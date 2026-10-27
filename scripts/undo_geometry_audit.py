#!/usr/bin/env python3
"""Audit of the stage-3 "undo" statistics (paper Table 2) against plain vector geometry.

Question: do the per-coordinate statistics of Table 2 (reversal rate, undo fraction,
zig-zag share, wasted motion, ...) carry information beyond the Gram matrix of the
block increments dx_k = x(k+1) - x(k)?

Same 500 validation images (one per class, seed 0), same checkpoints and the same
massive channel as ``paper/scripts/pooled_feature_dynamics.py`` (195 for ConvNeXt-T,
236 for the weight-tied model). Only the integer-time states x(0..9) are stored.

Steps (from repo root):
    python scripts/undo_geometry_audit.py integrate            # states -> outputs/undo_audit/
    python scripts/undo_geometry_audit.py analyze              # A2-A7 tables
    python scripts/undo_geometry_audit.py b4                   # Table 9 forward check

Nothing under outputs/pooled_dynamics or paper/ is written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper" / "scripts"))

OUT = ROOT / "outputs" / "undo_audit"
POOLED_CACHE = ROOT / "outputs" / "pooled_dynamics"
N_IMAGES = 500
SAMPLE_SEED = 0
BATCH = 8
SURROGATE_SEEDS = (0, 1, 2)
MASSIVE = {"convnext": 195, "convnext_shared": 236}

RUNS = {
    "shared_R1": dict(model="convnext_shared", depth=9, step=1.0),
    "shared_R100": dict(model="convnext_shared", depth=900, step=0.01),
    "convnext_R1": dict(model="convnext", repeats=1, step=1.0),
    "convnext_R100": dict(model="convnext", repeats=100, step=0.01),
    "rand_g1_R1": dict(model="random", gain=1.0, repeats=1, step=1.0),
    "rand_g1_R100": dict(model="random", gain=1.0, repeats=100, step=0.01),
    "rand_g1.35_R1": dict(model="random", gain=1.35, repeats=1, step=1.0),
    "rand_g1.35_R100": dict(model="random", gain=1.35, repeats=100, step=0.01),
}
TRAINED = ["shared_R1", "shared_R100", "convnext_R1", "convnext_R100"]
RANDOM = [r for r in RUNS if r.startswith("rand")]


def dataset_and_indices():
    import scripts.feature_map_explorer as fme
    from data.imagenet import ImageNetDataset
    from data.transforms.transforms import build_val_transforms

    ds = ImageNetDataset(split="validation", transforms=build_val_transforms())
    labels = ds.ds.data.column("label").to_pylist()
    idx = fme.sample_one_per_class(labels, 500, SAMPLE_SEED)[:N_IMAGES]
    return ds, idx, labels


# ----------------------------------------------------------------------------- integration
def build_run(spec: dict, device):
    """(enter, fields, head) with fields returning x + f(x)."""
    if spec["model"] == "random":
        from scripts.compare_solver_convergence import build_convnext
        from scripts.random_init_convergence import random_model

        model = random_model("blocks", spec["gain"], 0).to(device).eval()
        enter, fields, _, _ = build_convnext(model, repeats=spec["repeats"], euler_step=spec["step"])
        return enter, fields, None
    from plot_trajectory_pca import build

    enter, fields, head, _ = build(spec["model"], depth=spec.get("depth", 0),
                                   repeats=spec.get("repeats", 1), step=spec["step"], device=device)
    return enter, fields, head


@torch.no_grad()
def integrate(run: str, device) -> None:
    spec = RUNS[run]
    path = OUT / f"states_{run}.npy"
    if path.is_file() and (OUT / f"micro_{run}.npz").is_file():
        print(f"[{run}] cached")
        return
    ds, idx, labels = dataset_and_indices()
    enter, fields, head = build_run(spec, device)
    step = spec["step"]
    D = len(fields)
    u = round(1 / step)
    assert D == 9 * u
    switch = {k * u for k in range(1, 9)}
    load = lambda ii: torch.stack([ds[int(i)][0] for i in ii]).to(device)

    if spec["model"] == "random":
        massive = None
    else:
        x = enter(load(idx[:BATCH]))
        for f in fields:
            x = x + step * (f(x) - x)
        massive = int(x.abs().amax(dim=(2, 3)).mean(dim=0).argmax())
        assert massive == MASSIVE[spec["model"]], massive
    x = enter(load(idx[:1]))
    C, H, W = x.shape[1:]
    keep = torch.tensor([c for c in range(C) if c != massive], device=device)
    states = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                       shape=(len(idx), 10, keep.numel(), H, W))
    length = np.zeros(len(idx))
    flips = np.zeros(len(idx))
    norm_x = np.zeros((len(idx), 10))
    massive_norm = np.zeros((len(idx), 10))
    preds = []
    for s in range(0, len(idx), BATCH):
        ii = idx[s : s + BATCH]
        x = enter(load(ii))
        B = x.shape[0]
        L = torch.zeros(B, device=device)
        fl = torch.zeros(B, device=device)
        prev = None
        for d in range(D + 1):
            if d % u == 0:
                xs = x.index_select(1, keep)
                states[s : s + B, d // u] = xs.cpu().numpy()
                norm_x[s : s + B, d // u] = xs.flatten(1).norm(dim=1).cpu().numpy()
                if massive is not None:
                    massive_norm[s : s + B, d // u] = x[:, massive].flatten(1).norm(dim=1).cpu().numpy()
            if d == D:
                break
            hf_full = fields[d](x) - x
            hf = hf_full.index_select(1, keep).flatten(1)
            sign = hf > 0
            if d in switch:
                fl += (sign != prev).float().mean(dim=1)
            prev = sign
            L += step * hf.norm(dim=1)
            x = x + step * hf_full
        length[s : s + B] = L.cpu().numpy()
        flips[s : s + B] = (fl / 8).cpu().numpy()
        if head is not None:
            preds.append(head(x).argmax(dim=1).cpu().numpy())
        print(f"  [{run}] {s + B}/{len(idx)}", flush=True)
    states.flush()
    np.savez(OUT / f"micro_{run}.npz", length=length, sign_flip_at_switch=flips, norm_x=norm_x,
             massive_norm=massive_norm, indices=np.asarray(idx),
             labels=np.asarray([labels[i] for i in idx]),
             preds=np.concatenate(preds) if preds else np.zeros(0),
             massive=-1 if massive is None else massive)
    print(f"[{run}] saved {path}")


# ----------------------------------------------------------------------------- metrics
def table2_block_metrics(xb: torch.Tensor, select_all: bool = False) -> dict:
    """Integer-time part of ``pooled_feature_dynamics._image_stats`` on states xb (10, F).

    ``select_all=False`` reproduces the code (median selection only for t50);
    ``select_all=True`` applies the |Δ_i| >= median selection to every metric, as the
    paper caption describes.
    """
    delta = xb[-1] - xb[0]
    sel = delta.abs() >= delta.abs().median()
    p = (xb[:, sel] - xb[0, sel]) / delta[sel]
    t50 = torch.arange(10, device=xb.device, dtype=xb.dtype)[(p >= 0.5).float().argmax(dim=0)]
    if select_all:
        xb = xb[:, sel]
    db = xb[1:] - xb[:-1]
    net = xb[-1] - xb[0]
    s = torch.sign(db)
    rev = (s[1:] == -s[:-1]).float().mean(dim=1)
    undo = (torch.minimum(db[:-1].abs(), db[1:].abs()) * (s[1:] != s[:-1])).sum(1) / db[:-1].abs().sum(1)
    cos_b = F.cosine_similarity(db[:-1], db[1:], dim=1)
    n_rev = (s[1:] != s[:-1]).sum(0)
    wb = db.abs().sum(0)
    back = (-db * torch.sign(net)).clamp_min(0).sum(0) / db.abs().sum(0).clamp_min(1e-12)
    w = net.abs()
    return {
        "straightness": float(net.norm() / db.norm(dim=1).sum()),
        "wasted_motion": float(1 - net.abs().sum() / db.abs().sum()),
        "monotone_share": float((w * (back < 0.05)).sum() / w.sum()),
        "cos_consecutive_blocks": float(cos_b.mean()),
        "reversal_rate": float(rev.mean()),
        "undo_fraction": float(undo.mean()),
        "zigzag_share": float((wb * (n_rev >= 2)).sum() / wb.sum()),
        "t50_median": float(t50.median()),
        "_rev_pairs": rev.cpu().numpy(),
        "_cos_pairs": cos_b.cpu().numpy(),
    }


METRICS = ["straightness", "wasted_motion", "monotone_share", "cos_consecutive_blocks",
           "reversal_rate", "undo_fraction", "zigzag_share", "t50_median"]


def gram_quantities(G: np.ndarray) -> dict:
    """A2/A3 from the 9x9 Gram matrix of one image."""
    d = np.sqrt(np.diag(G))
    cos = G / np.outer(d, d)
    straight = math.sqrt(G.sum()) / d.sum()
    sq = np.diag(G)
    s = np.array([G[k, k + 1 :].sum() / sq[k] for k in range(8)])
    n = np.array([G[k, k + 1] / sq[k] for k in range(8)])
    e = np.clip(-s, 0, 1)
    E = (e * d[:8]).sum() / d[:8].sum()
    return dict(cos=cos, straight=straight, s=s, n=n, E=E)


def haar_orthogonal(n: int, gen: torch.Generator, device) -> torch.Tensor:
    A = torch.randn(n, n, generator=gen, dtype=torch.float64).to(device)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


def gram_surrogate(xb: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """States with exactly the same increment Gram matrix, in a random 9-frame of R^F."""
    dx = (xb[1:] - xb[:-1]).double()
    G = dx @ dx.T
    Lc = torch.linalg.cholesky(G)
    Z = torch.randn(dx.shape[1], 9, generator=gen, dtype=torch.float64).to(xb.device)
    Q, _ = torch.linalg.qr(Z)                              # (F, 9), orthonormal columns
    Y = Lc @ Q.T                                           # (9, F), Y Y^T = G
    out = torch.cat([xb[:1].double(), xb[:1].double() + torch.cumsum(Y, dim=0)])
    return out.float()


def local_stats(x: torch.Tensor) -> dict:
    """Per-position (C-dim) cosines and survival, x of shape (10, C, P)."""
    dx = x[1:] - x[:-1]                                    # (9, C, P)
    a, b = dx[:-1], dx[1:]                                 # pairs k, k+1 for k=0..7
    na, nb = a.norm(dim=1), b.norm(dim=1)                  # (8, P)
    cos_p = (a * b).sum(1) / (na * nb).clamp_min(1e-12)
    rest = x[-1:] - x[1:9]                                 # x(9) - x(k+1), k=0..7
    s_p = (rest * a).sum(1) / (na**2).clamp_min(1e-12)
    # signed vs absolute local inner products, normalised by the global norms
    ab = (a * b).sum(1)                                    # (8, P)
    gl = a.flatten(1).norm(dim=1) * b.flatten(1).norm(dim=1)
    return dict(cos_p=cos_p.cpu().numpy(), s_p=s_p.cpu().numpy(), w=na.cpu().numpy(),
                cos_global=(ab.sum(1) / gl).cpu().numpy(),
                abs_local=(ab.abs().sum(1) / gl).cpu().numpy())


@torch.no_grad()
def analyze_run(run: str, device, surrogates: bool) -> dict:
    states = np.load(OUT / f"states_{run}.npy", mmap_mode="r")
    N, _, C, H, W = states.shape
    rots = {sd: haar_orthogonal(C, torch.Generator().manual_seed(1000 + sd), device).float()
            for sd in SURROGATE_SEEDS} if surrogates else {}
    res = {"G": np.zeros((N, 9, 9)), "orig": [], "paper": [], "local": [],
           "sur": {(k, sd, v): [] for k in ("gram", "rot") for sd in rots for v in ("orig", "paper")}}
    for i in range(N):
        x4 = torch.from_numpy(np.ascontiguousarray(states[i])).to(device)   # (10, C, H, W)
        xb = x4.flatten(1)
        dx = (xb[1:] - xb[:-1]).double()
        res["G"][i] = (dx @ dx.T).cpu().numpy()
        res["orig"].append(table2_block_metrics(xb))
        res["paper"].append(table2_block_metrics(xb, select_all=True))
        res["local"].append(local_stats(x4.flatten(2)))
        for sd, O in rots.items():
            g = torch.Generator().manual_seed(10_000 * sd + i)
            xg = gram_surrogate(xb, g)
            xr = torch.einsum("dc,tcp->tdp", O, x4.flatten(2)).flatten(1)
            for name, xs in (("gram", xg), ("rot", xr)):
                res["sur"][(name, sd, "orig")].append(table2_block_metrics(xs))
                res["sur"][(name, sd, "paper")].append(table2_block_metrics(xs, select_all=True))
        if (i + 1) % 100 == 0:
            print(f"  [{run}] analysed {i + 1}/{N}", flush=True)
    return res


# ----------------------------------------------------------------------------- reporting
def pct(a, q):
    return float(np.percentile(a, q))


def fmt_ms(a):
    a = np.asarray(a, dtype=float)
    return f"{a.mean():.3f}±{a.std(ddof=1):.3f}"


def report(results: dict) -> str:
    lines = []
    cached = {r: torch.load(POOLED_CACHE / f"{r}_n500.pt", weights_only=False)["scalars"]
              for r in TRAINED if (POOLED_CACHE / f"{r}_n500.pt").is_file()}
    micro = {r: np.load(OUT / f"micro_{r}.npz") for r in results}

    # --- validation against Table 2 cache
    lines.append("## Validation vs cached Table-2 scalars (max |diff| over images)\n")
    lines.append("| run | " + " | ".join(METRICS) + " | top1 (500) |")
    lines.append("|---" * (len(METRICS) + 2) + "|")
    for r in TRAINED:
        if r not in results or r not in cached:
            continue
        cells = []
        for m in METRICS:
            a = np.array([s[m] for s in cached[r]])
            b = np.array([s[m] for s in results[r]["orig"]])
            cells.append(f"{np.abs(a - b).max():.1e}")
        mc = micro[r]
        cells.append(f"{(mc['preds'] == mc['labels']).mean():.3f}")
        lines.append(f"| {r} | " + " | ".join(cells) + " |")
    lines.append("\nMicro-resolution quantities regenerated (straightness from micro path length, "
                 "sign flips at switch) vs cache:\n")
    for r in TRAINED:
        if r not in results or r not in cached:
            continue
        st = np.linalg.norm  # noqa
        st_m = np.array([s["straightness"] for s in cached[r]])
        fl_m = np.array([s["sign_flip_at_switch"] for s in cached[r]])
        st_new = np.array([math.sqrt(G.sum()) for G in results[r]["G"]]) / micro[r]["length"]
        lines.append(f"- {r}: straightness max|diff|={np.abs(st_m - st_new).max():.1e}, "
                     f"flips max|diff|={np.abs(fl_m - micro[r]['sign_flip_at_switch']).max():.1e}")

    # --- A2
    lines.append("\n## A2. Mean cosine matrix cos(dx_j, dx_k) over images\n")
    for r in results:
        cos = np.stack([gram_quantities(G)["cos"] for G in results[r]["G"]]).mean(0)
        lines.append(f"**{r}**\n")
        lines.append("| j\\k | " + " | ".join(str(k) for k in range(9)) + " |")
        lines.append("|---" * 10 + "|")
        for j in range(9):
            lines.append(f"| {j} | " + " | ".join(f"{cos[j, k]:.2f}" for k in range(9)) + " |")
        lines.append("")
    lines.append("Straightness from G = sqrt(1ᵀG1)/Σ_k sqrt(G_kk) vs Table-2 value:\n")
    lines.append("| run | from G (block path) | Table-2 code (cache) | max abs diff | mean consecutive cos | mean |cos| j≠k |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        gq = [gram_quantities(G) for G in results[r]["G"]]
        sg = np.array([q["straight"] for q in gq])
        consec = np.array([np.mean([q["cos"][k, k + 1] for k in range(8)]) for q in gq])
        off = np.array([np.abs(q["cos"][~np.eye(9, dtype=bool)]).mean() for q in gq])
        if r in cached:
            sm = np.array([s["straightness"] for s in cached[r]])
            lines.append(f"| {r} | {fmt_ms(sg)} | {fmt_ms(sm)} | {np.abs(sg - sm).max():.1e} | {fmt_ms(consec)} | {fmt_ms(off)} |")
        else:
            lines.append(f"| {r} | {fmt_ms(sg)} | — | — | {fmt_ms(consec)} | {fmt_ms(off)} |")

    # --- A3
    lines.append("\n## A3. Vector survival s_k and immediate n_k (mean [P10, P90] %neg)\n")
    for r in results:
        gq = [gram_quantities(G) for G in results[r]["G"]]
        S = np.stack([q["s"] for q in gq])
        Nn = np.stack([q["n"] for q in gq])
        E = np.array([q["E"] for q in gq])
        lines.append(f"**{r}** — E = {fmt_ms(E)} (P10 {pct(E, 10):.3f}, P90 {pct(E, 90):.3f})\n")
        lines.append("| k | s_k mean | s_k P10 | s_k P90 | s_k %neg | n_k mean | n_k P10 | n_k P90 | n_k %neg |")
        lines.append("|---" * 9 + "|")
        for k in range(8):
            a, b = S[:, k], Nn[:, k]
            lines.append(f"| {k} | {a.mean():.3f} | {pct(a, 10):.3f} | {pct(a, 90):.3f} | {100 * (a < 0).mean():.0f} "
                         f"| {b.mean():.3f} | {pct(b, 10):.3f} | {pct(b, 90):.3f} | {100 * (b < 0).mean():.0f} |")
        lines.append("")

    # --- A4
    lines.append("## A4. Gaussian prediction arccos(c)/π vs measured reversal rate (all coords, as Table 2)\n")
    lines.append("| run | mean c | pred | measured | mean |pred−meas| | max |pred−meas| | corr over (img,pair) | wasted+straight |")
    lines.append("|---" * 8 + "|")
    per_pair = {}
    for r in results:
        c = np.stack([m["_cos_pairs"] for m in results[r]["orig"]])
        rv = np.stack([m["_rev_pairs"] for m in results[r]["orig"]])
        pr = np.arccos(np.clip(c, -1, 1)) / np.pi
        per_pair[r] = (pr.mean(0), rv.mean(0))
        cc = np.corrcoef(pr.ravel(), rv.ravel())[0, 1]
        ws = np.array([m["wasted_motion"] + m["straightness"] for m in results[r]["orig"]])
        lines.append(f"| {r} | {c.mean():.3f} | {pr.mean():.4f} | {rv.mean():.4f} | {np.abs(pr - rv).mean():.4f} "
                     f"| {np.abs(pr - rv).max():.4f} | {cc:.3f} | {fmt_ms(ws)} |")
    lines.append("\nPer pair (pred / measured, mean over images):\n")
    lines.append("| pair k,k+1 | " + " | ".join(results) + " |")
    lines.append("|---" * (len(results) + 1) + "|")
    for k in range(8):
        lines.append(f"| {k},{k + 1} | " + " | ".join(
            f"{per_pair[r][0][k]:.3f} / {per_pair[r][1][k]:.3f}" for r in results) + " |")

    # --- A5
    lines.append("\n## A5. Surrogates (integer-time Table-2 pipeline; mean over images, ± = std across 3 seeds of the means)\n")
    for variant, title in (("orig", "code as run (selection only for t50)"),
                           ("paper", "paper definition (median selection on every metric)")):
        lines.append(f"### Variant: {title}\n")
        lines.append("| run | source | " + " | ".join(METRICS) + " |")
        lines.append("|---" * (len(METRICS) + 2) + "|")
        for r in results:
            real = results[r][variant]
            lines.append(f"| {r} | real | " + " | ".join(
                f"{np.mean([m[k] for m in real]):.3f}" for k in METRICS) + " |")
            for name, label in (("gram", "(a) same Gram"), ("rot", "(b) channel rot.")):
                keys = [(name, sd, variant) for sd in SURROGATE_SEEDS if (name, sd, variant) in results[r]["sur"]]
                if not keys or not results[r]["sur"][keys[0]]:
                    continue
                cells = []
                for k in METRICS:
                    means = np.array([np.mean([m[k] for m in results[r]["sur"][key]]) for key in keys])
                    cells.append(f"{means.mean():.3f}±{means.std(ddof=0):.3f}")
                lines.append(f"| {r} | {label} | " + " | ".join(cells) + " |")
        lines.append("")

    # --- A6
    lines.append("## A6. Per spatial position (383/384-dim increments), pooled over images, positions, k=0..7\n")
    lines.append("| run | global cos | cos_p P10 | P25 | P50 | P75 | P90 | %neg | ‖dx_p‖-wtd mean | %cos_p<−0.3 | Σ|⟨a_p,b_p⟩| / ‖a‖‖b‖ | s_p P10 | P50 | P90 | %neg | ‖dx_p‖-wtd mean | global s (mean) |")
    lines.append("|---" * 18 + "|")
    for r in results:
        loc = results[r]["local"]
        cp = np.concatenate([l["cos_p"].ravel() for l in loc])
        sp = np.concatenate([l["s_p"].ravel() for l in loc])
        w = np.concatenate([l["w"].ravel() for l in loc])
        cg = np.concatenate([l["cos_global"] for l in loc])
        al = np.concatenate([l["abs_local"] for l in loc])
        sg = np.stack([gram_quantities(G)["s"] for G in results[r]["G"]]).mean()
        lines.append(
            f"| {r} | {cg.mean():.3f} | {pct(cp, 10):.3f} | {pct(cp, 25):.3f} | {pct(cp, 50):.3f} | {pct(cp, 75):.3f} "
            f"| {pct(cp, 90):.3f} | {100 * (cp < 0).mean():.0f} | {(w * cp).sum() / w.sum():.3f} | {100 * (cp < -0.3).mean():.1f} "
            f"| {al.mean():.3f} | {pct(sp, 10):.3f} | {pct(sp, 50):.3f} | {pct(sp, 90):.3f} | {100 * (sp < 0).mean():.0f} "
            f"| {(w * sp).sum() / w.sum():.3f} | {sg:.3f} |")
    lines.append(f"\nNull for two independent random 383-dim vectors: std(cos_p) ≈ {1 / math.sqrt(383):.3f}.\n")

    # --- A7 extras: field strength rho on the R=1 path
    lines.append("## Field strength ρ = mean_k ‖dx_k‖/‖x_k‖ on R=1 states\n")
    lines.append("| run | ρ | ‖x(0)‖ | ‖x(9)‖ |")
    lines.append("|---|---|---|---|")
    for r in results:
        if not r.endswith("_R1"):
            continue
        G = results[r]["G"]
        nx = micro[r]["norm_x"]
        rho = (np.sqrt(np.stack([np.diag(g) for g in G])) / nx[:, :9]).mean()
        lines.append(f"| {r} | {rho:.3f} | {nx[:, 0].mean():.1f} | {nx[:, 9].mean():.1f} |")
    return "\n".join(lines) + "\n"


def to_jsonable(results):
    out = {}
    for r, res in results.items():
        out[r] = {
            "orig": [{k: v for k, v in m.items() if not k.startswith("_")} for m in res["orig"]],
            "paper": [{k: v for k, v in m.items() if not k.startswith("_")} for m in res["paper"]],
        }
    return out


# ----------------------------------------------------------------------------- B4
@torch.no_grad()
def b4(device) -> str:
    import scripts.feature_map_explorer as fme

    ds, idx, labels = dataset_and_indices()
    model = fme.load_interpoled_model("convnext_shared").to(device).eval()
    blk = model.deltifiedStage3[0]
    m = MASSIVE["convnext_shared"]
    head = lambda x: model.fc(model.globalPool(model.stage4(model.tail(x))))
    enter = lambda b: model.stage2(model.stage1(model.stem(b)))
    steps = (1, 2, 3, 9)
    acc = {D: [] for D in steps}
    rows = {D: {"norm": [], "norm_nm": [], "massive": [], "cos": [], "cos_nm": [], "rel_nm": []} for D in steps}
    checks = {}
    y_all = []
    for s in range(0, len(idx), BATCH):
        ii = idx[s : s + BATCH]
        imgs = torch.stack([ds[int(i)][0] for i in ii]).to(device)
        y = torch.tensor([labels[i] for i in ii], device=device)
        y_all.append(y.cpu())
        x0 = enter(imgs)
        xs = {}
        for D in steps:
            x = x0
            for _ in range(D):
                x = model._integrate_block(blk, x, 9.0 / D, None)
            xs[D] = x
            acc[D].append((head(x).argmax(1) == y).cpu())
        if s == 0:
            h0 = blk(x0) - x0
            checks["one_step_minus_x0_plus_9h"] = float((xs[1] - (x0 + 9.0 * h0)).abs().max())
            checks["eulerStep_attr"] = float(blk.eulerStep)
            cfg = {"block_indices": [0, 9, 10], "euler_step": 9.0}
            checks["custom_forward_vs_manual_logits"] = float(
                (model.custom_forward(imgs, cfg) - head(xs[1])).abs().max())
            cfg2 = {"block_indices": [0, 0, 9, 10], "euler_step": 4.5}
            checks["custom_forward_vs_manual_logits_D2"] = float(
                (model.custom_forward(imgs, cfg2) - head(xs[2])).abs().max())
            checks["norm_h0_over_norm_x0_nomassive"] = float(
                (torch.cat([h0[:, :m], h0[:, m + 1:]], 1).flatten(1).norm(dim=1)
                 / torch.cat([x0[:, :m], x0[:, m + 1:]], 1).flatten(1).norm(dim=1)).mean())
        nat = xs[9]
        nat_nm = torch.cat([nat[:, :m], nat[:, m + 1:]], 1).flatten(1)
        for D in steps:
            x = xs[D]
            x_nm = torch.cat([x[:, :m], x[:, m + 1:]], 1).flatten(1)
            r = rows[D]
            r["norm"].append(x.flatten(1).norm(dim=1).cpu())
            r["norm_nm"].append(x_nm.norm(dim=1).cpu())
            r["massive"].append(x[:, m].flatten(1).norm(dim=1).cpu())
            r["cos"].append(F.cosine_similarity(x.flatten(1), nat.flatten(1), dim=1).cpu())
            r["cos_nm"].append(F.cosine_similarity(x_nm, nat_nm, dim=1).cpu())
            r["rel_nm"].append(((x_nm - nat_nm).norm(dim=1) / nat_nm.norm(dim=1)).cpu())
    x0n = None
    lines = ["## B4. Weight-tied, Euler at T=9 with D steps (Δt=9/D), 500 images\n",
             "| D (Δt) | top-1 (500) | ‖x_T‖ | ‖x_T‖ w/o ch236 | ‖x_T[ch236]‖ | cos(x_T, x_T^D9) | same w/o ch236 | ‖x_T−x_T^D9‖/‖x_T^D9‖ w/o ch236 |",
             "|---" * 8 + "|"]
    for D in steps:
        r = {k: torch.cat(v).numpy() for k, v in rows[D].items()}
        a = torch.cat(acc[D]).float().mean().item()
        lines.append(f"| {D} ({9 / D:g}) | {100 * a:.1f} | {r['norm'].mean():.0f} | {r['norm_nm'].mean():.0f} "
                     f"| {r['massive'].mean():.0f} | {r['cos'].mean():.3f} | {r['cos_nm'].mean():.3f} | {r['rel_nm'].mean():.3f} |")
    lines.append("\nChecks: " + json.dumps(checks))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["integrate", "analyze", "b4"])
    ap.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cmd == "integrate":
        for r in args.runs:
            integrate(r, device)
            torch.cuda.empty_cache()
    elif args.cmd == "analyze":
        results = {}
        for r in args.runs:
            if not (OUT / f"states_{r}.npy").is_file():
                print(f"missing states for {r}, skipped")
                continue
            results[r] = analyze_run(r, device, surrogates=r in TRAINED)
        text = report(results)
        suffix = "" if args.runs == list(RUNS) else "_" + "-".join(args.runs)
        (OUT / f"report{suffix}.md").write_text(text)
        torch.save({r: {"G": res["G"], "local": res["local"]} for r, res in results.items()},
                   OUT / f"gram_local{suffix}.pt")
        (OUT / f"scalars{suffix}.json").write_text(json.dumps(to_jsonable(results)))
        print(text)
    else:
        text = b4(device)
        (OUT / "b4.md").write_text(text)
        print(text)


if __name__ == "__main__":
    main()
