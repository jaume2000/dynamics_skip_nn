#!/usr/bin/env python3
"""Per-coordinate trajectory statistics pooled over N images (not the mean trajectory).

For every image the full stage-3 trajectory is integrated on the GPU and reduced on the
spot, so RAM only holds small accumulators:
  * per-t histograms of progress p_i(t) and normalised velocity h_i/Δ_i   (bands)
  * 2D histogram of (p_i, h_i/Δ_i)                                         (phase density)
  * ROWS_PER_IMAGE random coordinates per image                           (recruitment)
  * one coordinate from each of the first N_LINES images                  (lines)
  * per-image scalars: straightness, wasted motion, block reversals, zig-zag share, ...
Only coordinates with |Δ_i| above the per-image median enter the normalised views; the
massive channel is dropped. "Blocks" are unit-time segments [k, k+1] (native blocks for
the standard network, the same intervals for the weight-tied one).

Images: one per class from 500 random classes (seed 0), the same set as the N500 suite.

Writes
  outputs/pooled_dynamics/<run>.pt
  figures/pooled_dynamics.pdf        weight-tied vs ConvNeXt-T at R=100
  figures/pooled_dynamics_R1.pdf     same at R=1
  figures/pooled_dynamics_table.{csv,tex}

Usage (from repo root):
    python paper/scripts/pooled_feature_dynamics.py                 # integrate missing runs, plot
    python paper/scripts/pooled_feature_dynamics.py --plot-only
    python paper/scripts/pooled_feature_dynamics.py --runs shared_R1 convnext_R1 --n-images 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper" / "scripts"))

from plot_trajectory_pca import SHARED_KEY, TITLES, build  # noqa: E402

FIG = ROOT / "paper" / "figures"
CACHE_DIR = ROOT / "outputs" / "pooled_dynamics"
RUNS = {
    "shared_R100": dict(model=SHARED_KEY, depth=900, repeats=None, step=0.01, label="R=100"),
    "convnext_R100": dict(model="convnext", depth=None, repeats=100, step=0.01, label="R=100"),
    "shared_R1": dict(model=SHARED_KEY, depth=9, repeats=None, step=1.0, label="R=1"),
    "convnext_R1": dict(model="convnext", depth=None, repeats=1, step=1.0, label="R=1"),
}
N_IMAGES = 500
SAMPLE_SEED = 0
BATCH = 8
MAX_STORED = 226
P_EDGES = np.linspace(-0.8, 1.8, 261)
H_EDGES = np.linspace(-1.5, 2.5, 401)
DENS_P = np.linspace(-0.3, 1.3, 321)
DENS_H = np.linspace(-1.0, 2.0, 601)
ROWS_PER_IMAGE = 6
N_LINES = 40
SEED = 0


# ----------------------------------------------------------------------------- integration
def _hist_rows(v: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Row-wise histogram of ``v`` (T, M) on fixed ``edges``; outliers go to the end bins."""
    idx = torch.bucketize(v.contiguous(), edges[1:-1])
    out = torch.zeros(v.shape[0], len(edges) - 1, device=v.device)
    return out.scatter_add_(1, idx, torch.ones_like(v))


def _hist2d(a: torch.Tensor, b: torch.Tensor, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
    a, b = a.flatten(), b.flatten()
    ok = (a >= ea[0]) & (a < ea[-1]) & (b >= eb[0]) & (b < eb[-1])
    ia = torch.bucketize(a[ok], ea[1:-1])
    ib = torch.bucketize(b[ok], eb[1:-1])
    nb = len(eb) - 1
    return torch.bincount(ia * nb + ib, minlength=(len(ea) - 1) * nb).view(len(ea) - 1, nb).float()


def _image_stats(x, h, t, unit_slots, gen, acc, keep_line: bool) -> dict:
    """Reduce one trajectory ``x``, ``h`` of shape (T, N) into ``acc``; return its scalars."""
    dev = x.device
    delta = x[-1] - x[0]
    sel = delta.abs() >= delta.abs().median()
    p = (x[:, sel] - x[0, sel]) / delta[sel]
    hn = h[:, sel] / delta[sel]
    acc["p_hist"] += _hist_rows(p, acc["p_edges"])
    acc["h_hist"] += _hist_rows(hn, acc["h_edges"])
    acc["dens"] += _hist2d(p, hn, acc["dens_p"], acc["dens_h"])

    hs = h[:, sel] * torch.sign(delta[sel])
    hs = hs / hs.abs().amax(dim=0, keepdim=True).clamp_min(1e-12)
    t_peak = t[hs.abs().argmax(dim=0)]
    t50 = t[(p >= 0.5).float().argmax(dim=0)]
    rows = torch.randperm(p.shape[1], generator=gen)[:ROWS_PER_IMAGE].to(dev)
    acc["rec_h"].append(hs[:, rows].T.cpu())
    acc["rec_p"].append(p[:, rows].T.cpu())
    acc["rec_tpeak"].append(t_peak[rows].cpu())
    acc["rec_t50"].append(t50[rows].cpu())
    if keep_line:
        j = rows[:1]
        acc["line_p"].append(p[:, j].T.cpu())
        acc["line_h"].append(hn[:, j].T.cpu())
        acc["line_t50"].append(t50[j].cpu())

    steps = x[1:] - x[:-1]
    back = (-steps * torch.sign(delta)).clamp_min(0).sum(0) / steps.abs().sum(0).clamp_min(1e-12)
    w = delta.abs()
    xb = x[unit_slots]
    db = xb[1:] - xb[:-1]
    net = xb[-1] - xb[0]
    s = torch.sign(db)
    rev = (s[1:] == -s[:-1]).float().mean(dim=1)
    undo = (torch.minimum(db[:-1].abs(), db[1:].abs()) * (s[1:] != s[:-1])).sum(1) / db[:-1].abs().sum(1)
    cos_b = torch.nn.functional.cosine_similarity(db[:-1], db[1:], dim=1)
    n_rev = (s[1:] != s[:-1]).sum(0)
    wb = db.abs().sum(0)
    built = (db.abs() * (s == torch.sign(net))).sum(1) / db.abs().sum(1)
    return {
        "monotone_share": float((w * (back < 0.05)).sum() / w.sum()),
        "wasted_motion": float(1 - net.abs().sum() / db.abs().sum()),
        "reversal_rate": float(rev.mean()),
        "undo_fraction": float(undo.mean()),
        "cos_consecutive_blocks": float(cos_b.mean()),
        "zigzag_share": float((wb * (n_rev >= 2)).sum() / wb.sum()),
        "t50_median": float(t50.median()),
        "tpeak_median": float(t_peak.median()),
        "built_per_block": built.cpu().numpy(),
        "block_norm": db.norm(dim=1).cpu().numpy(),
    }


@torch.no_grad()
def integrate(run: str, n_images: int, device) -> dict:
    import scripts.feature_map_explorer as fme
    from data.imagenet import ImageNetDataset
    from data.transforms.transforms import build_val_transforms

    spec = RUNS[run]
    step = spec["step"]
    dataset = ImageNetDataset(split="validation", transforms=build_val_transforms())
    labels = dataset.ds.data.column("label").to_pylist()
    indices = fme.sample_one_per_class(labels, 500, SAMPLE_SEED)[:n_images]
    enter, fields, head, bounds = build(spec["model"], depth=spec["depth"] or 0,
                                        repeats=spec["repeats"] or 1, step=step, device=device)
    D = len(fields)
    u = round(1 / step)
    assert abs(u * step - 1) < 1e-9 and D == 9 * u, (D, step)
    stride = max(1, -(-D // (MAX_STORED - 1)))
    stored = sorted(set(range(0, D + 1, stride)) | {k * u for k in range(10)})
    slot = {d: k for k, d in enumerate(stored)}
    unit_slots = [slot[k * u] for k in range(10)]
    t = torch.tensor(stored, device=device, dtype=torch.float32) * step
    switch_pairs = [k * u - 1 for k in range(1, 9)]

    def load(idx):
        return torch.stack([dataset[int(i)][0] for i in idx]).to(device)

    x = enter(load(indices[:BATCH]))
    for f in fields:
        x = x + step * (f(x) - x)
    massive = int(x.abs().amax(dim=(2, 3)).mean(dim=0).argmax())
    print(f"[{run}] D={D} step={step:g} stored={len(stored)} massive channel={massive}")

    edges = lambda e: torch.as_tensor(e, device=device, dtype=torch.float32)
    acc = {
        "p_edges": edges(P_EDGES), "h_edges": edges(H_EDGES),
        "dens_p": edges(DENS_P), "dens_h": edges(DENS_H),
        "p_hist": torch.zeros(len(stored), len(P_EDGES) - 1, device=device),
        "h_hist": torch.zeros(len(stored), len(H_EDGES) - 1, device=device),
        "dens": torch.zeros(len(DENS_P) - 1, len(DENS_H) - 1, device=device),
        "rec_h": [], "rec_p": [], "rec_tpeak": [], "rec_t50": [],
        "line_p": [], "line_h": [], "line_t50": [],
    }
    gen = torch.Generator().manual_seed(SEED)
    scalars, speeds, preds, flips_all = [], [], [], []
    n_done = 0
    for start in range(0, len(indices), BATCH):
        idx = indices[start : start + BATCH]
        x = enter(load(idx))
        keep = torch.tensor([c for c in range(x.shape[1]) if c != massive], device=device)
        B = x.shape[0]
        n_feat = keep.numel() * x.shape[2] * x.shape[3]
        X = torch.empty(len(stored), B, n_feat, device=device)
        H = torch.empty_like(X)
        length = torch.zeros(B, device=device)
        flips = torch.zeros(B, D, device=device)
        prev = None
        for d in range(D + 1):
            hf_full = fields[min(d, D - 1)](x) - x
            hf = hf_full.index_select(1, keep).flatten(1)
            sign = hf > 0
            if prev is not None:
                flips[:, d - 1] = (sign != prev).float().mean(dim=1)
            prev = sign
            if d in slot:
                X[slot[d]] = x.index_select(1, keep).flatten(1)
                H[slot[d]] = hf
            if d < D:
                length += step * hf.norm(dim=1)
                x = x + step * hf_full
        if head is not None:
            preds.append(head(x).argmax(dim=1).cpu())
        speeds.append(H.norm(dim=2).T.cpu())
        flips_all.append(flips.cpu())
        for b in range(B):
            sc = _image_stats(X[:, b], H[:, b], t, unit_slots, gen, acc,
                              keep_line=n_done < N_LINES)
            sc["straightness"] = float((X[-1, b] - X[0, b]).norm() / length[b])
            scalars.append(sc)
            n_done += 1
        del X, H
        print(f"  [{run}] {n_done}/{len(indices)}", flush=True)

    flips_all = torch.cat(flips_all).numpy()
    inside = np.setdiff1d(np.arange(D), switch_pairs)
    for i, sc in enumerate(scalars):
        sc["sign_flip_at_switch"] = float(flips_all[i, switch_pairs].mean())
        sc["sign_flip_inside"] = float(flips_all[i, inside].mean()) if len(inside) else float("nan")
    cat = lambda k: torch.cat(acc[k]).numpy()
    out = {
        "run": run, **{k: v for k, v in spec.items()},
        "t": t.cpu().numpy(), "block_bounds": bounds, "massive_channel": massive,
        "p_hist": acc["p_hist"].cpu().numpy(), "h_hist": acc["h_hist"].cpu().numpy(),
        "dens": acc["dens"].cpu().numpy(),
        "rec_h": cat("rec_h"), "rec_p": cat("rec_p"),
        "rec_tpeak": cat("rec_tpeak"), "rec_t50": cat("rec_t50"),
        "line_p": cat("line_p"), "line_h": cat("line_h"), "line_t50": cat("line_t50"),
        "speeds": torch.cat(speeds).numpy(),
        "scalars": scalars,
        "labels": np.asarray([labels[i] for i in indices]),
        "preds": torch.cat(preds).numpy() if preds else None,
    }
    return out


# ----------------------------------------------------------------------------- plotting
def _hist_quantiles(hist: np.ndarray, edges: np.ndarray, qs) -> np.ndarray:
    centers = 0.5 * (edges[1:] + edges[:-1])
    cdf = np.cumsum(hist, axis=1) / hist.sum(axis=1, keepdims=True)
    return np.stack([centers[np.argmax(cdf >= q, axis=1)] for q in qs])


def _bands(ax, t, hist, edges, color="0.2"):
    q05, q25, q50, q75, q95 = _hist_quantiles(hist, edges, [0.05, 0.25, 0.5, 0.75, 0.95])
    ax.fill_between(t, q05, q95, color=color, alpha=0.12, lw=0, label="P5–P95")
    ax.fill_between(t, q25, q75, color=color, alpha=0.25, lw=0, label="P25–P75")
    ax.plot(t, q50, color=color, lw=1.6, label="median")
    return q05, q95


def _vlines(ax, data):
    for b in data["block_bounds"]:
        ax.axvline(b, color="0.6", lw=0.5, ls=":", zorder=0)


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def draw_progress(ax, data, cax=None):
    t = data["t"]
    _bands(ax, t, data["p_hist"], P_EDGES)
    t50 = data["line_t50"]
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(*np.quantile(data["rec_t50"], [0.05, 0.95]))
    for row, tt in zip(data["line_p"], t50):
        ax.plot(t, row, color=cmap(norm(tt)), lw=0.6, alpha=0.85)
    ax.set_ylim(-0.5, 1.5)
    ax.set(xlabel=r"$t$", ylabel=r"progress $p_i(t)$")
    _vlines(ax, data)
    _style(ax)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def draw_recruitment(ax, data):
    order = np.argsort(data["rec_tpeak"], kind="stable")
    t = data["t"]
    im = ax.imshow(data["rec_h"][order], origin="lower", aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1, extent=(t[0], t[-1], 0, len(order)),
                   interpolation="antialiased", rasterized=True)
    ax.plot(data["rec_tpeak"][order], np.arange(len(order)) + 0.5, color="k", lw=0.7, alpha=0.6)
    ax.set(xlabel=r"$t$", ylabel="coordinate (sorted by peak time)")
    _vlines(ax, data)
    ax.tick_params(labelsize=8)
    return im


def draw_density(ax, data):
    H = data["dens"]
    marg = H.sum(axis=0)
    cdf = np.cumsum(marg) / marg.sum()
    centers = 0.5 * (DENS_H[1:] + DENS_H[:-1])
    lo, hi = centers[np.argmax(cdf >= 0.003)], centers[np.argmax(cdf >= 0.997)]
    cmap = plt.get_cmap("magma").with_extremes(bad=plt.get_cmap("magma")(0))
    im = ax.imshow(np.ma.masked_equal(H.T, 0), origin="lower", aspect="auto", cmap=cmap,
                   extent=(DENS_P[0], DENS_P[-1], DENS_H[0], DENS_H[-1]),
                   norm=LogNorm(vmin=1, vmax=H.max()), rasterized=True)
    pm = _hist_quantiles(data["p_hist"], P_EDGES, [0.5])[0]
    hm = _hist_quantiles(data["h_hist"], H_EDGES, [0.5])[0]
    ax.scatter(pm, hm, c=data["t"], cmap="viridis", s=2, zorder=3)
    ax.set_ylim(lo, hi)
    ax.set(xlabel=r"progress $p_i$", ylabel=r"$h_i/\Delta_i$")
    ax.tick_params(labelsize=8)
    return im


def plot_compare(datas: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(len(datas), 3, figsize=(12.0, 3.0 * len(datas)), layout="constrained",
                             squeeze=False)
    for r, data in enumerate(datas):
        name = f"{TITLES[data['model']]} ({data['label']})"
        sm = draw_progress(axes[r, 0], data)
        cb = fig.colorbar(sm, ax=axes[r, 0], pad=0.02)
        cb.set_label(r"$t_{50}$", fontsize=8)
        im = draw_recruitment(axes[r, 1], data)
        cb = fig.colorbar(im, ax=axes[r, 1], pad=0.02)
        cb.set_label(r"$\mathrm{sign}(\Delta_i)\,h_i/\max_t|h_i|$", fontsize=8)
        im = draw_density(axes[r, 2], data)
        fig.colorbar(im, ax=axes[r, 2], pad=0.02).set_label("count (log)", fontsize=8)
        axes[r, 0].set_ylabel(f"{name}\n" + r"progress $p_i(t)$", fontsize=9)
        if r == 0:
            axes[r, 0].set_title(r"progress $(x_i-x_i(0))/\Delta_i$", fontsize=9)
            axes[r, 1].set_title("velocity, sorted by peak time (red: towards final value)",
                                 fontsize=9)
            axes[r, 2].set_title(r"phase density $(p_i,\ h_i/\Delta_i)$", fontsize=9)
            axes[r, 0].legend(fontsize=7, frameon=False, loc="upper left")
    n = len(datas[0]["scalars"])
    fig.suptitle(f"Stage-3 coordinates pooled over {n} images ($|\\Delta_i|$ above each image's "
                 "median; massive channel removed)", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


TABLE_COLS = {
    "straightness": "straightness",
    "wasted_motion": "wasted motion",
    "monotone_share": "monotone share",
    "cos_consecutive_blocks": r"$\cos(\Delta x_k,\Delta x_{k+1})$",
    "reversal_rate": "reversal rate",
    "undo_fraction": "undo fraction",
    "zigzag_share": "zig-zag share",
    "sign_flip_at_switch": r"sign flips at $t\in\mathbb{N}$",
    "t50_median": r"median $t_{50}$",
}


def write_table(datas: list[dict], stem: Path) -> None:
    rows = []
    for data in datas:
        df = pd.DataFrame([{k: s[k] for k in TABLE_COLS} for s in data["scalars"]])
        row = {"model": TITLES[data["model"]], "R": data["label"], "n": len(df)}
        for k in TABLE_COLS:
            row[k] = df[k].mean()
            row[f"{k}_std"] = df[k].std()
        if data.get("preds") is not None:
            row["top1"] = float((data["preds"] == data["labels"]).mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(stem.with_suffix(".csv"), index=False)
    lines = [r"\begin{tabular}{ll" + "c" * len(TABLE_COLS) + "}", r"\toprule",
             "Model & $R$ & " + " & ".join(TABLE_COLS.values()) + r" \\", r"\midrule"]
    for r in rows:
        cells = [f"{r[k]:.2f}$\\pm${r[k + '_std']:.2f}" for k in TABLE_COLS]
        lines.append(f"{r['model']} & {r['R'].split('=')[1]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    stem.with_suffix(".tex").write_text("\n".join(lines) + "\n")
    print(f"Saved {stem.with_suffix('.csv')} and .tex")
    print(table[["model", "R", "n", *TABLE_COLS, *(["top1"] if "top1" in table else [])]]
          .round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    ap.add_argument("--n-images", type=int, default=N_IMAGES)
    ap.add_argument("--force", action="store_true", help="re-integrate even if cached")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {}
    for run in args.runs:
        cache = CACHE_DIR / f"{run}_n{args.n_images}.pt"
        if cache.is_file() and not args.force:
            data[run] = torch.load(cache, weights_only=False)
            print(f"loaded {cache}")
        elif args.plot_only:
            print(f"missing {cache}, skipped")
            continue
        else:
            data[run] = integrate(run, args.n_images, device)
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data[run], cache)
            print(f"cached -> {cache}")
            torch.cuda.empty_cache()
    suffix = "" if args.n_images == N_IMAGES else f"_n{args.n_images}"
    for runs, name in ((["shared_R100", "convnext_R100"], "pooled_dynamics"),
                       (["shared_R1", "convnext_R1"], "pooled_dynamics_R1")):
        have = [data[r] for r in runs if r in data]
        if have:
            plot_compare(have, FIG / f"{name}{suffix}.pdf")
    if data:
        write_table([data[r] for r in RUNS if r in data], FIG / f"pooled_dynamics_table{suffix}")


if __name__ == "__main__":
    main()
