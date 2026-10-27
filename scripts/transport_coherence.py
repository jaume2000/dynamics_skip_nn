#!/usr/bin/env python3
"""Basis-independent coherence of the stage-3 transport (phase 1.1 of the rewrite).

Uses the integer-time states and Gram matrices of ``scripts/undo_geometry_audit.py``
(same 500 images, massive channel removed) and re-integrates the R=100 runs once to
get, on the full micro grid,
  * the arc length of every unit block (sum of Δt‖h‖ over its micro-steps),
  * the projected progress p(t) = <x(t) - x0, û> / ‖x_T - x0‖, û = (x_T - x0)/‖x_T - x0‖.

Per configuration it reports
  * straightness with the real arc length and with the sum of block chords, and the
    intra-block factor (sum of chords / arc length),
  * the 9x9 cosine matrix of block increments and its mean off-diagonal value,
  * vector survival s_k and the erased fraction E,
  * alignment with the destination cos(dx_k, x_T - x0),
  * basis-free t50 (first t with p(t) >= 0.5) on the R=100 grid.

Writes outputs/transport_coherence/{summary.csv, per_block.csv, cos_<run>.npy,
progress_<run>.npz, report.md}.

Usage (from repo root):
    python scripts/transport_coherence.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import undo_geometry_audit as audit  # noqa: E402

OUT = ROOT / "outputs" / "transport_coherence"
CONFIGS = ["shared_R1", "shared_R100", "convnext_R1", "convnext_R100", "rand_g1.35_R1", "rand_g1.35_R100"]
LABELS = {
    "shared_R1": "tied R=1", "shared_R100": "tied R=100",
    "convnext_R1": "untied R=1", "convnext_R100": "untied R=100",
    "rand_g1.35_R1": "random untied g*=1.35 R=1", "rand_g1.35_R100": "random untied g*=1.35 R=100",
}


@torch.no_grad()
def micro_progress(run: str, device) -> dict:
    """Arc length per unit block and p(t) on the full grid for one refined run."""
    spec = audit.RUNS[run]
    states = np.load(audit.OUT / f"states_{run}.npy", mmap_mode="r")
    micro = np.load(audit.OUT / f"micro_{run}.npz")
    massive = int(micro["massive"])
    ds, idx, _ = audit.dataset_and_indices()
    enter, fields, _ = audit.build_run(spec, device)
    step = spec["step"]
    D = len(fields)
    u = round(1 / step)
    N = len(idx)
    p = np.zeros((N, D + 1), dtype=np.float32)
    arc = np.zeros((N, 9))
    for s in range(0, N, audit.BATCH):
        ii = idx[s : s + audit.BATCH]
        B = len(ii)
        x = enter(torch.stack([ds[int(i)][0] for i in ii]).to(device))
        keep = torch.tensor([c for c in range(x.shape[1]) if c != massive], device=device)
        x0 = torch.from_numpy(np.ascontiguousarray(states[s : s + B, 0])).to(device).flatten(1)
        xT = torch.from_numpy(np.ascontiguousarray(states[s : s + B, 9])).to(device).flatten(1)
        net = xT - x0
        nn_ = net.norm(dim=1)
        uhat = net / nn_[:, None]
        L = torch.zeros(B, 9, device=device)
        for d in range(D + 1):
            xs = x.index_select(1, keep).flatten(1)
            p[s : s + B, d] = (((xs - x0) * uhat).sum(1) / nn_).cpu().numpy()
            if d == D:
                break
            h = fields[d](x) - x
            L[:, d // u] += step * h.index_select(1, keep).flatten(1).norm(dim=1)
            x = x + step * h
        arc[s : s + B] = L.cpu().numpy()
        print(f"  [{run}] {s + B}/{N}", flush=True)
    # sanity: the re-integrated endpoint must reproduce the stored one
    assert np.abs(p[:, -1] - 1).max() < 1e-3, np.abs(p[:, -1] - 1).max()
    return {"p": p, "arc": arc, "t": np.arange(D + 1) * step}


def per_image(G: np.ndarray) -> dict:
    d = np.sqrt(np.diag(G))
    cos = G / np.outer(d, d)
    total = G.sum()
    s = np.array([G[k, k + 1 :].sum() / G[k, k] for k in range(8)])
    e = np.clip(-s, 0, 1)
    align = G.sum(1) / (d * math.sqrt(total))
    return dict(cos=cos, chord=d, net=math.sqrt(total), s=s,
                E=(e * d[:8]).sum() / d[:8].sum(), align=align)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gram = torch.load(audit.OUT / "gram_local.pt", weights_only=False)
    rows, block_rows, lines = [], [], []
    for run in CONFIGS:
        G = gram[run]["G"]
        q = [per_image(g) for g in G]
        cos = np.stack([x["cos"] for x in q])
        chord = np.stack([x["chord"] for x in q])
        net = np.array([x["net"] for x in q])
        if run.endswith("_R100"):
            cache = OUT / f"progress_{run}.npz"
            if cache.is_file():
                mp = dict(np.load(cache))
            else:
                mp = micro_progress(run, device)
                np.savez(cache, **mp)
            arc = mp["arc"]
            p, t = mp["p"], mp["t"]
            t50 = t[(p >= 0.5).argmax(axis=1)]
        else:
            arc = chord.copy()
            states = np.load(audit.OUT / f"states_{run}.npy", mmap_mode="r")
            p = np.zeros((len(G), 10), dtype=np.float32)
            for i in range(len(G)):
                xb = np.asarray(states[i], dtype=np.float64).reshape(10, -1)
                v = xb[-1] - xb[0]
                p[i] = (xb - xb[0]) @ v / (v @ v)
            t = np.arange(10, dtype=float)
            np.savez(OUT / f"progress_{run}.npz", p=p, t=t, arc=arc)
            t50 = np.full(len(G), np.nan)
        off = ~np.eye(9, dtype=bool)
        mean_cos = cos.mean(0)
        np.save(OUT / f"cos_{run}.npy", mean_cos)
        S = np.stack([x["s"] for x in q])
        A = np.stack([x["align"] for x in q])
        E = np.array([x["E"] for x in q])
        st_arc = net / arc.sum(1)
        st_chord = net / chord.sum(1)
        factor = chord.sum(1) / arc.sum(1)
        consec = np.array([np.mean([c[k, k + 1] for k in range(8)]) for c in cos])
        skip2 = np.array([np.mean([c[k, k + 2] for k in range(7)]) for c in cos])
        offd = np.array([c[off].mean() for c in cos])
        row = dict(config=LABELS[run], n=len(G),
                   straight_arc=st_arc.mean(), straight_arc_std=st_arc.std(ddof=1),
                   straight_chord=st_chord.mean(), straight_chord_std=st_chord.std(ddof=1),
                   intra_factor=factor.mean(), intra_factor_std=factor.std(ddof=1),
                   cos_k_k1=consec.mean(), cos_k_k1_std=consec.std(ddof=1),
                   cos_k_k2=skip2.mean(), cos_k_k2_std=skip2.std(ddof=1),
                   cos_offdiag=offd.mean(), cos_offdiag_std=offd.std(ddof=1),
                   s_min=S.min(), s_neg_frac=(S < 0).mean(), E=E.mean(), E_std=E.std(ddof=1),
                   align_mean=A.mean(),
                   t50=np.nanmean(t50), t50_std=np.nanstd(t50, ddof=1) if np.isfinite(t50).any() else np.nan)
        rows.append(row)
        for k in range(9):
            fk = chord[:, k] / arc[:, k]
            block_rows.append(dict(
                config=LABELS[run], k=k,
                s_k=S[:, k].mean() if k < 8 else np.nan,
                s_k_p10=np.percentile(S[:, k], 10) if k < 8 else np.nan,
                s_k_p90=np.percentile(S[:, k], 90) if k < 8 else np.nan,
                align=A[:, k].mean(), align_p10=np.percentile(A[:, k], 10),
                align_p90=np.percentile(A[:, k], 90),
                chord=chord[:, k].mean(), arc=arc[:, k].mean(), intra_factor=fk.mean(),
                # equivalent circular-arc turning angle: chord/arc = sin(θ/2)/(θ/2)
                turn_deg=float(np.degrees(_turn_angle(fk.mean())))))
        lines.append(f"{LABELS[run]}: p(t) at integer t = "
                     + ", ".join(f"{p[:, int(round(k / (t[1] - t[0])))].mean():.3f}" for k in range(10)))
    summary = pd.DataFrame(rows)
    blocks = pd.DataFrame(block_rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    blocks.to_csv(OUT / "per_block.csv", index=False)
    text = ["# Transport coherence (500 images, massive channel removed)", "",
            summary.round(4).to_string(index=False), "", blocks.round(4).to_string(index=False),
            "", "Projected progress p(t) at integer t (mean over images):", ""] + [f"- {l}" for l in lines]
    (OUT / "report.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


def _turn_angle(ratio: float) -> float:
    """Solve sin(θ/2)/(θ/2) = ratio for θ in [0, 2π) by bisection."""
    if ratio >= 1:
        return 0.0
    lo, hi = 0.0, 2 * math.pi - 1e-9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if math.sin(mid / 2) / (mid / 2) > ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    main()
