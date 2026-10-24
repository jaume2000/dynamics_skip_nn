#!/usr/bin/env python3
"""Hero figure: stage-3 trajectories of the weight-tied and the original ConvNeXt-T.

Reads the caches written by ``plot_trajectory_pca.py`` (projected states without the
massive channel) and draws, per model, the PCA of all states with every path coloured by
depth t, a few highlighted paths with dots at integer t, and the speed ‖f(x(t))‖ below.

Writes figures/hero_trajectories.{pdf,png}.

Usage (from repo root):
    python paper/scripts/plot_hero.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_trajectory_pca import CACHE_DIR, CLASSES, FIG, pca  # noqa: E402

RUNS = [
    ("convnext_shared_D100_ES0.09", "Weight-tied: one autonomous field",
     r"$\dot x = f(x)$"),
    ("convnext_R100_ES0.01", "Independent blocks: a new field every unit of time",
     r"$\dot x = f_{\lfloor t\rfloor}(x)$"),
]
HIGHLIGHT = [289, 404, 985]
T_END = 9.0
CMAP = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "plasma_trunc", plt.get_cmap("plasma")(np.linspace(0.0, 0.85, 256)))
NORM = plt.Normalize(0.0, T_END)


def load(tag: str) -> dict:
    """Projected trajectories in their own PCA plane; cached as ``<tag>_pca2.npz``."""
    small = CACHE_DIR / f"{tag}_pca2.npz"
    if small.exists():
        return dict(np.load(small))
    data = torch.load(CACHE_DIR / f"{tag}.pt", weights_only=False)
    X = data.pop("states").numpy()
    n, T, P = X.shape
    mu, V, ev = pca(X.reshape(-1, P))
    Z = X @ V.T - mu @ V.T
    del X
    Z[..., 0] *= np.sign(Z[:, -1, 0].mean() - Z[:, 0, 0].mean())
    Z[..., 1] *= np.sign(np.median(Z[:, -1, 1]) - np.median(Z[:, 0, 1])) or 1.0
    run = {
        "Z": Z,
        "t": data["t"].numpy(),
        "labels": data["labels"].numpy(),
        "speeds": data["speeds"].numpy(),
        "t_speed": data.get("t_speed", data["t"]).numpy(),
        "var": np.float64(ev[:2].sum()),
    }
    np.savez(small, **run)
    return run


def coloured_line(ax, xy: np.ndarray, t: np.ndarray, **kw) -> None:
    seg = np.stack([xy[:-1], xy[1:]], axis=1)
    lc = LineCollection(seg, cmap=CMAP, norm=NORM, capstyle="round", **kw)
    lc.set_array(0.5 * (t[:-1] + t[1:]))
    ax.add_collection(lc)


def draw_paths(ax, run: dict) -> None:
    Z, t, labels = run["Z"], run["t"], run["labels"]
    for z in Z:
        coloured_line(ax, z, t, linewidths=0.45, alpha=0.35, zorder=1)
    ints = [int(np.abs(t - k).argmin()) for k in range(1, int(T_END))]
    for c in HIGHLIGHT:
        i = int(np.flatnonzero(labels == c)[0])
        ax.plot(Z[i, :, 0], Z[i, :, 1], color="white", lw=3.2, zorder=2, solid_capstyle="round")
        coloured_line(ax, Z[i], t, linewidths=1.8, zorder=3)
        ax.scatter(Z[i, ints, 0], Z[i, ints, 1], s=11, c="white", edgecolors="0.15",
                   linewidths=0.6, zorder=4)
    ax.scatter(Z[:, 0, 0], Z[:, 0, 1], s=4, c="0.55", lw=0, zorder=2)
    ax.scatter(Z[:, -1, 0], Z[:, -1, 1], s=7, c=[CMAP(1.0)], edgecolors="0.2",
               linewidths=0.25, zorder=2)
    ax.autoscale()
    ax.margins(0.04)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"PC1   (PC1–2: {100 * float(run['var']):.0f}% of the variance)", fontsize=8, color="0.35")
    ax.set_ylabel("PC2", fontsize=8, color="0.35")
    for s in ax.spines.values():
        s.set_visible(False)


def draw_speed(ax, run: dict) -> None:
    ts, sp = run["t_speed"], run["speeds"]
    lo, hi = np.percentile(sp, [10, 90], axis=0)
    ax.fill_between(ts, lo, hi, color="0.85", lw=0)
    coloured_line(ax, np.stack([ts, sp.mean(0)], axis=1), ts, linewidths=2.2)
    for k in range(1, int(T_END)):
        ax.axvline(k, color="0.75", lw=0.5, ls=":", zorder=0)
    ax.set_xlim(0, T_END)
    ax.set_ylim(0, hi.max() * 1.05)
    ax.set_xlabel(r"depth $t$", fontsize=8)
    ax.set_ylabel(r"speed $\|f(x(t))\|$", fontsize=8)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.6), layout="constrained",
                             gridspec_kw={"height_ratios": [3.2, 1]})
    for col, (tag, title, eq) in enumerate(RUNS):
        run = load(tag)
        draw_paths(axes[0, col], run)
        axes[0, col].set_title(f"{title}\n{eq}", fontsize=10, loc="left")
        draw_speed(axes[1, col], run)
        del run
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    cb = fig.colorbar(sm, ax=axes[:, 1], shrink=0.6, aspect=30, pad=0.01)
    cb.set_label(r"depth $t$", fontsize=8)
    cb.set_ticks(range(0, int(T_END) + 1))
    cb.ax.tick_params(labelsize=7)
    names = ", ".join(CLASSES[c] for c in HIGHLIGHT)
    print(f"highlighted: {names}")
    out = FIG / "hero_trajectories.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
