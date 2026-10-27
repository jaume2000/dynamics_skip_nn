#!/usr/bin/env python3
"""Figures for the transport-coherence analysis of stage 3 (Section 3.4).

  transport_cosine.pdf     mean 9x9 cosine of block increments: random untied (g*),
                           trained untied, weight-tied (all at R=100)
  transport_survival.pdf   vector survival s_k and alignment cos(dx_k, x_T - x_0) per block
  transport_progress.pdf   projected progress p(t) on the R=100 grid (mean, P10-P90)

Inputs: outputs/transport_coherence/ (scripts/transport_coherence.py).

Usage (from repo root):
    python paper/scripts/plot_transport_coherence.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs" / "transport_coherence"
FIG = ROOT / "paper" / "figures"

RUNS = [
    ("rand_g1.35_R100", "random untied ($g^\\star$)", "random untied g*=1.35 R=100", "0.45"),
    ("convnext_R100", "ConvNeXt-T (untied)", "untied R=100", "#1f77b4"),
    ("shared_R100", "Weight-tied", "tied R=100", "#9467bd"),
]


def _save(fig, name: str) -> None:
    for ext in (".pdf", ".png"):
        fig.savefig(FIG / f"{name}{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {FIG / name}.pdf/.png")


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.5)
    ax.tick_params(labelsize=7.5)


def plot_cosine() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3), constrained_layout=True)
    for ax, (run, title, _, _) in zip(axes, RUNS):
        c = np.load(SRC / f"cos_{run}.npy")
        im = ax.imshow(c, vmin=-1, vmax=1, cmap="RdBu_r")
        off = c[~np.eye(9, dtype=bool)].mean()
        ax.set_title(f"{title}\nmean off-diag. {off:.2f}", fontsize=8)
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("block $j$", fontsize=7.5)
    axes[0].set_ylabel("block $k$", fontsize=7.5)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label(r"$\cos(\Delta x_k,\Delta x_j)$", fontsize=7.5)
    _save(fig, "transport_cosine")


def plot_survival() -> None:
    df = pd.read_csv(SRC / "per_block.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.1), constrained_layout=True)
    for run, label, key, color in RUNS:
        d = df[df["config"] == key].sort_values("k")
        s = d.dropna(subset=["s_k"])
        axes[0].plot(s["k"], s["s_k"], "o-", color=color, ms=3.5, lw=1.3, label=label)
        axes[0].fill_between(s["k"], s["s_k_p10"], s["s_k_p90"], color=color, alpha=0.18, lw=0)
        axes[1].plot(d["k"], d["align"], "o-", color=color, ms=3.5, lw=1.3, label=label)
        axes[1].fill_between(d["k"], d["align_p10"], d["align_p90"], color=color, alpha=0.18, lw=0)
    axes[0].axhline(0, color="k", lw=0.7)
    axes[0].axhline(-1, color="k", lw=0.7, ls=":")
    axes[0].set_ylabel(r"survival $s_k$", fontsize=9)
    axes[1].set_ylabel(r"$\cos(\Delta x_k,\,x_T-x_0)$", fontsize=9)
    axes[1].set_ylim(0, 1)
    for ax, n in zip(axes, (8, 9)):
        _style(ax)
        ax.set_xticks(range(n))
        ax.set_xlabel("block $k$", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)
    _save(fig, "transport_survival")


def plot_progress() -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.2), constrained_layout=True)
    ax.plot([0, 9], [0, 1], color="0.7", lw=0.8, ls="--", zorder=0)
    for run, label, _, color in RUNS:
        z = np.load(SRC / f"progress_{run}.npz")
        p, t = z["p"], z["t"]
        ax.plot(t, p.mean(0), color=color, lw=1.4, label=label)
        ax.fill_between(t, np.percentile(p, 10, 0), np.percentile(p, 90, 0), color=color, alpha=0.18, lw=0)
    for k in range(1, 9):
        ax.axvline(k, color="0.85", lw=0.5, zorder=0)
    ax.axhline(0.5, color="0.6", lw=0.6, ls=":")
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.05, 1.02)
    ax.set_xticks(range(10))
    ax.set_xlabel("stage-3 depth $t$ ($R{=}100$)", fontsize=9)
    ax.set_ylabel(r"progress $p(t)$", fontsize=9)
    _style(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _save(fig, "transport_progress")


def plot_summary() -> None:
    """Body figure: cosine heatmaps and p(t) on top, survival and alignment below."""
    df = pd.read_csv(SRC / "per_block.csv")
    fig = plt.figure(figsize=(7.0, 3.7))
    W, H = 7.0, 3.7
    side, y0 = 1.25, 2.05
    for i, (run, title, _, _) in enumerate(RUNS):
        ax = fig.add_axes([(0.45 + i * 1.42) / W, y0 / H, side / W, side / H])
        c = np.load(SRC / f"cos_{run}.npy")
        im = ax.imshow(c, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.set_xticks(range(0, 9, 2))
        ax.set_yticks(range(0, 9, 2))
        ax.tick_params(labelsize=6, length=2)
        ax.set_xlabel("block $j$", fontsize=7, labelpad=1)
        if i == 0:
            ax.set_ylabel("block $k$", fontsize=7, labelpad=1)
    cax = fig.add_axes([(0.45 + 2 * 1.42 + side + 0.07) / W, y0 / H, 0.08 / W, side / H])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label(r"$\cos(\Delta x_k,\Delta x_j)$", fontsize=7)

    ax = fig.add_axes([5.62 / W, y0 / H, 1.33 / W, side / H])
    ax.plot([0, 9], [0, 1], color="0.7", lw=0.8, ls="--", zorder=0)
    for run, label, _, color in RUNS:
        z = np.load(SRC / f"progress_{run}.npz")
        p, t = z["p"], z["t"]
        ax.plot(t, p.mean(0), color=color, lw=1.3, label=label)
        ax.fill_between(t, np.percentile(p, 10, 0), np.percentile(p, 90, 0), color=color, alpha=0.18, lw=0)
    ax.axhline(0.5, color="0.6", lw=0.6, ls=":")
    ax.set_xlim(0, 9)
    ax.set_xticks(range(0, 10, 3))
    ax.set_xlabel("depth $t$ ($R{=}100$)", fontsize=7, labelpad=1)
    ax.set_ylabel(r"progress $p(t)$", fontsize=7, labelpad=1)
    _style(ax)
    ax.tick_params(labelsize=6, length=2)

    for j, (col, lab, n) in enumerate([("s_k", r"survival $s_k$", 8),
                                       ("align", r"$\cos(\Delta x_k,\,x_T-x_0)$", 9)]):
        ax = fig.add_axes([(0.45 + j * 3.35) / W, 0.35 / H, 3.05 / W, 1.25 / H])
        for run, label, key, color in RUNS:
            d = df[df["config"] == key].sort_values("k").dropna(subset=[col])
            ax.plot(d["k"], d[col], "o-", color=color, ms=3, lw=1.2, label=label)
            ax.fill_between(d["k"], d[f"{col}_p10"], d[f"{col}_p90"], color=color, alpha=0.18, lw=0)
        if col == "s_k":
            ax.axhline(0, color="k", lw=0.7)
            ax.legend(frameon=False, fontsize=6.5)
        else:
            ax.set_ylim(0, 1)
        ax.set_xticks(range(n))
        ax.set_xlabel("block $k$", fontsize=7, labelpad=1)
        ax.set_ylabel(lab, fontsize=7, labelpad=1)
        _style(ax)
        ax.tick_params(labelsize=6, length=2)
    _save(fig, "transport_summary")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    plot_cosine()
    plot_survival()
    plot_progress()
    plot_summary()
