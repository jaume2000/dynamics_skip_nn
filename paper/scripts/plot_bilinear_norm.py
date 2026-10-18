#!/usr/bin/env python3
"""ConvNeXt residual-norm: plain vs bilinear (R=100), with across-image bands.

Uses the N=500 suite tables (P5–P95), same style as the teaser right panel.
Refinement is N=100 Euler sub-steps per block (Δt=0.01).

Usage (from repo root):
    python paper/scripts/plot_bilinear_norm.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "bilinear_norm.pdf"
PLAIN = ROOT / "outputs/featureMaps_interpoled/convnext_R100_ES0.01_N500/table_norms.csv"
BILINEAR = ROOT / "outputs/featureMaps_interpoled/convnext_R100_ES0.01_bilinear_N500/table_norms.csv"
COLOR_PLAIN = "#1f77b4"
COLOR_BILINEAR = "#ff7f0e"


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.55)


def _plot(ax: plt.Axes, df: pd.DataFrame, *, color: str, label: str) -> None:
    x = df["t"].to_numpy(dtype=float)
    ax.fill_between(x, df["norm_h_q05"], df["norm_h_q95"], color=color, alpha=0.28, lw=0, zorder=1)
    ax.plot(x, df["norm_h"].to_numpy(dtype=float), color=color, lw=1.4, label=label, zorder=2)


def plot_bilinear_norm(out_path: Path = OUT) -> Path:
    plain = pd.read_csv(PLAIN)
    bilin = pd.read_csv(BILINEAR)

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    _plot(ax, plain, color=COLOR_PLAIN, label=r"Plain ($\theta_t=\theta_{\lfloor t \rfloor}$)")
    _plot(ax, bilin, color=COLOR_BILINEAR, label=r"Bilinear ($\theta_t$ lerp)")
    for k in range(1, 9):
        ax.axvline(k, color="0.75", lw=0.6, ls=":", zorder=0)

    ax.set_xlabel(r"Depth $t$ (stage 3)", fontsize=11)
    ax.set_ylabel(r"$\|f_k(x)\|$", fontsize=11)
    ax.set_xlim(0, 9)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.tick_params(labelsize=9)
    _style(ax)
    ax.set_title(r"ConvNeXt-T, $R{=}100$ (mean $\pm$ P5--P95, 500 images)", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    plot_bilinear_norm()
