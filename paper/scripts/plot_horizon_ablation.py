#!/usr/bin/env python3
"""Shared ConvNeXt: top-1 vs integration horizon T.

Two sweeps at once:
  - fixed step Δt=1, vary depth D (so T=D)
  - fixed depth D=9, vary step (so T=9·Δt)

Usage (from repo root):
    python paper/scripts/plot_horizon_ablation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "horizon_ablation.pdf"
SHARED_CSV = ROOT / "outputs/sharedConvnextAblation/results.csv"
COLOR_GROW = "#d62728"
COLOR_FIXED_D = "#9467bd"


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.55)


def plot_horizon(out_path: Path = OUT) -> Path:
    df = pd.read_csv(SHARED_CSV)
    rk1 = df[df["method"] == "RK1"].copy()

    grow = rk1[(rk1["euler_step"] - 1.0).abs() < 1e-6].sort_values("depth")
    grow = grow.drop_duplicates(subset=["depth"], keep="first")
    # Keep a readable range around the peak (drop extreme collapse tails for clarity).
    grow = grow[(grow["T"] >= 0.5) & (grow["T"] <= 128)]

    fixed_d = rk1[rk1["depth"] == 9].sort_values("euler_step")
    fixed_d = fixed_d.drop_duplicates(subset=["euler_step"], keep="first")
    fixed_d = fixed_d[(fixed_d["T"] >= 0.05) & (fixed_d["T"] <= 72)]

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(
        grow["T"],
        100.0 * grow["top1acc"],
        color=COLOR_GROW,
        lw=1.8,
        marker="o",
        ms=4.0,
        label=r"Fixed $\Delta t{=}1$ (more steps, $T$ grows)",
    )
    ax.plot(
        fixed_d["T"],
        100.0 * fixed_d["top1acc"],
        color=COLOR_FIXED_D,
        lw=1.8,
        marker="s",
        ms=4.0,
        label=r"Fixed $R{=}1$ (9 steps; $T{=}9\cdot\Delta t$)",
    )
    ax.axvline(9.0, color="0.35", lw=1.0, ls="--", alpha=0.8)
    ax.text(9.15, 35.0, r"$T{=}9$", fontsize=9, color="0.35")

    ax.set_xscale("log")
    ax.set_xlabel(r"Integration horizon $T$", fontsize=11)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=11)
    ax.set_ylim(-2, 88)
    ax.legend(fontsize=8, frameon=False, loc="lower center")
    ax.tick_params(labelsize=9)
    _style(ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    plot_horizon()
