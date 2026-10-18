#!/usr/bin/env python3
"""Weight-tied ConvNeXt accuracy for fractional depth refinement (N < 1).

Fixed horizon T=9 with D < 9 (equivalently N = D/9 in (0, 1)).
Baseline trained depth is D=9 ↔ N=1.

Usage (from repo root):
    python paper/scripts/plot_shared_fractional.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "shared_fractional.pdf"
SHARED_CSV = ROOT / "outputs/sharedConvnextAblation/results.csv"
SHARED_BASELINE_DEPTH = 9.0
COLOR = "#9467bd"


def _load_shared_t9(method: str) -> pd.DataFrame:
    df = pd.read_csv(SHARED_CSV)
    m = df[df["method"] == method].copy()
    if "T" in m.columns:
        refine = m[(m["T"] - 9.0).abs() < 1e-3].copy()
    else:
        refine = m[((m["depth"] * m["euler_step"]) - 9.0).abs() < 1e-2].copy()
    refine = refine.sort_values("depth").drop_duplicates(subset=["depth"], keep="first")
    refine["R"] = refine["depth"].astype(float) / SHARED_BASELINE_DEPTH
    return refine.sort_values("R")


def plot_shared_fractional(out_path: Path = OUT) -> Path:
    rk1 = _load_shared_t9("RK1")
    rk4 = _load_shared_t9("RK4")
    # Fractional steps plus the R=1 baseline for reference.
    rk1 = rk1[rk1["R"] <= 1.0 + 1e-9]
    rk4 = rk4[rk4["R"] <= 1.0 + 1e-9]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(
        rk1["R"],
        100.0 * rk1["top1acc"].to_numpy(),
        color=COLOR,
        lw=2.0,
        marker="o",
        ms=5.0,
        label="RK1",
    )
    ax.plot(
        rk4["R"],
        100.0 * rk4["top1acc"].to_numpy(),
        color=COLOR,
        lw=1.6,
        ls="--",
        marker="s",
        ms=4.5,
        label="RK4",
    )
    ax.axvline(1.0, color="0.7", lw=0.8, ls=":", zorder=0)

    ax.set_xscale("log")
    ax.set_xlabel(r"$R$ (repetitions; $9R$ shared steps, $\Delta t{=}1/R$)", fontsize=10)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=11)
    ax.set_xlim(0.08, 1.2)
    ax.set_xticks([1 / 9, 2 / 9, 3 / 9, 4 / 9, 6 / 9, 1.0])
    ax.set_xticklabels([r"$1/9$", r"$2/9$", r"$3/9$", r"$4/9$", r"$6/9$", r"$1$"])
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.55)
    ax.legend(
        handles=[
            Line2D([0], [0], color=COLOR, lw=2.0, marker="o", ms=5.0, label="RK1"),
            Line2D([0], [0], color=COLOR, lw=1.6, ls="--", marker="s", ms=4.5, label="RK4"),
        ],
        fontsize=9,
        loc="lower right",
        frameon=False,
    )
    ax.set_title(r"Weight-tied ConvNeXt: fractional $R\leq 1$ ($T{=}9$)", fontsize=11, pad=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"RK1 depths: {rk1['depth'].astype(int).tolist()}")
    print(f"RK4 depths: {rk4['depth'].astype(int).tolist()}")
    return out_path


if __name__ == "__main__":
    plot_shared_fractional()
