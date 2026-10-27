#!/usr/bin/env python3
"""ImageNet top-1 / CE of S4+head on stage-3 prefixes (tied vs untied).

Solid curves: R=20 (Δt=0.05) full-val runs.
Connected markers: native discrete prefixes (R=1, one eval per block).

Usage (from repo root):
    python paper/scripts/plot_prefix_s4_head_acc.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "prefix_s4_head_acc.pdf"

FINE = [
    {
        "csv": ROOT / "outputs/prefix_s4_head_acc_shared_R20/results.csv",
        "label": "Weight-tied",
        "color": "#7B2D8E",
        "disc_key": "shared_D9",
    },
    {
        "csv": ROOT / "outputs/prefix_s4_head_acc_convnext_R20/results.csv",
        "label": "ConvNeXt-T",
        "color": "#2F5FA6",
        "disc_key": "convnext",
    },
]
DISC = ROOT / "outputs/prefix_s4_head_acc/results.csv"


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle=":", linewidth=0.55, alpha=0.5)
    for k in range(1, 9):
        ax.axvline(k, color="0.82", lw=0.55, zorder=0)
    ax.set_xlim(0, 9)
    ax.set_xticks(range(0, 10, 1))
    ax.set_xlabel(r"Stage-3 depth $t$", fontsize=10, labelpad=2)
    ax.tick_params(labelsize=9)


def plot_prefix_acc(out_path: Path = OUT) -> Path:
    disc = pd.read_csv(DISC)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

    for spec in FINE:
        df = pd.read_csv(spec["csv"]).sort_values("t")
        color = spec["color"]
        axes[0].plot(
            df["t"], 100.0 * df["top1acc"],
            color=color, lw=1.55, zorder=2,
        )
        axes[1].plot(
            df["t"], df["loss"],
            color=color, lw=1.55, zorder=2,
        )

        d = disc.loc[disc["model"] == spec["disc_key"]].sort_values("block")
        mk = dict(color=color, ls="--", lw=1.15, marker="o", ms=4.2,
                  markerfacecolor="white", markeredgewidth=1.15, zorder=3)
        axes[0].plot(d["block"], 100.0 * d["top1acc"], **mk)
        if "loss" in d.columns:
            axes[1].plot(d["block"], d["loss"], **mk)

    axes[0].set_ylabel("Top-1 (%)", fontsize=9)
    axes[0].set_ylim(0, 85)
    axes[1].set_ylabel("Cross-entropy", fontsize=9)
    axes[1].set_ylim(0, 6)

    handles = []
    for spec in FINE:
        handles.append(Line2D([0], [0], color=spec["color"], lw=1.6, label=spec["label"] + r" $R{=}20$"))
        handles.append(
            Line2D(
                [0], [0], color=spec["color"], lw=1.15, ls="--", marker="o", ms=4.2,
                markerfacecolor="white", markeredgewidth=1.1,
                label=spec["label"] + r" native ($R{=}1$)",
            )
        )
    for ax in axes:
        _style(ax)
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=7.5,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.2,
        handlelength=2.2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Saved {out_path.with_suffix('.png')}")
    return out_path


if __name__ == "__main__":
    plot_prefix_acc()
