#!/usr/bin/env python3
"""Plot ImageNet top-1 of S4+head on stage-3 prefixes (tied vs untied).

Requires outputs/prefix_s4_head_acc/results.csv from scripts/prefix_s4_head_acc.py.

Usage (from repo root):
    python paper/scripts/plot_prefix_s4_head_acc.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "outputs" / "prefix_s4_head_acc" / "results.csv"
OUT = ROOT / "paper" / "figures" / "prefix_s4_head_acc.pdf"

SERIES = [
    ("shared_D9", "Weight-tied", "#9467bd"),
    ("convnext", "ConvNeXt-T", "#1f77b4"),
]


def plot_prefix_acc(out_path: Path = OUT) -> Path:
    df = pd.read_csv(CSV)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for model, label, color in SERIES:
        m = df[df["model"] == model].sort_values("block")
        ax.plot(
            m["block"],
            100.0 * m["top1acc"].to_numpy(),
            color=color,
            lw=2.0,
            marker="o",
            ms=5.0,
            label=label,
        )
    ax.set_xlabel("Stage-3 blocks applied")
    ax.set_ylabel("Top-1 (%)")
    ax.set_xticks(range(1, 10))
    ax.set_ylim(0, 85)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = plot_prefix_acc()
    print(f"Wrote {path}")
