#!/usr/bin/env python3
"""Refined vs native R=1 stage-3 path: state norms, L2 distance and cosine along depth.

Writes
  figures/dist_cos_r1.pdf          ConvNeXt-T only (Section 3.2)
  figures/dist_cos_r1_compare.pdf  ConvNeXt-T vs weight-tied side by side (Section 3.3)

Reads the N=500 feature-map suite (massive channel ignored; same images in every run).

Usage (from repo root):
    python paper/scripts/plot_dist_r1.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
COLOR = "#1f77b4"
COLOR_R1 = "#ff7f0e"
COLOR_X = "0.45"

RUNS = {
    "convnext": {
        "fine": ROOT / "outputs/featureMaps_interpoled/convnext_R100_ES0.01_N500_ignore1",
        "native": ROOT / "outputs/featureMaps_interpoled/convnext_R1_ES1_N500_ignore1",
        "step": 0.01,
        "fine_label": r"$R{=}100$",
        "title": r"ConvNeXt-T ($R{=}100$, $\Delta t{=}0.01$)",
    },
    "shared": {
        "fine": ROOT / "outputs/featureMaps/convnext_shared_D900_ES0.01_N500_ignore1",
        "native": ROOT / "outputs/featureMaps/convnext_shared_D9_ES1_N500_ignore1",
        "step": 0.01,
        "fine_label": r"$R{=}100$",
        "title": r"Weight-tied ($R{=}100$, $\Delta t{=}0.01$)",
    },
}


def _band(ax, t, df: pd.DataFrame, col: str, *, as_cos: bool, label: str) -> None:
    tf = (lambda v: 1.0 - v) if as_cos else (lambda v: v)
    mean, q05, q95 = tf(df[col]), tf(df[f"{col}_q05"]), tf(df[f"{col}_q95"])
    if as_cos:
        q05, q95 = q95, q05
    ax.fill_between(t, q05, q95, color=COLOR, alpha=0.18, lw=0)
    ax.plot(t, mean, color=COLOR, lw=1.4, label=label)
    end = df["is_block_end"].astype(bool)
    ax.plot(t[end], tf(df.loc[end, col]), "o", color="black", ms=3, label="block end")


def _style(ax) -> None:
    for k in range(1, 9):
        ax.axvline(k, color="0.85", lw=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def draw_column(ax_d, ax_c, key: str, *, legends: bool) -> pd.DataFrame:
    spec = RUNS[key]
    pairs = pd.read_csv(spec["fine"] / "table_pairs.csv")
    norms = pd.read_csv(spec["fine"] / "table_norms.csv")
    norms_r1 = pd.read_csv(spec["native"] / "table_norms.csv")
    t = pairs["t"] + spec["step"]  # state after micro-step d sits at t_{d+1}

    ax_d.plot(norms["t"], norms["norm_x"], color=COLOR_X, lw=1.2, ls="--",
              label=rf"$\|x^{{(R)}}\|$, {spec['fine_label']}")
    ax_d.plot(norms_r1["t"], norms_r1["norm_x"], "s", color=COLOR_R1, ms=3.5,
              label=r"$\|x_{R=1}\|$")
    _band(ax_d, t, pairs, "dist_x_r1", as_cos=False, label=r"$\|x^{(R)} - x_{R=1}\|$")
    _band(ax_c, t, pairs, "cos_dist_x_r1", as_cos=True, label="mean (band: P5–P95)")
    for ax in (ax_d, ax_c):
        _style(ax)
    ax_c.set_xlim(0, 9)
    ax_c.set_xticks(range(10))
    ax_c.set_xlabel(r"depth $t$", fontsize=9)
    if legends:
        ax_d.legend(fontsize=6.5, frameon=False, loc="upper left")
        ax_c.legend(fontsize=6.5, frameon=False, loc="lower left")
    return pairs


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def plot_single(out_path: Path = FIG / "dist_cos_r1.pdf") -> None:
    fig, (ax_d, ax_c) = plt.subplots(2, 1, figsize=(3.4, 3.9), sharex=True)
    pairs = draw_column(ax_d, ax_c, "convnext", legends=True)
    ax_d.set_ylabel("norm / distance", fontsize=9)
    ax_c.set_ylabel(r"$\cos(x^{(R)}, x_{R=1})$", fontsize=9)
    fig.tight_layout(h_pad=0.4)
    _save(fig, out_path)
    end = pairs[pairs["is_block_end"].astype(bool)]
    print("  convnext block-end cos:", (1 - end["cos_dist_x_r1"]).round(3).tolist())


def plot_compare(out_path: Path = FIG / "dist_cos_r1_compare.pdf") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 3.9), sharex=True, sharey="row")
    for col, key in enumerate(("convnext", "shared")):
        pairs = draw_column(axes[0, col], axes[1, col], key, legends=(col == 0))
        axes[0, col].set_title(RUNS[key]["title"], fontsize=9)
        end = pairs[pairs["is_block_end"].astype(bool)]
        print(f"  {key} block-end cos:", (1 - end["cos_dist_x_r1"]).round(3).tolist())
    axes[0, 0].set_ylabel("norm / distance", fontsize=9)
    axes[1, 0].set_ylabel(r"$\cos(x^{(R)}, x_{R=1})$", fontsize=9)
    fig.tight_layout(h_pad=0.4, w_pad=0.8)
    _save(fig, out_path)


if __name__ == "__main__":
    plot_single()
    plot_compare()
