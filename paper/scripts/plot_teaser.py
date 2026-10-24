#!/usr/bin/env python3
"""Build the paper teaser figure (accuracy vs N | residual-norm along depth).

Usage (from repo root):
    python paper/scripts/plot_teaser.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "teaser.pdf"

REFINE_SOURCES = {
    "ConvNeXt": ROOT / "outputs/convnext_interpolation/results.csv",
    "ResNet-50": ROOT / "outputs/resnet_interpolation/results.csv",
    "ResNet-101": ROOT / "outputs/resnet101_interpolation/results.csv",
    "Swin": ROOT / "outputs/swin_interpolation/results.csv",
}
SHARED_CSV = ROOT / "outputs/sharedConvnextAblation/results.csv"
SHARED_BASELINE_DEPTH = 9.0  # R=1 ↔ D=9, so N = D / 9

# Residual-field norms: N500 means + variance bands (same style as feature_map_explorer).
NORM_REFINED = ROOT / "outputs/featureMaps_interpoled/convnext_R100_ES0.01_N500/table_norms.csv"
NORM_UNIT = ROOT / "outputs/featureMaps_interpoled/convnext_R1_ES1_N500/table_norms.csv"
NORM_SHARED_REFINED = ROOT / "outputs/featureMaps/convnext_shared_D900_ES0.01_N500/table_norms.csv"
NORM_SHARED_UNIT = ROOT / "outputs/featureMaps/convnext_shared_D9_ES1_N500/table_norms.csv"

COLORS = {
    "ConvNeXt": "#1f77b4",
    "ResNet-50": "#ff7f0e",
    "ResNet-101": "#2ca02c",
    "Swin": "#d62728",
    "Weight-tied": "#9467bd",
    # Distinct hues for unit-step overlays on the right panel.
    "ConvNeXt-unit": "#17becf",
    "Weight-tied-unit": "#e377c2",
}

# Shared marker / line weights (kept light so overlapping series stay readable).
LW_RK1 = 1.15
LW_RK4 = 0.95
MS_RK1 = 2.8
MS_RK4 = 2.4
LW_NORM = 1.05
LW_NORM_UNIT = 0.95
MS_NORM_UNIT = 3.0


def _load_refine(csv_path: Path, method: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    m = df[df["method"] == method].copy()
    refine = m[(m["euler_step"] * m["repeats"] - 1.0).abs() < 1e-6].copy()
    refine["R"] = refine["repeats"].astype(float)
    return refine.sort_values("R")


def _load_shared_t9(method: str) -> pd.DataFrame:
    """Fixed horizon T=9. R = D/9 so baseline D=9 maps to R=1 (same as non-tied)."""
    df = pd.read_csv(SHARED_CSV)
    m = df[df["method"] == method].copy()
    if "T" in m.columns:
        refine = m[(m["T"] - 9.0).abs() < 1e-3].copy()
    else:
        refine = m[((m["depth"] * m["euler_step"]) - 9.0).abs() < 1e-2].copy()
    refine = refine.sort_values("depth").drop_duplicates(subset=["depth"], keep="first")
    refine["R"] = refine["depth"].astype(float) / SHARED_BASELINE_DEPTH
    return refine.sort_values("R")


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.55)


def _plot_norm_with_spread(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    color: str,
    label: str,
    lw: float = 1.5,
    ls: str = "-",
    marker: str | None = None,
    ms: float = 4.0,
    alpha: float = 1.0,
    band_alpha: float = 0.22,
) -> None:
    """Mean ``norm_h`` with a P5–P95 across-image band (N=500)."""
    x = df["t"].to_numpy(dtype=float)
    ax.fill_between(
        x,
        df["norm_h_q05"],
        df["norm_h_q95"],
        color=color,
        alpha=band_alpha,
        lw=0,
        zorder=1,
    )
    ax.plot(
        x,
        df["norm_h"].to_numpy(dtype=float),
        color=color,
        lw=lw,
        ls=ls,
        marker=marker,
        ms=ms,
        mew=0.4,
        alpha=alpha,
        label=label,
        zorder=2,
    )


def plot_teaser(out_path: Path = OUT) -> Path:
    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(12.0, 3.4),
        gridspec_kw={"wspace": 0.22},
    )

    # ---- Left: top-1 vs N (RK1 solid/circles, RK4 dashed/squares) ----
    y_vals: list[float] = []
    for label, path in REFINE_SOURCES.items():
        color = COLORS[label]
        rk1 = _load_refine(path, "RK1")
        rk4 = _load_refine(path, "RK4")
        rk1_y = 100.0 * rk1["top1acc"].to_numpy()
        rk4_y = 100.0 * rk4["top1acc"].to_numpy()
        y_vals.extend(rk1_y.tolist())
        y_vals.extend(rk4_y.tolist())
        ax_l.plot(
            rk1["R"], rk1_y, color=color, lw=LW_RK1, marker="o", ms=MS_RK1, mew=0.4, label=label
        )
        ax_l.plot(
            rk4["R"], rk4_y, color=color, lw=LW_RK4, ls="--", marker="s", ms=MS_RK4, mew=0.4
        )

    # Weight-tied: only R >= 1 (fractional R < 1 are in plot_shared_fractional.py).
    shared_rk1 = _load_shared_t9("RK1")
    shared_rk4 = _load_shared_t9("RK4")
    shared_rk1 = shared_rk1[shared_rk1["R"] >= 1.0 - 1e-9]
    shared_rk4 = shared_rk4[shared_rk4["R"] >= 1.0 - 1e-9]
    shared_rk1_y = 100.0 * shared_rk1["top1acc"].to_numpy()
    shared_rk4_y = 100.0 * shared_rk4["top1acc"].to_numpy()
    y_vals.extend(shared_rk1_y.tolist())
    y_vals.extend(shared_rk4_y.tolist())

    color = COLORS["Weight-tied"]
    ax_l.plot(
        shared_rk1["R"],
        shared_rk1_y,
        color=color,
        lw=LW_RK1,
        marker="o",
        ms=MS_RK1,
        mew=0.4,
        label="Weight-tied",
    )
    ax_l.plot(
        shared_rk4["R"],
        shared_rk4_y,
        color=color,
        lw=LW_RK4,
        ls="--",
        marker="s",
        ms=MS_RK4,
        mew=0.4,
    )

    model_handles = [
        Line2D(
            [0], [0], color=COLORS[name], lw=LW_RK1, marker="o", ms=MS_RK1, mew=0.4, label=name
        )
        for name in list(REFINE_SOURCES) + ["Weight-tied"]
    ]
    style_handles = [
        Line2D([0], [0], color="0.25", lw=LW_RK1, marker="o", ms=MS_RK1, mew=0.4, label="RK1"),
        Line2D(
            [0], [0], color="0.25", lw=LW_RK4, ls="--", marker="s", ms=MS_RK4, mew=0.4, label="RK4"
        ),
    ]
    ax_l.legend(
        handles=model_handles + style_handles,
        fontsize=7.5,
        loc="upper right",
        frameon=False,
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=2.2,
        labelspacing=0.3,
        borderaxespad=0.3,
    )

    ax_l.set_xscale("log")
    ax_l.set_xlabel(r"$R$ (repetitions; $\Delta t{=}1/R$)", fontsize=11)
    ax_l.set_ylabel("Top-1 accuracy (%)", fontsize=11)
    ax_l.set_ylim(min(y_vals) - 0.4, max(y_vals) + 0.5)
    ax_l.set_xlim(0.8, 2.0e3)  # include weight-tied D=10000 (R≈1111)
    ax_l.tick_params(labelsize=9)
    _style_axes(ax_l)
    ax_l.set_title("Depth refinement", fontsize=11, pad=6)

    # ---- Right: ||f|| along depth (N500 mean + P5–P95 across images) ----
    refined = pd.read_csv(NORM_REFINED)
    unit = pd.read_csv(NORM_UNIT)
    shared_refined = pd.read_csv(NORM_SHARED_REFINED)
    shared_unit = pd.read_csv(NORM_SHARED_UNIT)

    _plot_norm_with_spread(
        ax_r,
        refined,
        color=COLORS["ConvNeXt"],
        label=r"ConvNeXt ($R{=}100$)",
        lw=LW_NORM,
        band_alpha=0.4,
    )
    _plot_norm_with_spread(
        ax_r,
        unit,
        color=COLORS["ConvNeXt-unit"],
        label=r"ConvNeXt ($R{=}1$)",
        lw=LW_NORM_UNIT,
        ls="--",
        marker="s",
        ms=MS_NORM_UNIT,
        alpha=0.55,
        band_alpha=0.12,
    )
    _plot_norm_with_spread(
        ax_r,
        shared_refined,
        color=COLORS["Weight-tied"],
        label=r"Weight-tied ($R{=}100$)",
        lw=LW_NORM,
        band_alpha=0.4,
    )
    _plot_norm_with_spread(
        ax_r,
        shared_unit,
        color=COLORS["Weight-tied-unit"],
        label=r"Weight-tied ($R{=}1$)",
        lw=LW_NORM_UNIT,
        ls="--",
        marker="s",
        ms=MS_NORM_UNIT,
        alpha=0.55,
        band_alpha=0.12,
    )
    for k in range(1, 9):
        ax_r.axvline(k, color="0.75", lw=0.6, ls=":", zorder=0)

    ax_r.legend(fontsize=7.5, loc="upper left", frameon=False, labelspacing=0.3)

    ax_r.set_xlabel(r"Depth $t$ (stage 3)", fontsize=11)
    ax_r.set_ylabel(r"$\|f_k(x)\|$", fontsize=11)
    ax_r.set_xlim(0, 9)
    ax_r.tick_params(labelsize=9)
    _style_axes(ax_r)
    ax_r.set_title("Residual update norm", fontsize=11, pad=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.canvas.draw()
    right = ax_r.get_tightbbox(fig.canvas.get_renderer())
    right = right.transformed(fig.dpi_scale_trans.inverted()).padded(0.05)
    speed_path = out_path.with_name("teaser_speed.png")
    fig.savefig(speed_path, bbox_inches=right, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path} and {speed_path}")
    return out_path


if __name__ == "__main__":
    plot_teaser()
