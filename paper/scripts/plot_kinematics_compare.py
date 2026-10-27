#!/usr/bin/env python3
"""Shared vs untied stage-3 kinematics (2x4), massive channel ignored.

Columns: ||v||, ω, acceleration decomposition (||a||, a_t, a_n), κ.
Rows: weight-tied ConvNeXt | standard ConvNeXt-T.
Inter-block pairs are dropped for derivative metrics on the untied row: the
piecewise field makes v discontinuous at integer depths, so finite-difference
ω/a/κ diverge as Δt→0 and are not representative.

Usage (from repo root):
    python paper/scripts/plot_kinematics_compare.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "kinematics_compare.pdf"

RUNS = {
    "shared": {
        "dir": ROOT / "outputs/featureMaps/convnext_shared_D900_ES0.01_N500_ignore1",
        "title": "Weight-tied",
        "color": "#9467bd",
    },
    "untied": {
        "dir": ROOT / "outputs/featureMaps_interpoled/convnext_R100_ES0.01_N500_ignore1",
        "title": "ConvNeXt-T (untied)",
        "color": "#1f77b4",
    },
}

# Acceleration panel (shared across both rows for legend consistency).
C_A = "#2ca02c"  # ||a||
C_AT = "#1f77b4"  # a_t
C_AN = "#ff7f0e"  # a_n

COL_TITLES = [
    r"Velocity $\|v\|$",
    r"Angular velocity $\omega$",
    r"Acceleration",
    r"Curvature $\kappa$",
]


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle=":", linewidth=0.55, alpha=0.5)
    for k in range(1, 9):
        ax.axvline(k, color="0.82", lw=0.55, zorder=0)
    ax.set_xlim(0, 9)
    ax.set_xticks(range(0, 10, 2))
    ax.tick_params(labelsize=7.5)


def _band(
    ax: plt.Axes,
    t: np.ndarray,
    mean: np.ndarray,
    q05: np.ndarray,
    q95: np.ndarray,
    *,
    color: str,
    lw: float = 1.25,
    label: str | None = None,
    z: int = 2,
) -> None:
    ax.fill_between(t, q05, q95, color=color, alpha=0.20, lw=0, zorder=z - 1)
    ax.plot(t, mean, color=color, lw=lw, label=label, zorder=z)


def _pairs_no_ib(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path / "table_pairs.csv")
    if "is_inter_block" in df.columns:
        df = df.loc[~df["is_inter_block"].astype(bool)].copy()
    return df


def _draw_velocity(ax: plt.Axes, run_dir: Path, color: str) -> None:
    n = pd.read_csv(run_dir / "table_norms.csv")
    _band(
        ax,
        n["t"].to_numpy(float),
        n["norm_h"].to_numpy(float),
        n["norm_h_q05"].to_numpy(float),
        n["norm_h_q95"].to_numpy(float),
        color=color,
        label=r"$\|v\|$",
    )
    ax.set_ylabel(r"$\|v\|$", fontsize=8)


def _draw_omega(ax: plt.Axes, pairs: pd.DataFrame, color: str) -> None:
    _band(
        ax,
        pairs["t"].to_numpy(float),
        pairs["omega_h"].to_numpy(float),
        pairs["omega_h_q05"].to_numpy(float),
        pairs["omega_h_q95"].to_numpy(float),
        color=color,
        label=r"$\omega$",
    )
    ax.set_ylabel(r"$\omega$ [rad$/t$]", fontsize=8)


def _draw_accel(ax: plt.Axes, pairs: pd.DataFrame) -> None:
    t = pairs["t"].to_numpy(float)
    _band(
        ax,
        t,
        pairs["acc_h"].to_numpy(float),
        pairs["acc_h_q05"].to_numpy(float),
        pairs["acc_h_q95"].to_numpy(float),
        color=C_A,
        lw=1.15,
        label=r"$\|a\|$",
        z=2,
    )
    _band(
        ax,
        t,
        pairs["acceleration_tangential"].to_numpy(float),
        pairs["acceleration_tangential_q05"].to_numpy(float),
        pairs["acceleration_tangential_q95"].to_numpy(float),
        color=C_AT,
        lw=1.15,
        label=r"$a_t$",
        z=3,
    )
    _band(
        ax,
        t,
        pairs["acceleration_normal"].to_numpy(float),
        pairs["acceleration_normal_q05"].to_numpy(float),
        pairs["acceleration_normal_q95"].to_numpy(float),
        color=C_AN,
        lw=1.15,
        label=r"$a_n$",
        z=4,
    )
    ax.axhline(0.0, color="0.55", lw=0.7, zorder=1)
    ax.set_ylabel(r"acceleration $[1/t]$", fontsize=8)


def _draw_curvature(ax: plt.Axes, pairs: pd.DataFrame, color: str) -> None:
    _band(
        ax,
        pairs["t"].to_numpy(float),
        pairs["curvature"].to_numpy(float),
        pairs["curvature_q05"].to_numpy(float),
        pairs["curvature_q95"].to_numpy(float),
        color=color,
        label=r"$\kappa$",
    )
    ax.set_ylabel(r"$\kappa$ $[1/\mathrm{length}]$", fontsize=8)


def plot_kinematics_compare(out_path: Path = OUT) -> Path:
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(11.2, 4.55),
        sharex=True,
        constrained_layout=True,
    )

    row_keys = ("shared", "untied")
    for r, key in enumerate(row_keys):
        spec = RUNS[key]
        pairs = _pairs_no_ib(spec["dir"])
        color = spec["color"]

        _draw_velocity(axes[r, 0], spec["dir"], color)
        _draw_omega(axes[r, 1], pairs, color)
        _draw_accel(axes[r, 2], pairs)
        _draw_curvature(axes[r, 3], pairs, color)

        for c in range(4):
            _style(axes[r, c])
            if r == 0:
                axes[r, c].set_title(COL_TITLES[c], fontsize=9.5, pad=4)
            if r == 1:
                axes[r, c].set_xlabel(r"Depth $t$", fontsize=8.5)

        # Row label on the leftmost axis.
        axes[r, 0].annotate(
            spec["title"],
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-0.42, 0.5),
            textcoords="axes fraction",
            ha="center",
            va="center",
            rotation=90,
            fontsize=9.5,
            fontweight="semibold",
            color=color,
        )

    # Shared legend for the acceleration column (top row).
    accel_handles = [
        Line2D([0], [0], color=C_A, lw=1.4, label=r"$\|a\|$"),
        Line2D([0], [0], color=C_AT, lw=1.4, label=r"$a_t$ (signed)"),
        Line2D([0], [0], color=C_AN, lw=1.4, label=r"$a_n$"),
    ]
    axes[0, 2].legend(
        handles=accel_handles,
        fontsize=7,
        frameon=False,
        loc="upper right",
        handlelength=1.4,
        borderaxespad=0.2,
    )

    # Protocol details live in the LaTeX caption; keep the figure itself clean.

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Saved {out_path.with_suffix('.png')}")
    return out_path


if __name__ == "__main__":
    plot_kinematics_compare()
