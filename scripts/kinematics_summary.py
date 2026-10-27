#!/usr/bin/env python3
"""Scalar summary of the stage-3 kinematics behind Figure 6 (tied D=900, untied R=100).

Reads the per-step tables used by paper/scripts/plot_kinematics_compare.py and writes
outputs/kinematics_summary/summary.csv with the values quoted in Section 3.3.
Means over the 500 images are the curves of the figure; inter-block pairs of the untied
model are kept only for the heading jump at block boundaries.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "tied": ROOT / "outputs/featureMaps/convnext_shared_D900_ES0.01_N500_ignore1",
    "untied": ROOT / "outputs/featureMaps_interpoled/convnext_R100_ES0.01_N500_ignore1",
}
OUT = ROOT / "outputs" / "kinematics_summary"
DT = 0.01


def at(df: pd.DataFrame, col: str, t: float) -> float:
    return float(df.iloc[(df["t"] - t).abs().argmin()][col])


def window(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    return df[(df["t"] >= lo) & (df["t"] < hi)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, d in RUNS.items():
        pairs_all = pd.read_csv(d / "table_pairs.csv")
        ib = pairs_all["is_inter_block"].astype(bool) if "is_inter_block" in pairs_all else np.zeros(len(pairs_all), bool)
        p = pairs_all[~ib]
        n = pd.read_csv(d / "table_norms.csv")
        r = {"model": name}
        r["share_a_n"] = float((p["acceleration_normal"] / p["acc_h"]).mean())
        r["share_abs_a_t"] = float((p["acceleration_tangential"].abs() / p["acc_h"]).mean())
        r["acc_first"] = float(p["acc_h"].iloc[0])
        r["acc_last"] = float(p["acc_h"].iloc[-1])
        r["omega_end"] = float(p["omega_h"].iloc[-1])
        r["kappa_end"] = float(p["curvature"].iloc[-1])
        r["v_end"] = float(n["norm_h"].iloc[-1])
        if name == "tied":
            early = window(p, 0, 1.5)
            i = early["omega_h"].idxmax()
            r["omega_peak_early"] = float(early.loc[i, "omega_h"])
            r["t_omega_peak_early"] = float(early.loc[i, "t"])
            r["kappa_peak_early"] = float(early["curvature"].max())
            r["v_at_kappa_peak"] = at(n, "norm_h", float(early.loc[early["curvature"].idxmax(), "t"]))
            r["t_a_t_max"] = float(p.loc[p["acceleration_tangential"].idxmax(), "t"])
            r["t_a_t_min"] = float(p.loc[p["acceleration_tangential"].idxmin(), "t"])
            sign = np.sign(p["acceleration_tangential"].to_numpy())
            cross = np.where((sign[:-1] > 0) & (sign[1:] <= 0))[0]
            r["t_a_t_zero"] = float(p["t"].iloc[cross[0] + 1]) if len(cross) else math.nan
            mid = window(p, 2, 6)
            j = mid["curvature"].idxmin()
            r["kappa_min_mid"] = float(mid.loc[j, "curvature"])
            r["t_kappa_min_mid"] = float(mid.loc[j, "t"])
            r["v_at_kappa_min"] = at(n, "norm_h", r["t_kappa_min_mid"])
            late = window(p, 5.5, 8)
            r["t_omega_rebound"] = float(late.loc[late["omega_h"].idxmax(), "t"])
            r["t_v_peak"] = float(n.loc[n["norm_h"].idxmax(), "t"])
            r["v_peak"] = float(n["norm_h"].max())
        else:
            blocks = [window(p, k, k + 1) for k in range(9)]
            r["omega_block_start_range"] = f"{min(b['omega_h'].iloc[0] for b in blocks):.2f}-{max(b['omega_h'].iloc[0] for b in blocks):.2f}"
            r["omega_block_end_range"] = f"{min(b['omega_h'].iloc[-1] for b in blocks):.2f}-{max(b['omega_h'].iloc[-1] for b in blocks):.2f}"
            r["kappa_block0_start"] = float(blocks[0]["curvature"].iloc[0])
            r["kappa_block8_mean"] = float(blocks[8]["curvature"].mean())
            r["v_after_block0"] = at(n, "norm_h", 0.5)
            r["v_start_block8"] = at(n, "norm_h", 8.0)
            jumps = pairs_all[ib]
            ang = np.degrees(jumps["omega_h"].to_numpy() * DT)
            r["boundary_turn_deg_range"] = f"{ang.min():.0f}-{ang.max():.0f}"
            r["boundary_turn_deg_mean"] = float(ang.mean())
            r["boundary_omega_range"] = f"{jumps['omega_h'].min():.0f}-{jumps['omega_h'].max():.0f}"
        rows.append(r)
    df = pd.DataFrame(rows).T
    df.to_csv(OUT / "summary.csv", header=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
