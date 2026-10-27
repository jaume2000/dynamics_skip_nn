#!/usr/bin/env python3
"""Plot top-1 vs stage-3 block evaluations (paper Fig. block-evals).

Uses only existing CSVs under outputs/. Optional: run missing equispaced
untied subsets via the helpers in scripts/convnext_interpolation.py
(does not modify that file).

  python paper/scripts/plot_block_evals.py
  python paper/scripts/plot_block_evals.py --run-missing
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT_FIG = REPO / "paper" / "figures" / "block_evals_acc.pdf"
OUT_CSV = REPO / "outputs" / "block_evals_figure" / "points.csv"
EQ_CSV = REPO / "outputs" / "block_evals_figure" / "equispaced_results.csv"

# Paper colours
TIED = "#7B2D8E"
UNTIED = "#2F5FA6"
RANDOM = "#6B6B6B"


def native_prefixes() -> pd.DataFrame:
    df = pd.read_csv(REPO / "outputs" / "prefix_s4_head_acc" / "results.csv")
    rows = []
    for _, r in df.iterrows():
        model = "tied" if str(r["model"]).startswith("shared") else "untied"
        rows.append(
            {
                "series": f"prefix_{model}",
                "evals": int(r["step"]),  # one Euler step per block
                "top1": 100.0 * float(r["top1acc"]),
                "label": f"prefix {model} t={int(r['step'])}",
            }
        )
    return pd.DataFrame(rows)


def tied_coarse_euler() -> pd.DataFrame:
    """Table 10 / sharedConvnextAblation: Euler at T=9 with 1..9 steps."""
    df = pd.read_csv(REPO / "outputs" / "sharedConvnextAblation" / "results.csv")
    sub = df[(df["T"] == 9.0) & (df["method"] == "RK1") & (df["depth"] <= 9)].copy()
    sub = sub.sort_values("depth")
    rows = []
    for _, r in sub.iterrows():
        rows.append(
            {
                "series": "tied_coarse_euler",
                "evals": int(r["depth"]),  # Euler: 1 eval per step
                "top1": 100.0 * float(r["top1acc"]),
                "label": f"tied Euler steps={int(r['depth'])}",
            }
        )
    return pd.DataFrame(rows)


def app_e_best_subseq() -> pd.DataFrame:
    """Hand-curated best / notable App E configs at T≈9, plus full stage."""
    # (evals, top1%, label) — evals = number of block applications (Euler)
    points = [
        (1, 39.67, "best single B5 Δt=9"),
        (2, 57.57, "B3,5 Δt=4.5"),
        (3, 60.62, "B0,4,8 Δt=3"),
        (9, 74.69, "every other twice Δt=1"),
        (9, 80.96, "native nine"),
    ]
    return pd.DataFrame(
        [
            {
                "series": "untied_best_subseq",
                "evals": e,
                "top1": t,
                "label": lab,
            }
            for e, t, lab in points
        ]
    )


def equispaced_specs() -> list[dict]:
    specs = []
    for L in range(1, 10):
        idx = np.round(np.linspace(0, 8, L)).astype(int).tolist()
        # unique while preserving order
        seen = set()
        uniq = []
        for i in idx:
            if i not in seen:
                seen.add(i)
                uniq.append(int(i))
        dt = 9.0 / L
        specs.append(
            {
                "name": f"eq_L{L}_" + "-".join(map(str, uniq)),
                "blocks": uniq,
                "euler_step": dt,
                "method": "RK1",
                "weight_interpolation": "plain",
                "L": L,
                "evals": len(uniq),  # one Euler eval per listed block
            }
        )
    return specs


def load_or_empty_equispaced() -> pd.DataFrame:
    if EQ_CSV.exists():
        return pd.read_csv(EQ_CSV)
    return pd.DataFrame(columns=["series", "evals", "top1", "label", "L", "blocks"])


def run_missing_equispaced() -> pd.DataFrame:
    """Launch missing equispaced evals using convnext_interpolation helpers."""
    from scripts.convnext_interpolation import (
        apply_experiment,
        last_metric,
        load_interpoled_convnext,
    )
    from scripts.validate import validate

    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault(
        "HF_DATASETS_CACHE",
        str(Path.home() / ".cache" / "huggingface" / "datasets"),
    )

    existing = load_or_empty_equispaced()
    done = set(existing["label"].tolist()) if len(existing) else set()
    ckpt = REPO / "outputs" / "convnextv1_imagenet" / "weights" / "last.pth"
    model = load_interpoled_convnext(ckpt)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rows = existing.to_dict("records") if len(existing) else []
    out_dir = EQ_CSV.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in equispaced_specs():
        label = f"eq L={spec['L']} blocks={spec['blocks']} Δt={spec['euler_step']:g}"
        if label in done:
            print(f"skip {label}")
            continue
        blocks, name = apply_experiment(model, spec)
        hist = validate(model, batch_size=2, num_workers=2)
        top1 = 100.0 * last_metric(hist, "top1acc")
        row = {
            "series": "untied_equispaced",
            "evals": spec["evals"],
            "top1": top1,
            "label": label,
            "L": spec["L"],
            "blocks": ",".join(map(str, blocks)),
            "euler_step": spec["euler_step"],
            "name": name,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(EQ_CSV, index=False)
        print(f"{label}: {top1:.2f}%")
    return pd.DataFrame(rows)


def plot(points: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    # horizontal full-depth lines
    ax.axhline(80.96, color=UNTIED, ls="-", lw=0.8, alpha=0.35)
    ax.axhline(80.34, color=TIED, ls="-", lw=0.8, alpha=0.35)

    def scatter_line(mask, color, marker, ls, label, z=2):
        sub = points[mask].sort_values("evals")
        if sub.empty:
            return
        ax.plot(
            sub["evals"],
            sub["top1"],
            color=color,
            marker=marker,
            ls=ls,
            lw=1.6,
            ms=5.5,
            label=label,
            zorder=z,
        )

    scatter_line(points["series"] == "prefix_tied", TIED, "o", "-", "tied prefixes")
    scatter_line(
        points["series"] == "prefix_untied", UNTIED, "s", "-", "untied prefixes"
    )
    scatter_line(
        points["series"] == "untied_best_subseq",
        UNTIED,
        "D",
        "--",
        "untied best subsequence",
    )
    scatter_line(
        points["series"] == "untied_equispaced",
        UNTIED,
        "^",
        ":",
        "untied equispaced",
    )
    scatter_line(
        points["series"] == "tied_coarse_euler",
        TIED,
        "x",
        ":",
        "tied coarse Euler",
        z=1,
    )

    ax.set_xlabel("Stage-3 block evaluations")
    ax.set_ylabel("ImageNet top-1 (%)")
    ax.set_xlim(0.5, 9.5)
    ax.set_xticks(range(1, 10))
    ax.set_ylim(0, 85)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    fig.savefig(OUT_FIG.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_FIG}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-missing",
        action="store_true",
        help="Evaluate missing equispaced untied subsets on GPU",
    )
    args = ap.parse_args()

    frames = [native_prefixes(), tied_coarse_euler(), app_e_best_subseq()]
    if args.run_missing:
        frames.append(run_missing_equispaced())
    else:
        eq = load_or_empty_equispaced()
        if len(eq):
            frames.append(eq)

    points = pd.concat(frames, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(OUT_CSV, index=False)
    plot(points)


if __name__ == "__main__":
    main()
