"""Overlay train/val curves of the four delta experiments.

  python scripts/plot_delta_experiments.py
  python scripts/plot_delta_experiments.py --output paper/figures/delta_experiments.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EXPERIMENTS = [
    ("delta_from_D9_lr1e-5_e100", "D9 → δ, LR 1e-5"),
    ("delta_from_D9_ramp0to1_e80", "D9 → δ, coeff 0→1 / 80"),
    ("delta_scratch_ramp150to250_e300", "scratch, coeff ramp / 300"),
    ("delta_scratch_full_e300", "scratch, coeff=1 / 300"),
]


def _read_history(experiment: str, split: str, metric: str) -> pd.Series | None:
    path = _REPO_ROOT / "outputs" / experiment / "results" / f"{split}_{metric}_history.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    col = "value" if "value" in df.columns else df.columns[-1]
    return df[col].astype(float)


def _read_r_ablation(experiment: str) -> pd.DataFrame | None:
    path = _REPO_ROOT / "outputs" / f"{experiment}_interpolation" / "results2.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def plot_r_ablations(output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    drawn = 0
    for name, label in EXPERIMENTS:
        df = _read_r_ablation(name)
        if df is None:
            continue
        rk1 = df[df["method"] == "RK1"].sort_values("repeats")
        if rk1.empty:
            continue
        axes[0].plot(rk1["repeats"], 100 * rk1["top1acc"], marker="o", label=label)
        axes[1].plot(rk1["repeats"], rk1["loss"], marker="o", label=label)
        drawn += 1
    if drawn == 0:
        return
    axes[0].set_ylabel("val top-1 (%)")
    axes[1].set_ylabel("val loss")
    for ax in axes:
        ax.set_xlabel("R (repeats per block)")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Delta ConvNeXt R-refine (plain RK1)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "outputs" / "delta_experiments" / "training_curves.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=False)
    panels = [
        (axes[0, 0], "val", "top1acc", "val top-1"),
        (axes[0, 1], "val", "loss", "val loss"),
        (axes[1, 0], "train", "top1acc", "train top-1"),
        (axes[1, 1], "train", "loss", "train loss"),
    ]
    drawn = 0
    for ax, split, metric, title in panels:
        for name, label in EXPERIMENTS:
            series = _read_history(name, split, metric)
            if series is None:
                continue
            ax.plot(series.values, label=label)
            drawn += 1
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    out = Path(args.output)
    if drawn:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"Wrote {out}")
    else:
        print("No history CSVs found under outputs/delta_* (training still running?)")
    plt.close(fig)
    plot_r_ablations(out.with_name("r_ablation.png"))


if __name__ == "__main__":
    main()
