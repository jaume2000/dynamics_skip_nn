#!/usr/bin/env python3
"""Per-coordinate trajectory views (progress, recruitment, phase density) for one run.

Thin wrapper around ``scripts/trajectory_views.py`` reading a feature-map run folder
(``mean_maps.pt`` + ``config.json``). The default run is the weight-tied D=900 one; any
other run gets its folder name as suffix.

Writes
  figures/feature_progress[_<run>].pdf
  figures/feature_recruitment[_<run>].pdf
  figures/feature_phase_density[_<run>].pdf

Usage (from repo root):
    python paper/scripts/plot_feature_dynamics.py
    python paper/scripts/plot_feature_dynamics.py \
        --run outputs/featureMaps_interpoled/convnext_R100_ES0.01_c289_n1_ignore1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.trajectory_views import save_trajectory_views  # noqa: E402

FIG = ROOT / "paper" / "figures"
DEFAULT_RUN = ROOT / "outputs/featureMaps/convnext_shared_D900_ES0.01_c289_n1_ignore1"
TITLES = {
    "convnext_shared": "Weight-tied ConvNeXt",
    "convnext": "ConvNeXt-T",
    "convnext_droppath0": "ConvNeXt-T (drop-path 0)",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "swin": "Swin-T",
}


def block_bounds(cfg: dict) -> list[float]:
    """Times where the stage-3 schedule switches residual block (empty for weight-tied)."""
    blocks = cfg.get("blocks") or []
    es = float(cfg["euler_step"])
    return [(d + 1) * es for d in range(len(blocks) - 1) if blocks[d] != blocks[d + 1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    cfg = json.loads((run / "config.json").read_text())
    maps = torch.load(run / "mean_maps.pt", map_location="cpu", weights_only=False)

    model = TITLES.get(cfg.get("model"), cfg.get("model", "?"))
    n = cfg.get("n_images", 1)
    who = "one image" if n == 1 else f"mean of {n} images"
    title = f"{model}, stage 3, {who} (D={cfg['D']}, $\\Delta t$={cfg['euler_step']:g})"
    suffix = "" if run.resolve() == DEFAULT_RUN.resolve() else f"_{run.name}"

    stats = save_trajectory_views(
        maps["mean_x"], maps["mean_h"], FIG, euler_step=cfg["euler_step"],
        ignored_channels=cfg.get("ignored_channels"), title=title,
        block_bounds=block_bounds(cfg), suffix=f"{suffix}.pdf",
    )
    for kind in ("progress", "recruitment", "phase_density"):
        pdf = FIG / f"feature_{kind}{suffix}.pdf"
        print(f"Saved {pdf}")
    print({k: round(v, 3) for k, v in stats.items()})


if __name__ == "__main__":
    main()
