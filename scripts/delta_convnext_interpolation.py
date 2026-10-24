"""Classic R-refine ablation of a trained Delta ConvNeXt (unique per-block δ).

Each residual 0..8 is repeated R times with ES=1/R (horizon per block stays 1,
total T=9), integrated with RK1/RK2/RK4 for R in {1,2,4,8,16,32,64,128}.
Weight interpolation is ``plain`` only: bilinear would need a lerp of two
distinct (shared+δ) fields, which custom_forward does not expose.

  DELTA_CHECKPOINT=outputs/delta_from_D9_lr1e-5_e100/weights/last.pth \\
  DELTA_INTERP_OUT=outputs/delta_from_D9_lr1e-5_e100_interpolation \\
    python scripts/delta_convnext_interpolation.py

  python scripts/delta_convnext_interpolation.py --experiment delta_from_D9_lr1e-5_e100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.backbones.delta_convnext import CustomForwardConfig, DeltaConvNext
from scripts.validate import validate
from utils.env import load_dotenv

load_dotenv()

N_BLOCKS = 9
TAIL = [N_BLOCKS, N_BLOCKS + 1]
REPEATS = [1, 2, 4, 8, 16, 32, 64, 128]
METHODS = ["RK1", "RK2", "RK4"]


class _CustomForwardModel(nn.Module):
    def __init__(self, model: DeltaConvNext, cnf: CustomForwardConfig):
        super().__init__()
        self.model = model
        self.cnf = cnf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.custom_forward(x, self.cnf)


def load_delta_convnext(checkpoint: Path) -> DeltaConvNext:
    model = DeltaConvNext(useDeltas=True)
    model.rewire()
    model.set_delta_coeff(1.0)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    incompatible = model.load_state_dict(state, strict=False)
    tail_aliases = {
        f"tail.{index}.{part}"
        for index in (model.depths[2], model.depths[2] + 1)
        for part in ("weight", "bias")
    }
    stale = [k for k in incompatible.missing_keys if k not in tail_aliases]
    if stale or incompatible.unexpected_keys:
        raise RuntimeError(f"missing={stale} unexpected={incompatible.unexpected_keys}")
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    print(f"Loaded {checkpoint}" + (f" (epoch {epoch})" if epoch is not None else ""))
    return model


def make_config(repeats: int, euler_step: float, method: str) -> CustomForwardConfig:
    blocks = [i for i in range(N_BLOCKS) for _ in range(repeats)]
    return {
        "block_indices": blocks + TAIL,
        "euler_step": float(euler_step),
        "method": method,
    }


def last_metric(history, name: str) -> float:
    return float(history.get(name).compute_last())


def plot_r_curves(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("repeats")
        axes[0].plot(sub["repeats"], 100 * sub["top1acc"], marker="o", label=method)
        axes[1].plot(sub["repeats"], sub["loss"], marker="o", label=method)
    axes[0].set_ylabel("val top-1 (%)")
    axes[1].set_ylabel("val loss")
    for ax in axes:
        ax.set_xlabel("R (repeats per block)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(REPEATS)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(output_path.name)
    fig.tight_layout()
    fig.savefig(output_path / "r_ablation.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default=os.environ.get("DELTA_EXPERIMENT"),
        help="outputs/<name>/weights/last.pth and outputs/<name>_interpolation/",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("DELTA_CHECKPOINT"),
        help="Override the checkpoint path",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("DELTA_INTERP_OUT"),
        help="Override the results directory",
    )
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "128")))
    parser.add_argument("--force", action="store_true", help="Overwrite results2.csv if it exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment:
        checkpoint = _REPO_ROOT / "outputs" / args.experiment / "weights" / "last.pth"
        output_path = _REPO_ROOT / "outputs" / f"{args.experiment}_interpolation"
    else:
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        output_path = Path(args.output) if args.output else None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    if args.output:
        output_path = Path(args.output)
    if checkpoint is None or output_path is None:
        raise SystemExit("Pass --experiment NAME, or DELTA_CHECKPOINT and DELTA_INTERP_OUT")
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    results2 = output_path / "results2.csv"
    if results2.exists() and not args.force:
        print(f"Skip: {results2} already exists (use --force to overwrite)")
        return

    model = load_delta_convnext(checkpoint)
    experiments = [
        {
            "name": f"R{r}_ES{1 / r:g}_{m}_plain",
            "repeats": r,
            "euler_step": 1.0 / r,
            "method": m,
        }
        for r in REPEATS
        for m in METHODS
    ]

    results = []
    for spec in experiments:
        cnf = make_config(spec["repeats"], spec["euler_step"], spec["method"])
        blocks = cnf["block_indices"][:-2]
        print(
            f"\n=== {spec['name']} ===\n"
            f"delta schedule: {len(blocks)} steps  "
            f"(euler_step={cnf['euler_step']:g}, method={cnf.get('method')})"
        )
        history = validate(
            model=_CustomForwardModel(model, cnf),
            batch_size=args.batch_size,
            output_path=output_path / spec["name"],
        )
        results.append({
            "name": spec["name"],
            "repeats": spec["repeats"],
            "blocks": ",".join(map(str, blocks)),
            "euler_step": spec["euler_step"],
            "method": spec["method"],
            "weight_interpolation": "plain",
            "top1acc": last_metric(history, "top1acc"),
            "loss": last_metric(history, "loss"),
        })

    output_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(results2, index=False)
    plot_r_curves(df, output_path)

    print("\n=== summary ===")
    print(f"{'name':<28} {'method':<6} {'R':>4} {'euler':>8} {'top1':>10} {'loss':>10}")
    for row in results:
        print(
            f"{row['name']:<28} {row['method']:<6} {row['repeats']:>4} "
            f"{row['euler_step']:>8g} {row['top1acc']:>10.5f} {row['loss']:>10.6f}"
        )
    print(f"\nSaved {len(results)} rows to {results2}")


if __name__ == "__main__":
    main()
