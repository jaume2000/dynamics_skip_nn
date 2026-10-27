"""ImageNet-val top-1 / CE loss of S4+head on stage-3 prefixes.

Discrete (default): after the 1st, 2nd, …, 9th stage-3 residual, classify with
the trained downsample + stage-4 + head (Shared D9 and/or ConvNeXt-T).

Refinement (``--R``): ``D = 9·R`` Euler steps with ``Δt = 1/R`` (horizon ``T = 9``)
and decode after every micro-step — e.g. ``--R 20`` → 180 points.
  - ``shared``: repeated shared residual
  - ``convnext``: InterpoledConvNext plain schedule (each block 0..8 × R)

  python scripts/prefix_s4_head_acc.py
  python scripts/prefix_s4_head_acc.py --R 20 --only shared
  python scripts/prefix_s4_head_acc.py --R 20 --only convnext
  python scripts/prefix_s4_head_acc.py --R 20 --only shared convnext
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.backbones.convnext import ConvNextV1
from models.backbones.delta_convnext import DeltaConvNext
from models.backbones.interpoled_convnext import InterpoledConvNextV1
from models.backbones.rk_integrate import residual_rk_step
from utils.env import load_dotenv

load_dotenv()

SHARED_CKPT = Path(
    os.environ.get(
        "SHARED_CONVNEXT_CHECKPOINT",
        str(_REPO_ROOT / "outputs" / "shared_convnextv1_imagenet" / "weights" / "last.pth"),
    )
)
CONVNEXT_CKPT = Path(
    os.environ.get(
        "CONVNEXT_CHECKPOINT",
        str(_REPO_ROOT / "outputs" / "convnextv1_imagenet" / "weights" / "last.pth"),
    )
)
N_BLOCKS = 9


def available_cpus() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)
    return len(os.sched_getaffinity(0))


def load_shared(checkpoint: Path) -> DeltaConvNext:
    model = DeltaConvNext(useDeltas=False)
    model.rewire()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    stale = [k for k in missing if not k.startswith("tail.")]
    if stale or unexpected:
        raise RuntimeError(f"shared load: missing={stale} unexpected={unexpected}")
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    print(f"Loaded shared {checkpoint}" + (f" (epoch {epoch})" if epoch is not None else ""))
    return model


def load_convnext(checkpoint: Path) -> ConvNextV1:
    model = ConvNextV1()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    print(f"Loaded convnext {checkpoint}" + (f" (epoch {epoch})" if epoch is not None else ""))
    return model


def load_interpoled_convnext(checkpoint: Path) -> InterpoledConvNextV1:
    model = InterpoledConvNextV1()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    print(
        f"Loaded interpoled convnext {checkpoint}"
        + (f" (epoch {epoch})" if epoch is not None else "")
    )
    return model


def schedule_blocks(n_blocks: int, repeats: int) -> list[int]:
    return [i for i in range(n_blocks) for _ in range(repeats)]


def stage3_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    if isinstance(model, DeltaConvNext):
        return [model.deltifiedStage3[i] for i in range(N_BLOCKS)]
    return [model.stage3[i] for i in range(N_BLOCKS)]


def s4_head(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, DeltaConvNext):
        x = model.tail(x)
        x = model._stage(4)(x)
    else:
        x = model.stage3[N_BLOCKS](x)
        x = model.stage3[N_BLOCKS + 1](x)
        x = model.stage4(x)
    x = model.globalPool(x)
    return model.fc(x)


def to_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    batch = batch.to(device, non_blocking=True)
    if device.type == "cuda":
        batch = batch.contiguous(memory_format=torch.channels_last)
    return batch


def _stem_to_stage2(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    x = model.stem(batch)
    x = model.stage1(x) if hasattr(model, "stage1") else model._stage(1)(x)
    x = model.stage2(x) if hasattr(model, "stage2") else model._stage(2)(x)
    return x


def _accumulate(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    correct: list[float],
    loss_sum: list[float],
    k: int,
) -> None:
    correct[k] += (logits.argmax(1) == y).sum().item()
    loss_sum[k] += F.cross_entropy(logits, y, reduction="sum").item()


@torch.inference_mode()
def evaluate_prefixes(
    model: torch.nn.Module,
    *,
    name: str,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> list[dict]:
    """Discrete stage-3 prefixes (k = 1..9)."""
    model.eval()
    blocks = stage3_blocks(model)
    n = len(blocks)
    correct = [0.0] * n
    loss_sum = [0.0] * n
    total = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    pbar = tqdm(loader, desc=name, leave=True)
    for step, (batch, y) in enumerate(pbar):
        if max_batches is not None and step >= max_batches:
            break
        batch = to_device(batch, device)
        y = y.to(device, non_blocking=True)

        x = _stem_to_stage2(model, batch)
        for k, block in enumerate(blocks):
            x = block(x)
            _accumulate(s4_head(model, x), y, correct=correct, loss_sum=loss_sum, k=k)
        total += y.shape[0]
        pbar.set_postfix(
            b9=f"{correct[-1] / max(total, 1):.3f}",
            loss=f"{loss_sum[-1] / max(total, 1):.3f}",
        )
    pbar.close()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    rows = []
    for k in range(n):
        acc = correct[k] / max(total, 1)
        loss = loss_sum[k] / max(total, 1)
        rows.append({
            "model": name,
            "step": k + 1,
            "block": k + 1,
            "t": float(k + 1),
            "R_frac": (k + 1) / float(N_BLOCKS),
            "top1acc": acc,
            "loss": loss,
            "n_samples": total,
            "time_s": elapsed,
            "euler_step": 1.0,
            "D": n,
            "R": 1,
        })
        print(f"  {name} after block {k + 1}: top1={100 * acc:.2f}%  loss={loss:.4f}  (n={total})")
    print(f"  {name} pass in {elapsed:.1f}s")
    return rows


@torch.inference_mode()
def evaluate_shared_refined(
    model: DeltaConvNext,
    *,
    R: int,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    method: str = "RK1",
) -> list[dict]:
    """S4+head after every shared Euler micro-step (D=9R, Δt=1/R, T=9)."""
    model.eval()
    D = N_BLOCKS * R
    euler_step = 1.0 / R
    # Block eulerStep must be 1 so _integrate_block sees f(y)=F(y), not F(y)·eu.
    block = model.deltifiedStage3[0]
    if abs(float(getattr(block, "eulerStep", 1.0)) - 1.0) > 1e-12:
        raise RuntimeError(
            f"expected shared block eulerStep=1 for refined probe, got {block.eulerStep}"
        )

    name = f"shared_R{R}"
    correct = [0.0] * D
    loss_sum = [0.0] * D
    total = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    print(f"{name}: D={D}  ES={euler_step:g}  T={D * euler_step:g}  method={method}")
    pbar = tqdm(loader, desc=name, leave=True)
    for step, (batch, y) in enumerate(pbar):
        if max_batches is not None and step >= max_batches:
            break
        batch = to_device(batch, device)
        y = y.to(device, non_blocking=True)

        x = _stem_to_stage2(model, batch)
        for k in range(D):
            x = model._integrate_block(block, x, euler_step, method)
            _accumulate(s4_head(model, x), y, correct=correct, loss_sum=loss_sum, k=k)
        total += y.shape[0]
        pbar.set_postfix(
            last=f"{correct[-1] / max(total, 1):.3f}",
            loss=f"{loss_sum[-1] / max(total, 1):.3f}",
        )
    pbar.close()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    rows = []
    for k in range(D):
        acc = correct[k] / max(total, 1)
        loss = loss_sum[k] / max(total, 1)
        t = (k + 1) * euler_step
        rows.append({
            "model": name,
            "step": k + 1,
            "block": (k + 1) / R,  # fractional block index in [1/R, 9]
            "t": t,
            "R_frac": t / float(N_BLOCKS),
            "top1acc": acc,
            "loss": loss,
            "n_samples": total,
            "time_s": elapsed,
            "euler_step": euler_step,
            "D": D,
            "R": R,
        })
        # Print integer-block landmarks + endpoints.
        if (k + 1) % R == 0 or k == 0 or k == D - 1:
            print(
                f"  {name} step {k + 1}/{D} (t={t:.3g}): "
                f"top1={100 * acc:.2f}%  loss={loss:.4f}"
            )
    print(f"  {name} pass in {elapsed:.1f}s")
    return rows


@torch.inference_mode()
def evaluate_convnext_refined(
    model: InterpoledConvNextV1,
    *,
    R: int,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    method: str = "RK1",
) -> list[dict]:
    """S4+head after every plain-interpoled Euler micro-step (each block × R)."""
    model.eval()
    D = N_BLOCKS * R
    euler_step = 1.0 / R
    blocks = schedule_blocks(N_BLOCKS, R)
    assert len(blocks) == D
    model.set_schedule(blocks, euler_step, method=method, weight_interpolation="plain")

    name = f"convnext_R{R}"
    correct = [0.0] * D
    loss_sum = [0.0] * D
    total = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    print(
        f"{name}: D={D}  ES={euler_step:g}  T={D * euler_step:g}  "
        f"method={method}  wi=plain"
    )
    pbar = tqdm(loader, desc=name, leave=True)
    for step, (batch, y) in enumerate(pbar):
        if max_batches is not None and step >= max_batches:
            break
        batch = to_device(batch, device)
        y = y.to(device, non_blocking=True)

        x = _stem_to_stage2(model, batch)
        for k, bi in enumerate(blocks):
            x = residual_rk_step(model.stage3[bi], x, euler_step, method)
            _accumulate(s4_head(model, x), y, correct=correct, loss_sum=loss_sum, k=k)
        total += y.shape[0]
        pbar.set_postfix(
            last=f"{correct[-1] / max(total, 1):.3f}",
            loss=f"{loss_sum[-1] / max(total, 1):.3f}",
        )
    pbar.close()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    rows = []
    for k in range(D):
        acc = correct[k] / max(total, 1)
        loss = loss_sum[k] / max(total, 1)
        t = (k + 1) * euler_step
        rows.append({
            "model": name,
            "step": k + 1,
            "block": (k + 1) / R,
            "t": t,
            "R_frac": t / float(N_BLOCKS),
            "top1acc": acc,
            "loss": loss,
            "n_samples": total,
            "time_s": elapsed,
            "euler_step": euler_step,
            "D": D,
            "R": R,
        })
        if (k + 1) % R == 0 or k == 0 or k == D - 1:
            print(
                f"  {name} step {k + 1}/{D} (t={t:.3g}, blk={blocks[k]}): "
                f"top1={100 * acc:.2f}%  loss={loss:.4f}"
            )
    print(f"  {name} pass in {elapsed:.1f}s")
    return rows


SERIES_STYLE = {
    "shared_D9": ("Weight-tied D9", "#9467bd"),
    "shared_R20": ("Weight-tied R20", "#9467bd"),
    "convnext": ("ConvNeXt-T", "#1f77b4"),
    "convnext_R20": ("ConvNeXt-T R20", "#1f77b4"),
}


def plot_curves(df: pd.DataFrame, out_dir: Path, *, R: int | None) -> None:
    """Write top-1 and CE-loss curves vs continuous depth t."""
    out_dir.mkdir(parents=True, exist_ok=True)
    series = []
    for model, g in df.groupby("model", sort=False):
        g = g.sort_values("t")
        series.append((model, g))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))

    for model, g in series:
        label, color = SERIES_STYLE.get(model, (model, None))
        kw = {"lw": 1.8, "label": label}
        if color is not None:
            kw["color"] = color
        axes[0].plot(g["t"], 100.0 * g["top1acc"].to_numpy(), **kw)
        axes[1].plot(g["t"], g["loss"].to_numpy(), **kw)

    axes[0].set_ylabel("Top-1 (%)")
    axes[1].set_ylabel("CE loss")
    for ax in axes:
        ax.set_xlabel(r"Stage-3 depth $t$")
        ax.set_xlim(0, N_BLOCKS)
        ax.set_xticks(range(0, N_BLOCKS + 1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        if R is not None and R > 1:
            for b in range(1, N_BLOCKS + 1):
                ax.axvline(b, color="0.85", lw=0.6, zorder=0)

    title = (
        f"S4+head on stage-3 prefixes (R={R}, Δt={1 / R:g})"
        if R is not None and R > 1
        else "S4+head on stage-3 prefixes"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    pdf = out_dir / "prefix_curves.pdf"
    png = out_dir / "prefix_curves.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"Wrote {pdf} and {png}")


def default_out_dir(R: int | None, only: list[str]) -> Path:
    if R is not None and R > 1:
        tag = "_".join(only)
        return _REPO_ROOT / "outputs" / f"prefix_s4_head_acc_{tag}_R{R}"
    return Path(
        os.environ.get(
            "PREFIX_S4_OUT",
            str(_REPO_ROOT / "outputs" / "prefix_s4_head_acc"),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--R",
        type=int,
        default=None,
        help="Refinement: D=9R Euler steps with Δt=1/R for shared and/or convnext.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="RK1",
        choices=["RK1", "RK2", "RK4", "EULER"],
        help="Integrator for --R probe (default RK1).",
    )
    parser.add_argument(
        "--only",
        choices=["shared", "convnext"],
        nargs="+",
        default=["shared", "convnext"],
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default depends on --R / --only).",
    )
    parser.add_argument(
        "--merge-csv",
        type=str,
        nargs="*",
        default=None,
        help="Extra results.csv paths to merge into the plot (e.g. prior shared_R20).",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from data.imagenet import ImageNetDataset
    from data.transforms.transforms import build_val_transforms

    if args.R is not None and args.R < 1:
        raise SystemExit("--R must be >= 1")

    out_dir = Path(args.out) if args.out is not None else default_out_dir(args.R, args.only)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = min(4, max(0, available_cpus() // 2))
    print(f"device={device}  batch={args.batch_size}  workers={num_workers}  out={out_dir}")

    val_loader = DataLoader(
        ImageNetDataset(split="validation", transforms=build_val_transforms()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    rows: list[dict] = []
    if "shared" in args.only:
        model = load_shared(SHARED_CKPT).to(device)
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)
        if args.R is not None and args.R > 1:
            rows += evaluate_shared_refined(
                model,
                R=args.R,
                loader=val_loader,
                device=device,
                max_batches=args.max_batches,
                method=args.method.upper(),
            )
        else:
            rows += evaluate_prefixes(
                model, name="shared_D9", loader=val_loader, device=device,
                max_batches=args.max_batches,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "convnext" in args.only:
        if args.R is not None and args.R > 1:
            model = load_interpoled_convnext(CONVNEXT_CKPT).to(device)
            if device.type == "cuda":
                model.to(memory_format=torch.channels_last)
            rows += evaluate_convnext_refined(
                model,
                R=args.R,
                loader=val_loader,
                device=device,
                max_batches=args.max_batches,
                method=args.method.upper(),
            )
        else:
            model = load_convnext(CONVNEXT_CKPT).to(device)
            if device.type == "cuda":
                model.to(memory_format=torch.channels_last)
            rows += evaluate_prefixes(
                model, name="convnext", loader=val_loader, device=device,
                max_batches=args.max_batches,
            )
        del model

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    if not args.no_plot:
        plot_df = df
        if args.merge_csv:
            extras = [pd.read_csv(p) for p in args.merge_csv]
            plot_df = pd.concat([df, *extras], ignore_index=True)
        plot_curves(plot_df, out_dir, R=args.R)


if __name__ == "__main__":
    main()

