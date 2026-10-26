"""ImageNet-val top-1 of S4+head on stage-3 prefixes.

For Shared D9 and standard ConvNeXt-T, take the feature after the 1st, 2nd, …,
9th stage-3 residual and classify it with the trained downsample + stage-4 +
head (no remaining stage-3 blocks). One val pass per model: blocks are applied
incrementally.

  python scripts/prefix_s4_head_acc.py
  python scripts/prefix_s4_head_acc.py --batch-size 64 --num-workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.backbones.convnext import ConvNextV1
from models.backbones.delta_convnext import DeltaConvNext
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
OUTPUT_PATH = Path(
    os.environ.get(
        "PREFIX_S4_OUT",
        str(_REPO_ROOT / "outputs" / "prefix_s4_head_acc"),
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


@torch.inference_mode()
def evaluate_prefixes(
    model: torch.nn.Module,
    *,
    name: str,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> list[dict]:
    model.eval()
    blocks = stage3_blocks(model)
    correct = [0] * N_BLOCKS
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

        x = model.stem(batch)
        x = model.stage1(x) if hasattr(model, "stage1") else model._stage(1)(x)
        x = model.stage2(x) if hasattr(model, "stage2") else model._stage(2)(x)

        for k, block in enumerate(blocks):
            x = block(x)
            pred = s4_head(model, x)
            correct[k] += (pred.argmax(1) == y).sum().item()
        total += y.shape[0]
        pbar.set_postfix(b9=f"{correct[-1] / max(total, 1):.3f}")
    pbar.close()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    rows = []
    for k in range(N_BLOCKS):
        acc = correct[k] / max(total, 1)
        rows.append({
            "model": name,
            "block": k + 1,
            "top1acc": acc,
            "n_samples": total,
            "time_s": elapsed,
        })
        print(f"  {name} after block {k + 1}: {100 * acc:.2f}%  (n={total})")
    print(f"  {name} pass in {elapsed:.1f}s")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--only",
        choices=["shared", "convnext"],
        nargs="+",
        default=["shared", "convnext"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from data.imagenet import ImageNetDataset
    from data.transforms.transforms import build_val_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = min(4, max(0, available_cpus() // 2))
    print(f"device={device}  batch={args.batch_size}  workers={num_workers}")

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
        rows += evaluate_prefixes(
            model, name="shared_D9", loader=val_loader, device=device,
            max_batches=args.max_batches,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "convnext" in args.only:
        model = load_convnext(CONVNEXT_CKPT).to(device)
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)
        rows += evaluate_prefixes(
            model, name="convnext", loader=val_loader, device=device,
            max_batches=args.max_batches,
        )
        del model

    df = pd.DataFrame(rows)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_PATH / "results.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== S4+head top-1 on ImageNet val ===")
    print(f"{'block':>6}  {'shared D9':>12}  {'ConvNeXt':>12}")
    shared = {r["block"]: r["top1acc"] for r in rows if r["model"] == "shared_D9"}
    conv = {r["block"]: r["top1acc"] for r in rows if r["model"] == "convnext"}
    for k in range(1, N_BLOCKS + 1):
        s = f"{100 * shared[k]:.2f}" if k in shared else "—"
        c = f"{100 * conv[k]:.2f}" if k in conv else "—"
        print(f"{k:>6}  {s:>12}  {c:>12}")
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
