"""Delta ConvNeXt experiments (D=9 stage-3).

Variants (set VARIANT, or pass the name as the first CLI arg):

  d9_lowlr       Fine-tune the D=9 shared checkpoint + zero-init deltas at 1e-5.
                 The earlier wu10/e50/1e-3 run dropped 79.5 → 72.2 at peak LR and
                 only recovered as cosine decayed; 1e-4 still dipped ~1.5 pt;
                 1e-5 (the 9→27 run) did not. 100 epochs, warmup 2.

  d9_ramp        Same shared checkpoint, but δ is multiplied by a coefficient
                 that goes 0 → 1 over epochs 0..60 and stays 1 until epoch 80.
                 Peak LR 1e-4 (the ramp is the stabilizer).

  scratch_ramp   From scratch. Coefficient stays 0 through epoch 150, then
                 rises linearly to 1 at epoch 250; training ends at 300.
                 Standard shared recipe: 1e-3, warmup 10, global batch 1024.

  scratch_full   From scratch, coefficient always 1, 300 epochs. Control for
                 scratch_ramp. Unique δ per step on one shared W (classic).

  scratch_grouped3x3
                 From scratch, same recipe as scratch_full, but stage-3 is
                 3 distinct (W, δ) blocks each applied 3 times (shared δ
                 within each block only).

Leonardo:
  VARIANT=d9_lowlr source .env && sbatch --account="$SLURM_ACCOUNT" \\
      jobs/train_delta_experiment.sh
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from data.imagenet import ImageNetDataset
from data.transforms.transforms import (
    build_train_batch_transforms,
    build_train_transforms,
    build_val_transforms,
)
from engine.trainer import Trainer
from metrics.metricDeltaCoeff import MetricDeltaCoeff
from metrics.metricDeltaRatio import MetricDeltaRatio
from metrics.metricGradNorm import MetricGradNorm
from metrics.metricHistory import DictHistoryMetrics
from metrics.metricLoss import MetricLoss
from metrics.metricLR import MetricLR
from metrics.top1acc import Top1AccMetric
from torch.optim import AdamW
from models.backbones.delta_convnext import DeltaConvNext
from optim.cosineSchedule import CosineWithWarmup
from optim.deltaCoeffSchedule import delta_coeff_at_epoch
from optim.paramGroups import build_param_groups_with_delta_weight_decay
from timm.loss import SoftTargetCrossEntropy
from utils.env import load_dotenv
import torch


load_dotenv()

VARIANTS: dict[str, dict] = {
    "d9_lowlr": {
        "experiment_name": "delta_from_D9_lr1e-5_e100",
        "pretrained": "outputs/shared_convnextv1_imagenet/weights/last.pth",
        "epochs": 100,
        "warmup_epochs": 2,
        "lr": 1e-5,
        "min_lr": 1e-7,
        "global_batch_size": 512,
        "hold_until": 0,
        "ramp_end": 0,
    },
    "d9_ramp": {
        "experiment_name": "delta_from_D9_ramp0to1_e80",
        "pretrained": "outputs/shared_convnextv1_imagenet/weights/last.pth",
        "epochs": 80,
        "warmup_epochs": 2,
        "lr": 1e-4,
        "min_lr": 1e-7,
        "global_batch_size": 512,
        "hold_until": 0,
        "ramp_end": 60,
    },
    "scratch_ramp": {
        "experiment_name": "delta_scratch_ramp150to250_e300",
        "pretrained": None,
        "epochs": 300,
        "warmup_epochs": 10,
        "lr": 1e-3,
        "min_lr": 1e-7,
        "global_batch_size": 1024,
        "hold_until": 150,
        "ramp_end": 250,
    },
    "scratch_full": {
        "experiment_name": "delta_scratch_full_e300",
        "pretrained": None,
        "epochs": 300,
        "warmup_epochs": 10,
        "lr": 1e-3,
        "min_lr": 1e-7,
        "global_batch_size": 1024,
        "hold_until": 0,
        "ramp_end": 0,
        "rewire": "classic",
    },
    "scratch_grouped3x3": {
        "experiment_name": "delta_scratch_grouped3x3_e300",
        "pretrained": None,
        "epochs": 300,
        "warmup_epochs": 10,
        "lr": 1e-3,
        "min_lr": 1e-7,
        "global_batch_size": 1024,
        "hold_until": 0,
        "ramp_end": 0,
        "rewire": "grouped3x3",
    },
}

_argv_variant = next((a for a in sys.argv[1:] if not a.startswith("-")), "")
VARIANT = os.environ.get("VARIANT", _argv_variant)
if VARIANT not in VARIANTS:
    names = ", ".join(VARIANTS)
    raise SystemExit(f"VARIANT must be one of: {names} (got {VARIANT!r})")

cfg = VARIANTS[VARIANT]
# Do not read EXPERIMENT_NAME from .env: that file still names the shared
# baseline and would overwrite these runs.
EXPERIMENT_NAME = os.environ.get("DELTA_EXPERIMENT_NAME", cfg["experiment_name"])
EXPERIMENT_PATH = Path("outputs") / EXPERIMENT_NAME
PRETRAINED = os.environ.get("PRETRAINED", cfg["pretrained"] or "")
PRETRAINED_PATH = Path(PRETRAINED) if PRETRAINED else None
EPOCHS = int(os.environ.get("EPOCHS", cfg["epochs"]))
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", cfg["warmup_epochs"]))
LR = float(os.environ.get("LR", cfg["lr"]))
MIN_LR = float(os.environ.get("MIN_LR", cfg["min_lr"]))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", cfg["global_batch_size"]))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
HOLD_UNTIL = int(os.environ.get("DELTA_HOLD_UNTIL", cfg["hold_until"]))
RAMP_END = int(os.environ.get("DELTA_RAMP_END", cfg["ramp_end"]))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.05"))
DELTA_WEIGHT_DECAY = float(os.environ.get("DELTA_WEIGHT_DECAY", "0.05"))
USE_DDP = True
RETAKE = os.environ.get("RETAKE", "0") == "1"


def available_cpus() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)
    return len(os.sched_getaffinity(0))


def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def load_shared_checkpoint(model: DeltaConvNext, checkpoint: Path, rank: int) -> None:
    stDict = torch.load(checkpoint, map_location="cpu")
    state = stDict["model"] if isinstance(stDict, dict) and "model" in stDict else stDict
    incompatible = model.load_state_dict(state, strict=False)
    tail_aliases = {
        f"tail.{index}.{part}"
        for index in (model.depths[2], model.depths[2] + 1)
        for part in ("weight", "bias")
    }
    allowed_missing = tail_aliases | {k for k in incompatible.missing_keys if "Delta" in k}
    stale = set(incompatible.missing_keys) - allowed_missing
    if stale or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Refusing to train on a partial load of {checkpoint}: "
            f"missing {sorted(stale)}, "
            f"unexpected {sorted(incompatible.unexpected_keys)}"
        )
    if rank == 0:
        n_delta = sum(1 for k in incompatible.missing_keys if "Delta" in k)
        epoch = stDict.get("epoch") if isinstance(stDict, dict) else None
        print(
            f"Loaded {checkpoint}"
            + (f" (epoch {epoch})" if epoch is not None else "")
            + f"; {n_delta} zero-init deltas; epoch 1 val top1 must land near 0.80"
        )


local_rank, rank, world_size = setup_ddp() if USE_DDP else (0, 0, 1)
if GLOBAL_BATCH_SIZE % max(world_size, 1):
    raise ValueError(
        f"GLOBAL_BATCH_SIZE={GLOBAL_BATCH_SIZE} not divisible by {world_size} ranks"
    )
per_rank = GLOBAL_BATCH_SIZE // max(world_size, 1)
if per_rank % BATCH_SIZE:
    # 4 GPUs × 256 cannot make a global batch of 512; drop to one microbatch.
    BATCH_SIZE = per_rank
GRAD_ACCUM_STEPS = per_rank // BATCH_SIZE
use_ddp = USE_DDP and world_size > 1 and torch.cuda.is_available()

if torch.cuda.is_available():
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cpu")
    use_ddp = False

model = DeltaConvNext(useDeltas=True)
rewire_mode = cfg.get("rewire", "classic")
if rewire_mode == "grouped3x3":
    model.rewire_grouped(n_groups=3, repeats=3)
elif rewire_mode == "classic":
    model.rewire(sharedBlock=5)
else:
    raise ValueError(f"Unknown rewire mode {rewire_mode!r}")
if not RETAKE and PRETRAINED_PATH is not None:
    if not PRETRAINED_PATH.is_file():
        raise FileNotFoundError(
            f"Shared checkpoint not found at {PRETRAINED_PATH}. "
            "Copy outputs/shared_convnextv1_imagenet from the previous project."
        )
    load_shared_checkpoint(model, PRETRAINED_PATH, rank)
elif rank == 0 and not RETAKE:
    print("Training from scratch (random init + zero-init deltas)")
model = model.to(device)

if use_ddp:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if rank == 0:
        print(
            f"DDP on {world_size} GPUs (local_rank={local_rank}), "
            f"grad accumulation x{GRAD_ACCUM_STEPS}"
        )
elif device.type == "cuda":
    print("Using single GPU: cuda:0")
else:
    print("Using CPU")

train_dataset = ImageNetDataset(split="train", transforms=build_train_transforms())
val_dataset = ImageNetDataset(split="validation", transforms=build_val_transforms())

num_workers = max(1, available_cpus() // world_size) if use_ddp else available_cpus()
if rank == 0:
    print(f"Dataloader workers per process: {num_workers}")

train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=num_workers > 0,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=num_workers > 0,
)

param_groups = build_param_groups_with_delta_weight_decay(
    model, weight_decay=WEIGHT_DECAY, delta_weight_decay=DELTA_WEIGHT_DECAY
)
if rank == 0:
    msg = f"Weight decay {WEIGHT_DECAY} on {sum(p.numel() for p in param_groups[0]['params'])} params"
    delta_groups = [g for g in param_groups if g["weight_decay"] == DELTA_WEIGHT_DECAY]
    if delta_groups:
        msg += (
            f", delta weight decay {DELTA_WEIGHT_DECAY} on "
            f"{sum(p.numel() for g in delta_groups for p in g['params'])} params"
        )
    no_wd = [g for g in param_groups if g["weight_decay"] == 0.0]
    if no_wd:
        msg += f", none on {sum(p.numel() for g in no_wd for p in g['params'])} (norms, biases, layer scale)"
    print(msg)

optimizer = AdamW(param_groups, lr=LR, betas=(0.9, 0.999))
steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
scheduler = CosineWithWarmup(
    optimizer,
    warmup_steps=WARMUP_EPOCHS * steps_per_epoch,
    total_steps=EPOCHS * steps_per_epoch,
    min_lr=MIN_LR,
)
if rank == 0:
    print(
        f"VARIANT={VARIANT}  peak LR {LR:.2e} after {WARMUP_EPOCHS} warmup epochs, "
        f"cosine over {EPOCHS} epochs, global batch {GLOBAL_BATCH_SIZE}, "
        f"delta coeff hold={HOLD_UNTIL} ramp_end={RAMP_END}"
    )


def on_epoch_start(epoch: int, unwrapped) -> None:
    coeff = delta_coeff_at_epoch(epoch, HOLD_UNTIL, RAMP_END)
    unwrapped.set_delta_coeff(coeff)
    if rank == 0:
        print(f"epoch {epoch + 1}/{EPOCHS}: delta_coeff={coeff:.4f}")


train_history_metrics = DictHistoryMetrics(EXPERIMENT_PATH, split="train")
train_history_metrics.addHistoryMetric("loss", MetricLoss, higher_is_better=False)
train_history_metrics.addHistoryMetric("top1acc", Top1AccMetric)
train_history_metrics.addHistoryMetric("gradnorm", MetricGradNorm)
train_history_metrics.addHistoryMetric("lr", MetricLR)
train_history_metrics.addHistoryMetric("deltacoeff", MetricDeltaCoeff, optional=True)
train_history_metrics.addHistoryMetric("deltaratio", MetricDeltaRatio, optional=True, sample_every=20)
val_history_metrics = DictHistoryMetrics(EXPERIMENT_PATH, split="val")
val_history_metrics.addHistoryMetric("loss", MetricLoss, higher_is_better=False)
val_history_metrics.addHistoryMetric("top1acc", Top1AccMetric)

trainer = Trainer(
    experiment_name=EXPERIMENT_NAME,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=SoftTargetCrossEntropy(),
    device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    batch_transforms=build_train_batch_transforms(),
    num_classes=1000,
    amp=False,
    gradient_clipping=None,
    grad_accum_steps=GRAD_ACCUM_STEPS,
    retake=RETAKE,
    on_epoch_start=on_epoch_start,
)
trainer.fit(EPOCHS, train_history_metrics, val_history_metrics)

if use_ddp:
    dist.destroy_process_group()
