"""Finetune shared ConvNeXt-T stage-4 + head on a D-unrolled stage-3 (S1–S3 frozen).

Loads ``outputs/shared_convnextv1_imagenet/weights/last.pth``, sets stage-3 length
to ``D`` with Euler step ``9/D`` (horizon T=9), freezes stem/S1/S2/S3, trains only
stage4 + globalPool + fc.

Env:
  D                default 900 (Δt=0.01, same step as R=100)
  EULER_STEP       default 9/D (keep T=9)
  EPOCHS           default 10
  LR               default 1e-6
  WARMUP_EPOCHS    default 1
  EXPERIMENT_NAME  default convnext_shared_D{D}_ES{es}_finetune_s4
  BASELINE_WEIGHTS default outputs/shared_convnextv1_imagenet/weights/last.pth
  RETAKE           0|1
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from timm.loss import SoftTargetCrossEntropy

from data.imagenet import ImageNetDataset
from data.transforms.transforms import (
    build_train_batch_transforms,
    build_train_transforms,
    build_val_transforms,
)
from engine.trainer import Trainer
from metrics.metricGradNorm import MetricGradNorm
from metrics.metricHistory import DictHistoryMetrics
from metrics.metricLoss import MetricLoss
from metrics.metricLR import MetricLR
from metrics.top1acc import Top1AccMetric
from models.backbones.delta_convnext import DeltaConvNext
from optim.cosineSchedule import CosineWithWarmup
from optim.paramGroups import build_param_groups
from scripts.feature_map_explorer import load_shared_convnext
from utils.env import load_dotenv


load_dotenv()

D = int(os.environ.get("D", "900"))
_es_env = os.environ.get("EULER_STEP")
EULER_STEP = float(_es_env) if _es_env is not None else 9.0 / D
METHOD = os.environ.get("METHOD", "RK1")  # logged only; DeltaConvnextBlock uses eulerStep

_es_tag = f"{EULER_STEP:g}".replace(".", "p")
EXPERIMENT_NAME = os.environ.get(
    "EXPERIMENT_NAME",
    f"convnext_shared_D{D}_ES{_es_tag}_finetune_s4",
)
EXPERIMENT_PATH = Path("outputs") / EXPERIMENT_NAME
BASELINE_WEIGHTS = Path(
    os.environ.get(
        "BASELINE_WEIGHTS",
        "outputs/shared_convnextv1_imagenet/weights/last.pth",
    )
)

USE_DDP = True
EPOCHS = int(os.environ.get("EPOCHS", "10"))
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "1"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "1024"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
LR = float(os.environ.get("LR", "1e-6"))
MIN_LR = float(os.environ.get("MIN_LR", "1e-8"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.05"))
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


def freeze_encoder(model: DeltaConvNext) -> None:
    """Freeze stem + stages 1–3 (shared block + unrolled S3). Train S4 + head."""
    model.freezeStages([0, 1, 2, 3])
    model.stage4.requires_grad_(True)
    model.globalPool.requires_grad_(True)
    model.fc.requires_grad_(True)


local_rank, rank, world_size = setup_ddp() if USE_DDP else (0, 0, 1)
use_ddp = USE_DDP and world_size > 1 and torch.cuda.is_available()
if GLOBAL_BATCH_SIZE % (BATCH_SIZE * world_size):
    raise ValueError(
        f"GLOBAL_BATCH_SIZE={GLOBAL_BATCH_SIZE} not divisible by {BATCH_SIZE} x {world_size}"
    )
GRAD_ACCUM_STEPS = GLOBAL_BATCH_SIZE // (BATCH_SIZE * world_size)

if torch.cuda.is_available():
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cpu")
    use_ddp = False

if not RETAKE:
    if not BASELINE_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Shared baseline weights not found at {BASELINE_WEIGHTS}. "
            "Train shared ConvNeXt first, or set BASELINE_WEIGHTS."
        )
    # load_shared_convnext: rewire + strict=False (tail.* aliases deltifiedStage3 tail).
    model = load_shared_convnext(BASELINE_WEIGHTS)
else:
    model = DeltaConvNext(useDeltas=False)
    model.rewire()
    if rank == 0:
        print("RETAKE=1: skipping baseline load; Trainer restores experiment checkpoint")

# Unroll after loading so checkpoint matches the trained D=9 shared layout.
model.setStage3Length(D)
# setStage3Length sets eulerStep = 9/D from the pre-unroll length; override if requested.
if abs(model.eulerStep - EULER_STEP) > 1e-12:
    if rank == 0:
        print(f"Overriding eulerStep {model.eulerStep:g} -> {EULER_STEP:g}")
    model.eulerStep = EULER_STEP
    for block in model.deltifiedStage3:
        if hasattr(block, "eulerStep"):
            block.eulerStep = EULER_STEP

freeze_encoder(model)
model = model.to(device)

T = D * EULER_STEP
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
if rank == 0:
    print(
        f"Schedule: D={D}, ES={EULER_STEP:g}, method={METHOD}, T={T:g} "
        f"({D} shared residual steps + tail)"
    )
    print(f"Trainable params: {n_train:,} | frozen: {n_frozen:,}")
    print(f"Experiment: {EXPERIMENT_NAME}")

if use_ddp:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if rank == 0:
        print(f"DDP on {world_size} GPUs, grad accumulation x{GRAD_ACCUM_STEPS}")

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

param_groups = build_param_groups(model, weight_decay=WEIGHT_DECAY)
if rank == 0:
    print(
        f"Weight decay {WEIGHT_DECAY} on {sum(p.numel() for p in param_groups[0]['params']):,} params, "
        f"none on {sum(p.numel() for p in param_groups[1]['params']):,}"
    )
optimizer = AdamW(param_groups, lr=LR, betas=(0.9, 0.999))
steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
scheduler = CosineWithWarmup(
    optimizer,
    warmup_steps=WARMUP_EPOCHS * steps_per_epoch,
    total_steps=EPOCHS * steps_per_epoch,
    min_lr=MIN_LR,
)
if rank == 0:
    print(f"Peak LR: {LR:.2e} after {WARMUP_EPOCHS} warmup epochs, cosine over {EPOCHS} epochs")

train_history_metrics = DictHistoryMetrics(EXPERIMENT_PATH, split="train")
train_history_metrics.addHistoryMetric("loss", MetricLoss, higher_is_better=False)
train_history_metrics.addHistoryMetric("top1acc", Top1AccMetric)
train_history_metrics.addHistoryMetric("gradnorm", MetricGradNorm)
train_history_metrics.addHistoryMetric("lr", MetricLR)
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
)

if not RETAKE:
    model_eval = model.module if hasattr(model, "module") else model
    model_eval.eval()
    correct = 0
    total = 0
    if rank == 0:
        print("Running epoch-0 probe (unrolled S3, frozen encoder, pretrained head)...")
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i >= 40:
                    break
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model_eval(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.numel()
        print(f"Epoch-0 probe top1 ≈ {100.0 * correct / max(total, 1):.2f}% on {total} samples")
    if use_ddp:
        dist.barrier()

trainer.fit(EPOCHS, train_history_metrics, val_history_metrics)

if use_ddp:
    dist.destroy_process_group()
