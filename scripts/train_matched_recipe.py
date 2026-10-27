"""Matched-recipe ConvNeXt-T: untied or weight-tied stage-3, seeded.

Same optim as the original tied D9 run: AdamW, peak LR 1e-3 after 10 warmup
epochs, cosine to 1e-7, weight decay 0.05, 300 epochs, global batch 1024
(grad-accum when world_size < 4 so 2-GPU half-nodes keep the recipe).

Env:
  MODEL              tied | untied   (required)
  SEED               int             (required)
  EXPERIMENT_NAME    default {tied:shared_convnextv1_matched_seed{S},
                              untied:convnextv1_matched_seed{S}}
  GLOBAL_BATCH_SIZE  default 1024
  BATCH_SIZE         default 256 (per GPU / micro-batch)
  RETAKE             0|1

  torchrun --standalone --nproc_per_node=2 scripts/train_matched_recipe.py
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
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
from models.backbones.convnext import ConvNextV1
from models.backbones.delta_convnext import DeltaConvNext
from optim.cosineSchedule import CosineWithWarmup
from optim.paramGroups import build_param_groups_with_delta_weight_decay
from utils.env import load_dotenv

load_dotenv()

MODEL = os.environ.get("MODEL", "").strip().lower()
if MODEL not in ("tied", "untied"):
    raise SystemExit(f"MODEL must be tied|untied (got {MODEL!r})")

_seed_raw = os.environ.get("SEED", "").strip()
if not _seed_raw:
    raise SystemExit("SEED is required (non-negative int)")
SEED = int(_seed_raw)

_default_name = (
    f"shared_convnextv1_matched_seed{SEED}"
    if MODEL == "tied"
    else f"convnextv1_matched_seed{SEED}"
)
# Prefer MATCHED_EXPERIMENT_NAME so .env's EXPERIMENT_NAME cannot hijack the run.
EXPERIMENT_NAME = os.environ.get("MATCHED_EXPERIMENT_NAME") or os.environ.get(
    "EXPERIMENT_NAME", _default_name
)
# If EXPERIMENT_NAME came from .env and looks like another experiment, force default
# unless MATCHED_EXPERIMENT_NAME was set.
if "MATCHED_EXPERIMENT_NAME" not in os.environ and EXPERIMENT_NAME in {
    "shared_convnextv1_imagenet",
    "convnextv1_imagenet",
    "convnextv1_imagenet_tied_schedule",
}:
    EXPERIMENT_NAME = _default_name

EXPERIMENT_PATH = Path("outputs") / EXPERIMENT_NAME
EPOCHS = 300
WARMUP_EPOCHS = 10
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "1024"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
LR = 1e-3
MIN_LR = 1e-7
WEIGHT_DECAY = 0.05
DROP_PATH_RATE = 0.1
RETAKE = os.environ.get("RETAKE", "0") == "1"


def available_cpus() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(slurm_cpus) if slurm_cpus else len(os.sched_getaffinity(0))


local_rank = int(os.environ.get("LOCAL_RANK", 0))
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
use_ddp = world_size > 1 and torch.cuda.is_available()
if use_ddp:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

if GLOBAL_BATCH_SIZE % max(world_size, 1):
    raise ValueError(
        f"GLOBAL_BATCH_SIZE={GLOBAL_BATCH_SIZE} not divisible by world_size={world_size}"
    )
per_rank = GLOBAL_BATCH_SIZE // max(world_size, 1)
if per_rank % BATCH_SIZE:
    raise ValueError(
        f"per-rank batch {per_rank} not divisible by micro-batch BATCH_SIZE={BATCH_SIZE}"
    )
GRAD_ACCUM_STEPS = per_rank // BATCH_SIZE

device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
random.seed(SEED + rank)
np.random.seed(SEED + rank)
torch.manual_seed(SEED + rank)
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED + rank)

if MODEL == "untied":
    model = ConvNextV1(drop_path_rate=DROP_PATH_RATE).to(device)
elif MODEL == "tied":
    model = DeltaConvNext(useDeltas=False)
    model.rewire()
    model = model.to(device)
else:
    raise AssertionError(MODEL)

if use_ddp:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

train_dataset = ImageNetDataset(split="train", transforms=build_train_transforms())
val_dataset = ImageNetDataset(split="validation", transforms=build_val_transforms())
num_workers = max(1, available_cpus() // world_size) if use_ddp else available_cpus()
train_sampler = (
    DistributedSampler(train_dataset, shuffle=True, seed=SEED) if use_ddp else None
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=train_sampler is None,
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
    model, weight_decay=WEIGHT_DECAY, delta_weight_decay=WEIGHT_DECAY
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
    print(
        f"MODEL={MODEL} SEED={SEED} EXPERIMENT_NAME={EXPERIMENT_NAME}\n"
        f"peak LR {LR:.2e} after {WARMUP_EPOCHS} warmup epochs, cosine to {MIN_LR:.0e} "
        f"over {EPOCHS} epochs; global batch {GLOBAL_BATCH_SIZE} "
        f"({world_size}×{BATCH_SIZE}×accum{GRAD_ACCUM_STEPS}); "
        f"steps/epoch={steps_per_epoch}"
    )
    sys.stdout.flush()

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
trainer.fit(EPOCHS, train_history_metrics, val_history_metrics)

if use_ddp:
    dist.destroy_process_group()
