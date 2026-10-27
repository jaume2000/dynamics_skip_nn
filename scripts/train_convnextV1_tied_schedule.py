"""Untied ConvNeXt-T trained with the weight-tied recipe (phase 2.3, not launched).

Identical to scripts/train_deltaConvnextV1.py (DDP, 256 images per GPU on 4 GPUs = 1024
global, AdamW, weight decay 0.05, peak lr 1e-3 after 10 warmup epochs, cosine to 1e-7,
300 epochs, same augmentation and batch transforms, no AMP) except that the model is the
standard untied ConvNeXt-T with drop-path rate 0.1, as in scripts/train_convnextV1.py.

Neither original run fixed a seed; this one seeds Python, NumPy, torch and the sampler
with 0 so it can be repeated, which cannot reproduce the exact sample order of either.

  torchrun --standalone --nproc_per_node=4 scripts/train_convnextV1_tied_schedule.py
"""

import os
import random
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
from data.transforms.transforms import build_train_batch_transforms, build_train_transforms, build_val_transforms
from engine.trainer import Trainer
from metrics.metricGradNorm import MetricGradNorm
from metrics.metricHistory import DictHistoryMetrics
from metrics.metricLoss import MetricLoss
from metrics.metricLR import MetricLR
from metrics.top1acc import Top1AccMetric
from models.backbones.convnext import ConvNextV1
from optim.cosineSchedule import CosineWithWarmup
from optim.paramGroups import build_param_groups_with_delta_weight_decay
from utils.env import load_dotenv

load_dotenv()
# Fixed name: the SLURM job sources .env, whose EXPERIMENT_NAME belongs to other runs.
EXPERIMENT_NAME = "convnextv1_imagenet_tied_schedule"
EXPERIMENT_PATH = Path("outputs") / EXPERIMENT_NAME
DROP_PATH_RATE = 0.1
EPOCHS = 300
WARMUP_EPOCHS = 10
BATCH_SIZE = 256
LR = 1e-3
MIN_LR = 1e-7
WEIGHT_DECAY = 0.05
SEED = 0
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
if world_size != 4 and rank == 0:
    print(f"WARNING: the tied run used 4 GPUs x {BATCH_SIZE}; world_size={world_size} changes the global batch")
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
random.seed(SEED + rank)
np.random.seed(SEED + rank)
torch.manual_seed(SEED + rank)

model = ConvNextV1(drop_path_rate=DROP_PATH_RATE).to(device)
if use_ddp:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

train_dataset = ImageNetDataset(split="train", transforms=build_train_transforms())
val_dataset = ImageNetDataset(split="validation", transforms=build_val_transforms())
num_workers = max(1, available_cpus() // world_size) if use_ddp else available_cpus()
train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=SEED) if use_ddp else None
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=train_sampler is None,
                          sampler=train_sampler, num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers,
                        pin_memory=True, persistent_workers=num_workers > 0)

param_groups = build_param_groups_with_delta_weight_decay(model, weight_decay=WEIGHT_DECAY,
                                                          delta_weight_decay=WEIGHT_DECAY)
optimizer = AdamW(param_groups, lr=LR, betas=(0.9, 0.999))
steps_per_epoch = len(train_loader)
scheduler = CosineWithWarmup(optimizer, warmup_steps=WARMUP_EPOCHS * steps_per_epoch,
                             total_steps=EPOCHS * steps_per_epoch, min_lr=MIN_LR)
if rank == 0:
    print(f"drop_path_rate={DROP_PATH_RATE}; peak LR {LR:.2e} after {WARMUP_EPOCHS} warmup epochs, "
          f"cosine to {MIN_LR:.0e} over {EPOCHS} epochs; seed {SEED}")

train_history_metrics = DictHistoryMetrics(EXPERIMENT_PATH, split="train")
train_history_metrics.addHistoryMetric("loss", MetricLoss, higher_is_better=False)
train_history_metrics.addHistoryMetric("top1acc", Top1AccMetric)
train_history_metrics.addHistoryMetric("gradnorm", MetricGradNorm)
train_history_metrics.addHistoryMetric("lr", MetricLR)
val_history_metrics = DictHistoryMetrics(EXPERIMENT_PATH, split="val")
val_history_metrics.addHistoryMetric("loss", MetricLoss, higher_is_better=False)
val_history_metrics.addHistoryMetric("top1acc", Top1AccMetric)

trainer = Trainer(experiment_name=EXPERIMENT_NAME, model=model, optimizer=optimizer,
                  scheduler=scheduler, criterion=SoftTargetCrossEntropy(), device=device,
                  train_loader=train_loader, val_loader=val_loader,
                  batch_transforms=build_train_batch_transforms(), num_classes=1000, amp=False,
                  gradient_clipping=None, retake=RETAKE)
trainer.fit(EPOCHS, train_history_metrics, val_history_metrics)

if use_ddp:
    dist.destroy_process_group()
