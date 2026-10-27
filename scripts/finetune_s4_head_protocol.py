"""Common stage-4 + head finetune protocol for untied and weight-tied ConvNeXt-T.

Stage 3 is refined at fixed horizon T=9 (untied: R Euler repeats of each block with
Δt=1/R; tied: 9R applications of the shared block with Δt=1/R). Stem and stages 1-3 are
frozen *and kept in eval mode*, so their drop path is off; stage 4 keeps its own drop
path. The older scripts (finetune_s4_r_convnext.py, finetune_s4_d_shared_convnext.py)
froze the encoder but left it in train mode, so drop path stayed active in the frozen
stages of both models (S1-S3 untied, S1-S2 tied).

Protocol: AdamW, constant lr 5e-6 (no warmup, no decay), weight decay 0.05, global
batch 1024, standard training augmentation, 10 full epochs, seed 0. A full-validation
evaluation is run before the first step and written to ``epoch0.json``; the Trainer
logs train/val histories after every epoch; ``config.json`` stores the protocol.

Env:
  MODEL            untied | tied
  REPEATS          R (default 1)
  EPOCHS           default 10
  LR               default 5e-6 (constant)
  FT_EXPERIMENT_NAME  default ft_s4_{MODEL}_R{R}_protocol
  FT_RETAKE           0|1

  torchrun --nproc_per_node=2 scripts/finetune_s4_head_protocol.py
"""

from __future__ import annotations

import json
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
from models.backbones.interpoled_convnext import InterpoledConvNextV1
from optim.cosineSchedule import CosineWithWarmup
from optim.paramGroups import build_param_groups
from utils.env import load_dotenv

load_dotenv()

MODEL = os.environ.get("MODEL", "untied")
REPEATS = int(os.environ.get("REPEATS", "1"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
LR = float(os.environ.get("LR", "5e-6"))
WEIGHT_DECAY = 0.05
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "1024"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
SEED = 0
# FT_* names because the SLURM jobs source .env, which sets EXPERIMENT_NAME and RETAKE.
RETAKE = os.environ.get("FT_RETAKE", "0") == "1"
EXPERIMENT_NAME = os.environ.get("FT_EXPERIMENT_NAME", f"ft_s4_{MODEL}_R{REPEATS}_protocol")
EXPERIMENT_PATH = Path("outputs") / EXPERIMENT_NAME
WEIGHTS = {
    "untied": Path("outputs/convnextv1_imagenet/weights/last.pth"),
    "tied": Path("outputs/shared_convnextv1_imagenet/weights/last.pth"),
}
TRAINABLE = ("stage4", "globalPool", "fc")


def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def build_model() -> torch.nn.Module:
    if MODEL == "untied":
        model = InterpoledConvNextV1(blocks=[i for i in range(9) for _ in range(REPEATS)],
                                     euler_step=1.0 / REPEATS)
        if not RETAKE:
            model.load_state_dict(torch.load(WEIGHTS["untied"], map_location="cpu")["model"], strict=True)
    elif MODEL == "tied":
        from scripts.feature_map_explorer import load_shared_convnext

        model = load_shared_convnext(None if RETAKE else WEIGHTS["tied"])
        model.setStage3Length(9 * REPEATS)
        assert abs(model.eulerStep - 1.0 / REPEATS) < 1e-12, model.eulerStep
    else:
        raise ValueError(MODEL)
    for name, child in model.named_children():
        child.requires_grad_(name in TRAINABLE)
    return model


def keep_frozen_in_eval(model: torch.nn.Module) -> list[str]:
    """Patch ``model.train`` so fully frozen top-level children always stay in eval."""
    frozen = [n for n, c in model.named_children()
              if any(True for _ in c.parameters()) and not any(p.requires_grad for p in c.parameters())]
    base_train = model.train

    def train(mode: bool = True):
        base_train(mode)
        for n in frozen:
            getattr(model, n).eval()
        return model

    model.train = train
    return frozen


@torch.no_grad()
def full_validation(model, loader, device, use_ddp) -> dict:
    model.eval()
    stats = torch.zeros(3, device=device, dtype=torch.float64)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        stats += torch.tensor([(logits.argmax(1) == y).sum().item(), criterion(logits, y).item(), y.numel()],
                              device=device, dtype=torch.float64)
    if use_ddp:
        dist.all_reduce(stats)
    return {"top1acc": float(stats[0] / stats[2]), "loss": float(stats[1] / stats[2]), "n": int(stats[2])}


def main() -> None:
    local_rank, rank, world_size = setup_ddp()
    use_ddp = world_size > 1 and torch.cuda.is_available()
    if GLOBAL_BATCH_SIZE % (BATCH_SIZE * world_size):
        raise ValueError(f"GLOBAL_BATCH_SIZE={GLOBAL_BATCH_SIZE} not divisible by {BATCH_SIZE}x{world_size}")
    grad_accum = GLOBAL_BATCH_SIZE // (BATCH_SIZE * world_size)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    random.seed(SEED + rank)
    np.random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)

    model = build_model()
    frozen = keep_frozen_in_eval(model)
    model = model.to(device)
    model.train()
    active_dp = [n for n, m in model.named_modules()
                 if type(m).__name__ == "StochasticDepth" and m.training and not n.startswith("stage4")]
    assert not active_dp, active_dp
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_ds = ImageNetDataset(split="train", transforms=build_train_transforms())
    val_ds = ImageNetDataset(split="validation", transforms=build_val_transforms())
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))))
    workers = max(1, cpus // world_size)
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=SEED) if use_ddp else None
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=train_sampler is None,
                              sampler=train_sampler, num_workers=workers, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, sampler=val_sampler,
                            num_workers=workers, pin_memory=True, persistent_workers=True)

    config = dict(model=MODEL, repeats=REPEATS, euler_step=1.0 / REPEATS, horizon=9.0,
                  stage3_applications=9 * REPEATS, weights=str(WEIGHTS[MODEL]), epochs=EPOCHS,
                  lr=LR, schedule="constant", warmup_epochs=0, weight_decay=WEIGHT_DECAY,
                  global_batch=GLOBAL_BATCH_SIZE, world_size=world_size, seed=SEED,
                  trainable=list(TRAINABLE), frozen_in_eval=frozen, trainable_params=n_train)
    EXPERIMENT_PATH.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (EXPERIMENT_PATH / "config.json").write_text(json.dumps(config, indent=2))
        print(json.dumps(config, indent=2))

    if not RETAKE:
        # DistributedSampler pads the last shard, so n can exceed 50 000 by < world_size.
        ev = full_validation(model, val_loader, device, use_ddp)
        if rank == 0:
            (EXPERIMENT_PATH / "epoch0.json").write_text(json.dumps(ev, indent=2))
            print(f"epoch-0 full validation: {ev}")
        model.train()

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = AdamW(build_param_groups(model, weight_decay=WEIGHT_DECAY), lr=LR, betas=(0.9, 0.999))
    steps = len(train_loader) // grad_accum
    scheduler = CosineWithWarmup(optimizer, warmup_steps=0, total_steps=EPOCHS * steps, min_lr=LR)

    train_hist = DictHistoryMetrics(EXPERIMENT_PATH, split="train")
    train_hist.addHistoryMetric("loss", MetricLoss, higher_is_better=False)
    train_hist.addHistoryMetric("top1acc", Top1AccMetric)
    train_hist.addHistoryMetric("gradnorm", MetricGradNorm)
    train_hist.addHistoryMetric("lr", MetricLR)
    val_hist = DictHistoryMetrics(EXPERIMENT_PATH, split="val")
    val_hist.addHistoryMetric("loss", MetricLoss, higher_is_better=False)
    val_hist.addHistoryMetric("top1acc", Top1AccMetric)

    trainer = Trainer(experiment_name=EXPERIMENT_NAME, model=model, optimizer=optimizer,
                      scheduler=scheduler, criterion=SoftTargetCrossEntropy(), device=device,
                      train_loader=train_loader, val_loader=DataLoader(
                          val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers,
                          pin_memory=True, persistent_workers=True),
                      batch_transforms=build_train_batch_transforms(), num_classes=1000, amp=False,
                      gradient_clipping=None, grad_accum_steps=grad_accum, retake=RETAKE)
    trainer.fit(EPOCHS, train_hist, val_hist)
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
