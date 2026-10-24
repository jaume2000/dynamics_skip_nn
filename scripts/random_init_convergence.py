#!/usr/bin/env python3
"""Do randomly initialised residual stages also converge to a continuous limit?

Every Conv2d/Linear gets Xavier-normal (gain 1, zero bias); then the stage-3
blocks are re-drawn with Xavier gain ``g``, so the stage-3 input is identical
across gains and only the field strength changes. LayerScale γ is set to 1
(the ConvNeXt default 1e-6 makes f ≈ 0).

For each gain, the stage-3 endpoint x_T (horizon T = 9) of Euler at
R ∈ {1, 10, 100, 1000} and RK4 at R = 1 is compared against RK4 at R = 100.
The trained ConvNeXt-T and weight-tied checkpoints are measured the same way as
reference rows, with their massive-activation channel zeroed (as in the paper
metrics); random networks have no such channel and use all channels.

Usage (from repo root):
  python scripts/random_init_convergence.py
  python scripts/random_init_convergence.py --gains 0.5 1 2 --n-images 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data.imagenet import ImageNetDataset  # noqa: E402
from data.transforms.transforms import build_val_transforms  # noqa: E402
from scripts import feature_map_explorer as fme  # noqa: E402
from scripts.compare_solver_convergence import (  # noqa: E402
    build_convnext,
    build_shared,
    integrate_x,
    pairwise_stats,
)

EULER_R = (1, 10, 100, 1000)
REF = ("RK4", 100)
MASSIVE_CHANNEL = {"blocks": 195, "tied": 236}
KINDS = ("blocks", "tied", "resnet50", "resnet101", "swin")


def fme_key(kind: str) -> str:
    return "convnext" if kind == "blocks" else kind


def xavier_(module: nn.Module, gain: float) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def stage3_modules(kind: str, model: nn.Module) -> list[nn.Module]:
    if kind == "tied":
        # DeltaConvnextBlock keeps its layers in an unregistered ``_shared`` attribute.
        return [model.deltifiedStage3[0]._shared]
    return fme.stage3_raw_blocks(fme_key(kind), model)


def random_model(kind: str, gain: float, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if kind == "tied":
        model = fme.load_shared_convnext(None)
    elif kind == "blocks":
        model = fme.load_interpoled_convnext(None)
    else:
        model = fme.load_interpoled_model(kind, random_init=True)
        torch.manual_seed(seed)
    with torch.no_grad():
        xavier_(model, 1.0)
        for blk in stage3_modules(kind, model):
            xavier_(blk, gain)
        fme.set_layerscale_value(model, 1.0)
    return model


def trained_model(kind: str) -> nn.Module:
    return fme.load_interpoled_model("convnext_shared" if kind == "tied" else fme_key(kind))


def build_generic(key: str, model: nn.Module, repeats: int, euler_step: float):
    n_blocks = fme.stage3_n_blocks(key, model)
    blocks = [i for i in range(n_blocks) for _ in range(repeats)]
    field = fme.stage3_field_blocks(key, model, blocks, euler_step=euler_step)
    enter = lambda batch, m=model: fme.enter_stage3(key, m, batch)
    r1 = [fme._wrap_stage3_block(key, b) for b in fme.stage3_raw_blocks(key, model)]
    return enter, field, r1, euler_step


def builder(kind: str):
    if kind == "tied":
        return lambda m, R: build_shared(m, D=9 * R, euler_step=1.0 / R)
    if kind == "blocks":
        return lambda m, R: build_convnext(m, repeats=R, euler_step=1.0 / R)
    return lambda m, R: build_generic(kind, m, repeats=R, euler_step=1.0 / R)


def dominant_channel(x: torch.Tensor) -> tuple[int, float]:
    """Channel with the largest mean spatial norm, and its ratio to the median channel."""
    per_ch = x.flatten(2).norm(dim=2).mean(0)
    top = int(per_ch.argmax())
    return top, float(per_ch[top] / per_ch.median().clamp_min(1e-30))


@torch.no_grad()
def step_ratio(
    enter, r1_blocks, idx, *, dataset, device, batch_size, ignore: list[int] | None = None
) -> tuple[float, float, float]:
    """Mean ||f_k(x_k)|| / ||x_k|| along the R=1 path, plus mean ||x_0||, ||x_T||."""

    def norm(t):
        if ignore:
            t = t.clone()
            t[:, ignore] = 0
        return t.flatten(1).norm(dim=1)

    ratios, n0, nT = [], [], []
    for s in range(0, len(idx), batch_size):
        imgs = torch.stack([dataset[i][0] for i in idx[s : s + batch_size]]).to(device)
        x = enter(imgs)
        n0.append(norm(x))
        per_block = []
        for blk in r1_blocks:
            h = blk(x) - x
            per_block.append(norm(h) / norm(x))
            x = x + h
        ratios.append(torch.stack(per_block).mean(0))
        nT.append(norm(x))
    cat = lambda t: float(torch.cat(t).mean())
    return cat(ratios), cat(n0), cat(nT)


def measure(model, kind, label, gain, idx, kw, ignores=(None,)) -> list[dict]:
    """``ignores``: channel lists to zero in the metrics; ``"auto"`` = dominant channel."""
    build = builder(kind)
    xs = {}
    for meth, R in [("RK1", R) for R in EULER_R] + [("RK4", 1), REF]:
        enter, field, _, es = build(model, R)
        xs[(meth, R)] = integrate_x(enter, field, idx, euler_step=es, method=meth, **kw)
    enter, _, r1, _ = build(model, 1)
    top, top_ratio = dominant_channel(xs[REF])
    print(f"  {kind} {label} g={gain:g}: dominant channel {top} ({top_ratio:.1f}x median)")
    rows = []
    for ignore in ignores:
        if ignore == "auto":
            ignore = [top]
        rho, n0, nT = step_ratio(enter, r1, idx, ignore=ignore, **kw)
        for (meth, R), x in xs.items():
            if (meth, R) == REF:
                continue
            s = pairwise_stats(x, xs[REF], ignore)
            rows.append(dict(
                model=kind, init=label, gain=gain, ignore_ch=ignore[0] if ignore else None,
                solver="Euler" if meth == "RK1" else meth, R=R,
                rel=s["rel_mean"], cos=s["cos_mean"], angle_deg=s["angle_mean_deg"],
                step_ratio=rho, norm_x0=n0, norm_xT=nT, finite=bool(torch.isfinite(x).all()),
            ))
            print(f"  {kind:<9} {label:<8} g={gain:<5g} ign={str(ignore):<6} "
                  f"{rows[-1]['solver']:<5} R={R:<5} "
                  f"rel={s['rel_mean']:.4f} ang={s['angle_mean_deg']:6.2f}°  "
                  f"|f|/|x|={rho:.3f} |x0|={n0:.0f} |xT|={nT:.0f}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gains", type=float, nargs="*", default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--models", nargs="+", default=["blocks", "tied"], choices=KINDS)
    p.add_argument("--n-images", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-trained", action="store_true")
    p.add_argument("--out", type=Path, default=_REPO / "outputs" / "random_init_convergence" / "xavier.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ImageNetDataset(split="validation", transforms=build_val_transforms())
    labels = ds.ds.data.column("label").to_pylist()
    idx = fme.sample_one_per_class(labels, args.n_images, args.seed)
    kw = dict(batch_size=args.batch_size, dataset=ds, device=device)

    rows: list[dict] = []
    for kind in args.models:
        if not args.no_trained:
            m = trained_model(kind).to(device).eval()
            ignores = [[MASSIVE_CHANNEL[kind]]] if kind in MASSIVE_CHANNEL else [None, "auto"]
            rows += measure(m, kind, "trained", float("nan"), idx, kw, ignores)
            del m
        for g in args.gains:
            m = random_model(kind, g, args.seed).to(device).eval()
            rows += measure(m, kind, "xavier", g, idx, kw)
            del m
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
