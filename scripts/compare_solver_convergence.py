#!/usr/bin/env python3
"""Pairwise stage-3 endpoint agreement across solvers / step sizes.

Tests whether refined discretisations share a continuous limit:
  - Euler vs RK4 at the same (D or R, Δt)
  - Euler R vs 2R (Cauchy), same horizon T

Usage (from repo root):
  python scripts/compare_solver_convergence.py
  python scripts/compare_solver_convergence.py --n-images 32 --batch-size 8
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

load_dotenv(_REPO / ".env")

from data.imagenet import ImageNetDataset  # noqa: E402
from data.transforms.transforms import build_val_transforms  # noqa: E402
from scripts import feature_map_explorer as fme  # noqa: E402


@torch.no_grad()
def integrate_x(
    enter_fn,
    field_blocks: list,
    indices: list[int],
    *,
    euler_step: float,
    method: str | None,
    batch_size: int,
    dataset,
    device: torch.device,
) -> torch.Tensor:
    """Return final stage-3 state ``x`` for every image, shape ``[N, C, H, W]``."""
    method = method.upper() if isinstance(method, str) else method
    D = len(field_blocks)
    es = float(euler_step)
    outs: list[torch.Tensor] = []

    def field_at(block):
        return lambda z: block(z) - z

    for start in tqdm(range(0, len(indices), batch_size), desc=f"D={D} {method or 'RK1'}", leave=False):
        batch_idx = indices[start : start + batch_size]
        imgs = torch.stack([dataset[i][0] for i in batch_idx]).to(device)
        x = enter_fn(imgs)
        for d in range(D):
            f = field_at(field_blocks[d])
            x = fme.rk_step(f, x, es, method)
        outs.append(x.detach().float().cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def integrate_r1(
    enter_fn,
    r1_blocks: list,
    indices: list[int],
    *,
    batch_size: int,
    dataset,
    device: torch.device,
) -> torch.Tensor:
    """Native one-step-per-block path (ES=1)."""
    outs: list[torch.Tensor] = []
    for start in tqdm(range(0, len(indices), batch_size), desc="R1", leave=False):
        batch_idx = indices[start : start + batch_size]
        imgs = torch.stack([dataset[i][0] for i in batch_idx]).to(device)
        x = enter_fn(imgs)
        for block in r1_blocks:
            x = block(x)
        outs.append(x.detach().float().cpu())
    return torch.cat(outs, dim=0)


def pairwise_stats(xa: torch.Tensor, xb: torch.Tensor, ignore_chs: list[int] | None = None) -> dict:
    """Per-image L2 / relative / cosine between two endpoint clouds."""
    # float64: in float32, cos rounds to 1 below ~0.2° and the angle reads 0.
    a, b = xa.double(), xb.double()
    if ignore_chs:
        a[:, ignore_chs] = 0
        b[:, ignore_chs] = 0
    af, bf = a.flatten(1), b.flatten(1)
    dist = (af - bf).norm(dim=1)
    rel_b = dist / bf.norm(dim=1).clamp_min(1e-30)
    rel_a = dist / af.norm(dim=1).clamp_min(1e-30)
    cos = F.cosine_similarity(af, bf, dim=1).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.arccos(cos))
    return {
        "n": int(a.shape[0]),
        "dist_mean": float(dist.mean()),
        "dist_median": float(dist.median()),
        "rel_mean": float(rel_b.mean()),
        "rel_median": float(rel_b.median()),
        "rel_to_a_mean": float(rel_a.mean()),
        "cos_mean": float(cos.mean()),
        "cos_median": float(cos.median()),
        "angle_mean_deg": float(angle.mean()),
        "angle_median_deg": float(angle.median()),
        "norm_a_mean": float(af.norm(dim=1).mean()),
        "norm_b_mean": float(bf.norm(dim=1).mean()),
    }


def rank_ignore_channels(x: torch.Tensor, h_proxy: torch.Tensor | None, k: int) -> list[int]:
    """Top-k channels by ||·|| over spatial dims, averaged over images (on ``x`` if no h)."""
    if k <= 0:
        return []
    t = h_proxy if h_proxy is not None else x
    # [N,C,H,W] → score per channel
    score = t.flatten(2).norm(dim=2).mean(0)  # [C]
    return score.topk(k).indices.tolist()


def fmt(s: dict) -> str:
    return (
        f"n={s['n']}  ||Δ||={s['dist_mean']:.3g} (med {s['dist_median']:.3g})  "
        f"rel={s['rel_mean']:.4f} (med {s['rel_median']:.4f})  "
        f"cos={s['cos_mean']:.4f} (med {s['cos_median']:.4f})  "
        f"∠={s['angle_mean_deg']:.2f}°  "
        f"||x_a||={s['norm_a_mean']:.1f} ||x_b||={s['norm_b_mean']:.1f}"
    )


def build_shared(model, D: int, euler_step: float):
    field = [model.deltifiedStage3[0]] * D
    enter = lambda batch, m=model: m.stage2(m.stage1(m.stem(batch)))
    r1 = [model.deltifiedStage3[0]] * int(model.stage3_length)
    return enter, field, r1, euler_step


def build_convnext(model, repeats: int, euler_step: float):
    n_blocks = fme.stage3_n_blocks("convnext", model)
    blocks = [i for i in range(n_blocks) for _ in range(repeats)]
    field = fme.stage3_field_blocks(
        "convnext", model, blocks, euler_step=euler_step, weight_interpolation="plain"
    )
    enter = lambda batch, m=model: fme.enter_stage3("convnext", m, batch)
    r1 = [fme._wrap_stage3_block("convnext", b) for b in fme.stage3_raw_blocks("convnext", model)]
    return enter, field, r1, euler_step


def run_pair(name, xa, xb, ignore):
    print(f"\n[{name}] full:     {fmt(pairwise_stats(xa, xb))}")
    if ignore:
        print(f"[{name}] ignore{len(ignore)}: {fmt(pairwise_stats(xa, xb, ignore))}  ch={ignore}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ignore-top-k", type=int, default=1)
    p.add_argument(
        "--skip-r200",
        action="store_true",
        help="Skip the Euler R=100 vs R=200 Cauchy checks (ConvNeXt and shared)",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    dataset = ImageNetDataset(split="validation", transforms=build_val_transforms())
    labels = dataset.ds.data.column("label").to_pylist()
    indices = fme.sample_one_per_class(labels, args.n_images, args.seed)
    print(f"n_images={len(indices)} seed={args.seed} ignore_top_k={args.ignore_top_k}")

    # ----- Shared ConvNeXt: R=100 (D=900, Δt=0.01), same step as ConvNeXt R=100 -----
    print("\n=== shared ConvNeXt  D=900  Δt=0.01  Euler vs RK4 vs R1 (+ D=1800 Cauchy) ===")
    shared = fme.load_interpoled_model("convnext_shared").to(device).eval()
    enter, field, r1_blocks, step = build_shared(shared, D=900, euler_step=0.01)
    x_eu = integrate_x(
        enter, field, indices, euler_step=step, method="RK1",
        batch_size=args.batch_size, dataset=dataset, device=device,
    )
    x_rk = integrate_x(
        enter, field, indices, euler_step=step, method="RK4",
        batch_size=args.batch_size, dataset=dataset, device=device,
    )
    x_r1 = integrate_r1(
        enter, r1_blocks, indices, batch_size=args.batch_size, dataset=dataset, device=device,
    )
    enter, field9, _, _ = build_shared(shared, D=9, euler_step=1.0)
    x_rk9 = integrate_x(
        enter, field9, indices, euler_step=1.0, method="RK4",
        batch_size=args.batch_size, dataset=dataset, device=device,
    )
    ign_s = rank_ignore_channels(x_eu, x_eu, args.ignore_top_k)
    run_pair("shared Euler↔RK4", x_eu, x_rk, ign_s)
    run_pair("shared Euler↔R1", x_eu, x_r1, ign_s)
    run_pair("shared RK4↔R1", x_rk, x_r1, ign_s)
    run_pair("shared RK4 D9↔RK4 D900", x_rk9, x_rk, ign_s)
    if not args.skip_r200:
        enter, field1800, _, _ = build_shared(shared, D=1800, euler_step=0.005)
        x1800 = integrate_x(
            enter, field1800, indices, euler_step=0.005, method="RK1",
            batch_size=args.batch_size, dataset=dataset, device=device,
        )
        run_pair("shared Euler D900↔D1800", x_eu, x1800, ign_s)
    del shared, x_eu, x_rk, x_r1, x_rk9
    torch.cuda.empty_cache()

    # ----- ConvNeXt: R=100 Euler vs RK4, and R=100 vs R=200 -----
    print("\n=== ConvNeXt-T  R=100  Δt=0.01  Euler vs RK4 (+ R=200 Cauchy) ===")
    conv = fme.load_interpoled_model("convnext").to(device).eval()
    enter, field100, r1_blocks, step100 = build_convnext(conv, repeats=100, euler_step=0.01)
    x100 = integrate_x(
        enter, field100, indices, euler_step=step100, method="RK1",
        batch_size=args.batch_size, dataset=dataset, device=device,
    )
    x100_rk = integrate_x(
        enter, field100, indices, euler_step=step100, method="RK4",
        batch_size=args.batch_size, dataset=dataset, device=device,
    )
    x_r1 = integrate_r1(
        enter, r1_blocks, indices, batch_size=args.batch_size, dataset=dataset, device=device,
    )
    ign_c = rank_ignore_channels(x100, x100, args.ignore_top_k)
    run_pair("convnext R100 Euler↔RK4", x100, x100_rk, ign_c)
    run_pair("convnext R100 Euler↔R1", x100, x_r1, ign_c)
    run_pair("convnext R100 RK4↔R1", x100_rk, x_r1, ign_c)

    if not args.skip_r200:
        enter, field200, _, step200 = build_convnext(conv, repeats=200, euler_step=0.005)
        x200 = integrate_x(
            enter, field200, indices, euler_step=step200, method="RK1",
            batch_size=args.batch_size, dataset=dataset, device=device,
        )
        run_pair("convnext Euler R100↔R200", x100, x200, ign_c)
        run_pair("convnext Euler R200↔R1", x200, x_r1, ign_c)

    print("\nDone.")


if __name__ == "__main__":
    main()
