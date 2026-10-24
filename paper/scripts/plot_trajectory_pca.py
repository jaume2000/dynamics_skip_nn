#!/usr/bin/env python3
"""Stage-3 trajectories of many images in a shared PCA plane, for any backbone.

Integrates x ← x + Δt·f(x) over stage 3 for ``N_PER_CLASS`` validation images of each
class in ``CLASSES``:
  convnext_shared  one weight-tied field, D steps (default D=100, Δt=0.09, T=9)
  others           each native residual repeated R times (default R=100, Δt=0.01, T=9)
The massive channel (top |x| channel at T, found on the first batch) is dropped and every
stored state is mapped on the GPU with a fixed Gaussian random projection to ``PROJ_DIM``
dimensions (Johnson–Lindenstrauss), so full trajectories never sit in RAM. PCA is fitted
on the projected states.

Writes
  outputs/trajectory_pca/<tag>.pt     cache (projected states, labels, speeds)
  figures/trajectory_pca[_<tag>].pdf  PCA of all states | PCA of final states | speed

Usage (from repo root):
    python paper/scripts/plot_trajectory_pca.py                       # weight-tied
    python paper/scripts/plot_trajectory_pca.py --model convnext      # ConvNeXt-T, R=100
    python paper/scripts/plot_trajectory_pca.py --model convnext --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIG = ROOT / "paper" / "figures"
CACHE_DIR = ROOT / "outputs" / "trajectory_pca"
CLASSES = {
    1: "goldfish",
    207: "golden retriever",
    289: "snow leopard",
    404: "airliner",
    417: "balloon",
    430: "basketball",
    555: "fire engine",
    948: "Granny Smith",
    980: "volcano",
    985: "daisy",
}
N_PER_CLASS = 20
SHARED_KEY = "convnext_shared"
TITLES = {
    SHARED_KEY: "Weight-tied ConvNeXt",
    "convnext": "ConvNeXt-T",
    "convnext_droppath0": "ConvNeXt-T (drop-path 0)",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "swin": "Swin-T",
}
# Stored states per trajectory (enough to resolve block switches at R=100).
MAX_STORED = 301
PROJ_DIM = 2048
BATCH = 10
SEED = 0


def build(model_key: str, *, depth: int, repeats: int, step: float, device):
    """``enter(batch)``, per-step fields, ``head(x)`` (or None) and block switch times."""
    import scripts.feature_map_explorer as fme

    model = fme.load_interpoled_model(model_key).to(device).eval()
    if model_key == SHARED_KEY:
        block = model.deltifiedStage3[0]
        assert block.eulerStep == 1.0
        enter = lambda b: model.stage2(model.stage1(model.stem(b)))
        fields = [block] * depth
        head = lambda x: model.fc(model.globalPool(model.stage4(model.tail(x))))
        return enter, fields, head, []
    n_blocks = fme.stage3_n_blocks(model_key, model)
    blocks = fme.resolve_blocks({"repeats": repeats}, n_blocks)
    fields = fme.stage3_field_blocks(model_key, model, blocks, euler_step=step)
    enter = lambda b: fme.enter_stage3(model_key, model, b)
    head = None
    if model_key in fme._CONVNEXT_KEYS:
        def head(x):
            x = model.stage3[n_blocks + 1](model.stage3[n_blocks](x))
            return model.fc(model.globalPool(model.stage4(x)))
    bounds = [(d + 1) * step for d in range(len(blocks) - 1) if blocks[d] != blocks[d + 1]]
    return enter, fields, head, bounds


@torch.no_grad()
def integrate(model_key: str, *, depth: int, repeats: int, step: float, device) -> dict:
    from data.imagenet import ImageNetDataset
    from data.transforms.transforms import build_val_transforms

    dataset = ImageNetDataset(split="validation", transforms=build_val_transforms())
    labels = np.asarray(dataset.ds.data.column("label").to_pylist())
    indices = np.concatenate([np.flatnonzero(labels == c)[:N_PER_CLASS] for c in CLASSES])
    enter, fields, head, bounds = build(model_key, depth=depth, repeats=repeats, step=step,
                                        device=device)
    D = len(fields)
    stride = max(1, -(-D // (MAX_STORED - 1)))
    stored = list(range(0, D + 1, stride))
    if stored[-1] != D:
        stored.append(D)

    def run(x, on_step=None):
        for d, field in enumerate(fields):
            h = field(x) - x
            if on_step:
                on_step(d, x, h)
            x = x + step * h
        if on_step:
            on_step(D, x, fields[-1](x) - x)
        return x

    def load(idx):
        return torch.stack([dataset[int(i)][0] for i in idx]).to(device)

    x_end = run(enter(load(indices[:BATCH])))
    massive = int(x_end.abs().amax(dim=(2, 3)).mean(dim=0).argmax())
    print(f"massive channel: {massive}")

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    proj = None
    states, speeds, preds, massive_ok = [], [], [], []
    for start in range(0, len(indices), BATCH):
        idx = indices[start : start + BATCH]
        x = enter(load(idx))
        keep = torch.tensor([c for c in range(x.shape[1]) if c != massive], device=device)
        if proj is None:
            n_feat = keep.numel() * x.shape[2] * x.shape[3]
            proj = (torch.randn(n_feat, PROJ_DIM, generator=gen) / PROJ_DIM**0.5).to(device)
        traj = torch.empty(len(idx), len(stored), PROJ_DIM)
        speed = torch.empty(len(idx), D + 1)
        slot = {d: k for k, d in enumerate(stored)}

        def on_step(d, x, h):
            speed[:, d] = h.index_select(1, keep).flatten(1).norm(dim=1).cpu()
            if d in slot:
                traj[:, slot[d]] = (x.index_select(1, keep).flatten(1) @ proj).cpu()

        x = run(x, on_step)
        massive_ok.append((x.abs().amax(dim=(2, 3)).argmax(dim=1) == massive).cpu())
        if head is not None:
            preds.append(head(x).argmax(dim=1).cpu())
        states.append(traj)
        speeds.append(speed)
        print(f"  integrated {start + len(idx)}/{len(indices)}", flush=True)
    massive_ok = torch.cat(massive_ok)
    print(f"massive channel {massive} is the top |x| channel in "
          f"{int(massive_ok.sum())}/{len(massive_ok)} images")
    return {
        "model": model_key,
        "states": torch.cat(states),
        "speeds": torch.cat(speeds),
        "labels": torch.as_tensor(labels[indices]),
        "preds": torch.cat(preds) if preds else None,
        "indices": torch.as_tensor(indices),
        "t": torch.as_tensor(stored) * step,
        "t_speed": torch.arange(D + 1) * step,
        "block_bounds": bounds,
        "massive_channel": massive,
    }


def pca(fit: np.ndarray, k: int = 2):
    """PCA via the P×P covariance, so no (n_samples × P) SVD factor is ever allocated."""
    mu = fit.mean(axis=0, dtype=np.float64)
    cov = np.zeros((fit.shape[1], fit.shape[1]))
    for chunk in np.array_split(np.arange(len(fit)), max(1, len(fit) // 4096)):
        c = fit[chunk] - mu
        cov += c.T @ c
    ev, vecs = np.linalg.eigh(cov)
    ev, vecs = ev[::-1].clip(min=0), vecs[:, ::-1]
    return mu, vecs[:, :k].T, ev / ev.sum()


def _draw(ax, Z, labels, colors, *, title: str) -> None:
    n, T, _ = Z.shape
    alpha = np.linspace(0.15, 0.9, T - 1)
    for i in range(n):
        seg = np.stack([Z[i, :-1], Z[i, 1:]], axis=1)
        rgba = np.tile(np.asarray(plt.matplotlib.colors.to_rgba(colors[labels[i]])), (T - 1, 1))
        rgba[:, 3] = alpha
        ax.add_collection(LineCollection(seg, colors=rgba, linewidths=0.9, capstyle="round"))
    ax.scatter(Z[:, 0, 0], Z[:, 0, 1], s=6, facecolors="none",
               edgecolors=[colors[c] for c in labels], linewidths=0.5, alpha=0.6, zorder=3)
    ax.scatter(Z[:, -1, 0], Z[:, -1, 1], s=12, c=[colors[c] for c in labels],
               edgecolors="k", linewidths=0.3, zorder=4)
    ax.autoscale()
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot(data: dict, path: Path) -> None:
    X = data.pop("states").numpy()
    labels = data["labels"].numpy()
    n, T, P = X.shape
    cmap = plt.get_cmap("tab10")
    colors = {c: cmap(k) for k, c in enumerate(CLASSES)}

    mu_all, V_all, ev_all = pca(X.reshape(-1, P))
    mu_end, V_end, ev_end = pca(X[:, -1])
    Z_all = X @ V_all.T - mu_all @ V_all.T
    Z_end = X @ V_end.T - mu_end @ V_end.T

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 0.8]})
    _draw(axes[0], Z_all, labels, colors,
          title=f"PCA of all states (PC1–2: {100 * ev_all[:2].sum():.0f}% var.)")
    axes[0].set(xlabel="PC1", ylabel="PC2")
    _draw(axes[1], Z_end, labels, colors,
          title=f"PCA of final states (PC1–2: {100 * ev_end[:2].sum():.0f}% var.)")
    axes[1].set(xlabel="PC1", ylabel="PC2")

    ax = axes[2]
    sp = data["speeds"].numpy()
    ts = data.get("t_speed", data["t"]).numpy()
    for c in CLASSES:
        ax.plot(ts, sp[labels == c].mean(axis=0), color=colors[c], lw=1.0)
    ax.plot(ts, sp.mean(axis=0), color="k", lw=2.0)
    for b in data.get("block_bounds") or []:
        ax.axvline(b, color="0.6", lw=0.5, ls=":", zorder=0)
    ax.set(xlabel=r"$t$", ylabel=r"$\|f(x(t))\|$", title="speed (class means; black: all)")
    ax.title.set_fontsize(9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [Line2D([0], [0], color=colors[c], lw=2, label=name) for c, name in CLASSES.items()]
    handles += [
        Line2D([0], [0], marker="o", ls="", mfc="none", mec="0.3", ms=4, label=r"$x(0)$"),
        Line2D([0], [0], marker="o", ls="", mfc="0.3", mec="k", ms=4, label=r"$x(T)$"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=7.5, frameon=False)
    fig.suptitle(TITLES.get(data.get("model", SHARED_KEY), data.get("model", "")), fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {path}")
    if data.get("preds") is not None:
        acc = (data["preds"] == data["labels"]).float().mean().item()
        print(f"  top-1 on these {n} images: {100 * acc:.1f}%")
    print(f"  explained var. all-states PC1..5: {np.round(100 * ev_all[:5], 1).tolist()}")
    print(f"  explained var. final-states PC1..5: {np.round(100 * ev_end[:5], 1).tolist()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SHARED_KEY, choices=sorted(TITLES))
    ap.add_argument("--depth", type=int, default=100, help="weight-tied: number of Euler steps D")
    ap.add_argument("--repeats", type=int, default=100, help="non-tied: repeats R per block")
    ap.add_argument("--step", type=float, default=None,
                    help="Euler step (default 9/D weight-tied, 1/R otherwise)")
    ap.add_argument("--force", action="store_true", help="re-integrate even if the cache exists")
    args = ap.parse_args()
    shared = args.model == SHARED_KEY
    step = args.step or (9.0 / args.depth if shared else 1.0 / args.repeats)
    sched = f"D{args.depth}" if shared else f"R{args.repeats}"
    tag = f"{args.model}_{sched}_ES{step:g}"
    cache = CACHE_DIR / f"{tag}.pt"
    if cache.is_file() and not args.force:
        data = torch.load(cache, weights_only=False)
        print(f"loaded {cache}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = integrate(args.model, depth=args.depth, repeats=args.repeats, step=step,
                         device=device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, cache)
        print(f"cached -> {cache}")
    default = shared and args.depth == 100 and args.step is None
    plot(data, FIG / ("trajectory_pca.pdf" if default else f"trajectory_pca_{tag}.pdf"))


if __name__ == "__main__":
    main()
