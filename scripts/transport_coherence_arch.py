#!/usr/bin/env python3
"""Transport coherence of stage 3 in Swin-T, ResNet-50 and ResNet-101 (phase 1.3).

Native path (R=1) on the 500 images of ``scripts/undo_geometry_audit.py``. For every
image the Gram matrix of the block increments dx_k = x_{k+1} - x_k gives
straightness, the cosine matrix, vector survival s_k and the erased fraction E.
Rows: pretrained torchvision weights, and Xavier random stage 3 with g=1 and the
matched gain g* of Appendix D (same construction as scripts/random_init_convergence.py).

ResNet blocks exclude layer3[0] (projection shortcut), and x_{k+1} is the block output,
so the field includes the ReLU after the residual sum. For Swin the four corner tokens
(massive norms, App. G) are reported both included and excluded.

Writes outputs/transport_coherence_arch/{summary.csv, gram.pt, report.md}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import undo_geometry_audit as audit  # noqa: E402
from scripts import feature_map_explorer as fme  # noqa: E402
from scripts.random_init_convergence import random_model  # noqa: E402

OUT = ROOT / "outputs" / "transport_coherence_arch"
G_STAR = {"swin": 1.8, "resnet50": 1.15, "resnet101": 0.95}
BATCH = 8


def corner_mask(H: int, W: int, device) -> torch.Tensor:
    m = torch.ones(H, W, dtype=torch.bool, device=device)
    for i in (0, H - 1):
        for j in (0, W - 1):
            m[i, j] = False
    return m


@torch.no_grad()
def grams(kind: str, model, ds, idx, device) -> dict[str, np.ndarray]:
    blocks = [fme._wrap_stage3_block(kind, b) for b in fme.stage3_raw_blocks(kind, model)]
    out = {"all": []}
    if kind == "swin":
        out["no_corners"] = []
    for s in range(0, len(idx), BATCH):
        imgs = torch.stack([ds[int(i)][0] for i in idx[s : s + BATCH]]).to(device)
        x = fme.enter_stage3(kind, model, imgs)
        xs = [x]
        for b in blocks:
            x = b(x)
            xs.append(x)
        X = torch.stack(xs, 1).double()  # B, n+1, C, H, W
        dx = X[:, 1:] - X[:, :-1]
        variants = {"all": dx.flatten(2)}
        if kind == "swin":
            m = corner_mask(*dx.shape[-2:], device)
            variants["no_corners"] = dx[..., m].flatten(2)
        for k, v in variants.items():
            out[k].append((v @ v.transpose(1, 2)).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def summarize(G: np.ndarray) -> dict:
    n = G.shape[1]
    d = np.sqrt(np.einsum("ikk->ik", G))
    cos = G / (d[:, :, None] * d[:, None, :])
    off = ~np.eye(n, dtype=bool)
    st = np.sqrt(G.sum((1, 2))) / d.sum(1)
    S = np.stack([[G[i, k, k + 1 :].sum() / G[i, k, k] for k in range(n - 1)] for i in range(len(G))])
    e = np.clip(-S, 0, 1)
    E = (e * d[:, : n - 1]).sum(1) / d[:, : n - 1].sum(1)
    consec = np.array([np.mean([c[k, k + 1] for k in range(n - 1)]) for c in cos])
    skip2 = np.array([np.mean([c[k, k + 2] for k in range(n - 2)]) for c in cos])
    align = G.sum(2) / (d * np.sqrt(G.sum((1, 2)))[:, None])
    return dict(n_blocks=n, straightness=st.mean(), straightness_std=st.std(ddof=1),
                cos_k_k1=consec.mean(), cos_k_k1_std=consec.std(ddof=1),
                cos_k_k2=skip2.mean(), cos_offdiag=np.array([c[off].mean() for c in cos]).mean(),
                s_min=S.min(), s_mean=S.mean(), s_neg_frac=(S < 0).mean(),
                E=E.mean(), align_mean=align.mean(), random_walk_straightness=1 / np.sqrt(n))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds, idx, _ = audit.dataset_and_indices()
    rows, store = [], {}
    for kind in ("swin", "resnet50", "resnet101"):
        inits = [("pretrained", None), ("xavier", 1.0), ("xavier", G_STAR[kind])]
        for init, g in inits:
            model = (fme.load_interpoled_model(kind) if g is None else random_model(kind, g, 0)).to(device).eval()
            res = grams(kind, model, ds, idx, device)
            for variant, G in res.items():
                key = f"{kind}_{init}_{'' if g is None else f'g{g:g}_'}{variant}"
                store[key] = G
                rows.append(dict(model=kind, init=init, gain=g, tokens=variant, **summarize(G)))
                print(rows[-1], flush=True)
            del model
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    torch.save(store, OUT / "gram.pt")
    (OUT / "report.md").write_text("# Stage-3 transport coherence, other architectures (R=1, 500 images)\n\n"
                                   + df.round(4).to_string(index=False) + "\n")
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
