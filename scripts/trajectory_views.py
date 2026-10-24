"""Per-coordinate views of a stage-3 trajectory x(t), h(t) = f(x(t)).

Every scalar coordinate x_i (channel × location) is normalised by its own net displacement
Δ_i = x_i(T) - x_i(0):
    progress  p_i(t) = (x_i(t) - x_i(0)) / Δ_i     (0 → 1 for a monotone coordinate)
    velocity  ĥ_i(t) = h_i(t) / Δ_i                 (integrates to 1 over [0, T])

Figures (``save_trajectory_views``):
  feature_progress       raw spaghetti coloured by x_i(0) | progress | normalised velocity
  feature_recruitment    coordinates sorted by peak-velocity time: velocity and progress
  feature_phase_density  (x, h) and (p, ĥ) phase portraits as 2D histograms

Inputs are ``mean_x`` / ``mean_h`` of shape ``(D+1, C, H, W)``; with n>1 images they are
image means, so the views describe the mean trajectory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, TwoSlopeNorm

# Coordinates whose |Δ| is below this quantile are mostly noise around 0 when normalised.
DELTA_QUANTILE = 0.5
# Cap on coordinates entering the 2D histograms (keeps ResNet-sized maps cheap).
MAX_DENSITY_COORDS = 40_000
# Individual coordinates drawn over the quantile bands of the progress / velocity panels.
N_LINES = 40
SEED = 0


def flatten_trajectory(mean_x, mean_h, *, ignored_channels=None, euler_step: float):
    """``(D+1, C, H, W)`` maps → ``t``, ``x``, ``h`` as ``(D+1, n_coords)`` float32 arrays.

    Drops ``ignored_channels`` and channels that are identically zero (``*_ignore`` runs
    store the massive channel zeroed out).
    """
    x = torch.as_tensor(mean_x).float().cpu()
    h = torch.as_tensor(mean_h).float().cpu()
    drop = x.abs().amax(dim=(0, 2, 3)) == 0
    for c in ignored_channels or []:
        drop[int(c)] = True
    keep = (~drop).nonzero().flatten()
    x = x.index_select(1, keep).reshape(x.shape[0], -1).numpy()
    h = h.index_select(1, keep).reshape(h.shape[0], -1).numpy()
    t = np.arange(x.shape[0]) * float(euler_step)
    return t, x, h


def normalise(x: np.ndarray, h: np.ndarray):
    delta = x[-1] - x[0]
    sel = np.abs(delta) >= np.quantile(np.abs(delta), DELTA_QUANTILE)
    p = (x[:, sel] - x[0, sel]) / delta[sel]
    hn = h[:, sel] / delta[sel]
    return sel, p, hn


def t_cross(t: np.ndarray, p: np.ndarray, level: float = 0.5) -> np.ndarray:
    return t[np.argmax(p >= level, axis=0)]


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def _block_lines(ax, block_bounds) -> None:
    for b in block_bounds or []:
        ax.axvline(b, color="0.55", lw=0.5, ls=":", zorder=0)


def _quantile_bands(ax, t, y, color="0.2"):
    q05, q25, q50, q75, q95 = np.nanquantile(y, [0.05, 0.25, 0.5, 0.75, 0.95], axis=1)
    ax.fill_between(t, q05, q95, color=color, alpha=0.12, lw=0, label="P5–P95")
    ax.fill_between(t, q25, q75, color=color, alpha=0.25, lw=0, label="P25–P75")
    ax.plot(t, q50, color=color, lw=1.6, label="median")


def _hist2d(ax, a, b, *, bins, rng):
    H, xe, ye = np.histogram2d(a.ravel(), b.ravel(), bins=bins, range=rng)
    cmap = plt.get_cmap("magma").with_extremes(bad=plt.get_cmap("magma")(0))
    return ax.imshow(
        np.ma.masked_equal(H.T, 0), origin="lower", aspect="auto", cmap=cmap,
        extent=(xe[0], xe[-1], ye[0], ye[-1]), norm=LogNorm(vmin=1, vmax=H.max()), rasterized=True,
    )


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=200 if path.suffix == ".png" else 300)
    if path.suffix == ".pdf":
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_progress(t, x, h, path: Path, *, title: str = "", block_bounds=None) -> None:
    sel, p, hn = normalise(x, h)
    t50 = t_cross(t, p)
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), layout="constrained")

    ax = axes[0]
    idx = rng.choice(x.shape[1], size=min(300, x.shape[1]), replace=False)
    x0 = x[0, idx]
    lim = np.quantile(np.abs(x0), 0.98)
    cmap = plt.get_cmap("coolwarm")
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    for i in idx[np.argsort(np.abs(x0))]:
        ax.plot(t, x[:, i], color=cmap(norm(x[0, i])), lw=0.6, alpha=0.75)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, pad=0.02).set_label(r"$x_i(0)$", fontsize=8)
    ax.set(xlabel=r"$t$", ylabel=r"$x_i(t)$", title=f"raw coordinates ({len(idx)} random)")

    ax = axes[1]
    _quantile_bands(ax, t, p)
    # Fixed draw order over *all* coordinates, so runs of the same backbone share most lines.
    pos_in_sel = np.cumsum(sel) - 1
    perm = np.random.default_rng(SEED + 1).permutation(x.shape[1])
    order = pos_in_sel[perm[sel[perm]][:N_LINES]]
    cmap_t = plt.get_cmap("viridis")
    tnorm = plt.Normalize(np.quantile(t50, 0.05), np.quantile(t50, 0.95))
    for i in order:
        ax.plot(t, p[:, i], color=cmap_t(tnorm(t50[i])), lw=0.7, alpha=0.9)
    sm = plt.cm.ScalarMappable(norm=tnorm, cmap=cmap_t)
    fig.colorbar(sm, ax=ax, pad=0.02).set_label(r"$t_{50}$ (half-way time)", fontsize=8)
    ax.set_ylim(-0.4, 1.4)
    ax.set(xlabel=r"$t$", ylabel=r"$p_i(t)$", title=r"progress $(x_i-x_i(0))/\Delta_i$")
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    ax = axes[2]
    _quantile_bands(ax, t, hn)
    for i in order:
        ax.plot(t, hn[:, i], color=cmap_t(tnorm(t50[i])), lw=0.5, alpha=0.6)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_ylim(np.nanquantile(hn, 0.01), np.nanquantile(hn, 0.995))
    ax.set(xlabel=r"$t$", ylabel=r"$h_i(t)/\Delta_i$", title=r"normalised velocity $h_i/\Delta_i$")

    for ax in axes:
        _style(ax)
        _block_lines(ax, block_bounds)
    fig.suptitle(
        f"{title}{'; ' if title else ''}panels 2–3: {sel.sum():,} coordinates with "
        r"$|\Delta_i|$ above the median",
        fontsize=9,
    )
    _save(fig, path)


def plot_recruitment(t, x, h, path: Path, *, title: str = "", block_bounds=None,
                     n_rows: int = 3000) -> None:
    sel, p, hn = normalise(x, h)
    delta = (x[-1] - x[0])[sel]
    rng = np.random.default_rng(SEED)
    rows = rng.choice(p.shape[1], size=min(n_rows, p.shape[1]), replace=False)
    hs = h[:, sel][:, rows] * np.sign(delta[rows])  # >0 = moving towards the final value
    hs = hs / np.abs(hs).max(axis=0, keepdims=True).clip(min=1e-12)
    t_peak = t[np.abs(hs).argmax(axis=0)]
    order = np.argsort(t_peak)
    pr = p[:, rows][:, order]
    t50 = t_cross(t, p[:, rows])[order]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True, layout="constrained")
    ext = (t[0], t[-1], 0, len(rows))
    im0 = axes[0].imshow(
        hs[:, order].T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
        extent=ext, interpolation="antialiased", rasterized=True,
    )
    axes[0].plot(t_peak[order], np.arange(len(rows)) + 0.5, color="k", lw=0.8, alpha=0.7)
    cb0 = fig.colorbar(im0, ax=axes[0], pad=0.02, fraction=0.05)
    cb0.set_label(r"$\mathrm{sign}(\Delta_i)\,h_i / \max_t|h_i|$", fontsize=8)
    axes[0].set(xlabel=r"$t$", ylabel="coordinate (sorted by peak-velocity time)",
                title="velocity: red = towards final value")

    im1 = axes[1].imshow(
        pr.T, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
        extent=ext, interpolation="antialiased", rasterized=True,
    )
    axes[1].plot(t50, np.arange(len(rows)) + 0.5, ".", color="cyan", ms=0.6, alpha=0.5)
    cb1 = fig.colorbar(im1, ax=axes[1], pad=0.02, fraction=0.05)
    cb1.set_label(r"progress $p_i(t)$", fontsize=8)
    axes[1].set(xlabel=r"$t$", title=r"progress (cyan: $t_{50}$)")
    for ax in axes:
        ax.tick_params(labelsize=8)
        _block_lines(ax, block_bounds)
    fig.suptitle(
        f"{title}{'; ' if title else ''}recruitment order of {len(rows):,} random coordinates "
        r"($|\Delta_i|$ > median)",
        fontsize=9,
    )
    _save(fig, path)


def plot_phase_density(t, x, h, path: Path, *, title: str = "") -> None:
    rng = np.random.default_rng(SEED)
    if x.shape[1] > MAX_DENSITY_COORDS:
        cols = np.sort(rng.choice(x.shape[1], size=MAX_DENSITY_COORDS, replace=False))
        x, h = x[:, cols], h[:, cols]
    sel, p, hn = normalise(x, h)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), layout="constrained")

    ax = axes[0]
    xl = np.quantile(np.abs(x), 0.999)
    hl = np.quantile(np.abs(h), 0.999)
    im = _hist2d(ax, x, h, bins=400, rng=[[-xl, xl], [-hl, hl]])
    fig.colorbar(im, ax=ax, pad=0.02).set_label("count (log)", fontsize=8)
    ax.set(xlabel=r"$x_i$", ylabel=r"$h_i$", title=f"{x.shape[1]:,} coordinates, all $t$")

    ax = axes[1]
    _hist2d(ax, p, hn, bins=400, rng=[[-0.3, 1.3], [-0.15, np.nanquantile(hn, 0.995)]])
    pm, hm = np.median(p, axis=1), np.median(hn, axis=1)
    sc = ax.scatter(pm, hm, c=t, cmap="viridis", s=2, zorder=3)
    fig.colorbar(sc, ax=ax, pad=0.02).set_label(r"$t$ (median trajectory)", fontsize=8)
    ax.set(xlabel=r"progress $p_i$", ylabel=r"$h_i/\Delta_i$",
           title=r"normalised: every coordinate goes $0\to1$")
    for ax in axes:
        ax.tick_params(labelsize=8)
    if title:
        fig.suptitle(title, fontsize=9)
    _save(fig, path)


def summary(t, x, h) -> dict[str, float]:
    """Scalars quoted in the text: straightness, monotone share, t50 / peak-time spread."""
    sel, p, _ = normalise(x, h)
    disp = np.linalg.norm(x[-1] - x[0])
    length = np.linalg.norm(np.diff(x, axis=0), axis=1).sum()
    delta = x[-1] - x[0]
    steps = np.diff(x, axis=0)
    back = np.clip(-steps * np.sign(delta), 0, None).sum(0) / (np.abs(steps).sum(0) + 1e-12)
    w = np.abs(delta)
    t50 = t_cross(t, p)
    t_peak = t[np.abs(h).argmax(axis=0)]
    q = lambda a, k: float(np.quantile(a, k))
    return {
        "straightness": float(disp / max(length, 1e-12)),
        "monotone_share_weighted": float((w * (back < 0.05)).sum() / w.sum()),
        "t50_q25": q(t50, 0.25), "t50_q50": q(t50, 0.5), "t50_q75": q(t50, 0.75),
        "tpeak_q25": q(t_peak, 0.25), "tpeak_q50": q(t_peak, 0.5), "tpeak_q75": q(t_peak, 0.75),
    }


def save_trajectory_views(
    mean_x, mean_h, out_dir: Path, *, euler_step: float, ignored_channels=None,
    title: str = "", block_bounds=None, suffix: str = ".png", write=None,
) -> dict[str, float]:
    """Write the three figures into ``out_dir``. ``write(path, fn)`` defaults to ``fn()``."""
    out_dir = Path(out_dir)
    write = write or (lambda _path, fn: fn())
    t, x, h = flatten_trajectory(mean_x, mean_h, ignored_channels=ignored_channels,
                                 euler_step=euler_step)
    p_prog = out_dir / f"feature_progress{suffix}"
    p_rec = out_dir / f"feature_recruitment{suffix}"
    p_den = out_dir / f"feature_phase_density{suffix}"
    write(p_prog, lambda: plot_progress(t, x, h, p_prog, title=title, block_bounds=block_bounds))
    write(p_rec, lambda: plot_recruitment(t, x, h, p_rec, title=title, block_bounds=block_bounds))
    write(p_den, lambda: plot_phase_density(t, x, h, p_den, title=title))
    return summary(t, x, h)
