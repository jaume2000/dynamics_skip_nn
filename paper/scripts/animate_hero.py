#!/usr/bin/env python3
"""Animated hero: the stage-3 trajectories of Figure 1 drawn as they are integrated.

Both models advance in sync on the same depth clock t ∈ [0, 9]: every path grows behind a
moving head, recent segments glow, and the mean speed is traced below with a cursor.
Uses the 2-D PCA caches written by ``plot_hero.py``.

Writes figures/hero_trajectories.mp4 and a poster (last frame) hero_trajectories_poster.jpg.

Usage (from repo root):
    python paper/scripts/plot_hero.py      # once, to build the caches
    python paper/scripts/animate_hero.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_hero import HIGHLIGHT, RUNS, T_END, load  # noqa: E402
from plot_trajectory_pca import FIG  # noqa: E402

FPS = 30
SECONDS = 9.0
HOLD = 2.0
GLOW = 0.6
BG = "#0b0b14"
FG = "#d8d8e0"
CMAP = plt.get_cmap("plasma")
NORM = plt.Normalize(0.0, T_END)


class Panel:
    def __init__(self, ax, ax_speed, run: dict, title: str, eq: str):
        self.Z, self.t = run["Z"], run["t"]
        self.ts, self.sp = run["t_speed"], run["speeds"].mean(0)
        labels = run["labels"]
        n, T, _ = self.Z.shape
        seg = np.stack([self.Z[:, :-1], self.Z[:, 1:]], axis=2).reshape(-1, 2, 2)
        self.tmid = np.tile(0.5 * (self.t[:-1] + self.t[1:]), n)
        self.base = CMAP(NORM(self.tmid))
        self.lines = LineCollection(seg, linewidths=0.55, capstyle="round", zorder=1)
        ax.add_collection(self.lines)
        self.hi = [int(np.flatnonzero(labels == c)[0]) for c in HIGHLIGHT]
        hseg = np.stack([self.Z[self.hi, :-1], self.Z[self.hi, 1:]], axis=2).reshape(-1, 2, 2)
        self.htmid = np.tile(0.5 * (self.t[:-1] + self.t[1:]), len(self.hi))
        self.hbase = CMAP(NORM(self.htmid))
        self.hlines = LineCollection(hseg, linewidths=1.8, capstyle="round", zorder=3)
        ax.add_collection(self.hlines)
        self.heads = ax.scatter(self.Z[:, 0, 0], self.Z[:, 0, 1], s=3, c=FG, lw=0, alpha=0.8, zorder=2)
        self.hheads = ax.scatter(self.Z[self.hi, 0, 0], self.Z[self.hi, 0, 1], s=22, c="white",
                                 edgecolors=BG, linewidths=0.6, zorder=4)
        ints = [int(np.abs(self.t - k).argmin()) for k in range(1, int(T_END))]
        self.marks = [(k, self.Z[self.hi][:, i]) for k, i in zip(range(1, int(T_END)), ints)]
        self.mark_art = ax.scatter([], [], s=10, c="white", edgecolors=BG, linewidths=0.5, zorder=4)
        lo, hi = self.Z.reshape(-1, 2).min(0), self.Z.reshape(-1, 2).max(0)
        pad = 0.04 * (hi - lo)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_axis_off()
        ax.set_title(f"{title}\n{eq}", color=FG, fontsize=11, loc="left")

        self.ax_speed = ax_speed
        ax_speed.set_facecolor(BG)
        ax_speed.set_xlim(0, T_END)
        ax_speed.set_ylim(0, self.sp.max() * 1.15)
        for k in range(1, int(T_END)):
            ax_speed.axvline(k, color="#33334a", lw=0.6, ls=":", zorder=0)
        sseg = np.stack([np.c_[self.ts[:-1], self.sp[:-1]], np.c_[self.ts[1:], self.sp[1:]]], axis=1)
        self.stmid = 0.5 * (self.ts[:-1] + self.ts[1:])
        self.sbase = CMAP(NORM(self.stmid))
        self.speed = LineCollection(sseg, linewidths=2.2, capstyle="round")
        ax_speed.add_collection(self.speed)
        self.cursor = ax_speed.axvline(0, color=FG, lw=0.8, alpha=0.6)
        ax_speed.set_xlabel(r"depth $t$", color=FG, fontsize=9)
        ax_speed.set_ylabel(r"speed $\|f(x(t))\|$", color=FG, fontsize=9)
        ax_speed.tick_params(colors=FG, labelsize=8)
        for s in ("top", "right"):
            ax_speed.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax_speed.spines[s].set_color("#55556a")

    def _position(self, now: float) -> np.ndarray:
        j = int(np.clip(np.searchsorted(self.t, now) - 1, 0, len(self.t) - 2))
        w = np.clip((now - self.t[j]) / (self.t[j + 1] - self.t[j]), 0.0, 1.0)
        return (1 - w) * self.Z[:, j] + w * self.Z[:, j + 1]

    @staticmethod
    def _alpha(tmid: np.ndarray, now: float, base: float, glow: float) -> np.ndarray:
        age = now - tmid
        return np.where(age < 0, 0.0, base + glow * np.exp(-np.maximum(age, 0) / GLOW))

    def update(self, now: float) -> None:
        rgba = self.base.copy()
        rgba[:, 3] = np.clip(self._alpha(self.tmid, now, 0.22, 0.55), 0, 1)
        self.lines.set_color(rgba)
        rgba = self.hbase.copy()
        rgba[:, 3] = np.where(self.htmid <= now, 1.0, 0.0)
        self.hlines.set_color(rgba)
        pos = self._position(now)
        self.heads.set_offsets(pos)
        self.hheads.set_offsets(pos[self.hi])
        done = [p for k, p in self.marks if k <= now]
        self.mark_art.set_offsets(np.concatenate(done) if done else np.empty((0, 2)))
        rgba = self.sbase.copy()
        rgba[:, 3] = np.where(self.stmid <= now, 1.0, 0.0)
        self.speed.set_color(rgba)
        self.cursor.set_xdata([now, now])


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), dpi=120,
                             gridspec_kw={"height_ratios": [3.4, 1], "hspace": 0.18, "wspace": 0.08},
                             facecolor=BG)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.08)
    panels = []
    for col, (tag, title, eq) in enumerate(RUNS):
        axes[0, col].set_facecolor(BG)
        panels.append(Panel(axes[0, col], axes[1, col], load(tag), title, eq))
    clock = fig.text(0.5, 0.955, "", color=FG, fontsize=13, ha="center")

    n_frames = int(SECONDS * FPS)
    times = np.r_[np.linspace(0.0, T_END, n_frames), np.full(int(HOLD * FPS), T_END)]
    out = FIG / "hero_trajectories.mp4"
    writer = FFMpegWriter(fps=FPS, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow",
                                      "-movflags", "+faststart"])
    with writer.saving(fig, str(out), dpi=120):
        for now in times:
            for p in panels:
                p.update(float(now))
            clock.set_text(f"t = {now:4.2f}")
            writer.grab_frame(facecolor=BG)
    fig.savefig(FIG / "hero_trajectories_poster.jpg", dpi=120, facecolor=BG,
                pil_kwargs={"quality": 88})
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
