"""Four-panel seed-0 trajectory figure for the archived SafeCEM Push-T pilot.

Replays the committed simulator states (no simulation): the unconstrained planner,
Safe CEM at d = 0 and d = 40 from the main sweep, and the paired d = 60 arena-check
comparison from ``arena_fix_states.npz``. Writes ``<out>.pdf`` and ``<out>.png``.

Usage::

    .venv/bin/python -I experiments/scripts/safecem_pilot_figure.py \
        archive/legacy/docs/safeDial/results results/safecem-trajectories
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated)
HAZARD = "#d03b3b"  # status: critical
INK, INK2, MUTED, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb"


def inside(xy: np.ndarray, box: list[float]) -> int:
    hit = (xy[:, 0] >= box[0]) & (xy[:, 0] <= box[1]) & (xy[:, 1] >= box[2]) & (xy[:, 1] <= box[3])
    return int(hit.sum())


def main(root: Path, out: Path) -> None:
    results = json.loads((root / "results.json").read_text())
    states = np.load(root / "results_states.npz")
    arena_states = np.load(root / "arena_fix_states.npz")
    cfg = results["config"]
    box = cfg["hazard_box"]
    goal_block, start_block = cfg["goal_block"], cfg["start_block"]

    def find(arm: str, **sel):
        for i, ep in enumerate(results["arms"][arm]["episodes"]):
            if all(ep[k] == v for k, v in sel.items()):
                return states[f"{arm}_{i}"]
        raise KeyError(sel)

    pen0 = find("penalty", lam=0.0, seed=0)
    s0 = find("safe", dial=0.0, seed=0)
    s40 = find("safe", dial=40.0, seed=0)
    probe, action = arena_states["probe_based_0"], arena_states["action_space_0"]

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": AXIS,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 4, figsize=(6.7, 2.05))
    fig.subplots_adjust(left=0.045, right=0.995, top=0.80, bottom=0.235, wspace=0.14)
    fig.patch.set_facecolor("white")

    def arena(ax, dial: float, title: str, sub: str) -> None:
        ax.set_facecolor(SURFACE)
        ax.add_patch(Rectangle((0, 0), 512, 512, fill=False, ec=AXIS, lw=0.8, zorder=1))
        if dial > 0:
            ax.add_patch(Rectangle((box[0] - dial, box[2] - dial), box[1] - box[0] + 2 * dial,
                                   box[3] - box[2] + 2 * dial, fc=HAZARD, ec="none",
                                   alpha=0.10, zorder=2))
        ax.add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2],
                               fc=HAZARD, ec="none", alpha=0.30, zorder=3))
        ax.set_xlim(-40, 552)
        ax.set_ylim(552, -40)
        ax.set_aspect("equal")
        ax.set_xticks([0, 256, 512])
        ax.set_yticks([0, 256, 512])
        ax.tick_params(length=2, width=0.6, pad=1.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left", fontweight="bold", pad=11)
        ax.text(0, 1.015, sub, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6.3, color=INK2)

    def paths(ax, s, color, lw=1.4, block=True, label=None, zorder=5) -> None:
        p, b = s[:, :2], s[:, 2:4]
        ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, solid_capstyle="round",
                zorder=zorder, label=label)
        ax.plot(p[0, 0], p[0, 1], "o", ms=4, color=color, mec="white", mew=0.6, zorder=zorder + 1)
        if block:
            ax.plot(b[:, 0], b[:, 1], color=INK, lw=2.2, solid_capstyle="round", zorder=zorder + 2)
            ax.plot(start_block[0], start_block[1], "s", ms=4.5, color=INK, mec="white",
                    mew=0.6, zorder=zorder + 3)

    def goal(ax) -> None:
        ax.plot(goal_block[0], goal_block[1], "s", ms=5, mfc="none", mec=INK2, mew=0.9, zorder=9)

    arena(axes[0], 0, "a  Unconstrained",
          f"λ = 0; {inside(pen0[:, :2], box)} of {len(pen0)} states in hazard")
    paths(axes[0], pen0, BLUE)
    goal(axes[0])
    thin = dict(arrowstyle="-", lw=0.5, shrinkB=2)
    axes[0].annotate("pusher", xy=(pen0[12, 0], pen0[12, 1]), xytext=(150, 60), fontsize=6,
                     color=BLUE, ha="left", arrowprops=dict(color=BLUE, **thin))
    axes[0].annotate("block", xy=(start_block[0], start_block[1]), xytext=(300, 420),
                     fontsize=6, color=INK, ha="center", arrowprops=dict(color=INK, **thin))
    axes[0].annotate("goal", xy=(goal_block[0], goal_block[1]), xytext=(430, 250), fontsize=6,
                     color=INK2, ha="center", arrowprops=dict(color=INK2, **thin))

    arena(axes[1], 0, "b  Safe CEM, d = 0",
          f"{inside(s0[:, :2], box)} of {len(s0)} states in hazard")
    paths(axes[1], s0, BLUE)
    goal(axes[1])

    arena(axes[2], 40, "c  Safe CEM, d = 40",
          f"{inside(s40[:, :2], box)} of {len(s40)} states in hazard")
    paths(axes[2], s40, BLUE)
    goal(axes[2])

    arena(axes[3], 60, "d  Safe CEM, d = 60", "two arena checks, same seed")
    paths(axes[3], action, BLUE, zorder=5,
          label="action-space check (repair): stays inside, moves the block")
    goal(axes[3])
    paths(axes[3], probe, ORANGE, lw=1.8, block=False, zorder=8,
          label="probe-based check (original): leaves the arena, block unmoved")
    exit_i = int(np.argmax((probe[:, 1] < -40) | (probe[:, 0] < -40)))
    escape = (probe[:, 0].min(), probe[:, 1].min())
    axes[3].annotate(f"probe-based:\nexits, reaches\n({escape[0]:.0f}, {escape[1]:.0f})",
                     xy=(probe[max(exit_i - 1, 0), 0], -40), xytext=(300, 55), fontsize=5.8,
                     color=ORANGE, ha="left", va="top",
                     arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.5, shrinkB=1))
    axes[3].text(345, 420, "action-space:\nstays in", fontsize=5.8, color=BLUE,
                 ha="center", va="top")
    handles, labels = axes[3].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2,
               fontsize=5.8, frameon=False, handlelength=1.4, columnspacing=1.6, labelcolor=INK2)
    for ax in axes[1:]:
        ax.set_yticklabels([])

    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
