#!/usr/bin/env python3
"""Animate the Safe CEM / Safety Dial episodes on Push-T.

Every frame is the real Push-T renderer replaying a committed simulator state, with the
hazard drawn on top:

  * the solid red box is the TRUE hazard, the one violations are always measured against;
  * the dotted box is that hazard inflated by the dial `d`, which is the keep-out region
    the planner actually tests candidate plans against.

States come from `docs/safeDial/results/` rather than `outputs/`, which is gitignored, so
the animations rebuild from a fresh clone. Output lands in `animations/`.

    uv run python experiments/scripts/make_animations.py
    uv run python experiments/scripts/make_animations.py --only dial_sweep --no-mp4

Provenance, and it matters for every d = 60 panel: the sweep in `results.json` ran with the
probe-based arena constraint, which cannot detect a genuine arena exit. That is why the
d = 60 sweep episode leaves the arena. The paired re-run in `arena_fix.json` carries both
variants; `dial60_arena_escape` animates the pair and `dial_response` takes its d = 60 point
from the fixed variant.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

# pygame needs a video driver even to draw onto an offscreen surface, and the env reads
# STABLEWM_HOME at import time.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("STABLEWM_HOME", str(REPO_ROOT / "data" / "stablewm"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

DOC_RESULTS = REPO_ROOT / "docs" / "safeDial" / "results"
OUT_DIR = REPO_ROOT / "animations"

ARENA = 512.0
C_HAZ = "#dc2626"
C_PUSH = "#1d4ed8"
C_BLOCK = "#334155"
C_OK = "#16a34a"
C_BAD = "#b91c1c"
C_GRID = "0.45"


# ---------------------------------------------------------------------------
# scene rendering
# ---------------------------------------------------------------------------


class SceneRenderer:
    """Replays a recorded simulator state through the real Push-T renderer.

    `_set_state` runs one physics substep, so the pose is pinned afterwards: what is drawn
    is exactly the recorded state, not the state plus 10 ms of drift.
    """

    def __init__(self, start_state, goal_state):
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401  (registers swm/PushT-v1)

        self.env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=int(ARENA))
        self.env.reset(seed=0, options={"state": start_state, "goal_state": goal_state})
        self.u = self.env.unwrapped
        # The green target the env draws comes from the variation space, not from
        # goal_state, so point it at the goal block pose the experiment actually used.
        self.u.goal_pose = np.asarray(
            [goal_state[2], goal_state[3], goal_state[4]], dtype=float
        )
        self._cache: dict[tuple, np.ndarray] = {}

    def frame(self, state):
        key = tuple(np.round(np.asarray(state, dtype=float), 4))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        s = np.asarray(state, dtype=float)
        self.u._set_state(s)
        self.u.agent.position = (float(s[0]), float(s[1]))
        self.u.agent.velocity = (0.0, 0.0)
        self.u.block.position = (float(s[2]), float(s[3]))
        self.u.block.angle = float(s[4])
        img = np.asarray(self.u.render()).copy()
        self._cache[key] = img
        return img


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------


def inside_box(xy, box):
    x0, x1, y0, y1 = box
    return (xy[0] >= x0) and (xy[0] <= x1) and (xy[1] >= y0) and (xy[1] <= y1)


def draw_hazard(ax, box, dial=None, label_dial=True):
    """Solid box = the true hazard. Dotted box = the hazard inflated by the dial."""
    x0, x1, y0, y1 = box
    ax.add_patch(
        Rectangle(
            (x0, y0), x1 - x0, y1 - y0, fc=C_HAZ, alpha=0.30, ec=C_HAZ, lw=1.8, zorder=2
        )
    )
    ax.text(
        (x0 + x1) / 2, (y0 + y1) / 2, "HAZARD", color=C_HAZ, fontsize=8, fontweight="bold",
        ha="center", va="center", zorder=3,
    )
    if dial:
        ax.add_patch(
            Rectangle(
                (x0 - dial, y0 - dial),
                (x1 - x0) + 2 * dial,
                (y1 - y0) + 2 * dial,
                fill=False, ec=C_HAZ, lw=1.4, ls=(0, (3, 2)), zorder=2,
            )
        )
        if label_dial:
            ax.text(
                (x0 + x1) / 2, y0 - dial - 5, f"keep-out: d = {dial:.0f} px",
                color=C_HAZ, fontsize=7.5, ha="center", va="bottom", zorder=4,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2),
            )


def draw_arena(ax):
    ax.add_patch(
        Rectangle((0, 0), ARENA, ARENA, fill=False, ec=C_GRID, lw=1.2, zorder=2)
    )


def episode_panel(ax, renderer, states, t, box, dial, title, subtitle="", limits=None):
    """One panel at step t: rendered scene, hazard, trails, and a live readout."""
    ax.clear()
    s = states
    tt = min(t, len(s) - 1)

    img = renderer.frame(s[tt])
    ax.imshow(img, extent=(0.0, ARENA, ARENA, 0.0), interpolation="bilinear", zorder=0)
    draw_arena(ax)
    draw_hazard(ax, box, dial=dial)

    ax.plot(s[: tt + 1, 0], s[: tt + 1, 1], "-", color=C_PUSH, lw=1.6, alpha=0.95, zorder=4)
    ax.plot(s[: tt + 1, 2], s[: tt + 1, 3], "-", color=C_BLOCK, lw=2.6, alpha=0.75, zorder=4)

    violating = inside_box(s[tt, :2], box)
    ax.scatter(
        s[tt, 0], s[tt, 1], s=170, facecolors="none", zorder=6,
        edgecolors=C_BAD if violating else C_PUSH, linewidths=2.6,
    )
    ax.scatter(s[0, 0], s[0, 1], s=30, marker="o", c=C_PUSH, alpha=0.5, zorder=5)

    viol_so_far = sum(inside_box(s[i, :2], box) for i in range(tt + 1))
    oob = not (0.0 <= s[tt, 0] <= ARENA and 0.0 <= s[tt, 1] <= ARENA)

    if limits is None:
        ax.set_xlim(-26, ARENA + 26)
        ax.set_ylim(ARENA + 26, -26)
    else:
        lo, hi = limits
        ax.set_xlim(lo, hi)
        ax.set_ylim(hi, lo)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, fontsize=10.5, pad=8)
    status = (
        f"step {tt + 1:>2}/{len(s)}   "
        f"hazard steps {viol_so_far}/{tt + 1} ({viol_so_far / (tt + 1):.2f})"
    )
    if oob:
        status += "   OUTSIDE ARENA"
    ax.text(
        0.5, -0.045, status, transform=ax.transAxes, ha="center", va="top",
        fontsize=8.5, color=C_BAD if (violating or oob) else "0.25",
        family="monospace",
    )
    if subtitle:
        ax.text(
            0.5, -0.105, subtitle, transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color="0.35",
        )
    if t >= len(s):
        ax.text(
            0.98, 0.02, "episode ended", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="0.4", style="italic",
        )


def legend_handles():
    return [
        Line2D([], [], color=C_PUSH, lw=2, label="pusher path"),
        Line2D([], [], color=C_BLOCK, lw=2.6, alpha=0.75, label="block path"),
        Line2D([], [], color=C_HAZ, lw=2, label="true hazard (violations measured here)"),
        Line2D([], [], color=C_HAZ, lw=1.4, ls=(0, (3, 2)), label="dial-inflated keep-out"),
    ]


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def fig_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def write_gif(frames, path, fps, colors=128):
    imgs = [
        Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=colors)
        for f in frames
    ]
    imgs[0].save(
        path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps),
        loop=0, optimize=True, disposal=2,
    )
    print(f"wrote {path.relative_to(REPO_ROOT)}  ({path.stat().st_size / 1e6:.1f} MB)")


def write_mp4(frames, path, fps):
    import imageio.v2 as iio

    h, w = frames[0].shape[:2]
    # libx264 needs even dimensions.
    crop = [f[: h - h % 2, : w - w % 2] for f in frames]
    iio.mimwrite(path, crop, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    print(f"wrote {path.relative_to(REPO_ROOT)}  ({path.stat().st_size / 1e6:.1f} MB)")


PREVIEW = {"on": False, "dir": None}


def save(frames, name, fps, want_mp4):
    if PREVIEW["on"]:
        out = PREVIEW["dir"] / f"{name}.png"
        Image.fromarray(frames[-1]).save(out)
        print(f"wrote preview {out}")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_gif(frames, OUT_DIR / f"{name}.gif", fps)
    if want_mp4:
        try:
            write_mp4(frames, OUT_DIR / f"{name}.mp4", fps)
        except Exception as exc:  # ffmpeg missing or unusable
            print(f"  mp4 skipped: {exc}")


# ---------------------------------------------------------------------------
# animations
# ---------------------------------------------------------------------------


def anim_dial_sweep(ctx, fps, want_mp4):
    """Turning the dial up grows the keep-out region and bends the route around it."""
    sweep, states, box = ctx["sweep"], ctx["states"], ctx["box"]
    panels = [
        ("penalty_0", None, r"no hazard term ($\lambda = 0$)", "drives straight through"),
        ("safe_0", 0.0, r"Safe CEM, dial $d = 0$", "avoid the box itself"),
        ("safe_3", 20.0, r"Safe CEM, dial $d = 20$ px", "20 px of clearance"),
        ("safe_6", 40.0, r"Safe CEM, dial $d = 40$ px", "40 px of clearance"),
    ]
    eps = [(states[k], d, t, sub) for k, d, t, sub in panels]
    n = max(len(e[0]) for e in eps)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 10.3), dpi=100)
    axes = axes.ravel()
    fig.subplots_adjust(left=0.02, right=0.98, top=0.89, bottom=0.135, wspace=0.02, hspace=0.36)
    frames = []
    for t in range(n + int(0.8 * fps)):
        for ax, (s, d, title, sub) in zip(axes, eps):
            episode_panel(ax, ctx["renderer"], s, t, box, d, title, subtitle=sub)
        fig.legend(
            handles=legend_handles(), loc="lower center", ncol=2, fontsize=8.5,
            frameon=False, bbox_to_anchor=(0.5, 0.004),
        )
        fig.suptitle(
            "The Safety Dial on Push-T: one trained model, conservatism set at deployment\n"
            "Safe CEM never enters the hazard at any dial; the unconstrained planner does.",
            fontsize=11.5, y=0.985,
        )
        frames.append(fig_to_rgb(fig))
        fig.legends.clear()
    plt.close(fig)
    save(frames, "dial_sweep", fps, want_mp4)


def anim_penalty_vs_safe(ctx, fps, want_mp4):
    """A weight trades safety off against the goal; a threshold refuses the trade."""
    sweep, states, box = ctx["sweep"], ctx["states"], ctx["box"]
    ep = {(e["lam"], e["seed"]): e for e in sweep["arms"]["penalty"]["episodes"]}
    safe_ep = {(e["dial"], e["seed"]): e for e in sweep["arms"]["safe"]["episodes"]}

    panels = [
        ("penalty_1", None, r"penalty CEM, $\lambda = 0$",
         f"violates {ep[(0.0, 1)]['frac_violating']:.2f} of steps"),
        ("penalty_7", None, r"penalty CEM, $\lambda = 1$",
         f"still violates {ep[(1.0, 1)]['frac_violating']:.2f} of steps"),
        ("safe_7", 40.0, r"Safe CEM, dial $d = 40$ px",
         f"violates {safe_ep[(40.0, 1)]['frac_violating']:.2f}"),
    ]
    eps = [(states[k], d, t, sub) for k, d, t, sub in panels]
    n = max(len(e[0]) for e in eps)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 6.2), dpi=100)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.855, bottom=0.20, wspace=0.03)
    frames = []
    for t in range(n + int(0.8 * fps)):
        for ax, (s, d, title, sub) in zip(axes, eps):
            episode_panel(ax, ctx["renderer"], s, t, box, d, title, subtitle=sub)
        fig.legend(
            handles=legend_handles(), loc="lower center", ncol=4, fontsize=8.5,
            frameon=False, bbox_to_anchor=(0.5, 0.006),
        )
        fig.suptitle(
            "A weight is not a threshold. Same seed, same planner, same hazard.\n"
            "Penalty CEM buys goal progress with a little violation; constraint-priority "
            "ranking will not make that exchange.",
            fontsize=11.5, y=0.985,
        )
        frames.append(fig_to_rgb(fig))
        fig.legends.clear()
    plt.close(fig)
    save(frames, "penalty_vs_safe_cem", fps, want_mp4)


def anim_dial60_escape(ctx, fps, want_mp4):
    """A probe cannot police the validity domain of the probe."""
    fix = ctx["fix"]
    fix_states = ctx["fix_states"]
    box = ctx["box"]
    dial = float(fix["config"]["dial"])

    probe = fix_states["probe_based_0"]
    action = fix_states["action_space_0"]
    pe = fix["variants"]["probe_based"]["episodes"][0]
    ae = fix["variants"]["action_space"]["episodes"][0]

    lo = min(-40.0, probe[:, 0].min(), probe[:, 1].min()) - 30
    hi = max(ARENA + 40.0, probe[:, 0].max(), probe[:, 1].max()) + 30
    wide = (lo, hi)

    eps = [
        (probe, "probe-based arena constraint (broken)",
         f"block moved {pe['block_moved_px']:.0f} px, "
         f"{pe['oob_frac']:.0%} of steps outside the arena "
         "-- zero violations, but only by leaving", wide),
        (action, "action-space arena constraint (fixed)",
         f"block moved {ae['block_moved_px']:.0f} px, "
         f"{ae['oob_frac']:.0%} of steps outside the arena", wide),
    ]
    n = max(len(e[0]) for e in eps)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 7.0), dpi=100)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.19, wspace=0.05)
    frames = []
    for t in range(n + int(0.8 * fps)):
        for ax, (s, title, sub, lim) in zip(axes, eps):
            episode_panel(
                ax, ctx["renderer"], s, t, box, dial, title, subtitle=sub, limits=lim
            )
            ax.text(
                0.02, 0.02, "axes widened far beyond the 512 px arena",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
                color="0.45", style="italic",
            )
        fig.legend(
            handles=legend_handles(), loc="lower center", ncol=4, fontsize=8.5,
            frameon=False, bbox_to_anchor=(0.5, 0.006),
        )
        fig.suptitle(
            r"Dial $d = 60$ px, same seed: tightening the dial pushes the planner into the "
            "probe's blind spot" "\n"
            "Left, the constraint is read off the probe: the pusher escapes the arena, where "
            "no probe output is valid, and\n"
            "scores zero violations vacuously. Right, the same constraint computed in action "
            "space closes the exploit.",
            fontsize=11, y=0.988,
        )
        frames.append(fig_to_rgb(fig))
        fig.legends.clear()
    plt.close(fig)
    save(frames, "dial60_arena_escape", fps, want_mp4)


def anim_dial_response(ctx, fps, want_mp4):
    """What the operator is actually turning, and what it costs."""
    sweep, states, box = ctx["sweep"], ctx["states"], ctx["box"]
    fix, fix_states = ctx["fix"], ctx["fix_states"]

    safe = {r["dial"]: r for r in sweep["arms"]["safe"]["summary"]}
    pen = sweep["arms"]["penalty"]["summary"]
    best_pen = min(r["viol_mean"] for r in pen if r["lam"] > 0)

    fixed = fix["variants"]["action_space"]["episodes"]

    def _ms(key):
        xs = [e[key] for e in fixed]
        return float(np.mean(xs)), float(np.std(xs))

    d60_block, d60_block_sd = _ms("final_block_err_px")
    d60_feas, d60_feas_sd = _ms("final_frac_feasible")
    d60_viol, _ = _ms("frac_violating")

    dials = [0.0, 20.0, 40.0, 60.0]
    routes = {
        0.0: states["safe_0"],
        20.0: states["safe_3"],
        40.0: states["safe_6"],
        60.0: fix_states["action_space_0"],
    }
    viol = [safe[d]["viol_mean"] for d in dials[:3]] + [d60_viol]
    viol_sd = [safe[d]["viol_std"] for d in dials[:3]] + [0.0]
    block = [safe[d]["block_mean"] for d in dials[:3]] + [d60_block]
    block_sd = [safe[d]["block_std"] for d in dials[:3]] + [d60_block_sd]
    # `feasible` in the sweep summary is a mean over seeds with no spread recorded; the
    # re-run keeps per-episode values, so only the d = 60 point gets a bar.
    feas = [safe[d]["feasible"] for d in dials[:3]] + [d60_feas]
    feas_sd = [0.0, 0.0, 0.0, d60_feas_sd]

    # Sweep d continuously, holding at each measured setting.
    hold, tween = int(1.4 * fps), int(0.7 * fps)
    schedule = [0.0] * hold
    for a, b in zip(dials, dials[1:]):
        schedule += list(np.linspace(a, b, tween, endpoint=False))
        schedule += [b] * hold

    start_img = ctx["renderer"].frame(routes[0.0][0])

    fig = plt.figure(figsize=(13.0, 6.6), dpi=100)
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1.25, 1.0], hspace=0.62, wspace=0.20,
        left=0.02, right=0.96, top=0.855, bottom=0.115,
    )
    ax_arena = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[2, 1])

    frames = []
    for d in schedule:
        nearest = min(dials, key=lambda k: abs(k - d))

        ax_arena.clear()
        ax_arena.imshow(
            start_img, extent=(0.0, ARENA, ARENA, 0.0), interpolation="bilinear",
            alpha=0.30, zorder=0,
        )
        draw_arena(ax_arena)
        draw_hazard(ax_arena, box, dial=d if d > 0 else None, label_dial=False)
        for k in dials:
            s = routes[k]
            on = abs(k - nearest) < 1e-9
            ax_arena.plot(
                s[:, 0], s[:, 1], "-", color=C_PUSH if on else "0.6",
                lw=2.4 if on else 1.0, alpha=1.0 if on else 0.30, zorder=5 if on else 4,
                label=f"realised route, d = {k:.0f}" if on else None,
            )
        ax_arena.scatter(routes[nearest][0, 0], routes[nearest][0, 1], s=42, c=C_PUSH, zorder=6)
        ax_arena.set_xlim(-26, ARENA + 26)
        ax_arena.set_ylim(ARENA + 26, -26)
        ax_arena.set_aspect("equal")
        ax_arena.set_xticks([])
        ax_arena.set_yticks([])
        for spine in ax_arena.spines.values():
            spine.set_visible(False)
        ax_arena.set_title(
            f"keep-out region at d = {d:5.1f} px    route shown: d = {nearest:.0f}",
            fontsize=10.5, family="monospace",
        )
        ax_arena.legend(fontsize=8, loc="upper right", frameon=False)

        for ax, ys, sd, ylab, ttl in (
            (ax_v, viol, viol_sd, "violation fraction", "Safety: zero at every dial"),
            (ax_b, block, block_sd, "final block error (px)",
             "Task cost: non-monotone within 3-seed noise"),
            (ax_f, feas, feas_sd, "fraction of plans feasible",
             "The dial does bite on the feasible set"),
        ):
            ax.clear()
            ax.errorbar(
                dials, ys, yerr=sd, fmt="-o", color=C_PUSH, lw=1.8, ms=5,
                capsize=3, ecolor=C_PUSH, elinewidth=1.1,
            )
            ax.axvline(d, color=C_HAZ, lw=1.4, ls=(0, (3, 2)))
            ax.set_ylabel(ylab, fontsize=8.5)
            ax.set_title(ttl, fontsize=9.5, loc="left")
            ax.grid(alpha=0.28)
            ax.tick_params(labelsize=8)
            ax.set_xlim(-4, 64)
        ax_v.axhline(best_pen, color="#d97706", lw=1.2, ls=":")
        ax_v.text(
            2, best_pen, f"best penalty weight ($\\lambda=10$): {best_pen:.3f}",
            fontsize=7.5, color="#d97706", va="bottom", ha="left",
        )
        ax_v.set_ylim(-0.004, max(0.022, best_pen * 2.0))
        ax_b.set_ylim(0, max(b + e for b, e in zip(block, block_sd)) * 1.18)
        ax_f.set_xlabel("dial  d  (px of required clearance)", fontsize=9)

        fig.suptitle(
            "Dial response: the operator sets d at deployment, from one trained model\n"
            "Violations stay at exactly 0.000 across the dial; what d buys is measured on the "
            "task and feasibility axes.",
            fontsize=11.5, y=0.975,
        )
        fig.text(
            0.5, 0.012,
            "d = 0, 20, 40 from the 3-seed sweep; d = 60 from the paired action-space-constraint "
            "re-run, because the sweep's d = 60 left the arena (see dial60_arena_escape).",
            ha="center", fontsize=7.5, color="0.4",
        )
        frames.append(fig_to_rgb(fig))
        fig.texts = [fig.texts[0]] if fig.texts else []
    plt.close(fig)
    save(frames, "dial_response", fps, want_mp4)


ANIMATIONS = {
    "dial_sweep": anim_dial_sweep,
    "penalty_vs_safe": anim_penalty_vs_safe,
    "dial60_escape": anim_dial60_escape,
    "dial_response": anim_dial_response,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(ANIMATIONS), default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument(
        "--preview", type=Path, default=None,
        help="render only the final frame of each animation as a PNG into this directory",
    )
    args = ap.parse_args()

    if args.preview is not None:
        PREVIEW["on"] = True
        PREVIEW["dir"] = args.preview
        args.preview.mkdir(parents=True, exist_ok=True)

    sweep_path = DOC_RESULTS / "results.json"
    states_path = DOC_RESULTS / "results_states.npz"
    fix_path = DOC_RESULTS / "arena_fix.json"
    fix_states_path = DOC_RESULTS / "arena_fix_states.npz"
    for p in (sweep_path, states_path, fix_path, fix_states_path):
        if not p.is_file():
            print(f"missing {p}")
            return 1

    sweep = json.loads(sweep_path.read_text())
    cfg = sweep["config"]
    start_state = np.array(
        [*cfg["start_pusher"], *cfg["start_block"], cfg["block_theta"], 0.0, 0.0]
    )
    goal_state = np.array(
        [*cfg["goal_pusher"], *cfg["goal_block"], cfg["block_theta"], 0.0, 0.0]
    )

    ctx = {
        "sweep": sweep,
        "states": np.load(states_path),
        "fix": json.loads(fix_path.read_text()),
        "fix_states": np.load(fix_states_path),
        "box": cfg["hazard_box"],
        "renderer": SceneRenderer(start_state, goal_state),
    }

    names = args.only or list(ANIMATIONS)
    for name in names:
        print(f"--- {name}")
        ANIMATIONS[name](ctx, args.fps, not args.no_mp4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
