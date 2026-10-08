#!/usr/bin/env python3
"""Build every figure used by docs/safeDial.

Figures are written into docs/safeDial/latex/figures/, which is tracked by git, so the report
compiles from a fresh clone. outputs/ is gitignored and must not be the only home for a result.

    uv run python experiments/scripts/make_report_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = REPO_ROOT / "outputs" / "safe_dial"
DOC_RESULTS = REPO_ROOT / "docs" / "safeDial" / "results"
FIG_DIR = REPO_ROOT / "docs" / "safeDial" / "latex" / "figures"

ARENA = 512.0
C_SAFE = "#2563eb"
C_PEN = "#d97706"
C_PEN2 = "#b91c1c"
C_HAZ = "#dc2626"


def draw_arena(ax, hazard_box, *, dial=None, title=""):
    ax.add_patch(Rectangle((0, 0), ARENA, ARENA, fill=False, ec="0.35", lw=1.2))
    x0, x1, y0, y1 = hazard_box
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=C_HAZ, alpha=0.22, ec=C_HAZ, lw=1.4))
    if dial:
        ax.add_patch(
            Rectangle(
                (x0 - dial, y0 - dial),
                (x1 - x0) + 2 * dial,
                (y1 - y0) + 2 * dial,
                fill=False,
                ec=C_HAZ,
                lw=1.1,
                ls=":",
            )
        )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.set_xticks([0, 256, 512])
    ax.set_yticks([0, 256, 512])
    ax.tick_params(labelsize=7)


def fig_trajectories(sweep, states, hazard_box, out):
    """Bird's-eye view: what the planner actually did, per arm."""
    panels = [
        ("penalty_0", r"unconstrained ($\lambda=0$)", None),
        ("safe_0", r"Safe CEM, $d=0$", 0.0),
        ("safe_6", r"Safe CEM, $d=40$", 40.0),
        ("safe_9", r"Safe CEM, $d=60$ (breakdown)", 60.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))
    axes = axes.ravel()
    for ax, (key, title, dial) in zip(axes, panels):
        draw_arena(ax, hazard_box, dial=dial, title=title)
        if key not in states.files:
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center")
            continue
        s = states[key]
        ax.plot(s[:, 0], s[:, 1], "-", color=C_SAFE, lw=1.6, label="pusher")
        ax.plot(s[:, 2], s[:, 3], "-", color="0.25", lw=2.4, label="block")
        ax.scatter(*s[0, :2], c=C_SAFE, s=45, zorder=5, marker="o", ec="w")
        ax.scatter(*s[0, 2:4], c="0.25", s=55, zorder=5, marker="s", ec="w")
        ax.scatter(*s[-1, 2:4], c="#16a34a", s=70, zorder=5, marker="*", ec="w")
        lo = min(0, s[:, 0].min(), s[:, 1].min()) - 20
        hi = max(ARENA, s[:, 0].max(), s[:, 1].max()) + 20
        if lo < -50 or hi > ARENA + 50:
            ax.set_xlim(lo, hi)
            ax.set_ylim(hi, lo)
            ax.set_title(title + "\n(axes widened: left the arena)", fontsize=9)
        else:
            ax.set_xlim(-20, ARENA + 20)
            ax.set_ylim(ARENA + 20, -20)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(
        "Push-T trajectories. Red box is the hazard, dotted is the dial-inflated box the planner avoids.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_dial_response(sweep, out):
    """The dial bites on the planner's feasible set even where the safety axis saturates."""
    safe = sweep["arms"]["safe"]["summary"]
    pen = sweep["arms"]["penalty"]["summary"]
    d = [r["dial"] for r in safe]
    lam = [r["lam"] for r in pen]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.errorbar(d, [r["viol_mean"] for r in safe], yerr=[r["viol_std"] for r in safe],
                marker="o", color=C_SAFE, capsize=3, label="Safe CEM")
    ax.errorbar(lam, [r["viol_mean"] for r in pen], yerr=[r["viol_std"] for r in pen],
                marker="s", color=C_PEN, capsize=3, label=r"penalty ($\lambda$)")
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"dial $d$ (px)  or  weight $\lambda$")
    ax.set_ylabel("hazard violation fraction")
    ax.set_title("Safety: Safe CEM reaches exactly zero")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.errorbar(d, [r["block_mean"] for r in safe], yerr=[r["block_std"] for r in safe],
                marker="o", color=C_SAFE, capsize=3)
    ax.axhline(60.0, ls=":", c="k", lw=1)
    ax.annotate("block never moved (60 px)", (d[0], 60.0), fontsize=7, va="bottom")
    ax.set_xlabel(r"dial $d$ (px)")
    ax.set_ylabel("final block error (px)")
    ax.set_title("Task cost of conservatism")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(d, [r["feasible"] for r in safe], marker="o", color=C_SAFE)
    ax.set_xlabel(r"dial $d$ (px)")
    ax.set_ylabel("fraction of candidate plans feasible")
    ax.set_title("The dial tightens the feasible set")
    ax.grid(alpha=0.3)

    fig.suptitle("Dial response, three seeds, error bars are one standard deviation", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_arena_fix(out):
    """Paired comparison at dial 60: probe-based constraint versus action-space constraint."""
    path = DOC_RESULTS / "arena_fix.json"
    if not path.is_file():
        print(f"skip {out.name}: {path} not found (run validate_arena_fix.py --both)")
        return
    payload = json.loads(path.read_text())
    variants = payload["variants"]
    if "probe_based" not in variants:
        print(f"skip {out.name}: need --both to get the paired comparison")
        return

    metrics = [
        ("oob_frac", "fraction of steps\noutside the arena", 1.0),
        ("block_moved_px", "block displacement (px)", None),
        ("final_block_err_px", "final block error (px)", None),
    ]
    labels = ["probe-based\n(broken)", "action-space\n(fixed)"]
    colors = [C_PEN2, C_SAFE]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for ax, (key, ylabel, ymax) in zip(axes, metrics):
        vals, errs = [], []
        for v in ("probe_based", "action_space"):
            xs = [e[key] for e in variants[v]["episodes"]]
            vals.append(np.mean(xs))
            errs.append(np.std(xs))
        ax.bar(labels, vals, yerr=errs, capsize=5, color=colors, alpha=0.85)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if ymax:
            ax.set_ylim(0, ymax * 1.15)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(labelsize=8)
    fig.suptitle(
        r"Dial $d=60$, same three seeds: a probe cannot enforce a constraint outside its own validity domain",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_arena_fix_traj(hazard_box, out):
    """Trajectories for the paired dial-60 comparison."""
    path = DOC_RESULTS / "arena_fix_states.npz"
    if not path.is_file():
        print(f"skip {out.name}: {path} not found")
        return
    st = np.load(path)
    pairs = [("probe_based", "probe-based constraint (broken)"), ("action_space", "action-space constraint (fixed)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    for ax, (prefix, title) in zip(axes, pairs):
        keys = sorted(k for k in st.files if k.startswith(prefix))
        if not keys:
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center")
            continue
        draw_arena(ax, hazard_box, dial=60.0, title=title)
        lo, hi = 0.0, ARENA
        for k in keys:
            s = st[k]
            ax.plot(s[:, 0], s[:, 1], "-", lw=1.4, alpha=0.9, label=f"pusher {k.split('_')[-1]}")
            ax.plot(s[:, 2], s[:, 3], "-", lw=2.4, color="0.25", alpha=0.8)
            lo = min(lo, s[:, 0].min(), s[:, 1].min())
            hi = max(hi, s[:, 0].max(), s[:, 1].max())
        pad = 0.05 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(hi + pad, lo - pad)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        r"Dial $d=60$: with the probe-based constraint the pusher leaves the arena entirely and the block never moves",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=SWEEP_DIR / "results.json")
    ap.add_argument("--states", type=Path, default=SWEEP_DIR / "results_states.npz")
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOC_RESULTS.mkdir(parents=True, exist_ok=True)

    sweep_path = args.sweep if args.sweep.is_file() else DOC_RESULTS / "results.json"
    states_path = args.states if args.states.is_file() else DOC_RESULTS / "results_states.npz"
    if not sweep_path.is_file():
        print(f"no sweep results at {args.sweep} or {sweep_path}")
        return 1

    sweep = json.loads(sweep_path.read_text())
    hazard_box = sweep["config"]["hazard_box"]

    fig_dial_response(sweep, FIG_DIR / "dial_response.png")
    if states_path.is_file():
        fig_trajectories(sweep, np.load(states_path), hazard_box, FIG_DIR / "trajectories.png")
    fig_arena_fix(FIG_DIR / "arena_fix.png")
    fig_arena_fix_traj(hazard_box, FIG_DIR / "arena_fix_traj.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
