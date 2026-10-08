#!/usr/bin/env python3
"""Plot the Safety Dial trade-off from `safe_dial_pusht.py` output.

Left panel is the operating curve the dial is meant to produce: hazard violation on x,
task error on y, one point per dial value, error bars over seeds. Right panel is the
out-of-bounds rate, which is what exposes the escape exploit in the penalty arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "safe_dial"

STYLE = {
    "safe": dict(color="#2563eb", marker="o", label="Safe CEM (dial $d$)"),
    "penalty": dict(color="#d97706", marker="s", label=r"penalty CEM ($\lambda$), arena constrained"),
    "penalty-noarena": dict(color="#b91c1c", marker="^", label=r"penalty CEM ($\lambda$), no arena constraint"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=OUT_DIR / "results.json")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "safe_dial_tradeoff.png")
    args = ap.parse_args()

    payload = json.loads(args.results.read_text())
    cfg = payload["config"]
    arms = payload["arms"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    for name, arm in arms.items():
        st = STYLE.get(name, dict(marker="x", label=name))
        key = arm["key"]
        rows = arm["summary"]
        xs = [r["viol_mean"] for r in rows]
        ys = [r["block_mean"] for r in rows]
        xe = [r["viol_std"] for r in rows]
        ye = [r["block_std"] for r in rows]
        ax.errorbar(xs, ys, xerr=xe, yerr=ye, capsize=3, lw=1.4, ms=7, **st)
        for r in rows:
            ax.annotate(
                f"{key[0]}={r[key]:g}",
                (r["viol_mean"], r["block_mean"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )
    unmoved = (
        (cfg["goal_block"][0] - cfg["start_block"][0]) ** 2
        + (cfg["goal_block"][1] - cfg["start_block"][1]) ** 2
    ) ** 0.5
    ax.axhline(unmoved, ls=":", c="k", lw=1.2)
    ax.annotate(
        f"block never moved ({unmoved:.0f} px)",
        (ax.get_xlim()[1], unmoved),
        ha="right",
        va="bottom",
        fontsize=8,
    )
    ax.set_xlabel("fraction of steps inside the true hazard box (lower is safer)")
    ax.set_ylabel("final block error, px (lower is better)")
    ax.set_title("Safety-performance operating curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    for name, arm in arms.items():
        st = {k: v for k, v in STYLE.get(name, dict(marker="x", label=name)).items()}
        key = arm["key"]
        rows = arm["summary"]
        ax.plot([r[key] for r in rows], [r["oob_mean"] for r in rows], lw=1.4, ms=7, **st)
    ax.set_xlabel(r"dial $d$ (px) or penalty weight $\lambda$")
    ax.set_ylabel("fraction of steps outside the arena")
    ax.set_title("The escape exploit")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Push-T, frozen pusht/lewm checkpoint: a deployment-time dial versus a training-time weight",
        fontsize=11,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
