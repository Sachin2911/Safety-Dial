"""Trajectory and outcome figures for a PushT or Cube stage from saved traces. No simulation."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import manip as M  # noqa: E402

COLOURS = dict(replay="#b55a00", scripted="#1b7f3a", clean="#7a7a7a", monitored="#1f5f8b",
               final="#7a7a7a", ap="#1f5f8b")


def draw_pusht(ax, trace, result, exclusion):
    states = trace["states"]
    for angle_state, colour, alpha in ((states[0], "#9aa0a6", 0.35), (trace["goal_state"], "#2e7d32", 0.25)):
        for polygon in M.tee_polygons(angle_state[2:4], angle_state[4]):
            ax.fill(*np.asarray(polygon).T, color=colour, alpha=alpha, linewidth=0)
    centre = trace["hazard_centre"]
    if centre.size:
        ax.add_patch(plt.Circle(centre, exclusion, color="#d32f2f", alpha=0.12, linewidth=0))
        ax.add_patch(plt.Circle(centre, M.TASKS["pusht"]["hazard_radius"], color="#d32f2f", alpha=0.5, linewidth=0))
    pusher = states[:, :2]
    ax.plot(pusher[:, 0], pusher[:, 1], color=COLOURS[result["arm"]], linewidth=1.2, alpha=0.9)
    ax.plot(pusher[0, 0], pusher[0, 1], "o", color=COLOURS[result["arm"]], markersize=3)
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.set_aspect("equal")


def draw_cube(ax, trace, result, exclusion):
    states = trace["states"]
    centre = trace["hazard_centre"]
    if centre.size:
        ax.add_patch(plt.Circle(centre, exclusion, color="#d32f2f", alpha=0.12, linewidth=0))
        ax.add_patch(plt.Circle(centre, M.TASKS["cube"]["hazard_radius"], color="#d32f2f", alpha=0.5, linewidth=0))
    goal = trace["goal_state"]
    ax.add_patch(plt.Circle(goal[:2], 0.04, color="#2e7d32", alpha=0.25, linewidth=0))
    pinch, cube = states[:, :2], states[:, 3:5]
    ax.plot(pinch[:, 0], pinch[:, 1], color=COLOURS[result["arm"]], linewidth=1.2, alpha=0.9)
    ax.plot(cube[:, 0], cube[:, 1], color=COLOURS[result["arm"]], linewidth=1.0, linestyle="--", alpha=0.7)
    ax.plot(cube[0, 0], cube[0, 1], "s", color="#444444", markersize=3)
    ax.set_xlim(0.25, 0.6)
    ax.set_ylim(-0.35, 0.35)
    ax.set_aspect("equal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=sorted(M.TASKS))
    parser.add_argument("stage")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    stage = M.OUT / args.task / args.stage
    summary = json.loads((stage / "summary.json").read_text())
    out = Path(args.out) if args.out else stage
    exclusion = M.exclusion(args.task)
    draw = draw_pusht if args.task == "pusht" else draw_cube
    results = summary["results"]
    families = sorted({r["family"] for r in results if r.get("family")})
    if families:
        for family in families:
            cases = sorted({r["case"] for r in results if r["family"] == family})
            fig, axes = plt.subplots(1, len(cases), figsize=(3.2 * len(cases), 3.4), squeeze=False)
            for ax, case in zip(axes[0], cases):
                for r in results:
                    if r["case"] != case or r["arm"] == "clean":
                        continue
                    trace = np.load(stage / r["id"] / "trace.npz")
                    draw(ax, trace, r, exclusion)
                ax.set_title(case, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"{args.task} {family}: replay (orange), scripted (green), AP monitored (blue)", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / f"{args.task}-{family}-trajectories.png", dpi=130)
            plt.close(fig)
    else:
        cases = sorted({r["case"] for r in results})
        columns = 10
        rows = int(np.ceil(len(cases) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.4 * rows), squeeze=False)
        for index, case in enumerate(cases):
            ax = axes[index // columns][index % columns]
            for r in results:
                if r["case"] == case:
                    trace = np.load(stage / r["id"] / "trace.npz")
                    draw(ax, trace, r, exclusion)
            marks = "".join("A" if r["arm"] == "ap" and r["success"] else "F" if r["arm"] == "final" and r["success"] else ""
                            for r in results if r["case"] == case)
            ax.set_title(f"{case.split('-')[-1]} {marks}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{args.task} no-hazard panel: final-goal (grey), AP (blue); letters mark completions", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{args.task}-long-trajectories.png", dpi=110)
        plt.close(fig)
    print("figures written to", out)


if __name__ == "__main__":
    main()
