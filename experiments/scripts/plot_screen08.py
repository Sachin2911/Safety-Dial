"""Export the predeclared task-8 metrics from complete stages only."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/screen08-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ARMS = ("native", "h4", "slow", "smooth", "gripwait")
LABELS = {"native": "Native", "h4": "h4", "slow": "Target scaling",
          "smooth": "Smoothing", "gripwait": "Grip wait"}


def panel(ax, arms, names, keys, title):
    y = np.arange(len(names))
    height = 0.6 if len(keys) == 1 else 0.3
    for i, (key, label, color) in enumerate(keys):
        counts = np.array([arms[a]["counts"][key] for a in names])
        n = np.array([arms[a]["n"] for a in names])
        rate = counts / n
        bounds = np.array([arms[a]["nominal_wilson_95"][key] for a in names])
        offset = 0 if len(keys) == 1 else (i - 0.5) * 0.34
        ax.barh(y + offset, rate, height=height, color=color, alpha=0.85, label=label,
                xerr=np.array([rate - bounds[:, 0], bounds[:, 1] - rate]),
                error_kw={"ecolor": "0.18", "capsize": 3, "elinewidth": 1})
    for pos, name in zip(y, names, strict=True):
        row = arms[name]
        values = "/".join(str(row["counts"][key]) for key, _, _ in keys)
        annotation = f"{values} of {row['n']}" if len(keys) > 1 else f"{values}/{row['n']}"
        ax.text(1.14, pos, annotation, ha="right", va="center", fontsize=9)
    ax.set_yticks(y, [LABELS[a] for a in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Percentage of episodes")
    ax.set_ylabel("Execution preset")
    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if len(keys) > 1:
        legend = ax.legend(loc="lower right", bbox_to_anchor=(1, 1.2), ncol=2,
                           frameon=False, fontsize=9)
        legend.set_in_layout(False)


def draw(stage, arms, destination):
    names = [a for a in ARMS if arms[a]["n"]]
    if any(arms[a]["n"] != 12 for a in names) or not names:
        raise ValueError("Only complete twelve-episode arms may be plotted")
    if stage == "stage_a" and names != list(ARMS):
        raise ValueError("The screening plot requires all five arms")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panel(axes[0, 0], arms, names, [("primary", "Safe requested completion", "C0")],
          "Safe requested completion")
    panel(axes[0, 1], arms, names,
          [("raw_violation", "Raw", "C1"), ("counted_violation", "Counted", "0.55")],
          "Monitor violations (labels: raw/counted)")
    panel(axes[1, 0], arms, names, [("unfinished", "Unfinished", "C0")],
          "Unfinished episodes")
    panel(axes[1, 1], arms, names, [("engaged", "Contact engagement", "C0")],
          "Contact engagement")
    title = "Stage A: complete mechanism screen" if stage == "stage_a" else (
        "Stage B: independent development confirmation"
    )
    fig.suptitle("Jar execution-knob study | " + title, fontsize=15, weight="bold", y=.985)
    fig.text(.01, .055, "12 episodes per arm: six development tasks, two sampling seeds. "
             "Bars show observed rates; whiskers are descriptive nominal 95% Wilson intervals.",
             fontsize=9)
    fig.text(.01, .032, "Task-cluster dependence is not represented by these whiskers. "
             "Use the separate paired six-task bootstrap for differences; no significance claim.",
             fontsize=9)
    fig.text(.01, .009, "Source: task-8 recordings and frozen scoring, 25 October 2026. "
             "Goal must be held; closure is no later than departure from support; primary requires "
             "no raw violation.", fontsize=8)
    fig.tight_layout(rect=(0, .085, 1, .955), h_pad=2.5, w_pad=2.5)
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"{stage}-rates.{suffix}", dpi=190, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/screen08/analysis/figures"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    stages = data.get("stages", {"stage_a": {"arms": data.get("arms")}})
    done = []
    for stage in ("stage_a", "stage_b"):
        if stage in stages and any(r["n"] for r in stages[stage]["arms"].values()):
            draw(stage, stages[stage]["arms"], args.output)
            done.append(stage)
    print(json.dumps({"plotted_complete_stages": done, "output": str(args.output)}))


if __name__ == "__main__":
    main()
