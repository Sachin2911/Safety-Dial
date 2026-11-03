"""Plots from saved pilot evidence only; use the rented host."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/adapter-pilot-20261101/run-v1"


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    panel = json.loads((OUT / "sources/configs/adapter-pilot/panel.json").read_text())
    target = OUT / "graphs"
    target.mkdir(exist_ok=True)
    arms = ["native", "untrained", "cma", "random"]
    labels = ["Native AP", "Untrained", "CMA-ES", "Random search"]
    fig, ax = plt.subplots(figsize=(7.6, 3.8), layout="constrained")
    x = np.arange(4)
    for offset, metric, label, color in [
        (-0.25, "safe", "Safe completions", "#16817a"),
        (0, "complete", "All completions", "#4267b2"),
        (0.25, "violations", "Violations", "#c55245"),
    ]:
        values = [summary["evaluation"][arm][metric] for arm in arms]
        bars = ax.bar(x + offset, values, width=0.24, label=label, color=color)
        ax.bar_label(bars, padding=2)
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylim=(0, 14),
        ylabel="Episodes out of 12",
        title="Recurrent-adapter pilot: four new layouts, three planner seeds",
    )
    ax.legend(ncol=3, loc="upper center", fontsize=8)
    for ext in ["png", "pdf"]:
        fig.savefig(target / f"evaluation.{ext}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), layout="constrained", sharey=True)
    for ax, optimizer in zip(axes, ["cma", "random"]):
        rows = [r for r in summary["candidates"] if r["optimizer"] == optimizer]
        values = np.asarray([r["safe"] for r in rows])
        ax.scatter(np.arange(1, 33), values, color="#4267b2", s=20)
        ax.step(np.arange(1, 33), np.maximum.accumulate(values), color="#16817a", where="post")
        ax.set(
            title=optimizer.upper(),
            xlabel="Candidate (eight episodes each)",
            ylim=(-0.3, 8.5),
            ylabel="Training safe completions / 8",
        )
    for ext in ["png", "pdf"]:
        fig.savefig(target / f"training.{ext}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10), layout="constrained")
    for i, case in enumerate(panel["evaluation"]):
        for j, arm in enumerate(arms):
            ax = axes[i, j]
            for seed, alpha in zip(panel["evaluation_seeds"], [1, 0.7, 0.45]):
                with np.load(
                    OUT / "episodes" / f"eval-{arm}_{case['id']}_{seed}" / "trace.npz"
                ) as t:
                    ax.plot(t["states"][:, 0], t["states"][:, 1], alpha=alpha, lw=1)
            ax.add_patch(plt.Circle(case["hazard_centre"], 18, color="#c55245", alpha=0.2))
            ax.scatter(*case["initial"], marker="o", color="#333333", s=15)
            ax.scatter(*case["goal"], marker="*", color="#16817a", s=60)
            ax.set(xlim=(20, 204), ylim=(204, 20), aspect="equal", xticks=[], yticks=[])
            if i == 0:
                ax.set_title(labels[j])
            if j == 0:
                ax.set_ylabel(case["id"].replace("eval-", ""))
    for ext in ["png", "pdf"]:
        fig.savefig(target / f"trajectories.{ext}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
