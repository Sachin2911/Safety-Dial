"""Frozen paired summaries and crossed-cluster uncertainty, on the rented host."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/replication-20261101/run-v1"
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.pilot_common import save  # noqa: E402


def metrics(rows):
    return {
        "n": len(rows),
        **{
            key: sum(r[key] for r in rows)
            for key in ["safe_completion", "success", "violation", "unfinished", "engaged"]
        },
    }


def main():
    assert json.loads((OUT / "audit.json").read_text())["passed"]
    summary = json.loads((OUT / "summary.json").read_text())
    panel = json.loads((OUT / "sources/configs/adapter-replication/panel.json").read_text())
    config = json.loads((OUT / "sources/configs/adapter-replication/run.json").read_text())
    evaluation = [r for r in summary["results"] if r["stage"] == "evaluation"]
    arms = ["native", "untrained", "cma", "random"]
    repeats = [r["id"] for r in config["repeats"]]
    groups, per_layout = {}, {}
    values = np.zeros((4, 3, 12))
    for ai, arm in enumerate(arms):
        for ri, rid in enumerate(repeats):
            rows = [
                r
                for r in evaluation
                if r["arm"] == arm and r["repeat"] == (None if arm == "native" else rid)
            ]
            groups[f"{rid}-{arm}"] = metrics(rows)
            per_layout[f"{rid}-{arm}"] = {}
            for ci, case in enumerate(panel["evaluation"]):
                subset = [r for r in rows if r["case"] == case["id"]]
                assert len(subset) == 3
                per_layout[f"{rid}-{arm}"][case["id"]] = metrics(subset)
                values[ai, ri, ci] = sum(r["safe_completion"] for r in subset) / 3
    rng = np.random.default_rng(config["bootstrap_seed"])
    draws = []
    for _ in range(config["bootstrap_replicates"]):
        rid = rng.integers(0, 3, 3)
        cid = rng.integers(0, 12, 12)
        sampled = values[:, rid][:, :, cid].mean(axis=(1, 2))
        draws.append([sampled[2] - sampled[0], sampled[2] - sampled[3]])
    draws = np.array(draws)
    comparisons = {}
    for i, (name, ai) in enumerate([("cma_minus_native", 0), ("cma_minus_random", 3)]):
        comparisons[name] = dict(
            mean_difference=float((values[2] - values[ai]).mean()),
            per_repeat=(values[2] - values[ai]).mean(axis=1).tolist(),
            exploratory_95_interval=np.quantile(draws[:, i], [0.025, 0.975]).tolist(),
        )
    result = dict(
        groups=groups,
        per_layout=per_layout,
        comparisons=comparisons,
        native_shared_not_independent=True,
        bootstrap_draws=config["bootstrap_replicates"],
        cluster_units="training repeat and layout; planner seeds retained together",
        resources=summary["resources"],
        seconds=summary["seconds"],
    )
    save(OUT / "analysis.json", result)
    plots = OUT / "graphs"
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    colors = ["#557b9d", "#b1b1b1", "#208e75", "#d49347"]
    for ax, rid in zip(axes, repeats):
        safe = [groups[f"{rid}-{a}"]["safe_completion"] for a in arms]
        ax.bar(arms, safe, color=colors)
        ax.set_ylim(0, 36)
        ax.set_title(f"Repeat {int(rid[1:]) + 1}")
        for i, count in enumerate(safe):
            ax.text(i, count + 0.5, f"{count}/36", ha="center", fontsize=10)
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Safe completions / 36")
    fig.suptitle("Twelve new layouts × three planner seeds; native shared across repeats")
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(plots / ("09-adapter-replication." + suffix), dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, rid in zip(axes, repeats):
        for optimizer, color in [("cma", colors[2]), ("random", colors[3])]:
            candidates = [
                c
                for c in summary["candidates"]
                if c["repeat"] == rid and c["optimizer"] == optimizer
            ]
            spent, best, xx, yy = 0, None, [], []
            for candidate in candidates:
                spent += candidate["actions"]
                if candidate["eligible"]:
                    if best is None or candidate["score"] > best["score"]:
                        best = candidate
                    xx.append(spent)
                    yy.append(best["safe"])
            ax.step(xx, yy, where="post", color=color, label=optimizer)
        ax.set_xlim(0, 40000)
        ax.set_ylim(0, 8)
        ax.set_title(rid)
        ax.set_xlabel("Actual training actions")
        ax.legend()
    axes[0].set_ylabel("Incumbent safe training completions / 8")
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(plots / ("09-training." + suffix), dpi=170)
    plt.close(fig)
    print(json.dumps(dict(groups=groups, comparisons=comparisons)), flush=True)


if __name__ == "__main__":
    main()
