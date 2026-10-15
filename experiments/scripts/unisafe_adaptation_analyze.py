"""Audit and summarize declared adaptation rollouts without adding interactions."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - radius, center + radius]


def geometry(state):
    c = np.asarray(state["cube_positions"])

    def stacked(i, j):
        delta = c[i] - c[j]
        height = abs(delta[2])
        return np.linalg.norm(delta[:2]) < .05 and abs(height - .03) < .01 and height > .014

    failure = c[1, 2] < .13 or c[2, 2] < .13 or np.linalg.norm(c[2, :2] - c[0, :2]) > .10
    success = (
        stacked(2, 0) and not stacked(1, 0) and not stacked(2, 1)
        and c[1, 2] >= .13 and np.linalg.norm(c[1, :2] - c[0, :2]) > .07
    )
    assert bool(failure) == state["physical_failure"]
    assert bool(success) == state["physical_success"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="delay-0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    groups, paired, paths = defaultdict(list), defaultdict(dict), defaultdict(dict)
    total_steps, physics_steps, queries, times, sources = 0, 0, Counter(), [], {}
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text())
        assert summary["status"] == "complete"
        assert summary["script_sha256"] == digest(
            run / "implementation/experiments/scripts/unisafe_adaptation_worker.py"
        )
        for key, expected in (("uncertainty_threshold", 4.3), ("reachability_threshold", .1),
                              ("backup_postprocess", "released"), ("filter_step", 200000)):
            assert summary["settings"][key] == expected
        assert summary["environment_steps"] == sum(e["steps"] for e in summary["episodes"])
        assert summary["physics_steps"] == summary["environment_steps"] * 5
        total_steps += summary["environment_steps"]
        physics_steps += summary["physics_steps"]
        queries.update(summary["query_counts"])
        times.append(summary["elapsed_seconds"])
        sources[str(run / "summary.json")] = digest(run / "summary.json")
        for e in summary["episodes"]:
            path = run / f"episode-{e['index']:04d}"
            rows = [json.loads(line) for line in (path / "steps.jsonl").read_text().splitlines()]
            assert len(rows) == e["steps"]
            assert e["final_state"] == rows[-1]["after"]
            assert e["interventions"] == sum(r["intervened"] for r in rows)
            assert e["physical_failure"] == any(r["after"]["physical_failure"] for r in rows)
            for i, row in enumerate(rows):
                assert row["step"] == i
                if i:
                    assert row["before"] == rows[i - 1]["after"]
                geometry(row["before"])
                geometry(row["after"])
                task = np.array(row["raw_task_action"])
                task[-1] *= 2
                np.testing.assert_allclose(task, row["task_proposed_action"], atol=1e-7, rtol=0)
                if row["adapter_gripper_override"]:
                    task[-1] = -1
                np.testing.assert_allclose(task, row["proposed_action"], atol=1e-7, rtol=0)
                assert row["backup_candidate_action"][-1] == row["proposed_action"][-1]
                assert row["value_trigger"] == (min(row["next_q1"], row["next_q2"]) < .1)
                assert row["uncertainty_trigger"] == (row["uncertainty"] > 4.3)
                assert row["would_intervene"] == (row["value_trigger"] or row["uncertainty_trigger"])
                assert row["intervened"] == (e["mode"] == "filtered" and row["would_intervene"])
                selected = row["backup_candidate_action"] if row["intervened"] else row["proposed_action"]
                assert selected == row["filtered_action"]
                physical = np.clip(np.asarray(selected) * .15, -.15, .15)
                np.testing.assert_allclose(physical, row["environment_action"], atol=2e-8, rtol=0)
                np.testing.assert_allclose(physical / .15, row["executed_normalized_action"], atol=2e-7, rtol=0)
            name = e["job"]["name"]
            assert e["seed"] not in paired[name], "Use distinct names for repeated training replicas"
            paired[name][e["seed"]] = e
            paths[name][e["seed"]] = path
            groups[name].append(e)
    aggregate = {}
    for name, episodes in groups.items():
        n = len(episodes)
        counts = Counter(e["outcome"] for e in episodes)
        aggregate[name] = {
            "episodes": n, "outcomes": dict(counts),
            "safe_success_rate": counts["success"] / n,
            "safe_success_wilson95": wilson(counts["success"], n),
            "physical_failure_wilson95": wilson(counts["physical_failure"], n),
            "environment_steps": sum(e["steps"] for e in episodes),
            "interventions": sum(e["interventions"] for e in episodes),
            "mean_utility": float(np.mean([e["utility"] for e in episodes])),
            "episodes_detail": [{k: e[k] for k in (
                "seed", "outcome", "steps", "interventions", "adapter_override_steps", "job"
            )} for e in episodes],
        }
    comparisons = {}
    if args.baseline in paired:
        rng = np.random.default_rng(20261015)
        for name in paired:
            if name == args.baseline:
                continue
            seeds = sorted(set(paired[name]) & set(paired[args.baseline]))
            if not seeds:
                continue
            delta, camera_errors, state_errors = [], [], []
            for seed in seeds:
                delta.append(int(paired[name][seed]["safe_success"]) - int(paired[args.baseline][seed]["safe_success"]))
                a, b = paths[name][seed], paths[args.baseline][seed]
                x, y = np.load(a / "initial_state.npz"), np.load(b / "initial_state.npz")
                state_errors.append(max(float(np.abs(x[k] - y[k]).max()) for k in x.files))
                x = np.load(a / "reset_cameras.npz")["cameras"].astype(float)
                y = np.load(b / "reset_cameras.npz")["cameras"].astype(float)
                camera_errors.append(np.abs(x - y).mean(axis=(1, 2, 3)).tolist())
            draws = rng.choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
            comparisons[name] = {
                "paired_initial_condition_seeds": seeds,
                "safe_success_rate_difference": float(np.mean(delta)),
                "paired_seed_bootstrap95": np.quantile(draws, [.025, .975]).tolist(),
                "max_initial_physical_error": max(state_errors),
                "reset_camera_mae_by_seed_front_wrist": camera_errors,
                "scope": "Descriptive seed-cluster bootstrap. Sequential rendered inputs may differ; this is not exact counterfactual branching.",
            }
    result = {
        "status": "complete", "groups": aggregate, "comparisons": comparisons,
        "environment_steps": total_steps, "physics_steps": physics_steps,
        "model_queries": dict(queries), "worker_elapsed_seconds_sum": sum(times),
        "artifact_audit": "passed all episode states/actions, frozen filter settings and source hashes",
        "source_summary_sha256": sources,
        "limits": "Development results are exploratory; confidence intervals do not correct for selection. Renderer history makes independence approximate. Frozen filters do not guarantee unchanged safety rates.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axis = plt.subplots(figsize=(9, 4), layout="constrained")
    labels = list(aggregate)
    base = np.zeros(len(labels))
    for outcome, color in (("success", "#2e8b57"), ("physical_failure", "#c94b4b"),
                            ("boundary_exit", "#e8a83f"), ("timeout", "#8899aa")):
        counts = np.array([aggregate[k]["outcomes"].get(outcome, 0) for k in labels])
        axis.bar(labels, counts, bottom=base, label=outcome, color=color)
        base += counts
    axis.set_ylabel("Episodes")
    axis.legend(loc="upper right")
    fig.savefig(args.output / "outcomes.png", dpi=150)
    plt.close(fig)
    for name, value in aggregate.items():
        print(name, value["outcomes"], "steps", value["environment_steps"])
    print("Verified control steps:", total_steps)


if __name__ == "__main__":
    main()
