"""Summarize paired UNISafe physical outcomes and flag unfinished episodes for inspection."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def wilson(k, n):
    z = 1.96
    center = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(center - half), float(center + half)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = json.loads((args.run / "summary.json").read_text())
    output = args.run / "analysis"
    output.mkdir(exist_ok=True)
    results, rows, unfinished, traces = {}, [], [], {}
    for episode in run["episodes"]:
        directory = args.run / f"seed-{episode['seed']:03d}-{episode['mode']}"
        steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
        traces[(episode["seed"], episode["mode"])] = steps
        tail = steps[-60:]
        positions = np.asarray([s["after"]["cube_positions"][1] for s in tail])
        eef = np.asarray([s["after"]["eef_pos"] for s in tail])
        proposed = np.asarray([s["proposed_action"] for s in tail])
        row = {
            k: episode[k]
            for k in (
                "seed",
                "mode",
                "outcome",
                "steps",
                "simulation_time_s",
                "completion_time_s",
                "interventions",
                "physical_failure",
                "boundary_exit",
                "timeout",
            )
        }
        row.update(
            intervention_fraction=episode["interventions"] / episode["steps"],
            tail_intervention_fraction=float(np.mean([s["intervened"] for s in tail])),
            tail_middle_motion_extent_m=float(np.linalg.norm(np.ptp(positions, axis=0))),
            tail_eef_motion_extent_m=float(np.linalg.norm(np.ptp(eef, axis=0))),
            tail_gripper_open_fraction=float(
                np.mean([abs(s["after"]["gripper_pos"][0]) > 0.03 for s in tail])
            ),
            closest_eef_middle_distance_m=min(s["after"]["eef_middle_distance_m"] for s in steps),
            tail_proposal_change_mean=float(
                np.linalg.norm(np.diff(proposed, axis=0), axis=1).mean()
            )
            if len(tail) > 1
            else 0.0,
            final_middle_base_xy_distance_m=episode["final_state"]["middle_base_xy_distance_m"],
            final_eef_middle_distance_m=episode["final_state"]["eef_middle_distance_m"],
            uncertainty_median=float(np.median([s["uncertainty"] for s in steps])),
            next_min_q_median=float(np.median([s["next_min_q"] for s in steps])),
            failure_prediction_max=max(s["failure_prediction"] for s in steps),
            next_eef_error_median_m=float(
                np.median(
                    [
                        s["executed_next_eef_error_m"]
                        for s in steps
                        if "executed_next_eef_error_m" in s
                    ]
                )
            ),
        )
        # Operational flag only, not a causal diagnosis or proof of avoidability.
        row["low_progress_tail"] = len(tail) == 60 and row["tail_middle_motion_extent_m"] < 0.002
        rows.append(row)
        if episode["timeout"]:
            unfinished.append(row)
            fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
            times = [s["simulation_time_s"] for s in steps]
            axes[0].plot(
                times,
                [s["after"]["middle_base_xy_distance_m"] for s in steps],
                label="Middle/base XY",
            )
            axes[0].plot(
                times, [s["after"]["top_base_xy_distance_m"] for s in steps], label="Top/base XY"
            )
            axes[0].axhline(
                0.07, color="green", linestyle="--", label="Middle distance success component"
            )
            axes[0].set_ylabel("Distance (m)")
            axes[0].legend()
            axes[1].plot(times, [s["uncertainty"] for s in steps], label="Uncertainty")
            axes[1].axhline(run["settings"]["uncertainty_threshold"], linestyle="--", color="red")
            axes[1].set_ylabel("Uncertainty")
            axes[2].plot(times, [s["next_min_q"] for s in steps], label="Safety Q")
            axes[2].axhline(run["settings"]["reachability_threshold"], linestyle="--", color="red")
            axes[2].set_ylabel("Next safety Q")
            axes[3].step(times, [s["intervened"] for s in steps], label="Intervention")
            axes[3].plot(times, [s["failure_prediction"] for s in steps], label="Predicted failure")
            axes[3].legend()
            axes[3].set_xlabel("Simulation time (s)")
            fig.suptitle(f"Seed {episode['seed']} / {episode['mode']}: unfinished episode")
            fig.savefig(output / f"seed-{episode['seed']:03d}-{episode['mode']}.png", dpi=120)
            plt.close(fig)
    for mode in run["settings"]["modes"]:
        group = [e for e in run["episodes"] if e["mode"] == mode]
        counts = Counter(e["outcome"] for e in group)
        n = len(group)
        violations = sum(e["physical_failure"] or e["boundary_exit"] for e in group)
        results[mode] = {
            "episodes": n,
            "outcome_counts": dict(counts),
            "safe_success_rate": counts["success"] / n,
            "safe_success_wilson_95": wilson(counts["success"], n),
            "physical_failure_count": sum(e["physical_failure"] for e in group),
            "any_violation_count": violations,
            "any_violation_rate": violations / n,
            "violation_wilson_95": wilson(violations, n),
            "timeout_count": sum(e["timeout"] for e in group),
            "total_steps": sum(e["steps"] for e in group),
            "total_interventions": sum(e["interventions"] for e in group),
            "mean_episode_simulation_time_s": float(
                np.mean([e["simulation_time_s"] for e in group])
            ),
            "success_completion_times_s": [
                e["completion_time_s"] for e in group if e["safe_success"]
            ],
        }
    paired = []
    for seed in run["settings"]["seeds"]:
        group = {e["mode"]: e for e in run["episodes"] if e["seed"] == seed}
        if len(group) == 2:
            filtered = traces[(seed, "filtered")]
            unfiltered = traces[(seed, "unfiltered")]
            first_intervention = next(
                (i for i, step in enumerate(filtered) if step["intervened"]), None
            )
            prefix_length = min(len(filtered), len(unfiltered))
            if first_intervention is not None:
                prefix_length = min(prefix_length, first_intervention)
            prefix_difference = max(
                (
                    float(
                        np.max(
                            np.abs(
                                np.asarray(filtered[i]["proposed_action"])
                                - np.asarray(unfiltered[i]["proposed_action"])
                            )
                        )
                    )
                    for i in range(prefix_length)
                ),
                default=0.0,
            )
            paired.append(
                {
                    "seed": seed,
                    **{mode: e["outcome"] for mode, e in group.items()},
                    "first_filtered_intervention_step": first_intervention,
                    "preintervention_shared_steps": prefix_length,
                    "preintervention_max_proposal_difference": prefix_difference,
                }
            )
    with (output / "episodes.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    analysis = {
        "by_mode": results,
        "paired_outcomes": paired,
        "paired_initial_conditions": run["paired_initial_conditions"],
        "unfinished_episodes": unfinished,
        "counts": {
            k: run[k]
            for k in ("environment_steps", "physics_steps", "elapsed_seconds", "training_updates")
        },
        "query_counts": dict(sum((Counter(e["query_counts"]) for e in run["episodes"]), Counter())),
        "low_progress_definition": "Last 60 control steps (3 s): middle-block bounding-box diagonal < 2 mm; descriptive flag, not cause or avoidability",
        "inference_limit": "Small migrated-stack pilot; no historical reproduction or general safety guarantee",
    }
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    names = ("success", "physical_failure", "boundary_exit", "timeout", "other_termination")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for i, (mode, result) in enumerate(results.items()):
        bottom = 0
        for j, name in enumerate(names):
            count = result["outcome_counts"].get(name, 0)
            ax.bar(i, count, bottom=bottom, color=f"C{j}", label=name if i == 0 else None)
            bottom += count
    ax.set_xticks(range(len(results)), list(results))
    ax.set_ylabel("Episode count")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.savefig(output / "outcomes.png", dpi=150)
    plt.close(fig)
    print(json.dumps(results, indent=2))
    print("Analysis:", output)


if __name__ == "__main__":
    main()
