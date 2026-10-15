"""Validate matched prefixes and quantify the physical switching probes."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def trace(path):
    return [json.loads(line) for line in (path / "steps.jsonl").read_text().splitlines()]


def motion_metrics(rows):
    f = np.asarray([r["intervened"] for r in rows])
    arm = np.asarray([r["executed_normalized_action"][:6] for r in rows])
    proposed = np.asarray([r.get("task_proposed_action", r["proposed_action"])[:6] for r in rows])
    eef = np.asarray([r["before"]["eef_pos"] for r in rows] + [rows[-1]["after"]["eef_pos"]])
    changed = f[1:] != f[:-1]
    jumps = np.linalg.norm(np.diff(arm, axis=0), axis=1)
    policy_jumps = np.linalg.norm(np.diff(proposed, axis=0), axis=1)
    starts = np.flatnonzero(f & ~np.r_[False, f[:-1]])
    ends = np.flatnonzero(f & ~np.r_[f[1:], False])
    tail = rows[50:]
    return {
        "switches": int(changed.sum()),
        "interventions": int(f.sum()),
        "single_step_intervention_bursts": int(((ends - starts + 1) == 1).sum()),
        "first_intervention_step": int(np.flatnonzero(f)[0]) if f.any() else None,
        "median_command_jump": float(np.median(jumps)),
        "median_command_jump_at_switch": float(np.median(jumps[changed]))
        if changed.any()
        else None,
        "median_command_jump_without_switch": float(np.median(jumps[~changed]))
        if (~changed).any()
        else None,
        "median_task_proposal_jump": float(np.median(policy_jumps)),
        "eef_path_length_m": float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum()),
        "median_eef_acceleration_m_s2": float(
            np.median(np.linalg.norm(np.diff(eef, n=2, axis=0), axis=1) / 0.05**2)
        ),
        "post_step50_median_command_jump": float(np.median(jumps[50:]))
        if len(jumps) > 50
        else None,
        "post_step50_block_extent_xyz_m": np.ptp(
            [r["after"]["cube_positions"][1] for r in tail], axis=0
        ).tolist()
        if tail
        else None,
        "max_middle_base_xy_m": max(r["after"]["middle_base_xy_distance_m"] for r in rows),
        "minimum_middle_height_m": min(r["after"]["cube_positions"][1][2] for r in rows),
    }


def prefix_check(reference, candidate):
    a, b = trace(reference), trace(candidate)
    n = min(len(a), len(b))
    first = next(
        (
            i
            for i in range(n)
            if not np.array_equal(
                a[i]["executed_normalized_action"], b[i]["executed_normalized_action"]
            )
        ),
        None,
    )
    count = first if first is not None else n
    aa, bb = np.load(reference / "observations.npz"), np.load(candidate / "observations.npz")
    before_match = all(a[i]["before"] == b[i]["before"] for i in range(min(count + 1, n)))
    images_match = np.array_equal(aa["cameras"][: count + 1], bb["cameras"][: count + 1])
    initial_a, initial_b = (
        np.load(reference / "initial_state.npz"),
        np.load(candidate / "initial_state.npz"),
    )
    initial_error = max(float(np.abs(initial_a[k] - initial_b[k]).max()) for k in initial_a.files)
    assert before_match and images_match and initial_error == 0, (
        candidate,
        first,
        before_match,
        images_match,
        initial_error,
    )
    return {
        "first_executed_action_difference_step": first,
        "identical_preceding_actions": count,
        "camera_inputs_bitwise_equal_through_first_change": images_match,
        "physical_observations_equal_through_first_change": before_match,
        "max_initial_physical_error": initial_error,
        "scope": "Replayed matched observed prefix; no simulator snapshot restoration claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    args = parser.parse_args()
    output = args.runs / "analysis"
    output.mkdir(exist_ok=True)
    reference = args.primary / "seed-001-filtered"
    ref_rows = trace(reference)
    ref_summary = json.loads((reference / "summary.json").read_text())
    results = [
        {"name": "recorded_reference", "episode": ref_summary, "metrics": motion_metrics(ref_rows)}
    ]
    accounting, all_queries, plot_rows = [], Counter(), [("Reference", ref_rows)]
    for job in json.loads((args.runs / "plan.json").read_text())["jobs"]:
        directory = args.runs / job["name"]
        run = json.loads((directory / "summary.json").read_text())
        warmup = {}
        for mode in ("unfiltered", "filtered"):
            name = f"seed-000-{mode}"
            check = prefix_check(args.primary / name, directory / name)
            assert check["first_executed_action_difference_step"] is None
            warmup[mode] = check
        target = directory / "seed-001-filtered"
        rows = trace(target)
        episode = json.loads((target / "summary.json").read_text())
        results.append(
            {
                "name": job["name"],
                "episode": episode,
                "metrics": motion_metrics(rows),
                "warmup_checks": warmup,
                "prefix_check": prefix_check(reference, target),
            }
        )
        plot_rows.append((job["name"], rows))
        queries = sum((Counter(e["query_counts"]) for e in run["episodes"]), Counter())
        all_queries.update(queries)
        accounting.append(
            {
                "name": job["name"],
                "episodes": len(run["episodes"]),
                **{
                    k: run[k]
                    for k in (
                        "environment_steps",
                        "physics_steps",
                        "elapsed_seconds",
                        "training_updates",
                    )
                },
                "queries": dict(queries),
            }
        )
    offline = json.loads((args.offline / "summary.json").read_text())
    result = {
        "status": "complete",
        "results": results,
        "run_accounting": accounting,
        "new_physical_totals": {
            k: sum(row[k] for row in accounting)
            for k in (
                "episodes",
                "environment_steps",
                "physics_steps",
                "elapsed_seconds",
                "training_updates",
            )
        },
        "physical_run_model_queries": dict(all_queries),
        "offline_probe": offline,
        "limits": [
            "Hand-selected, hand-timed single-case probes; no population success claims.",
            "Changing the safety estimator or bypassing it changes the controller; these are diagnostic controls.",
            "Normalized command differences are not physical jerk measurements.",
            "The target policy's learned reason for opening cannot be uniquely inferred from behavior alone.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
    for label, rows in plot_rows:
        t = [r["simulation_time_s"] for r in rows]
        axes[0].plot(t, [r["after"]["middle_base_xy_distance_m"] * 100 for r in rows], label=label)
        axes[1].plot(t, np.cumsum([r["intervened"] for r in rows]), label=label)
        arms = np.asarray([r["executed_normalized_action"][:6] for r in rows])
        axes[2].plot(t[1:], np.cumsum(np.linalg.norm(np.diff(arms, axis=0), axis=1)), label=label)
    axes[0].axhline(7, linestyle="--", color="black", label="One success component")
    axes[0].set_ylabel("Middle/base XY (cm)")
    axes[0].legend(ncol=2, fontsize=9)
    axes[1].set_ylabel("Cumulative interventions")
    axes[2].set_ylabel("Cumulative command change")
    axes[2].set_xlabel("Simulation time (s)")
    fig.savefig(output / "physical_probes.png", dpi=150)
    plt.close(fig)
    for item in results:
        e, m = item["episode"], item["metrics"]
        print(
            item["name"],
            e["outcome"],
            e["steps"],
            "switches",
            m["switches"],
            "command jump",
            round(m["median_command_jump"], 3),
            "first changed step",
            item.get("prefix_check", {}).get("first_executed_action_difference_step"),
        )
    print("New interaction totals:", result["new_physical_totals"])


if __name__ == "__main__":
    main()
