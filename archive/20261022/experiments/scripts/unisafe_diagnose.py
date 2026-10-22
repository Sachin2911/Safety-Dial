"""Audit exploratory continuation runs, matching and complete interaction accounting."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_steps(directory):
    return [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]


def initial_comparison(reference, candidate):
    a, b = np.load(reference / "initial_state.npz"), np.load(candidate / "initial_state.npz")
    aa = np.load(reference / "observations.npz")["cameras"][0].astype(float)
    bb = np.load(candidate / "observations.npz")["cameras"][0].astype(float)
    return {
        "max_physical_state_difference": max(float(np.abs(a[k] - b[k]).max()) for k in a.files),
        "initial_camera_mae_0_255": np.abs(aa - bb).mean(axis=(1, 2, 3)).tolist(),
    }


def prefix_comparison(reference, candidate, length=300):
    a, b = read_steps(reference), read_steps(candidate)
    n = min(length, len(a), len(b))
    ignored = {"inference_seconds", "continuation_seed"}
    semantic = lambda row: {k: v for k, v in row.items() if k not in ignored}
    mismatch = next((i for i in range(n) if semantic(a[i]) != semantic(b[i])), None)
    aa, bb = np.load(reference / "observations.npz"), np.load(candidate / "observations.npz")
    images_equal = np.array_equal(aa["cameras"][: n + 1], bb["cameras"][: n + 1])
    return {
        "compared_steps": n,
        "first_trace_mismatch": mismatch,
        "camera_prefix_bitwise_equal": images_equal,
        "complete_300_step_prefix_matches": n == length and mismatch is None and images_equal,
        "scope": "Recorded prefix comparison, not simulator snapshot/restore validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--followups", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--environment-check", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads((args.followups / "plan.json").read_text())["jobs"]
    diagnostics, accounting = [], []
    all_queries = Counter()
    paths = [("integration_pilot", args.pilot), ("primary", args.baseline)]
    paths += [(job["name"], args.followups / job["name"]) for job in jobs]
    for name, directory in paths:
        summary = json.loads((directory / "summary.json").read_text())
        queries = sum((Counter(e["query_counts"]) for e in summary["episodes"]), Counter())
        all_queries.update(queries)
        accounting.append(
            {
                "name": name,
                "directory": str(directory),
                "episodes": len(summary["episodes"]),
                **{
                    k: summary[k]
                    for k in (
                        "environment_steps",
                        "physics_steps",
                        "elapsed_seconds",
                        "training_updates",
                    )
                },
                "query_counts": dict(queries),
            }
        )
        if name in ("integration_pilot", "primary"):
            continue
        for episode in summary["episodes"]:
            folder = f"seed-{episode['seed']:03d}-{episode['mode']}"
            steps = read_steps(directory / folder)
            nominal = steps[:300]
            tail = nominal[-60:]
            positions = np.asarray([s["after"]["cube_positions"][1] for s in tail])
            diagnostics.append(
                {
                    "job": name,
                    **{
                        k: episode[k]
                        for k in (
                            "seed",
                            "policy_seed",
                            "outcome",
                            "steps",
                            "physical_failure",
                            "boundary_exit",
                            "interventions",
                            "completion_time_s",
                        )
                    },
                    "matching_to_primary": initial_comparison(
                        args.baseline / folder, directory / folder
                    ),
                    "unfinished_at_15s": len(steps) > 300 or episode["outcome"] == "timeout",
                    "nominal_deadline_middle_motion_extent_m": float(
                        np.linalg.norm(np.ptp(positions, axis=0))
                    ),
                    "nominal_deadline_state": nominal[-1]["after"],
                    "interventions_after_15s": sum(s["intervened"] for s in steps[300:]),
                    "continuation_seed_applied": next(
                        (
                            s["continuation_seed"]
                            for s in steps
                            if s.get("continuation_seed") is not None
                        ),
                        None,
                    ),
                }
            )
    prefixes = {}
    for seed in (0, 1):
        folder = f"seed-{seed:03d}-filtered"
        prefixes[str(seed)] = prefix_comparison(
            args.followups / f"horizon-seed-{seed}" / folder,
            args.followups / f"continuation-seed-{seed}" / folder,
        )
    original = args.baseline / "seed-000-unfiltered"
    fresh_filtered = args.followups / "horizon-seed-0/seed-000-filtered"
    original_steps, fresh_steps = read_steps(original), read_steps(fresh_filtered)
    first = next((i for i, step in enumerate(fresh_steps) if step["intervened"]), None)
    prefix = min(first + 1 if first is not None else 300, len(original_steps))
    original_images = np.load(original / "observations.npz")["cameras"]
    fresh_images = np.load(fresh_filtered / "observations.npz")["cameras"]
    fresh_start_check = {
        **initial_comparison(original, fresh_filtered),
        "first_intervention_zero_based_step": first,
        "camera_inputs_equal_through_first_intervention": np.array_equal(
            original_images[:prefix], fresh_images[:prefix]
        ),
        "proposals_equal_through_first_intervention": all(
            original_steps[i]["proposed_action"] == fresh_steps[i]["proposed_action"]
            for i in range(prefix)
        ),
        "physical_observations_equal_through_first_intervention": all(
            original_steps[i]["before"] == fresh_steps[i]["before"] for i in range(prefix)
        ),
        "state_before_first_intervention": fresh_steps[first]["before"]
        if first is not None
        else None,
    }
    engineering = json.loads((args.environment_check / "summary.json").read_text())
    # Preserve this check separately: outcomes were injected, not policy achievements.
    totals = {
        k: sum(row[k] for row in accounting)
        for k in ("environment_steps", "physics_steps", "elapsed_seconds", "training_updates")
    }
    output = {
        "diagnostics": diagnostics,
        "continuation_prefix_checks": prefixes,
        "fresh_start_seed0_comparison": fresh_start_check,
        "run_accounting": accounting,
        "autonomous_totals_including_pilot_and_discarded_followups": totals,
        "model_query_totals": dict(all_queries),
        "environment_acceptance": engineering,
        "failed_initial_integration": {
            "path": "runs/unisafe/pilot-integration-20261015",
            "environment_steps": 0,
            "reason": "Preprocessor key error before the first environment step",
            "timing_and_partial_model_queries": "not instrumented in that failed attempt",
        },
        "limits": [
            "Follow-ups were selected after inspecting the first pilot; no confirmatory rate claims.",
            "Changing the RNG changes future task-posterior and imagined-filter draws together.",
            "Initial physical matching does not imply pixel-identical starts or mid-episode branching.",
            "Failed bounded alternatives do not establish impossibility.",
            "Earlier setup and migration debugging are outside this baseline-stage ledger.",
        ],
    }
    destination = args.followups / "diagnosis.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print("Diagnosis and accounting:", destination)
    print(json.dumps(prefixes, indent=2))
    for row in diagnostics:
        print(
            row["job"],
            row["seed"],
            row["outcome"],
            row["steps"],
            "unfinished@15s",
            row["unfinished_at_15s"],
        )


if __name__ == "__main__":
    main()
