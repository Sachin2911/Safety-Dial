"""Recorded-input activation and matching diagnostics; no counterfactual physics."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.maniguard_followup_budget import atomic
from helpers.maniguard_part2_protocol import RUN
from helpers.maniguard_phase3_protocol import sha


def main():
    records = json.loads((RUN / "part2/results.json").read_text())["records"]
    episodes = []
    all_traces = {}
    for record in records:
        output = Path(record["output"])
        trace = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
        all_traces[record["episode_id"]] = trace
        executions = {r["step"]: r for r in trace if r["type"] == "executed"}
        native = json.loads(next((output / "episode").rglob("*_ltl.json")).read_text())
        query_groups = []
        current = None
        substantial = actual_saturation = tiny_changes = 0
        for row in trace:
            if row["type"] == "query":
                if current:
                    query_groups.append(current)
                current = {"step": row["step"], "cumulative": [], "correction_total": 0}
            elif row["type"] == "correction":
                current["cumulative"].append(row["accumulated"])
                current["correction_total"] = row["accumulated"]
            elif row["type"] == "proposal":
                p = np.asarray(row["original"])
                executed = np.asarray(executions[row["step"]]["action"])
                difference = np.max(np.abs(p[:7] - executed[:7]))
                substantial += int(difference > 1e-6)
                tiny_changes += int(0 < difference <= 1e-6)
                cap = row["event"]["step_limit"]
                q = np.asarray(row["frame"]["joints"])
                actual_saturation += int(cap is not None and np.any(np.abs(p[:7] - q) > cap + 1e-6))
        if current:
            query_groups.append(current)
        activation = {}
        for minimum in (4, 6):
            for threshold in (0, .5, 1, 2):
                eligible = [g for g in query_groups if len(g["cumulative"]) >= minimum]
                hits = [g for g in eligible if any(v >= threshold for v in g["cumulative"][minimum - 1:])]
                activation[f"min{minimum}-threshold{threshold}"] = {"prefixes_with_eligible_observed_length": len(eligible), "observed_prefixes_reaching_threshold": len(hits)}
        first = native["violation_step"]
        initial = next(r for r in trace if r["type"] == "initial")
        episodes.append({"episode_id": record["episode_id"], "condition": record["condition"],
                         "scene": record["scene"], "seed": record["seed"],
                         "exact_changed_commands": record["costs"]["changed_steps"],
                         "substantial_arm_corrections_gt_1e_minus6_rad": substantial,
                         "numerical_changes_at_most_1e_minus6_rad": tiny_changes,
                         "bound_saturations_gt_tolerance": actual_saturation,
                         "maximum_accumulated_prefix_correction": max(g["correction_total"] for g in query_groups),
                         "prefixes": len(query_groups), "fixed_log_activation": activation,
                         "early_point5_crossings_actions4_to7": sum(any(v >= .5 for v in g["cumulative"][3:7]) for g in query_groups),
                         "point5_crossings_at_eighth_action": sum(len(g["cumulative"]) >= 8 and g["cumulative"][7] >= .5 for g in query_groups),
                         "first_violation_step": first,
                         "first_violation_AP": native["log"][first]["ap"] if first is not None else None,
                         "source_trace_sha256": sha(output / "trace.jsonl"),
                         "initial_scene_name": initial["scene"]["name"]})
    matching = []
    for block in range(4):
        jobs = []
        for record in records:
            job = json.loads((Path(record["output"]) / "job.json").read_text())
            if job["block"] == block:
                jobs.append((job, record))
        identity = next(r for j, r in jobs if j["condition"] == "A")
        with np.load(Path(identity["output"]) / "initial_observation.npz") as data:
            reference = {key: data[key].copy() for key in data.files}
        chunk_a = next(r["actions"] for r in all_traces[identity["episode_id"]] if r["type"] == "chunk")
        for job, record in jobs:
            if job["condition"] == "A":
                continue
            with np.load(Path(record["output"]) / "initial_observation.npz") as data:
                comparisons = {key: {"exact_equal": bool(np.array_equal(reference[key], data[key])),
                                     "rms": float(np.sqrt(np.mean((reference[key].astype(float) - data[key].astype(float)) ** 2)))}
                               for key in data.files}
            chunk = next(r["actions"] for r in all_traces[record["episode_id"]] if r["type"] == "chunk")
            matching.append({"block": block, "condition": job["condition"],
                             "same_scene_and_base_seed": (job["scene"], job["seed"]) == (identity["scene"], identity["seed"]),
                             "initial": comparisons,
                             "first_policy_chunk_exact_equal": bool(np.array_equal(chunk_a, chunk))})
    result = {"schema": "maniguard-followup-recorded-input-diagnostics-v1", "at": time.time(),
              "scope": "Post hoc Part2 mechanism diagnosis on recorded factual inputs. Domain activation is a fixed-log support check, not a counterfactual execution or outcome prediction.",
              "chronology_deviation": "Before engineering, synthetic CPU checks verified the controller. Recorded physical-input replay first audited the engineering jobs; it was not run before the engineering jobs as the Part2 plan requested. All engineering replay passed before full comparison freeze.",
              "episodes": episodes, "matching_against_A": matching, "script_sha256": sha(__file__)}
    atomic(RUN / "part2/diagnostics.json", result)
    for c in "ABCD":
        selected = [e for e in episodes if e["condition"] == c]
        print(c, {"substantial_corrections": sum(e["substantial_arm_corrections_gt_1e_minus6_rad"] for e in selected),
                  "bound_saturations": sum(e["bound_saturations_gt_tolerance"] for e in selected),
                  "maximum_prefix_correction": max(e["maximum_accumulated_prefix_correction"] for e in selected)})
    print({"initial_state_exact_matches": sum(r["initial"]["state"]["exact_equal"] for r in matching),
           "relative_state_comparisons": len(matching),
           "exact_image_matches": sum(r["initial"][key]["exact_equal"] for r in matching for key in ("overview", "wrist")),
           "image_comparisons": 2 * len(matching),
           "first_chunk_exact_matches": sum(r["first_policy_chunk_exact_equal"] for r in matching)})


if __name__ == "__main__":
    main()
