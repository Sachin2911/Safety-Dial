"""Audit and summarize the frozen PushT configuration study, without new interactions."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.pusht_tuning import select_configuration, summarize
from helpers.safety_data import write_json
from helpers.safety_model import file_digest, pusht_constraints


def audit_trial(root, config, case):
    folder = root / case["split"] / config["id"] / case["id"]
    result = json.loads((folder / "result.json").read_text())
    replans = json.loads((folder / "replans.json").read_text())
    with np.load(folder / "trace.npz", allow_pickle=False) as trace:
        states, actions, dense = trace["states"], trace["actions"], trace["dense"]
    if len(states) != len(actions) + 1 or len(dense) != 1 + len(actions) * 10:
        raise ValueError("Incomplete executed trace")
    positions = torch.tensor(dense[:, :2], dtype=torch.float64)
    constraints = pusht_constraints(case["case"])
    distances = [constraint.clearance(positions, 15).numpy() for constraint in constraints]
    margin_breach = bool((np.min(np.stack(distances), axis=0) < 6 - 1e-6).any())
    reported = result["margin_breach"] or result["arena_margin_breach"]
    if margin_breach != reported:
        raise ValueError("Dense monitor and saved result disagree")
    goal = np.asarray(result["goal_state"])
    position_error = np.linalg.norm(states[-1, :4] - goal[:4])
    angle_error = abs(states[-1, 4] - goal[4])
    angle_error = min(angle_error, 2 * np.pi - angle_error)
    actual_goal = bool(position_error < 20 and angle_error < np.pi / 9)
    if actual_goal != result["success"]:
        raise ValueError("Original joint task-goal outcome is wrong")
    if not result["reference_predictors_forbidden"]:
        raise ValueError("Reference prediction was not forbidden")
    if result["safety_model"]["reference_controls_execution"]:
        raise ValueError("Reference-controlled evidence cannot enter this study")
    if result["resources"].get("analytic_path_steps", 0):
        raise ValueError("Unexpected analytic prediction work")
    h = config["params"]["horizon"]
    n = 750 // h
    completed = sum(row["executed"] != "none" for row in replans)
    expected_steps = 11250 * len(replans) + h * completed
    expected_candidates = 15 * n * len(replans) + completed
    if result["resources"]["predictor_steps"] != expected_steps:
        raise ValueError("Actual predictor work differs from the frozen search allowance")
    if result["resources"]["model_candidate_rollouts"] != expected_candidates:
        raise ValueError("Candidate-query accounting mismatch")
    for row in replans:
        if row["executed"] != "none" and row["samples"] != n:
            raise ValueError("Candidate allocation differs from horizon budget")
    return result, {
        "case": case["id"],
        "configuration": config["id"],
        "replans": len(replans),
        "search_predictor_steps": 11250 * len(replans),
        "forecast_predictor_steps": h * completed,
        "minimum_full_pusher_clearance_px": float(np.min(distances)),
        "initial_full_pusher_clearance_px": float(np.min(np.stack(distances)[:, 0])),
        "initial_margin_breach": bool(np.min(np.stack(distances)[:, 0]) < 6 - 1e-6),
        "actual_initial_state": states[0].tolist(),
    }


def report(root):
    if (root / "invalidated.json").exists():
        raise ValueError("Invalidated engineering trials cannot support a tuning comparison")
    panel = json.loads((root / "prepared-panel.json").read_text())
    design = panel["design"]
    dev_freeze = json.loads((root / "development-freeze.json").read_text())
    eval_freeze = json.loads((root / "evaluation-freeze.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    if dev_freeze["panel_sha256"] != file_digest(root / "prepared-panel.json"):
        raise ValueError("Development panel changed after freeze")
    if eval_freeze["selection_sha256"] != file_digest(root / "selection.json"):
        raise ValueError("Selection changed after reserved evaluation began")
    for name, digest in panel["sources"].items():
        if file_digest(root / "source" / name) != digest:
            raise ValueError("Frozen source snapshot changed")
    for name, digest in selection["development_trial_result_sha256"].items():
        if file_digest(root / name) != digest:
            raise ValueError("Development result changed after selection")
    cases = panel["cases"]
    dev = [c for c in cases if c["split"] == "development"]
    evaluation = [c for c in cases if c["split"] == "evaluation"]
    if {c["source_episode"] for c in dev} & {c["source_episode"] for c in evaluation}:
        raise ValueError("Source episodes overlap across study partitions")
    all_rows, audits, development = [], [], {}
    initial_by_case = {}
    for config in panel["configurations"]:
        development[config["id"]] = []
        for case in dev:
            row, audit = audit_trial(root, config, case)
            development[config["id"]].append(row)
            all_rows.append(row)
            audits.append(audit)
            if case["id"] in initial_by_case:
                if initial_by_case[case["id"]] != audit["actual_initial_state"]:
                    raise ValueError("Configurations received different initial states")
            initial_by_case[case["id"]] = audit["actual_initial_state"]
    recomputed = select_configuration(panel["configurations"], development, [c["id"] for c in dev])
    if recomputed["selected"] != selection["selected"]:
        raise ValueError("Stored selection does not follow the preregistered rule")
    current, selected = panel["configurations"][0], selection["selected"]
    arms = [current] if selected["id"] == "current" else [current, selected]
    eval_rows = {}
    for config in arms:
        eval_rows[config["id"]] = []
        for case in evaluation:
            row, audit = audit_trial(root, config, case)
            eval_rows[config["id"]].append(row)
            all_rows.append(row)
            audits.append(audit)
            if case["id"] in initial_by_case:
                if initial_by_case[case["id"]] != audit["actual_initial_state"]:
                    raise ValueError("Evaluation arms received different initial states")
            initial_by_case[case["id"]] = audit["actual_initial_state"]
    totals = summarize(all_rows)
    preparation = panel["preparation_resources"]
    resources = {**totals["resources"]}
    for key, value in preparation.items():
        resources[key] = resources.get(key, 0) + value
    if totals["episodes"] > design["budget"]["max_rollouts"]:
        raise ValueError("Episode budget exceeded")
    if resources["environment_actions"] > design["budget"]["max_environment_actions"]:
        raise ValueError("Interaction budget exceeded")
    total_physics = resources["executed_physics_steps"] + resources["reset_physics_steps"]
    if total_physics > design["budget"]["max_total_physics_steps"]:
        raise ValueError("Physics budget exceeded")
    baseline, chosen = eval_rows["current"], eval_rows[selected["id"]]
    pairs = [
        {
            "case": c["id"],
            "current_safe": baseline[i]["safe_completion"],
            "selected_safe": chosen[i]["safe_completion"],
            "current_actions": baseline[i]["actions"],
            "selected_actions": chosen[i]["actions"],
        }
        for i, c in enumerate(evaluation)
    ]
    result = {
        "study": design["study"],
        "audited": True,
        "selected_configuration": selected,
        "current_configuration": current,
        "development": recomputed["development_summaries"],
        "evaluation": {
            "current": summarize(baseline),
            "selected": summarize(chosen),
            "shared_observations": len(arms) == 1,
            "paired": pairs,
        },
        "all_unique_trials": totals,
        "total_resources_including_preparation": resources,
        "total_physics_steps": total_physics,
        "external_compute_usd": None if design.get("rental") else 0,
        "external_compute_budget_usd": design["budget"]["external_compute_usd"],
        "rental": design.get("rental"),
        "audit": {
            "source_partitions_disjoint": True,
            "selection_precedes_reserved_evaluation": True,
            "original_joint_goal_verified": True,
            "dense_physics_safety_verified": True,
            "initial_states_match_across_configurations": True,
            "reference_predictions_forbidden": True,
            "search_predictor_steps_per_replan": 11250,
            "trial_checks": audits,
        },
        "limitations": [
            design["limitations"],
            "Three development and three evaluation cases",
            "One planner seed per case and one random-search optimizer run",
            "No evolutionary, ablation or cross-environment claim",
        ],
    }
    initial_breaches = sorted({row["case"] for row in audits if row["initial_margin_breach"]})
    result["panel_admission"] = {
        "all_observed_starts_clear": not initial_breaches,
        "initially_margin_violating_case_ids": initial_breaches,
        "all_frozen_cases_retained_in_primary_denominators": True,
        "no_replacement_or_extra_trials": True,
    }
    if initial_breaches:
        result["limitations"].append(
            "Native reset settling moved at least one supplied start inside its 6 px margin. "
            "The layout admitted raw input coordinates rather than verifying the settled "
            "start. All frozen cases remain scored; this limits panel validity."
        )
    combined = dict(resources)
    prior = design.get("prior_invalidated_resources", {})
    regression = design.get("regression_test_budget", {})
    prior_local = design.get("prior_all_local_resources_including_invalid_and_tests")
    if prior_local is not None:
        for key, value in prior_local.items():
            combined[key] = combined.get(key, 0) + value
        for key, value in design["remote_regression_test_budget"].items():
            if isinstance(value, int):
                combined[key] = combined.get(key, 0) + value
        result["prior_separate_local_host_resources"] = prior_local
        result["prior_local_completed_trials"] = design["prior_local_completed_trials"]
        result["remote_regression_test_resources"] = design["remote_regression_test_budget"]
    else:
        for key, value in prior.items():
            combined[key] = combined.get(key, 0) + value
        combined["environment_resets"] += regression.get("new_environment_resets", 0)
        combined["reset_physics_steps"] += regression.get("new_reset_physics_steps", 0)
        combined["environment_actions"] += regression.get("new_environment_actions", 0)
    combined_physics = combined["executed_physics_steps"] + combined["reset_physics_steps"]
    if "combined_budget_including_failed_stage" in design:
        limits = design["combined_budget_including_failed_stage"]
        if combined["environment_actions"] > limits["max_environment_actions"]:
            raise ValueError("Combined interaction allocation exceeded")
        if combined_physics > limits["max_physics_steps"]:
            raise ValueError("Combined physics allocation exceeded")
        if combined["environment_resets"] > limits["max_resets"]:
            raise ValueError("Combined reset allocation exceeded")
    result["invalidated_stage_charged_resources"] = prior
    result["native_regression_test_resources"] = regression
    result["combined_resources_including_invalid_stage_and_tests"] = combined
    result["combined_total_physics_steps"] = combined_physics
    if "repair_of" in design:
        invalid = json.loads((ROOT / design["repair_of"]).read_text())
        if invalid["resources_including_preparation"] != prior:
            raise ValueError("Invalid engineering cost record differs from corrected design")
        result["invalidated_engineering_episodes"] = invalid["completed_invalid_trials"]
    result["trial_rows"] = all_rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/pusht-tuning-20261029-v1")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = report(args.root)
    write_json(args.root / "report.json", result)
    fields = [
        "configuration",
        "case",
        "split",
        "safe_completion",
        "violations",
        "unfinished",
        "actions",
        "object_displacement_px",
        "predictor_steps",
        "model_candidate_rollouts",
        "episode_seconds",
    ]
    with (args.root / "valid-trials.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["trial_rows"]:
            writer.writerow(
                {
                    "configuration": row["configuration_id"],
                    "case": row["study_case_id"],
                    "split": row["study_split"],
                    "safe_completion": row["safe_completion"],
                    "violations": row["margin_breach"] or row["arena_margin_breach"],
                    "unfinished": row["unfinished"],
                    "actions": row["actions"],
                    "object_displacement_px": row["object_displacement_px"],
                    "predictor_steps": row["resources"]["predictor_steps"],
                    "model_candidate_rollouts": row["resources"]["model_candidate_rollouts"],
                    "episode_seconds": row["episode_seconds"],
                }
            )
    print(
        json.dumps(
            {
                "selected": result["selected_configuration"],
                "evaluation": result["evaluation"],
                "unique_episodes": result["all_unique_trials"]["episodes"],
                "total_resources": result["total_resources_including_preparation"],
            }
        )
    )


if __name__ == "__main__":
    main()
