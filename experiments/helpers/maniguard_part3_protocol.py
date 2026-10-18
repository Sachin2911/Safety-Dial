"""Prospective data partitions and forecast gates, sealed before audit outcomes."""

import json
from dataclasses import asdict
from pathlib import Path

from helpers.maniguard_followup_budget import atomic, clock
from helpers.maniguard_part2_controller import conditions
from helpers.maniguard_part2_protocol import implementation_files as part2_files
from helpers.maniguard_phase2_families import ROOT, validate_scene_access
from helpers.maniguard_phase3_protocol import sha

ENCODER = {"repo_id": "facebook/dinov2-small",
           "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
           "license": "apache-2.0", "input_size": 224, "pool_grid": [2, 2],
           "cameras": ["overview", "wrist"], "features": 3072,
           "source": "https://huggingface.co/facebook/dinov2-small"}
COLLECT = ("task_0046/base", "task_0006/base")
PARTITIONS = {"fit": ["task_0039/base", "task_0010/base"],
              "calibration": ["task_0046/base"], "audit": ["task_0006/base"]}


def recording_plan():
    jobs = []
    for b, (scene, seed) in enumerate(
        (scene, 5000 + 10 * i + r) for i, scene in enumerate(COLLECT) for r in range(2)
    ):
        for condition in ("BC" if b % 2 == 0 else "CB"):
            jobs.append({"episode_id": f"part3-record-b{b}-{condition}", "part": 3,
                         "purpose": "train", "condition": condition, "block": b,
                         "stage": "common_recording", "scene": scene, "seed": seed,
                         "action_horizon": 1100, "controller": asdict(conditions()[condition])})
    return {"schema": "maniguard-part3-recording-plan-v1", "jobs": jobs}


def criteria():
    return {
        "minimum_full_episodes": {"fit": 8, "calibration": 4, "audit": 4},
        "minimum_changing_windows_audit": 100,
        "changing_definition": "four-step latent displacement exceeds fitting-window median",
        "minimum_near_threshold_episodes_calibration": 2,
        "minimum_near_threshold_episodes_audit": 2,
        "near_threshold_margin_deg": 10,
        "minimum_upright_crossing_episodes_audit": 1,
        "minimum_persistence_improvement_fraction_1_and_4": 0.05,
        "minimum_action_shuffled_improvement_fraction_4": 0.02,
        "maximum_relative_ridge_error_4": 1.05,
        "maximum_actual_latent_margin_mae_deg": 10,
        "maximum_predicted_latent_near_margin_mae_deg": 10,
        "maximum_predicted_progress_mae_m": 0.03,
        "maximum_false_reassurance_fraction": 0.10,
        "false_reassurance": "actual margin<0, predicted margin>5 degrees",
        "nominal_acceptance_fraction": 0.5,
        "minimum_actual_acceptance_fraction_audit_each": 0.1,
        "minimum_progress_ranking_spearman": 0.25,
        "minimum_progress_varying_windows": 30,
        "maximum_screening_seconds_per_16_candidates": 30,
        "gate_rule": "all accuracy/usefulness criteria pass; missing support is inconclusive; never relax",
        "inference_limit": "factual prefix prediction/ranking only; no validated counterfactual outcomes",
    }


def implementation_files():
    files = part2_files()
    for directory in ("experiments/helpers", "experiments/scripts"):
        files += list((ROOT / directory).glob("maniguard_part3_*.py"))
    return sorted(set(files))


def freeze_data_decision(path):
    """No model fit or forecast-audit inspection is performed here."""
    if Path(path).exists():
        raise FileExistsError("Data decision already sealed")
    old = ROOT / "runs/maniguard/phase3-part1/20261018-56578764"
    donors = []
    for audit in sorted((old / "attempts").glob("*/audit.json")):
        result = json.loads(audit.read_text())
        if result.get("purpose") != "train" or not result.get("valid"):
            continue
        data = audit.parent / "data"
        donors.append({"episode_id": result["episode_id"], "scene": result["scene"],
                       "audit_sha256": sha(audit), "steps": result["steps"],
                       "exact_initial_observation": (data / "initial_observation.npz").exists(),
                       "exact_later_observations": len(list((data / "observations").glob("*.npz"))),
                       "numeric_margins": False,
                       "used_for_model": False})
    for scene in COLLECT:
        validate_scene_access("clutter", scene, "train")
    value = {"schema": "maniguard-part3-data-decision-v1", "status": "sealed",
             "clock": clock(), "common_recording_required": True,
             "reason": "Part1 has no exact later RGB or numerical margins; Part2 has only two scenes. Two new training tasks permit distinct fit/calibration/audit task groups.",
             "part1_donor_inventory": donors, "part1_comparison_excluded": True,
             "part1_donor_used_for_model": False, "part2_recording_used_once_as_common_data": True,
             "additional_full_episodes": 8, "recording_schedule": recording_plan(),
             "partitions": PARTITIONS, "forecast_criteria": criteria(), "encoder": ENCODER,
             "sampling": "all exact recorded steps; whole-task grouping; no frame interpolation",
             "model_spec": {"pca_dimensions": 32, "history_steps": 1, "members": 5,
                            "hidden_units": [128, 128], "epochs": 30,
                            "optimizer": "Adam, lr .001, batch256; episode bootstrap seed20261021+i",
                            "loss": "MSE standardized next-state residual, no privileged inputs",
                            "readouts": "ridge min upright margin and negative target-goal distance",
                            "ridge_alpha": 1.0, "member_seed": 20261021,
                            "calibration_bins": [0, 0.5, 1, 2, 4, 8, 1000000],
                            "minimum_bin_windows": 30,
                            "nominal_acceptance": 0.5,
                            "pca_or_projection_fitted_on": "fit partition only"}}
    atomic(path, value)
    return value
