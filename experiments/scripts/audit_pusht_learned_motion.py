"""Verify the saved motion checkpoint against saved executed physics, without rollouts."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.learned_motion import load_motion
from helpers.safety_model import file_digest, pusht_constraints, scaler_digest


def audit(checkpoint, training, results, scalers):
    model, provenance = load_motion(checkpoint)
    spec = model.spec
    parameters = json.loads(Path(scalers).read_text())["columns"]["action"]
    mean, scale = np.asarray(parameters["mean"]), np.asarray(parameters["scale"])
    if scaler_digest(mean, scale) != spec.action_scaler_sha256:
        raise ValueError("Audit action scaler differs from checkpoint")
    with np.load(training, allow_pickle=False) as data:
        partitions = {
            split: set(data["groups"][data["splits"] == split].tolist())
            for split in ("train", "calibration", "test")
        }
        for first, second in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
            if partitions[first] & partitions[second]:
                raise ValueError("Training/calibration/test source groups overlap")
    expected_dataset = provenance["report"]["dataset_sha256"]
    if file_digest(training) != expected_dataset:
        raise ValueError("Training dataset differs from checkpoint provenance")
    rows = []
    query_count = 0
    with torch.inference_mode():
        for kind, folder in (("island", "island-r1"), ("bar", "bar"), ("gate", "gate")):
            root = Path(results) / folder
            result = json.loads((root / "result.json").read_text())
            if not result["reference_predictors_forbidden"]:
                raise ValueError("Reference predictor guard was not active")
            if result["safety_model"]["reference_controls_execution"]:
                raise ValueError("This was reference-controlled execution")
            with np.load(root / "episode/trace.npz", allow_pickle=False) as trace:
                states, actions, dense = trace["states"], trace["actions"], trace["dense"]
            if len(dense) != 1 + len(actions) * spec.substeps:
                raise ValueError("Saved physics trace is incomplete")
            max_error, compared_points = 0.0, 0
            for start in range(0, len(actions), spec.action_block):
                n = min(spec.actions, len(actions) - start)
                normalized = np.zeros((spec.actions, spec.action_dim))
                normalized[:n] = (actions[start : start + n] - mean) / scale
                candidates = torch.tensor(normalized).reshape(
                    1, 1, spec.max_horizon, spec.action_block * spec.action_dim
                )
                observed = torch.tensor(states[start, [0, 1, 5, 6]])[None]
                prediction = model(observed, candidates)[0, 0, : n * spec.substeps]
                begin = start * spec.substeps + 1
                target = torch.tensor(dense[begin : begin + n * spec.substeps, :2])
                error = (prediction - target).norm(dim=-1)
                max_error = max(max_error, float(error.max()))
                compared_points += len(error)
                query_count += 1
            points = torch.tensor(dense[:, :2])
            clearance = min(
                float(c.clearance(points, spec.body_radius).min())
                for c in pusht_constraints(result["case"])
            )
            goal = np.asarray(result["goal_state"])
            position_error = float(np.linalg.norm(states[-1, :4] - goal[:4]))
            angle_error = abs(float(states[-1, 4] - goal[4]))
            angle_error = min(angle_error, 2 * np.pi - angle_error)
            passed = (
                max_error <= float(model.error_allowance)
                and clearance >= 6
                and position_error < 20
                and angle_error < np.pi / 9
                and result["safe_completion"]
                and not result["unfinished"]
            )
            rows.append(
                {
                    "layout": kind,
                    "passed": bool(passed),
                    "executed_actions": len(actions),
                    "compared_physics_points_including_overlapping_prefixes": compared_points,
                    "maximum_reloaded_motion_error_px": max_error,
                    "minimum_full_pusher_clearance_px": clearance,
                    "joint_goal_position_error_px": position_error,
                    "goal_angle_error_rad": angle_error,
                }
            )
    return {
        "passed": all(row["passed"] for row in rows),
        "checkpoint_sha256": file_digest(checkpoint),
        "partitions_disjoint": True,
        "source_groups": {k: len(v) for k, v in partitions.items()},
        "new_simulator_interactions": 0,
        "offline_learned_model_queries": query_count,
        "calibration_allowance_px": float(model.error_allowance),
        "results": rows,
        "scope": "Reloaded artifact and existing three known-scene traces; no new success trials",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--scalers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(4)
    result = audit(args.checkpoint, args.training, args.results, args.scalers)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise RuntimeError("Saved PushT artifact failed completion audit")


if __name__ == "__main__":
    main()
