"""Replay stored actions and scoring, then check live task-9 engineering evidence."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.unisafe09_adapter import Adapter
from helpers.unisafe09_controller import backup_arm_action
from helpers.unisafe09_environment import normalized_to_environment
from helpers.unisafe09_outcomes import EpisodeScore
from helpers.unisafe09_protocol import digest
from helpers.unisafe09_rendering import validate_cameras


def validate_attempt(folder):
    if (folder / "invalidation.json").exists():
        raise ValueError("Attempt was invalidated; retain it outside performance estimates")
    summary = json.loads((folder / "summary.json").read_text())
    job = summary["job"]
    rows = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
    if len(rows) != summary["controls"]:
        raise ValueError("Control trace is incomplete")
    score = EpisodeScore(job["continuation"])
    adapter = Adapter(job["condition_spec"]) if job["condition_spec"] else None
    for index, row in enumerate(rows):
        if row["step"] != index:
            raise ValueError("Trace step order differs")
        task = torch.tensor([row["task_proposed_action"]])
        proposed = task
        if adapter:
            proposed, expected_log = adapter(
                task, {"gripper_pos": torch.tensor([row["before"]["gripper_pos"]])}, None
            )
            for key, value in expected_log.items():
                if row[key] != value:
                    raise ValueError(f"Adapter replay differs at {index}: {key}")
        torch.testing.assert_close(
            proposed, torch.tensor([row["proposed_action"]]), rtol=0, atol=1e-7
        )
        filtered = proposed.clone()
        if row["intervened"]:
            filtered[:, :6] = backup_arm_action(
                torch.tensor([row["backup_raw_arm_action"]]), "released"
            )
        torch.testing.assert_close(
            filtered, torch.tensor([row["filtered_action"]]), rtol=0, atol=1e-7
        )
        torch.testing.assert_close(
            normalized_to_environment(filtered),
            torch.tensor([row["environment_action"]]),
            rtol=0,
            atol=1e-7,
        )
        camera = np.load(folder / "cameras" / f"{index:04d}.npz")["cameras"]
        validate_cameras(camera)
        if hashlib.sha256(camera.tobytes()).hexdigest() != row["camera_sha256"]:
            raise ValueError("Recorded camera hash differs")
        terms = row["termination_terms"]
        finished = score.advance(
            success=terms["success"],
            physical_failure=row["after"]["physical_failure"]
            or terms["cube_2_dropping"]
            or terms["cube_3_dropping"],
            boundary=terms["ee_position"],
        )
        if finished != (index == len(rows) - 1):
            raise ValueError("Episode stopped at the wrong control")
        if score.benchmark and not finished and row["automatic_resets"]:
            raise ValueError("Continuation includes a reset")
        if not finished and row["episode_length"] != index + 1:
            raise ValueError("Live episode counter reset before termination")
    for key, value in score.summary().items():
        if summary[key] != value:
            raise ValueError(f"Outcome replay differs: {key}")
    expected_queries = {
        "encoder": 1,
        "posterior": 2,
        "task_actor": 1,
        "uncertainty_ensemble": 1,
        "imagined_step": 1,
        "safety_actor": 2,
        "safety_critic": 2,
        "failure_head": 1,
    }
    if summary["query_counts"] != {k: v * len(rows) for k, v in expected_queries.items()}:
        raise ValueError("Query accounting differs from the controller contract")
    if summary["rollout_physics_steps"] != len(rows) * 5:
        raise ValueError("Physics accounting differs")
    if summary["physics_steps"] != len(rows) * 5 + summary["reset_internal_physics_steps"]:
        raise ValueError("Reset/internal physics accounting is incomplete")
    if job["condition"] == "native_direct":
        if not all(row["recorded_input_identity_passed"] for row in rows):
            raise ValueError("Recorded-input native identity failed")
        if summary["shadow_query_counts"] != summary["query_counts"]:
            raise ValueError("Native shadow query accounting differs")
    for name in (
        "initial_state.npz",
        "terminal_state.npz",
        "reset_cameras.npz",
        "terminal_cameras.npz",
    ):
        with np.load(folder / name) as arrays:
            if not arrays.files:
                raise ValueError(f"Empty array artifact: {name}")
            for array in arrays.values():
                if not np.isfinite(array).all():
                    raise ValueError(f"Non-finite snapshot: {name}")
    return summary, rows


def paired_prefix(first, second):
    folder_a, folder_b = first[0], second[0]
    a, b = first[2], second[2]
    with (
        np.load(folder_a / "initial_state.npz") as left,
        np.load(folder_b / "initial_state.npz") as right,
    ):
        initial = {key: np.array_equal(left[key], right[key]) for key in left.files}
    common = 0
    first_changed = None
    for index, (x, y) in enumerate(zip(a, b)):
        if x["environment_action"] != y["environment_action"] and first_changed is None:
            first_changed = index
        if (
            common == index
            and x["before"] == y["before"]
            and x["camera_sha256"] == y["camera_sha256"]
        ):
            common += 1
    return {
        "initial_physical_arrays_equal": initial,
        "observed_prefix_controls": common,
        "first_changed_executed_action": first_changed,
        "same_benchmark_outcome": first[1]["outcome"] == second[1]["outcome"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/unisafe09/remote")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engineering", action="store_true")
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    valid = {}
    invalid = []
    for folder in sorted((args.root / "attempts").glob("*")):
        intent = json.loads((folder / "intent.json").read_text())
        if (folder / "invalidation.json").exists():
            invalid.append(folder.name)
            continue
        if intent["freeze_sha256"] != digest(args.freeze):
            continue
        if not (folder / "summary.json").exists():
            invalid.append(folder.name)
            continue
        summary, rows = validate_attempt(folder)
        job_id = summary["job"]["id"]
        if job_id in valid:
            raise ValueError("Duplicate valid job")
        valid[job_id] = (folder, summary, rows)
    if set(valid) != {j["id"] for j in freeze["jobs"]}:
        raise ValueError("The complete frozen stage has not finished")
    checks = {
        "complete_valid_panel": True,
        "all_action_and_outcome_replays_passed": True,
        "accounting_and_raw_artifacts_complete": True,
    }
    prefixes = {}
    if args.engineering:
        checks.update(
            native_recorded_input_identity=valid["engineering-00"][1]["identity_checks"] > 0,
            live_translation_activity=valid["engineering-03"][1]["translation_active_steps"] > 0,
            live_history_override=valid["engineering-04"][1]["gripper_override_steps"] > 0,
            live_latch_override=valid["engineering-05"][1]["gripper_override_steps"] > 0,
            live_40_control_continuation=any(
                x[1]["continuation_steps"] == 40 for x in valid.values()
            ),
            no_optimizer_updates=all(x[1]["training_updates"] == 0 for x in valid.values()),
        )
        for i, j in ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)):
            prefixes[f"{i}-{j}"] = paired_prefix(
                valid[f"engineering-{i:02d}"], valid[f"engineering-{j:02d}"]
            )
    # Exclude first shader-startup job and its extra identity queries from steady-state projection.
    timing = [x for x in valid.values() if x[1]["identity_checks"] == 0]
    startup = max(x[1]["startup_seconds"] for x in timing)
    control = max(
        (x[1]["elapsed_seconds"] - x[1]["startup_seconds"]) / x[1]["controls"] for x in timing
    )
    overhead = max(
        max(
            0.0,
            json.loads((x[0] / "process-result.json").read_text())["elapsed_seconds"]
            - x[1]["elapsed_seconds"],
        )
        for x in timing
    )
    result = {
        "utc": utc(),
        "passed": all(checks.values()),
        "checks": checks,
        "freeze_sha256": digest(args.freeze),
        "invalid_attempts": invalid,
        "paired_prefixes": prefixes,
        "episodes": [x[1] for x in valid.values()],
        "runtime_projection": {
            "max_steady_startup_seconds": startup,
            "max_measured_seconds_per_control": control,
            "max_process_overhead_seconds": overhead,
            "safety_multiplier": 1.25,
        },
        "artifact_bytes": sum(p.stat().st_size for p in args.root.rglob("*") if p.is_file()),
    }
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "checks": checks,
                "runtime_projection": result["runtime_projection"],
            }
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
