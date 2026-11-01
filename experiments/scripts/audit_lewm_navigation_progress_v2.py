"""Audit a completed frozen progress-target pilot from saved evidence only.

No environment, renderer, world model or safety head is loaded or queried. Cached
feature/readout arithmetic runs only on the declared rented GPU host.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm_navigation_retrieval_v2 import (
    FrozenGoalReadout,
    RetrievalController,
    RetrievalMemory,
)
from helpers.portable_motion import clearance, shape_constraints


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def maximum_run(indices):
    longest = current = 0
    previous = object()
    for index in indices:
        current = current + 1 if index == previous else 1
        longest, previous = max(longest, current), index
    return longest


def audit(protocol_path, output_path):
    protocol = json.loads(protocol_path.read_text())
    design = protocol["evaluation"]
    require(os.environ.get("RUNPOD_POD_ID") == design["execution_pod_id"], "Wrong rented host")
    require(torch.cuda.is_available(), "Rented GPU required")
    require(not output_path.exists(), "Use a new saved-evidence audit namespace")
    run = ROOT / protocol["output_namespace"]
    state = json.loads((run / "state.json").read_text())
    require(state["phase"] == "complete" and state["active"] is None, "Pilot must be terminal")
    require(len(state["jobs"]) == len(protocol["jobs"]) == 16, "All frozen jobs required")
    for name, expected in protocol["source_sha256"].items():
        require(digest(ROOT / name) == expected, "Frozen source changed: " + name)
    require(digest(ROOT / design["memory_path"]) == design["memory_sha256"], "Memory changed")
    readout_config = design["goal_readout"]
    require(digest(ROOT / readout_config["path"]) == readout_config["sha256"], "Readout changed")
    memory = RetrievalMemory.from_npz(ROOT / design["memory_path"])
    readout = FrozenGoalReadout.from_npz(ROOT / readout_config["path"])
    records = []
    for planned in protocol["jobs"]:
        job_id, case = planned["job_id"], planned["case"]
        folder = run / "jobs" / job_id / "episode"
        result = json.loads((folder / "result.json").read_text())
        history = json.loads((folder / "selector-history.json").read_text())
        with np.load(folder / "trace.npz", allow_pickle=False) as saved:
            trace = {name: saved[name].copy() for name in saved.files}
        require(result["error"] is None and not result["frame_flags"], "Technical/image failure")
        require(result["candidate_projection"], "Candidate projection must remain enabled")
        require(result["shape"] == case["shape"], "Original hazard changed")
        np.testing.assert_allclose(trace["states"][0], case["initial_xy"], atol=1e-5, rtol=0)
        np.testing.assert_allclose(trace["original_goal"], case["goal_xy"], atol=1e-5, rtol=0)
        np.testing.assert_array_equal(trace["actions"], np.clip(trace["proposed"], -1, 1))
        require(
            len(trace["actions"]) == result["recorded_successful_actions"], "Action trace count"
        )
        require(result["environment_actions"] == len(trace["actions"]), "Failed action attempt")
        require(len(trace["actions"]) <= planned["max_actions"], "Frozen action cap exceeded")
        geometry = shape_constraints(case["shape"])
        clear = clearance(torch.as_tensor(trace["dense"]), geometry, design["body_radius"])
        violation = bool((clear < design["margin"] - 1e-6).any())
        final_distance = float(np.linalg.norm(trace["states"][-1] - trace["original_goal"]))
        success = final_distance < 16
        require(violation == result["violation"], "Original violation monitor changed")
        require(success == result["success"], "Original physical-goal monitor changed")
        require(result["unfinished"] == (not success), "Original unfinished outcome changed")
        require(
            result["safe_completion"]
            == bool(success and not violation and not result["controller_failure"]),
            "Original safe completion changed",
        )
        observed_distance, target_progress, indices, direct_count = [], [], [], 0
        estimated_errors = []
        native = planned["method"] == "native"
        actor = None
        if native:
            require(not history, "Native target path must remain disabled")
        else:
            actor = RetrievalController(
                memory,
                planned["parameters"],
                case["shape"],
                body_radius=design["body_radius"],
                margin=design["margin"],
                allowance=design["motion_allowance"],
                goal_readout=readout,
                controller_config=design["controller_config"],
            )
            expected_prefix = {"recent_target_xy": [], "previous_current_xy": None}
            for query_index, row in enumerate(history):
                require(row["causal_context"] == expected_prefix, "Causal history prefix changed")
                current = torch.tensor(row["current_latent"], dtype=torch.float32, device="cuda")
                final = torch.tensor(row["final_latent"], dtype=torch.float32, device="cuda")
                features, available = actor.features(
                    current, final, row["current_xy"], context=row["causal_context"]
                )
                arrays = [x.detach().cpu().contiguous().numpy() for x in (features, available)]
                for array, name in zip(arrays, ("features", "available"), strict=True):
                    require(
                        hashlib.sha256(array.tobytes()).hexdigest() == row[name + "_sha256"],
                        "Exact selector reconstruction failed: " + name,
                    )
                require(row["memory_sha256"] == memory.fingerprint, "Selector memory identity")
                require(
                    row["goal_readout_sha256"] == readout.fingerprint, "Selector readout identity"
                )
                require(row["query"] == query_index, "Selector query sequence")
                np.testing.assert_array_equal(row["current_xy"], trace["states"][query_index * 5])
                chosen = int(actor.logits(features, available, planned["parameters"]).argmax())
                index = row["memory_index"]
                require(
                    chosen == (memory.count if index is None else index), "Target argmin changed"
                )
                require(bool(available[chosen]), "Selected masked target")
                goal_hat = actor._last_estimated_goal.detach().cpu().numpy()
                np.testing.assert_array_equal(goal_hat, row["estimated_goal_xy"])
                estimated_errors.append(float(np.linalg.norm(goal_hat - trace["original_goal"])))
                xy = np.asarray(row["current_xy"])
                if index is None:
                    direct_count += 1
                    target = goal_hat
                    require(np.linalg.norm(target - xy) <= 30 + 1e-5, "Nonlocal direct-goal target")
                else:
                    target = memory.tables["successor_xy"][index].cpu().numpy()
                    distance = float(np.linalg.norm(target - xy))
                    observed_distance.append(distance)
                    require(6 - 1e-5 <= distance <= 30 + 1e-5, "Observed target outside fixed band")
                    indices.append(index)
                np.testing.assert_array_equal(target, row["selected_target_xy_estimate"])
                target_progress.append(
                    float(
                        np.linalg.norm(xy - trace["original_goal"])
                        - np.linalg.norm(target - trace["original_goal"])
                    )
                )
                expected_prefix = {
                    "recent_target_xy": (expected_prefix["recent_target_xy"] + [target.tolist()])[
                        -6:
                    ],
                    "previous_current_xy": row["current_xy"],
                }
        resources = result["resources"]
        readout_attempts = resources.get("goal_readout_call_attempts", 0)
        gap = readout_attempts - len(history)
        require(native or gap in (0, 1), "Unexpected readout/selection accounting gap")
        require(not native or readout_attempts == 0, "Native performed an online readout")
        require(gap == 0 or result["controller_failure"], "Unexplained selector provenance gap")
        records.append(
            {
                "job_id": job_id,
                "case_id": case["case_id"],
                "method": planned["method"],
                "safe_completion": result["safe_completion"],
                "violation": violation,
                "unfinished": result["unfinished"],
                "engaged": result["engaged"],
                "actions": result["environment_actions"],
                "initial_xy": trace["states"][0].tolist(),
                "original_goal_xy": trace["original_goal"].tolist(),
                "final_xy": trace["states"][-1].tolist(),
                "initial_goal_distance": result["initial_original_goal_distance"],
                "final_goal_distance": final_distance,
                "bounded_progress": result["progress"],
                "observed_targets": len(observed_distance),
                "direct_original_targets": direct_count,
                "observed_target_distance_min": min(observed_distance, default=None),
                "observed_target_distance_median": float(np.median(observed_distance))
                if observed_distance
                else None,
                "observed_target_distance_max": max(observed_distance, default=None),
                "target_actual_progress_positive": sum(x > 0 for x in target_progress),
                "target_actual_progress_median": float(np.median(target_progress))
                if target_progress
                else None,
                "unique_observed_targets": len(set(indices)),
                "maximum_consecutive_observed_repeat": maximum_run(indices),
                "readout_error_max_px": max(estimated_errors, default=None),
                "controller_failure": result["controller_failure"],
                "missing_terminal_support_query_provenance": bool(gap),
                "planning_seconds": result["planning_seconds"],
                "episode_seconds": result["episode_seconds"],
                "protected_body_path_length": result["protected_body_path_length"],
                "episode_result_sha256": digest(folder / "result.json"),
                "trace_sha256": digest(folder / "trace.npz"),
                "selector_history_sha256": digest(folder / "selector-history.json"),
            }
        )
    summary = {}
    for method in sorted({row["method"] for row in records}):
        rows = [row for row in records if row["method"] == method]
        summary[method] = {
            "episodes": len(rows),
            **{
                key: sum(row[key] for row in rows)
                for key in ("safe_completion", "violation", "unfinished", "engaged", "actions")
            },
            "controller_refusals": sum(bool(row["controller_failure"]) for row in rows),
            "bounded_progress_sum": sum(row["bounded_progress"] for row in rows),
        }
    report = {
        "verified": True,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_sha256": digest(protocol_path),
        "audit_source_sha256": digest(Path(__file__)),
        "new_world_model_or_safety_queries": 0,
        "new_simulator_actions_or_resets": 0,
        "cached_linear_readout_reconstructions": readout.calls,
        "summary": summary,
        "records": records,
        "limitation": "V2 runner serializes initial controller artifact only. Any terminal empty-support refusal has attempted-work accounting but its missing selector input is explicitly flagged; no evidence is invented or replayed.",
        "interpretation": "Architecture pilot only. No optimizer training or evolutionary advantage is established.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.protocol, args.out)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
