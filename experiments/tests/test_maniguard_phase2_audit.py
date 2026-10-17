"""Preparation audits distinguish measured evidence from unproved equivalence."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from test_maniguard_phase2_results import synthetic_episode, write_json, write_trace

from helpers.maniguard_phase2_audit import audit_preparation, audit_sources, compare_identity


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identity_pair(tmp_path):
    result = []
    original_source = ("from helpers.maniguard_runtime import STUDY\n"
                       "STUDY.start(scene_info, obs)\n")
    for phase2, label in ((False, "original"), (True, "identity")):
        directory, job, _, trace, _ = synthetic_episode(tmp_path / label, stochastic=False)
        job.update(phase="phase2" if phase2 else "phase2_baseline_check",
                   purpose="engineering", max_steps_override=32, action_horizon=32)
        if not phase2:
            job["adapter"] = {"kind": "identity", "horizon": 8}
            trace = [row for row in trace if row["type"] not in
                     ("phase2_features", "phase2_actor", "decision") and not
                     (row["type"] == "observed" and row["step"] == 0)]
        source = original_source
        if phase2:
            source = source.replace("maniguard_runtime", "maniguard_phase2_runtime").replace(
                "STUDY.start(scene_info, obs)", "STUDY.start(scene_info, obs, env, robot, monitor)")
        (directory / "benchmark_instrumented.py").write_text(source)
        provenance = json.loads((directory / "runtime-provenance.json").read_text())
        provenance["runtime_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        if phase2:
            provenance["phase1_runtime_sha256"] = hashlib.sha256(original_source.encode()).hexdigest()
        write_json(directory / "runtime-provenance.json", provenance)
        state, overview, wrist = np.zeros(8, np.float32), np.zeros((4, 4, 3), np.uint8), np.ones(
            (4, 4, 3), np.uint8)
        np.savez(directory / "initial_observation.npz", state=state, overview=overview, wrist=wrist)
        trace.insert(1, {"type": "initial", "state": state.tolist()})
        for row in trace:
            if row["type"] in ("query", "observed"):
                row["state"] = (state + row["step"] * .01).tolist()
            if row["type"] == "query":
                row["image_hashes"] = {
                    "overview_image": hashlib.sha256(overview.tobytes()).hexdigest(),
                    "wrist_images": hashlib.sha256(wrist.tobytes()).hexdigest()}
        write_json(directory / "job.json", job)
        write_trace(directory / "trace.jsonl", trace)
        result.append(directory)
    return result


def test_identical_recorded_start_allows_recorded_trajectory_comparison(tmp_path):
    paths = identity_pair(tmp_path)
    report = compare_identity(*paths)
    assert report["valid"], report["invalid_reasons"]
    assert report["status"] == "recorded_trajectory_equal"
    assert report["recorded_starts_match"] and all(report["trajectory_comparison"].values())
    assert report["exact_identity_transformation"] == [True, True]
    assert not report["performance_claim"]


def test_same_seed_but_different_images_blocks_identical_start_claim(tmp_path):
    paths = identity_pair(tmp_path)
    with np.load(paths[1] / "initial_observation.npz") as data:
        state, overview, wrist = (data[key].copy() for key in ("state", "overview", "wrist"))
    overview[0, 0, 0] = 100
    np.savez(paths[1] / "initial_observation.npz", state=state, overview=overview, wrist=wrist)
    trace = [json.loads(line) for line in (paths[1] / "trace.jsonl").read_text().splitlines()]
    next(row for row in trace if row["type"] == "query")["image_hashes"][
        "overview_image"] = hashlib.sha256(overview.tobytes()).hexdigest()
    write_trace(paths[1] / "trace.jsonl", trace)
    report = compare_identity(*paths)
    assert report["valid"], report["invalid_reasons"]
    assert report["status"] == "not_comparable_different_recorded_starts"
    assert report["trajectory_comparison"] is None and not report["recorded_starts_match"]


@pytest.mark.parametrize("mutation", ["initial_hash", "schedule", "transformation", "source"])
def test_identity_checks_reject_broken_implementation_evidence(tmp_path, mutation):
    paths = identity_pair(tmp_path)
    trace = [json.loads(line) for line in (paths[1] / "trace.jsonl").read_text().splitlines()]
    if mutation == "initial_hash":
        next(row for row in trace if row["type"] == "query")["image_hashes"][
            "wrist_images"] = "bad"
    elif mutation == "schedule":
        next(row for row in trace if row["type"] == "query")["step"] = 1
    elif mutation == "transformation":
        next(row for row in trace if row["type"] == "proposal")["adapted"][0] = .2
    else:
        path = paths[1] / "benchmark_instrumented.py"
        path.write_text(path.read_text() + "# unexpected edit\n")
    write_trace(paths[1] / "trace.jsonl", trace)
    report = compare_identity(*paths)
    assert not report["valid"] and report["invalid_reasons"]


def test_archive_hashes_are_validated_without_requiring_current_workspace_match(tmp_path):
    directory, *_ = synthetic_episode(tmp_path)
    source = audit_sources(directory, root=tmp_path / "later-workspace", verify_upstream=False)
    assert source["valid"] and source["workspace_differences"]
    assert not source["loaded_module_attestation"]
    archive = directory / "study_source/configs/maniguard/phase2/encoder.json"
    archive.write_text(archive.read_text() + " ")
    source = audit_sources(directory, verify_upstream=False)
    assert not source["valid"]


def preparation_fixture(tmp_path):
    directory = tmp_path / "preparation"
    jobs = [{"job_id": "short-" + str(i), "max_steps": 32} for i in range(3)]
    manifest = {"short_jobs": jobs, "full_jobs": [{"job_id": "full-0", "max_steps": None,
                                                  "full_horizon": 2000}],
                "budget": {"short_attempts": 3, "full_attempts": 1, "policy_action_ceiling": 2096,
                           "integration_gpu_hour_ceiling": 1, "full_preparation_gpu_hour_ceiling": 1}}
    manifest["sha256"] = digest(manifest)
    write_json(directory / "preparation.json", manifest)
    receipts = []
    for i, status in enumerate(("valid", "invalid", "started")):
        output = directory / ("short-" + str(i))
        receipt = {"job": jobs[i], "short_check": True, "status": status,
                   "output": str(output), "timeout_seconds": 900,
                   "protocol_sha256": manifest["sha256"],
                   "ledger": {"phase": "completed" if status == "valid" else "incomplete",
                              "policy_control_steps": [32, 5, 0][i], "physics_steps": 100,
                              "policy_queries": i + 1, "elapsed_seconds": 8}}
        if status != "started":
            receipt["elapsed_seconds"] = [10, 20][i]
        write_json(directory / "attempts" / (jobs[i]["job_id"] + ".json"), receipt)
        receipts.append(receipt)
    return directory, manifest, receipts


def test_invalid_and_pending_attempts_consume_slots_costs_and_timeout_exposure(tmp_path):
    directory, _, _ = preparation_fixture(tmp_path)
    report = audit_preparation(directory)
    assert report["valid"], report["invalid_reasons"]
    short = report["ceilings"]["short"]
    assert short["attempts"] == 3 and short["remaining_slots"] == 0
    assert short["statuses"] == {"valid": 1, "invalid": 1, "started": 1}
    assert short["launcher_seconds_measured"] == 30
    assert short["pending_timeout_seconds_reserved"] == 900
    assert report["assigned_action_exposure"] == 96
    assert report["measured_cost_totals"]["policy_control_steps"] == 37
    assert report["measured_cost_totals"]["physics_steps"] == 300
    assert report["incomplete_cost_attempts"] == ["short-1", "short-2"]


def test_duplicate_or_orphan_work_cannot_disappear_from_preparation_budget(tmp_path):
    directory, _, receipts = preparation_fixture(tmp_path)
    write_json(directory / "attempts/duplicate.json", receipts[0])
    write_json(directory / "orphan/job.json", {"max_steps_override": 32, "action_horizon": 32})
    write_json(directory / "orphan/ledger.json", {"physics_steps": 50, "policy_control_steps": 3})
    report = audit_preparation(directory)
    assert not report["valid"] and report["attempt_count"] == 5
    assert report["ceilings"]["short"]["remaining_slots"] == -2
    assert report["measured_cost_totals"]["physics_steps"] == 350  # duplicate receipt is not new work
    assert any("no attempt receipt" in reason for reason in report["invalid_reasons"])


def test_receipt_assignment_mutation_is_not_silently_accepted(tmp_path):
    directory, _, receipts = preparation_fixture(tmp_path)
    receipt = copy.deepcopy(receipts[0])
    receipt["job"]["max_steps"] = 4000
    write_json(directory / "attempts/short-0.json", receipt)
    report = audit_preparation(directory)
    assert not report["valid"]
    assert report["assigned_action_exposure"] > report["policy_action_ceiling"]


def test_malformed_receipt_still_consumes_an_unclassified_attempt(tmp_path):
    directory, _, _ = preparation_fixture(tmp_path)
    (directory / "attempts/broken.json").write_text("{unfinished")
    report = audit_preparation(directory)
    assert not report["valid"] and report["attempt_count"] == 4
    assert report["unclassified_attempts"] == 1
