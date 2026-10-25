"""Protect complete-panel admission and byte-preserving shutdown handover."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.screen08_preservation import completion, sha256, snapshot, verify_snapshot
from helpers.screen08_protocol import protocol


def completed_panel(root, survivors=()):
    output = root / "outputs/screen08"
    output.mkdir(parents=True)
    plan = protocol()
    frozen = root / "frozen.py"
    frozen.write_text("frozen experiment source\n")
    plan["source_sha256"] = {"frozen.py": sha256(frozen)}
    plan_path = root / "progress/receipts/08-screen-plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps(plan))
    decision = {"plan_sha256": sha256(plan_path), "survivors": list(survivors),
                "admit": bool(survivors), "reason": "admitted" if survivors else "no_survivors",
                "evidence_sha256": {}}
    jobs = plan["engineering_jobs"] + plan["jobs"]
    if survivors:
        jobs += [j for j in plan["stage_b_jobs"] if j["arm"] in ("native", *survivors)]
    for job in jobs:
        directory = output / "episodes" / f"job{job['job']}"
        directory.mkdir(parents=True)
        (directory / "job.json").write_text(json.dumps(job))
        validated = directory / "validation.json"
        validated.write_text(json.dumps({
            "screen_mode": job["arm"],
            "row": {"status": "completed", "seed": job["seed"],
                    "scene_name": f"task_{job['task']:04d}/base"},
        }))
        if job["stage"] == "stage_a":
            decision["evidence_sha256"][str(validated.relative_to(output))] = sha256(validated)
    (output / "stage-b-decision.json").write_text(json.dumps(decision))
    for name in ("pipeline-ended", "study-complete", "engineering-complete",
                 "stage_a-complete", "stage_b-complete"):
        (output / f"{name}.txt").write_text("2026-10-25T12:00:00Z\n")
    (output / "pipeline-exit-code.txt").write_text("0\n")
    return output


def test_completion_markers_cannot_hide_missing_screen_episode(tmp_path):
    output = completed_panel(tmp_path)
    assert len(completion(tmp_path)["validated_jobs"]) == 65
    (output / "episodes/job159/validation.json").unlink()
    with pytest.raises(ValueError, match="complete scheduled panel"):
        completion(tmp_path)


def test_confirmation_requires_all_selected_arms_and_its_completion_marker(tmp_path):
    output = completed_panel(tmp_path, ("h4", "slow"))
    assert completion(tmp_path)["confirmation_jobs"] == 36
    (output / "stage_b-complete.txt").unlink()
    with pytest.raises(ValueError, match="confirmation has not finished"):
        completion(tmp_path)
    (output / "stage_b-complete.txt").write_text("finished\n")
    (output / "episodes/job201/validation.json").unlink()
    with pytest.raises(ValueError, match="complete scheduled panel"):
        completion(tmp_path)


def test_changed_selection_evidence_is_rejected(tmp_path):
    output = completed_panel(tmp_path)
    path = output / "episodes/job100/validation.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="evidence changed"):
        completion(tmp_path)


def test_failed_pipeline_is_not_a_completed_study(tmp_path):
    output = completed_panel(tmp_path)
    (output / "pipeline-exit-code.txt").write_text("1\n")
    assert completion(tmp_path)["status"] == "needs_attention"


def test_live_process_blocks_snapshot_and_verified_copy_detects_tampering(tmp_path, monkeypatch):
    completed_panel(tmp_path)
    monkeypatch.setenv("CONTAINER_ID", "57639938")
    monkeypatch.setattr("helpers.screen08_preservation.active_experiment_processes", lambda: [12])
    assert snapshot(tmp_path, 57639938)["reason"] == "waiting_for_process_exit"
    assert not (tmp_path / "outputs/screen08-completed-snapshot").exists()
    monkeypatch.setattr("helpers.screen08_preservation.active_experiment_processes", lambda: [])
    state = snapshot(tmp_path, 57639938)
    destination = Path(state["path"])
    assert verify_snapshot(destination)["instance_id"] == 57639938
    copied = destination / "raw/episodes/job100/validation.json"
    copied.write_text("changed after copying\n")
    with pytest.raises(ValueError, match="inventory or bytes differ"):
        verify_snapshot(destination)


def test_snapshot_refuses_another_allocation(tmp_path, monkeypatch):
    completed_panel(tmp_path)
    monkeypatch.setenv("CONTAINER_ID", "different-allocation")
    monkeypatch.setattr("helpers.screen08_preservation.active_experiment_processes", lambda: [])
    with pytest.raises(ValueError, match="recorded task-8 allocation"):
        snapshot(tmp_path, 57639938)
