"""Decision checks for the amended resume path; all runtime operations are stubs."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import resume_dsrl06_engineering as entry  # noqa: E402
from run_dsrl06 import atomic_json, timestamp  # noqa: E402
from helpers.dsrl06_protocol import frozen_jobs  # noqa: E402


def setup(tmp_path, monkeypatch, *, authority=True, remaining=20000):
    output = tmp_path / "run"
    output.mkdir()
    learner = output / "learner"
    learner.mkdir()
    begin, end = "2026-10-24T23:48:24+00:00", "2026-10-24T23:50:26+00:00"
    events = [
        {"event": "update_reserved", "label": label, "utc": at}
        for label, at in (
            ("author_reference", "2026-10-24T23:48:26+00:00"),
            ("bridge_parity", "2026-10-24T23:49:32+00:00"),
        )
    ]
    (learner / "learner-ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in events))
    original_request = {
        "body": {"path": "recorded-inputs.npz", "sha256": "inputs", "updates": 128, "parity": True}
    }
    request_path = output / "http/engineering-profile.request.json"
    atomic_json(request_path, original_request)
    plan_path = tmp_path / "freeze.json"
    atomic_json(plan_path, {"amendment": "explicit test fixture"})
    calls = []
    state = {
        "episodes": {
            job: {"valid": True, "finished_utc": "2026-10-24T23:51:52+00:00"}
            for job in ("E0", "E1", "E2")
        },
        "engineering_overlap": ["E0", "E1", "E2"],
        "learner_process": "learner",
    }
    amendment = {"original_profile_success": False, "explicit_measurement_association": True}
    profile = {
        "source_parity": {"maximum_state_difference": 0.0},
        "timed_updates": 128,
        "median_update_seconds": 0.01,
        "request_sha256": "new-exact-request",
    }
    queue = SimpleNamespace(
        output=output,
        learner_dir=learner,
        state=state,
        plan={"amendment": {"measurement_association": amendment}, "jobs": frozen_jobs()},
        rental={"instance_id": 7, "allocation_deadline_utc": "2026-10-25T04:50:22+00:00"},
    )
    queue.processes = SimpleNamespace(
        spec=lambda key: {"created_utc": begin},
        status=lambda key: {"finished_utc": end, "returncode": 1},
    )
    queue.save = lambda: None
    queue.wait = lambda key: calls.append(("wait", key))
    queue.learner = lambda phase: calls.append(("learner", phase))

    def http(key, route, body):
        calls.append((route, {"key": key, "body": dict(body)}))
        return profile if route == "/profile" else {}

    queue.http = http
    queue.checkpoint = lambda name: {"path": "private/checkpoint", "name": name}
    queue.cpu_actor_parity = lambda: {"valid": True, "median_cpu_seconds": 0.003}
    queue.authority = lambda: {"gate": {"valid": True, "admit": authority}}

    def start(job, **kwargs):
        calls.append(("start", job["job_id"]))
        state["episodes"][job["job_id"]] = {
            "valid": True,
            "started_utc": "2026-10-25T00:10:00+00:00",
            "finished_utc": "2026-10-25T00:12:00+00:00",
        }

    queue.start_episode = start
    queue.wait_episodes = lambda jobs: calls.append(("wait_episodes", jobs))
    queue.sync_counters = lambda: {"usage": {"engineering_updates": 136}}
    queue.close = lambda **kwargs: calls.append(("close", kwargs))
    monkeypatch.setattr(entry, "Coordinator", lambda *args, **kwargs: queue)
    monkeypatch.setattr(
        entry.time, "time", lambda: timestamp(queue.rental["allocation_deadline_utc"]) - remaining
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "resume_dsrl06_engineering.py",
            "--plan",
            str(plan_path),
            "--repair-receipt",
            str(tmp_path / "repair.json"),
        ],
    )
    return queue, calls, request_path, original_request


def result(queue):
    return json.loads((queue.output / "stage-a-summary.json").read_text())


def test_negative_authority_never_starts_remaining_engineering_episodes(tmp_path, monkeypatch):
    queue, calls, _, _ = setup(tmp_path, monkeypatch, authority=False)
    entry.main()
    summary = result(queue)
    assert summary["authority_admit"] is False
    assert summary["admit"] is False
    assert not any(name == "start" for name, _ in calls)
    assert calls[-1] == ("close", {"stop": True})


def test_positive_checks_and_budget_admit_only_after_repaired_calibration(tmp_path, monkeypatch):
    queue, calls, _, _ = setup(tmp_path, monkeypatch)
    entry.main()
    summary = result(queue)
    assert summary["admit"] is True
    assert summary["source_parity_valid"] is True
    assert summary["topology_three_rollouts"] is True
    assert [value for name, value in calls if name == "start"] == ["E4", "E5"]
    request = next(value["body"] for name, value in calls if name == "/profile")
    assert request["updates"] == 128
    assert request["parity"] is True
    assert request["operation_id"] == "engineering-profile-repair-01"
    assert calls[-1] == ("close", {"stop": False})


def test_original_failed_request_and_two_call_coexistence_remain_separate(tmp_path, monkeypatch):
    queue, _, request_path, original_request = setup(tmp_path, monkeypatch)
    entry.main()
    summary = result(queue)
    assert summary["original_profile_success"] is False
    assert summary["original_profile_http_exit"]["returncode"] == 1
    assert summary["topology_gradient_calls_observed"] == 2
    assert summary["three_rollout_profile_overlap_seconds"] == 122
    assert summary["profile"]["timed_updates"] == 128
    assert summary["measurement_association_amendment"]["explicit_measurement_association"]
    assert json.loads(request_path.read_text()) == original_request
    assert "operation_id" not in original_request["body"]


def test_insufficient_projected_budget_does_not_admit_learning(tmp_path, monkeypatch):
    queue, calls, _, _ = setup(tmp_path, monkeypatch, remaining=100)
    entry.main()
    assert result(queue)["admit"] is False
    assert calls[-1] == ("close", {"stop": True})


def test_claiming_original_profile_success_is_rejected(tmp_path, monkeypatch):
    queue, calls, _, _ = setup(tmp_path, monkeypatch)
    queue.plan["amendment"]["measurement_association"]["original_profile_success"] = True
    with pytest.raises(ValueError, match="must remain failed"):
        entry.main()
    assert not any(name == "/profile" or name == "start" for name, _ in calls)
