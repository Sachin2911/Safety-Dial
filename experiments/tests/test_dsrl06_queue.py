"""CPU queue recovery and accounting checks, without robotics or neural calls."""

import copy
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_dsrl06 import (  # noqa: E402
    AdmissionClosed,
    Coordinator,
    PendingOperation,
    Processes,
    atomic_json,
    identity_alive,
    http_worker,
    process_identity,
    sha256,
    validate_amendment,
    verify_plan,
)
from helpers.dsrl06_protocol import CAPS, ProtocolLedger, frozen_jobs  # noqa: E402


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_dsrl06.py"


def minimal(tmp_path):
    queue = Coordinator.__new__(Coordinator)
    queue.root = tmp_path
    queue.output = tmp_path / "outputs/dsrl06"
    queue.output.mkdir(parents=True)
    queue.state_path = queue.output / "queue-state.json"
    queue.state = {"episodes": {}, "server_generations": {}}
    queue.plan = {
        "jobs": frozen_jobs(),
        "prior_engineering_updates": 2,
        "prior_engineering_receipt": "sha256:prior",
    }
    queue.plan_path = tmp_path / "plan.json"
    queue.plan_path.write_text("{}")
    queue.ledger = ProtocolLedger(queue.output / "resources.jsonl")
    queue.query_ledger = queue.output / "queries.sqlite"
    queue.learner_dir = queue.output / "learner"
    queue.repair = None
    queue.stage = "engineering"
    now = dt.datetime.now(dt.timezone.utc)
    queue.rental = {
        "instance_id": 7,
        "created_utc": now.isoformat(),
        "allocation_deadline_utc": (now + dt.timedelta(hours=6)).isoformat(),
    }
    queue.behavior = sys.executable
    queue.processes = SimpleNamespace(launch=lambda *args, **kwargs: None)
    queue.actor_url = "http://127.0.0.1:9009"
    return queue


def append(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        for row in values:
            stream.write(json.dumps(row) + "\n")


def wait_state(processes, key, desired):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = processes.status(key)
        if status["state"] == desired:
            return status
        time.sleep(0.02)
    pytest.fail(f"Managed process never reached {desired}: {status}")


def test_managed_resume_adopts_live_worker_and_never_reexecutes(tmp_path):
    processes = Processes(tmp_path / "processes", sys.executable, SCRIPT)
    output = tmp_path / "count"
    command = [
        sys.executable,
        "-c",
        "import pathlib,time; "
        f"p=pathlib.Path({str(output)!r}); p.write_text(p.read_text()+'x' if p.exists() else 'x'); "
        "time.sleep(.3)",
    ]
    processes.launch("same", command, cwd=tmp_path)
    first = wait_state(processes, "same", "running")
    resumed = Processes(tmp_path / "processes", sys.executable, SCRIPT)
    second = resumed.launch("same", command, cwd=tmp_path)
    assert second["pid"] == first["pid"]
    assert wait_state(resumed, "same", "exited")["returncode"] == 0
    assert resumed.launch("same", command, cwd=tmp_path)["state"] == "exited"
    assert output.read_text() == "x"
    with pytest.raises(ValueError, match="cannot change"):
        resumed.launch("same", [sys.executable, "-c", "pass"], cwd=tmp_path)


def test_managed_stop_waits_for_wrapper_and_actual_child(tmp_path):
    processes = Processes(tmp_path / "processes", sys.executable, SCRIPT)
    processes.launch("sleep", [sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path)
    wait_state(processes, "sleep", "running")
    child_file = processes.path("sleep").with_suffix(".child.json")
    deadline = time.monotonic() + 10
    while not child_file.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert child_file.exists()
    child = json.loads(child_file.read_text())
    assert identity_alive(child)
    processes.stop("sleep")
    assert not identity_alive(child)
    assert all(process_identity(pid) is None for pid in processes.actual_pids())


def test_unknown_launch_intent_is_pending_without_spawning(tmp_path):
    processes = Processes(tmp_path / "processes", sys.executable, SCRIPT)
    spec = {
        "command": ["false"],
        "cwd": str(tmp_path),
        "environment": {},
        "kind": "process",
        "receipt": str(tmp_path / "missing.exit"),
        "identity": str(tmp_path / "missing.pid"),
    }
    atomic_json(processes.path("unknown"), spec)
    assert processes.launch("unknown", ["false"], cwd=tmp_path)["state"] == "unresolved"


def test_source_counters_include_failed_attempts_queries_and_prior_updates_once(tmp_path):
    queue = minimal(tmp_path)
    journal = queue.output / "episodes/E0.attempt1/recording.journal.jsonl"
    append(
        journal,
        [{"event": "warmup_attempt"}, {"event": "before_action"}, {"event": "before_action"}],
    )
    with sqlite3.connect(queue.query_ledger) as db:
        db.execute("CREATE TABLE queries (status TEXT)")
        db.executemany("INSERT INTO queries VALUES (?)", [("failed",), ("reserved",)])
    learner = queue.learner_dir / "learner-ledger.jsonl"
    append(
        learner,
        [
            {
                "event": "update_reserved",
                "phase": "engineering",
                "origin": "workstation_cpu_preflight",
                "receipt": "sha256:prior",
            }
        ]
        * 2
        + [{"event": "update_reserved", "phase": "engineering"}] * 3
        + [{"event": "update_reserved", "phase": "training"}] * 4,
    )
    first = queue.sync_counters()["usage"]
    assert first["policy_controls"] == 2
    assert first["warmup_controls"] == 1
    assert first["model_queries"] == 2
    assert first["engineering_updates"] == 5
    assert first["learning_updates"] == 4
    assert first["learner_updates"] == 9
    assert queue.sync_counters()["usage"] == first
    queue.state = json.loads(queue.state_path.read_text())
    assert queue.sync_counters()["usage"] == first
    append(journal, [{"event": "before_action"}])
    assert queue.sync_counters()["usage"]["policy_controls"] == 3


def test_partial_source_line_is_counted_once_after_completion_and_truncation_fails(tmp_path):
    queue = minimal(tmp_path)
    path = queue.output / "episodes/E0.attempt1/recording.journal.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"event":"before_action"')
    assert queue.sync_counters()["usage"]["policy_controls"] == 0
    with path.open("a") as stream:
        stream.write("}\n")
    assert queue.sync_counters()["usage"]["policy_controls"] == 1
    path.write_text("")
    with pytest.raises(ValueError, match="truncated"):
        queue.sync_counters()


def test_invalid_attempt_needs_repair_and_restarts_its_sampler_lane(tmp_path):
    queue = minimal(tmp_path)
    job = frozen_jobs()["engineering_jobs"][0]
    queue.ledger.reserve_start("E0")
    queue.ledger.finish_start("E0", 1, valid=False)
    queue.state["episodes"]["E0"] = {"valid": False, "attempt": 1, "lane": 0}
    queue.state["server_generations"]["0"] = "server0.collection"
    with pytest.raises(AdmissionClosed, match="recorded repair"):
        queue.start_episode(job)
    operations = []
    queue.repair = {
        "job_id": "E0",
        "attempt": 2,
        "repair": "Fix renderer library path",
        "receipt_sha256": "repairhash",
    }
    queue.processes.stop = lambda key: operations.append(("stop", key))
    queue.server = lambda lane, generation: operations.append(("restart", lane, generation))
    entry = queue.start_episode(job)
    assert entry["attempt"] == 2
    assert operations == [("stop", "server0.collection"), ("restart", 0, "repair-E0-2")]
    assert queue.ledger.summary()["usage"]["replacements"] == 1


def test_valid_poor_episode_is_never_repeated(tmp_path):
    queue = minimal(tmp_path)
    queue.state["episodes"]["E0"] = {"valid": True, "success": False, "attempt": 1}
    assert queue.start_episode(frozen_jobs()["engineering_jobs"][0])["attempt"] == 1
    assert queue.ledger.summary()["usage"]["starts"] == 0


@pytest.mark.parametrize("warm,required", [(False, 3600), (True, 2700)])
def test_start_reserves_full_episode_and_preservation_before_charging(
    tmp_path, monkeypatch, warm, required
):
    queue = minimal(tmp_path)
    queue.stage = "pilot"
    if warm:
        queue.state["episodes"]["prior"] = {"valid": True, "lane": 0}
    deadline = dt.datetime.fromisoformat(queue.rental["allocation_deadline_utc"]).timestamp()
    monkeypatch.setattr(time, "time", lambda: deadline - required + 1)
    with pytest.raises(AdmissionClosed, match="reservation"):
        queue.start_episode(frozen_jobs()["engineering_jobs"][0])
    assert queue.ledger.summary()["usage"]["starts"] == 0
    monkeypatch.setattr(time, "time", lambda: deadline - required)
    queue.start_episode(frozen_jobs()["engineering_jobs"][0])
    assert queue.ledger.summary()["usage"]["starts"] == 1


def test_scientific_freeze_and_explicit_repair_amendment(tmp_path):
    source = tmp_path / "implementation.py"
    source.write_text("pass\n")
    old = {
        "authorization": {"user_launch": True},
        "jobs": frozen_jobs(),
        "caps": dict(CAPS),
        "money": {"total_usd": 15},
        "source_sha256": {source.name: sha256(source)},
    }
    path = tmp_path / "plan.json"
    atomic_json(path, old)
    assert verify_plan(tmp_path, path, "engineering") == old
    new = {
        **old,
        "source_sha256": {source.name: "updated"},
        "supersedes_sha256": "old",
        "amendment": {"type": "engineering_repair"},
    }
    receipt = {"old_freeze_sha256": "old", "new_freeze_sha256": "new", "repair": "Fix parser"}
    validate_amendment(old, new, receipt, "old", "new")
    changed = copy.deepcopy(new)
    changed["jobs"]["engineering_jobs"][0]["seed"] += 1
    with pytest.raises(ValueError, match="scientific"):
        validate_amendment(old, changed, receipt, "old", "new")
    with pytest.raises(ValueError, match="explicit"):
        validate_amendment(old, new, None, "old", "new")
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="Frozen source"):
        verify_plan(tmp_path, path, "engineering")


def test_engineering_and_preservation_clocks_are_both_enforced(tmp_path, monkeypatch):
    queue = minimal(tmp_path)
    start = dt.datetime.fromisoformat(queue.rental["created_utc"]).timestamp()
    monkeypatch.setattr(time, "time", lambda: start + 5401)
    with pytest.raises(AdmissionClosed):
        queue.guard()
    queue.stage = "pilot"
    end = dt.datetime.fromisoformat(queue.rental["allocation_deadline_utc"]).timestamp()
    monkeypatch.setattr(time, "time", lambda: end - 1199)
    with pytest.raises(AdmissionClosed):
        queue.guard()


def fake_pilot(tmp_path, *, transitions=250):
    queue = minimal(tmp_path)
    queue.stage = "pilot"
    summary = queue.output / "stage-a-summary.json"
    atomic_json(summary, {"admit": True})
    queue.plan.update(
        stage_a_summary_sha256=sha256(summary), actor_evaluation_mode="cpu_published_actor"
    )
    status = {"phase": "training", "warmup_complete": False, "actor_version": 0}
    operations = []
    completed_episodes = 0

    def http(key, route, body):
        nonlocal completed_episodes
        operations.append((route, copy.deepcopy(body)))
        if route == "/append":
            completed_episodes += 3
            if completed_episodes % 6 == 0 and completed_episodes * transitions > 500:
                status["warmup_complete"] = True
            return {"new_transitions": 3 * transitions}
        if route == "/publish":
            status["actor_version"] += 1
        if route == "/export-actor":
            return {"path": "published-actor", "actor_version": status["actor_version"]}
        return {}

    def start(job, **kwargs):
        operations.append(("start", {**job, **kwargs}))
        directory = queue.output / job["job_id"]
        directory.mkdir()
        atomic_json(
            directory / "descriptor.json",
            {
                "job_id": job["job_id"],
                "order_index": int(job["job_id"][1:]) if job["stage"] == "training" else -1,
                "actor_version": kwargs["actor_version"],
            },
        )
        queue.state["episodes"][job["job_id"]] = {"valid": True, "directory": str(directory)}

    queue.server = lambda *args: None
    queue.learner = lambda phase: status
    queue.get_status = lambda: status.copy()
    queue.http = http
    queue.start_episode = start
    queue.wait_episodes = lambda ids: operations.append(("wait", list(ids)))
    queue.checkpoint = lambda name: queue.state.update(last_checkpoint=name)
    queue.wait = lambda *args: None
    queue.actor_server = lambda *args: None
    queue.sync_counters = lambda: {}
    return queue, operations


def test_pilot_collects_six_gaussian_episodes_then_matched_rounds_and_four_arm_blocks(tmp_path):
    queue, operations = fake_pilot(tmp_path)
    queue.pilot()
    starts = [row for name, row in operations if name == "start"]
    training = [row for row in starts if row["stage"] == "training"]
    assert len(training) == 24
    assert all(row["mode"] == "gaussian" for row in training[:6])
    assert all(row["mode"] == "actor" for row in training[6:])
    assert [row["actor_version"] for row in training[6:9]] == [0, 0, 0]
    assert [row["actor_version"] for row in training[9:12]] == [1, 1, 1]
    assert len([row for route, row in operations if route == "/train"]) == 6
    assert len([row for route, row in operations if route == "/publish"]) == 6
    assert queue.state["last_checkpoint"] == "episode24"
    evaluations = [row for row in starts if row["stage"] == "evaluation"]
    assert len(evaluations) == 48
    for block in range(12):
        assert {row["lane"] for row in evaluations[block * 4 : block * 4 + 4]} == {0, 1, 2, 3}
    assert all(row["actor_version"] == 6 for row in evaluations)
    assert any(route == "/export-actor" for route, _ in operations)


def test_insufficient_warmup_stops_after_twelve_without_learning_credit(tmp_path):
    queue, operations = fake_pilot(tmp_path, transitions=1)
    with pytest.raises(AdmissionClosed, match="twelve"):
        queue.pilot()
    starts = [row for route, row in operations if route == "start"]
    assert len(starts) == 12
    assert all(row["mode"] == "gaussian" for row in starts)
    assert not any(route == "/train" for route, _ in operations)


def test_lost_http_response_remains_pending_without_duplicate_mutation(tmp_path):
    queue = minimal(tmp_path)
    queue.wait = lambda *args, **kwargs: {"returncode": 1}
    with pytest.raises(PendingOperation, match="no response"):
        queue.finish_http("train-round02")


def test_completed_source_update_recovers_a_lost_http_response_without_resending(tmp_path):
    queue = minimal(tmp_path)
    request = queue.output / "http/train-round02.request.json"
    atomic_json(request, {"url": queue.actor_url + "/train", "body": {"round_id": "round02"}})
    append(
        queue.learner_dir / "learner-ledger.jsonl",
        [{"event": "train_completed", "round_id": "round02", "updates": 15000}],
    )
    queue.wait = lambda *args, **kwargs: {"returncode": 1}
    result = queue.finish_http("train-round02")
    assert result["updates"] == 15000
    assert (queue.output / "http/train-round02.response.json").is_file()


def test_engineering_waits_for_native_responses_before_learner_then_freezes_overlap(tmp_path):
    queue = minimal(tmp_path)
    calls = []
    ready = False

    def server(lane):
        calls.append(("server", lane))
        queue.state["server_generations"][str(lane)] = f"server{lane}.collection"

    def start(job):
        calls.append(("start", job["job_id"]))
        queue.state["episodes"][job["job_id"]] = {"valid": None}

    def wait_episodes(ids):
        for job_id in ids:
            queue.state["episodes"][job_id]["valid"] = True
            queue.state["episodes"][job_id]["finished_utc"] = "2026-10-25T10:02:00+00:00"

    def tick():
        nonlocal ready
        ready = True
        calls.append(("native_responses", True))

    def learner(phase):
        assert ready
        assert queue.state["episodes"]["E3"]["valid"]
        calls.append(("learner", phase))

    queue.server, queue.start_episode, queue.wait_episodes = server, start, wait_episodes
    queue.first_response = lambda job_id: job_id == "E3" or ready
    queue.tick, queue.learner = tick, learner
    queue.poll_episodes = lambda: None
    queue.processes.stop = lambda key: calls.append(("stop", key))
    queue.processes.spec = lambda key: {"created_utc": "2026-10-25T10:00:00+00:00"}
    queue.processes.status = lambda key: {"finished_utc": "2026-10-25T10:01:00+00:00"}
    queue.profile = lambda: calls.append(("profile", None))
    queue.finish_http = lambda key: {
        "median_update_seconds": 0.001,
        "source_parity": {"maximum_state_difference": 0},
    }
    queue.checkpoint = lambda name: {"path": name}
    queue.cpu_actor_parity = lambda: {"valid": True, "median_cpu_seconds": 0.003}
    queue.authority = lambda: {"gate": {"valid": True, "admit": False, "reason": "absent"}}
    queue.sync_counters = lambda: {}
    summary = queue.engineering()
    assert calls[:2] == [("server", 3), ("start", "E3")]
    assert calls.index(("stop", "server3.collection")) < calls.index(("learner", "engineering"))
    assert calls.index(("native_responses", True)) < calls.index(("learner", "engineering"))
    assert summary["overlapping_jobs_at_profile"] == ["E0", "E1", "E2"]
    assert not summary["admit"]
    assert not any(name == "start" and value in ("E4", "E5") for name, value in calls)


def test_round_sampling_mode_and_actor_version_survive_mid_round_resume(tmp_path):
    queue, operations = fake_pilot(tmp_path)
    queue.state["rounds"] = {"round00": {"warmup": True, "actor_version": 0}}
    queue.state["completed_rounds"] = {f"round{i:02d}": True for i in range(1, 8)}
    queue.get_status = lambda: {"phase": "training", "warmup_complete": True, "actor_version": 0}
    queue.training_phase()
    starts = [row for name, row in operations if name == "start"]
    assert len(starts) == 3
    assert all(row["mode"] == "gaussian" for row in starts)
    assert not any(route == "/train" for route, _ in operations)


def test_http_error_body_is_preserved_immutably_and_failure_is_never_success(tmp_path, monkeypatch):
    path = tmp_path / "profile.request.json"
    output = tmp_path / "profile.response.json"
    atomic_json(
        path,
        {"url": "http://127.0.0.1:9009/profile", "body": {"parity": True}, "output": str(output)},
    )
    body = {"error": "ValueError", "message": "Bridge differs from direct author update: 0.01"}
    calls = []

    def failure(*args, **kwargs):
        calls.append((args, kwargs))
        raise urllib.error.HTTPError(
            "http://127.0.0.1:9009/profile",
            409,
            "Conflict",
            {},
            io.BytesIO(json.dumps(body).encode()),
        )

    monkeypatch.setattr(urllib.request, "urlopen", failure)
    with pytest.raises(urllib.error.HTTPError):
        http_worker(path)
    assert len(calls) == 1
    assert not output.exists()
    error_path = tmp_path / "profile.error.json"
    receipt = json.loads(error_path.read_text())
    assert receipt["http_status"] == 409
    assert receipt["body"] == body
    assert receipt["body_is_json"] is True
    assert receipt["request_sha256"] == sha256(path)
    original = error_path.read_bytes()
    with pytest.raises(urllib.error.HTTPError):
        http_worker(path)
    assert error_path.read_bytes() == original
    body["message"] = "A different diagnostic error"
    with pytest.raises(ValueError, match="immutable HTTP error"):
        http_worker(path)
    assert error_path.read_bytes() == original


def test_non_json_http_failure_is_recorded_without_retry(tmp_path, monkeypatch):
    path = tmp_path / "request.json"
    output = tmp_path / "reply.json"
    atomic_json(path, {"url": "http://127.0.0.1:9009/profile", "body": {}, "output": str(output)})

    def failure(*args, **kwargs):
        raise urllib.error.HTTPError(
            "http://127.0.0.1:9009/profile",
            503,
            "Unavailable",
            {},
            io.BytesIO(b"service unavailable"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", failure)
    with pytest.raises(urllib.error.HTTPError):
        http_worker(path)
    receipt = json.loads((tmp_path / "reply.error.json").read_text())
    assert receipt["http_status"] == 503
    assert receipt["body"] == "service unavailable"
    assert receipt["body_is_json"] is False
    assert not output.exists()


def test_profile_recovery_requires_exact_request_fingerprint_and_operation_identity(tmp_path):
    queue = minimal(tmp_path)
    request = {
        "path": "recorded.npz",
        "sha256": "inputs",
        "updates": 128,
        "parity": True,
        "operation_id": "engineering-profile-repair-01",
    }
    path = queue.output / "http/profile.request.json"
    atomic_json(path, {"url": queue.actor_url + "/profile", "body": request})

    def fingerprint(body):
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    queue.get_status = lambda: {"engineering_updates": 136}
    ledger = queue.learner_dir / "learner-ledger.jsonl"
    append(
        ledger,
        [
            {"event": "profile_completed", "timed_updates": 128, "source_parity": {}},
            {
                "event": "profile_completed",
                "timed_updates": 0,
                "source_parity": {},
                "request_sha256": fingerprint({**request, "updates": 0}),
            },
            {
                "event": "profile_completed",
                "timed_updates": 128,
                "source_parity": {},
                "request_sha256": fingerprint({**request, "operation_id": "old-profile"}),
            },
        ],
    )
    assert queue.recover_http("profile") is None
    append(
        ledger,
        [
            {
                "event": "profile_completed",
                "timed_updates": 128,
                "source_parity": {"maximum_state_difference": 0.0},
                "request_sha256": fingerprint(request),
            }
        ],
    )
    recovered = queue.recover_http("profile")
    assert recovered["timed_updates"] == 128
    assert recovered["request_sha256"] == fingerprint(request)
    assert recovered["status"]["engineering_updates"] == 136


def test_known_http_failure_cannot_be_recovered_as_completed_profile(tmp_path):
    queue = minimal(tmp_path)
    request = {"updates": 128, "parity": True, "operation_id": "failed-profile"}
    path = queue.output / "http/profile.request.json"
    atomic_json(path, {"url": queue.actor_url + "/profile", "body": request})
    fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    append(
        queue.learner_dir / "learner-ledger.jsonl",
        [
            {
                "event": "profile_completed",
                "timed_updates": 128,
                "source_parity": {},
                "request_sha256": fingerprint,
            }
        ],
    )
    atomic_json(queue.output / "http/profile.error.json", {"http_status": 409})
    assert queue.recover_http("profile") is None
    with pytest.raises(PendingOperation, match="preserved failure"):
        queue.finish_http("profile")
