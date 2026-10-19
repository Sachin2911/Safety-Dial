"""Protect durable failure receipts, live-handle resumption and immutable jobs."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from helpers.maniguard_latent_queue import PhysicalQueue, process_identity

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("child_code", [0, 7])
def test_worker_persists_real_child_exit_status(tmp_path, child_code):
    request, receipt = tmp_path / "request.json", tmp_path / "receipt.json"
    request.write_text(json.dumps({"command": [sys.executable, "-c",
                                               f"raise SystemExit({child_code})"],
                                   "cwd": str(tmp_path)}))
    result = subprocess.run([sys.executable, str(ROOT / "experiments/scripts/maniguard_latent_worker.py"),
                             "--request", str(request), "--receipt", str(receipt)], check=False)
    saved = json.loads(receipt.read_text())
    assert saved["exit_code"] == child_code
    assert result.returncode == (0 if child_code == 0 else 1)
    assert saved["finished_epoch"] >= saved["started_epoch"]


def queue(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"physical_cutoff_utc": "2026-10-20T02:30:00Z"}))
    return PhysicalQueue(tmp_path / "run", protocol)


def test_queue_rejects_changed_job_after_resume(tmp_path):
    q = queue(tmp_path)
    job = {"id": "a", "family": "jar", "scene": "task_0019/base", "seed": 1}
    q.plan([job])
    with pytest.raises(ValueError):
        q.plan([{**job, "seed": 2}])
    assert json.loads(q.path.read_text())["jobs"]["a"]["definition"] == job


def test_live_handle_is_observed_without_relaunch(tmp_path, monkeypatch):
    q = queue(tmp_path)
    q.plan([{"id": "a"}])
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.1)"],
                              start_new_session=True)
    identity = process_identity(worker.pid)
    assert identity is not None
    q.state["jobs"]["a"].update(status="running", attempt_ids=["a-a0"])
    attempt = {"id": "a-a0", "process_identity": identity, "deadline_epoch": time.time() + 30}
    q.state["attempts"].append(attempt)
    q.save()
    monkeypatch.setattr(q, "_reconcile", lambda row, record: record)

    def duplicate(*args, **kwargs):
        pytest.fail("A live saved worker must never trigger an overlapping rollout")

    monkeypatch.setattr(subprocess, "Popen", duplicate)
    assert q.evaluate("a") is attempt
    worker.wait()
    assert process_identity(worker.pid) is None
    assert process_identity(os.getpid()) is not None


def test_colocated_jobs_require_declared_workers_and_separate_endpoints(tmp_path):
    q = queue(tmp_path)
    job = {"id": "a", "family": "clutter", "scene": "task_0039/base", "seed": 1,
           "controller": "controller.npz", "worker": 0}
    with pytest.raises(ValueError, match="declared"):
        q.command(job, tmp_path / "episode")
    q.protocol["workers"] = 2
    for worker in [0, 1]:
        assigned = {**job, "worker": worker}
        command = q.command(assigned, tmp_path / str(worker))
        assert command[command.index("--port") + 1] == str(18000 + worker)
        assert command[command.index("--simulation-device") + 1] == str(worker)
        assert command[command.index("--policy-device") + 1] == str(worker)
        assert q.environment(assigned)["CUDA_VISIBLE_DEVICES"] == str(worker)
