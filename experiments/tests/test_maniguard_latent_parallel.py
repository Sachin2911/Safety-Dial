"""Exercise real concurrent children, saved-handle recovery and one global budget."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from helpers import maniguard_latent_parallel as parallel
from helpers import maniguard_latent_queue as serial

ROOT = Path(__file__).resolve().parents[2]


def paired(tmp_path, monkeypatch):
    root = tmp_path / "project"
    python = root / "data/maniguard/envs/tools/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    script = root / "experiments/scripts/maniguard_latent_worker.py"
    script.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "experiments/scripts/maniguard_latent_worker.py", script)
    monkeypatch.setattr(parallel, "ROOT", root)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"workers": 2, "physical_cutoff_utc": "2099-01-01T00:00:00Z"}))
    q = parallel.PairedPhysicalQueue(tmp_path / "run", protocol)
    jobs = [{"id": f"j{i}", "worker": i, "stage": "development", "engineering": False}
            for i in range(2)]
    q.plan(jobs)

    def command(job, output):
        code = ("import json,os,time;from pathlib import Path;"
                f"p=Path({str(output)!r});p.mkdir();started=time.time();time.sleep(.5);"
                "(p/'evidence.json').write_text(json.dumps({'started':started,'finished':time.time(),"
                "'device':os.environ['CUDA_VISIBLE_DEVICES']}))")
        return [sys.executable, "-c", code]

    monkeypatch.setattr(q, "command", command)
    monkeypatch.setattr(serial, "read_episode", lambda *a, **k: {"score": 0})
    monkeypatch.setattr(parallel, "read_episode", lambda *a, **k: {"score": 0})
    return q, protocol


def test_concurrent_processes_resume_both_handles_without_extra_physics(tmp_path, monkeypatch):
    q, protocol = paired(tmp_path, monkeypatch)
    for name in ["j0", "j1"]:
        q.launch(q.state["jobs"][name])
    handles = list(q.handles.values())
    q.lock.close()
    resumed = parallel.PairedPhysicalQueue(tmp_path / "run", protocol)

    def duplicate(*args, **kwargs):
        pytest.fail("A saved live worker must never be relaunched")

    monkeypatch.setattr(subprocess, "Popen", duplicate)
    resumed.plan([{"id": "previous-wave", "worker": 0, "stage": "engineering", "engineering": True}])
    resumed.state["jobs"]["previous-wave"].update(status="completed", output=str(tmp_path))
    assert resumed.evaluate_batch(["previous-wave"]) == [{"score": 0}]
    assert resumed.evaluate_batch(["j0", "j1"]) == [{"score": 0}, {"score": 0}]
    evidence = [json.loads((Path(resumed.state["jobs"][name]["output"]) / "evidence.json").read_text())
                for name in ["j0", "j1"]]
    assert max(row["started"] for row in evidence) < min(row["finished"] for row in evidence)
    assert {row["device"] for row in evidence} == {"0", "1"}
    assert len(resumed.state["attempts"]) == 2
    assert all(row["status"] == "completed" for row in resumed.state["jobs"].values())
    for handle in handles:
        handle.wait()


def test_wave_cannot_overbook_global_budget_or_same_device(tmp_path, monkeypatch):
    q, _ = paired(tmp_path, monkeypatch)
    q.state["attempts"] = [{"engineering": False} for _ in range(63)]
    with pytest.raises(RuntimeError, match="64-full"):
        q.evaluate_batch(["j0", "j1"])
    q.state["attempts"] = []
    q.state["jobs"]["j1"]["definition"]["worker"] = 0
    with pytest.raises(ValueError, match="each declared worker"):
        q.evaluate_batch(["j0", "j1"])
