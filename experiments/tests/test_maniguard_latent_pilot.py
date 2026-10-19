"""Exercise matched accounting, interruption recovery and finalist protection."""

import json
from pathlib import Path

import pytest

from helpers.maniguard_latent_pilot import Pilot


class Queue:
    def __init__(self):
        self.jobs, self.completed = {}, {}
        self.interrupt_after = None

    def plan(self, jobs):
        for job in jobs:
            if job["id"] in self.jobs:
                assert job == self.jobs[job["id"]]
            self.jobs[job["id"]] = job

    def evaluate(self, name, **kwargs):
        if name not in self.completed:
            if self.interrupt_after == len(self.completed):
                raise RuntimeError("injected coordinator interruption")
            job = self.jobs[name]
            # All candidates tie. Identifier selection and paid reevaluation must
            # remain deterministic without any independent-outcome feedback.
            if job["stage"] == "independent_audit":
                assert (Path(job["controller"]).parent / "sealed-finalists.json").exists()
            self.completed[name] = {"score": 0}
        return self.completed[name]


def protocol(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({
        "status": "capacity_selected_frozen", "family": "jar",
        "capacity": {"fits": True, "episode_seconds": 180},
        "training_panels": [[{"scene": "task_0019/base", "seed": 100 + 10 * g},
                             {"scene": "task_0009/base", "seed": 101 + 10 * g}]
                            for g in range(3)],
        "audit_blocks": [{"scene": f"task_{i:04d}/base", "seed": 200 + i,
                          "order": ["native", "fixed", "ea", "random"]}
                         for i in range(4)]}))
    return path


def test_matched_search_is_identical_after_partial_generation_resume(tmp_path):
    path = protocol(tmp_path)
    interrupted = tmp_path / "interrupted"
    uninterrupted = tmp_path / "uninterrupted"
    interrupted.mkdir()
    uninterrupted.mkdir()
    q = Queue()
    q.interrupt_after = 13
    pilot = Pilot(interrupted, path, queue=q)
    with pytest.raises(RuntimeError, match="interruption"):
        pilot.run_search()
    assert len(q.completed) == 13
    assert not (interrupted / "search/sealed-finalists.json").exists()
    q.interrupt_after = None
    resumed = Pilot(interrupted, path, queue=q)
    resumed.run_search()
    resumed.run_audit()
    full = Pilot(uninterrupted, path, queue=Queue())
    full.run_search()
    assert resumed.state["optimizers"] == full.state["optimizers"]
    assert len(q.completed) == 56
    assert sum(j["stage"] == "shared_initial" for j in q.jobs.values()) == 8
    for method in ["ea", "random"]:
        assert sum(name.startswith(method + "-g") for name in q.completed) == 16
        search = resumed.state["optimizers"][method]
        assert len(search["candidates"]) == 12
        assert [c["parent"] for c in search["candidates"][4::4]] == [
            "initial-0", f"{method}-g1-0"]
    assert resumed.state["finalists"]["fixed"]["candidate_id"] == "initial-1"
    # Resuming a completed run must not acquire more physical outcomes.
    resumed.run_search()
    resumed.run_audit()
    assert len(q.completed) == 56


def test_audit_cannot_run_before_sealing_and_protocol_cannot_change(tmp_path):
    path = protocol(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    pilot = Pilot(run, path, queue=Queue())
    with pytest.raises(ValueError, match="Seal"):
        pilot.run_audit()
    changed = json.loads(path.read_text())
    changed["training_panels"][0][0]["seed"] += 1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="changed"):
        Pilot(run, path, queue=Queue())
