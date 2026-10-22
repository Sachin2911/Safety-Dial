"""CPU contract checks. Synthetic fixtures are never experimental evidence."""

import importlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_phase2_families import validate_scene_access
from helpers.maniguard_phase2_outcomes import derive_outcome
from helpers.maniguard_phase3_adapter import NEUTRAL, Controller, SafetyDial
from helpers.maniguard_phase3_budget import clock, guard, register, run_owned
from helpers.maniguard_phase3_results import audit, replay
from helpers.maniguard_phase3_search import (
    COMPARISON,
    GRID,
    Evolution,
    comparison_jobs,
    select_allocation,
    training_jobs,
)


def frame(joints=0, fingers=0.02):
    return SimpleNamespace(
        joints=np.full(7, joints, dtype=np.float32), fingers=np.full(2, fingers, dtype=np.float32)
    )


@pytest.mark.parametrize("genome", [NEUTRAL, (3, 3, 2, 0), (1, 2, 1, 2)])
def test_zero_is_bitwise_identity_and_eight_action_schedule(genome):
    adapter = SafetyDial(Controller.from_genome(genome), 0)
    p = np.array([1.3, -2.4, 2.8, 1e-5, -0.0, 1.0, -3.0, -1.0], dtype=np.float32)
    assert adapter.horizon == 8
    for _ in range(20):
        action, _ = adapter.act(p, frame())
        assert action.tobytes() == p.tobytes()
        adapter.applied(np.clip(action, -0.5, 0.5))
    adapter.reset()
    assert adapter.last_executed is None and adapter.previous_fingers is None


def test_neutral_exact_at_every_dial():
    for d in (0, 0.5, 1):
        adapter = SafetyDial(Controller(), d)
        p = np.linspace(-3, 3, 8, dtype=np.float32)
        a, _ = adapter.act(p, frame())
        assert a.tobytes() == p.tobytes()
        assert adapter.horizon == 8


def test_actual_clipped_history_and_measured_cap_reference():
    adapter = SafetyDial(Controller(None, None, 0.5, 4), 1)
    p = np.ones(8, dtype=np.float32)
    a, _ = adapter.act(p, frame())
    np.testing.assert_array_equal(a[:7], 0.5)
    adapter.applied(np.clip(a, -0.1, 0.1))
    a, e = adapter.act(p, frame(0.3))
    np.testing.assert_allclose(a[:7], 0.55)
    np.testing.assert_allclose(e["reference"], 0.1)
    capped = SafetyDial(Controller(0.04, 0.02, 0.5, 4), 1)
    a, _ = capped.act(p, frame(0.3))
    np.testing.assert_allclose(a[:7], 0.34)
    assert a[-1] == p[-1]


def test_finger_heuristic_and_disabled_cap():
    adapter = SafetyDial(Controller(None, 0.02, 1.0, 8), 1)
    p = np.ones(8, dtype=np.float32)
    p[-1] = -1
    a, e = adapter.act(p, frame())
    assert e["grasp_evidence"] is False and e["step_limit"] is None
    np.testing.assert_array_equal(a, p)
    adapter.applied(a)
    a, e = adapter.act(p, frame())
    assert e["grasp_evidence"] is True and e["step_limit"] == 0.02
    np.testing.assert_allclose(a[:7], 0.02)
    adapter.applied(a)
    _, e = adapter.act(p, frame(fingers=0.04))
    assert e["grasp_evidence"] is False


@pytest.mark.parametrize("horizon,mid", [(4, 6), (8, 8), (16, 12)])
def test_midpoint_fixed_input_interpolation_and_horizons(horizon, mid):
    cfg = Controller(0.04, 0.02, 0.5, horizon)
    p = np.ones(8, dtype=np.float32)
    full, _ = SafetyDial(cfg, 1).act(p, frame())
    adapter = SafetyDial(cfg, 0.5)
    half, _ = adapter.act(p, frame())
    np.testing.assert_array_equal(half[:7], p[:7] + 0.5 * (full[:7] - p[:7]))
    assert half[-1] == p[-1] and adapter.horizon == mid
    with pytest.raises(AttributeError):
        adapter.dial = 0.8


@pytest.mark.parametrize("dial", [-1, 2, float("nan"), float("inf"), True])
def test_invalid_dial(dial):
    with pytest.raises(ValueError):
        SafetyDial(Controller(), dial)


def test_reject_invalid_configuration_and_nonfinite_actions():
    for genome in ((0, 0, 0), (4, 0, 0, 0), (True, 0, 0, 0)):
        with pytest.raises(ValueError):
            Controller.from_genome(genome)
    with pytest.raises(ValueError):
        Controller(free_step_limit=float("inf"))
    with pytest.raises(ValueError):
        SafetyDial(Controller(), 0).act(np.full(8, np.nan), frame())
    adapter = SafetyDial(Controller(), 0)
    adapter.act(np.ones(8), frame())
    with pytest.raises(ValueError):
        adapter.act(np.ones(8), frame())


def valid_outcomes(scores):
    return [
        [{"valid": True, "full_episode": True, "dial": 1, "U": u} for _ in range(2)] for u in scores
    ]


def test_seed_replay_elite_reevaluation_stable_ties_and_final_population():
    a, b = Evolution(), Evolution()
    p = a.ask()
    assert p == b.ask() and p[0]["genome"] == list(NEUTRAL)
    assert len({tuple(x["genome"]) for x in p}) == 4 and len(GRID) == 144
    ranking = a.tell(p, valid_outcomes([1, 1, 0, -2]))
    assert ranking["order"] == [0, 1, 2, 3]
    next_population = a.ask()
    assert next_population[0]["genome"] == p[0]["genome"]
    assert next_population[0]["kind"] == "elite_reevaluation"
    assert len({tuple(x["genome"]) for x in next_population}) == 4
    ranking = a.tell(next_population, valid_outcomes([-2, 0, 1, 0]))
    finalist = a.finalist(next_population, ranking)
    assert finalist["slot"] == 2 and finalist["final_population"] == 1
    invalid = valid_outcomes([0, 0, 0, 0])
    invalid[2][1]["valid"] = False
    with pytest.raises(ValueError):
        a.tell(next_population, invalid)


@pytest.mark.parametrize("size,populations,reps", [(48, 3, 4), (28, 2, 2)])
def test_exact_allocation_shared_once_and_scene_seed_matching(size, populations, reps):
    rng = np.random.default_rng(20261019)
    population = Evolution().ask()
    jobs = [j for g in range(populations) for j in training_jobs(g, population, rng)]
    validation = comparison_jobs(size, population[0]["genome"], rng)
    assert len(jobs) + len(validation) == size
    assert len({j["episode_id"] for j in jobs + validation}) == size
    for j in range(2):
        for r in range(reps):
            block = [v for v in validation if v["block"] == f"v{j}-r{r}"]
            assert {v["dial"] for v in block} == {0, 0.5, 1}
            assert {v["seed"] for v in block} == {2000 + 10 * j + r}
    for scene in COMPARISON:
        validate_scene_access("clutter", scene, "validation")
        with pytest.raises(ValueError):
            validate_scene_access("clutter", scene, "train")


def test_timing_selector_has_no_outcome_inputs_and_refuses_overrun():
    assert select_allocation([300.0] * 8, 20000)["variant"] == 48
    assert select_allocation([300.0] * 8, 10000)["variant"] == 28
    assert select_allocation([300.0] * 8, 8000)["variant"] is None
    assert select_allocation([500.0] * 8, 26000)["t_plan"] == 625
    with pytest.raises(ValueError):
        select_allocation([{"duration": 300, "U": 1}] * 8, 20000)


def test_allocation_clock_never_resets():
    c = clock()
    assert c["hard_deadline_utc"] - c["started_utc"] == 28800
    assert clock(now=c["at"] + 600)["elapsed_seconds"] == c["elapsed_seconds"] + 600


def test_importing_base_study_does_not_start_legacy_trace(tmp_path, monkeypatch):
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"phase": "phase3_part1"}))
    monkeypatch.setenv("SAFETY_DIAL_MANIGUARD_JOB", str(path))
    import helpers.maniguard_runtime as base

    importlib.reload(base)
    assert base.STUDY is None and not list(tmp_path.glob("*.jsonl"))


def test_failed_attempt_is_not_zero_fitness(tmp_path):
    result = audit(tmp_path)
    assert result["valid"] is False and result["U"] is None and result["costs"] == {}


def test_native_requested_completion_and_raw_violation():
    native = {
        "status": "completed",
        "ltl_monitored": True,
        "ltl_violated": True,
        "success": True,
        "steps": 1,
        "counted_violation": False,
    }
    obs = [{"step": i, "ap": {"fragile_ok": True}} for i in range(2)]
    goals = [
        {
            "step": 1,
            "success": True,
            "detail": {"held": True, "intersects": True, "mode": "held_intersection"},
        }
    ]
    outcome = derive_outcome("clutter", native, obs, goals)
    assert outcome["valid"] and outcome["C_req"] and outcome["V"] and outcome["U"] == -2
    assert derive_outcome("clutter", native, obs[1:], goals)["valid"] is False


def test_timeout_kills_owned_subprocess_tree_and_records_cost(tmp_path):
    child_file = tmp_path / "child.json"
    code = (
        "import subprocess,time,json; "
        "p=subprocess.Popen(['sleep','60']); "
        f"open({str(child_file)!r},'w').write(json.dumps(p.pid)); time.sleep(60)"
    )
    result = run_owned([sys.executable, "-c", code], tmp_path / "attempt", "episode", timeout=0.3)
    assert result["status"] == "deadline-interrupted" and result["elapsed_seconds"] >= 0.3
    pid = json.loads(child_file.read_text())
    # A zombie is terminated; the container init is responsible for reaping it.
    stat = Path(f"/proc/{pid}/stat")
    assert not stat.exists() or stat.read_text().split(") ")[1].split()[0] == "Z"


def test_checked_source_anchors(tmp_path):
    upstream = Path(__file__).resolve().parents[2] / "third_party/ManiGuard"
    if not upstream.is_dir():
        pytest.skip("Pinned upstream checkout not present")
    from helpers.maniguard_phase3_source import prepare_runtime

    path = prepare_runtime(tmp_path)
    source = path.read_text()
    assert source.count("from helpers.maniguard_phase3_runtime import STUDY") == 1
    assert "STUDY.start(scene_info, obs, env, robot, monitor)" in source
    assert "STUDY.clip_action(step_idx, ctrl_action, action_space)" in source
    compile(source, str(path), "exec")


def test_noop_mutations_have_bounded_uniform_fallback():
    class NoopRng:
        def integers(self, high):
            return 0

        def random(self):
            return 0.0

    engine = Evolution()
    engine.parents = [[0, 0, 0, 0], [0, 0, 0, 1]]
    engine.rng = NoopRng()
    population = engine.ask()
    assert len({tuple(r["genome"]) for r in population}) == 4
    assert all(r["fallback"] and r["draws"] == 100 for r in population[1:])


@pytest.mark.parametrize("length,horizon", [(3, 16), (16, 4), (16, 8), (16, 16)])
def test_query_schedule_and_short_exhaustion_replay(length, horizon):
    genome = [0, 0, 0, [4, 8, 16].index(horizon)]
    job = {"genome": genome, "dial": 1}
    adapter = SafetyDial(Controller.from_genome(genome), 1)
    trace = [{"type": "observed", "step": 0, "monitor": {"valid": True, "error": None}}]
    chunk = np.ones((length, 8), dtype=np.float32)
    chunk[:, -1] = -1
    steps = 35
    index = 0
    for step in range(steps):
        if index == 0:
            adapter.on_query(None)
            trace += [
                {"type": "query", "step": step},
                {"type": "chunk", "step": step, "actions": chunk.tolist()},
            ]
        f = frame()
        action, event = adapter.act(chunk[index], f)
        trace.append(
            {
                "type": "proposal",
                "step": step,
                "chunk_index": index,
                "event": event,
                "original": chunk[index].tolist(),
                "adapted": action.tolist(),
                "frame": {"joints": f.joints.tolist(), "fingers": f.fingers.tolist()},
            }
        )
        trace.append(
            {
                "type": "action_space",
                "step": step,
                "low": [-2.0] * 8,
                "high": [2.0] * 8,
                "dtype": "float32",
                "clipped": action.tolist(),
            }
        )
        adapter.applied(action)
        trace += [
            {"type": "executed", "step": step, "action": action.tolist()},
            {"type": "observed", "step": step + 1, "monitor": {"valid": True, "error": None}},
            {"type": "goal", "step": step + 1},
        ]
        index = (index + 1) % min(horizon, length)
    assert replay(job, trace, steps)["query_count"] == int(np.ceil(steps / min(horizon, length)))
    trace[-2]["step"] = 90
    with pytest.raises(ValueError):
        replay(job, trace, steps)


def test_independent_guard_forces_cleanup_without_coordinator(tmp_path):
    now = time.time()
    stamps = {
        "started_utc": now - 28800 + 0.5,
        "physical_deadline_utc": now - 3600 + 0.5,
        "hard_deadline_utc": now + 0.5,
    }
    allocation = tmp_path / "allocation.json"
    allocation.write_text(
        json.dumps(
            {
                "gpus": 1,
                **{
                    k: datetime.fromtimestamp(v).astimezone().isoformat() for k, v in stamps.items()
                },
            }
        )
    )
    process = subprocess.Popen(["sleep", "60"], start_new_session=True)
    register(tmp_path, process, "episode")
    try:
        guard(tmp_path, allocation, interval=0.03)
        assert process.wait(timeout=1) != 0
        assert (tmp_path / "deadline-events.jsonl").is_file()
    finally:
        if process.poll() is None:
            process.kill()


def test_native_unbounded_action_space_is_logged_without_changing_commands(monkeypatch):
    monkeypatch.delenv("SAFETY_DIAL_MANIGUARD_JOB", raising=False)
    from helpers.maniguard_phase3_runtime import Part1Study

    study = object.__new__(Part1Study)
    records = []
    study.write = lambda kind, **values: records.append({"type": kind, **values})
    space = SimpleNamespace(
        low=np.full(8, -np.inf, dtype=np.float32), high=np.full(8, np.inf, dtype=np.float32)
    )
    action = np.linspace(-3, 3, 8, dtype=np.float32)
    result = study.clip_action(0, action, space)
    assert result.tobytes() == action.tobytes()
    record = records[0]
    assert record["low"] == ["-inf"] * 8 and record["high"] == ["inf"] * 8
    json.dumps(record, allow_nan=False, default=lambda x: x.tolist())
    np.testing.assert_array_equal(np.asarray(record["low"], dtype=record["dtype"]), space.low)


def test_engineering_retry_identifiers_stay_fixed_across_three_actions(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/maniguard_phase3_pilot.py"
    spec = importlib.util.spec_from_file_location("part1_pilot_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    jobs = []
    monkeypatch.setattr(module, "ensure_policy", lambda *args: None)

    def launch(directory, job):
        jobs.append(job)
        return directory / job["episode_id"], {}, 0

    monkeypatch.setattr(module, "launch", launch)
    monkeypatch.setattr(module, "audited", lambda *args: {"engineering_valid": True})
    module.engineering(tmp_path, 2)
    assert [j["episode_id"] for j in jobs] == [f"engineering-r2-{i}" for i in range(3)]
