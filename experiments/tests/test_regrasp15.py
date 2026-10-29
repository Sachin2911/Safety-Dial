"""Task 15: gate, check, selection, the live hook and the independent replay validator."""

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from regrasp15_check import CheckConfig, Gate, features, project, select
from regrasp15_kinematics import eef_local, fit_base, yaw_matrix
from regrasp15_runtime import Runtime, candidate_seed
from regrasp15_sampler import candidate_noise
from regrasp15_validation import validate_recording
from tracking import SCORING, STATE_FIELDS

ROOT = Path(__file__).resolve().parents[2]
AP_NAMES = ["jar_closed", "jar_dropped", "jar_on_support", "jar_upright"]
YAW = -np.pi / 2
BASE = np.array([0.3, -0.2, 0.0])
Q0 = np.array([0.0, -0.4, 0.0, -2.1, 0.0, 1.8, 0.8])


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"experiments/scripts/{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def world_eef(q7):
    return BASE + yaw_matrix(YAW) @ eef_local(q7)


def chunk_from(q7, rate, gripper=1.0):
    steps = np.arange(1, 17)[:, None]
    chunk = np.zeros((16, 8), dtype=np.float32)
    chunk[:, :7] = q7 + rate * steps * np.array([1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    chunk[:, 7] = gripper
    return chunk


def test_gate_latches_closes_at_lift_and_always_gate_ignores_lift():
    closure = Gate(CheckConfig(mode="select", speed_limit_mps=0.1, candidates=1))
    always = Gate(CheckConfig(mode="select", speed_limit_mps=0.1, candidates=1, gate="always"))
    for step in range(30):
        on_support = step < 25
        for gate in (closure, always):
            gate.update(step, step >= 5, on_support, True, False)
        if step == 13:
            assert not closure.open and always.open
        if step == 14:
            assert closure.open and closure.latched_at == 14
    assert closure.closed_at == 25 and closure.reason == "lift" and not closure.open
    assert always.open and always.closed_at is None
    always.update(30, True, False, False, False)
    assert not always.open and always.reason == "violation"


def test_select_prefers_native_then_closest_passing_then_least_excess():
    q = Q0
    eef = world_eef(q)
    jar = eef + np.array([0.05, 0.0, -0.1])
    config = CheckConfig(mode="select", speed_limit_mps=0.1, candidates=3)
    slow, fast = chunk_from(q, 0.001), chunk_from(q, 0.03)
    assert features(fast, q, eef, jar, YAW, 0.22)["max_near_speed_mps"] > 0.1
    assert select(slow, [fast], q, eef, jar, YAW, config)[:2] == (0, "native_passes")
    choice, rule, *_ = select(fast, [chunk_from(q, 0.04), slow, chunk_from(q, 0.002)],
                              q, eef, jar, YAW, config)
    assert rule == "closest_passing" and choice == 3
    choice, rule, *_ = select(fast, [chunk_from(q, 0.05), chunk_from(q, 0.02)],
                              q, eef, jar, YAW, config)
    assert (choice, rule) == (2, "least_excess")
    result, scale, check = project(fast, q, eef, jar, YAW,
                                   CheckConfig(mode="project", speed_limit_mps=0.1))
    assert scale < 1 and check["max_near_speed_mps"] <= 0.1 + 1e-9
    np.testing.assert_array_equal(result[:, 7], fast[:, 7])


def test_candidate_noise_and_seed_are_reproducible():
    a, b = candidate_noise(7, 4), candidate_noise(7, 4)
    assert a.shape == (4, 16, 32) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, candidate_noise(8, 4))
    assert candidate_seed(5, 16) == candidate_seed(5, 16) != candidate_seed(5, 24)


class Recorder:
    def __init__(self, episode_seed):
        self.rows = []
        self.ap_names = AP_NAMES
        self.episode_seed = episode_seed


class Client:
    def __init__(self, k):
        self.k = k
        self.requests = []

    def infer(self, request):
        self.requests.append(request)
        seed = request["regrasp15"]["seed"]
        rng = np.random.default_rng(seed)
        q = request["observation/state"][:7].astype(np.float64)
        rates = rng.uniform(0.0005, 0.04, self.k)
        return {"candidates": np.stack([chunk_from(q, r) for r in rates]).astype(np.float64),
                "noise_seed": np.int64(seed), "decode_s": np.float64(0.01)}


def episode(tmp_path, monkeypatch, settings, n=60, lift_at=45):
    monkeypatch.setenv("SAFETY_DIAL_REGRASP_CONFIG", str(tmp_path / "adapter-config.json"))
    monkeypatch.setenv("SAFETY_DIAL_REGRASP_JOURNAL", str(tmp_path / "adapter.jsonl"))
    (tmp_path / "adapter-config.json").write_text(json.dumps(settings))
    cfg = {"action_frequency": 20, "execute_horizon": 8, "state_mode": "joint",
           "controller_preset": "joint_position_raw", "ik_eef_to_joint": False,
           "max_steps": 2000, "source_profile": None, "gripper_binarize": True,
           "camera_resolution": 256, "success_hold_steps": 10}
    t = np.arange(n + 1)[:, None]
    q = np.zeros((n + 1, 9), dtype=np.float32)
    q[:, :7] = Q0 + 0.2 * np.sin(t / 9.0 + np.arange(7))
    eef = np.stack([world_eef(row[:7].astype(np.float64)) for row in q]).astype(np.float32)
    jar = np.tile((eef[0] + np.array([0.06, 0.02, -0.12])).astype(np.float32), (n + 1, 1))
    ap = np.zeros((n + 1, 4), dtype=bool)
    ap[:, 0] = t[:, 0] >= 5
    ap[:, 2] = t[:, 0] < lift_at
    ap[:, 3] = True
    quaternion = np.array([0.0, 0.0, math.sin(YAW / 2), math.cos(YAW / 2)])
    robot = SimpleNamespace(default_arm="arm", arm_control_idx={"arm": np.arange(7)},
                            get_position_orientation=lambda: (BASE.copy(), quaternion))
    recorder = Recorder(episode_seed=4242)
    config = CheckConfig(**settings)
    client = Client(config.candidates)
    runtime = Runtime(SimpleNamespace(**cfg), robot, recorder)
    executed, journal_rows = [], [{"event": "observer_created"}]
    for step in range(0, n, 8):
        while len(recorder.rows) <= step:
            i = len(recorder.rows)
            recorder.rows.append({"step": np.int64(i), "q": q[i], "eef_pos": eef[i],
                                  "jar_pos": jar[i], "ap": ap[i]})
        index = step // 8
        native = chunk_from(q[step, :7].astype(np.float64), 0.03 if index % 2 else 0.001)
        runtime.emit("query_attempt", step=step)
        runtime.emit("query_returned", step=step)
        obs = {"observation/image_left": np.zeros((256, 256, 3), np.uint8),
               "observation/wrist_image": np.zeros((256, 256, 3), np.uint8),
               "observation/state": q[step, :8], "prompt": "jar", "episode_seed": 4242}
        chunk = runtime.choose(step, native, obs, client)
        executed.append(np.asarray(chunk, dtype=np.float32))
        journal_rows.append({"event": "proposal", "step": step})
    while len(recorder.rows) <= n:
        i = len(recorder.rows)
        recorder.rows.append({"step": np.int64(i), "q": q[i], "eef_pos": eef[i],
                              "jar_pos": jar[i], "ap": ap[i]})
    row = dict.fromkeys(SCORING)
    row.update(status="completed", ltl_monitored=True, steps=n, scene_name="task_0001/base",
               seed=150003, episode_seed=4242, success=False, ltl_violated=False,
               counted_violation=False, safety_evaluated=True)
    runtime.finish(row)
    journal_rows += [{"event": "applied", "step": s} for s in range(n)]
    journal_rows.append({"event": "finish", "result": row})
    chunks = np.stack(executed)
    raw = chunks[np.arange(n) // 8, np.arange(n) % 8]
    data = {key: np.zeros(n + 1) for key in STATE_FIELDS}
    data.update(step=np.arange(n + 1), q=q, dq=np.zeros_like(q), eef_pos=eef, jar_pos=jar,
                gripper_q=np.zeros((n + 1, 2)), policy_state=np.zeros((n + 1, 8)), ap=ap,
                ap_valid=np.ones(n + 1, dtype=bool), exec_step=np.arange(n),
                exec_proposal=np.arange(n) // 8, exec_offset=np.arange(n) % 8,
                exec_raw=raw, exec_ctrl=raw, exec_clipped=raw,
                proposal_step=np.arange(0, n, 8), proposal_chunk=chunks)
    meta = {"config": cfg, "result": row, "arm_control_idx": list(range(7)),
            "execute_horizon": 8, "action_frequency": 20, "ap_names": AP_NAMES,
            "recorder_overhead_s": 0.1, "episode_seed": 4242}
    records = tmp_path / "recordings"
    records.mkdir()
    path = records / "task_0001_base.npz"
    np.savez_compressed(path, **data, meta=json.dumps(meta))
    path.with_suffix(".journal.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in journal_rows))
    row_path = tmp_path / "results.jsonl"
    row_path.write_text(json.dumps(row) + "\n")
    sidecar = tmp_path / "task_0001_base_ltl.json"
    sidecar.write_text(json.dumps({"log": [
        {"step": i, "ap": dict(zip(AP_NAMES, map(bool, ap[i]), strict=True))}
        for i in range(n + 1)]}))
    return (path, row_path, sidecar, tmp_path / "adapter.jsonl", settings), client, chunks


SETTINGS = {
    "native": {"mode": "native", "gate": "closure"},
    "select": {"mode": "select", "speed_limit_mps": 0.1, "candidates": 6, "gate": "closure"},
    "project": {"mode": "project", "speed_limit_mps": 0.1, "gate": "closure"},
    "always": {"mode": "select", "speed_limit_mps": 0.1, "candidates": 6, "gate": "always"},
}


@pytest.mark.parametrize("arm", sorted(SETTINGS))
def test_live_hook_roundtrip_replays_every_decision(tmp_path, monkeypatch, arm):
    args, client, _ = episode(tmp_path, monkeypatch, SETTINGS[arm])
    result = validate_recording(*args)
    adapter = result["adapter"]
    assert result["queries"] == 8 and adapter["base_yaw_gap_rad"] < 1e-6
    if arm == "always":
        assert adapter["gated"] == 8 and adapter["gate_closed_step"] is None
    else:
        assert adapter["gate_latched_step"] == 14 and adapter["gate_closed_reason"] == "lift"
        assert adapter["gated"] == 4  # queries at 16, 24, 32 and 40
    if arm == "native":
        assert adapter["vetoes"] == 0 and not client.requests
    if arm in {"select", "always"}:
        assert adapter["vetoes"] > 0 and len(client.requests) == adapter["vetoes"]
        assert adapter["closest_passing"] + adapter["least_excess"] == adapter["vetoes"]
        assert all("episode_seed" in r and "regrasp15" in r for r in client.requests)
    if arm == "project":
        assert adapter["projected"] == adapter["vetoes"] > 0
    assert result["outcome"]["closure_latched_step"] == 14


def test_validator_rejects_a_changed_candidate_and_a_shut_gate_edit(tmp_path, monkeypatch):
    args, _, chunks = episode(tmp_path, monkeypatch, SETTINGS["select"])
    journal_path = args[3]
    original = journal_path.read_text()
    journal = [json.loads(x) for x in original.splitlines()]
    vetoed = next(e for e in journal if e["event"] == "decision" and "candidates" in e)
    choice = vetoed["choice"]
    vetoed["candidates"][max(choice - 1, 0)][0][0] += 0.25
    journal_path.write_text("".join(json.dumps(e) + "\n" for e in journal))
    with pytest.raises(ValueError):
        validate_recording(*args)
    journal_path.write_text(original)
    validate_recording(*args)
    data = dict(np.load(args[0]))
    data["proposal_chunk"] = data["proposal_chunk"].copy()
    data["proposal_chunk"][0, 0, 0] += 0.01  # Query 0 runs with the gate shut.
    data["exec_raw"] = data["proposal_chunk"][np.arange(60) // 8, np.arange(60) % 8]
    np.savez_compressed(args[0], **data)
    with pytest.raises(ValueError, match="gate was shut"):
        validate_recording(*args)


def test_base_fit_recovers_the_live_yaw():
    rng = np.random.default_rng(3)
    q = Q0 + rng.uniform(-0.3, 0.3, (50, 7))
    eef = np.stack([world_eef(row) for row in q])
    yaw, position, error = fit_base(q, eef)
    assert abs((yaw - YAW + np.pi) % (2 * np.pi) - np.pi) < 1e-6 and error < 1e-6
    np.testing.assert_allclose(position, BASE, atol=1e-6)


def test_patch_is_exact_and_idempotent(tmp_path):
    source = ROOT / "third_party/ManiGuard/maniguard/eval/benchmark.py"
    if not source.exists():
        pytest.skip("ManiGuard checkout is not present")
    patch = module("patch_regrasp15")
    patched = patch.patched(source.read_text())
    assert patched.count("action_adapter.choose(") == 1
    assert patched.count("action_adapter.finish(result)") == 2


def test_stage_c_choice_and_one_sided_contrast(tmp_path, monkeypatch):
    analyze = module("analyze_regrasp15")
    monkeypatch.setattr(analyze, "OUT", tmp_path)
    design = tmp_path / "design.json"
    design.write_text(json.dumps({"stage_c_seeds": list(range(12)), "stage_c_tasks": [0, 3]}))
    monkeypatch.setattr(analyze, "DESIGN", design)

    def validated(arm, task, strict, violation):
        return {"event": "validated", "arm": arm, "task": task, "stage": "stage_b",
                "outcome": {**dict.fromkeys(analyze.OUTCOMES, False), "strict_safe": strict,
                            "raw_violation": violation},
                "adapter": dict.fromkeys(analyze.ACTIVITY, 0)}

    rows = ([validated("native", 0, i < 2, i == 3) for i in range(8)]
            + [validated("select-0.10", 0, i < 4, False) for i in range(8)]
            + [validated("select-0.06", 0, i < 4, i < 1) for i in range(8)])
    (tmp_path / "attempts_stage_a.jsonl").write_text("")
    (tmp_path / "attempts_stage_b.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    analyze.decide()
    decision = json.loads((tmp_path / "stage-c-decision.json").read_text())
    assert decision["arm"] == "select-0.10" and decision["ranking"][0] == "select-0.10"
    stage_c = ([dict(validated("native", t, i < 3, False), stage="stage_c")
                for t in (0, 3) for i in range(24)]
               + [dict(validated("select-0.10", t, i < 15, False), stage="stage_c")
                  for t in (0, 3) for i in range(24)])
    result = analyze.contrast(stage_c, "select-0.10")
    assert result["difference"] > 0.4 and result["barnard_one_sided_p"] < 0.001
    worse = analyze.contrast([dict(r, arm={"native": "select-0.10", "select-0.10": "native"}[
        r["arm"]]) for r in stage_c], "select-0.10")
    assert worse["barnard_one_sided_p"] > 0.9


def test_stage_d_jobs_follow_the_declaration(tmp_path, monkeypatch):
    stage_d = module("stage_d_regrasp15")
    declaration = tmp_path / "decl.json"
    declaration.write_text(json.dumps({"arm": "project-0.10", "tasks": [0, 3], "seeds": [7, 8],
                                       "ceilings": {"simulator_starts": 250}}))
    monkeypatch.setattr(stage_d, "DECLARATION", declaration)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(stage_d.run_regrasp15, "OUT", out)
    queue = object.__new__(stage_d.StageD)
    queue.design = {"stage_a_jobs": [{"job": 0}], "arms": {"project-0.10": {}},
                    "ceilings": {"simulator_starts": 200}}
    queue.history = [{"event": "validated", "job": 0}]
    with pytest.raises(ValueError, match="after the stage C queue"):
        queue.stage_jobs()
    (out / "stage_c-complete.json").write_text("{}")
    jobs = queue.stage_jobs()
    assert [(j["job"], j["task"], j["seed"]) for j in jobs] == [
        (500, 0, 7), (501, 3, 7), (502, 0, 8), (503, 3, 8)]
    assert {j["arm"] for j in jobs} == {"project-0.10"}
