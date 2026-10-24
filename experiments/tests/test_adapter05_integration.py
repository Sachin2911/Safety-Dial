"""Exercise the live hook and the independent recording validator with known states."""

import importlib.util
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from adapter05_runtime import Runtime
from helpers.adapter05_validation import validate_adapter_recording
from helpers.execution_adapter import AdapterConfig
from helpers.tracking import SCORING, STATE_FIELDS

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"experiments/scripts/{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def fixture_recording(tmp_path, monkeypatch, mode):
    n = 11  # Cross a chunk boundary and finish partway through the next chunk.
    settings = asdict(AdapterConfig(mode=mode))
    settings_path = tmp_path / "adapter-config.json"
    settings_path.write_text(json.dumps(settings))
    monkeypatch.setenv("SAFETY_DIAL_ADAPTER_CONFIG", str(settings_path))
    monkeypatch.setenv("SAFETY_DIAL_ADAPTER_JOURNAL", str(tmp_path / "adapter.jsonl"))
    cfg = {
        "action_frequency": 20,
        "state_mode": "joint",
        "controller_preset": "joint_position_raw",
        "ik_eef_to_joint": False,
        "max_steps": 2000,
        "source_profile": "legacy",
        "gripper_binarize": True,
        "camera_resolution": 256,
        "success_hold_steps": 10,
    }
    indices = [6, 4, 2, 0, 1, 3, 5]
    q = np.arange((n + 1) * 9, dtype=np.float32).reshape(n + 1, 9) / 500
    dq = np.full_like(q, 0.36)
    robot = SimpleNamespace(
        default_arm="arm",
        arm_control_idx={"arm": np.array(indices)},
        get_joint_positions=lambda: q[robot.step],
        get_joint_velocities=lambda: dq[robot.step],
        step=0,
    )
    runtime = Runtime(SimpleNamespace(**cfg), robot)
    for _ in range(10):
        runtime.warmup_before(np.zeros(8))
        runtime.warmup_after()
    chunks = np.random.default_rng(9).uniform(-0.2, 0.2, (math.ceil(n / 8), 16, 8))
    raw = chunks[np.arange(n) // 8, np.arange(n) % 8]
    corrected = []
    for step, command in enumerate(raw):
        robot.step = step
        command = command.copy()
        command[-1] = np.sign(command[-1]) if abs(command[-1]) > 0.01 else -1.0
        corrected.append(runtime.apply(step, command))
        runtime.after_step(step)
    row = dict.fromkeys(SCORING)
    row.update(
        status="completed",
        ltl_monitored=True,
        steps=n,
        scene_name="task_0001/base",
        seed=390,
        success=False,
        counted_violation=False,
        safety_evaluated=True,
    )
    runtime.finish(row)
    data = {key: np.zeros(n + 1) for key in STATE_FIELDS}
    data.update(
        step=np.arange(n + 1),
        q=q,
        dq=dq,
        eef_pos=np.zeros((n + 1, 3)),
        gripper_q=np.zeros((n + 1, 2)),
        policy_state=np.zeros((n + 1, 8)),
        ap=np.zeros((n + 1, 1), dtype=bool),
        ap_valid=np.ones(n + 1, dtype=bool),
        exec_step=np.arange(n),
        exec_proposal=np.arange(n) // 8,
        exec_offset=np.arange(n) % 8,
        exec_raw=raw.astype(np.float32),
        exec_ctrl=np.asarray(corrected, dtype=np.float32),
        exec_clipped=np.asarray(corrected, dtype=np.float32),
        proposal_step=np.arange(0, n, 8),
        proposal_chunk=chunks.astype(np.float32),
    )
    meta = {
        "config": cfg,
        "result": row,
        "arm_control_idx": indices,
        "execute_horizon": 8,
        "action_frequency": 20,
        "ap_names": ["example"],
        "recorder_overhead_s": 0.1,
    }
    records = tmp_path / "recordings"
    records.mkdir()
    path = records / "base.npz"
    np.savez_compressed(path, **data, meta=json.dumps(meta))
    row_path = tmp_path / "results.jsonl"
    row_path.write_text(json.dumps(row) + "\n")
    sidecar = tmp_path / "base_ltl.json"
    sidecar.write_text(
        json.dumps({"log": [{"step": i, "ap": {"example": False}} for i in range(n + 1)]})
    )
    return path, row_path, sidecar


@pytest.mark.parametrize("mode", ["native", "fixed", "adaptive"])
def test_hook_roundtrip_and_validator_detect_corrupted_command(tmp_path, monkeypatch, mode):
    paths = fixture_recording(tmp_path, monkeypatch, mode)
    result = validate_adapter_recording(*paths)
    assert result["adapter_replay_steps"] == 11
    assert result["queries"] == 2
    assert result["adapter_mode"] == mode
    journal_path = tmp_path / "adapter.jsonl"
    journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
    next(e for e in journal if e["event"] == "command")["q"][0] += 0.1
    journal_path.write_text("".join(json.dumps(e) + "\n" for e in journal))
    with pytest.raises(AssertionError):
        validate_adapter_recording(*paths)


def test_frozen_panel_is_disjoint_and_balanced_across_lanes():
    plan = module("make_adapter05_plan").make_plan()
    assert len(plan["jobs"]) == 36
    heldout = {2, 5, 8, 14, 16, 23, 7, 18, 12}
    assert not heldout.intersection(j["task"] for j in plan["jobs"] + plan["engineering_jobs"])
    cells = {(j["task"], j["seed"], j["arm"]) for j in plan["jobs"]}
    assert len(cells) == 36
    for arm in plan["arms"]:
        for lane in range(2):
            assert sum(j["arm"] == arm and j["preferred_lane"] == lane for j in plan["jobs"]) == 6


def test_resume_counts_previous_starts_and_obeys_deadline(tmp_path, monkeypatch):
    runner = module("run_adapter05")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setenv("CONTAINER_ID", "123")
    plan = module("make_adapter05_plan").make_plan()
    plan["source_sha256"] = {}
    plan["max_total_starts"] = 2
    plan_path, rental_path = tmp_path / "plan.json", tmp_path / "rental.json"
    plan_path.write_text(json.dumps(plan))
    rental_path.write_text(json.dumps({"instance_id": 123, "created_utc": runner.utc()}))
    check = runner.Check(plan_path, rental_path, "engineering")
    job = plan["engineering_jobs"][0]
    assert check.reserve(job, 1)
    check.queue_lock.close()
    check.ledger.close()
    resumed = runner.Check(plan_path, rental_path, "engineering")
    assert resumed.starts == 1
    assert resumed.reserve(job, 2)
    assert not resumed.reserve(plan["engineering_jobs"][1], 1)
    resumed.admission = 0
    assert not resumed.reserve(job, 2)
    resumed.queue_lock.close()
    resumed.ledger.close()
