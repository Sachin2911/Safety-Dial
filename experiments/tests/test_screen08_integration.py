"""Exercise the live hook and the independent recording validator with known states."""

import importlib.util
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from screen08_runtime import Runtime
from helpers.screen08_validation import validate_recording
from helpers.screen08_controller import ScreenConfig
from helpers.tracking import SCORING, STATE_FIELDS, load_recording

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"experiments/scripts/{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def fixture_recording(tmp_path, monkeypatch, mode):
    n = 19  # Cross a chunk boundary and finish partway through the next chunk.
    settings = asdict(ScreenConfig(mode=mode))
    horizon = ScreenConfig(mode=mode).horizon
    settings_path = tmp_path / "screen-config.json"
    settings_path.write_text(json.dumps(settings))
    monkeypatch.setenv("SAFETY_DIAL_SCREEN_CONFIG", str(settings_path))
    monkeypatch.setenv("SAFETY_DIAL_SCREEN_JOURNAL", str(tmp_path / "screen.jsonl"))
    cfg = {
        "action_frequency": 20,
        "execute_horizon": horizon,
        "state_mode": "joint",
        "controller_preset": "joint_position_raw",
        "ik_eef_to_joint": False,
        "max_steps": 2000,
        "source_profile": None,
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
    runtime.emit("action_bounds", low=np.full(8,-.1), high=np.full(8,.1))
    for _ in range(10):
        runtime.warmup_before(np.zeros(8))
        runtime.warmup_after()
    chunks = np.random.default_rng(9).uniform(-0.2, 0.2, (math.ceil(n / horizon), 16, 8))
    raw = chunks[np.arange(n) // horizon, np.arange(n) % horizon]
    corrected = []
    for step, command in enumerate(raw):
        if step % horizon == 0:
            runtime.emit("query_attempt", step=step)
            runtime.emit("query_returned", step=step)
            runtime.proposal(step, chunks[step // horizon])
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
        exec_proposal=np.arange(n) // horizon,
        exec_offset=np.arange(n) % horizon,
        exec_raw=raw.astype(np.float32),
        exec_ctrl=np.asarray(corrected, dtype=np.float32),
        exec_clipped=np.clip(np.asarray(corrected), -.1, .1).astype(np.float32),
        proposal_step=np.arange(0, n, horizon),
        proposal_chunk=chunks.astype(np.float32),
    )
    meta = {
        "config": cfg,
        "result": row,
        "arm_control_idx": indices,
        "execute_horizon": horizon,
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


@pytest.mark.parametrize("mode", ["native", "h4", "slow", "smooth", "gripwait"])
def test_hook_roundtrip_and_validator_detect_corrupted_command(tmp_path, monkeypatch, mode):
    paths = fixture_recording(tmp_path, monkeypatch, mode)
    result = validate_recording(*paths)
    assert result["replayed_controls"] == 19
    assert result["queries"] == math.ceil(19 / ScreenConfig(mode=mode).horizon)
    assert result["screen_mode"] == mode
    journal_path = tmp_path / "screen.jsonl"
    journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
    next(e for e in journal if e["event"] == "command")["q"][0] += 0.1
    journal_path.write_text("".join(json.dumps(e) + "\n" for e in journal))
    with pytest.raises(AssertionError):
        validate_recording(*paths)


@pytest.mark.parametrize("profile", ["high_fidelity", "missing"])
def test_validator_rejects_wrong_or_missing_resolved_render_profile(tmp_path, monkeypatch, profile):
    paths = fixture_recording(tmp_path, monkeypatch, "native")
    data, meta = load_recording(paths[0])
    if profile == "missing":
        del meta["config"]["source_profile"]
    else:
        meta["config"]["source_profile"] = profile
    np.savez_compressed(paths[0], **data, meta=json.dumps(meta))
    with pytest.raises(ValueError, match="source_profile"):
        validate_recording(*paths)

