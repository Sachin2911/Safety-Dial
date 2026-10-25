"""Validate both execution horizons and replay every task-8 controller action.

The base checks retain task 4's recorder/scoring validation, with horizon explicit.
Historical validators and results remain unchanged.
"""

import json
import math
from pathlib import Path

import numpy as np

from helpers.screen08_controller import Controller, ScreenConfig
from helpers.tracking import SCORING, STATE_FIELDS, load_recording


def validate_base_recording(path, row_path, sidecar_path, *, expected_horizon):
    data, meta = load_recording(path)
    rows = [json.loads(line) for line in Path(row_path).read_text().splitlines() if line]
    if len(rows) != 1:
        raise ValueError(f"Expected one row, found {len(rows)}")
    row = rows[0]
    if row.get("status") != "completed" or not row.get("ltl_monitored"):
        raise ValueError("Episode did not complete with its monitor attached")
    n = row["steps"]
    if not 0 < n <= 2000:
        raise ValueError("Invalid executed-step count")
    if meta["action_frequency"] != 20 or len(meta["arm_control_idx"]) != 7:
        raise ValueError("Unexpected control frequency or arm joint mapping")
    for key, expected in {"state_mode": "joint", "controller_preset": "joint_position_raw",
                          "ik_eef_to_joint": False, "camera_resolution": 256,
                          "success_hold_steps": 10}.items():
        if meta["config"].get(key) != expected:
            raise ValueError(f"Unexpected evaluation setting: {key}")
    for key in STATE_FIELDS:
        if key not in data or len(data[key]) != n + 1:
            raise ValueError(f"{key}: expected N+1 state rows")
    for key in ("exec_step", "exec_proposal", "exec_offset", "exec_raw", "exec_ctrl",
                "exec_clipped"):
        if key not in data or len(data[key]) != n:
            raise ValueError(f"{key}: expected N action rows")
    if not np.array_equal(data["step"], np.arange(n + 1)):
        raise ValueError("State step identifiers are not contiguous")
    if not np.array_equal(data["exec_step"], np.arange(n)):
        raise ValueError("Executed step identifiers are not 0..N-1")
    horizon = meta["execute_horizon"]
    if horizon != expected_horizon or data["proposal_chunk"].shape != (
        math.ceil(n / horizon), 16, 8
    ):
        raise ValueError("Unexpected proposal count or chunk shape")
    if not np.array_equal(data["proposal_step"], np.arange(0, n, horizon)):
        raise ValueError("Unexpected policy query steps")
    if not np.array_equal(data["exec_proposal"], np.arange(n) // horizon):
        raise ValueError("Incorrect executed proposal identifiers")
    if not np.array_equal(data["exec_offset"], np.arange(n) % horizon):
        raise ValueError("Incorrect within-chunk offsets")
    expected = data["proposal_chunk"][data["exec_proposal"], data["exec_offset"]]
    if not np.array_equal(expected, data["exec_raw"]):
        raise ValueError("Executed raw actions differ from their proposals")
    for key in ("q", "dq", "eef_pos", "gripper_q", "policy_state", "proposal_chunk",
                "exec_raw", "exec_ctrl", "exec_clipped"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Non-finite values in {key}")
    for key in SCORING:
        if key not in meta["result"] or key not in row or meta["result"][key] != row[key]:
            raise ValueError(f"Recorded scoring differs from result row: {key}")
    sidecar = json.loads(Path(sidecar_path).read_text())
    propositions = np.array([[entry["ap"][name] for name in meta["ap_names"]]
                             for entry in sidecar["log"]], dtype=bool)
    if not data["ap_valid"].all() or not np.array_equal(data["ap"], propositions):
        raise ValueError("Recorded propositions differ from monitor sidecar")
    if not np.array_equal([entry["step"] for entry in sidecar["log"]], data["step"]):
        raise ValueError("Monitor sidecar steps differ from recording")
    journal_path = Path(path).with_suffix(".journal.jsonl")
    if journal_path.exists():
        journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
        applied = [entry["step"] for entry in journal if entry["event"] == "applied"]
        proposals = [entry["step"] for entry in journal if entry["event"] == "proposal"]
        if not np.array_equal(applied, data["exec_step"]):
            raise ValueError("Crash-surviving journal disagrees with executed-step array")
        if not np.array_equal(proposals, data["proposal_step"]):
            raise ValueError("Crash-surviving journal disagrees with proposal array")
        if journal[-1]["event"] != "finish" or journal[-1]["result"] != row:
            raise ValueError("Journal is incomplete or records a different result")
    return {"steps": n, "warmup_control_steps": 10,
            "queries": len(data["proposal_chunk"]), "row": row,
            "recorder_overhead_s": meta["recorder_overhead_s"]}


def validate_recording(path, row_path, sidecar_path):
    directory = Path(path).parent.parent
    settings = json.loads((directory / "screen-config.json").read_text())
    config = ScreenConfig(**settings)
    result = validate_base_recording(
        path, row_path, sidecar_path, expected_horizon=config.horizon
    )
    journal = [json.loads(x) for x in (directory / "screen.jsonl").read_text().splitlines()]
    created = [e for e in journal if e["event"] == "created"]
    commands = [e for e in journal if e["event"] == "command"]
    proposals = [e for e in journal if e["event"] == "raw_proposal"]
    n = result["steps"]
    if len(created) != 1 or created[0]["settings"] != settings:
        raise ValueError("Executed screen configuration differs from frozen job")
    if len(commands) != n:
        raise ValueError("Wrong number of modified commands")
    for event, steps in {
        "applied": list(range(n)), "warmup_applied": list(range(10)),
        "query_attempt": list(range(0, n, config.horizon)),
        "query_returned": list(range(0, n, config.horizon)),
        "raw_proposal": list(range(0, n, config.horizon)),
    }.items():
        if [e["step"] for e in journal if e["event"] == event] != steps:
            raise ValueError(f"Screen journal has inconsistent {event} events")
    if journal[-1]["event"] != "finish" or journal[-1]["result"] != result["row"]:
        raise ValueError("Screen journal is incomplete or scoring differs")
    data, meta = load_recording(path)
    if created[0]["arm_control_idx"] != meta["arm_control_idx"]:
        raise ValueError("Controller and recorder disagree on joint order")
    for key, expected in {"max_steps": 2000, "source_profile": None,
                          "gripper_binarize": True}.items():
        if key not in meta["config"] or meta["config"][key] != expected:
            raise ValueError(f"Unexpected evaluation contract: {key}")
    controller = Controller(config)
    bounds = [e for e in journal if e["event"] == "action_bounds"]
    if len(bounds) != 1:
        raise ValueError("Missing action clipping bounds")
    low, high = np.asarray(bounds[0]["low"]), np.asarray(bounds[0]["high"])
    if low.shape != (8,) or high.shape != (8,) or not np.all(low <= high):
        raise ValueError("Invalid action bounds")
    for step, entry in enumerate(commands):
        if entry["step"] != step:
            raise ValueError("Noncontiguous controller commands")
        dtype = entry["command_dtype"]
        if step % config.horizon == 0:
            index = step // config.horizon
            chunk = np.asarray(proposals[index]["chunk"], dtype=dtype)
            np.testing.assert_array_equal(chunk.astype(np.float32), data["proposal_chunk"][index])
            controller.proposal(step, chunk)
        native = np.asarray(entry["native_command"], dtype=dtype)
        expected_native = data["exec_raw"][step].copy()
        expected_native[-1] = (
            np.sign(expected_native[-1]) if abs(expected_native[-1]) > 0.01 else -1.0
        )
        np.testing.assert_array_equal(native.astype(np.float32), expected_native)
        np.testing.assert_array_equal(
            np.asarray(entry["q"], dtype=np.float32), data["q"][step, meta["arm_control_idx"]]
        )
        modified, replay = controller.apply(step, native, entry["q"])
        np.testing.assert_array_equal(modified, entry["modified_command"])
        np.testing.assert_array_equal(modified.astype(np.float32), data["exec_ctrl"][step])
        np.testing.assert_array_equal(
            np.clip(modified, low.astype(dtype), high.astype(dtype)).astype(np.float32),
            data["exec_clipped"][step],
        )
        for key in ("changed", "gripper_close_denied", "scaling_active",
                    "position_history_length", "offset"):
            if replay[key] != entry[key]:
                raise ValueError(f"Controller replay differs: {key}")
        if config.mode in {"native", "h4"} and modified.tobytes() != native.tobytes():
            raise ValueError("Native command passthrough is not byte-identical")
        if config.mode != "gripwait" and modified[-1].tobytes() != native[-1].tobytes():
            raise ValueError("Unexpected gripper modification")
    result.update({
        "screen_mode": config.mode,
        "replayed_controls": n,
        "changed_controls": sum(e["changed"] for e in commands),
        "closures_denied": sum(e["gripper_close_denied"] for e in commands),
        "mechanism_active": config.mode == "h4" or any(e["changed"] for e in commands),
    })
    return result

