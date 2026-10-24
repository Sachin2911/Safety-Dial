"""Validate factual control boundaries and replay every adapter command."""

import json
from pathlib import Path

import numpy as np

from helpers.execution_adapter import AdapterConfig, ExecutionAdapter
from helpers.tracking import load_recording, validate_recording


def validate_adapter_recording(path, row_path, sidecar_path):
    result = validate_recording(path, row_path, sidecar_path)
    directory = Path(path).parent.parent
    settings = json.loads((directory / "adapter-config.json").read_text())
    journal = [json.loads(line) for line in (directory / "adapter.jsonl").read_text().splitlines()]
    created = [e for e in journal if e["event"] == "created"]
    commands = [e for e in journal if e["event"] == "command"]
    applied = [e["step"] for e in journal if e["event"] == "applied"]
    warmups = [e["step"] for e in journal if e["event"] == "warmup_applied"]
    if len(created) != 1 or created[0]["settings"] != settings:
        raise ValueError("Executed adapter config differs from the job")
    n = result["steps"]
    if len(commands) != n or applied != list(range(n)) or warmups != list(range(10)):
        raise ValueError("Adapter journal action/warmup counts are inconsistent")
    if journal[-1]["event"] != "finish" or journal[-1]["result"] != result["row"]:
        raise ValueError("Adapter journal is incomplete or scoring differs")
    data, meta = load_recording(path)
    for event in ("query_attempt", "query_returned"):
        if [e["step"] for e in journal if e["event"] == event] != list(range(0, n, 8)):
            raise ValueError("Adapter journal policy query count differs from the proposals")
    for key, expected in {
        "max_steps": 2000,
        "source_profile": "legacy",
        "gripper_binarize": True,
    }.items():
        if meta["config"].get(key) != expected:
            raise ValueError(f"Unexpected evaluation contract: {key}")
    if created[0]["arm_control_idx"] != meta["arm_control_idx"]:
        raise ValueError("Adapter and recorder disagree on joint mapping")
    controller = ExecutionAdapter(AdapterConfig(**settings))
    for step, entry in enumerate(commands):
        if entry["step"] != step:
            raise ValueError("Adapter command indices are not contiguous")
        native = np.asarray(entry["native_command"], dtype=entry["command_dtype"])
        expected_native = data["exec_raw"][step].copy()
        expected_native[-1] = (
            np.sign(expected_native[-1]) if abs(expected_native[-1]) > 0.01 else -1.0
        )
        np.testing.assert_array_equal(native.astype(np.float32), expected_native)
        for key in ("q", "dq"):
            np.testing.assert_array_equal(
                np.asarray(entry[key], dtype=np.float32), data[key][step, meta["arm_control_idx"]]
            )
        corrected, replay = controller.apply(native, entry["q"], entry["dq"], step)
        np.testing.assert_array_equal(corrected, entry["corrected_command"])
        np.testing.assert_array_equal(corrected.astype(np.float32), data["exec_ctrl"][step])
        for key in (
            "coefficients",
            "next_coefficients",
            "correction",
            "reference_velocity",
            "reference_acceleration",
        ):
            np.testing.assert_allclose(replay[key], entry[key], rtol=1e-12, atol=1e-12)
        if native[-1].tobytes() != corrected[-1].tobytes():
            raise ValueError("Adapter modified the gripper command")
        if np.max(np.abs(corrected[:7] - native[:7])) > settings["correction_limit"] + 1e-6:
            raise ValueError("Corrected target exceeded the frozen correction bound")
        if settings["mode"] == "native" and native.tobytes() != corrected.tobytes():
            raise ValueError("Native mode is not exact pass-through")
    result["adapter_mode"] = settings["mode"]
    result["adapter_replay_steps"] = n
    return result
