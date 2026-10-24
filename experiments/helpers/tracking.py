"""Offline validation and alignment for native Jar joint-position recordings."""

import json
import math
from pathlib import Path

import numpy as np


SCORING = (
    "scene_name", "seed", "episode_seed", "status", "steps", "success", "success_step",
    "success_first_step", "ltl_violated", "ltl_violation_step", "ltl_violation_count",
    "first_contact_step", "ever_grasped", "grasp_steps", "outcome", "counted_violation",
    "safety_evaluated",
)
STATE_FIELDS = (
    "step", "wall_s", "recorder_s", "q", "dq", "eef_pos", "eef_quat", "gripper_q",
    "policy_state", "jar_pos", "jar_quat", "jar_link_pose", "jar_joint_q", "jar_joint_dq",
    "jar_lin_vel", "jar_ang_vel", "jar_robot_contacts", "jar_finger_contact", "grasping",
    "other_obj_pos", "ap", "ap_valid", "monitor_doomed",
)


def load_recording(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files if key != "meta"}
        meta = json.loads(str(archive["meta"]))
    return data, meta


def validate_recording(path, row_path, sidecar_path):
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
    if horizon != 8 or data["proposal_chunk"].shape != (math.ceil(n / 8), 16, 8):
        raise ValueError("Unexpected proposal count or chunk shape")
    if not np.array_equal(data["proposal_step"], np.arange(0, n, 8)):
        raise ValueError("Unexpected policy query steps")
    if not np.array_equal(data["exec_proposal"], np.arange(n) // 8):
        raise ValueError("Incorrect executed proposal identifiers")
    if not np.array_equal(data["exec_offset"], np.arange(n) % 8):
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
    return {"steps": n, "queries": len(data["proposal_chunk"]), "row": row,
            "recorder_overhead_s": meta["recorder_overhead_s"]}


def aligned_arm(data, meta, lag=0):
    """At lag 0, match action t to state t before execution; lag 1 is after it.

    Do not subtract the first N joint rows positionally. Explicit step identifiers
    prevent accidental alignment changes when analysing a slice of an episode.
    """
    state_steps = np.asarray(data["step"])
    wanted = np.asarray(data["exec_step"]) + lag
    indices = np.searchsorted(state_steps, wanted)
    valid = indices < len(state_steps)
    valid[valid] &= state_steps[indices[valid]] == wanted[valid]
    target = np.asarray(data["exec_clipped"])[valid, :7]
    actual = np.asarray(data["q"])[indices[valid]][:, meta["arm_control_idx"]]
    return target, actual, np.asarray(data["exec_step"])[valid]


def first_step(data, predicate):
    indices = np.flatnonzero(predicate)
    return int(data["step"][indices[0]]) if len(indices) else None


def events(data, meta):
    ap = {name: data["ap"][:, i] for i, name in enumerate(meta["ap_names"])}
    return {
        "contact": first_step(data, data["jar_robot_contacts"] > 0),
        "finger_contact": first_step(data, data["jar_finger_contact"]),
        "grasp": first_step(data, data["grasping"] == 1),
        "closed": first_step(data, ap["jar_closed"]),
        "lift": first_step(data, ~ap["jar_on_support"]),
        "violation": meta["result"].get("ltl_violation_step"),
    }


def requested_sequence(event):
    return event["closed"] is not None and (
        event["lift"] is None or event["closed"] <= event["lift"]
    )
