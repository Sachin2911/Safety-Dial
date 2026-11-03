"""Validate one task 15 episode and replay every adapter decision from its journal.

The base checks keep task 4's recorder and scoring validation with the eight-control
horizon. On top of them: the journal's configuration equals the job's; the adapter's
state at each query equals the recording; the gate replays from the recorded
propositions; native chunks pass through byte-identically whenever the gate is shut or
the check passes; selection and projection replay exactly; candidate seeds follow the
frozen rule; and the live base yaw agrees with a fit over the episode's own arm and
hand positions. Outcome labels use the strict definition of tasks 6 and 7.
"""

import json
import math
from pathlib import Path

import numpy as np

from regrasp15_check import EXECUTED, CheckConfig, Gate, excess, features, project, select
from regrasp15_kinematics import fit_base
from regrasp15_runtime import candidate_seed, gate_inputs
from tracking import SCORING, STATE_FIELDS, load_recording

SELECT_RULES = {"closest_passing", "least_excess"}


def validate_base_recording(path, row_path, sidecar_path):
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
                          "success_hold_steps": 10, "max_steps": 2000,
                          "gripper_binarize": True}.items():
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
    if meta["execute_horizon"] != EXECUTED or data["proposal_chunk"].shape != (
        math.ceil(n / EXECUTED), 16, 8
    ):
        raise ValueError("Unexpected proposal count or chunk shape")
    if not np.array_equal(data["proposal_step"], np.arange(0, n, EXECUTED)):
        raise ValueError("Unexpected policy query steps")
    if not np.array_equal(data["exec_proposal"], np.arange(n) // EXECUTED):
        raise ValueError("Incorrect executed proposal identifiers")
    if not np.array_equal(data["exec_offset"], np.arange(n) % EXECUTED):
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
    journal = [json.loads(line) for line in journal_path.read_text().splitlines() if line]
    applied = [entry["step"] for entry in journal if entry["event"] == "applied"]
    proposals = [entry["step"] for entry in journal if entry["event"] == "proposal"]
    if not np.array_equal(applied, data["exec_step"]):
        raise ValueError("Crash-surviving journal disagrees with executed-step array")
    if not np.array_equal(proposals, data["proposal_step"]):
        raise ValueError("Crash-surviving journal disagrees with proposal array")
    if journal[-1]["event"] != "finish" or journal[-1]["result"] != row:
        raise ValueError("Recorder journal is incomplete or records a different result")
    return data, meta, row


def outcome(data, meta, row):
    """Strict outcome labels plus the closure-gate timeline shared by every arm."""
    names = meta["ap_names"]
    closed = data["ap"][:, names.index("jar_closed")]
    on_support = data["ap"][:, names.index("jar_on_support")]
    first_closed = int(np.argmax(closed)) if closed.any() else None
    first_departure = int(np.argmax(~on_support)) if (~on_support).any() else None
    order_failed = first_departure is not None and (
        first_closed is None or first_departure < first_closed
    )
    requested = first_closed is not None and not order_failed
    success, violated = bool(row["success"]), bool(row["ltl_violated"])
    gate = Gate(CheckConfig(mode="select", speed_limit_mps=1.0, candidates=1))
    for t in range(len(data["ap"])):
        gate.update(t, *gate_inputs(data["ap"][t], names))
    latched = gate.latched_at
    violation_step = row.get("ltl_violation_step") if violated else None
    return {
        "success": success,
        "raw_violation": violated,
        "counted_violation": bool(row.get("counted_violation")),
        "violation_step": violation_step,
        "first_closed_step": first_closed,
        "first_departure_step": first_departure,
        "order_failed": order_failed,
        "requested_success": bool(success and requested),
        "strict_safe": bool(success and requested and not violated),
        "unfinished": not success,
        "engaged": row.get("first_contact_step") is not None,
        "ever_grasped": bool(row.get("ever_grasped")),
        "closure_latched_step": latched,
        "closure_gate_opened": latched is not None,
        "post_gate_tip": bool(latched is not None and violated and violation_step is not None
                              and violation_step >= latched),
        "post_gate_stall": bool(latched is not None and not success and not violated),
        "post_gate_success": bool(latched is not None and success and requested
                                  and not violated),
        "steps": int(row["steps"]),
    }


def validate_recording(path, row_path, sidecar_path, adapter_journal, settings):
    data, meta, row = validate_base_recording(path, row_path, sidecar_path)
    config = CheckConfig(**settings)
    journal = [json.loads(x) for x in Path(adapter_journal).read_text().splitlines() if x]
    created = [e for e in journal if e["event"] == "created"]
    if len(created) != 1 or created[0]["settings"] != config.settings():
        raise ValueError("Executed adapter configuration differs from the frozen job")
    if created[0]["arm_control_idx"] != meta["arm_control_idx"]:
        raise ValueError("Adapter and recorder disagree on joint order")
    n = row["steps"]
    queries = list(range(0, n, EXECUTED))
    for event in ("query_attempt", "query_returned", "decision"):
        if [e["step"] for e in journal if e["event"] == event] != queries:
            raise ValueError(f"Adapter journal has inconsistent {event} events")
    if journal[-1]["event"] != "finish" or journal[-1]["result"] != row:
        raise ValueError("Adapter journal is incomplete or scoring differs")
    bases = [e for e in journal if e["event"] == "base"]
    if len(bases) != 1:
        raise ValueError("Expected one base-pose record")
    yaw = float(bases[0]["yaw"])
    arm = meta["arm_control_idx"]
    q_all = data["q"][:, arm].astype(np.float64)
    eef_all = data["eef_pos"].astype(np.float64)
    sample = np.linspace(0, n, min(n + 1, 200)).astype(int)
    fit_yaw, _, fit_error = fit_base(q_all[sample], eef_all[sample])
    yaw_gap = abs((fit_yaw - yaw + np.pi) % (2 * np.pi) - np.pi)
    if fit_error > 1e-3 or yaw_gap > 0.01:
        raise ValueError(f"Live base yaw differs from the fitted base by {yaw_gap:.4f} rad")
    gate = Gate(config)
    names = meta["ap_names"]
    decisions = [e for e in journal if e["event"] == "decision"]
    tally = {"gated": 0, "vetoes": 0, "native_passes": 0, "closest_passing": 0,
             "least_excess": 0, "projected": 0, "candidate_decodes": 0}
    scales, chosen_distance, decode_s = [], [], []
    for index, decision in enumerate(decisions):
        step = decision["step"]
        for t in range(gate.seen, step + 1):
            gate.update(t, *gate_inputs(data["ap"][t], names))
        if gate.open != decision["gate_open"]:
            raise ValueError(f"Gate replay differs at step {step}")
        q, eef, jar = (np.asarray(decision[k], dtype=np.float64) for k in ("q", "eef", "jar"))
        if (not np.array_equal(q, q_all[step]) or not np.array_equal(eef, eef_all[step])
                or not np.array_equal(jar, data["jar_pos"][step].astype(np.float64))):
            raise ValueError(f"Adapter state differs from the recording at step {step}")
        native = np.asarray(decision["native"], dtype=np.float32)
        executed = data["proposal_chunk"][index]
        action = decision["action"]
        if not gate.open:
            if action != "gate_shut" or native.tobytes() != executed.tobytes():
                raise ValueError(f"Native chunk changed while the gate was shut at {step}")
            continue
        tally["gated"] += 1
        feat = features(native, q, eef, jar, yaw, config.near_radius_m)
        if feat["max_near_speed_mps"] != decision["native_near_speed_mps"]:
            raise ValueError(f"Native check replay differs at step {step}")
        if action == "native":
            passes = config.mode == "native" or excess(feat, config.speed_limit_mps) == 0.0
            if not passes or native.tobytes() != executed.tobytes():
                raise ValueError(f"Native passthrough is invalid at step {step}")
            tally["native_passes"] += config.mode != "native"
            continue
        tally["vetoes"] += 1
        if action == "projected":
            if config.mode != "project":
                raise ValueError("Projection outside a projection arm")
            result, scale, _ = project(native, q, eef, jar, yaw, config)
            if (result.astype(np.float32).tobytes() != executed.tobytes()
                    or scale != decision["scale"]):
                raise ValueError(f"Projection replay differs at step {step}")
            tally["projected"] += 1
            scales.append(scale)
            continue
        if config.mode != "select" or action not in SELECT_RULES:
            raise ValueError(f"Unknown adapter action {action}")
        if decision["seed"] != candidate_seed(meta["episode_seed"], step):
            raise ValueError("Candidate seed differs from the frozen rule")
        candidates = np.asarray(decision["candidates"], dtype=np.float32)
        if candidates.shape != (config.candidates, 16, 8) or not np.isfinite(candidates).all():
            raise ValueError("Candidate set is not finite (K, 16, 8)")
        choice, rule, _, _, distance = select(native, candidates, q, eef, jar, yaw, config)
        if choice != decision["choice"] or rule != action:
            raise ValueError(f"Selection replay differs at step {step}")
        expected = native if choice == 0 else candidates[choice - 1]
        if expected.tobytes() != executed.tobytes():
            raise ValueError(f"Executed chunk is not the selected one at step {step}")
        tally[rule] += 1
        tally["candidate_decodes"] += config.candidates
        chosen_distance.append(distance[choice])
        decode_s.append(float(decision["decode_s"]))
    for t in range(gate.seen, n + 1):
        gate.update(t, *gate_inputs(data["ap"][t], names))
    latched = [e["step"] for e in journal if e["event"] == "gate_latched"]
    closed = [(e["step"], e["reason"]) for e in journal if e["event"] == "gate_closed"]
    if latched != ([gate.latched_at] if gate.latched_at is not None else []):
        raise ValueError("Journal gate latch differs from replay")
    if closed != ([(gate.closed_at, gate.reason)] if gate.closed_at is not None else []):
        raise ValueError("Journal gate closure differs from replay")
    counts = journal[-1]["counts"]
    if (counts["gated_queries"] != tally["gated"] or counts["vetoes"] != tally["vetoes"]
            or counts["candidate_decodes"] != tally["candidate_decodes"]):
        raise ValueError("Adapter counters differ from the replayed decisions")
    return {
        "row": row,
        "steps": n,
        "queries": len(queries),
        "mode": config.mode,
        "gate": config.gate,
        "speed_limit_mps": config.speed_limit_mps if math.isfinite(config.speed_limit_mps)
        else None,
        "candidates": config.candidates,
        "adapter": {
            **tally,
            "gate_latched_step": gate.latched_at,
            "gate_closed_step": gate.closed_at,
            "gate_closed_reason": gate.reason,
            "mean_projection_scale": float(np.mean(scales)) if scales else None,
            "median_chosen_joint_rms": float(np.median(chosen_distance))
            if chosen_distance else None,
            "median_decode_s": float(np.median(decode_s)) if decode_s else None,
            "base_fk_residual_m": float(bases[0]["fk_residual_m"]),
            "base_yaw_gap_rad": float(yaw_gap),
        },
        "outcome": outcome(data, meta, row),
        "recorder_overhead_s": meta["recorder_overhead_s"],
    }
