"""Validate one task 19 episode and replay every adapter decision from its journal.

Reuses task 15's base recording checks and strict outcome labels
(`regrasp15_validation.validate_base_recording` and `outcome`) and replays the evo19
projection exactly: gate, adapter state against the recording, native passthrough when
the gate is shut or the check passes, and every projected chunk and scale.
"""

import json
from pathlib import Path

import numpy as np

from evo19_check import (EvoConfig, excess, features, project, retrace, tilt_deg, triggered,
                         window_contacts)
from regrasp15_check import EXECUTED, Gate
from regrasp15_kinematics import fit_base
from regrasp15_runtime import gate_inputs
from regrasp15_validation import outcome, validate_base_recording


def validate_recording(path, row_path, sidecar_path, adapter_journal, settings):
    data, meta, row = validate_base_recording(path, row_path, sidecar_path)
    config = EvoConfig(**settings)
    journal = [json.loads(x) for x in Path(adapter_journal).read_text().splitlines() if x]
    created = [e for e in journal if e["event"] == "created"]
    if len(created) != 1 or created[0]["settings"] != config.settings():
        raise ValueError("Executed adapter configuration differs from the job")
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
    tally = {"gated": 0, "vetoes": 0, "native_passes": 0, "projected": 0}
    scales = []
    origin = None
    decisions = [e for e in journal if e["event"] == "decision"]
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
        if config.mode == "shield":
            tilt = tilt_deg(data["jar_quat"][step])
            contacts = window_contacts(data["jar_robot_contacts"], step, config)
            if abs(tilt - decision["tilt_deg"]) > 1e-4 or contacts != decision["contacts"]:
                raise ValueError(f"Shield inputs differ from the recording at step {step}")
            if not triggered(tilt, contacts, config):
                if action != "native" or native.tobytes() != executed.tobytes():
                    raise ValueError(f"Shield passthrough is invalid at step {step}")
                origin = None
                tally["native_passes"] += 1
                continue
            start = step if origin is None else origin
            result, origin = retrace(native, q_all, eef_all, start, config.retreat_m)
            if (action != "rewound" or decision["origin"] != start
                    or decision["target"] != origin
                    or result.astype(np.float32).tobytes() != executed.tobytes()):
                raise ValueError(f"Retrace replay differs at step {step}")
            tally["vetoes"] += 1
            tally["projected"] += 1
            continue
        feat = features(native, q, eef, jar, yaw, config)
        if feat["max_near_speed_mps"] != decision["native_near_speed_mps"]:
            raise ValueError(f"Native check replay differs at step {step}")
        if action == "native":
            passes = config.mode == "native" or excess(feat, config.speed_limit_mps) == 0.0
            if not passes or native.tobytes() != executed.tobytes():
                raise ValueError(f"Native passthrough is invalid at step {step}")
            tally["native_passes"] += config.mode != "native"
            continue
        if action != "projected" or config.mode != "project":
            raise ValueError(f"Unknown adapter action {action}")
        tally["vetoes"] += 1
        result, scale, _ = project(native, q, eef, jar, yaw, config)
        if result.astype(np.float32).tobytes() != executed.tobytes() or scale != decision["scale"]:
            raise ValueError(f"Projection replay differs at step {step}")
        tally["projected"] += 1
        scales.append(scale)
    for t in range(gate.seen, n + 1):
        gate.update(t, *gate_inputs(data["ap"][t], names))
    latched = [e["step"] for e in journal if e["event"] == "gate_latched"]
    if latched != ([gate.latched_at] if gate.latched_at is not None else []):
        raise ValueError("Journal gate latch differs from replay")
    counts = journal[-1]["counts"]
    if counts["gated_queries"] != tally["gated"] or counts["vetoes"] != tally["vetoes"]:
        raise ValueError("Adapter counters differ from the replayed decisions")
    return {
        "row": row,
        "steps": n,
        "settings": config.settings(),
        "adapter": {**tally, "gate_latched_step": gate.latched_at,
                    "gate_closed_step": gate.closed_at, "gate_closed_reason": gate.reason,
                    "mean_projection_scale": float(np.mean(scales)) if scales else None,
                    "base_yaw_gap_rad": float(yaw_gap)},
        "outcome": outcome(data, meta, row),
        "recorder_overhead_s": meta["recorder_overhead_s"],
    }
