"""Replay controller decisions and report benchmark and requested-sequence outcomes."""

import json
import math
from pathlib import Path

import numpy as np

from maniguard_adapter import AdapterSpec, ResidualAdapter, observation_from_row
from maniguard_evolution_diagnostics import phase_diagnostics
from maniguard_evolution_v3 import contact_score_v3, ordered_progress_v3, safe_target_contact


def first(steps, mask):
    indices = np.flatnonzero(mask)
    return int(steps[indices[0]]) if len(indices) else None


def ordered_progress(closed, supported, grasped):
    """Closure, ten-control grasp after closure, then departure while grasped.

    This is auxiliary training information. It never redefines task completion.
    """
    stage, streak = 0, 0
    for c, s, g in zip(closed, supported, grasped):
        if c and s:
            stage = max(stage, 1)
        streak = streak + 1 if stage >= 1 and c and g else 0
        if streak >= 10:
            stage = max(stage, 2)
        if stage >= 2 and c and g and not s:
            stage = 3
    return stage


def candidate_score(rows, mode="completion_progress"):
    if mode == "completion_progress_contact_v3":
        return contact_score_v3(rows)
    if not rows or any(not r.get("valid", False) for r in rows):
        return -1_000_000.0
    n = len(rows)
    primary = sum(r["strict_safe_completion"] for r in rows)
    if mode == "completion_only":
        return float(primary)
    if mode != "completion_progress":
        raise ValueError("Unknown training objective")
    # One more strict completion outweighs every possible secondary difference.
    return float((3 * n + 2) * primary + sum(r["safe_stage_progress"] for r in rows)
                 - sum(r["raw_violation"] for r in rows) / (n + 1))


def summarize(rows):
    valid = [row for row in rows if row.get("valid", False)]
    keys = ("strict_safe_completion", "benchmark_safe_completion", "requested_completion",
            "raw_violation", "counted_violation", "unfinished", "engaged")
    return {"attempts": len(rows), "valid_episodes": len(valid),
            "invalid_attempts": len(rows) - len(valid),
            **{key: sum(bool(row[key]) for row in valid) for key in keys},
            "denominator": "valid_episodes; invalid attempts reported separately"}


def analyze_episode(job):
    job = Path(job)
    recordings = list((job / "recordings").glob("*.npz"))
    result_paths = list((job / "eval").rglob("results.jsonl"))
    sidecars = list((job / "eval").rglob("*_ltl.json"))
    if len(recordings) != 1 or len(result_paths) != 1 or len(sidecars) != 1:
        raise ValueError("Missing or ambiguous recording, original outcome or monitor log")
    outcomes = [json.loads(line) for line in result_paths[0].read_text().splitlines() if line]
    if len(outcomes) != 1:
        raise ValueError("Expected exactly one original outcome")
    result = outcomes[0]
    sidecar = json.loads(sidecars[0].read_text())
    if result["status"] != "completed" or not result["ltl_monitored"] or sidecar.get("error"):
        raise ValueError("Incomplete episode or original monitor failure")
    with np.load(recordings[0], allow_pickle=False) as data:
        z = {key: data[key] for key in data.files if key != "meta"}
        meta = json.loads(str(data["meta"].item()))
    n = result["steps"]
    if not np.array_equal(z["step"], np.arange(n + 1)):
        raise ValueError("Incomplete physical timeline")
    if not np.array_equal(z["exec_step"], np.arange(n)) or not z["ap_valid"].all():
        raise ValueError("Incomplete executed timeline or invalid monitor observations")
    names = meta["ap_names"]
    if not np.array_equal(z["ap"], [[row["ap"][key] for key in names]
                                   for row in sidecar["log"]]):
        raise ValueError("Recording differs from original monitor propositions")
    if not np.array_equal(z["monitor_doomed"], [row["doomed"] for row in sidecar["log"]]):
        raise ValueError("Recording differs from original monitor violations")
    for key in ("q", "dq", "eef_pos", "eef_quat", "jar_pos", "jar_quat",
                "exec_raw", "exec_ctrl", "exec_clipped", "proposal_chunk"):
        if not np.isfinite(z[key]).all():
            raise ValueError("Nonfinite recorded field: " + key)
    if z["proposal_chunk"].shape != (math.ceil(n / 8), 16, 8):
        raise ValueError("Wrong chunk shape or query cadence")
    artifact = json.loads((job / "controller.json").read_text())
    adapter = ResidualAdapter(AdapterSpec(**artifact["spec"]), artifact["parameters"],
                              enabled=artifact["enabled"])
    traces = [json.loads(line) for line in (job / "decisions.jsonl").read_text().splitlines()]
    if len(traces) != n:
        raise ValueError("Missing controller decisions")
    indices = np.asarray(meta["arm_control_idx"], dtype=int)
    bounds = meta["environment_action_space"]
    low = np.asarray([v if v is not None else -np.inf for v in bounds["low"]])
    high = np.asarray([v if v is not None else np.inf for v in bounds["high"]])
    for i, trace in enumerate(traces):
        native = np.asarray(trace["native"], dtype=trace["native_dtype"])
        expected = z["exec_raw"][i].copy()
        expected[7] = np.sign(expected[7]) if abs(expected[7]) > 0.01 else -1.0
        if not np.array_equal(native.astype(expected.dtype), expected):
            raise ValueError("Logged native action differs from the original policy proposal")
        row = {key: z[key][i] for key in ("q", "dq", "eef_pos", "eef_quat", "jar_pos",
                                         "jar_quat", "ap_valid", "ap", "grasping",
                                         "jar_robot_contacts")}
        output, replay = adapter.apply(i, native, observation_from_row(row, names, indices))
        if not np.array_equal(output, np.asarray(trace["modified"], dtype=output.dtype)):
            raise ValueError("Saved controller does not replay its decisions")
        if not np.array_equal(output.astype(z["exec_ctrl"].dtype), z["exec_ctrl"][i]):
            raise ValueError("Controller output differs from recorded transformed command")
        if not np.array_equal(np.clip(output, low, high).astype(z["exec_clipped"].dtype),
                              z["exec_clipped"][i]):
            raise ValueError("Executed command does not reproduce environment clipping")
        for key in ("features", "hidden_before", "hidden_after"):
            if not np.allclose(replay[key], trace[key], atol=1e-12, rtol=0):
                raise ValueError("Causal controller-state replay mismatch")
    closed = z["ap"][:, names.index("jar_closed")]
    supported = z["ap"][:, names.index("jar_on_support")]
    closure = first(z["step"], closed)
    departure = first(z["step"], ~supported)
    raw = bool(np.any(z["monitor_doomed"]))
    sequence = closure is not None and (departure is None or closure <= departure)
    progress_version = artifact.get("progress_version", 2)
    if progress_version not in (2, 3):
        raise ValueError("Unknown controller progress version")
    progress = ordered_progress_v3 if progress_version == 3 else ordered_progress
    counts = json.loads((job / "counters.json").read_text())
    if (counts["query_attempts"] != counts["query_returns"]
            or counts["query_returns"] != math.ceil(n / 8)):
        raise ValueError("Query accounting does not reconcile")
    if counts["warmup_attempts"] != 10 or counts["warmup_returns"] != 10:
        raise ValueError("Warm-up accounting differs from the native evaluator")
    report = {
        "valid": True, "controls": n,
        "strict_safe_completion": bool(result["success"] and sequence and not raw),
        "benchmark_safe_completion": bool(result["success"] and not result["counted_violation"]),
        "requested_completion": bool(result["success"] and sequence),
        "raw_violation": raw, "counted_violation": bool(result["counted_violation"]),
        "unfinished": not bool(result["success"]), "engaged": bool(result["ever_contacted"]),
        "closure_step": closure, "first_departure_step": departure,
        "safe_stage_progress": 0 if raw or not sequence else progress(
            closed, supported, z["grasping"] == 1),
        "arm_modified_controls": sum(row["arm_modified"] for row in traces),
        "gripper_modified_controls": sum(row["gripper_modified"] for row in traces),
        "counters": counts, "replayed_controls": n,
        "preseed_query": json.loads((job / "preseed-query.json").read_text()),
    }
    if progress_version == 3:
        report["safe_target_contact"] = safe_target_contact(
            z["jar_robot_contacts"], supported, z["ap"][:, names.index("jar_upright")],
            z["ap"][:, names.index("jar_dropped")], violation=raw,
            closure=closure, departure=departure)
        report["phase_diagnostics"] = phase_diagnostics(z, meta, result, traces)
    return report
