"""Validate original outcomes and exactly replay this probe's action decisions."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from jar_grasp_probe import GraspController


def first(steps, mask):
    indices = np.flatnonzero(mask)
    return int(steps[indices[0]]) if len(indices) else None


def analyze(job, plan):
    recording = list((job / "recordings").glob("*.npz"))
    rows = list((job / "eval").rglob("results.jsonl"))
    sidecars = list((job / "eval").rglob("*_ltl.json"))
    if len(recording) != 1 or len(rows) != 1 or len(sidecars) != 1:
        raise ValueError("Missing or ambiguous recording, outcome row or original LTL sidecar")
    result_rows = [json.loads(line) for line in rows[0].read_text().splitlines() if line]
    if len(result_rows) != 1:
        raise ValueError("Expected exactly one original result")
    result = result_rows[0]
    if result["status"] != "completed" or not result["ltl_monitored"]:
        raise ValueError("Episode did not finish validly with the original monitor")
    sidecar = json.loads(sidecars[0].read_text())
    if sidecar.get("error"):
        raise ValueError("Original monitor recorded an error")
    specification = json.loads((job / "job.json").read_text())
    with np.load(recording[0], allow_pickle=False) as archive:
        z = {key: archive[key] for key in archive.files if key != "meta"}
        meta = json.loads(str(archive["meta"].item()))
    n = result["steps"]
    if not np.array_equal(z["step"], np.arange(n + 1)) or len(z["exec_step"]) != n:
        raise ValueError("Incomplete state or executed-command timeline")
    if not np.array_equal(z["exec_step"], np.arange(n)) or not z["ap_valid"].all():
        raise ValueError("Incomplete control steps or invalid monitor state")
    names = meta["ap_names"]
    logged_ap = np.asarray([[row["ap"][name] for name in names] for row in sidecar["log"]])
    if not np.array_equal(z["ap"], logged_ap):
        raise ValueError("Recording and original monitor propositions differ")
    if not np.array_equal(z["monitor_doomed"], [row["doomed"] for row in sidecar["log"]]):
        raise ValueError("Recording and original monitor violation states differ")
    for key in ("q", "dq", "eef_pos", "eef_quat", "jar_pos", "jar_quat", "exec_raw",
                "exec_ctrl", "exec_clipped", "proposal_chunk"):
        if not np.isfinite(z[key]).all():
            raise ValueError("Nonfinite recorded field: " + key)
    if z["proposal_chunk"].shape[0] != math.ceil(n / 8):
        raise ValueError("Wrong policy-query cadence")
    if z["proposal_chunk"].shape[1:] != (16, 8):
        raise ValueError("Wrong policy chunk shape")
    decisions = [json.loads(line) for line in (job / "decisions.jsonl").read_text().splitlines()]
    if len(decisions) != n:
        raise ValueError("Missing action decisions")
    controller = GraspController(plan["controller"], specification["arm"] == "pose_triggered_grasp")
    bounds = meta["environment_action_space"]
    low = np.asarray([value if value is not None else -np.inf for value in bounds["low"]])
    high = np.asarray([value if value is not None else np.inf for value in bounds["high"]])
    overrides, opportunities, triggers = 0, 0, []
    for i, trace in enumerate(decisions):
        original = z["exec_raw"][i].copy()
        original[-1] = np.sign(original[-1]) if abs(original[-1]) > 0.01 else -1.0
        if not np.array_equal(original, trace["native"]):
            raise ValueError("Native target does not reproduce the original proposal conversion")
        observation = trace["observation"]
        if not np.allclose(observation["eef_pos"], z["eef_pos"][i], atol=1e-6, rtol=0):
            raise ValueError("Decision did not use the current pre-action pose")
        for key in ("eef_quat", "jar_pos", "jar_quat"):
            # The recorder stores float32; live data may use a wider dtype.
            if not np.allclose(observation[key], z[key][i], atol=1e-6, rtol=0):
                raise ValueError("Decision state disagrees with recording: " + key)
        if observation["ap"] != dict(zip(names, z["ap"][i].tolist())):
            raise ValueError("Decision used different safety predicates")
        if observation["grasp"] != int(z["grasping"][i]):
            raise ValueError("Decision used different grasp state")
        output, replay = controller.apply(i, original, observation)
        if not np.array_equal(output, z["exec_ctrl"][i]):
            raise ValueError("Decision replay differs from modified command")
        if not np.array_equal(np.clip(output, low, high), z["exec_clipped"][i]):
            raise ValueError("Environment input does not reproduce the original clipping")
        if not np.array_equal(output[:7], original[:7]):
            raise ValueError("Probe changed arm coordinates")
        for key in ("mode_before", "mode_after", "age", "attempted", "override", "eligible"):
            if replay[key] != trace[key]:
                raise ValueError("State-machine replay mismatch: " + key)
        overrides += trace["override"]
        opportunities += trace["eligible"]
        if trace["mode_before"] == "idle" and trace["mode_after"] == "acquiring":
            triggers.append(i)
    if len(triggers) > 1:
        raise ValueError("More than one grasp attempt")
    closed = first(z["step"], z["ap"][:, names.index("jar_closed")])
    departure = first(z["step"], ~z["ap"][:, names.index("jar_on_support")])
    raw_violation = bool(np.any(z["monitor_doomed"]))
    sequence = closed is not None and (departure is None or closed <= departure)
    sustained, streak = False, 0
    if triggers:
        for i in range(triggers[0] + 1, n + 1):
            streak = streak + 1 if z["grasping"][i] == 1 and not z["monitor_doomed"][i] else 0
            sustained |= streak >= 10
    counters = json.loads((job / "counters.json").read_text())
    if counters["query_attempts"] != counters["query_returns"] or counters["query_returns"] != math.ceil(n / 8):
        raise ValueError("Query accounting does not reconcile")
    if counters["warmup_attempts"] != 10 or counters["warmup_returns"] != 10:
        raise ValueError("Warmup accounting differs from the native evaluator")
    report = {
        "valid": True, "job": specification, "controls": n,
        "primary": bool(result["success"] and sequence and not raw_violation),
        "benchmark_safe_completion": bool(result["success"] and not result["counted_violation"]),
        "requested_completion": bool(result["success"] and sequence),
        "raw_violation": raw_violation, "counted_violation": bool(result["counted_violation"]),
        "unfinished": not bool(result["success"]), "engaged": bool(result["ever_contacted"]),
        "closure_step": closed, "first_departure_step": departure,
        "first_grasp_step": first(z["step"], z["grasping"] == 1),
        "eligible_controls": opportunities, "overridden_controls": overrides,
        "trigger_steps": triggers, "post_trigger_sustained_grasp": bool(sustained),
        "decision_replay_controls": n, "arm_targets_preserved": True,
        "counters": counters,
        "preseed_query": json.loads((job / "preseed-query.json").read_text()),
        "scoring_note": "Original benchmark and requested sequence reported separately",
    }
    (job / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.job, json.loads(args.plan.read_text()))))
