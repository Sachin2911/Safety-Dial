"""Describe how the jar fails in every task 19 episode, from the recorded simulator state.

For each valid episode (search and confirmation) this reads the episode recording and
writes per-episode descriptors to outputs/evo19/analysis/mechanism.json:

- the violation step, which monitor proposition failed there, and whether the jar had
  been closed before it;
- jar tilt from vertical (from the jar quaternion, x y z w), its value at the violation and
  the lead time from the start of the final excursion above 5, 8 and 12 degrees;
- gripper width, robot-jar contact and grasp state in the 40 controls before;
- hand speed (finite differences of the end-effector position at 20 Hz): the peak in the
  40 controls before the violation, the mean over the five controls up to the moment the
  tilt last rose through 5 degrees, and the peak from that moment to the violation;
- for strict-safe episodes, the largest tilt before first closure and before first grasp;
- the first policy query (every 8 controls) at which the frozen shield rule holds: tilt at
  least 8.9196 degrees and robot-jar contact within the last 8 controls. The policy is not
  queried at or after a violation, so the rule "fires" only at a query before any
  violation. In shield episodes that is the first retrace (checked below against the
  logged retrace counts); in native episodes it is where the rule would first have fired,
  which is meaningful only up to that query because the arms diverge after it;
- an aligned tilt trace (120 controls before to 10 after the violation) for figures.

Descriptive only: these are simulator measurements, not inputs to any arm's decisions
beyond what the shield itself used.
"""

import json
from pathlib import Path

import numpy as np
from revalidate_evo19 import counted_episode, finished_runs

ROOT = Path(__file__).resolve().parents[2]
REMOTE = ROOT / "outputs/evo19/remote"
OUT = ROOT / "outputs/evo19/analysis/mechanism.json"
HZ = 20
WINDOW = 40
TRACE = (120, 10)
QUERY_EVERY = 8
RULE = {"tilt_limit_deg": 8.9196, "contact_window": 8}


def tilt_deg(quat):
    quat = np.asarray(quat, dtype=np.float64)
    return np.degrees(np.arccos(np.clip(1 - 2 * (quat[:, 0] ** 2 + quat[:, 1] ** 2), -1, 1)))


def last_rise(series, threshold, end):
    """Start of the final excursion at or above `threshold` that reaches control `end`."""
    above = series[: end + 1] >= threshold
    if not above[end]:
        return None
    start = end
    while start > 0 and above[start - 1]:
        start -= 1
    return start


def first_fire(tilt, contacts):
    """First query control where the frozen shield rule holds, or None."""
    w = RULE["contact_window"]
    for q in range(0, len(tilt), QUERY_EVERY):
        if tilt[q] >= RULE["tilt_limit_deg"] and contacts[max(0, q - w + 1):q + 1].max() > 0:
            return q
    return None


def describe(job, outcome, recording):
    with np.load(recording, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files if k != "meta"}
        meta = json.loads(str(archive["meta"]))
    names = meta["ap_names"]
    tilt = tilt_deg(data["jar_quat"])
    eef = data["eef_pos"].astype(np.float64)
    speed = np.r_[0.0, np.linalg.norm(np.diff(eef, axis=0), axis=1) * HZ]
    width = data["gripper_q"].astype(np.float64).sum(axis=1)
    contacts = data["jar_robot_contacts"].astype(int)
    grasping = data["grasping"].astype(int)
    closed_at = outcome["first_closed_step"]
    grasp_steps = np.flatnonzero(grasping == 1)
    v = outcome["violation_step"]
    fire = first_fire(tilt, contacts)
    if fire is not None and v is not None and fire >= v:
        fire = None  # no policy query at or after a violation
    row = {
        "job": job["id"], "kind": job["kind"], "stage": job.get("stage"), "task": job["task"],
        "seed": job["seed"], "mode": job["settings"]["mode"],
        "strict_safe": outcome["strict_safe"], "raw_violation": outcome["raw_violation"],
        "success": outcome["success"], "gate_opened": outcome["closure_gate_opened"],
        "first_closed_step": closed_at,
        "first_grasp_step": int(grasp_steps[0]) if grasp_steps.size else None,
        "steps": int(outcome["steps"]),
        "category": ("strict_safe" if outcome["strict_safe"] else
                     "violation" if outcome["raw_violation"] else
                     "stall_after_closure" if outcome["closure_gate_opened"] else "never_closed"),
        "rule_first_fire_step": fire,
        "max_tilt_deg": round(float(tilt.max()), 3),
    }
    if outcome["strict_safe"]:
        end_closed = closed_at if closed_at is not None else len(tilt) - 1
        end_grasp = row["first_grasp_step"] if row["first_grasp_step"] is not None else end_closed
        row["max_tilt_before_closure_deg"] = round(float(tilt[: end_closed + 1].max()), 3)
        row["max_tilt_before_grasp_deg"] = round(float(tilt[: end_grasp + 1].max()), 3)
    if not outcome["raw_violation"] or v is None:
        return row
    v = int(v)
    before = data["ap"][max(v - 1, 0)].astype(bool)
    at = data["ap"][v].astype(bool)
    failed = [names[i] for i in range(len(names)) if before[i] != at[i]]
    lo = max(v - WINDOW, 0)
    rise5 = last_rise(tilt, 5.0, v)
    row.update({
        "violation_step": v,
        "failed": failed,
        "kind_of_violation": ("tip" if "jar_upright" in failed else
                              "lift_before_close" if "jar_on_support" in failed else
                              "drop" if "jar_dropped" in failed else "other"),
        "before_closure": closed_at is None or v < closed_at,
        "tilt_at_violation_deg": round(float(tilt[v]), 3),
        "lead_controls": {str(t): (None if (s := last_rise(tilt, t, v)) is None else v - s)
                          for t in (5, 8, 12)},
        "gripper_width_before_m": round(float(width[v - 1]), 4),
        "open_gripper_before": bool(width[v - 1] > 0.07),
        "contact_in_window": bool(contacts[lo:v + 1].max() > 0),
        "grasping_before": bool(grasping[v - 1] == 1),
        "peak_hand_speed_window_mps": round(float(speed[lo:v + 1].max()), 4),
        "hand_speed_at_tilt5_mps": (None if rise5 is None
                                    else round(float(speed[max(rise5 - 4, 0):rise5 + 1].mean()), 4)),
        "peak_hand_speed_tilt5_to_violation_mps": (None if rise5 is None
                                                   else round(float(speed[rise5:v + 1].max()), 4)),
        "trace_tilt_deg": [round(float(x), 2) for x in
                           tilt[max(v - TRACE[0], 0):min(v + TRACE[1], len(tilt) - 1) + 1]],
        "trace_offset": -min(TRACE[0], v),
    })
    return row


def main():
    rows, problems = [], []
    state = json.loads((ROOT / "outputs/evo19/search/state.json").read_text())
    runs = finished_runs()
    # Counted episodes (the state's record), plus the one determinism repeat.
    jobs = [j for j in state["jobs"].values() if j["status"] == "done"]
    jobs += [json.loads(p.read_text())["job"] | {"outcome": json.loads(p.read_text())["outcome"]}
             for p in sorted(REMOTE.glob("*/results/z-repeat-*.json"))]
    for job in sorted(jobs, key=lambda j: j["id"]):
        episode = counted_episode(job["id"], runs)
        recording = next(episode.glob("recordings/*_base.npz"), None)
        if recording is None:
            problems.append({"job": job["id"], "problem": "no local recording"})
            continue
        job = dict(job)
        job["stage"] = job.get("stage", 1) if job["kind"] == "search" else None
        rows.append(describe(job, job["outcome"], recording))
    keys = {kind: {(r["task"], r["seed"]) for r in rows if r["kind"] == kind}
            for kind in ("confirm_native", "confirm_evolved")}
    paired = keys["confirm_native"] & keys["confirm_evolved"]
    violations = [r for r in rows if r.get("violation_step") is not None
                  and (r["kind"] == "search" or (r["task"], r["seed"]) in paired)]
    summary = {}
    for label, group in (("native_confirmation", [r for r in violations if r["kind"] == "confirm_native"]),
                         ("shield_confirmation", [r for r in violations if r["kind"] == "confirm_evolved"]),
                         ("search", [r for r in violations if r["kind"] == "search"])):
        if not group:
            continue
        tips = [r for r in group if r["kind_of_violation"] == "tip"]
        speeds = [r["hand_speed_at_tilt5_mps"] for r in tips if r["hand_speed_at_tilt5_mps"] is not None]
        leads8 = [r["lead_controls"]["8"] for r in tips if r["lead_controls"]["8"] is not None]
        summary[label] = {
            "violations": len(group),
            "by_kind": {k: sum(r["kind_of_violation"] == k for r in group)
                        for k in ("tip", "lift_before_close", "drop", "other")},
            "before_closure": sum(r["before_closure"] for r in group),
            "tips_with_open_gripper": sum(r["open_gripper_before"] for r in tips),
            "tips_with_contact_in_window": sum(r["contact_in_window"] for r in tips),
            "tips_hand_speed_at_tilt5_mps_median_iqr": (
                [round(float(np.percentile(speeds, q)), 4) for q in (50, 25, 75)] if speeds else None),
            "tips_slower_than_0p10_at_tilt5": sum(s < 0.10 for s in speeds),
            "tips_peak_speed_tilt5_to_violation_mps_median": (
                round(float(np.median([r["peak_hand_speed_tilt5_to_violation_mps"] for r in tips
                                       if r["peak_hand_speed_tilt5_to_violation_mps"] is not None])), 4)
                if speeds else None),
            "tips_below_0p10_throughout_tilt5_to_violation": sum(
                r["peak_hand_speed_tilt5_to_violation_mps"] < 0.10 for r in tips
                if r["peak_hand_speed_tilt5_to_violation_mps"] is not None),
            "tips_with_speed_measured": len(speeds),
            "tips_lead_from_8deg_controls_median_iqr": (
                [float(np.percentile(leads8, q)) for q in (50, 25, 75)] if leads8 else None),
            "tips_lead_from_8deg_at_least_8_controls": sum(x >= 8 for x in leads8),
        }
    # Rule activity on the paired confirmation set (task-seeds valid in both arms).
    for label, kind in (("native", "confirm_native"), ("shield", "confirm_evolved")):
        group = [r for r in rows if r["kind"] == kind and (r["task"], r["seed"]) in paired]
        outcome = {}
        for r in group:
            cell = outcome.setdefault(r["category"], {"episodes": 0, "rule_fired": 0,
                                                      "steps_median": None, "_steps": []})
            cell["episodes"] += 1
            cell["rule_fired"] += r["rule_first_fire_step"] is not None
            cell["_steps"].append(r["steps"])
        for cell in outcome.values():
            cell["steps_median"] = float(np.median(cell.pop("_steps")))
        summary[f"rule_{label}_confirmation"] = dict(sorted(outcome.items()))
    summary["paired_task_seeds"] = len(paired)
    # The offline rule must reproduce the shield's own logged decisions.
    retraced = {j["id"]: bool((j.get("adapter") or {}).get("projected")) for j in jobs}
    summary["shield_rule_vs_logged_retraces_mismatches"] = [
        r["job"] for r in rows if r["kind"] == "confirm_evolved"
        and (r["rule_first_fire_step"] is not None) != retraced[r["job"]]]
    safe_native = [r for r in rows if r["kind"] == "confirm_native" and r["strict_safe"]]
    if safe_native:
        closure = [r["max_tilt_before_closure_deg"] for r in safe_native]
        grasp = [r["max_tilt_before_grasp_deg"] for r in safe_native]
        summary["native_strict_safe_tilt"] = {
            "episodes": len(safe_native),
            "max_before_closure_deg_median_p90_max": [round(float(np.percentile(closure, q)), 2)
                                                      for q in (50, 90, 100)],
            "max_before_grasp_deg_median_p90_max": [round(float(np.percentile(grasp, q)), 2)
                                                    for q in (50, 90, 100)],
            "reaching_8p92_before_grasp": sum(x >= 8.92 for x in grasp),
        }
    OUT.write_text(json.dumps({"summary": summary, "episodes": rows, "problems": problems,
                               "definitions": __doc__}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(len(rows), "episodes;", len(problems), "problems")


if __name__ == "__main__":
    main()
