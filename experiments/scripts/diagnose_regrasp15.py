"""Describe what native pi0.5 does after the jar lid closes, from existing recordings.

Read-only design evidence for task 15. It loads the accounted per-control recordings
already preserved from tasks 4, 5, 7 and 8, finds each episode's phase events and
summarizes the end-effector geometry before post-closure tips and before successful
grasps. No simulator, model or network call is made.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "tracking04": "outputs/tracking04/remote/episodes/*/recordings/task_*_base.npz",
    "adapter05": "outputs/adapter05/remote/episodes/*/recordings/task_*_base.npz",
    "dsrl07": "outputs/dsrl07/remote-final/episodes/*/recordings/task_*_base.npz",
    "screen08": "outputs/screen08/completed-snapshot/raw/episodes/*/recordings/task_*_base.npz",
}
STABLE = 10  # controls the closure predicate must hold before the latch opens


def arm_of(path, source):
    job = json.loads((path.parents[1] / "job.json").read_text())
    for key in ("arm", "mode", "condition"):
        if key in job:
            return str(job[key]), job
    return "native" if source == "tracking04" else "unknown", job


def first(mask, start=0):
    idx = np.flatnonzero(mask[start:])
    return int(idx[0] + start) if idx.size else None


def stable_start(mask, length=STABLE):
    run = 0
    for t, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == length:
            return t - length + 1
    return None


def benchmark_row(path):
    rows = sorted(path.parents[1].rglob("results.jsonl"))
    if len(rows) != 1:
        raise ValueError("Episode needs exactly one benchmark results row")
    found = [json.loads(x) for x in rows[0].read_text().splitlines() if x.strip()]
    if len(found) != 1:
        raise ValueError("Episode needs exactly one benchmark results row")
    return found[0]


def window_features(d, a, b):
    a = max(a, 0)
    eef = d["eef_pos"][a:b + 1].astype(np.float64)
    jar = d["jar_pos"][a:b + 1].astype(np.float64)
    rel = eef - jar
    horizontal = np.linalg.norm(rel[:, :2], axis=1)
    speed = np.linalg.norm(np.diff(eef, axis=0), axis=1) * 20.0 if len(eef) > 1 else np.zeros(1)
    width = d["gripper_q"][a:b + 1].astype(np.float64).sum(axis=1)
    return {
        "min_horizontal_m": float(horizontal.min()),
        "end_horizontal_m": float(horizontal[-1]),
        "end_dz_m": float(rel[-1, 2]),
        "min_dz_m": float(rel[:, 2].min()),
        "max_speed_mps": float(speed.max()),
        "mean_speed_mps": float(speed.mean()),
        "end_width_m": float(width[-1]),
        "min_width_m": float(width.min()),
        "contact_steps": int((d["jar_robot_contacts"][a:b + 1] > 0).sum()),
        "grasp_steps": int((d["grasping"][a:b + 1] == 1).sum()),
    }


def episode(path, source):
    d = np.load(path)
    arm, job = arm_of(path, source)
    ap = d["ap"].astype(bool)
    closed, on_support, upright = ap[:, 0], ap[:, 2], ap[:, 3]
    contact = d["jar_robot_contacts"] > 0
    grasp = d["grasping"] == 1
    result = benchmark_row(path)
    violation = result.get("ltl_violation_step") if result.get("ltl_violated") else None
    success = result.get("success_step") if result.get("success") else None
    closure = stable_start(closed)
    row = {
        "source": source,
        "leaf": path.parents[1].name,
        "task": int(path.stem.split("_")[1]),
        "arm": arm,
        "steps": int(len(d["step"]) - 1),
        "first_contact": first(contact),
        "first_closed_raw": first(closed),
        "stable_closure": closure,
        "first_grasp": first(grasp),
        "violation": violation,
        "success": success,
        "lid_q_min": float(d["jar_joint_q"].min()),
    }
    if closure is not None:
        after = slice(closure, None)
        row["grasp_after_closure"] = first(grasp, closure)
        row["off_support_after_closure"] = first(~on_support, closure)
        row["not_upright_after_closure"] = first(~upright, closure)
        row["contact_after_closure"] = first(contact, closure)
        row["closed_toggles_after"] = int(np.abs(np.diff(closed[after].astype(int))).sum())
    if violation is not None:
        row["violation_window"] = window_features(d, violation - 40, violation)
        row["violation_kind"] = (
            "tip" if not upright[min(violation, len(upright) - 1)]
            else "off_support_open" if not on_support[min(violation, len(on_support) - 1)]
            else "other"
        )
    if row.get("grasp_after_closure") is not None:
        g = row["grasp_after_closure"]
        row["grasp_window"] = window_features(d, g - 40, g)
    if closure is not None:
        row["closure_state"] = window_features(d, closure, closure)
    return row


def outcome(row):
    c, v, s = row["stable_closure"], row["violation"], row["success"]
    if c is None:
        return "violation_before_closure" if v is not None else "no_closure"
    if v is not None and v < c:
        return "violation_before_closure"
    if s is not None and (v is None or s < v):
        return "success_after_closure"
    if v is not None:
        return "violation_after_closure"
    return "stall_after_closure"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/regrasp15/design")
    args = parser.parse_args()
    rows = []
    for source, pattern in SOURCES.items():
        for path in sorted(ROOT.glob(pattern)):
            try:
                row = episode(path, source)
            except (KeyError, ValueError) as error:
                print("skip", path, type(error).__name__, error)
                continue
            row["outcome"] = outcome(row)
            rows.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "episodes.json").write_text(json.dumps(rows, indent=1) + "\n")
    print("episodes", len(rows))
    native = [r for r in rows if r["arm"] in ("native",)]
    for label, group in (("all", rows), ("native", native)):
        counts = {}
        for r in group:
            counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
        print(label, len(group), json.dumps(counts, sort_keys=True))
    print("\nper task (native): closure / success-after / violation-after / stall-after / before")
    for task in sorted({r["task"] for r in native}):
        g = [r for r in native if r["task"] == task]
        c = [r for r in g if r["outcome"].endswith("after_closure")]
        print(task, len(g), len(c), sum(r["outcome"] == "success_after_closure" for r in g),
              sum(r["outcome"] == "violation_after_closure" for r in g),
              sum(r["outcome"] == "stall_after_closure" for r in g),
              sum(r["outcome"] == "violation_before_closure" for r in g))


if __name__ == "__main__":
    main()
