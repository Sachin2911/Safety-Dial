"""Descriptive task-progress and geometry diagnostics from already saved episodes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.safety_data import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = args.root
    panel = json.loads((out / "panel.json").read_text())
    report = json.loads((out / "report.json").read_text())
    design = json.loads((out / "preparation-freeze.json").read_text())["design"]
    name = report["environment"]
    radius = report["motion_report"]["spec"]["body_radius"]
    allowance = report["motion_report"]["calibration"]["allowance"]
    margin = design["margin"][name]
    geometry = []
    for case in panel["cases"]:
        shape = case["shape"]
        if shape["kind"] == "gate":
            gap = shape["gap"] - 2 * (radius + margin + allowance)
            geometry.append(
                {
                    "case": case["id"],
                    "physical_gap_width": shape["gap"],
                    "gap_after_body_and_margin": shape["gap"] - 2 * (radius + margin),
                    "gap_after_calibration_tube": gap,
                    "direct_gap_closed_under_tube": gap <= 0,
                }
            )
    rows = []
    for path in sorted(out.glob("evaluation/**/result.json")):
        r = json.loads(path.read_text())
        folder = path.parent
        with np.load(folder / "trace.npz") as z:
            state = z["states"]
        with np.load(out / "inputs" / (r["case_id"] + ".npz")) as z:
            goal = z["goal"]
        row = {
            "case": r["case_id"],
            "configuration": r["configuration"],
            "condition": r["condition"],
            "safe_completion": r["safe_completion"],
            "actions": r["actions"],
            "stopping": "goal"
            if r["success"]
            else ("no_checked_plan" if r["controller_failure"] else "action_cap"),
            "protected_body_path_length": r["protected_body_path_length"],
        }
        if name == "tworoom":
            error = np.linalg.norm(state[:, -2:] - goal, axis=-1)
        elif name == "reacher":
            error = np.abs(state[:, :2] - goal).max(-1)
            wrapped = np.abs((state[:, :2] - goal + np.pi) % (2 * np.pi) - np.pi).max(-1)
            row.update(
                final_wrapped_joint_error=float(wrapped[-1]),
                best_wrapped_joint_error=float(wrapped.min()),
            )
        else:
            monitor = [
                json.loads(line)
                for line in (folder / "object-states.jsonl").read_text().splitlines()
            ]
            xyz = np.asarray([p["cube_xyz"] for p in monitor])
            error = np.linalg.norm(xyz - goal, axis=-1)
            row.update(
                object_path_length=float(np.linalg.norm(np.diff(xyz, axis=0), axis=-1).sum()),
                maximum_object_displacement=float(np.linalg.norm(xyz - xyz[0], axis=-1).max()),
            )
        row.update(
            initial_task_error=float(error[0]),
            final_task_error=float(error[-1]),
            best_task_error=float(error.min()),
        )
        rows.append(row)
    result = {
        "environment": name,
        "geometry": geometry,
        "reserved_task_progress": rows,
        "scope": "Posthoc descriptive diagnostics only. No new simulator/model queries, selection changes or endpoint changes.",
        "interpretation": "Endpoint clearance was screened; whole-task connectivity and action-budget feasibility were not certified. A closed direct gate opening does not prove all routes impossible. Wrapped Reacher error is secondary; the native unwrapped success rule remains primary.",
    }
    write_json(out / "task-progress-diagnostics.json", result)
    print(
        json.dumps({"environment": name, "reserved_rows": len(rows), "gate_diagnostics": geometry}),
        flush=True,
    )


if __name__ == "__main__":
    main()
