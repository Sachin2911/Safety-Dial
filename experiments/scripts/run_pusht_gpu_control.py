"""Matched historical/current GPU reproduction, with a frozen diagnostic trigger."""

import json
import os
from pathlib import Path

import run_pusht_reproduction as reproduction


def main():
    root = Path("/workspace/Safety-Dial")
    design = json.loads((root / "design.json").read_text())
    if os.environ.get("RUNPOD_POD_ID") != design["pod_id"]:
        raise RuntimeError("Only the declared rented host may execute this comparison")
    if design["arm_order"] != ["historical-cuda", "current-cuda"]:
        raise RuntimeError("Only the two frozen GPU arms are permitted")
    reproduction.main()

    import numpy as np

    report = json.loads((root / "results/report.json").read_text())
    if len(report["rows"]) != 6:
        raise RuntimeError("Exactly six audited episodes are required")
    comparisons = []
    for case in design["case_order"]:
        traces = []
        for arm in design["arm_order"]:
            path = root / "results" / arm / case / "episode/trace.npz"
            with np.load(path, allow_pickle=False) as data:
                traces.append({k: data[k] for k in ("states", "actions", "dense")})
        left, right = traces
        row = {"case": case}
        for key in ("states", "actions", "dense"):
            shared = min(len(left[key]), len(right[key]))
            delta = float(np.abs(left[key][:shared] - right[key][:shared]).max())
            row[key] = {
                "same_shape": left[key].shape == right[key].shape,
                "exact_prefix": bool(np.array_equal(left[key][:shared], right[key][:shared])),
                "shared_samples": shared,
                "maximum_absolute_difference": delta,
            }
        outcomes = [r for r in report["rows"] if r["case"] == case]
        row["same_outcomes"] = all(
            outcomes[0][k] == outcomes[1][k]
            for k in ("safe_completion", "violation", "unfinished", "actions", "error")
        )
        row["diagnostic_trigger"] = not row["same_outcomes"] or any(
            not row[k]["same_shape"] or row[k]["maximum_absolute_difference"] > 1e-6
            for k in ("states", "actions", "dense")
        )
        comparisons.append(row)
    report["paired_gpu_comparisons"] = comparisons
    report["conditional_diagnostic"] = {
        "required": any(r["diagnostic_trigger"] for r in comparisons),
        "protocol": design["conditional_diagnostic"],
        "status": "pending"
        if any(r["diagnostic_trigger"] for r in comparisons)
        else "not_triggered",
    }
    reproduction.write(root / "results/report.json", report)
    reproduction.write(
        root / "results/gpu-control-episodes-complete.json",
        {
            "verified": True,
            "report_sha256": reproduction.digest(root / "results/report.json"),
            "diagnostic_required": report["conditional_diagnostic"]["required"],
        },
    )
    print(json.dumps({"paired_gpu_comparisons": comparisons}), flush=True)


if __name__ == "__main__":
    main()
