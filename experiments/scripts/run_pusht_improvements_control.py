"""Run the frozen GPU-only before/after control and audit refinement activation."""

import json
import os
from pathlib import Path

import run_pusht_reproduction as reproduction


def main():
    root = Path("/workspace/Safety-Dial")
    design = json.loads((root / "design.json").read_text())
    if os.environ.get("RUNPOD_POD_ID") != design["pod_id"]:
        raise RuntimeError("Declared rented pod only")
    if design["arm_order"] != ["baseline-gpu", "improved-gpu"]:
        raise RuntimeError("Only the frozen six-episode comparison is permitted")
    reproduction.main()
    import numpy as np

    report = json.loads((root / "results/report.json").read_text())
    checks = []
    for case in design["case_order"]:
        folder = root / "results/improved-gpu" / case
        replans = json.loads((folder / "episode/replans.json").read_text())
        fits = [fit for plan in replans for fit in plan.get("refinements", [])]
        if not fits:
            raise RuntimeError("No refinement activation recorded")
        for fit in fits:
            if fit["feasible_candidates"]:
                assert fit["feasible_only"]
                assert fit["fitted_elites"] == min(25, fit["feasible_candidates"])
            assert fit["spread_min"] >= 0.05 - 1e-7
        with np.load(root / "assets/prior-gpu-traces" / case / "trace.npz") as old:
            with np.load(root / "results/baseline-gpu" / case / "episode/trace.npz") as new:
                n = min(len(old["states"]), len(new["states"]))
                baseline_prefix_equal = bool(np.array_equal(old["states"][:n], new["states"][:n]))
        checks.append(
            {
                "case": case,
                "refinement_iterations": len(fits),
                "few_safe_iterations": sum(0 < f["feasible_candidates"] < 25 for f in fits),
                "zero_safe_iterations": sum(f["feasible_candidates"] == 0 for f in fits),
                "best_checked_selections": sum(
                    p.get("executed") == "best_checked" for p in replans
                ),
                "baseline_prior_gpu_state_prefix_exact": baseline_prefix_equal,
            }
        )
    for arm in design["arm_order"]:
        rows = [r for r in report["rows"] if r["arm"] == arm]
        report["summaries"][arm]["safe_completion_at"] = {
            str(k): sum(r["safe_completion"] and r["actions"] <= k for r in rows)
            for k in (50, 100, 150)
        }
    report["refinement_audit"] = checks
    report["scope"] = (
        "Three known layouts on one scene/seed; bundled engineering control, not an ablation or generalization result."
    )
    reproduction.write(root / "results/report.json", report)
    reproduction.write(
        root / "results/improvements-control-complete.json",
        {
            "verified": True,
            "episodes": len(report["rows"]),
            "refinement_audit": checks,
            "report_sha256": reproduction.digest(root / "results/report.json"),
        },
    )
    print(json.dumps({"summaries": report["summaries"], "refinement_audit": checks}), flush=True)


if __name__ == "__main__":
    main()
