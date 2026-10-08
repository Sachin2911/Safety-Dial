"""Predeclared checkpoint coverage for the E4 goal-retention report."""
from __future__ import annotations

from pathlib import Path

from helpers.runManifest import file_sha256
from helpers.studyGates import acquisition_repeatability_gate


def checkpoint_plan(report: dict, checkpoint_root: Path) -> list[dict]:
    """Largest budget of every arm/seed; include E5's passing budget when different.

    Checkpoints are not ranked on goal-retention outcomes, and a failed checkpoint is
    retained in the report. Intermediate acquisition curves do not inherit these results.
    """
    if (report.get("status") != "complete" or report.get("gate", {}).get("diagnostic")
            or not report.get("gate", {}).get("e2_passed")
            or not report.get("split_audit", {}).get("passes")):
        raise ValueError("Transfer retention requires a completed qualified acquisition study")
    if report.get("source_family_protocol", {}).get("version") != "block-start-goal-checkerboard-v1":
        raise ValueError("Transfer retention requires the geometric source-family protocol")
    arms = report.get("arms", {})
    if not {"random", "boundary"} <= set(arms):
        raise ValueError("Transfer retention requires both random and boundary")
    seeds = set(arms["random"])
    if len(seeds) < 3 or any(set(values) != seeds for values in arms.values()):
        raise ValueError("Every reported acquisition arm needs the same three or more seeds")
    budgets, all_curves = set(), set()
    for values in arms.values():
        for run in values.values():
            if not run.get("curve"):
                raise ValueError("Missing acquisition curve")
            sequence = tuple(point["budget_added"] for point in run["curve"])
            if list(sequence) != sorted(set(sequence)):
                raise ValueError("Acquisition budget curve is not ordered and unique")
            all_curves.add(sequence)
            budgets.add(max(sequence))
    if len(budgets) != 1 or len(all_curves) != 1:
        raise ValueError("Acquisition arms have different final budgets")
    closedloop = acquisition_repeatability_gate(report)
    if closedloop["passes"]:
        budgets.add(closedloop["selected_budget"])
    plan = []
    for arm in sorted(arms):
        for seed in sorted(seeds):
            for budget in sorted(budgets):
                points = [p for p in arms[arm][seed]["curve"] if p["budget_added"] == budget]
                if len(points) != 1:
                    raise ValueError("Missing or duplicate transfer checkpoint budget")
                point = points[0]
                name = f"{report['run_id']}-{arm}-s{seed}-b{budget}"
                path = Path(checkpoint_root) / name / "weights.pt"
                if file_sha256(path) != point.get("weights_sha256"):
                    raise ValueError(f"Transfer checkpoint changed: {name}")
                if not point.get("hf_revision") or not point.get("hf_repo"):
                    raise ValueError(f"Pinned private checkpoint receipt is missing: {name}")
                plan.append({"name": name, "arm": arm, "acquisition_seed": seed,
                    "budget_added": budget, "weights": str(path.resolve()),
                    "weights_sha256": point["weights_sha256"], "hf_revision": point["hf_revision"],
                    "hf_repo": point["hf_repo"]})
    return plan
