"""Machine-readable gates for conditional acquisition and closed-loop experiments."""
from __future__ import annotations

import math


def acquisition_repeatability_gate(report: dict) -> dict:
    """Require repeatable paired improvement at two budgets across three acquisition seeds.

    A diagnostic, a leaked split or episode-only transfer labels cannot unlock E5.
    Bounds are paired root-bootstrap intervals of boundary minus random FSA.
    The selected budget must also improve on the original model and retain ordinary
    latent prediction within the declared E2 tolerance in every acquisition seed.
    """
    reasons = []
    if report.get("status") != "complete":
        reasons.append("Acquisition report must be a completed, non-diagnostic run")
    if report.get("gate", {}).get("diagnostic") or not report.get("gate", {}).get("e2_passed"):
        reasons.append("E2 must pass and acquisition must not be a bounded diagnostic")
    if not report.get("split_audit", {}).get("passes"):
        reasons.append("Source trajectories are not disjoint across development and final test")
    if report.get("source_family_protocol", {}).get("version") != "block-start-goal-checkerboard-v1":
        reasons.append("A geometric start/goal transfer protocol is required")
    arms = report.get("arms", {})
    seeds = sorted(set(arms.get("random", {})) & set(arms.get("boundary", {})))
    if len(seeds) < 3:
        reasons.append("At least three paired acquisition seeds are required")
    qualifying = []
    for budget, comp in report.get("boundary_vs_random_test", {}).items():
        intervals = comp.get("paired_root_intervals_by_seed", {})
        if len(seeds) >= 3 and all(
            seed in intervals and math.isfinite(intervals[seed].get("hi", math.nan))
            and intervals[seed]["hi"] < 0 and intervals[seed].get("n_boot", 0) > 0
            for seed in seeds
        ):
            qualifying.append(int(budget))
    if len(qualifying) < 2:
        reasons.append("Boundary minus random paired intervals must exclude zero at two budgets in every seed")
    usable_budgets = []
    baseline = report.get("no_update", {}).get("test", {}).get("at_matched", {}).get("fsa", math.nan)
    base_retention = report.get("no_update_retention", {}).get("latent_mse_tf", math.nan)
    for budget in qualifying:
        valid = True
        for seed in seeds:
            points = [p for p in arms["boundary"][seed].get("curve", []) if p.get("budget_added") == budget]
            if not points:
                valid = False
                break
            point = points[0]
            score = point["eval"]["test"]["at_matched"]["fsa"]
            retention = point.get("retention", {}).get("latent_mse_tf", math.nan)
            valid &= (math.isfinite(score) and math.isfinite(baseline) and score < baseline
                      and math.isfinite(retention) and math.isfinite(base_retention)
                      and retention <= 1.15 * base_retention)
        if valid:
            usable_budgets.append(budget)
    if not usable_budgets:
        reasons.append("No qualifying budget improves on the original model while meeting retention tolerance")
    return {"passes": not reasons, "reasons": reasons, "paired_seeds": seeds,
            "qualifying_budgets": sorted(qualifying), "usable_budgets": sorted(usable_budgets),
            "selected_budget": max(usable_budgets) if usable_budgets and not reasons else None,
            "required_acquisition_seeds": 3, "required_consistent_budgets": 2,
            "retention_max_relative_mse": 1.15,
            "limitation": "Independent training seeds and goal-reaching retention are separate requirements"}


def require_closedloop_gate(report: dict) -> dict:
    gate = acquisition_repeatability_gate(report)
    if not gate["passes"]:
        raise ValueError("Closed-loop gate is not met: " + "; ".join(gate["reasons"]))
    return gate


def require_goal_retention(report: dict, checkpoint_sha256: str, *,
                           cases_path=None, case_sha256=None, n_blocks=None) -> dict:
    """Require paired goal-retention evidence for the exact final repaired weights."""
    if report.get("repaired_weights_sha256") != checkpoint_sha256:
        raise ValueError("Goal-retention report does not identify the selected repaired checkpoint")
    if report.get("comparison", {}).get("passes") is not True:
        raise ValueError("The selected repaired checkpoint has not passed goal-reaching retention")
    from helpers.pushtRetention import case_identity, compare_retention
    from helpers.pushtReplay import Root
    from helpers.runManifest import file_sha256
    import json
    from pathlib import Path

    comparison = compare_retention(report.get("base", report.get("no_update", {})),
                                   report.get("adapted", {}))
    if n_blocks is not None and comparison["n_blocks"] != n_blocks:
        raise ValueError("Goal-retention horizon differs from the frozen E2 protocol")
    if cases_path is not None:
        cases_path = Path(cases_path)
        if not case_sha256 or file_sha256(cases_path) != case_sha256:
            raise ValueError("Frozen E2 case file changed")
        if report.get("case_file_sha256") != case_sha256:
            raise ValueError("Goal-retention report uses different frozen E2 cases")
        expected = [case_identity(Root.from_dict(row)) for row in json.loads(cases_path.read_text())]
        if expected != [row.get("case_sha256") for row in report["adapted"]["rows"]]:
            raise ValueError("Goal-retention rows do not match the frozen E2 cases")
    if not comparison["passes"]:
        raise ValueError("Goal retention requires complete paired fixed-horizon evidence: "
                         + "; ".join(comparison.get("reasons", ["paired goal regression"])))
    return {"passes": True, "n_paired_cases": comparison["n_paired_cases"],
            "repaired_weights_sha256": checkpoint_sha256, "case_file_sha256": report.get("case_file_sha256"),
            "comparison": comparison}
