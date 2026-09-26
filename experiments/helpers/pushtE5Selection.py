"""Resolve a predeclared E5 checkpoint from immutable E3/E4 evidence, without simulation."""
from __future__ import annotations

import json
from pathlib import Path

from helpers.pushtFeasibility import require_feasibility_report
from helpers.pushtRetention import case_identity, compare_retention
from helpers.pushtReplay import Root
from helpers.runManifest import file_sha256, validate_run_id
from helpers.splitIntegrity import bank_identity, validate_bank_splits
from helpers.studyGates import acquisition_repeatability_gate, require_goal_retention

BUDGETS = [64, 128, 256, 512]
SEEDS = {"0", "1", "2"}


def _json(path):
    return json.loads(Path(path).read_text())


def qualified_protocol(report):
    """Require the complete fixed prospective design, not a favorable subset of it."""
    if (report.get("status") != "complete" or report.get("gate", {}).get("diagnostic") is not False
            or report.get("gate", {}).get("e2_passed") is not True
            or report.get("rounds") != [64, 64, 128, 256] or report.get("seed_branches") != 128
            or set(report.get("arms", {})) != {"random", "boundary"}):
        raise ValueError("E5 resolution requires the complete predeclared random/boundary acquisition design")
    for arm in ("random", "boundary"):
        if set(report["arms"][arm]) != SEEDS:
            raise ValueError("E5 resolution requires the predeclared three acquisition seeds 0, 1, 2")
        for seed in SEEDS:
            curve = report["arms"][arm][seed].get("curve", [])
            if [point.get("budget_added") for point in curve] != BUDGETS:
                raise ValueError("Every arm/seed must contain all four additional budgets")
            for point in curve:
                ledger = point.get("ledger", {})
                cost = point.get("charged_steps")
                if type(cost) is not int or cost <= 0 or ledger.get("total_steps") != cost or sum(ledger.get("steps", {}).values()) != cost:
                    raise ValueError("Acquisition checkpoint has an invalid charged-step ledger")
    for seed in SEEDS:
        if [p["charged_steps"] for p in report["arms"]["random"][seed]["curve"]] != [p["charged_steps"] for p in report["arms"]["boundary"][seed]["curve"]]:
            raise ValueError("Paired acquisition arms have unequal simulator costs")


def validate_e5_cases(banks, report, e2, spec):
    from helpers.branchBank import Bank
    from helpers.pushtClosedLoop import closedloop_arm_specs, fixed_cases, penalty_reference_protocol
    from helpers.pushtLayouts import load_layouts
    from helpers.pushtSourceFamilies import validate_geometric_bank

    penalty_reference_protocol(spec["lam"])
    if spec["blocks"] <= 0 or spec["samples"] < 30 or spec["iterations"] <= 0:
        raise ValueError("Invalid predeclared E5 planning budget")
    margin = float(report["no_update"]["margin_matched_dev"])
    fixed = float(e2["arms"]["fixed_margin"]["margin"])
    closedloop_arm_specs(None, None, dial=margin, fixed_margin=fixed)
    bank = Bank(banks / "test")
    try:
        validate_geometric_bank(bank)
        layouts = load_layouts(banks / "test" / "layouts.json")[0]
        return fixed_cases(bank, layouts, episodes_per_layout=10, seed=spec["case_seed"], margin=max(margin, fixed))
    finally:
        bank.h5.close()


def resolve_selection(spec: dict) -> dict:
    """Scientific negatives return a false gate; mismatched evidence raises ValueError."""
    paths = {key: Path(spec[key]).resolve() for key in
             ("acquisition_report", "e2_report", "feasibility_report", "retention_report", "cases",
              "banks_dir", "checkpoint_root", "e5_output_dir")}
    e5_run_id = validate_run_id(spec.get("e5_run_id") or paths["e5_output_dir"].name)
    acquisition, e2 = _json(paths["acquisition_report"]), _json(paths["e2_report"])
    qualified_protocol(acquisition)
    run_id = validate_run_id(acquisition["run_id"])
    seed = str(spec["acquisition_seed"])
    if seed not in SEEDS:
        raise ValueError("Select a predeclared acquisition seed, independent of retention outcomes")
    if (Path(acquisition["e2_report"]).resolve() != paths["e2_report"]
            or acquisition.get("e2_report_sha256") != file_sha256(paths["e2_report"])
            or e2.get("gate", {}).get("passes") is not True):
        raise ValueError("E3 does not identify this exact passing E2 report")
    for key in ("banks_dir", "assets_run", "probes_run", "clips_dir"):
        if Path(acquisition[key]).resolve() != Path(e2[key]).resolve():
            raise ValueError(f"E3/E2 {key} identities differ")
    if Path(acquisition["banks_dir"]).resolve() != paths["banks_dir"]:
        raise ValueError("Selected final-test banks differ from E3")
    if Path(acquisition["checkpoint_dir"]).resolve() != paths["checkpoint_root"]:
        raise ValueError("Checkpoint parent differs from E3's recorded output path")
    identities = {name: bank_identity(paths["banks_dir"] / name) for name in ("dev", "test", "stress")}
    if acquisition.get("evaluation_bank_identity") != identities:
        raise ValueError("The evaluation banks changed since E3")
    validate_bank_splits({name: paths["banks_dir"] / name for name in identities})
    feasibility = require_feasibility_report(paths["feasibility_report"], paths["banks_dir"] / "dev")
    for record in (e2.get("development_feasibility", {}), acquisition.get("development_feasibility", {})):
        if (record.get("sha256") != feasibility["sha256"]
                or Path(record.get("report", "")).resolve() != paths["feasibility_report"]):
            raise ValueError("Feasible-route evidence differs from the E2/E3 gate")
    case_sha = file_sha256(paths["cases"])
    if (Path(e2["goal_retention_cases"]).resolve() != paths["cases"]
            or case_sha != e2.get("goal_retention_cases_sha256")):
        raise ValueError("Retention must reuse the exact frozen E2 cases")
    transfer = _json(paths["retention_report"])
    if (transfer.get("status") != "complete" or transfer.get("acquisition_run") != run_id
            or transfer.get("case_file_sha256") != case_sha
            or transfer.get("n_blocks") != e2["goal_retention"]["n_blocks"]):
        raise ValueError("Final-checkpoint retention does not match the completed E3/E2 protocol")
    gate = acquisition_repeatability_gate(acquisition)
    if acquisition.get("closedloop_gate") != gate:
        raise ValueError("Stored E3 repeatability gate differs from recomputed evidence")
    result = {"spec": spec, "e5_run_id": e5_run_id, "status": "gate_stopped", "gate": gate, "simulator_steps": 0,
              "model_queries": 0, "selection_rule": "boundary arm, predeclared acquisition seed, largest gate-qualified budget",
              "bank_identities": identities,
              "input_sha256": {str(paths[key]): file_sha256(paths[key]) for key in
                               ("acquisition_report", "e2_report", "feasibility_report", "retention_report", "cases")}}
    if not gate["passes"]:
        return result
    budget = gate["selected_budget"]
    point = next(p for p in acquisition["arms"]["boundary"][seed]["curve"] if p["budget_added"] == budget)
    name = f"{run_id}-boundary-s{seed}-b{budget}"
    weights = paths["checkpoint_root"] / name / "weights.pt"
    sha = file_sha256(weights)
    if sha != point.get("weights_sha256"):
        raise ValueError("Selected checkpoint SHA256 differs from E3")
    receipt_path = weights.parent / "hf_upload.json"
    receipt = _json(receipt_path)
    if (not point.get("hf_repo") or not point.get("hf_revision")
            or receipt.get("repo_id") != point["hf_repo"] or receipt.get("revision") != point["hf_revision"]
            or receipt.get("path") != f"adapted/{name}" or receipt.get("run_id") != name):
        raise ValueError("Selected checkpoint lacks its matching pinned Hugging Face upload receipt")
    entry = transfer.get("checkpoints", {}).get(name, {})
    retention_path = paths["retention_report"].parent / name / "retention.json"
    if entry.get("weights_sha256") != sha or Path(entry.get("report", "")).resolve() != retention_path:
        raise ValueError("Retention does not contain the selected acquisition checkpoint")
    retention = _json(retention_path)
    if retention.get("repaired_weights_sha256") != sha or retention.get("case_file_sha256") != case_sha:
        raise ValueError("Selected retention model/case hashes differ")
    expected = [case_identity(Root.from_dict(row)) for row in _json(paths["cases"])]
    for model in ("base", "adapted"):
        if [row.get("case_sha256") for row in retention.get(model, {}).get("rows", [])] != expected:
            raise ValueError("Selected retention rows do not identify the frozen E2 cases in order")
    comparison = compare_retention(retention["base"], retention["adapted"])
    result["selected"] = {"name": name, "acquisition_seed": seed, "budget_added": budget,
                          "weights": str(weights), "weights_sha256": sha, "retention_report": str(retention_path),
                          "hf_repo": point["hf_repo"], "hf_revision": point["hf_revision"]}
    result["input_sha256"].update({str(path): file_sha256(path) for path in (weights, receipt_path, retention_path)})
    if comparison["passes"] is not True or retention.get("comparison", {}).get("passes") is not True:
        result["gate"] = {"passes": False, "reason": "The preselected checkpoint failed exact-case goal retention", "retention": comparison}
        return result
    result["goal_retention"] = require_goal_retention(retention, sha, cases_path=paths["cases"],
        case_sha256=case_sha, n_blocks=e2["goal_retention"]["n_blocks"])
    result["cases"] = validate_e5_cases(paths["banks_dir"], acquisition, e2, spec)
    if paths["e5_output_dir"].exists():
        raise FileExistsError(f"Preserving existing E5 output: {paths['e5_output_dir']}")
    result["status"] = "ready"
    result["e5_arguments"] = ["--acquisition-report", str(paths["acquisition_report"]), "--repaired", str(weights),
        "--retention-report", str(retention_path), "--feasibility-report", str(paths["feasibility_report"]),
        "--banks-dir", str(paths["banks_dir"]), "--output-dir", str(paths["e5_output_dir"]), "--run-id", e5_run_id,
        "--acquisition-seed", seed, "--episodes-per-layout", "10", "--blocks", str(spec["blocks"]),
        "--samples", str(spec["samples"]), "--iterations", str(spec["iterations"]),
        "--case-seed", str(spec["case_seed"]), "--lam", str(spec["lam"])]
    return result
