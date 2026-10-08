from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import pushtE5Selection as selection
from helpers.pushtRetention import RETENTION_PROTOCOL, TERMINATION_POLICY, case_identity
from helpers.pushtReplay import Root
from helpers.runManifest import file_sha256
from helpers.studyGates import acquisition_repeatability_gate


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    witness, cases_path = tmp_path / "witness.json", tmp_path / "cases.json"
    write_json(witness, {})
    cases = [Root(i, np.zeros(7), np.ones(7), np.zeros((10, 2)), f"case-{i}", {"episode": i}) for i in range(20)]
    write_json(cases_path, [root.to_dict() for root in cases])
    feasibility = {"report": str(witness), "sha256": file_sha256(witness), "passes": True}
    monkeypatch.setattr(selection, "require_feasibility_report", lambda *args: feasibility)
    monkeypatch.setattr(selection, "bank_identity", lambda path: {"bank": path.name})
    monkeypatch.setattr(selection, "validate_bank_splits", lambda paths: {"passes": True})
    monkeypatch.setattr(selection, "validate_e5_cases", lambda *args: [{"case_id": i} for i in range(20)])
    e2_path, acquisition_path = tmp_path / "e2.json", tmp_path / "acquisition.json"
    checkpoint_root, banks = tmp_path / "checkpoints", tmp_path / "banks"
    common = {key: str(tmp_path / key) for key in ("assets_run", "probes_run", "clips_dir")}
    common["banks_dir"] = str(banks)
    e2 = {**common, "gate": {"passes": True}, "development_feasibility": feasibility,
          "goal_retention_cases": str(cases_path), "goal_retention_cases_sha256": file_sha256(cases_path),
          "goal_retention": {"n_blocks": 50}}
    write_json(e2_path, e2)
    report = {**common, "run_id": "acq-fixed", "status": "complete", "gate": {"diagnostic": False, "e2_passed": True},
              "rounds": [64, 64, 128, 256], "seed_branches": 128, "split_audit": {"passes": True},
              "source_family_protocol": {"version": "block-start-goal-checkerboard-v1"},
              "e2_report": str(e2_path), "e2_report_sha256": file_sha256(e2_path),
              "checkpoint_dir": str(checkpoint_root), "development_feasibility": feasibility,
              "evaluation_bank_identity": {name: {"bank": name} for name in ("dev", "test", "stress")},
              "no_update": {"test": {"at_matched": {"fsa": .2}}}, "no_update_retention": {"latent_mse_tf": .01},
              "arms": {"random": {}, "boundary": {}}, "boundary_vs_random_test": {}}
    for arm in report["arms"]:
        for seed in selection.SEEDS:
            report["arms"][arm][seed] = {"curve": [{"budget_added": budget, "charged_steps": 1000 + budget,
                "ledger": {"total_steps": 1000 + budget, "steps": {"branch": 1000 + budget}},
                "retention": {"latent_mse_tf": .009}, "eval": {"test": {"at_matched": {"fsa": .1}}},
                "hf_repo": "private/model", "hf_revision": "revision", "weights_sha256": "pending"}
                for budget in selection.BUDGETS]}
    for budget in selection.BUDGETS:
        report["boundary_vs_random_test"][str(budget)] = {"paired_root_intervals_by_seed": {
            seed: {"point": -.03, "lo": -.05, "hi": -.01, "n_boot": 1000} for seed in selection.SEEDS}}
    name = "acq-fixed-boundary-s0-b512"
    weights = checkpoint_root / name / "weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fixture-only weights")
    sha = file_sha256(weights)
    report["arms"]["boundary"]["0"]["curve"][-1]["weights_sha256"] = sha
    report["closedloop_gate"] = acquisition_repeatability_gate(report)
    write_json(acquisition_path, report)
    receipt_path = weights.parent / "hf_upload.json"
    write_json(receipt_path, {"repo_id": "private/model", "revision": "revision", "path": "adapted/" + name, "run_id": name})
    summary = {"n_blocks": 50, "steps_per_block": 5, "protocol": RETENTION_PROTOCOL, "termination_policy": TERMINATION_POLICY}
    rows = [{"root_id": root.root_id, "episode": root.meta["episode"], "case_sha256": case_identity(root),
             "valid": True, "requested_steps": 250, "executed_steps": 250, "censored": False, "censored_future": False,
             "arena_exit": False, "terminated": False, "truncated": False, "completed_on_verified_goal": False,
             "final_coverage": .9, "final_pose_error_px": 10.} for root in cases]
    retention = {"base": {"rows": rows, "summary": summary}, "adapted": {"rows": copy.deepcopy(rows), "summary": summary},
                 "comparison": {"passes": True}, "repaired_weights_sha256": sha, "case_file_sha256": file_sha256(cases_path)}
    retention_path = tmp_path / "retention" / name / "retention.json"
    write_json(retention_path, retention)
    transfer_path = tmp_path / "retention/transfer_retention.json"
    write_json(transfer_path, {"status": "complete", "acquisition_run": "acq-fixed", "case_file_sha256": file_sha256(cases_path),
        "n_blocks": 50, "checkpoints": {name: {"weights_sha256": sha, "report": str(retention_path)}}})
    spec = {"acquisition_report": str(acquisition_path), "e2_report": str(e2_path), "feasibility_report": str(witness),
            "retention_report": str(transfer_path), "cases": str(cases_path), "banks_dir": str(banks),
            "checkpoint_root": str(checkpoint_root), "e5_output_dir": str(tmp_path / "closedloop"),
            "e5_run_id": "pusht-closedloop-continuation-20260926-2",
            "acquisition_seed": "0", "blocks": 10, "samples": 300, "iterations": 30, "case_seed": 20261005, "lam": .05}
    return spec, report, retention_path, weights, receipt_path


def test_selection_is_deterministic_and_pins_exact_evidence_without_simulation(evidence):
    spec, _, _, weights, _ = evidence
    result = selection.resolve_selection(spec)
    assert result["status"] == "ready" and result["gate"]["passes"]
    assert result["selected"]["budget_added"] == 512
    assert result["selected"]["weights"] == str(weights)
    assert result["selected"]["acquisition_seed"] == "0"
    assert result["simulator_steps"] == result["model_queries"] == 0
    assert len(result["cases"]) == 20
    assert result["e5_run_id"] == spec["e5_run_id"] != Path(spec["e5_output_dir"]).name
    arguments = result["e5_arguments"]
    assert arguments[arguments.index("--run-id") + 1] == spec["e5_run_id"]
    assert result == selection.resolve_selection(spec)


@pytest.mark.parametrize("mismatch", ["weights", "receipt", "cases", "bank", "missing_budget", "costs"])
def test_selection_rejects_changed_or_incomplete_identity(evidence, mismatch):
    spec, report, _, weights, receipt = evidence
    if mismatch == "weights":
        weights.write_bytes(b"changed")
    elif mismatch == "receipt":
        blob = json.loads(receipt.read_text())
        blob["revision"] = "wrong"
        write_json(receipt, blob)
    elif mismatch == "cases":
        Path(spec["cases"]).write_text("[]")
    elif mismatch == "bank":
        report["evaluation_bank_identity"]["test"] = {"bank": "wrong"}
    elif mismatch == "missing_budget":
        report["arms"]["random"]["2"]["curve"].pop(0)
    else:
        report["arms"]["random"]["1"]["curve"][0]["charged_steps"] += 1
    write_json(Path(spec["acquisition_report"]), report)
    with pytest.raises(ValueError):
        selection.resolve_selection(spec)


def test_failed_repeatability_records_gate_stop_without_selecting_a_checkpoint(evidence):
    spec, report, _, weights, _ = evidence
    for comparison in report["boundary_vs_random_test"].values():
        comparison["paired_root_intervals_by_seed"]["2"]["hi"] = .1
    report["closedloop_gate"] = acquisition_repeatability_gate(report)
    write_json(Path(spec["acquisition_report"]), report)
    weights.unlink()  # The negative gate never needs to inspect or select weights.
    result = selection.resolve_selection(spec)
    assert result["status"] == "gate_stopped" and not result["gate"]["passes"]
    assert "selected" not in result and "e5_arguments" not in result


def test_failed_preselected_retention_never_searches_another_seed(evidence):
    spec, _, path, _, _ = evidence
    retention = json.loads(path.read_text())
    for row in retention["adapted"]["rows"]:
        row["final_coverage"] = .1
    retention["comparison"]["passes"] = False
    write_json(path, retention)
    result = selection.resolve_selection(spec)
    assert result["status"] == "gate_stopped" and not result["gate"]["passes"]
    assert result["selected"]["acquisition_seed"] == "0"
    assert "e5_arguments" not in result


def test_existing_closedloop_output_is_preserved(evidence):
    spec, *_ = evidence
    destination = Path(spec["e5_output_dir"])
    destination.mkdir()
    (destination / "sentinel").write_text("preserve")
    with pytest.raises(FileExistsError, match="Preserving"):
        selection.resolve_selection(spec)
    assert (destination / "sentinel").read_text() == "preserve"


def test_execution_rejects_modified_resolved_arguments_before_exec(tmp_path, monkeypatch):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/pusht_e5_selection.py"
    spec = importlib.util.spec_from_file_location("selection_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = {"status": "ready", "gate": {"passes": True}, "spec": {}, "e5_arguments": ["--repaired", "correct.pt"]}
    changed = copy.deepcopy(original)
    changed["e5_arguments"][-1] = "different.pt"
    record = tmp_path / "selection.json"
    write_json(record, changed)
    monkeypatch.setattr(sys, "argv", [str(path), "run", "--selection", str(record)])
    monkeypatch.setattr(module, "resolve_selection", lambda spec: original)
    monkeypatch.setattr(module.os, "execv", lambda *args: pytest.fail("Executed modified resolved arguments"))
    with pytest.raises(ValueError, match="changed since selection"):
        module.main()


def test_selection_preserves_legacy_output_basename_identity(evidence):
    spec, *_ = evidence
    spec.pop("e5_run_id")
    result = selection.resolve_selection(spec)
    assert result["e5_run_id"] == Path(spec["e5_output_dir"]).name


@pytest.mark.parametrize("run_id", ["../unsafe", "path/child", "not a run"])
def test_selection_rejects_invalid_explicit_e5_identity(evidence, run_id):
    spec, *_ = evidence
    spec["e5_run_id"] = run_id
    with pytest.raises(ValueError):
        selection.resolve_selection(spec)
