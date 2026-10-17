"""Preparation reports preserve failures, reservation intervals and selection gates."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers import maniguard_phase2_preparation_report as reporting
from helpers.maniguard_phase2_optim import record_hash
from helpers.maniguard_phase2_services import PRIOR_COUNTS, ReservationBudget


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def source_protocols():
    return tuple(json.loads((ROOT / "configs/maniguard/phase2" / name).read_text())
                 for name in ("preparation.json", "pilot.json"))


def measured_case(job):
    short = job["max_steps"] is not None
    return {"valid": True, "family": job["family"], "scene": job["scene"], "seed": job["seed"],
            "purpose": job["purpose"], "mode": job["mode"], "fixed_choices": job["fixed_choices"],
            "full_episode": not short, "short_check": short, "steps": 32 if short else 100,
            "costs": {key: 0 if key == "actor_rpc_attempts" else 1 for key in PRIOR_COUNTS},
            "outcome": {"U": 0, "V": False, "S_req": False},
            "decisions": [{"features": values.tolist()} for values in np.random.default_rng(4).uniform(
                -1, 1, size=(3, 64))]}


def report_fixture(tmp_path, monkeypatch, *, complete=False):
    manifest, pilot = source_protocols()
    preparation, setup = tmp_path / "preparation", tmp_path / "setup"
    write(preparation / "preparation.json", manifest)
    write(tmp_path / "pilot.json", pilot)
    for name in ("encoder-smoke.json", "encoder-loader-failure-v1.json"):
        value = json.loads((ROOT / "runs/maniguard/phase2/setup" / name).read_text())
        write(setup / name, value)
    write(setup / "reservation.json", {"start_unix_seconds": 1000, "allocated_gpus": 1})
    jobs = manifest["short_jobs"] + manifest["full_jobs"] if complete else manifest["short_jobs"][:2]
    measured = {}
    for job in jobs:
        episode = preparation / job["job_id"]
        row = measured_case(job)
        row["costs"].update(policy_control_steps=row["steps"], physics_steps=200, policy_queries=4)
        ledger = {**row["costs"], "phase": "completed", "elapsed_seconds": 10,
                  "physics_accounting_attached": True}
        row["ledger"] = ledger
        measured[str(episode)] = row
        write(episode / "job.json", {**job, "max_steps_override": job["max_steps"],
              "phase": "phase2_baseline_check" if job["baseline_executor"] else "phase2"})
        write(preparation / "attempts" / (job["job_id"] + ".json"), {
            "job": job, "output": str(episode), "short_check": job["max_steps"] is not None,
            "protocol_sha256": manifest["sha256"], "status": "valid", "returncode": 0,
            "ledger": ledger, "elapsed_seconds": 10, "timeout_seconds": 900,
            "audit_history": [{"status": "invalid", "evidence": {"costs": {"physics_steps": 9999}}}],
        })
    monkeypatch.setattr(reporting, "read_episode", lambda path: copy.deepcopy(measured[str(path)]))
    monkeypatch.setattr(reporting, "read_original", lambda path: copy.deepcopy(measured[str(path)]))
    monkeypatch.setattr(reporting, "audit_sources", lambda path: {"valid": True})
    return preparation, setup, tmp_path / "pilot.json", measured


def test_partial_report_counts_reaudited_work_once_and_emits_no_usable_prior(tmp_path, monkeypatch):
    preparation, setup, pilot, _ = report_fixture(tmp_path, monkeypatch)
    output = tmp_path / "report"
    report = reporting.build_report(preparation, setup, output, pilot_path=pilot, timestamp=1100)
    assert not report["preparation_complete"] and not report["training_ready"]
    assert not report["reference_selection"]["ready"] and report["reference_selection"]["fixed"] is None
    assert report["episode_counts_known"]["physics_steps"] == 400
    assert report["total_counts_known"]["encoder_images"] == 11  # baseline zero + one fixture + smoke ten
    assert report["reservation"]["allocated_gpu_seconds"] == 100  # not 100 + episode durations
    assert not (output / "prior-accounting.json").exists()
    assert report["accounting"]["attempt_count"] == 2
    assert all(row["audit_history"][0]["status"] == "invalid" for row in report["terminal_attempts"])
    with pytest.raises(FileExistsError):
        reporting.build_report(preparation, setup, output, pilot_path=pilot)


def test_pending_receipt_does_not_open_active_trace_or_ledger(tmp_path, monkeypatch):
    preparation, setup, pilot, measured = report_fixture(tmp_path, monkeypatch)
    path = preparation / "attempts/short-jar-identity.json"
    receipt = json.loads(path.read_text())
    receipt.update(status="started")
    receipt.pop("returncode")
    receipt.pop("ledger")
    write(path, receipt)
    active = Path(receipt["output"])
    (active / "ledger.json").write_text("not valid JSON and must not be read")
    measured.pop(str(active))  # Any attempt to behaviorally read this episode raises.
    report = reporting.build_report(preparation, setup, tmp_path / "report", pilot_path=pilot, timestamp=1100)
    assert report["pending_receipts"] == ["short-jar-identity"]
    assert len(report["unknown_counts"]) == len(PRIOR_COUNTS)
    assert report["accounting"]["ceilings"]["short"]["attempts"] == 2
    assert not report["prior_accounting_ready"]


def test_completed_prior_uses_immutable_sources_and_continuous_reservation(tmp_path, monkeypatch):
    preparation, setup, pilot, _ = report_fixture(tmp_path, monkeypatch, complete=True)
    output = tmp_path / "report"
    report = reporting.build_report(preparation, setup, output, pilot_path=pilot, timestamp=1100)
    assert report["preparation_complete"] and report["prior_accounting_ready"], report["invalid_reasons"]
    assert not report["training_ready"]
    prior = json.loads((output / "prior-accounting.json").read_text())
    assert set(prior["counts"]) == set(PRIOR_COUNTS)
    assert prior["counts"]["physics_steps"] == 36 * 200
    assert prior["counts"]["actor_rpc_attempts"] == 0
    assert all(Path(source["path"]).is_relative_to(output) for source in prior["sources"])
    launcher = output / "audit_source/experiments/scripts/maniguard_phase2_prepare.py"
    assert launcher.read_bytes() == (ROOT / "experiments/scripts/maniguard_phase2_prepare.py").read_bytes()
    assert any(source["path"] == str(launcher) for source in prior["sources"])
    write(setup / "reservation.json", {"edited_after_report": True})
    for path in (preparation / "attempts").glob("*.json"):
        path.write_text("edited after report")
    budget = ReservationBudget(prior, json.loads(pilot.read_text())["resource_envelope"], wall=lambda: 1150)
    assert budget.snapshot()["allocated_gpu_seconds"] == 150
    assert report["sha256"] == record_hash({key: value for key, value in report.items() if key != "sha256"})


def test_missing_failed_work_count_blocks_prior_even_when_all_receipts_terminal(tmp_path, monkeypatch):
    preparation, setup, pilot, measured = report_fixture(tmp_path, monkeypatch, complete=True)
    row = measured[str(preparation / "short-jar-identity")]
    row["valid"] = False
    del row["costs"]["physics_steps"]
    report = reporting.build_report(preparation, setup, tmp_path / "report", pilot_path=pilot, timestamp=1100)
    assert report["preparation_complete"] and not report["prior_accounting_ready"]
    assert {"job_id": "short-jar-identity", "count": "physics_steps"} in report["unknown_counts"]


def test_reference_selection_requires_all_24_and_applies_declared_ties():
    manifest, _ = source_protocols()
    rows = {job["job_id"]: measured_case(job) for job in manifest["full_jobs"]}
    absent = rows.pop(manifest["full_jobs"][-1]["job_id"])
    assert not reporting.select_references(manifest, rows)["ready"]
    rows[manifest["full_jobs"][-1]["job_id"]] = absent
    selected = reporting.select_references(manifest, rows)
    assert selected["fixed"]["name"] == "cap006"  # complete tie resolves by name
    for name, row in rows.items():
        if name.endswith("-smooth075"):
            row["outcome"].update(U=1, S_req=True)
    assert reporting.select_references(manifest, rows)["fixed"]["name"] == "smooth075"
    for name, row in rows.items():
        row["outcome"].update(U=0, S_req=False, V=name.endswith("-smooth075"))
    assert reporting.select_references(manifest, rows)["fixed"]["name"] == "cap006"
    for name, row in rows.items():
        row["outcome"]["V"] = False
        row["costs"]["policy_queries"] = 2 if name.endswith("-cap006") else 1
    selected = reporting.select_references(manifest, rows)
    assert selected["fixed"]["name"] == "smooth075"
    assert not selected["contextual_threshold_changed"]


def test_constant_features_do_not_fake_context_dependence_from_recurrent_history():
    _, pilot = source_protocols()
    report = reporting.activation_diagnostic(pilot, [{"features": np.zeros((8, 64)).tolist()}])
    assert report["valid"] and not report["passed"]
    assert not any(row["choices_depend_on_inputs"] for row in report["candidates"])
    assert not report["controller_changes"] and not report["physical_work"]


def test_recorded_varying_inputs_activate_sampled_and_mutated_controllers():
    _, pilot = source_protocols()
    features = np.random.default_rng(83).uniform(-1, 1, (20, 64)).tolist()
    report = reporting.activation_diagnostic(pilot, [{"features": features}])
    assert report["passed"] and report["sampled_choice_context_dependence"]
    assert report["mutated_choice_context_dependence"]
    identity = [row for row in report["candidates"] if row["origin"] == "identity"]
    assert all(not row["choices_depend_on_inputs"] for row in identity)


def test_failed_terminal_lower_bound_ledger_cannot_become_exact_prior(tmp_path, monkeypatch):
    preparation, setup, pilot, measured = report_fixture(tmp_path, monkeypatch, complete=True)
    row = measured[str(preparation / "short-jar-identity")]
    row["valid"] = False
    row["ledger"]["phase"] = "process_exited_without_result"
    report = reporting.build_report(preparation, setup, tmp_path / "report", pilot_path=pilot, timestamp=1100)
    assert not report["prior_accounting_ready"] and report["count_totals_are_lower_bounds"]
    assert any(item.get("reason") == "incomplete ledger is a lower bound" for item in report["unknown_counts"])
    assert report["episode_counts_known"]["physics_steps"] == 36 * 200


@pytest.mark.parametrize("field,value", [
    ("error", "launch failed"), ("cleanup_error", "cleanup deadline expired"),
    ("timed_out", True), ("interrupted", True), ("orphaned_descendants", [812]),
    ("process_group_remaining", True),
])
def test_process_failure_flags_match_launcher_and_cannot_be_erased(tmp_path, monkeypatch, field, value):
    from scripts.maniguard_phase2_prepare import process_succeeded

    preparation, setup, pilot, _ = report_fixture(tmp_path, monkeypatch)
    path = preparation / "attempts/short-jar-identity.json"
    receipt = json.loads(path.read_text())
    receipt.update(status="invalid", process={field: value})
    write(path, receipt)
    assert not process_succeeded(receipt)
    output = tmp_path / "report"
    report = reporting.build_report(preparation, setup, output, pilot_path=pilot, timestamp=1100)
    terminal = next(row for row in report["terminal_attempts"] if row["job_id"] == "short-jar-identity")
    assert not terminal["valid"] and not terminal["process_audit"]["valid"]
    assert terminal["process_audit"]["process"][field] == value
    assert any(field in reason for reason in terminal["invalid_reasons"])
    audited = json.loads((output / "audited_episodes/short-jar-identity.json").read_text())
    assert not audited["valid"] and audited["process_audit"] == terminal["process_audit"]
    assert report["episode_counts_known"]["physics_steps"] == 400  # Failure still consumes its work.


@pytest.mark.parametrize("receipt", [
    {"returncode": 0},
    {"returncode": 0, "process": {"cleanup_error": None, "timed_out": False,
                                  "orphaned_descendants": [], "process_group_remaining": False}},
    {"returncode": 1},
    {"returncode": 0, "process_error": "launch failed before process evidence"},
])
def test_process_audit_preserves_legacy_success_and_launcher_failures(receipt):
    from scripts.maniguard_phase2_prepare import process_succeeded

    assert reporting.audit_process(receipt)["valid"] == process_succeeded(receipt)


def test_zero_exit_cleanup_failure_cannot_select_reference_or_pass_freeze_review(tmp_path, monkeypatch):
    preparation, setup, pilot, _ = report_fixture(tmp_path, monkeypatch, complete=True)
    path = next((preparation / "attempts").glob("full-*.json"))
    receipt = json.loads(path.read_text())
    receipt.update(status="invalid", process={"cleanup_error": "orphan cleanup failed"})
    write(path, receipt)
    report = reporting.build_report(preparation, setup, tmp_path / "report", pilot_path=pilot, timestamp=1100)
    assert not report["reference_selection"]["ready"]
    assert not report["ready_for_freeze_review"]
    assert report["prior_accounting_ready"]  # Exact cost accounting may include invalid attempts.
    assert report["total_counts_known"]["physics_steps"] == 7200


def test_terminal_setup_lifecycles_and_diagnosis_are_immutable_without_extra_gpu_time(tmp_path, monkeypatch):
    preparation, setup, pilot, _ = report_fixture(tmp_path, monkeypatch, complete=True)
    failed = {"family": "clutter", "status": "failed", "started_at": 1005, "finished_at": 1020,
              "elapsed_seconds": 15, "error": "TimeoutError: Managed policy startup deadline expired"}
    successful = {"family": "clutter", "status": "started", "started_at": 1030, "finished_at": 1040}
    diagnosis = {"classification": "provenance schema mismatch",
                 "historical_server_traceback_recovered": False,
                 "evidence_scope": "Inferred from receipt and source; original traceback was not retained."}
    legacy = {"family": "stack", "started_unix_seconds": 1040, "observed_ready_unix_seconds": 1050,
              "active_episode_pids": [], "metadata": {"serve_config": "stack"}}
    values = {"service-clutter-v1/service.json": failed,
              "service-clutter-v1/diagnosis.json": diagnosis,
              "service-clutter-v2/service.json": successful,
              "policy-switch-stack.json": legacy}
    for name, value in values.items():
        write(setup / name, value)
    (setup / "service-clutter-v2/policy.log").write_text("still mutable while server runs")
    output = tmp_path / "report"
    report = reporting.build_report(preparation, setup, output, pilot_path=pilot, timestamp=1100)
    assert report["prior_accounting_ready"] and not report["pending_setup_lifecycle"]
    assert {row["status"] for row in report["setup_lifecycle"]} == {"failed", "started", "ready_observed"}
    assert report["reservation"]["allocated_gpu_seconds"] == 100  # Includes both loads and switch once.
    assert report["total_counts_known"]["physics_steps"] == 7200
    assert report["setup_counts_known"]["encoder_images"] == 10
    assert not list(output.rglob("policy.log"))
    prior = json.loads((output / "prior-accounting.json").read_text())
    for name, value in values.items():
        snapshot = output / "inputs/setup" / name
        assert json.loads(snapshot.read_text()) == value
        assert any(source["path"] == str(snapshot) for source in prior["sources"])
        write(setup / name, {"edited_after_report": True})
    captured_diagnosis = json.loads((output / "inputs/setup/service-clutter-v1/diagnosis.json").read_text())
    assert captured_diagnosis["historical_server_traceback_recovered"] is False
    budget = ReservationBudget(prior, json.loads(pilot.read_text())["resource_envelope"], wall=lambda: 1150)
    assert budget.snapshot()["allocated_gpu_seconds"] == 150


def test_pending_setup_lifecycle_is_not_a_prior_source_and_blocks_prior(tmp_path, monkeypatch):
    preparation, setup, pilot, _ = report_fixture(tmp_path, monkeypatch, complete=True)
    write(setup / "service-clutter-v3/service.json", {
        "status": "checking", "started_at": 1080, "finished_at": 1085,
    })  # PolicyService also writes finished_at on intermediate persists.
    (setup / "service-clutter-v3/diagnosis.json").write_text("active, must not be parsed")
    write(setup / "policy-switch-stack.json", {"family": "stack", "started_unix_seconds": 1090})
    output = tmp_path / "report"
    report = reporting.build_report(preparation, setup, output, pilot_path=pilot, timestamp=1100)
    assert not report["prior_accounting_ready"] and not (output / "prior-accounting.json").exists()
    assert report["setup_lifecycle"] == []
    assert set(report["pending_setup_lifecycle"]) == {"service-clutter-v3/service.json", "policy-switch-stack.json"}
    assert not (output / "inputs/setup/service-clutter-v3/service.json").exists()
    assert not (output / "inputs/setup/policy-switch-stack.json").exists()
    assert report["episode_counts_known"]["physics_steps"] == 7200


def test_malformed_terminal_lifecycle_blocks_prior(tmp_path, monkeypatch):
    preparation, setup, pilot, _ = report_fixture(tmp_path, monkeypatch, complete=True)
    write(setup / "service-clutter-v1/service.json", {"status": "failed", "started_at": 1080})
    report = reporting.build_report(preparation, setup, tmp_path / "report", pilot_path=pilot, timestamp=1100)
    assert not report["prior_accounting_ready"]
    assert any("valid start/finish interval" in reason for reason in report["invalid_reasons"])
