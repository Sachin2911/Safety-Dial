"""Task 15 recovery: cohort integrity, cumulative admission, gates and completion."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECOVERY = json.loads((ROOT / "configs/regrasp15r1/recovery.json").read_text())


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"experiments/scripts/{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def validated(job, attempt=1, steps=2000, stage="stage_c", arm="native"):
    return {"event": "validated", "job": job, "attempt": attempt, "stage": stage, "arm": arm,
            "outcome": {"steps": steps}}


def start(job, attempt=1):
    return {"event": "start_reserved", "job": job, "attempt": attempt}


def exit_(job, attempt=1, complete=True):
    return {"event": "simulator_exit", "job": job, "attempt": attempt, "completed_row": complete}


def test_recovery_jobs_replicate_the_frozen_schedule(tmp_path, monkeypatch):
    run = module("run_regrasp15")
    design = json.loads((ROOT / "configs/regrasp15/design.json").read_text())
    decision = json.loads((ROOT / RECOVERY["stage_c_decision"]["path"]).read_text())
    monkeypatch.setattr(run, "OUT", tmp_path)
    (tmp_path / "stage-a-pass.json").write_text(json.dumps({"pass": True}))
    (tmp_path / "stage-c-decision.json").write_text(json.dumps(
        {**decision, "evidence_sha256": {}}))
    queue = object.__new__(run.Queue)
    queue.design, queue.stage = design, "stage_c"
    queue.history = [{"event": "validated", "job": j["job"]} for j in design["stage_a_jobs"]]
    assert queue.stage_jobs() == RECOVERY["stage_c_jobs"]
    assert RECOVERY["engineering_jobs"] == design["stage_a_jobs"]
    assert {j["arm"] for j in RECOVERY["stage_c_jobs"]} == {"native", "select-0.06"}
    assert len(RECOVERY["stage_c_jobs"]) == 96 and len(RECOVERY["stage_d_jobs"]) == 48
    assert [j["job"] for j in RECOVERY["stage_d_jobs"]] == list(range(500, 548))
    assert {j["arm"] for j in RECOVERY["stage_d_jobs"]} == {"project-0.10"}
    ids = [j["job"] for k in ("engineering_jobs", "stage_c_jobs", "stage_d_jobs")
           for j in RECOVERY[k]]
    assert len(ids) == len(set(ids))
    tasks = {j["task"] for k in ("stage_c_jobs", "stage_d_jobs") for j in RECOVERY[k]}
    assert tasks == {0, 3, 6, 9}


def test_stage_d_jobs_match_the_declaration_rule():
    plan = module("plan_regrasp15r1")
    declaration = json.loads((ROOT / "progress/receipts/15-regrasp-stage-d.json").read_text())
    assert plan.stage_d_jobs(declaration) == RECOVERY["stage_d_jobs"]


def test_queue_is_rebound_to_the_recovery_namespace():
    recovery = module("run_regrasp15r1")
    assert recovery.base.OUT == ROOT / "outputs/regrasp15r1"
    assert recovery.base.SERVER_LEDGER == ROOT / "outputs/regrasp15r1/server/candidates.jsonl"
    assert recovery.RENTAL.name == "rental-regrasp15r1-20261026.json"


def test_completion_requires_every_job_and_no_unresolved_start():
    run = module("run_regrasp15r1")
    jobs = [{"job": 1}, {"job": 2}]
    history = [start(1), exit_(1), validated(1), start(2), exit_(2), validated(2)]
    assert run.completion(history, jobs, True)["complete"]
    assert not run.completion(history, jobs, False)["complete"]
    missing = run.completion(history[:3], jobs, True)
    assert missing["missing_jobs"] == [2] and not missing["complete"]
    hanging = run.completion(history[:3] + [start(2)], jobs, True)
    assert hanging["unresolved_starts"] == [[2, 1]] and not hanging["complete"]
    abandoned = history[:3] + [start(2), exit_(2, complete=False), start(2, 2),
                               exit_(2, 2, complete=False), {"event": "job_abandoned", "job": 2}]
    result = run.completion(abandoned, jobs, True)
    assert result["abandoned_jobs"] == [2] and not result["complete"]


def test_cumulative_usage_imports_history_and_charges_worst_cases():
    run = module("run_regrasp15r1")
    imports = RECOVERY["imports"]
    history = [start(300), exit_(300), validated(300, steps=1601), start(301),
               exit_(301, complete=False), start(301, 2)]
    used = run.cumulative_usage(history, RECOVERY, decodes=640)
    assert used["simulator_starts"] == imports["simulator_starts"] + 3
    assert used["replacements"] == imports["replacements"] + 1
    assert used["policy_controls"] == imports["policy_controls"] + 1601 + 2 * 2000
    assert used["native_queries"] == (imports["native_queries"] + 201 + 1 + 2 * 251
                                      + RECOVERY["probe_native_queries_reserved"])
    assert used["candidate_decodes"] == imports["candidate_decodes"] + 640


def test_refusals_cover_every_ceiling_and_the_admission_margin():
    run = module("run_regrasp15r1")
    ceilings = RECOVERY["ceilings"]
    fine = {"simulator_starts": 100, "replacements": 0, "policy_controls": 200_000,
            "native_queries": 30_000, "candidate_decodes": 60_000}
    assert run.refusal_reasons(fine, RECOVERY, 1, now=0, admission=1, stop=False) == []
    assert run.refusal_reasons(fine, RECOVERY, 1, now=1, admission=1, stop=True) == [
        "less_than_45_minutes_before_deadline", "stop_file"]
    for key in fine:
        over = dict(fine, **{key: ceilings[key]})
        reasons = run.refusal_reasons(over, RECOVERY, 2, now=0, admission=1, stop=False)
        assert key in reasons
    near = dict(fine, policy_controls=ceilings["policy_controls"] - 1999)
    assert run.refusal_reasons(near, RECOVERY, 1, now=0, admission=1,
                               stop=False) == ["policy_controls"]


def test_gates_need_engineering_readmission_and_a_complete_stage_c(tmp_path):
    run = module("run_regrasp15r1")
    engineering = [start(j["job"]) for j in RECOVERY["engineering_jobs"]] + [
        e for j in RECOVERY["engineering_jobs"] for e in (exit_(j["job"]), validated(j["job"]))]

    def check(stage, history):
        run.gate(stage, history, RECOVERY, out=tmp_path, plan_sha="f" * 64, instance=7)

    check("stage_a", [])
    with pytest.raises(ValueError, match="Stage A must be complete"):
        check("stage_c", engineering)
    (tmp_path / "stage_a-complete.json").write_text(json.dumps(
        run.completion(engineering, RECOVERY["engineering_jobs"], True)))
    (tmp_path / "stage-a-pass.json").write_text(json.dumps({"pass": True}))
    with pytest.raises(ValueError, match="re-admission"):
        check("stage_c", engineering)
    (tmp_path / "admission-c.json").write_text(json.dumps(
        {"admit_stage_c": True, "freeze_sha256": "e" * 64, "instance_id": 7}))
    with pytest.raises(ValueError, match="does not admit"):
        check("stage_c", engineering)
    (tmp_path / "admission-c.json").write_text(json.dumps(
        {"admit_stage_c": True, "freeze_sha256": "f" * 64, "instance_id": 7}))
    check("stage_c", engineering)
    with pytest.raises(ValueError, match="complete stage C"):
        check("stage_d", engineering)
    c_jobs = RECOVERY["stage_c_jobs"]
    partial = engineering + [e for j in c_jobs[:-1] for e in (start(j["job"]), exit_(j["job"]),
                                                              validated(j["job"]))]
    # A stored record claiming completion does not override the ledger.
    (tmp_path / "stage_c-complete.json").write_text(json.dumps(
        {"complete": True, "finished": True, "missing_jobs": [], "unresolved_starts": []}))
    with pytest.raises(ValueError, match="complete stage C"):
        check("stage_d", partial)
    full = partial + [start(c_jobs[-1]["job"]), exit_(c_jobs[-1]["job"]),
                      validated(c_jobs[-1]["job"])]
    (tmp_path / "stage_c-complete.json").write_text(json.dumps(
        run.completion(full, c_jobs, True)))
    check("stage_d", full)


def test_admission_simulation_honours_the_margin(tmp_path):
    admit = module("admit_regrasp15r1")
    infra = __import__("helpers.regrasp15_infra", fromlist=["FinancialBook"])
    book = tmp_path / "finance.json"
    book.write_text(json.dumps({
        "schema": "dsrl06-finance-v1", "created_utc": "2026-10-26T00:00:00+00:00",
        "limits": infra.FinancialBook.limits(), "other_actual_usd": 0.0,
        "allocations": [{"instance_id": 1, "created_utc": "2026-10-26T00:00:00+00:00",
                         "rate_usd_hour": 0.5, "offer": {},
                         "active_stop_verified_utc": "2026-10-26T10:00:00+00:00",
                         "provider_status": "destroyed", "transfer_estimate_usd": 0.1}]}))
    model = {"startup_s": 0.0, "per_step_s": 1.0, "per_request_s": 0.0}
    loads = {arm: {"steps": 3598.5, "requests": 0.0}
             for arm in ("native", "select-0.06", "project-0.10")}
    toy = {"stage_c_jobs": [{"stage": "stage_c", "arm": "native"}] * 6,
           "stage_d_jobs": [{"stage": "stage_d", "arm": "project-0.10"}] * 6}
    # 10 h remain (money would allow 11.6 h at this rate); one-hour episodes may start
    # until 10 h - 50 min after creation, so starts at 0 to 9 h are admitted.
    result = admit.admit(rate=0.25, setup_s=0.0, engineering_s=0.0, speed=1.0, replacements=0,
                         book_path=book, recovery=toy, model=model, loads=loads)
    assert result["nominal_allocation_hours"] == pytest.approx(10.0)
    assert result["admitted"] == {"stage_c": 6, "stage_d": 4}
    assert result["stage_c_complete"] and not result["stage_d_complete"]
    pricier = admit.admit(rate=0.3, setup_s=0.0, engineering_s=0.0, speed=1.0, replacements=0,
                          book_path=book, recovery=toy, model=model, loads=loads)
    assert pricier["nominal_allocation_hours"] == pytest.approx(2.9 / 0.3)


def test_recovery_book_imports_both_original_allocations():
    book = ROOT / "outputs/regrasp15r1/control/finance.json"
    if not book.exists():
        pytest.skip("The recovery book is local evidence")
    data = json.loads(book.read_text())
    original = {a["label"]: a for a in data["allocations"][:2]}
    absent = original["jar-regrasp15-1025214924"]
    assert absent["transfer_actual_usd"] == 0.0
    assert absent["active_stop_verified_utc"] == absent["created_utc"]
    first = original["jar-regrasp15-1025215704"]
    assert first["instance_id"] == 57727030
    assert first["active_stop_verified_utc"] == "2026-10-26T05:45:00+00:00"
    assert first["transfer_actual_usd"] == 0.155
    assert data["limits"]["total_usd"] == 10.0 and data["limits"]["aggregate_hours"] == 20.0
