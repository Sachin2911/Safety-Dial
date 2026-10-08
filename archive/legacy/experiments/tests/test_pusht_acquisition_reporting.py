"""Oracle pairing must bind actual action bytes before any branch is queried."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pushtAcquisitionReporting import (
    candidate_pool_identity,
    result_provenance,
    validate_reference,
    validate_reference_pool,
)


def candidates():
    pool = SimpleNamespace(candidates=[SimpleNamespace(
        root_index=root, key=(root, position), features={}, executed=False,
        proposal=SimpleNamespace(kind="nominal" if position == 0 else "random",
                                 tape=np.full((2, 5, 2), position / 10, dtype=np.float32)))
        for root in range(2) for position in range(4)])
    return pool, [0, 1, 0, 1, 0, 1], [pool.candidates[0], pool.candidates[4]]


def identity():
    return candidate_pool_identity(*candidates())


def reference():
    bank = {name: "a" * 64 for name in ("roots.json", "branches.h5", "layouts.json")}
    ident = identity()
    return {"run_id": "prospective", "status": "complete",
            "gate": {"diagnostic": False, "e2_passed": True},
            "e2_report_sha256": "b" * 64, "recipe": {"steps": 10, "lr": 5e-5},
            "input_sha256": {"model.pt": "c" * 64, "proposal.py": "d" * 64},
            "evaluation_bank_identity": {name: deepcopy(bank) for name in ("dev", "test", "stress")},
            "acquisition_root_identity": {**bank, "contexts.npz": "e" * 64},
            "acquisition_seeds": [0, 1, 2], "seed_branches": 2, "rounds": [2, 2], "budgets_added": [2, 4],
            "candidate_pool_identities": {str(seed): deepcopy(ident) for seed in (0, 1, 2)},
            "upstream_revisions": {"assets": {"repo_id": "private/models", "revision": "f" * 40,
                                                  "path": "assets/run", "receipt_sha256": "a" * 64}},
            "arms": {arm: {str(seed): {"root_schedule": ident["root_schedule"].copy(),
                "curve": [{"budget_added": budget, "charged_steps": budget * 25 + 100,
                           "ledger": {"total_steps": budget * 25 + 100},
                           "hf_repo": "private/models", "hf_revision": "f" * 40,
                           "weights_sha256": "c" * 64,
                           "training_time_cumulative_s": 1.5 * budget,
                           "scoring_time_cumulative_s": 0.3 * budget}
                          for budget in (2, 4)]} for seed in (0, 1, 2)} for arm in ("random", "boundary")}}


def current(ref):
    report = deepcopy(ref)
    report.update(run_id="oracle", status="running", arms={}, candidate_pool_identities={})
    return report


def test_identity_is_json_stable_and_excludes_postquery_features():
    pool, schedule, seed = candidates()
    before = candidate_pool_identity(pool, schedule, seed)
    pool.candidates[0].features["c_hat"] = -100
    pool.candidates[0].features["true_error"] = 100
    assert before == candidate_pool_identity(pool, schedule, seed)
    assert json.loads(json.dumps(before)) == before
    assert before["root_schedule"] == schedule
    assert before["common_seed_keys"] == [[0, 0], [1, 0]]
    assert before["candidate_count"] == 8


@pytest.mark.parametrize("change", ["byte", "dtype", "shape", "endian", "kind", "key", "root", "order", "schedule", "seed"])
def test_every_physical_or_order_identity_change_is_detected(change):
    pool, schedule, seed = candidates()
    before = candidate_pool_identity(pool, schedule, seed)
    c = pool.candidates[-1]
    if change == "byte":
        c.proposal.tape[0, 0, 0] = np.nextafter(c.proposal.tape[0, 0, 0], np.float32(1))
    elif change == "dtype":
        c.proposal.tape = c.proposal.tape.astype(np.float64)
    elif change == "shape":
        c.proposal.tape = c.proposal.tape.reshape(4, 5)
    elif change == "endian":
        c.proposal.tape = c.proposal.tape.astype(">f4")
    elif change == "kind":
        c.proposal.kind = "stress"
    elif change == "key":
        c.key = (1, 8)
    elif change == "root":
        c.root_index, c.key = 2, (2, 0)
    elif change == "order":
        pool.candidates[-1], pool.candidates[-2] = pool.candidates[-2], pool.candidates[-1]
    elif change == "schedule":
        schedule[-2:] = [1, 0]
    elif change == "seed":
        seed[0] = pool.candidates[1]
    after = candidate_pool_identity(pool, schedule, seed)
    assert before["sha256"] != after["sha256"]


def test_memory_strides_do_not_change_the_executed_action_identity():
    pool, schedule, seed = candidates()
    before = candidate_pool_identity(pool, schedule, seed)
    for candidate in pool.candidates:
        candidate.proposal.tape = np.asfortranarray(candidate.proposal.tape)
    assert candidate_pool_identity(pool, schedule, seed) == before


@pytest.mark.parametrize("change", ["queried", "duplicate", "exhausted", "nonfinite", "wrong_seed", "seed_tape"])
def test_invalid_prequery_pool_fails_closed(change):
    pool, schedule, seed = candidates()
    if change == "queried":
        pool.candidates[0].executed = True
    elif change == "duplicate":
        pool.candidates[1].key = pool.candidates[0].key
    elif change == "exhausted":
        schedule.extend([0, 0])
    elif change == "nonfinite":
        pool.candidates[0].proposal.tape[0, 0, 0] = np.nan
    elif change == "wrong_seed":
        seed.reverse()
    elif change == "seed_tape":
        seed[0] = deepcopy(seed[0])
        seed[0].proposal.tape[0, 0, 0] = 0.2
    with pytest.raises(ValueError):
        candidate_pool_identity(pool, schedule, seed)


def test_reference_validates_before_per_seed_pools_are_constructed():
    ref = reference()
    cur = current(ref)
    validate_reference(ref, cur)
    for seed in (0, 1, 2):
        validate_reference_pool(ref, seed, identity())
    cur["candidate_pool_identities"] = {"0": identity()}
    validate_reference(ref, cur)


@pytest.mark.parametrize("field", ["e2_report_sha256", "recipe", "input_sha256", "evaluation_bank_identity",
                                   "acquisition_root_identity", "acquisition_seeds", "seed_branches", "rounds"])
def test_reference_rejects_changed_study_inputs(field):
    ref = reference()
    cur = current(ref)
    if field == "e2_report_sha256":
        cur[field] = "1" * 64
    elif field == "recipe":
        cur[field]["lr"] *= 2
    elif field == "input_sha256":
        cur[field]["proposal.py"] = "1" * 64
    elif field == "evaluation_bank_identity":
        cur[field]["test"]["branches.h5"] = "1" * 64
    elif field == "acquisition_root_identity":
        cur[field]["contexts.npz"] = "1" * 64
    elif field == "acquisition_seeds":
        cur[field] = [0]
    elif field == "seed_branches":
        cur[field] += 1
    elif field == "rounds":
        cur[field] = [1, 3]
        cur["budgets_added"] = [1, 4]
    with pytest.raises(ValueError, match="differs"):
        validate_reference(ref, cur)


@pytest.mark.parametrize("change", ["incomplete", "diagnostic", "e2", "missing_arm", "missing_seed", "missing_budget",
                                   "pool_missing", "different_schedule", "missing_hashes", "missing_context"])
def test_reference_requires_complete_paired_qualified_evidence(change):
    ref = reference()
    cur = current(ref)
    if change == "incomplete":
        ref["status"] = "running"
    elif change == "diagnostic":
        ref["gate"]["diagnostic"] = True
    elif change == "e2":
        ref["gate"]["e2_passed"] = False
    elif change == "missing_arm":
        del ref["arms"]["boundary"]
    elif change == "missing_seed":
        del ref["arms"]["random"]["2"]
    elif change == "missing_budget":
        ref["arms"]["random"]["1"]["curve"].pop()
    elif change == "pool_missing":
        del ref["candidate_pool_identities"]["2"]
    elif change == "different_schedule":
        ref["arms"]["boundary"]["0"]["root_schedule"][-2:] = [1, 0]
    elif change == "missing_hashes":
        ref["input_sha256"] = {}
    elif change == "missing_context":
        del ref["acquisition_root_identity"]["contexts.npz"]
    with pytest.raises(ValueError):
        validate_reference(ref, cur)


def test_per_seed_check_rejects_changed_actions_and_common_seed_keys_before_query():
    ref = reference()
    pool, schedule, seeds = candidates()
    pool.candidates[-1].proposal.tape[0, 0, 0] = 0.8
    with pytest.raises(ValueError, match="differs from reference seed"):
        validate_reference_pool(ref, 0, candidate_pool_identity(pool, schedule, seeds))
    pool, schedule, seeds = candidates()
    seeds[0] = pool.candidates[1]
    with pytest.raises(ValueError, match="differs from reference seed"):
        validate_reference_pool(ref, 0, candidate_pool_identity(pool, schedule, seeds))
    with pytest.raises(ValueError, match="differs from reference seed"):
        validate_reference_pool(ref, 4, identity())


def test_identity_cannot_be_edited_without_changing_its_digest():
    ref = reference()
    tampered = identity()
    tampered["root_schedule"][-2:] = [1, 0]
    with pytest.raises(ValueError, match="digest"):
        validate_reference_pool(ref, 0, tampered)


def test_provenance_contains_receipts_and_every_checkpoint_cost_without_io(monkeypatch):
    ref = reference()
    ref["arms"]["random"]["0"]["oracle_pool_steps"] = 800
    monkeypatch.setattr(Path, "read_bytes", lambda *a, **k: pytest.fail("No filesystem reads"))
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("No filesystem reads"))
    result = result_provenance(ref)
    assert len(result["checkpoints"]) == 12
    assert result["upstream_revisions"] == ref["upstream_revisions"]
    point = result["checkpoints"][0]
    assert point["hf_revision"] == "f" * 40
    assert point["costs"]["charged_steps"] == 150
    assert point["costs"]["training_time_cumulative_s"] == 3
    assert point["oracle_pool_steps"] == 800
    result["input_sha256"]["model.pt"] = "changed"
    assert ref["input_sha256"]["model.pt"] == "c" * 64
    json.dumps(result, allow_nan=False)
