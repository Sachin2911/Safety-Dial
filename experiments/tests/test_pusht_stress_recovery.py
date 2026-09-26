from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import pushtBankSampling as original
from helpers import pushtStressRecovery as recovery

REGULAR = np.array([256., 256., 192., 192., 0., 0., 0.])
# Fixed saved initial geometry only; no future states, labels or model outputs.
EDGE = np.array([444.34205199859304, 332.8026023120523, 440.1402260064304,
    408.1297500317543, 1.5965972525658516, 34.072732651764056, -19.986448375474318])
EDGE_HAZARD = np.array([399.82943080863805, 512.7261010612381])


def old_proposals(seed, state, centre):
    root = SimpleNamespace(meta={"nominal_plan": np.zeros((5, 5, 2)).tolist()})
    return original.exact_proposals(np.random.default_rng(seed), root,
        SimpleNamespace(state=state), 16, stress=True, hazard_centre=centre)


@pytest.mark.parametrize("seed", [0, 7, 22, 20260926])
def test_original_twenty_trial_draws_actions_and_audits_are_exact(seed):
    expected, audit = old_proposals(seed, REGULAR, [350., 256.])
    if not audit["complete"]:
        with pytest.raises(recovery.StressProposalShortfall) as exc:
            recovery.exact_stress_proposals(np.random.default_rng(seed), REGULAR, [350., 256.], max_trials=20)
        assert exc.value.audit == audit
        return
    actual, found = recovery.exact_stress_proposals(np.random.default_rng(seed), REGULAR, [350., 256.], max_trials=20)
    assert found == audit
    for a, e in zip(actual, expected):
        assert a.kind == e.kind and a.params == e.params
        assert a.tape.dtype == np.float32 and a.tape.tobytes() == e.tape.tobytes()


def test_saved_edge_geometry_extends_failed_search_without_changing_prefix():
    seed = 20260926
    _, old_audit = old_proposals(seed, EDGE, EDGE_HAZARD)
    assert not old_audit["complete"]
    with pytest.raises(recovery.StressProposalShortfall) as exc:
        recovery.exact_stress_proposals(np.random.default_rng(seed), EDGE, EDGE_HAZARD, max_trials=20)
    assert exc.value.audit == old_audit
    proposals, extended = recovery.exact_stress_proposals(np.random.default_rng(seed), EDGE, EDGE_HAZARD)
    assert extended["trials"][:len(old_audit["trials"])] == old_audit["trials"]
    assert extended["complete"] and len(proposals) == 16
    assert [p.kind for p in proposals] == ["stress"] * 8 + ["toward_hazard"] * 8
    assert len({p.tape.tobytes() for p in proposals}) == 16
    assert all(not recovery.exits_arena(p.tape, EDGE[:2]) for p in proposals)


def test_arena_guard_sees_float32_and_shortfall_is_closed(monkeypatch):
    seen = []
    def reject(tape, state):
        seen.append(tape.dtype)
        return True
    monkeypatch.setattr(recovery, "exits_arena", reject)
    with pytest.raises(recovery.StressProposalShortfall) as exc:
        recovery.exact_stress_proposals(np.random.default_rng(1), REGULAR, [350., 256.], max_trials=3)
    assert seen == [np.dtype("float32")] * 3
    assert exc.value.audit["filled"] == 0
    assert [r["outcome"] for r in exc.value.audit["trials"]] == ["arena_rejected"] * 3


def test_duplicate_tapes_never_fill_unique_slots(monkeypatch):
    monkeypatch.setattr(recovery, "_draw_tape", lambda *args: (np.zeros((5, 5, 2)), {}))
    with pytest.raises(recovery.StressProposalShortfall) as exc:
        recovery.exact_stress_proposals(np.random.default_rng(1), REGULAR, [350., 256.], max_trials=3)
    assert exc.value.audit["filled"] == 1
    assert [r["outcome"] for r in exc.value.audit["trials"]] == ["accepted"] + ["duplicate_rejected"] * 3


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_invalid_trial_limit_rejected(bad):
    with pytest.raises(ValueError, match="positive integer"):
        recovery.exact_stress_proposals(np.random.default_rng(1), REGULAR, [350., 256.], max_trials=bad)


def test_bulk_plan_hashes_exact_order_inputs_and_tapes_without_io():
    roots = [{"root_id": "alpha"}, SimpleNamespace(root_id="beta")]
    states = np.stack([REGULAR, EDGE])
    centres = np.array([[350., 256.], EDGE_HAZARD])
    pools, plan = recovery.preflight_stress_roots(roots, states, centres, expected_roots=2)
    assert plan["complete"] and plan["root_count"] == 2 and plan["branch_count"] == 32
    assert plan["protocol"]["seed"] == 20280927
    assert plan["protocol"]["maximum_trials_per_slot"] == 4096
    assert json.loads(json.dumps(plan)) == plan
    h = hashlib.sha256()
    rng = np.random.default_rng(20280927)
    for i, (row, proposals) in enumerate(zip(plan["roots"], pools)):
        replay, audit = recovery.exact_stress_proposals(rng, states[i], centres[i])
        assert row["audit"] == audit
        assert row["state_sha256"] == hashlib.sha256(states[i].astype(np.float64).tobytes()).hexdigest()
        assert row["hazard_centre_sha256"] == hashlib.sha256(centres[i].astype(np.float64).tobytes()).hexdigest()
        for j, (proposal, expected) in enumerate(zip(proposals, replay)):
            assert proposal.tape.tobytes() == expected.tape.tobytes()
            assert row["accepted_tape_sha256"][j] == hashlib.sha256(proposal.tape.tobytes()).hexdigest()
            h.update(row["root_id"].encode() + b"\0" + proposal.kind.encode() + b"\0" + proposal.tape.tobytes())
    assert h.hexdigest() == plan["pool_sha256"]
    plan["protocol"]["seed"] = -1
    assert recovery.STRESS_RECOVERY_PROTOCOL["seed"] == 20280927


def test_bulk_preflight_fails_at_root_without_returning_partial_pool(monkeypatch):
    def fail(*args, **kwargs):
        raise recovery.StressProposalShortfall({"complete": False, "requested": 16, "filled": 3, "trials": []})
    monkeypatch.setattr(recovery, "exact_stress_proposals", fail)
    with pytest.raises(recovery.StressProposalShortfall) as exc:
        recovery.preflight_stress_roots([{"root_id": "failure"}], [REGULAR], [[350., 256.]], expected_roots=1)
    assert exc.value.root_id == "failure" and exc.value.audit["filled"] == 3


@pytest.mark.parametrize("case", ["wrong_count", "duplicate_ids", "nonfinite_state", "wrong_hazard_shape"])
def test_bulk_rejects_malformed_fixed_inputs_before_drawing(case, monkeypatch):
    roots = [{"root_id": "alpha"}, {"root_id": "beta"}]
    states = np.stack([REGULAR, EDGE])
    centres = np.array([[350., 256.], EDGE_HAZARD])
    expected = 2
    if case == "wrong_count":
        expected = 128
    if case == "duplicate_ids":
        roots[1] = roots[0]
    if case == "nonfinite_state":
        states[1, 2] = np.nan
    if case == "wrong_hazard_shape":
        centres = centres[:, :1]
    def forbid(*args, **kwargs): raise AssertionError("must validate before drawing")
    monkeypatch.setattr(recovery, "exact_stress_proposals", forbid)
    with pytest.raises(ValueError):
        recovery.preflight_stress_roots(roots, states, centres, expected_roots=expected)
