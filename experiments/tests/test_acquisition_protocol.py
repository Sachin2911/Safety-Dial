"""Regression checks for prospective acquisition accounting and evidence gates."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.acquisition import Candidate, CandidatePool, root_schedule, select
from helpers.acquisitionSafety import (
    COLLECTION_VERSION,
    MeteredEnv,
    copy_ledger,
    enforce_gate,
    require_new_paths,
    validate_acquisition_cache,
)
from helpers.branchBank import Proposal
from helpers.pushtReplay import StepLedger


def pool_fixture(n_roots=96, per_root=12):
    return CandidatePool([
        Candidate(ri, Proposal(np.zeros((5, 5, 2)), "random"), (ri, j),
                  {"c_hat": float(j - 3)})
        for ri in range(n_roots) for j in range(per_root)
    ])


def test_matched_schedule_keeps_actual_prefix_budgets_equal_across_all_rounds():
    pool = pool_fixture()
    rounds = [128, 64, 64, 128, 256]
    schedule = root_schedule(pool, sum(rounds), np.random.default_rng(41))
    costs = {}
    chosen_by_arm = {}
    for arm, seed in [("random", 2), ("boundary", 93)]:
        for candidate in pool.candidates:
            candidate.executed = False
        offset, charged, budget_costs, keys = 0, 0, [], []
        for n in rounds:
            selected = select(arm, pool, n, np.random.default_rng(seed),
                              schedule=schedule[offset:offset + n])
            assert [c.root_index for c in selected] == schedule[offset:offset + n]
            charged += sum(25 + 10 + 10 * (c.root_index % 4) for c in selected)
            keys.extend(c.key for c in selected)
            for c in selected:
                c.executed = True
            offset += n
            budget_costs.append(charged)
        assert len(set(keys)) == sum(rounds)
        costs[arm], chosen_by_arm[arm] = budget_costs, keys
    assert costs["random"] == costs["boundary"]
    assert chosen_by_arm["random"] != chosen_by_arm["boundary"]


def test_schedule_fails_when_pool_cannot_supply_budget():
    with pytest.raises(ValueError, match="cannot schedule"):
        root_schedule(pool_fixture(2, 2), 5, np.random.default_rng(1))
    with pytest.raises(ValueError, match="predeclared"):
        select("random", pool_fixture(2, 1), 2, np.random.default_rng(1), schedule=[0, 0])


def test_boundary_selection_uses_only_prequeried_predictions():
    pool = pool_fixture(2, 8)
    chosen = select("boundary", pool, 2, np.random.default_rng(1), schedule=[0, 1])
    assert all(c.features["c_hat"] == 0 for c in chosen)
    assert not any(c.executed for c in pool.candidates)


def test_proposals_use_observed_root_state_without_simulator(monkeypatch):
    import helpers.acquisition as acquisition

    observed = np.asarray([1., 2., 3., 4., 0., 0., 0.])
    root = SimpleNamespace(root_id="root", meta={"state_at_root": observed})
    bank = SimpleNamespace(roots=[root])
    seen = []

    def propose(rng, supplied_root, ctx, **kwargs):
        seen.append(ctx.state)
        return [Proposal(np.zeros((5, 5, 2)), "nominal")], 0

    monkeypatch.setattr(acquisition, "propose", propose)
    pool = CandidatePool.build(np.random.default_rng(1), bank, 16, n_stress=2, n_toward=2,
                               layouts_by_root={"root": {"familiar": SimpleNamespace(centre=(10, 10))}})
    assert len(pool.candidates) == 1
    np.testing.assert_array_equal(seen[0], observed)


def test_meter_counts_failed_steps_and_preserves_prior_costs():
    class Env:
        def step(self, action):
            if action == "fail":
                raise RuntimeError("step failed")
            return action

    ledger = StepLedger()
    metered = MeteredEnv(Env(), ledger, "branch")
    assert metered.step("ok") == "ok"
    with pytest.raises(RuntimeError):
        metered.step("fail")
    assert ledger.total == 2
    copied = StepLedger()
    copy_ledger(ledger.to_dict(), copied, "common_")
    assert copied.counts == {"common_branch": 2}


@pytest.mark.parametrize("diagnostic", [False, True])
def test_learned_selection_cannot_bypass_failed_e2(diagnostic):
    with pytest.raises(ValueError, match="Learned acquisition"):
        enforce_gate({"gate": {"passes": False}}, diagnostic=diagnostic,
                     arms=["learned"], seeds=[0])


def test_bounded_diagnostic_is_explicit_and_one_seed():
    report = {"gate": {"passes": False}}
    with pytest.raises(ValueError, match="explicit --diagnostic"):
        enforce_gate(report, diagnostic=False, arms=["random", "boundary"], seeds=[0])
    with pytest.raises(ValueError, match="one acquisition seed"):
        enforce_gate(report, diagnostic=True, arms=["random", "boundary"], seeds=[0, 1])
    assert enforce_gate(report, diagnostic=True, arms=["random", "boundary"], seeds=[0])["interpretation"] == "development_only"


def test_existing_outputs_are_preserved(tmp_path):
    sentinel = tmp_path / "acquisition.json"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        require_new_paths([tmp_path])
    assert sentinel.read_text() == "preserve"


def write_cache(path):
    path.mkdir()
    roots = [{"root_id": "r0", "prefix": np.zeros((10, 2)).tolist(), "meta": {"episode": 12}}]
    manifest = {"bank": "acq_roots", "seed": 7, "n_roots": 1, "collection_version": COLLECTION_VERSION}
    (path / "roots.json").write_text(json.dumps({"roots": roots, "manifest": manifest,
        "ledger": {"steps": {"root_generation": 45}, "total_steps": 45}}))
    (path / "layouts.json").write_text(json.dumps({"layouts": [{"root_id": "r0", "family": "familiar"}]}))
    with h5py.File(path / "branches.h5", "w") as f:
        for key in ["tape", "states", "block_vel", "block_ang_vel", "n_contacts", "root_index", "kind", "params"]:
            f.create_dataset(key, shape=(0,), dtype="f4")
    np.savez_compressed(path / "contexts.npz", frames=np.zeros((1, 3, 224, 224, 3), np.uint8),
                        history_actions=np.zeros((1, 2, 5, 2)), root_ids=np.asarray(["r0"]))


def test_cached_roots_require_complete_matching_provenance(tmp_path):
    path = tmp_path / "roots"
    write_cache(path)
    assert len(validate_acquisition_cache(path, n_roots=1, seed=7)["roots"]) == 1
    with pytest.raises(ValueError, match="manifest"):
        validate_acquisition_cache(path, n_roots=2, seed=7)
    (path / "contexts.npz").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        validate_acquisition_cache(path, n_roots=1, seed=7)


def test_prequeried_or_misaccounted_cache_is_rejected(tmp_path):
    path = tmp_path / "roots"
    write_cache(path)
    with h5py.File(path / "branches.h5", "a") as f:
        del f["tape"]
        f.create_dataset("tape", shape=(1,), dtype="f4")
    with pytest.raises(ValueError, match="prequeried"):
        validate_acquisition_cache(path, n_roots=1, seed=7)


def test_cli_gate_precedes_loading_model(monkeypatch, tmp_path):
    import scripts.pusht_e3_acquisition as runner

    e2 = tmp_path / "e2.json"
    e2.write_text(json.dumps({"gate": {"passes": False}}))
    monkeypatch.setattr(sys, "argv", ["e3", "--e2-report", str(e2)])
    monkeypatch.setattr(runner, "load_model", lambda *_: pytest.fail("GPU/model load occurred before gate"))
    with pytest.raises(ValueError, match="E2 gate did not pass"):
        runner.main()


def test_early_stop_padding_spends_real_equal_steps_without_inventing_branch_data(monkeypatch):
    import scripts.pusht_e3_acquisition as runner

    class Env:
        def __init__(self):
            self.steps = 0
            self.resets = 0

        def step(self, action):
            self.steps += 1

        def reset(self, **kwargs):
            self.resets += 1

    root = SimpleNamespace(prefix=np.zeros((10, 2)), seed=1, root_id="r0",
                           start_state=np.zeros(7), goal_state=np.zeros(7))
    bank = SimpleNamespace(roots=[root])
    candidate = Candidate(0, Proposal(np.zeros((5, 5, 2)), "random"), (0, 0))

    def reset_root(env, root, **kwargs):
        for action in root.prefix:
            env.step(action)

    monkeypatch.setattr(runner, "reset_root", reset_root)
    costs = []
    for executed in [4, 25]:
        env, ledger = Env(), StepLedger()

        def execute_tape(metered, tape, **kwargs):
            for action in tape[:executed]:
                metered.step(action)
            return SimpleNamespace(actions=tape, executed_steps=executed, censored=executed < 25)

        monkeypatch.setattr(runner, "execute_tape", execute_tape)
        branches = runner.execute_candidates(env, bank, [candidate], ledger, "branch")
        assert len(branches) == 1
        assert branches[0].log.executed_steps == executed
        assert env.steps == ledger.total == sum(ledger.counts.values()) == 35
        assert ledger.counts.get("branch_discarded_padding", 0) == 25 - executed
        assert ledger.branches["branch"] == 1
        costs.append(ledger.total)
    assert costs[0] == costs[1]


def test_acquisition_evaluation_keeps_unresolved_acceptance_undefined(monkeypatch):
    import scripts.pusht_e3_acquisition as runner

    rows = [{"branch": 0, "root": 0, "layout": "familiar", "kind": "random",
             "cmin_dense": 2.0, "cmin_imagined": 3.0, "unsafe_composite": False,
             "censored": True, "observation_valid": False}]
    bank = SimpleNamespace(h5={"states": np.zeros((1, 26, 7))},
                           roots=[SimpleNamespace(root_id="r0", meta={"source_family": "familiar"})])
    monkeypatch.setattr(runner, "Imaginer", lambda *args: None)
    monkeypatch.setattr(runner, "evaluate_model_on_bank", lambda *args, **kwargs: [dict(r) for r in rows])
    out, _ = runner.evaluate_all(None, None, None, {"dev": (bank, [])},
                                {"dev": SimpleNamespace(imaginer=None)}, "cpu", 1.0)
    metric = out["dev"]["at_matched"]
    assert np.isnan(metric["fsa"])
    assert metric["fsa_lower"] == 0 and metric["fsa_upper"] == 1
    assert metric["n_accepted_censored"] == 1
    assert out["dev"]["clearance_error"]["n"] == 0
