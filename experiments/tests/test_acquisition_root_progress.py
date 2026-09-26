"""Finite source exhaustion retains actual costs and its pre-outcome reservation."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.branchBank import Bank
from scripts import pusht_e3_acquisition as runner


def test_all_reserved_sources_are_frozen_before_queries_and_failure_is_durable(tmp_path, monkeypatch):
    destination = tmp_path / "roots"
    expert = tmp_path / "expert.h5"
    expert.write_bytes(b"mock dataset identity")
    monkeypatch.setattr(runner, "H5_PATH", expert)
    reserved = list(range(3020))
    observed = []

    class Environment:
        closed = False

        def step(self, action):
            observed.append(action)

        def close(self):
            self.closed = True

    env = Environment()
    monkeypatch.setattr(runner, "make_env", lambda: env)
    monkeypatch.setattr(runner, "NominalPlanner", lambda *args: None)

    def pairs(path, episodes, rng, count):
        assert episodes == reserved[3000:8000]
        assert count == len(episodes) == 20  # More than the old eight trials per root.
        return [{"episode": ep, "t0": 0, "goal_offset": 25,
                 "start": np.zeros(7), "goal": np.ones(7)} for ep in episodes]

    monkeypatch.setattr(runner, "expert_pairs", pairs)

    def reject(env, planner, pair, **kwargs):
        plan = json.loads((destination / "sampling-plan.json").read_text())
        assert len(plan["candidates"]) == 20
        assert plan["source_episodes"] == reserved[3000:8000]
        assert plan["max_accepted_roots_per_source"] == 1
        env.step(np.zeros(2))
        raise ValueError("Nominal root has a censored or out-of-domain prefix")

    monkeypatch.setattr(runner, "build_root", reject)
    with pytest.raises(RuntimeError, match="Only 0/1 roots"):
        runner.build_acq_roots(1, 19, {"roles": {"reserve": reserved}}, None, None,
                              "cpu", destination)
    assert env.closed and len(observed) == 20
    progress = json.loads((destination / "progress.json").read_text())
    journal = [json.loads(row) for row in (destination / "attempts.jsonl").read_text().splitlines()]
    assert progress["status"] == "incomplete"
    assert progress["attempts_completed"] == progress["scheduled_candidates"] == 20
    assert progress["ledger"]["total_steps"] == sum(row["actual_steps"] for row in journal) == 20
    assert [row["episode"] for row in journal] == reserved[3000:8000]
    assert all(not row["accepted"] for row in journal)
    with pytest.raises(ValueError, match="incomplete bank"):
        Bank(destination)


def test_oracle_does_not_rank_or_discard_unobserved_futures():
    from types import SimpleNamespace

    def branch(key, censored, trainable):
        return SimpleNamespace(params={"key": key},
                               log=SimpleNamespace(censored=censored, valid_for_training=trainable))

    complete = branch([0, 0], False, True)
    exited = branch([0, 1], False, False)
    censored = branch([1, 0], True, False)
    gate = runner.oracle_observation_gate([complete, exited, censored])
    assert not gate["gate"]["passes"]
    assert gate["n_queried_branches"] == 3
    assert gate["n_censored"] == 1 and gate["n_invalid_training"] == 2
    assert gate["censored_keys"] == [[1, 0]]
    assert runner.oracle_observation_gate([complete, exited])["gate"]["passes"]
    assert not runner.oracle_observation_gate([])["gate"]["passes"]


def test_oracle_pool_preserves_censored_and_complete_branches_before_gate(tmp_path):
    from types import SimpleNamespace
    from helpers.branchBank import Branch
    from helpers.pushtReplay import DenseLog, Root, StepLedger

    source = tmp_path / "source"
    source.mkdir()
    (source / "layouts.json").write_text(json.dumps({"layouts": []}))
    state = np.array([200., 200., 250., 250., 0., 0., 0.])
    root = Root(7, state, state, np.zeros((10, 2)), root_id="r0")
    branches = []
    for index, observed_steps in enumerate((25, 10)):
        observed = np.arange(26) <= observed_steps
        log = DenseLog(np.tile(state, (26, 1)), np.zeros((26, 2)), np.zeros(26),
                       np.zeros(26, dtype=int), np.zeros((25, 2)),
                       [np.zeros((224, 224, 3), np.uint8) for _ in range(6)],
                       observed=observed, observation_valid=np.ones(26, bool))
        branches.append(Branch("r0", np.zeros((5, 5, 2), np.float32), "random",
                               {"key": [0, index]}, log))
    ledger = StepLedger()
    ledger.add("pool", 70, branches=2)  # Prefix + actual query + discarded allowance.
    destination = tmp_path / "pool"
    gate = runner.preserve_oracle_pool(destination, SimpleNamespace(roots=[root], dir=source),
                                      branches, ledger, run_id="test-oracle", seed=0, hf=None)
    assert not gate["gate"]["passes"] and gate["n_censored"] == 1
    saved = Bank(destination)
    try:
        assert len(saved) == 2 and saved.ledger["total_steps"] == 70
        np.testing.assert_array_equal(saved.h5["observed"][1], branches[1].log.observed)
        assert json.loads((destination / "manifest.json").read_text())["data"]["all_paid_outcomes_retained"]
    finally:
        saved.h5.close()


def test_oracle_interruption_keeps_seed_trace_and_both_paid_ledgers(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import h5py
    from helpers.acquisition import Candidate
    from helpers.branchBank import Proposal
    from helpers.pushtReplay import DenseLog, Root, StepLedger

    source = tmp_path / "source"
    source.mkdir()
    (source / "layouts.json").write_text(json.dumps({"layouts": []}))
    state = np.array([200., 200., 250., 250., 0., 0., 0.])
    root = Root(7, state, state, np.zeros((10, 2)), root_id="r0")
    bank = SimpleNamespace(roots=[root], dir=source)
    common, oracle = StepLedger(), StepLedger()

    def ledger():
        total = StepLedger()
        total.add("shared_root_generation", 10)
        runner.copy_ledger(common.to_dict(), total)
        runner.copy_ledger(oracle.to_dict(), total)
        return total

    class Environment:
        calls = 0

        def step(self, action):
            self.calls += 1
            if self.calls == 47:
                raise RuntimeError("second oracle-query step failed")

    def reset(env, root, **kwargs):
        for action in root.prefix:
            env.step(action)

    def execute(env, actions, **kwargs):
        for action in actions:
            env.step(action)
        return DenseLog(np.tile(state, (26, 1)), np.zeros((26, 2)), np.zeros(26),
                        np.zeros(26, dtype=int), np.asarray(actions),
                        [np.zeros((224, 224, 3), np.uint8) for _ in range(6)],
                        observed=np.ones(26, bool), observation_valid=np.ones(26, bool))

    monkeypatch.setattr(runner, "reset_root", reset)
    monkeypatch.setattr(runner, "execute_tape", execute)
    pool = runner.OraclePoolProgress(tmp_path / "pool", bank, ledger)
    candidates = [Candidate(0, Proposal(np.zeros((5, 5, 2), np.float32), "random"), (0, i))
                  for i in range(2)]
    env = Environment()
    runner.execute_candidates(env, bank, candidates[:1], common, "seed", on_branch=pool.add)
    with pytest.raises(RuntimeError, match="step failed"):
        runner.execute_candidates(env, bank, candidates[1:], oracle, "oracle_pool", on_branch=pool.add)
    failure = pool.abort("RuntimeError")
    assert failure["stored_branches"] == 1
    assert failure["ledger"]["steps"] == {"shared_root_generation": 10, "seed": 35, "oracle_pool": 12}
    assert failure["ledger"]["total_steps"] == 57
    assert env.calls == 47
    saved = json.loads((tmp_path / "pool" / "roots.json").read_text())
    assert saved["status"] == "incomplete" and saved["ledger"] == failure["ledger"]
    with h5py.File(tmp_path / "pool" / "branches.h5") as h5:
        assert h5["tape"].shape[0] == 1
        assert json.loads(h5["params"][0])["key"] == [0, 0]
        assert h5["observed"][0].all()
    with pytest.raises(ValueError, match="incomplete bank"):
        Bank(tmp_path / "pool")
