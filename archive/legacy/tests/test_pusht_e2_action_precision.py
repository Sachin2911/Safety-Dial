"""E2 executes exactly the recorded controls, with the existing arena guard reapplied."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from helpers import branchBank
from helpers.branchBank import Bank, BankWriter, Proposal, exits_arena
from helpers.pushtReplay import DenseLog, Root, StepLedger
from scripts.pusht_e2_repair import canonical_proposals


def test_executed_action_values_equal_persisted_float32_tape(tmp_path, monkeypatch):
    raw = np.random.default_rng(304).uniform(-.01, .01, (5, 5, 2))
    assert not np.array_equal(raw, raw.astype(np.float32))
    proposals, rejected = canonical_proposals([Proposal(raw, "random", {})], [250., 250.])
    assert rejected == 0 and proposals[0].tape.dtype == np.float32
    captured = []
    monkeypatch.setattr(branchBank, "reset_root", lambda *args, **kwargs: None)
    def run_actions(_env, actions, **_kwargs):
        captured.append(np.asarray(actions, dtype=np.float64).copy())
        observed = np.arange(26) <= 7
        return DenseLog(np.zeros((26, 7)), np.zeros((26, 2)), np.zeros(26), np.zeros(26, int),
            actions, None, observed=observed, observation_valid=observed)
    monkeypatch.setattr(branchBank, "execute_tape", run_actions)
    state = np.array([250., 250., 200., 200., 0., 0., 0.])
    root = Root(1, state, state, np.zeros((0, 2)), "adapt-0", {"episode": 1})
    branches = branchBank.execute_proposals(object(), root, proposals, record_frames=False)
    writer = BankWriter(tmp_path / "bank", with_frames=False)
    writer.add_root(root)
    writer.add_branches(branches)
    writer.finish(StepLedger())
    bank = Bank(tmp_path / "bank")
    assert np.array_equal(captured[0], bank.h5["tape"][0].reshape(-1, 2).astype(np.float64))
    assert bank.h5["observed"][0].sum() == 8  # queried censored outcome is still stored
    bank.h5.close()


def test_float32_rounding_rechecks_the_same_arena_boundary():
    tape = np.zeros((5, 5, 2), np.float64)
    tape[0, 0, 0] = .1000000001
    xy = np.array([512. - 100. * tape[0, 0, 0], 250.])
    assert not exits_arena(tape, xy)
    assert exits_arena(tape.astype(np.float32), xy)
    proposals, rejected = canonical_proposals([Proposal(tape, "random", {})], xy)
    assert proposals == [] and rejected == 1
