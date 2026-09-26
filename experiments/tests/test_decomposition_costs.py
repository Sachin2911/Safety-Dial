"""E1 evaluation charges real cache fills and failed attempts, without simulation."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import decomposition, pushtReplay
from helpers.pushtReplay import StepLedger


def fixture(monkeypatch, *, fail_at=None, encode_failure=False):
    roots = [SimpleNamespace(root_id=f"r{i}", meta={"episode": i}, prefix=np.zeros((n, 2)))
             for i, n in enumerate((10, 15))]
    indices = [0, 0, 1, 0, 1]
    states = np.tile([256., 256., 256., 256., 0., 0., 0.], (26, 1))
    branches = [{"root_index": ri, "states": states.copy(), "kind": "nominal",
                 "tape": np.zeros((5, 5, 2)), "frames": states[::5, 2:5].copy(),
                 "observed": np.ones(26, bool), "observation_valid": np.ones(26, bool),
                 "terminated": np.zeros(26, bool), "truncated": np.zeros(26, bool),
                 "n_contacts": np.zeros(26, int), "pusher_block_contacts": np.zeros(26, int),
                 "contact_kind": "pusher_block"} for ri in indices]
    class Bank:
        def __len__(self):
            return len(branches)
        def branch(self, j, **kwargs):
            return branches[j]
        def root_of(self, j):
            return roots[indices[j]]
    bank = Bank()
    bank.roots = roots
    bank.h5 = {key: np.stack([b[key] for b in branches]) for key in
               ("observed", "observation_valid", "terminated", "truncated")}
    class Env:
        attempts = closed = 0
        def step(self, action):
            self.attempts += 1
            if self.attempts == fail_at:
                raise RuntimeError("failed attempted step")
        def close(self):
            self.closed += 1
    env = Env()
    replayed = []
    def reset_root(metered, root):
        replayed.append(root.root_id)
        for action in root.prefix:
            metered.step(action)
        return SimpleNamespace(frames=np.tile([256., 256., 0.], (3, 1)),
                               history_actions=np.zeros((2, 5, 2)))
    class Imaginer:
        encodes = rollouts = 0
        def encode(self, frames):
            self.encodes += 1
            if encode_failure and len(frames) == 3:
                raise RuntimeError("history encode failed")
            return frames
        def rollout(self, history, actions, tape):
            self.rollouts += 1
            return np.tile([256., 256., 0.], (1, 5, 1))
    class Probe:
        calls = 0
        def predict_pose(self, z):
            self.calls += 1
            return z
    monkeypatch.setattr(pushtReplay, "make_env", lambda: env)
    monkeypatch.setattr(pushtReplay, "reset_root", reset_root)
    monkeypatch.setattr(decomposition, "clearance_trace", lambda poses, hazard: np.ones(len(poses)))
    layouts = [SimpleNamespace(root_id=root.root_id, family=family, shape=None)
               for root in roots for family in ("familiar", "heldout")]
    return bank, layouts, Imaginer(), Probe(), env, replayed


def test_metered_cache_fills_once_per_root_without_changing_outputs_or_model_calls(monkeypatch):
    bank, layouts, imaginer, probe, env, replayed = fixture(monkeypatch)
    reference = decomposition.analyse_bank("dev", bank, layouts, imaginer, probe, verbose=False)
    reference_calls = imaginer.encodes, imaginer.rollouts, probe.calls
    assert env.closed == 1
    bank, layouts, imaginer, probe, env, replayed = fixture(monkeypatch)
    ledger = StepLedger()
    result = decomposition.analyse_bank("dev", bank, layouts, imaginer, probe,
                                        verbose=False, evaluation_ledger=ledger)
    assert result == reference
    assert (imaginer.encodes, imaginer.rollouts, probe.calls) == reference_calls == (7, 5, 10)
    assert replayed == ["r0", "r1"]
    assert len(result["rows"]) == 10  # Two layouts and five branches do not multiply replay costs.
    assert ledger.to_dict() == {"steps": {"dev_history": 25}, "branches": {}, "total_steps": 25}
    assert env.attempts == ledger.total
    assert env.closed == 1


def test_shared_ledger_keeps_bank_replays_in_separate_categories(monkeypatch):
    ledger = StepLedger()
    for name in ("dev", "test"):
        bank, layouts, imaginer, probe, env, _ = fixture(monkeypatch)
        decomposition.analyse_bank(name, bank, layouts, imaginer, probe,
                                   verbose=False, evaluation_ledger=ledger)
        assert env.closed == 1
    assert ledger.counts == {"dev_history": 25, "test_history": 25}
    assert ledger.total == 50


def test_failed_step_charges_actual_attempts_and_closes_the_cache_env(monkeypatch):
    bank, layouts, imaginer, probe, env, replayed = fixture(monkeypatch, fail_at=13)
    ledger = StepLedger()
    with pytest.raises(RuntimeError, match="failed attempted step"):
        decomposition.analyse_bank("stress", bank, layouts, imaginer, probe,
                                   verbose=False, evaluation_ledger=ledger)
    assert replayed == ["r0", "r1"]
    assert ledger.counts == {"stress_history": 13}
    assert ledger.total == env.attempts == 13
    assert imaginer.rollouts == 2
    assert env.closed == 1


def test_model_failure_after_replay_keeps_paid_steps_and_closes_env(monkeypatch):
    bank, layouts, imaginer, probe, env, replayed = fixture(monkeypatch, encode_failure=True)
    ledger = StepLedger()
    with pytest.raises(RuntimeError, match="history encode failed"):
        decomposition.analyse_bank("dev", bank, layouts, imaginer, probe,
                                   verbose=False, evaluation_ledger=ledger)
    assert replayed == ["r0"]
    assert ledger.total == env.attempts == 10
    assert imaginer.rollouts == 0
    assert env.closed == 1
