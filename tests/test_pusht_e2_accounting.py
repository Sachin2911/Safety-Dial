"""E2 history fills are charged once per cache and remain separate from collection."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from helpers import predictorAdapt, pushtReplay
from helpers.pushtReplay import StepLedger
from scripts import pusht_e2_repair as e2


class FakeEnv:
    def __init__(self):
        self.steps = 0
        self.closed = False

    def step(self, action):
        self.steps += 1

    def close(self):
        self.closed = True


class FakeImaginer:
    process = None

    def encode(self, frames):
        return torch.zeros((len(frames), 2))


class FakeBank:
    roots = [SimpleNamespace(prefix=np.zeros((3, 2))), SimpleNamespace(prefix=np.zeros((5, 2)))]

    def branch_valid_for_training(self, j):
        return j != 3

    def branch(self, j, frames=True):
        return {"root_index": [0, 0, 1][j], "frames": np.zeros((6, 2, 2, 3)),
                "tape": np.zeros((5, 5, 2))}


def test_context_and_evaluation_caches_charge_only_actual_fills(monkeypatch):
    envs = []
    def make_env():
        env = FakeEnv()
        envs.append(env)
        return env
    def reset_root(env, root):
        for action in root.prefix:
            env.step(action)
        return SimpleNamespace(frames=np.zeros((3, 2, 2, 3)), history_actions=np.zeros((2, 5, 2)))
    monkeypatch.setattr(e2, "make_env", make_env)
    monkeypatch.setattr(pushtReplay, "make_env", make_env)
    monkeypatch.setattr(pushtReplay, "reset_root", reset_root)
    monkeypatch.setattr(predictorAdapt, "blocks_to_model", lambda _process, blocks: torch.zeros((len(blocks), 10)))
    context, evaluation = StepLedger(), StepLedger()
    clips = e2.metered_branch_clips(FakeImaginer(), FakeBank(), [0, 1, 2, 3], context)
    assert len(clips) == 3  # invalid branch is not queried or trained on
    assert context.counts == {"adapt_training_history": 8}
    assert envs[0].steps == 8 and envs[0].closed
    cache = e2.metered_root_cache(FakeImaginer(), FakeBank(), evaluation, "test_history")
    first = cache.get(0)
    assert cache.get(0) is first
    cache.get(1)
    assert evaluation.counts == {"test_history": 8}
    # A distinct correction cache really repeats the replay and must be charged again.
    correction = e2.metered_root_cache(FakeImaginer(), FakeBank(), context, "readout_correction_history")
    correction.get(0)
    correction.get(1)
    correction.get(1)
    assert context.total == 16 and sum(env.steps for env in envs) == 24


def test_combined_ledger_keeps_disjoint_costs_and_does_not_mutate_inputs():
    def ledger(n):
        return {"steps": {"history": n}, "branches": {"history": 2}, "total_steps": n}
    inputs = [ledger(n) for n in (11, 13, 17, 19, 23)]
    result = e2.combined_simulator_ledger(*inputs)
    assert result["total_steps"] == 83
    assert result["steps"] == {"collection_history": 11, "context_history": 13,
        "evaluation_history": 17, "goal_no_update_history": 19, "goal_adapted_history": 23}
    assert all(item["branches"] == {"history": 2} for item in inputs)
    assert list(result["branches"].values()) == [2] * 5
