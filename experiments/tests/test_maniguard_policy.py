"""Check random-state isolation between matched physical comparison episodes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_policy import EpisodeSeededPolicy


class FakePolicy:
    def infer(self, observation):
        assert set(observation) == {"state"}
        result = self._rng
        self._rng += 1
        return result


def test_sampling_advances_within_episode_and_resets_between_same_seed_episodes():
    wrapped = EpisodeSeededPolicy(FakePolicy(), int)
    obs = {"study_episode_id": "baseline", "episode_seed": 42, "state": [0]}
    assert wrapped.infer(obs) == 42
    assert wrapped.infer(obs) == 43
    assert obs["episode_seed"] == 42
    assert obs["study_episode_id"] == "baseline"
    obs["study_episode_id"] = "adapter"
    assert wrapped.infer(obs) == 42
    assert wrapped.infer(obs) == 43
