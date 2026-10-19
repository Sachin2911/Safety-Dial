"""CPU mathematical/interface checks; no simulated outcomes or efficacy evidence."""

import json

import numpy as np
import pytest

from helpers.maniguard_latent import LatentController, initial_weights
from helpers.maniguard_latent_learning import ParameterReinforce, gaussian_score


def test_score_matches_independent_log_density_finite_difference():
    rng = np.random.default_rng(512)
    raw, center = rng.normal(size=(2, 4, 44))
    sigma = .37
    analytic = gaussian_score(raw, center, sigma)

    def log_density(location):
        return -np.sum((raw - location)**2) / (2 * sigma**2)

    for coordinate in [(0, 0), (1, 12), (2, 24), (3, 43)]:
        plus, minus = center.copy(), center.copy()
        plus[coordinate] += 1e-5
        minus[coordinate] -= 1e-5
        difference = (log_density(plus) - log_density(minus)) / 2e-5
        assert analytic[coordinate] == pytest.approx(difference, rel=1e-7, abs=1e-7)


def test_partial_batch_recovery_preserves_draws_scores_and_next_generation():
    learner = ParameterReinforce()
    samples = learner.propose()
    values = [[-2, 0], [1, 1], [0, 0], [-2, -2]]
    for sample, value in zip(samples[:2], values[:2], strict=True):
        learner.record(sample["id"], value)
    restored = ParameterReinforce(state=json.loads(json.dumps(learner.state())))
    assert restored.propose() == samples
    assert restored.state() == learner.state()
    with pytest.raises(ValueError, match="every case"):
        restored.update()
    for sample, value in zip(samples[2:], values[2:], strict=True):
        learner.record(sample["id"], value)
        restored.record(sample["id"], value)
    learner.update()
    restored.update()
    assert restored.state() == learner.state()
    assert restored.propose() == learner.propose()
    for sample in restored.pending:
        assert np.max(np.abs(LatentController(sample["weights"]).weights)) <= 1


def test_constant_returns_do_not_move_mean_or_expand_declared_case_budget():
    learner = ParameterReinforce(batches=1)
    samples = learner.propose()
    for sample in samples:
        learner.record(sample["id"], [0, 0])
        learner.record(sample["id"], [0, 0])  # Receipt replay is idempotent.
    with pytest.raises(ValueError, match="cannot change"):
        learner.record(samples[0]["id"], [1, 0])
    assert learner.update().identity
    assert learner.history[0]["valid_case_scores"] == 8
    assert learner.history[0]["gradient_norm"] == 0
    with pytest.raises(ValueError, match="allowance exhausted"):
        learner.propose()
    corrupted = learner.state()
    corrupted["pending"] = samples
    with pytest.raises(ValueError, match="Pending learning batch"):
        ParameterReinforce(state=corrupted)


def test_shared_initialization_and_corrupt_pending_artifacts():
    shared = [{"id": f"initial-{i}", "weights": weights.tolist(),
               "scores": [1, 0] if i == 2 else [0, 0]}
              for i, weights in enumerate(initial_weights())]
    learner = ParameterReinforce(shared_initialization=shared)
    np.testing.assert_allclose(learner.mean_controller().weights, initial_weights()[2])
    samples = learner.propose()
    state = learner.state()
    state["pending"][0]["raw"][0][0] += 1
    with pytest.raises(ValueError, match="recorded draw"):
        ParameterReinforce(state=state)
    with pytest.raises(ValueError, match="two valid"):
        learner.record(samples[0]["id"], [True, 0])
