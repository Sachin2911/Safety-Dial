"""Checks that protect physical comparison and causal feature contracts."""

import json

import numpy as np
import pytest

from helpers.maniguard_latent import (
    CODE_DIM,
    FEATURE_DIM,
    CandidateSearch,
    LatentController,
    arm_basis,
    feature_vector,
    initial_weights,
    outcome_score,
)


def test_steering_support_and_native_identity():
    basis = arm_basis()
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-6)
    assert not basis[7:].any()
    zero, positive, negative, _ = initial_weights()
    features = np.zeros(FEATURE_DIM)
    features[0] = 1
    assert LatentController(zero).identity
    assert not LatentController(zero).offset(features, 16, 32).any()
    plus = LatentController(positive).offset(features, 16, 32)
    minus = LatentController(negative).offset(features, 16, 32)
    np.testing.assert_array_equal(plus, -minus)
    assert np.any(plus[:8, :7]) and not np.any(plus[:, 7:])
    np.testing.assert_array_equal(plus[0], plus[-1])


def test_features_use_executed_history_and_bounded_semantics():
    context = [np.arange(8) / 8] * 3
    initial = feature_vector(np.zeros(7), np.zeros(2), np.zeros(3), *context)
    assert initial.shape == (FEATURE_DIM,)
    assert initial[0] == 1 and not initial[10:20].any()
    updated = feature_vector(np.zeros(7), np.zeros(2), np.zeros(3), *context,
                             previous_executed=np.r_[np.full(7, .1), -1],
                             previous_eef=np.full(3, -.025))
    np.testing.assert_allclose(updated[10:20], .5)
    np.testing.assert_array_equal(updated[-24:], np.tile(context[0], 3))
    with pytest.raises(ValueError):
        feature_vector(np.zeros(7), np.zeros(2), np.zeros(3), np.full(8, np.nan),
                       context[1], context[2])
    with pytest.raises(ValueError):
        LatentController(np.full((CODE_DIM, FEATURE_DIM), 1.01))


@pytest.mark.parametrize("completion,violation,score", [
    (False, False, 0), (True, False, 1), (False, True, -2), (True, True, -2)])
def test_objective_preserves_unsafe_completion_penalty(completion, violation, score):
    assert outcome_score(completion, violation) == score


def test_search_resume_lineage_domain_and_equal_elite_cost():
    for method in ["ea", "random"]:
        search = CandidateSearch(method)
        for i, weights in enumerate(initial_weights()):
            search.record(f"initial-{i}", weights, [0, 1] if i == 2 else [0, 0])
        restored = CandidateSearch(method, state=json.loads(json.dumps(search.state())))
        cohort = search.propose()
        assert cohort == restored.propose()
        assert len(cohort) == 4
        assert cohort[0]["parent"] == "initial-2"
        np.testing.assert_array_equal(cohort[0]["weights"], initial_weights()[2])
        for candidate in cohort:
            assert np.max(np.abs(candidate["weights"])) <= 1
            search.record(candidate["id"], candidate["weights"], [-2, 0], candidate["parent"])
        search.propose()
        with pytest.raises(ValueError):
            search.propose()


def test_incomplete_candidate_evidence_cannot_select_finalist():
    search = CandidateSearch("ea")
    with pytest.raises(ValueError):
        search.record("a", initial_weights()[0], [1])
    with pytest.raises(ValueError):
        search.propose()
