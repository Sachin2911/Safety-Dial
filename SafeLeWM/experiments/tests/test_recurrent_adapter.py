"""Synthetic checks only, executed on the rented GPU host before simulator work."""

import json
from pathlib import Path
import pickle
import sys

import cma
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.pilot_common import features, score  # noqa: E402
from helpers.qualification import segment_distance  # noqa: E402
from helpers.recurrent_adapter import (  # noqa: E402
    AdapterSpec,
    RecurrentActionAdapter,
    standardized_parameters,
)


@pytest.mark.parametrize("action_dim", [2, 7])
def test_neutral_recovers_base_and_reset(action_dim):
    actor = RecurrentActionAdapter(AdapterSpec(396, action_dim))
    inputs = torch.randn(3, 396)
    base = torch.rand(3, action_dim) * 2 - 1
    state = actor.initial_state(3)
    executed, updated, _ = actor(inputs, state, base)
    torch.testing.assert_close(executed, base, rtol=0, atol=0)
    assert not torch.equal(state, updated)
    repeated = actor(inputs, actor.initial_state(3), base)
    torch.testing.assert_close(repeated[1], updated, rtol=0, atol=0)


def test_parameter_roundtrip_authority_and_gradients():
    actor = RecurrentActionAdapter(AdapterSpec(396, 2))
    neutral, scales = actor.parameters_vector(), actor.coordinate_scales()
    z = np.random.default_rng(7101).normal(0, 0.5, neutral.shape)
    parameters = standardized_parameters(neutral, scales, z)
    actor.load_vector(parameters)
    np.testing.assert_array_equal(parameters, actor.parameters_vector())
    inputs, base = torch.randn(4, 396), torch.zeros(4, 2)
    executed, _, _ = actor(inputs, actor.initial_state(4), base)
    assert torch.isfinite(executed).all() and executed.abs().max() <= 1
    assert executed.abs().max() > 0.01
    executed.square().sum().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in actor.parameters())
    assert actor.features.weight.grad.abs().sum() > 0
    assert actor.memory.weight_ih.grad.abs().sum() > 0


def test_reject_invalid_parameters():
    actor = RecurrentActionAdapter(AdapterSpec(396, 2))
    with pytest.raises(ValueError):
        actor.load_vector([0])
    invalid = actor.parameters_vector()
    invalid[0] = np.nan
    with pytest.raises(ValueError):
        actor.load_vector(invalid)


def test_feature_information_and_causal_history():
    current, goal = np.arange(192), -np.arange(192)
    value = features(
        current, goal, np.array([112, 56]), [140, 84], [0.1, 0.2], [-0.3, -0.4], [5, -5], 100
    )
    assert value.shape == (396,)
    np.testing.assert_allclose(
        value[384:], [0, -0.5, 0.125, 0.125, 18 / 224, 0.1, 0.2, -0.3, -0.4, 1, -1, 0.5]
    )


def test_score_exact_lexicographic_order():
    totals = []
    for safe in range(9):
        for complete in range(safe, 9):
            for violations in range(9):
                totals.append(
                    ((safe, complete, -violations), 81 * safe + 9 * complete - violations)
                )
    ordered = sorted(totals)
    assert all(a[1] < b[1] for a, b in zip(ordered[:-1], ordered[1:]))
    rows = [dict(safe_completion=False, success=False, violation=False) for _ in range(8)]
    assert score(rows)["score"] == 0
    rows[0] = dict(safe_completion=False, success=True, violation=True)
    assert score(rows)["score"] == 8


def test_diagonal_cma_updates_and_checkpoint():
    es = cma.CMAEvolutionStrategy(
        np.zeros(19),
        0.5,
        dict(
            seed=7102,
            popsize=8,
            CMA_diagonal=True,
            verbose=-9,
            verb_log=0,
        ),
    )
    initial = np.array(es.mean).copy()
    candidates = es.ask()
    es.tell(candidates, [float(np.square(x - 0.2).sum()) for x in candidates])
    assert es.countiter == 1 and es.countevals == 8
    assert not np.array_equal(initial, es.mean)
    restored = pickle.loads(pickle.dumps(es))
    np.testing.assert_array_equal(restored.mean, es.mean)
    assert restored.countiter == 1


def test_frozen_panels_have_safe_and_risky_routes():
    panel = json.loads((ROOT / "configs/adapter-pilot/panel.json").read_text())
    assert len(panel["training"]) == 8 and len(panel["evaluation"]) == 4
    for case in panel["training"] + panel["evaluation"]:
        for label in ["safe", "risky"]:
            points = [case["initial"], *case[label + "_path"]]
            assert points[-1] == case["goal"]
            minimum = min(
                segment_distance(a, b, case["hazard_centre"])
                for a, b in zip(points[:-1], points[1:])
            )
            assert (minimum > 18) == (label == "safe")
            for a, b in zip(points[:-1], points[1:]):
                for x, y in np.linspace(a, b, 101):
                    assert 21 <= x <= 203 and 21 <= y <= 203
                    assert x < 100 or x > 124 or 43.75 <= y <= 54.25
