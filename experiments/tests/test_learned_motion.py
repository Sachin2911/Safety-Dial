"""Learned prediction, causal masking, independent calibration and oracle-free checking."""

import collections

import numpy as np
import pytest
import torch

from helpers.learned_motion import (
    DenseMotionModel,
    LearnedMotionChecker,
    MotionSpec,
    calibrate_motion,
    fit_motion,
    load_motion,
    motion_errors,
    save_motion,
)
from helpers.safety_model import Constraint


def test_supervised_fit_generalizes_to_new_inputs_and_preserves_causality():
    rng = np.random.default_rng(91)
    spec = MotionSpec()
    observed = rng.normal(size=(120, 4)) * 100
    actions = rng.normal(size=(120, 15, 2))
    inputs = np.c_[observed / 512, actions.reshape(120, -1), np.ones(120)]
    weights = rng.normal(size=(35, 300)) * 0.01
    # Synthetic causal data use arbitrary coefficients, not PushT controller equations.
    for t in range(15):
        weights[4 + (t + 1) * 2 : -1, t * 20 : (t + 1) * 20] = 0
    positions = (inputs @ weights).reshape(120, 150, 2) * 512
    model, ranks = fit_motion(spec, observed[:100], actions[:100], positions[:100])
    metrics, _ = motion_errors(model, observed[100:], actions[100:], positions[100:])
    assert metrics["max_position_error_px"] < 1e-8
    assert ranks[-1] == 35
    candidates = torch.tensor(actions[100:101]).reshape(1, 1, 3, 10)
    first = model(torch.tensor(observed[100:101]), candidates)
    changed = candidates.clone()
    changed[:, :, 2] += 100
    second = model(torch.tensor(observed[100:101]), changed)
    torch.testing.assert_close(first[:, :, :100], second[:, :, :100], atol=1e-10, rtol=0)


def test_calibration_rejects_correlated_or_nonfinite_evidence():
    model = DenseMotionModel(MotionSpec())
    result = calibrate_motion(model, np.ones(100), ["same-source"] * 100)
    assert not result["ready"]
    with pytest.raises(ValueError, match="finite"):
        calibrate_motion(model, [np.nan], ["a"])
    result = calibrate_motion(model, np.arange(12) * 0.01, [str(i) for i in range(12)])
    assert result["ready"] and result["source_groups"] == 12
    assert result["allowance"] == pytest.approx(0.11)


def test_motion_checker_uses_only_observation_and_learned_weights(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Handwritten motion oracle was called")

    monkeypatch.setattr("helpers.safecem10_planner.pd_response", forbidden)
    model = DenseMotionModel(MotionSpec())
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.fill_(200 / 512)
    calibrate_motion(model, np.zeros(12), [str(i) for i in range(12)])
    counters = collections.Counter()
    checker = LearnedMotionChecker(
        model,
        [Constraint("circle", (256, 256), radius=12)],
        6,
        counters,
    )
    with pytest.raises(ValueError, match="native observed"):
        checker.observe(np.zeros(7))
    checker.observe([200, 200, 0, 0])
    violation, path, _ = checker.check(torch.zeros(1, 2, 3, 10))
    assert not violation.any()
    assert path.shape == (1, 2, 151, 2)
    assert counters["learned_candidate_checks"] == 2
    assert not hasattr(checker, "reference")
    with torch.no_grad():
        model.linear.bias.fill_(256 / 512)
    assert (checker.check(torch.zeros(1, 1, 3, 10))[0] > 0).all()


def test_uncalibrated_motion_cannot_execute_and_checkpoint_roundtrips(tmp_path):
    model = DenseMotionModel(MotionSpec())
    with pytest.raises(ValueError, match="calibration"):
        LearnedMotionChecker(model, [Constraint("circle", (0, 0), radius=1)], 6, {})
    calibrate_motion(model, np.zeros(12), [str(i) for i in range(12)])
    path = tmp_path / "motion.pt"
    save_motion(path, model, {"purpose": "test"})
    restored, _ = load_motion(path)
    torch.testing.assert_close(model.linear.weight, restored.linear.weight)
    assert restored.calibration["ready"]
    with pytest.raises(FileExistsError):
        save_motion(path, model, {})
