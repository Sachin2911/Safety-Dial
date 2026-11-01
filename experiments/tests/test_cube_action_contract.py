"""Regression for Cube's explicit legal-action contract, with no simulator calls."""

import collections
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_cube_bounded as runtime
from scripts.run_cube_goal_repair import RecordedCube


def test_output_matches_safety_models_normalized_action_clipping():
    mean = np.array([0.01, -0.003, 0.002, 0.0004, 0.159])
    scale = np.array([0.289, 0.393, 0.643, 0.392, 0.250])
    scaler = runtime.BoundedScaler(mean, scale)
    candidates = np.array([[-8.0, -1.0, 0.0, 3.0, 9.0], [4.0, 2.0, -5.0, -8.0, 0.2]])
    executed = scaler.inverse_transform(candidates)
    low = (-1 - mean) / scale
    high = (1 - mean) / scale
    assert np.all(np.abs(executed) <= 1)
    np.testing.assert_allclose(
        scaler.transform(executed), np.clip(candidates, low, high), atol=1e-12
    )


def test_out_of_bounds_commands_are_rejected_before_native_step():
    env = runtime.BoundedCube.__new__(runtime.BoundedCube)
    with pytest.raises(RuntimeError, match="native action Box"):
        env.step(np.array([0.0, 0.0, 1.01, 0.0, 0.0]))


def test_applied_action_sidecar_records_the_policy_command(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(RecordedCube, "step", lambda self, a: seen.append(a.copy()))
    env = runtime.BoundedCube.__new__(runtime.BoundedCube)
    env.counters = collections.Counter()
    env.journal = SimpleNamespace(name=str(tmp_path / "interactions.jsonl"))
    scaler = runtime.BoundedScaler(np.zeros(5), np.ones(5))
    action = scaler.inverse_transform(np.array([[2.0, -0.4, 0.0, -3.0, 0.2]]))[0]
    env.step(action)
    row = json.loads((tmp_path / "applied-actions.jsonl").read_text())
    assert row["commanded_before_bounds"] == [2.0, -0.4, 0.0, -3.0, 0.2]
    assert row["executed"] == [1.0, -0.4, 0.0, -1.0, 0.2]
    np.testing.assert_array_equal(seen[0], action)
