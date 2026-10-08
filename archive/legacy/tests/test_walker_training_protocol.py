"""Regression tests for episode leakage and training-only action normalisation."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from helpers.walkerProtocol import episode_partition, training_action_scaler, exact_replacement, indices_for_episode_roles  # noqa: E402


def test_overlapping_clips_never_cross_validation_boundary():
    clips = [(ep, start) for ep in range(10) for start in range(100)]
    train, val, roles = episode_partition(clips, seed=3, val_fraction=0.2)
    assert len(train) == 800 and len(val) == 200
    assert set(roles["training"]).isdisjoint(roles["validation"])
    assert {clips[i][0] for i in train}.isdisjoint({clips[i][0] for i in val})
    assert episode_partition(clips, seed=3, val_fraction=0.2) == (train, val, roles)


def test_one_long_trajectory_cannot_manufacture_validation():
    with pytest.raises(ValueError, match="at least two episodes"):
        episode_partition([(0, start) for start in range(10000)], seed=1)


def test_scaler_ignores_extreme_validation_actions_and_nonfinite_rows():
    actions = np.array([[1., 4.], [3., 8.], [np.nan, 7.], [1e6, -1e6]])
    ds = SimpleNamespace(offsets=[0, 3], lengths=[3, 1], get_col_data=lambda _: actions)
    mean, std = training_action_scaler(ds, [0])
    np.testing.assert_allclose(mean, [2., 6.])
    np.testing.assert_allclose(std, [1., 2.])


def test_exact_replacement_keeps_validation_episodes_and_total_budget():
    lengths = np.array([1000, 730, 820, 60])
    kept = exact_replacement(lengths, [0, 2, 3], 512, seed=2)
    assert kept[1] == lengths[1]
    assert kept.sum() + 512 == lengths.sum()
    assert np.all((kept >= 0) & (kept <= lengths))
    assert sum((kept > 0) & (kept < lengths)) == 1


def test_replacement_cannot_borrow_validation_data_to_meet_budget():
    with pytest.raises(ValueError, match="available ordinary training"):
        exact_replacement([30, 10000], [0], 100, seed=2)


def test_explicit_remapped_roles_reject_overlap_and_missing_episodes():
    clips = [(0, 0), (1, 0), (2, 0)]
    with pytest.raises(ValueError, match="disjoint"):
        indices_for_episode_roles(clips, {"training": [0, 1], "validation": [1, 2]})
    with pytest.raises(ValueError, match="omit"):
        indices_for_episode_roles(clips, {"training": [0], "validation": [1]})
    assert indices_for_episode_roles(clips, {"training": [0, 2], "validation": [1]}) == ([0, 2], [1])


def test_smoke_check_requires_both_losses_and_independent_validation():
    from importlib.util import module_from_spec, spec_from_file_location

    path = Path(__file__).resolve().parents[1] / "experiments/scripts/mainplan_check_smoke.py"
    spec = spec_from_file_location("smoke_check", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {"steps": 1000, "render_validation": {"passed": True},
              "episode_splits": {"training": [0], "validation": [1]}, "steps_per_epoch": 100,
              "history": [{"pred_loss": 1 / (i + 1), "sigreg_loss": 2 / (i + 1), "it_per_s": 2} for i in range(20)]}
    assert module.assess_smoke(report)["gate"]["passes"]
    report["episode_splits"]["validation"] = [0]
    assert not module.assess_smoke(report)["gate"]["passes"]
    report["episode_splits"]["validation"] = [1]
    for row in report["history"]:
        row["sigreg_loss"] = 1
    assert not module.assess_smoke(report)["gate"]["passes"]
