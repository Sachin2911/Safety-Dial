"""Canonical checkpoint/JSON comparisons must preserve values, not container syntax."""

import pytest

from analyze_maniguard_evolution import checkpoint_state_matches


def test_json_task_seed_lists_match_checkpoint_tuples():
    checkpoint = {"generation": 2, "cases": [(0, 292100), (3, 292100)],
                  "jobs": {"native": {"valid": True, "controls": 916}}}
    encoded = {"generation": 2, "cases": [[0, 292100], [3, 292100]],
               "jobs": {"native": {"valid": True, "controls": 916}}}
    assert checkpoint_state_matches(checkpoint, encoded)


def test_value_changes_are_not_hidden_by_normalization():
    assert not checkpoint_state_matches({"cases": [(0, 1)]}, {"cases": [[0, 2]]})
    assert not checkpoint_state_matches({"generation": 1}, {"generation": 2})


def test_nonfinite_state_cannot_pass_audit():
    with pytest.raises(ValueError):
        checkpoint_state_matches({"fitness": float("nan")}, {"fitness": None})
