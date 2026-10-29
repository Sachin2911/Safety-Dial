"""Check that uncertainty uses whole paired case and optimizer-run blocks."""

import numpy as np
import pytest

from helpers.lewm17_analysis import crossed_interval, matrix_for, paired_contrast


def test_pairing_cancels_shared_case_difficulty():
    baseline = np.array([[0.1, 0.2, 0.5, 0.6], [0.2, 0.3, 0.6, 0.7]])
    result = paired_contrast(baseline + 0.1, baseline)
    np.testing.assert_allclose(result["interval"], [0.1, 0.1])
    assert result["n_runs"] == 2 and result["n_cases"] == 4


def test_fixed_reference_not_five_independent_observations():
    reference = np.array([[0.0, 1.0, 0.5]])
    adaptive = np.tile(reference, (5, 1))
    result = paired_contrast(adaptive, reference)
    np.testing.assert_array_equal(result["interval"], [0.0, 0.0])
    assert crossed_interval(reference)["n_runs"] == 1


def test_incomplete_case_block_is_rejected():
    row = {
        "job": {"condition": "tuned", "case": {"id": "case"}, "seed": 179710},
        "requested_safe_completion": True,
    }
    with pytest.raises(ValueError):
        matrix_for([row], "tuned", ["case"])
