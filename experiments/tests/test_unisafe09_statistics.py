"""Independent binomial-tail checks and paired-analysis failure cases."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe09_statistics import exact_binomial, paired_binary, paired_mean, wilson


def test_exact_limits_invert_binomial_tails():
    for n in (16, 64):
        for k in (0, 1, n // 2, n - 1, n):
            lower, upper = exact_binomial(k, n, 0.025)
            if k:
                assert binom.sf(k - 1, n, lower) == pytest.approx(0.0125, abs=1e-10)
            else:
                assert lower == 0
            if k < n:
                assert binom.cdf(k, n, upper) == pytest.approx(0.0125, abs=1e-10)
            else:
                assert upper == 1


def test_paired_binary_preserves_uncertainty_when_no_discordance():
    result = paired_binary([False] * 64, [False] * 64)
    assert result["difference"] == 0
    assert result["interval"][0] < 0 < result["interval"][1]
    stronger = paired_binary([True] * 64, [False] * 64, 0.05 / 9)
    assert stronger["difference"] == 1 and stronger["interval"][0] > 0


def test_pairing_and_simultaneous_intervals():
    a, b = [1] * 10 + [0] * 6, [0] * 6 + [1] * 10
    result = paired_binary(a, b)
    assert result["candidate_only"] == result["reference_only"] == 6
    adjusted = paired_binary(a, b, 0.05 / 9)
    assert adjusted["interval"][0] <= result["interval"][0]
    assert adjusted["interval"][1] >= result["interval"][1]
    with pytest.raises(ValueError):
        paired_binary([True], [True, False])
    assert wilson(0, 64)[1] > 0


def test_bootstrap_reports_degenerate_zero_without_jitter():
    draws = np.random.default_rng(2026102609).integers(0, 64, (10000, 64))
    result = paired_mean(np.ones(64), np.ones(64), draws)
    assert result["interval"] == [0.0, 0.0] and result["degenerate"]
    result = paired_mean(np.ones(64), np.ones(64) * 2, draws)
    assert result["interval"] == [-1.0, -1.0]
