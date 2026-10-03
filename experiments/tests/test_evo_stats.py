"""Evolution-study statistics: Wilson, IQM, Holm and source-episode cluster bootstraps."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.evoStats import (  # noqa: E402
    bootstrap_ci,
    clustered_mean_ci,
    clustered_spearman,
    holm,
    iqm,
    mann_whitney,
    paired_gap_difference,
    spearman,
    wilson,
)


def test_wilson_matches_known_values_and_bounds_zero_events():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = wilson(0, 256)
    assert lo == 0.0 and 0.0 < hi < 3 / 256 * 1.5
    assert wilson(256, 256)[1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        wilson(3, 2)


def test_iqm_drops_the_outer_quartiles():
    assert iqm([0, 1, 2, 3, 4, 5, 6, 100]) == pytest.approx(np.mean([2, 3, 4, 5]))
    ci = bootstrap_ci(np.arange(20.0), n_boot=300)
    assert ci["lo"] <= ci["point"] <= ci["hi"]


def test_holm_step_down_and_monotone():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    adj = holm(p)
    # sorted 0.01, 0.03, 0.04, 0.5 -> 0.04, 0.09, 0.09 (monotone), 0.5
    assert adj.tolist() == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert (holm(np.array([0.9, 0.9])) <= 1).all()


def test_spearman_and_clustered_interval():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(16), 4)
    real = rng.normal(size=(30, 64)) + np.linspace(0, 3, 30)[:, None]
    same = clustered_spearman(real, real, clusters, n_boot=200)
    assert same["point"] == pytest.approx(1.0) and same["lo"] == pytest.approx(1.0)
    noise = clustered_spearman(rng.normal(size=(30, 64)), real, clusters, n_boot=200)
    assert noise["lo"] < 0.5 < same["lo"]
    assert np.isnan(spearman(np.ones(5), np.arange(5)))
    with pytest.raises(ValueError):
        clustered_spearman(real, real[:, :10], clusters)


def test_cluster_bootstrap_resamples_whole_episodes():
    clusters = np.repeat(np.arange(8), 4)
    values = np.repeat(np.arange(8.0), 4)  # constant within an episode
    ci = clustered_mean_ci(values, clusters, n_boot=500)
    assert ci["point"] == pytest.approx(3.5) and ci["lo"] < 3.5 < ci["hi"]
    flat = clustered_mean_ci(np.ones(32), clusters, n_boot=50)
    assert flat["lo"] == flat["hi"] == 1.0


def test_paired_gap_difference_sign_and_pairing():
    clusters = np.repeat(np.arange(10), 4)
    zeros = np.zeros((5, 40))
    real_hi = zeros.copy()
    real_hi[:, ::2] = 1.0  # half of the selected-at-N=4096 policies fail for real
    out = paired_gap_difference(real_hi, zeros, zeros, zeros, clusters, n_boot=300)
    assert out["point"] == pytest.approx(0.5) and out["lo"] > 0
    null = paired_gap_difference(zeros, zeros, zeros, zeros, clusters, n_boot=50)
    assert null["point"] == 0.0 and null["lo"] == null["hi"] == 0.0
    with pytest.raises(ValueError):
        paired_gap_difference(zeros, zeros[:, :5], zeros, zeros, clusters)


def test_mann_whitney_two_sided():
    out = mann_whitney(np.arange(10), np.arange(10) + 20)
    assert out["p"] < 0.001 and out["n_a"] == 10
