"""Calibrating on disjoint episodes and aligning history actions with next-frame targets."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'experiments'))
from helpers.evoNoiseDiagnostic import calibration_split, residual_summary, trace_windows


def test_disjoint_roles_ignore_excluded_episode_order():
    excluded = np.arange(0, 256, 4)
    fit, check = calibration_split(256, excluded)
    assert len(fit) == len(check) == 64
    assert not set(fit) & set(check) and not (set(fit) | set(check)) & set(excluded)
    fit2, check2 = calibration_split(256, excluded[::-1])
    assert np.array_equal(fit, fit2) and np.array_equal(check, check2)
    with pytest.raises(ValueError):
        calibration_split(128, excluded[excluded < 128])


def test_windows_end_at_correct_real_target_with_only_preceding_actions():
    z = np.arange(2*13*3).reshape(2, 13, 3)
    a = np.arange(2*12*60).reshape(2, 12, 60)
    zw, aw, zs, acts = trace_windows(z[:, :3], a[:, :2].reshape(2, 2, 10, 6), z[:, 3:], a[:, 2:])
    assert np.array_equal(zs, z) and np.array_equal(acts, a)
    for root in range(2):
        for step in range(10):
            row = root*10+step
            assert np.array_equal(zw[row], z[root, step:step+4])
            assert np.array_equal(aw[row], a[root, step:step+3])
            assert np.array_equal(zw[row, -1], z[root, step+3])
    shifted = z.copy()
    shifted[:, -1] += 1000
    other, _, _, _ = trace_windows(shifted[:, :3], a[:, :2].reshape(2, 2, 10, 6), shifted[:, 3:], a[:, 2:])
    assert np.array_equal(zw.reshape(2, 10, 4, 3)[:, :-1], other.reshape(2, 10, 4, 3)[:, :-1])


def test_residual_bias_and_spread_remain_distinct():
    result = residual_summary(np.ones((10, 192))*2)
    assert result['bias_energy_fraction'] == 1 and result['std_rms'] == 0
    assert result['rms'] == result['bias_rms'] == 2
    with pytest.raises(ValueError):
        residual_summary(np.ones((1, 192)))
