"""Checks for task features and neutral authority. Run on rented hosts only."""

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pusht_authority import features, distinct_start  # noqa: E402
from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter  # noqa: E402


def test_features_bind_native_units_and_history():
    f = features(
        np.arange(192),
        np.arange(192)[::-1],
        [256, 256, 300, 320, np.pi / 2, 20, -40],
        [356, 156],
        [0.1, 0.2],
        [0.3, 0.4],
        [50, -50],
        140,
    )
    assert f.shape == (402,)
    np.testing.assert_allclose(f[384:388], [0, 0, 44 / 512, 64 / 512])
    np.testing.assert_allclose(
        f[388:395], [1, 0, 0.2, -0.4, 100 / 512, -100 / 512, 33 / 512], atol=1e-6
    )
    np.testing.assert_allclose(f[395:], [0.1, 0.2, 0.3, 0.4, 0.5, -0.5, 0.5])


def test_neutral_parity_and_head_only_authority():
    a = RecurrentActionAdapter(AdapterSpec(402, 2, initialization_seed=13101))
    neutral = a.parameters_vector()
    assert len(neutral) == 19364
    base = torch.tensor([[0.4, -0.2]])
    out, _, _ = a(torch.zeros(1, 402), a.initial_state(), base)
    torch.testing.assert_close(out, base, rtol=0, atol=0)
    changed = neutral.copy()
    changed[-2:] = [0.6, -0.6]
    a.load_vector(changed)
    out, _, detail = a(torch.zeros(1, 402), a.initial_state(), base)
    assert not torch.equal(out, base)
    assert torch.all(out.abs() <= 1)
    torch.testing.assert_close(out, detail["proposed"].clamp(-1, 1), rtol=0, atol=0)
    np.testing.assert_array_equal(changed[:-132], neutral[:-132])


def test_scene_distance_wraps_angles_and_rejects_near_duplicates():
    assert not distinct_start([0, 0, 0, 0, 0], [1, 1, 1, 1, 2 * np.pi - 0.01])
    assert distinct_start([0, 0, 0, 0, 0], [40, 0, 0, 0, 0])
    assert distinct_start([0, 0, 0, 0, 0], [0, 0, 0, 0, np.pi / 8])
