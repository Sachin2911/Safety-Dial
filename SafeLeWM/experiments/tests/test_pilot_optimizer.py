"""Regression for the actual 19,172-dimensional, four-update pilot configuration."""

from pathlib import Path
import pickle
import sys
import warnings

import numpy as np
from cma.sigma_adaptation import CMAAdaptSigmaCSA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pilot_optimizer import make_optimizer, tell_ordered  # noqa: E402
from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter  # noqa: E402


def test_full_size_four_updates_preserve_configuration_and_ties():
    dimension = len(RecurrentActionAdapter(AdapterSpec(396, 2)).parameters_vector())
    assert dimension == 19172
    optimizer = make_optimizer(dimension, sigma=0.5, seed=8303)
    assert isinstance(optimizer.adapt_sigma, CMAAdaptSigmaCSA)
    assert optimizer.opts["CMA_diagonal"] is True
    assert optimizer.opts["CMA_mirrors"] == 0
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        for generation in range(4):
            candidates = optimizer.ask()
            saved = np.array(candidates).copy()
            scores = (
                [0] * 8 if generation == 1 else [-float(np.square(x).sum()) for x in candidates]
            )
            tell_ordered(optimizer, candidates, scores)
            np.testing.assert_array_equal(candidates, saved)
            assert optimizer.countiter == generation + 1
            assert optimizer.countevals == 8 * (generation + 1)
            assert np.isfinite(optimizer.mean).all() and 0 < optimizer.sigma < 100
        assert not recorded, [str(w.message) for w in recorded]
    restored = pickle.loads(pickle.dumps(optimizer))
    np.testing.assert_array_equal(restored.mean, optimizer.mean)
    assert restored.countiter == 4
    assert isinstance(restored.adapt_sigma, CMAAdaptSigmaCSA)
