"""The native legal action box applies to replay proposals too."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pusht_authority_worker_v2 import clip_native_action  # noqa: E402


def test_clip_recorded_command_without_mutating_proposal():
    proposed = np.array([1.0535855, -0.15184173], np.float32)
    before = proposed.copy()
    executed = clip_native_action(proposed)
    np.testing.assert_array_equal(proposed, before)
    np.testing.assert_array_equal(executed, np.array([1, -0.15184173], np.float32))


def test_in_bounds_commands_are_exactly_preserved():
    x = np.array([0.4, -1], np.float32)
    np.testing.assert_array_equal(clip_native_action(x), x)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_commands_are_rejected(value):
    with pytest.raises(ValueError, match="Nonfinite"):
        clip_native_action([value, 0])
