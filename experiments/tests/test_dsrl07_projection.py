"""A fit estimate must also leave enough time to admit the final episode block."""

import numpy as np
import pytest

from helpers.dsrl07_infra import stage_b_projection


@pytest.mark.parametrize("episode_seconds", [180, 900, 1500])
def test_full_schedule_reserves_final_admission_and_preservation(episode_seconds):
    result = stage_b_projection(episode_seconds, 0.05, 0.002)
    work_before_last_start = 19 * episode_seconds + 90000 * 0.05 + 3000 * 0.002
    work_before_preservation = work_before_last_start + episode_seconds
    assert result["collection_rounds"] == 20
    assert result["total_seconds"] - work_before_last_start >= 2700
    assert result["total_seconds"] - work_before_preservation >= 1500


@pytest.mark.parametrize("values", [(0, 0.05, 0.002), (180, np.nan, 0.002)])
def test_invalid_measurements_cannot_produce_a_fit(values):
    with pytest.raises(ValueError, match="finite positive"):
        stage_b_projection(*values)
