from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.e0Validation import compare_replays, live_geometry_check, timing_row
from helpers.pushtGeometry import Box
from helpers.pushtReplay import DenseLog, make_env


def log_fixture():
    state = np.tile(np.array([100, 100, 250, 250, 0, 0, 0], float), (26, 1))
    flags = np.zeros(26, bool)
    return DenseLog(state, np.zeros((26, 2)), np.zeros(26), np.zeros(26), np.zeros((25, 2)),
                    frames=[np.zeros((2, 2, 3))] * 6, terminated=flags.copy(), truncated=flags.copy(),
                    observed=~flags, observation_valid=~flags)


def test_replay_requires_three_repeats_and_compares_terminal_masks_and_contacts():
    log = log_fixture()
    with pytest.raises(ValueError, match="three"):
        compare_replays([log, log])
    assert compare_replays([log, log, log])["bitwise"]
    changed = copy.deepcopy(log)
    changed.n_contacts[5] = 1
    assert not compare_replays([log, changed, log])["bitwise"]
    changed = copy.deepcopy(log)
    changed.terminated[25] = True
    assert not compare_replays([log, changed, log])["flags_equal"]


def test_replay_frame_length_mismatch_cannot_pass_zip_comparison():
    log = log_fixture()
    changed = copy.deepcopy(log)
    changed.frames = changed.frames[:-1]
    assert not compare_replays([log, changed, log])["frames_equal"]


def test_timing_excludes_terminal_padding_and_incomplete_endpoint_block():
    log = log_fixture()
    log.observed[8:] = False
    log.observation_valid[8:] = False
    log.states[8:, 2:5] = 0  # Dangerous fake terminal padding must never be scored.
    result = timing_row(log, Box(0, 20, 0, 20))
    assert result["evaluated_steps"] == 5
    assert result["excluded_observed_tail_steps"] == 2
    assert not result["dense_unsafe"]
    assert result["censored"]


def test_exact_contact_is_a_violation_in_timing_truth():
    log = log_fixture()
    result = timing_row(log, Box(310, 320, 250, 280))
    assert result["dense_unsafe"] and result["interpolated_unsafe"]


def test_live_pymunk_geometry_agrees_with_production_and_independent_geometry():
    env = make_env()
    try:
        env.reset(seed=1)
        result = live_geometry_check(env, [[250, 250, 0], [120, 350, .42], [310, 140, 2.8]])
        assert result["passes"]
        assert result["independent_geometry_cases"] == 12
    finally:
        env.close()
