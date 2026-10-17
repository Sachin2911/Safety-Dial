"""AAC source parity and joint-port semantics; not manipulation evidence.

Native-format golden results below were collected by evaluating the actual
author select_chunk_size(method='gaussian_bernoulli') on these fixtures at
ff8c8e421f7805b5845a72d07fbd8f1171df74bf. Source SHA256 values are checked in
the configuration. Only the selector's pure dependencies were loaded for that
audit; CI requires no network, upstream checkout, policy or simulator.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.maniguard_phase2_aac import (
    SOURCE_HELPER_SHA256,
    SOURCE_REVISION,
    SOURCE_SELECTOR_SHA256,
    AACConfig,
    entropy_profile,
    load_config,
    select_horizon,
    select_joint_prefix,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/maniguard/phase2/aac.json"
LOW = 2.4529314838331233
HIGH = 10.452904964246203
JUMP_MEANS = [LOW, LOW, LOW, 4.452924853936393, 5.652920875998355,
              6.452918224039663, 7.024344901212025, 7.452914909091298]


def native_fixture(scales, toggle=False):
    """20 native six-coordinate samples; raw Cartesian x deltas are .1."""
    base = np.concatenate([np.eye(6), -np.eye(6), np.zeros((8, 6))])
    values = base[:, None, :] * np.array(scales)[None, :, None]
    signed_gripper = np.tile(np.r_[np.ones(10), -np.ones(10)][:, None], (1, len(scales)))
    if toggle:
        signed_gripper[:, 2:] *= -1
    binary = (signed_gripper >= 0).astype(int)
    # All rotations are identity; this is the exact native Cartesian movement
    # construction for this fixture. Sample-zero toggle weight upstream is .2.
    movement = np.cumsum(np.full(len(scales), .1))
    if toggle:
        movement[2:] += .2
    return values, binary, movement


@pytest.mark.parametrize("scales,threshold,toggle,expected_horizon,expected_entropy", [
    ([1, 1, 1] + [np.exp(8 / 6)] * 5, 0, False, 3, [LOW] * 3 + [HIGH] * 5),
    ([1, 1, 1] + [np.exp(8 / 6)] * 5, .45, False, 5, [LOW] * 3 + [HIGH] * 5),
    ([1] * 8, 0, False, 7, [LOW] * 8),
    ([1, 3], 0, False, 2, [LOW, 9.04457988264215]),
    ([1] * 4, .35, True, 3, [LOW] * 4),
    ([1, 2, 3, 4, 5], 5, False, 5,
     [LOW, 6.611793192319707, 9.04457988264215, 10.770670931937312, 12.109531598572882]),
])
def test_native_gaussian_bernoulli_selector_matches_actual_author_golden(
        scales, threshold, toggle, expected_horizon, expected_entropy):
    continuous, binary, movement = native_fixture(scales, toggle)
    profile = entropy_profile(continuous, binary, group_sizes=(3, 3))
    np.testing.assert_allclose(profile.total, expected_entropy, rtol=0, atol=2e-14)
    np.testing.assert_allclose(profile.gripper, np.log(2), rtol=0, atol=1e-15)
    assert select_horizon(profile.cumulative_mean, movement, threshold)[0] == expected_horizon
    if len(scales) == 8 and scales[-1] != 1:
        np.testing.assert_allclose(profile.cumulative_mean, JUMP_MEANS, rtol=0, atol=2e-14)


def test_author_constant_entropy_rounding_is_preserved_and_exact_ties_choose_first():
    continuous, binary, movement = native_fixture([1] * 8)
    profile = entropy_profile(continuous, binary, group_sizes=(3, 3))
    np.testing.assert_array_equal(profile.total, np.full(8, LOW))
    # The pinned author implementation also returns seven here: prefix means
    # differ by one ulp. Do not disguise this as an exact mathematical tie.
    assert select_horizon(profile.cumulative_mean, movement, 0) == (7, 7, 2)
    assert select_horizon(np.ones(8), movement, 0) == (2, 2, 2)


def test_mean_entropy_maximum_change_has_author_action_count_not_next_index():
    # Maximum change between means for prefixes three and four executes three.
    # Argmin mean, argmax mean, and an extra +1 all select a different horizon.
    assert select_horizon([0, 0, 0, 2, 3, 3.5], np.ones(6), None) == (3, 3, 2)


def test_movement_floor_uses_strict_first_crossing_and_full_horizon_fallback():
    means = np.ones(5)
    assert select_horizon(means, [0, .5, 1, 1.1, 1.5], 1) == (4, 2, 4)
    assert select_horizon(means, [0, 1.1, .1, .5, .4], 1) == (2, 2, 2)
    assert select_horizon(means, np.zeros(5), 0) == (5, 2, 5)
    assert select_horizon(means, np.zeros(5), None) == (2, 2, 2)


def test_multivariate_covariance_is_not_mean_variance_uncertainty():
    rng = np.random.default_rng(72)
    samples = rng.normal(size=(40, 1, 2))
    correlated = samples.copy()
    correlated[:, 0, 1] = correlated[:, 0, 0]
    correlated[:, 0, 1] *= samples[:, 0, 1].std() / correlated[:, 0, 1].std()
    np.testing.assert_allclose(samples.var(axis=0), correlated.var(axis=0))
    grip = np.zeros((40, 1))
    first = entropy_profile(samples, grip, group_sizes=(2,))
    second = entropy_profile(correlated, grip, group_sizes=(2,))
    assert first.continuous[0] > second.continuous[0] + 4


def test_joint_prefix_is_sample_zero_bit_exact_and_discarded_tail_cannot_influence_it():
    rng = np.random.default_rng(100)
    proposals = rng.normal(size=(20, 25, 8)).astype(np.float32)
    original = proposals.copy()
    config = AACConfig(movement_threshold=1e6)
    selected = select_joint_prefix(proposals, np.zeros(7), config)
    assert selected.horizon == 16 and selected.prefix.dtype == np.float32
    np.testing.assert_array_equal(selected.prefix, original[0, :16])
    np.testing.assert_array_equal(proposals, original)
    assert not np.shares_memory(selected.prefix, proposals)
    proposals[:, 16:] = np.nan
    again = select_joint_prefix(proposals, np.zeros(7), config)
    assert again.record() == selected.record()
    np.testing.assert_array_equal(again.prefix, selected.prefix)
    json.dumps(selected.record(), allow_nan=False)


@pytest.mark.parametrize("length", [1, 2, 3, 7, 16])
def test_short_chunks_are_not_padded_or_quantized(length):
    samples = np.zeros((20, length, 8), dtype=np.float32)
    samples[:, :, 7] = -1
    selected = select_joint_prefix(samples, np.zeros(7))
    assert selected.horizon == length  # stationary movement never clears the floor
    assert selected.prefix.shape == (length, 8)
    np.testing.assert_array_equal(selected.prefix, samples[0])
    assert len(selected.entropy.total) == length


def test_seventh_arm_joint_affects_entropy_independently_of_gripper():
    samples = np.zeros((20, 8, 8))
    values = np.linspace(-1, 1, 20)
    samples[:, :, 6] = .2 * values[:, None]
    samples[:, 3:, 6] *= np.exp(4)
    samples[:, :, 7] = -1
    selected = select_joint_prefix(samples, np.zeros(7), AACConfig(movement_threshold=None))
    assert selected.entropy_horizon == 3 and selected.horizon == 3
    assert selected.entropy.continuous[3] > selected.entropy.continuous[2] + 3.9
    np.testing.assert_array_equal(selected.entropy.gripper,
                                  np.full(8, selected.entropy.gripper[0]))


def test_joint_floor_uses_endpoint_delta_from_actual_joints_not_absolute_sum():
    q = np.full(7, 4.0)
    samples = np.empty((20, 5, 8))
    samples[:, :, :7] = q
    samples[:, :, 7] = -1
    stationary = select_joint_prefix(samples, q)
    np.testing.assert_array_equal(stationary.movement_magnitudes, np.zeros(5))
    assert stationary.horizon == 5
    samples[:, :, 2] += np.array([0, .05, .10, .15, .20])
    moved = select_joint_prefix(samples, q, AACConfig(movement_threshold=.6))
    np.testing.assert_allclose(moved.movement_magnitudes, [0, .25, .5, .75, 1])
    assert moved.movement_horizon == 4
    # Coordinate origin changes cannot manufacture additional movement.
    translated = samples.copy()
    translated[:, :, :7] += 2
    shifted = select_joint_prefix(translated, q + 2, AACConfig(movement_threshold=.6))
    np.testing.assert_allclose(shifted.entropy.total, moved.entropy.total)
    np.testing.assert_allclose(shifted.movement_magnitudes, moved.movement_magnitudes)
    assert shifted.horizon == moved.horizon


def test_gripper_toggle_uses_benchmark_open_state_without_rewriting_zero_command():
    samples = np.zeros((20, 4, 8))
    samples[:, :, 7] = [-1, -1, 0, 1]
    selected = select_joint_prefix(samples, np.zeros(7), AACConfig(movement_threshold=.1))
    np.testing.assert_array_equal(selected.movement_magnitudes, [0, 0, 0, .2])
    assert selected.movement_horizon == selected.horizon == 4
    np.testing.assert_array_equal(selected.prefix[:, 7], [-1, -1, 0, 1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_joint_gripper_threshold_matches_benchmark_and_preserves_raw_boundary_commands(dtype):
    threshold = dtype(.01)
    below = np.nextafter(threshold, dtype(-np.inf))
    above = np.nextafter(threshold, dtype(np.inf))
    commands = np.array([-1, -.01, -0.0, 0.0, .005, below, threshold, above], dtype=dtype)
    # Pinned ManiGuard benchmark.py: sign only when abs(command) > .01;
    # otherwise close. This oracle is the execution conversion, not AAC's rule.
    executed = np.array([np.sign(value) if abs(value) > .01 else -1 for value in commands])
    np.testing.assert_array_equal(executed, [-1, -1, -1, -1, -1, -1, -1, 1])
    samples = np.zeros((20, len(commands), 8), dtype=dtype)
    samples[:, :, 7] = commands
    original = samples.copy()
    selected = select_joint_prefix(samples, np.zeros(7), AACConfig(movement_threshold=.1))
    toggled = np.r_[False, np.maximum.accumulate(np.diff(executed) != 0)]
    np.testing.assert_array_equal(selected.movement_magnitudes, .2 * toggled)
    assert selected.movement_horizon == selected.horizon == len(commands)
    assert selected.prefix.dtype == samples.dtype
    assert selected.prefix.tobytes() == original[0].tobytes()
    assert samples.tobytes() == original.tobytes()
    assert not np.shares_memory(selected.prefix, samples)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_joint_gripper_entropy_uses_executed_state_including_deadband(dtype):
    threshold = dtype(.01)
    above = np.nextafter(threshold, dtype(np.inf))
    commands = np.array([-1, -.01, -0.0, 0.0, .005, threshold, above, 1], dtype=dtype)
    samples = np.zeros((20, len(commands), 8), dtype=dtype)
    samples[:10, :, 7] = -1
    samples[10:, :, 7] = commands
    selected = select_joint_prefix(samples, np.zeros(7), AACConfig(movement_threshold=None))
    # All twenty executed commands close in the first six columns. Only the
    # final two have a 50/50 open/close split. Small raw positives add no entropy.
    assert np.all(selected.entropy.gripper[:6] < 1e-6)
    np.testing.assert_allclose(selected.entropy.gripper[6:], np.log(2), rtol=0, atol=1e-15)
    np.testing.assert_array_equal(selected.movement_magnitudes, np.zeros(len(commands)))
    assert selected.prefix.tobytes() == samples[0, :selected.horizon].tobytes()


@pytest.mark.parametrize("kwargs", [
    {"sample_count": 1}, {"sample_count": True}, {"sample_count": 20.0},
    {"max_prefix": 50}, {"joint_scale_radians": 0}, {"joint_scale_radians": np.inf},
    {"covariance_epsilon": -1}, {"covariance_epsilon": np.nan},
    {"movement_threshold": -1}, {"movement_threshold": True},
    {"gripper_movement_weight": -1}, {"gripper_entropy_weight": np.inf},
])
def test_invalid_settings_fail_before_selection(kwargs):
    with pytest.raises(ValueError):
        AACConfig(**kwargs)


@pytest.mark.parametrize("samples,q", [
    (np.zeros((19, 8, 8)), np.zeros(7)),
    (np.zeros((20, 0, 8)), np.zeros(7)),
    (np.zeros((20, 8, 7)), np.zeros(7)),
    (np.zeros((20, 8, 8)), np.zeros(8)),
    (np.full((20, 8, 8), np.nan), np.zeros(7)),
    (np.zeros((20, 8, 8)), np.full(7, np.inf)),
    (np.ones((20, 8, 8), dtype=bool), np.zeros(7)),
    (np.ones((20, 8, 8), dtype=complex), np.zeros(7)),
    (np.full((20, 8, 8), 1e308), np.zeros(7)),
])
def test_invalid_or_overflowing_inputs_fail_without_commands(samples, q):
    with pytest.raises(ValueError):
        select_joint_prefix(samples, q)


def test_entropy_rejects_nonbinary_gripper_and_invalid_partition():
    with pytest.raises(ValueError, match="binary"):
        entropy_profile(np.zeros((20, 4, 7)), np.full((20, 4), .5), group_sizes=(7,))
    with pytest.raises(ValueError, match="partition"):
        entropy_profile(np.zeros((20, 4, 7)), np.ones((20, 4)), group_sizes=(3, 3))
    with pytest.raises(ValueError, match="N>=2"):
        entropy_profile(np.zeros((1, 4, 7)), np.ones((1, 4)), group_sizes=(7,))
    with pytest.raises(ValueError):
        select_horizon([0, np.inf], [0, 0], 1)


def test_config_records_pinned_license_unfrozen_status_and_shared_settings(tmp_path):
    data = json.loads(CONFIG.read_text())
    config = load_config(CONFIG)
    assert config == AACConfig()
    assert data["status"] == "unfrozen_pure_core"
    assert data["freeze_requirements"]["settings_frozen"] is False
    assert data["source"]["revision"] == SOURCE_REVISION
    assert data["source"]["selector_sha256"] == SOURCE_SELECTOR_SHA256
    assert data["source"]["helper_sha256"] == SOURCE_HELPER_SHA256
    assert data["source"]["license"] == "MIT"
    for field, bad in [("revision", "unversioned"), ("selector_sha256", "0" * 64),
                       ("helper_sha256", "0" * 64)]:
        altered = json.loads(CONFIG.read_text())
        altered["source"][field] = bad
        path = tmp_path / "bad-source.json"
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError, match="source pin"):
            load_config(path)
    data["settings"]["family"] = "jar"
    path = tmp_path / "unknown-setting.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="setting names"):
        load_config(path)
