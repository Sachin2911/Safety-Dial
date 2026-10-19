"""Exercise the pinned OpenPI inference path with a lightweight native-noise sampler."""

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("openpi")

from openpi.policies.policy import Policy

from helpers.maniguard_latent import FEATURE_DIM, LatentController, initial_weights
from helpers.maniguard_latent_policy import NativeNoisePolicy


def policy():
    instance = object.__new__(Policy)
    instance._model = SimpleNamespace(action_horizon=16, action_dim=32)
    instance._is_pytorch_model = False
    instance._rng = jax.random.key(0)
    instance._sample_kwargs = {}
    instance._input_transform = lambda value: value
    instance._output_transform = lambda value: {**value, "actions": value["actions"][:, :8]}
    instance.sampled_explicit = []

    def sample(key, observation, noise=None):
        instance.sampled_explicit.append(noise is not None)
        return jax.random.normal(key, (1, 16, 32)) if noise is None else noise

    instance._sample_actions = sample
    return instance


def observation():
    return {"state": np.zeros(32, dtype=np.float32),
            "image": {"base_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8)},
            "image_mask": {"base_0_rgb": np.asarray(True)}}


def payload(weights=None):
    return {**observation(), "study_episode_id": "physical-a", "episode_seed": 123,
            "latent_features": np.r_[1, np.zeros(FEATURE_DIM - 1)].astype(np.float32),
            "latent_weights": initial_weights()[0] if weights is None else weights}


def test_zero_preserves_actual_policy_rng_and_native_branch(tmp_path):
    base, reference = policy(), policy()
    reference._rng = jax.random.key(123)
    wrapper = NativeNoisePolicy(base, tmp_path / "queries.jsonl")
    for index in range(3):
        result = wrapper.infer(payload())
        expected = reference.infer(observation())
        np.testing.assert_array_equal(result["actions"], expected["actions"])
        np.testing.assert_array_equal(jax.random.key_data(base._rng),
                                      jax.random.key_data(reference._rng))
        assert result["latent_receipt"]["identity_native_path"]
        assert result["latent_receipt"]["query_index"] == index
    assert base.sampled_explicit == [False] * 3


def test_active_shift_uses_correct_split_and_leaves_gripper_noise_native(tmp_path):
    base, reference = policy(), policy()
    reference._rng = jax.random.key(123)
    expected = reference.infer(observation())["actions"]
    request = payload(initial_weights()[1])
    result = NativeNoisePolicy(base, tmp_path / "queries.jsonl").infer(request)
    offset = LatentController(request["latent_weights"]).offset(request["latent_features"], 16, 32)
    np.testing.assert_allclose(result["actions"], expected + offset[:, :8], atol=1e-6)
    np.testing.assert_array_equal(result["actions"][:, 7], expected[:, 7])
    np.testing.assert_array_equal(jax.random.key_data(base._rng),
                                  jax.random.key_data(reference._rng))
    assert base.sampled_explicit == [True]


def test_physical_service_rejects_audit_override(tmp_path):
    base = policy()
    wrapper = NativeNoisePolicy(base, tmp_path / "queries.jsonl")
    with pytest.raises(ValueError):
        wrapper.infer({**payload(), "latent_audit_offset": np.zeros((16, 32))})
    assert not base.sampled_explicit


def test_audit_zero_offset_matches_native_actual_noise(tmp_path):
    base = policy()
    wrapper = NativeNoisePolicy(base, tmp_path / "queries.jsonl", audit=True)
    native = wrapper.infer(payload())
    explicit = wrapper.infer({**payload(), "study_episode_id": "audit-b",
                              "latent_audit_offset": np.zeros((16, 32))})
    np.testing.assert_array_equal(native["actions"], explicit["actions"])
    assert native["latent_receipt"]["native_noise_sha256"] == \
        explicit["latent_receipt"]["native_noise_sha256"]
