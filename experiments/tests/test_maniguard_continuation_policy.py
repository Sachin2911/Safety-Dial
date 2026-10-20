"""CPU source-contract checks using a synthetic sampler, never a loaded VLA.

Run in the existing policy interpreter with JAX_PLATFORMS=cpu and one assigned
CPU. The fixture reuses the pinned Policy.infer path from the native unit tests.
No service, model weights, simulator or encoder is used.
"""

import os

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("openpi")

from test_maniguard_latent_policy import jax, observation, payload, policy

from helpers.maniguard_continuation_handoff import HandoffReference
from helpers.maniguard_latent import initial_weights
from helpers.maniguard_latent_policy import NativeNoisePolicy


@pytest.fixture(autouse=True)
def require_cpu_backend():
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert len(os.sched_getaffinity(0)) == 1
    assert jax.devices() and all(device.platform == "cpu" for device in jax.devices())


@pytest.mark.parametrize(
    "mode,branches",
    [
        ("native_only", ["native"] * 4),
        ("closure_only", ["closure"] * 4),
        ("clock_handoff", ["closure", "closure", "native", "native"]),
        ("privileged_lid_handoff", ["closure", "native", "native", "native"]),
    ],
)
def test_selector_weight_changes_keep_actual_native_rng_sequence(tmp_path, mode, branches):
    base, native = policy(), policy()
    native._rng = jax.random.key(123)
    selector = HandoffReference(initial_weights()[1], mode=mode, clock_switch_step=16)
    wrapper = NativeNoisePolicy(base, tmp_path / "synthetic-queries.jsonl")
    previous_after = None
    for index, branch in enumerate(branches):
        step = index * 8
        if selector.privileged:
            first = 0 if index == 0 else step - 7
            for observed in range(first, step + 1):
                selector.observe(observed, closed=observed > 0, supported=True, violated=False)
        decision = selector.select(step, payload()["latent_features"])
        assert decision["branch"] == branch
        weights = selector.native.weights if branch == "native" else selector.closure.weights
        result = wrapper.infer(payload(weights))
        expected = native.infer(observation())["actions"]
        receipt = result["latent_receipt"]
        assert receipt["query_index"] == index
        assert receipt["weights_sha256"] == decision["controller_sha256"]
        assert receipt["identity_native_path"] is (branch == "native")
        if previous_after is not None:
            assert receipt["rng_before"] == previous_after
        previous_after = receipt["rng_after"]
        np.testing.assert_array_equal(
            jax.random.key_data(base._rng), jax.random.key_data(native._rng)
        )
        if branch == "native":
            np.testing.assert_array_equal(result["actions"], expected)
        else:
            np.testing.assert_allclose(
                result["actions"], expected + decision["noise_offset"][:, :8], atol=1e-6
            )
        np.testing.assert_array_equal(result["actions"][:, 7], expected[:, 7])
    assert base.sampled_explicit == [branch == "closure" for branch in branches]
    assert len(native.sampled_explicit) == 4


def test_interleaved_episode_restarts_original_stream_and_would_break_a_chain(tmp_path):
    base = policy()
    wrapper = NativeNoisePolicy(base, tmp_path / "synthetic-queries.jsonl")
    request = payload(initial_weights()[1])
    first = wrapper.infer(request)["latent_receipt"]
    wrapper.infer({**request, "study_episode_id": "other-episode"})
    returned = wrapper.infer(request)["latent_receipt"]
    assert returned["query_index"] == first["query_index"] == 0
    assert returned["rng_before"] == first["rng_before"]
    assert returned["rng_before"] != first["rng_after"]
    assert returned["native_noise_sha256"] == first["native_noise_sha256"]
    assert len(base.sampled_explicit) == 3


def test_changing_seed_under_same_episode_also_restarts_its_query_sequence(tmp_path):
    base = policy()
    wrapper = NativeNoisePolicy(base, tmp_path / "synthetic-queries.jsonl")
    first = wrapper.infer(payload())["latent_receipt"]
    changed = wrapper.infer({**payload(), "episode_seed": 124})["latent_receipt"]
    assert changed["query_index"] == 0
    assert changed["rng_before"] != first["rng_after"]
    np.testing.assert_array_equal(changed["rng_before"], jax.random.key_data(jax.random.key(124)))
    assert len(base.sampled_explicit) == 2


def test_handoff_monitor_labels_are_not_an_extra_server_payload_channel(tmp_path):
    base = policy()
    wrapper = NativeNoisePolicy(base, tmp_path / "synthetic-queries.jsonl")
    with pytest.raises(ValueError, match="Unknown study payload"):
        wrapper.infer({**payload(), "study_handoff_label": True})
    assert not base.sampled_explicit and wrapper.last_episode is None
    assert not (tmp_path / "synthetic-queries.jsonl").exists()
