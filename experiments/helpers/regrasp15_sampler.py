"""Batched pi0.5 candidate decoding that leaves the native sampling stream untouched.

The native query keeps its ordinary Policy.infer call and key split. Candidates reuse
the policy's own input transform, jitted sampler and output transform, but take
explicit Gaussian latents drawn from a recorded numpy seed and a fixed JAX key that the
sampler ignores when noise is supplied, so the policy's key is never read or advanced.
The K latents are decoded in batches of at most 16 copies of the same observation, which
keeps the decode inside the server's share of a GPU it shares with the simulator.
"""

import time

import numpy as np

LATENT_SHAPE = (16, 32)
CHUNK_SHAPE = (16, 8)
MAX_BATCH = 16
RAW_KEYS = ("observation/image_left", "observation/wrist_image", "observation/state", "prompt")
CONFIG = "pi05-base_datagen_v1_jar_joint_2cam_lora"


def candidate_noise(seed, count):
    """Standard normal latents for `count` candidates, reproducible from the seed."""
    if count < 1:
        raise ValueError("At least one candidate latent is required")
    rng = np.random.default_rng(np.random.SeedSequence([15, int(seed)]))
    return rng.standard_normal((int(count), *LATENT_SHAPE)).astype(np.float32)


def load_policy(checkpoint, config=CONFIG):
    """Create the frozen policy exactly as maniguard.serve.openpi_native does."""
    import maniguard.openpi_sft.train_configs as train_configs
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    train_configs.register()
    return policy_config.create_trained_policy(train_config.get_config(config), str(checkpoint))


class CandidateSampler:
    """Decode explicit latents for one observation with the policy's own sampler."""

    def __init__(self, policy):
        import jax

        if "noise" in getattr(policy, "_sample_kwargs", {}):
            raise ValueError("The native sampler must not contain a preset noise override")
        self.policy = policy
        self.key = jax.random.key(0)  # Unused: sample_actions draws no noise when given some.

    def decode(self, obs, noise):
        """Return (K, 16, 8) chunks in the policy's output space and the decode seconds."""
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        noise = np.asarray(noise, dtype=np.float32)
        if noise.ndim != 3 or noise.shape[1:] != LATENT_SHAPE or not np.isfinite(noise).all():
            raise ValueError("Candidate latents must be finite (K, 16, 32) arrays")
        if set(obs) != set(RAW_KEYS):
            raise ValueError("Unexpected fields at the frozen VLA input boundary")
        policy = self.policy
        key_before = policy._rng
        count = noise.shape[0]
        # Same copy-then-transform order as Policy.infer, then copies instead of one.
        inputs = policy._input_transform(jax.tree.map(lambda x: x, dict(obs)))
        chunks, elapsed = [], 0.0
        for first in range(0, count, MAX_BATCH):
            part = noise[first:first + MAX_BATCH]
            size = part.shape[0]
            batched = jax.tree.map(
                lambda x, size=size: jnp.broadcast_to(
                    jnp.asarray(x)[np.newaxis, ...], (size, *np.shape(x))
                ),
                inputs,
            )
            observation = _model.Observation.from_dict(batched)
            start = time.monotonic()
            actions = policy._sample_actions(
                self.key, observation, **policy._sample_kwargs, noise=jnp.asarray(part)
            )
            actions = np.asarray(actions)
            elapsed += time.monotonic() - start
            state = np.asarray(batched["state"])
            chunks.extend(
                np.asarray(policy._output_transform({"state": state[k], "actions": actions[k]})[
                    "actions"
                ])
                for k in range(size)
            )
        chunks = np.stack(chunks)
        if policy._rng is not key_before:
            raise RuntimeError("Candidate decoding touched the native sampling key")
        if chunks.shape != (count, *CHUNK_SHAPE) or not np.isfinite(chunks).all():
            raise ValueError("Candidates are not finite (K, 16, 8) chunks")
        return chunks, elapsed
