"""Native-policy wrapper retaining OpenPI's exact zero-steering inference path."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from helpers.maniguard_latent import FEATURE_DIM, LatentController, finite


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.shape).encode() + str(value.dtype).encode()
                          + value.tobytes()).hexdigest()


class NativeNoisePolicy:
    def __init__(self, policy, ledger, *, audit=False):
        import jax

        if policy._is_pytorch_model:
            raise ValueError("This source-audited hook requires the pinned JAX policy")
        self.jax, self.policy = jax, policy
        self.horizon = int(policy._model.action_horizon)
        self.action_dim = int(policy._model.action_dim)
        if (self.horizon, self.action_dim) != (16, 32):
            raise ValueError("Loaded native model shape differs from the frozen (16,32) contract")
        self.ledger = Path(ledger)
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.audit = audit
        self.last_episode = None
        self.query_index = 0

    def infer(self, observation):
        obs = dict(observation)
        episode = obs.pop("study_episode_id")
        seed = obs.pop("episode_seed")
        if not isinstance(episode, str) or not episode or type(seed) is not int or seed < 0:
            raise ValueError("Every request needs an explicit episode identifier and seed")
        weights = obs.pop("latent_weights")
        features = finite(obs.pop("latent_features"), (FEATURE_DIM,), "actor features")
        controller = LatentController(weights)
        explicit = obs.pop("latent_audit_noise", None)
        audit_offset = obs.pop("latent_audit_offset", None)
        if explicit is not None and audit_offset is not None:
            raise ValueError("An audit request must select noise or offset, not both")
        is_probe = explicit is not None or audit_offset is not None
        if is_probe and not self.audit:
            raise ValueError("Explicit probe noise is forbidden on the physical-study service")
        if any(k.startswith(("latent_", "study_")) for k in obs):
            raise ValueError("Unknown study payload")
        if (episode, seed) != self.last_episode:
            self.policy._rng = self.jax.random.key(seed)
            self.last_episode = (episode, seed)
            self.query_index = 0
        started = time.monotonic()
        before = self.jax.random.key_data(self.policy._rng)
        # Policy.infer assigns the FIRST split to its next key and gives the
        # SECOND split to Pi0.sample_actions. Draw the same batched native shape.
        _, sample_key = self.jax.random.split(self.policy._rng)
        native = np.asarray(self.jax.random.normal(
            sample_key, (1, self.horizon, self.action_dim)))
        offset = controller.offset(features, self.horizon, self.action_dim)
        noise = None if controller.identity else native + offset[None]
        if explicit is not None:
            noise = finite(explicit, (1, self.horizon, self.action_dim), "audit noise")
        if audit_offset is not None:
            noise = native + finite(audit_offset, (self.horizon, self.action_dim),
                                    "audit offset")[None]
        receipt = {"episode": episode, "query_index": self.query_index,
                   "seed": seed, "weights_sha256": controller.sha256(),
                   "features_sha256": array_hash(features),
                   "native_noise_sha256": array_hash(native),
                   "offset_sha256": array_hash(offset),
                   "rng_before": np.asarray(before).tolist(),
                   "identity_native_path": noise is None, "audit": is_probe,
                   "code": controller.code(features).tolist(), "shape": list(native.shape)}
        if audit_offset is not None:
            receipt["audit_offset_sha256"] = array_hash(audit_offset)
        try:
            # Omitting noise, rather than passing zeros or an externally drawn
            # equivalent, preserves native RNG advancement and numerical inference.
            result = self.policy.infer(obs) if noise is None else self.policy.infer(obs, noise=noise)
            receipt["actions_sha256"] = array_hash(result["actions"])
            receipt["rng_after"] = np.asarray(
                self.jax.random.key_data(self.policy._rng)).tolist()
            receipt["status"] = "returned"
            result["latent_receipt"] = receipt
            return result
        except Exception as exc:
            receipt.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
            raise
        finally:
            receipt["elapsed_seconds"] = time.monotonic() - started
            with self.ledger.open("a") as stream:
                stream.write(json.dumps(receipt, allow_nan=False) + "\n")
            self.query_index += 1
