"""Serve the frozen Jar pi0.5 policy with the task 15 candidate path.

Native requests follow maniguard.serve.openpi_native exactly: episode_seed is stripped
and re-seeds the policy key only when it changes, then Policy.infer runs unchanged. A
request carrying a `regrasp15` field instead decodes K candidates for the same
observation from latents drawn with the recorded seed, through the policy's own
transforms and jitted sampler, without reading or advancing the native key. Every
candidate request is appended to a ledger.

Run with the OpenPI interpreter, PYTHONPATH=third_party/ManiGuard:experiments/helpers.
"""

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/helpers"))
from regrasp15_sampler import (  # noqa: E402
    CONFIG,
    RAW_KEYS,
    CandidateSampler,
    candidate_noise,
    load_policy,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MAX_K = 64


class RegraspPolicy:
    def __init__(self, policy, ledger):
        self._policy = policy
        self._sampler = CandidateSampler(policy)
        self._last_seed = None
        self._ledger = ledger
        self._lock = threading.Lock()

    @property
    def metadata(self):
        return self._policy.metadata

    def infer(self, obs):
        with self._lock:
            request = obs.pop("regrasp15", None)
            if request is None:
                seed = obs.pop("episode_seed", None)
                if seed is not None and seed != self._last_seed:
                    import jax

                    self._policy._rng = jax.random.key(int(seed))
                    self._last_seed = seed
                    logger.info(f"sampling RNG re-seeded: episode_seed={seed}")
                return self._policy.infer(obs)
            episode_seed = obs.pop("episode_seed", None)
            if set(request) != {"k", "seed"}:
                raise ValueError("A candidate request names only k and seed")
            k, seed = int(request["k"]), int(request["seed"])
            if not 1 <= k <= MAX_K:
                raise ValueError("Candidate count outside the frozen range")
            if set(obs) != set(RAW_KEYS):
                raise ValueError("Unexpected fields at the frozen VLA input boundary")
            chunks, decode_s = self._sampler.decode(obs, candidate_noise(seed, k))
            self._ledger.write(json.dumps({
                "utc": time.time(), "k": k, "seed": seed, "decode_s": decode_s,
                "episode_seed": None if episode_seed is None else int(episode_seed),
                "last_native_seed": None if self._last_seed is None else int(self._last_seed),
            }) + "\n")
            self._ledger.flush()
            return {"candidates": chunks, "noise_seed": np.int64(seed), "k": np.int64(k),
                    "decode_s": np.float64(decode_s)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    from openpi.serving import websocket_policy_server

    policy = load_policy(args.checkpoint)
    logger.info("Model loaded successfully.")
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.open("a") as ledger:
        server = websocket_policy_server.WebsocketPolicyServer(
            RegraspPolicy(policy, ledger),
            host=args.host,
            port=args.port,
            metadata={"serve_config": CONFIG, "checkpoint": str(args.checkpoint),
                      "regrasp15": True},
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
