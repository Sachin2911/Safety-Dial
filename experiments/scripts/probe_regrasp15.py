"""Warm up and check the task 15 policy server with synthetic 256 px observations.

--warmup compiles the native path and the K-candidate path (the first query of each
shape blocks for the JIT compile, which can outlast the client keepalive, so it is
retried on a fresh connection), then checks that a candidate request leaves the native
sampling stream untouched: the native draws after re-seeding must be identical with and
without a candidate request in between. --reseed-only sends one native query with
episode_seed 0 so the next episode's seed always re-seeds the server. Every query made
here is a model query and is counted in the accounting.

    third_party/openpi/.venv/bin/python -I experiments/scripts/probe_regrasp15.py --warmup --k 16
"""

import argparse
import sys
import time

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

PROMPT = (
    "Close the lid of the hinged jar holding the can of beans, then carry the closed jar "
    "into the green goal sphere on the table."
)


def observation(seed=None):
    rng = np.random.default_rng(0)
    obs = {
        "observation/image_left": rng.integers(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": PROMPT,
    }
    if seed is not None:
        obs["episode_seed"] = int(seed)
    return obs


class Probe:
    def __init__(self, host, port):
        self.host, self.port = host, port
        self.native = 0
        self.candidates = 0
        self.client = WebsocketClientPolicy(host=host, port=port)

    def infer(self, obs, *, patience=0.0):
        """One query. Within `patience` seconds, a dropped connection or a handshake that
        times out while the server is blocked in a compile is retried on a new one."""
        deadline = time.time() + patience
        attempt = 0
        while True:
            attempt += 1
            started = time.time()
            if "regrasp15" in obs:
                self.candidates += obs["regrasp15"]["k"]
            else:
                self.native += 1
            try:
                if self.client is None:
                    self.client = WebsocketClientPolicy(host=self.host, port=self.port)
                result = self.client.infer(dict(obs))
                print(f"query answered in {time.time() - started:.1f}s", flush=True)
                return result
            except Exception as exc:  # noqa: BLE001 - keepalive drop during compile
                print(f"attempt {attempt} dropped after {time.time() - started:.0f}s: "
                      f"{type(exc).__name__}", flush=True)
                self.client = None
                if time.time() >= deadline:
                    raise
                time.sleep(10)


def warmup(probe, k):
    print("server metadata:", probe.client.get_server_metadata(), flush=True)
    probe.infer(observation(0), patience=900)
    request = {**observation(), "regrasp15": {"k": k, "seed": 1}}
    first = probe.infer(request, patience=900)
    candidates = np.asarray(first["candidates"])
    if candidates.shape != (k, 16, 8) or not np.isfinite(candidates).all():
        print("unexpected candidate set", candidates.shape)
        return 1
    again = np.asarray(probe.infer(request)["candidates"])
    print(f"candidate decode {float(first['decode_s']):.2f}s; repeat max abs difference "
          f"{np.abs(again - candidates).max():.2e}", flush=True)
    draws = []
    for between in (False, True):
        probe.infer(observation(0))
        first_draw = np.asarray(probe.infer(observation(777))["actions"])
        if between:
            probe.infer(request)
        second_draw = np.asarray(probe.infer(observation(777))["actions"])
        draws.append((first_draw, second_draw))
    (a1, a2), (b1, b2) = draws
    gap = max(np.abs(a1 - b1).max(), np.abs(a2 - b2).max())
    distinct = np.abs(a1 - a2).max()
    print(f"native stream: re-seed gap {gap:.2e}, consecutive-draw difference {distinct:.2e}",
          flush=True)
    probe.infer(observation(0))
    # A disturbed stream would differ by about as much as two independent draws do.
    if distinct < 1e-3 or gap > 0.05 * distinct:
        print("candidate request disturbed the native stream, or draws did not advance")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--warmup", action="store_true")
    mode.add_argument("--reseed-only", action="store_true")
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args()
    probe = Probe(args.host, args.port)
    if args.reseed_only:
        actions = np.asarray(probe.infer(observation(0))["actions"])
        code = 0 if actions.shape == (16, 8) and np.isfinite(actions).all() else 1
    else:
        code = warmup(probe, args.k)
    print(f"model queries made by this probe: native {probe.native}, "
          f"candidate decodes {probe.candidates}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
