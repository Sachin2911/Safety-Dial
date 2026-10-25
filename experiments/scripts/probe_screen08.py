"""Warm up and check the Jar policy server with synthetic 256 px observations.

The native server runs JAX inference synchronously inside its websocket loop, so
the first query after a server start blocks for the JIT compile (minutes) and the
default openpi client keepalive closes that connection. Run this once after each
server start and before the official runner. Every query it makes is a model
query to be counted in the accounting (see progress/receipts).

    third_party/openpi/.venv/bin/python -I experiments/scripts/probe_screen08.py [--queries 3]
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


def observation(rng):
    return {
        "observation/image_left": rng.integers(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": PROMPT,
        "episode_seed": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--queries", type=int, default=3)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    queries = 0
    # The compile query may drop the connection; reconnect and count it anyway.
    for attempt in range(3):
        client = WebsocketClientPolicy(host=args.host, port=args.port)
        meta = client.get_server_metadata()
        try:
            queries += 1
            started = time.time()
            print("screen08_query_attempt", queries, flush=True)
            client.infer(observation(rng))
            print("screen08_query_returned", queries, flush=True)
            print(f"warm-up query answered in {time.time() - started:.1f}s")
            break
        except Exception as exc:  # noqa: BLE001 - keepalive drop during compile
            print(
                f"warm-up attempt {attempt + 1} dropped after {time.time() - started:.0f}s: {type(exc).__name__}"
            )
            time.sleep(5)
    else:
        print("server never answered a warm-up query")
        return 1
    print("server metadata:", meta)
    for i in range(args.queries):
        queries += 1
        started = time.time()
        print("screen08_query_attempt", queries, flush=True)
        result = client.infer(observation(rng))
        print("screen08_query_returned", queries, flush=True)
        actions = np.asarray(result["actions"])
        print(
            f"query {i}: {1000 * (time.time() - started):.0f} ms, actions {actions.shape} {actions.dtype}"
        )
        if actions.shape != (16, 8) or not np.all(np.isfinite(actions)):
            print("unexpected action chunk")
            return 1
    print(f"model queries made by this probe: {queries}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
