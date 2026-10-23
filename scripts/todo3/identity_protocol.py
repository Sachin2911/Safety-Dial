"""Protocol-level identity check of supervisor_proxy.py --mode off against the live server.

The episode-level check (compare_chunks.py on two recorded episodes) needs the native pipeline
to be bitwise reproducible, which on the host it is not (see receipts/03-identity.md). This
check removes the simulator: the same request bytes (fixed images and state, the Jar prompt,
one episode seed) go once to the server directly and once through the proxy, each sequence
started from a freshly re-seeded sampling key (the server re-seeds when the episode seed
changes, so a one-request sequence with a different seed is sent in between). A third
sequence repeats the direct one to measure the server's own run-to-run determinism.

    third_party/openpi/.venv/bin/python -I scripts/todo3/identity_protocol.py --direct 8000 --proxy 9000 --n 30 --out <json>

Every request is a model query and is counted in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import websockets.sync.client as _sync

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supervisor_proxy import packb, unpackb  # noqa: E402

PROMPT = ("Close the lid of the hinged jar holding the can of beans, then carry the closed jar "
          "into the green goal sphere on the table.")
STATE0 = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.04], dtype=np.float32)


def requests(n, seed, image_seed=1234):
    rng = np.random.default_rng(image_seed)
    out = []
    for i in range(n):
        st = STATE0 + np.float32(0.01) * np.float32(i) * np.array([1, -1, 1, -1, 1, -1, 1, 0], np.float32)
        out.append(packb({"observation/image_left": rng.integers(256, size=(256, 256, 3), dtype=np.uint8),
                          "observation/wrist_image": rng.integers(256, size=(256, 256, 3), dtype=np.uint8),
                          "observation/state": st, "prompt": PROMPT, "episode_seed": int(seed)}))
    return out


def run(port, reqs):
    conn = _sync.connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None)
    meta = conn.recv()
    replies, times = [], []
    for r in reqs:
        t0 = time.time(); conn.send(r); replies.append(conn.recv()); times.append(time.time() - t0)
    conn.close()
    return meta, replies, times


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direct", type=int, default=8000)
    ap.add_argument("--proxy", type=int, default=9000)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1528289969)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    reqs = requests(args.n, args.seed)
    sep = requests(1, 0)                                     # flips the server's last seed
    queries = 0
    run(args.direct, sep); queries += 1                      # every sequence below starts freshly keyed
    meta_d, direct, t_d = run(args.direct, reqs); queries += len(reqs)
    run(args.direct, sep); queries += 1
    meta_p, proxied, t_p = run(args.proxy, reqs); queries += len(reqs)
    run(args.direct, sep); queries += 1
    meta_d2, direct2, t_d2 = run(args.direct, reqs); queries += len(reqs)
    h = lambda b: hashlib.sha256(b).hexdigest()[:16]
    dec = lambda rs: [unpackb(r) for r in rs]
    dd, pp, dd2 = dec(direct), dec(proxied), dec(direct2)
    # the reply carries server_timing (wall-clock, never repeats): compare the action bytes and
    # every other key separately
    same_dp = [a["actions"].tobytes() == b["actions"].tobytes() for a, b in zip(dd, pp)]
    same_dd = [a["actions"].tobytes() == b["actions"].tobytes() for a, b in zip(dd, dd2)]
    other_keys = sorted({k for r in dd + pp + dd2 for k in r} - {"actions"})
    other_same = {k: int(sum(a.get(k) == b.get(k) for a, b in zip(dd, pp))) for k in other_keys}
    whole_same = int(sum(a == b for a, b in zip(direct, proxied)))
    chunks = [np.asarray(r["actions"]) for r in dd]
    chunks_p = [np.asarray(r["actions"]) for r in pp]
    chunks_d2 = [np.asarray(r["actions"]) for r in dd2]
    res = {
        "n": args.n, "seed": args.seed, "model_queries": queries,
        "metadata_identical": meta_d == meta_p == meta_d2,
        "direct_vs_proxied_actions_identical": int(sum(same_dp)), "direct_vs_direct2_actions_identical": int(sum(same_dd)),
        "direct_vs_proxied_whole_reply_identical": whole_same, "other_reply_keys": other_keys,
        "direct_vs_proxied_other_keys_identical": other_same,
        "request_bytes_sha256_first": h(reqs[0]), "reply_bytes": len(direct[0]),
        "chunk_shape": list(chunks[0].shape), "chunk_dtype": str(chunks[0].dtype),
        "max_abs_diff_direct_proxied": float(max(np.abs(a - b).max() for a, b in zip(chunks, chunks_p))),
        "max_abs_diff_direct_direct2": float(max(np.abs(a - b).max() for a, b in zip(chunks, chunks_d2))),
        "ms_direct": [round(1000 * t, 1) for t in t_d], "ms_proxied": [round(1000 * t, 1) for t in t_p],
        "reply_sha256_direct": [h(r) for r in direct], "reply_sha256_proxied": [h(r) for r in proxied],
        "reply_sha256_direct2": [h(r) for r in direct2],
    }
    json.dump(res, open(args.out, "w"), indent=1)
    print({k: v for k, v in res.items() if not k.startswith(("ms_", "reply_sha"))})
    print("PASS" if res["metadata_identical"] and sum(same_dp) == args.n else "FAIL")
    if sum(same_dd) != args.n:
        print(f"note: the server itself returned different actions for identical requests in {args.n - sum(same_dd)} of {args.n} cases")


if __name__ == "__main__":
    main()
