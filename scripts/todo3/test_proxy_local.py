#!/usr/bin/env python3
"""Local check of supervisor_proxy.py with a fake upstream, no GPU, no openpi.

    .venv/bin/python -I scripts/todo3/test_proxy_local.py

Fake upstream: speaks the openpi websocket protocol (metadata first, then one reply per
request) and returns a deterministic chunk (16, 8) that depends on the request count and
the state it was sent. A fake evaluator drives: direct, proxy off, proxy retry with a state
sequence that moves for 12 queries then stalls. Checks:
  1. off mode: every reply byte-identical to the direct reply, metadata too.
  2. retry mode: identical until the stall is detected at the expected query; the retreat is
     1 release chunk (hold, gripper +1) then 3 move chunks reaching the first-query pose;
     then pass-through resumes; at most 3 retreats; the log has the episode line.
  3. a seed change inside one connection resets the supervisor.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import websockets.asyncio.server as _server
import websockets.sync.client as _sync
from websockets.exceptions import ConnectionClosed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from supervisor_proxy import packb, unpackb  # noqa: E402

UP, OFF, RETRY = 18000, 19000, 19100
T, A = 16, 8


async def fake_upstream():
    async def handler(ws):
        await ws.send(packb({"serve_config": "fake", "checkpoint": "none"}))
        n = 0
        try:
            while True:
                obs = unpackb(await ws.recv())
                n += 1
                st = np.asarray(obs["observation/state"], dtype=np.float32)
                chunk = (np.arange(T * A, dtype=np.float32).reshape(T, A) * 1e-3 + n * 0.01 + st.sum())
                await ws.send(packb({"actions": chunk, "server_timing": {"infer_ms": 1.0}}))
        except ConnectionClosed:
            pass
    async with _server.serve(handler, "127.0.0.1", UP, compression=None, max_size=None):
        await asyncio.Future()


def states(n_move=12, n_total=60):
    """Arm moves 0.05 rad per query for n_move queries, then holds still."""
    q0 = np.linspace(-0.5, 0.5, 7).astype(np.float32)
    out = []
    for i in range(n_total):
        q = q0 + 0.05 * min(i, n_move)
        out.append(np.concatenate([q, [0.04]]).astype(np.float32))
    return out


def drive(port, seq, seed=12345, seed_switch_at=None):
    conn = _sync.connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None, legacy=True)
    meta = conn.recv()
    replies = []
    for i, st in enumerate(seq):
        s = seed if (seed_switch_at is None or i < seed_switch_at) else seed + 1
        obs = {"observation/image_left": np.zeros((4, 4, 3), np.uint8), "observation/wrist_image": np.zeros((4, 4, 3), np.uint8),
               "observation/state": st, "prompt": "close the jar", "episode_seed": s}
        conn.send(packb(obs))
        replies.append(conn.recv())
    conn.close()
    return meta, replies


def start(args, log):
    return subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "supervisor_proxy.py"), *args, "--log", log],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_port(port, t=10):
    for _ in range(int(t * 10)):
        try:
            _sync.connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None, legacy=True).close(); return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"port {port} not up")


def episode_lines(path, n, t=5.0):
    """Wait until the log has at least n episode_end lines (the proxy writes them after the
    client closes), then return them."""
    for _ in range(int(t * 20)):
        lines = [json.loads(l) for l in open(path)]
        eps = [l for l in lines if l["event"] == "episode_end"]
        if len(eps) >= n:
            return eps
        time.sleep(0.05)
    raise AssertionError(f"log has {len(eps)} episode lines, wanted {n}")


def main():
    import threading
    threading.Thread(target=lambda: asyncio.run(fake_upstream()), daemon=True).start()
    wait_port(UP)
    tmp = tempfile.mkdtemp(prefix="proxytest-")
    p_off = start(["--mode", "off", "--listen", str(OFF), "--upstream", str(UP)], f"{tmp}/off.jsonl")
    p_retry = start(["--mode", "retry", "--listen", str(RETRY), "--upstream", str(UP)], f"{tmp}/retry.jsonl")
    try:
        wait_port(OFF); wait_port(RETRY)
        seq = states()
        meta_d, direct = drive(UP, seq)
        meta_o, off = drive(OFF, seq)
        assert meta_d == meta_o, "metadata differs through proxy"
        assert all(a == b for a, b in zip(direct, off)) and len(direct) == len(off), "off mode not byte-identical"
        print(f"off mode: {len(off)} replies byte-identical to direct (metadata too)")

        meta_r, retry = drive(RETRY, seq)
        assert meta_r == meta_d
        # window fills at query 10 (patience 10); stall needs 10 identical states after the move
        # ends at query 13 (i=12 is the first stalled state), so the window [13..22] is still at
        # query 22: first retreat starts at query 22 (1-based).
        first_diff = next(i for i, (a, b) in enumerate(zip(direct, retry)) if a != b) + 1
        assert first_diff == 22, f"first retreat at query {first_diff}, expected 22"
        home = seq[0][:7].astype(np.float64)
        chunks = [np.asarray(unpackb(r)["actions"]) for r in retry]
        rel = chunks[21]
        assert rel.shape == (T, A) and rel.dtype == np.float32
        assert np.all(rel[:, -1] == 1.0) and np.allclose(rel[:, :7], seq[21][:7]), "release chunk wrong"
        move = chunks[22:25]
        assert all(np.all(m[:, -1] == 1.0) for m in move), "gripper not open during move"
        assert np.allclose(move[-1][7, :7], home, atol=1e-6), "move does not reach home at executed step 8 of chunk 3"
        assert np.all(np.diff(np.abs(move[0][:, 0] - home[0])) <= 1e-7), "move not monotone toward home"
        assert retry[25] == direct[25], "pass-through did not resume after the retreat"
        n_retreat_replies = sum(1 for a, b in zip(direct, retry) if a != b)
        print(f"retry mode: first retreat at query {first_diff}; {n_retreat_replies} replaced replies; "
              f"release chunk holds pose with gripper +1; move reaches home; pass-through resumes")
        # with the fake state still frozen, further retreats fire every patience window: cap at 3
        ep = episode_lines(f"{tmp}/retry.jsonl", 1)[-1]
        assert ep["n_retreats"] == 3 and ep["retreats"][0] == 22, ep
        assert n_retreat_replies == 12, n_retreat_replies
        print(f"log: episode line with retreats at queries {ep['retreats']} (cap 3), {ep['queries']} queries")

        # seed switch mid-connection resets the supervisor: no retreat right after the switch
        _, mixed = drive(RETRY, seq[:30] + seq[:20], seed_switch_at=30)
        eps = episode_lines(f"{tmp}/retry.jsonl", 3)[-2:]
        assert eps[0]["queries"] == 30 and eps[1]["queries"] == 20 and eps[1]["n_retreats"] == 0, eps
        print("seed change inside a connection starts a fresh episode in the log")
        print("ALL CHECKS PASSED")
    finally:
        p_off.terminate(); p_retry.terminate()


if __name__ == "__main__":
    main()
