#!/usr/bin/env python3
"""Retry-on-stall supervisor as a proxy policy server (todo 3, progress/03-retry-gate.md).

Sits between maniguard.eval.benchmark (openpi_client websocket protocol) and the real
policy server (maniguard.serve.openpi_native). Each evaluator connection gets its own
upstream connection, so the server's per-connection seeding is untouched. Every request
is forwarded to the upstream as the same bytes, and every upstream reply is forwarded
back as the same bytes, unless the supervisor is mid-retreat.

Modes:
  off    pass-through, byte-identical in both directions (the native arm).
  retry  pass-through plus the supervisor, driven only by observation/state
         (7 arm joints + 1 gripper) and episode_seed, both already in every request:
         - stall: over the last --patience queries (after --warmup queries) no arm joint
           moved more than --stall-rad from where it was at the start of the window;
         - retreat: --release-chunks chunks holding the current arm pose with the gripper
           open (+1), then --move-chunks chunks interpolating the arm back to the pose seen
           at the episode's first query, gripper open; then the policy resumes and the
           stall window restarts;
         - at most --max-retreats per episode.
         The retreat chunk has the upstream's shape (T, 8) and dtype; the upstream is
         still queried during a retreat so its state advances exactly as it would have.

Usage (openpi venv has websockets and msgpack; no openpi import needed):
    python -I scripts/todo3/supervisor_proxy.py --mode off   --listen 9000 --upstream 8000 --log /workspace/logs/proxy-off-9000.jsonl
    python -I scripts/todo3/supervisor_proxy.py --mode retry --listen 9100 --upstream 8000 --log /workspace/logs/proxy-retry-9100.jsonl

One JSONL line per episode (on seed change or disconnect) and one at every retreat start.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import os
import time
from collections import deque

import msgpack
import numpy as np
import websockets.asyncio.client as _client
import websockets.asyncio.server as _server
from websockets.exceptions import ConnectionClosed

log = logging.getLogger("supervisor_proxy")


# -- msgpack + numpy codec, identical to openpi_client.msgpack_numpy -----------------------
def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# -- the supervisor -------------------------------------------------------------------------
class RetrySupervisor:
    """Per-episode state machine. `observe(state)` is called once per query with the 8-D
    state; `chunk_for(upstream_chunk)` returns the chunk to send to the evaluator."""

    def __init__(self, *, patience: int, stall_rad: float, warmup: int, release_chunks: int,
                 move_chunks: int, max_retreats: int, execute_horizon: int):
        self.patience, self.stall_rad, self.warmup = patience, stall_rad, warmup
        self.release_chunks, self.move_chunks = release_chunks, move_chunks
        self.max_retreats, self.execute_horizon = max_retreats, execute_horizon
        self.reset()

    def reset(self):
        self.queries = 0
        self.home = None                      # arm pose at the first query
        self.window = deque(maxlen=self.patience)
        self.retreats: list[int] = []         # query index at which each retreat started
        self.retreat_step = None              # index of the next retreat chunk, or None
        self.retreat_from = None              # arm pose at retreat start
        self.trace: list[float] = []          # per query: max |q - window[0]| (nan while filling)
        self.last_state = None

    @property
    def retreating(self) -> bool:
        return self.retreat_step is not None

    def observe(self, state: np.ndarray) -> None:
        q = np.asarray(state, dtype=np.float64).reshape(-1)[:7]
        self.queries += 1
        self.last_state = q
        if self.home is None:
            self.home = q.copy()
        if self.retreating:
            self.trace.append(float("nan"))
            return
        self.window.append(q)
        moved = float(np.max(np.abs(q - self.window[0]))) if len(self.window) == self.patience else float("nan")
        self.trace.append(moved)
        stalled = (len(self.window) == self.patience and self.queries > self.warmup
                   and moved < self.stall_rad and len(self.retreats) < self.max_retreats)
        if stalled:
            self.retreats.append(self.queries)
            self.retreat_step = 0
            self.retreat_from = q.copy()

    def chunk_for(self, upstream_chunk: np.ndarray) -> np.ndarray | None:
        """None means: send the upstream bytes unchanged."""
        if not self.retreating:
            return None
        T, A = upstream_chunk.shape
        k = self.retreat_step
        out = np.empty((T, A), dtype=upstream_chunk.dtype)
        out[:, -1] = 1.0                      # gripper open (evaluator binarizes: +1 open)
        if A > 8:
            out[:, 7:-1] = 0.0                # never the case for the 8-D jar checkpoint
        if k < self.release_chunks:
            out[:, :7] = self.retreat_from
        else:
            moves = k - self.release_chunks
            S = self.move_chunks * self.execute_horizon
            for i in range(T):
                g = moves * self.execute_horizon + i + 1
                alpha = min(g / S, 1.0)
                out[i, :7] = self.retreat_from + alpha * (self.home - self.retreat_from)
        self.retreat_step = k + 1
        if self.retreat_step >= self.release_chunks + self.move_chunks:
            self.retreat_step = None
            self.window.clear()               # patience restarts after a retreat
        return out

    def summary(self) -> dict:
        return {"queries": self.queries, "retreats": self.retreats, "n_retreats": len(self.retreats),
                "home": None if self.home is None else [round(float(x), 5) for x in self.home],
                "trace_max_joint_change": [None if np.isnan(x) else round(x, 5) for x in self.trace]}


# -- the proxy ------------------------------------------------------------------------------
class Proxy:
    def __init__(self, args):
        self.args = args
        self.upstream_uri = f"ws://{args.upstream_host}:{args.upstream}"
        self.log_path = args.log
        self.conn_id = 0
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)

    def _write(self, record: dict) -> None:
        record = {"t": time.time(), "mode": self.args.mode, "listen": self.args.listen, **record}
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        log.info("%s", json.dumps(record)[:300])

    def _new_supervisor(self) -> RetrySupervisor:
        a = self.args
        return RetrySupervisor(patience=a.patience, stall_rad=a.stall_rad, warmup=a.warmup,
                               release_chunks=a.release_chunks, move_chunks=a.move_chunks,
                               max_retreats=a.max_retreats, execute_horizon=a.execute_horizon)

    async def handler(self, ws):
        self.conn_id += 1
        cid = self.conn_id
        sup = self._new_supervisor()
        seed = None
        async with _client.connect(self.upstream_uri, compression=None, max_size=None) as up:
            await ws.send(await up.recv())                      # metadata, verbatim
            try:
                while True:
                    req = await ws.recv()
                    if self.args.mode == "retry" and isinstance(req, (bytes, bytearray)):
                        obs = unpackb(req)
                        s = obs.get("episode_seed")
                        if s != seed:
                            if seed is not None:
                                self._write({"event": "episode_end", "conn": cid, "episode_seed": seed, **sup.summary()})
                            seed, sup = s, self._new_supervisor()
                        was_retreating = sup.retreating
                        sup.observe(obs["observation/state"])
                        if sup.retreating and not was_retreating:
                            self._write({"event": "retreat_start", "conn": cid, "episode_seed": seed,
                                         "query": sup.queries, "n_retreats": len(sup.retreats),
                                         "arm_state": [round(float(x), 5) for x in sup.last_state]})
                    await up.send(req)
                    resp = await up.recv()
                    if isinstance(resp, str):                   # upstream error text: pass on and stop
                        await ws.send(resp)
                        break
                    if self.args.mode == "retry" and sup.retreating:
                        result = unpackb(resp)
                        chunk = np.asarray(result["actions"])
                        new = sup.chunk_for(chunk)
                        if new is not None:
                            result["actions"] = new
                            result["supervisor"] = {"retreat_chunk": sup.retreat_step, "n_retreats": len(sup.retreats)}
                            resp = packb(result)
                    await ws.send(resp)
            except ConnectionClosed:
                pass
            finally:
                if self.args.mode == "retry" and seed is not None:
                    self._write({"event": "episode_end", "conn": cid, "episode_seed": seed, **sup.summary()})
                self._write({"event": "disconnect", "conn": cid})

    async def run(self):
        async with _server.serve(self.handler, self.args.listen_host, self.args.listen,
                                 compression=None, max_size=None) as server:
            self._write({"event": "listening", "upstream": self.upstream_uri,
                         "params": {k: getattr(self.args, k) for k in
                                    ("patience", "stall_rad", "warmup", "release_chunks", "move_chunks",
                                     "max_retreats", "execute_horizon")}})
            await server.serve_forever()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("off", "retry"), required=True)
    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--upstream-host", default="127.0.0.1")
    ap.add_argument("--upstream", type=int, required=True)
    ap.add_argument("--log", default=None, help="JSONL log path")
    ap.add_argument("--patience", type=int, default=10, help="queries in the stall window (10 x 8 steps = 4 s)")
    ap.add_argument("--stall-rad", type=float, default=0.01, help="max arm-joint change counted as stalled")
    ap.add_argument("--warmup", type=int, default=10, help="no retreat before this many queries")
    ap.add_argument("--release-chunks", type=int, default=1, help="chunks holding still with the gripper open")
    ap.add_argument("--move-chunks", type=int, default=3, help="chunks interpolating back to the first-query pose")
    ap.add_argument("--max-retreats", type=int, default=3)
    ap.add_argument("--execute-horizon", type=int, default=8, help="actions the evaluator executes per chunk")
    return ap.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(Proxy(parse_args(argv)).run())


if __name__ == "__main__":
    main()
