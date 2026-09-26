"""Walker2d root/branch banks for S4: roots from recorded episodes, exact-restore branches,
dense truth for both rules, block-endpoint frames, and clip construction for adaptation.

Roots come from `roots.h5` (held-out episodes of the set-A policies). The stress bank adds
roots within 0.5 s (62 steps) before a health violation in terminated episodes and roots
at speeds near the limit. A branch restores (qpos, qvel) exactly (walkerRules.py), so the
charged cost is the branch horizon (HORIZON_BLOCKS * FRAMESKIP env steps) plus root
generation, which is counted once per root as the recorded prefix.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from helpers.walkerRules import (
    FRAMESKIP,
    HISTORY,
    HORIZON_BLOCKS,
    SPEED_LIMIT,
    WalkerRoot,
    execute_branch,
    health_clearance,
    propose_tapes,
    roots_from_episode,
    speed_clearance,
)

MIN_PREFIX = (HISTORY - 1) * FRAMESKIP
HORIZON_STEPS = HORIZON_BLOCKS * FRAMESKIP


def iter_episodes(h5_path: Path):
    import hdf5plugin  # noqa: F401

    with h5py.File(h5_path, "r") as f:
        ln, off = f["ep_len"][:], f["ep_offset"][:]
        for i in range(len(ln)):
            a, b = int(off[i]), int(off[i] + ln[i])
            yield i, {k: f[k][a:b] for k in ("qpos", "qvel", "action", "x_velocity", "healthy", "terminated")}


def sample_roots(h5_path: Path, rng, n_roots: int, *, kind: str = "representative", episode_filter=None, prefix: str = "w") -> list[WalkerRoot]:
    """kind: representative (uniform over eligible steps) | stress (pre-violation or near the speed limit)."""
    roots = []
    eps = list(iter_episodes(h5_path))
    rng.shuffle(eps)
    for ei, ep in eps:
        if episode_filter is not None and not episode_filter(ei):
            continue
        n = len(ep["qpos"])
        hi = n - HORIZON_STEPS
        if hi <= MIN_PREFIX:
            continue
        if kind == "representative":
            steps = [int(rng.integers(MIN_PREFIX, hi))]
        else:
            cands = []
            if bool(ep["terminated"][-1]):
                cands += list(range(max(MIN_PREFIX, n - 62 - HORIZON_STEPS), hi))
            v = ep["x_velocity"]
            near = np.where(np.abs(v[MIN_PREFIX - 1 : hi - 1] - SPEED_LIMIT) < 0.3)[0] + MIN_PREFIX
            cands += near.tolist()
            if not cands:
                continue
            steps = [int(rng.choice(cands))]
        roots += roots_from_episode(ep, steps, episode=ei, prefix=prefix)
        if len(roots) >= n_roots:
            break
    return roots[:n_roots]


class WalkerBankWriter:
    def __init__(self, bank_dir: Path):
        self.dir = Path(bank_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(self.dir / "branches.h5", "w")
        T = HORIZON_STEPS
        spec = {"tape": ((0, T, 6), np.float32), "qpos": ((0, T + 1, 9), np.float64), "qvel": ((0, T + 1, 9), np.float64), "x_velocity": ((0, T), np.float64),
                "root_index": ((0,), np.int64), "kind": ((0,), h5py.string_dtype()), "params": ((0,), h5py.string_dtype()),
                "frames": ((0, HORIZON_BLOCKS + 1, 224, 224, 3), np.uint8)}
        for k, (shape, dt) in spec.items():
            kw = {"compression": "gzip", "compression_opts": 1, "chunks": (1, *shape[1:])} if k == "frames" else {"chunks": (64, *shape[1:])}
            self.h5.create_dataset(k, shape=shape, maxshape=(None, *shape[1:]), dtype=dt, **kw)
        self.roots: list[WalkerRoot] = []
        self.root_index: dict[str, int] = {}

    def add_root(self, root: WalkerRoot) -> int:
        if root.root_id not in self.root_index:
            self.roots.append(root)
            self.root_index[root.root_id] = len(self.roots) - 1
        return self.root_index[root.root_id]

    def add_branches(self, root: WalkerRoot, items: list[tuple[np.ndarray, str, dict]], env) -> list:
        ri = self.add_root(root)
        n0 = self.h5["tape"].shape[0]
        n = len(items)
        for k in self.h5:
            self.h5[k].resize(n0 + n, axis=0)
        logs = []
        for i, (tape, kind, params) in enumerate(items):
            log = execute_branch(env, root.qpos, root.qvel, tape, render_every=FRAMESKIP)
            j = n0 + i
            self.h5["tape"][j] = tape.astype(np.float32)
            self.h5["qpos"][j] = log.qpos
            self.h5["qvel"][j] = log.qvel
            self.h5["x_velocity"][j] = log.x_velocity
            self.h5["root_index"][j] = ri
            self.h5["kind"][j] = kind
            self.h5["params"][j] = json.dumps(params)
            self.h5["frames"][j] = np.stack(log.frames)
            logs.append(log)
        self.h5.flush()
        return logs

    def finish(self, meta: dict) -> None:
        (self.dir / "roots.json").write_text(json.dumps({"roots": [r.to_dict() for r in self.roots], "meta": meta, "n_branches": int(self.h5["tape"].shape[0])}) + "\n")
        self.h5.close()


class WalkerBank:
    def __init__(self, bank_dir: Path):
        self.dir = Path(bank_dir)
        blob = json.loads((self.dir / "roots.json").read_text())
        self.roots = [WalkerRoot.from_dict(r) for r in blob["roots"]]
        self.meta = blob["meta"]
        self.h5 = h5py.File(self.dir / "branches.h5", "r")

    def __len__(self) -> int:
        return int(self.h5["tape"].shape[0])

    def indices_for_root(self, ri: int) -> np.ndarray:
        return np.where(self.h5["root_index"][:] == ri)[0]

    def truth(self, j: int) -> dict:
        qp, xv = self.h5["qpos"][j], self.h5["x_velocity"][j]
        return {"speed": speed_clearance(xv), "health": health_clearance(qp[1:, 1], qp[1:, 2])}


def build_walker_bank(bank_dir: Path, roots: list[WalkerRoot], rng, env, *, n_tapes: int, stress: bool, seed: int, verbose=True) -> WalkerBank:
    import time

    w = WalkerBankWriter(bank_dir)
    t0 = time.time()
    for i, root in enumerate(roots):
        if stress:
            items = propose_tapes(rng, root, n_random=n_tapes // 2, sigmas=(0.3, 0.5), n_bursts=n_tapes - n_tapes // 2)[1:]
        else:
            items = propose_tapes(rng, root, n_random=n_tapes - 1, sigmas=(0.1, 0.2, 0.4))
        w.add_branches(root, items[:n_tapes], env)
        if verbose and (i + 1) % 16 == 0:
            print(f"[wbank:{bank_dir.name}] {i + 1}/{len(roots)} roots {time.time() - t0:.0f}s")
    w.finish({"seed": seed, "n_tapes": n_tapes, "stress": stress, "charged_steps_per_branch": HORIZON_STEPS})
    return WalkerBank(bank_dir)


def endpoint_targets(qpos: np.ndarray, x_velocity: np.ndarray) -> np.ndarray:
    """True (height, pitch, speed) at block endpoints 1..K, shape (K, 3)."""
    return np.stack([qpos[FRAMESKIP::FRAMESKIP, 1], qpos[FRAMESKIP::FRAMESKIP, 2], x_velocity[FRAMESKIP - 1 :: FRAMESKIP]], 1)


def interp_steps(vals: np.ndarray, block: int = FRAMESKIP) -> np.ndarray:
    """(K+1,) endpoint values -> (K*block,) per-step values by linear interpolation (row 0 = root)."""
    K = len(vals) - 1
    w = np.arange(1, block + 1) / block
    return (vals[:-1, None] * (1 - w) + vals[1:, None] * w).reshape(K * block)
