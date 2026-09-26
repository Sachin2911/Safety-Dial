"""Roots, proposals, branch execution and storage for the Push-T study.

Roots (pushT.md): a seeded reset with a start and goal, then `k` blocks of the nominal
controller (LeWM + CEM with the arena guard, `imagination.NominalPlanner`), stored as
(seed, reset options, recorded action prefix). Every replay charges `len(prefix)` env
steps; the prefix always starts with `HISTORY_FRAMES - 1` idle blocks so the model
conditions on three real frames (pushtReplay.py).

Start/goal pairs come from expert episodes in the `roots` split: the start is the expert
state at step `t0` with velocities zeroed, the goal is the expert state `goal_offset`
steps later. This is the regime the released checkpoint was evaluated in upstream
(le-wm `config/eval/pusht.yaml`: goals 25 steps ahead). The env's own random start/goal
sampler produces long-horizon goals on which the released planner flails even with the
arena guard (checked on 26 September 2026), so those are not used.

Proposal generator (common to every arm): the nominal 5-block plan at the root plus
bounded Gaussian perturbations of every action at declared noise scales; stress tapes aim
the pusher at T vertices and edges, or at a target point. Tapes whose commanded pusher
path leaves the arena are rejected before execution (the shared engineering control),
and rejections are counted as model-side work, not simulator steps.

Storage: `roots.json` (roots, ledger, manifest) and `branches.h5` (tapes, dense truth,
block-endpoint frames) per bank under `data/study/pusht/<bank>/`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from helpers.pushtAssets import ACTION_BLOCK, ACTION_SCALE, HISTORY_FRAMES, HORIZON_BLOCKS
from helpers.pushtGeometry import ARENA_HI, ARENA_LO, t_polygons
from helpers.pushtReplay import (
    DenseLog,
    Root,
    RootContext,
    StepLedger,
    execute_tape,
    idle_prefix,
    reset_root,
    run_actions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = REPO_ROOT / "data" / "study" / "pusht"
ACTION_CLIP = 0.6  # raw units; 0.6 * 100 px commanded offset per env step (expert std 0.21)


# --------------------------------------------------------------------------------------
# start/goal pairs from expert episodes
# --------------------------------------------------------------------------------------
def expert_pairs(h5_path: Path, episodes: list[int], rng: np.random.Generator, n: int, *,
                 t0_range=(0, 20), goal_offset_range=(25, 50)) -> list[dict]:
    """Sample n (episode, t0, goal_offset) triples and read their states."""
    import hdf5plugin  # noqa: F401

    out = []
    with h5py.File(h5_path, "r") as f:
        ep_len, ep_off = f["ep_len"][:], f["ep_offset"][:]
        eps = rng.choice(np.asarray(episodes), size=n, replace=len(episodes) < n)
        for ep in eps:
            ep = int(ep)
            L = int(ep_len[ep])
            goff = int(rng.integers(goal_offset_range[0], goal_offset_range[1] + 1))
            hi = min(t0_range[1], L - 1 - goff)
            if hi < t0_range[0]:
                goff = max(1, L - 1 - t0_range[0])
                hi = t0_range[0]
            t0 = int(rng.integers(t0_range[0], hi + 1))
            st = f["state"][int(ep_off[ep]) + t0 : int(ep_off[ep]) + t0 + goff + 1].astype(np.float64)
            start, goal = st[0].copy(), st[-1].copy()
            start[5:] = 0.0
            goal[5:] = 0.0
            out.append({"episode": ep, "t0": t0, "goal_offset": goff, "start": start, "goal": goal})
    return out


# --------------------------------------------------------------------------------------
# nominal controller loop
# --------------------------------------------------------------------------------------
@dataclass
class NominalRun:
    blocks: np.ndarray  # (k, ACTION_BLOCK, 2) executed nominal blocks
    log: DenseLog  # dense truth over the k blocks (from after the idle prefix)
    costs: list[float]
    solve_times: list[float]
    frac_feasible: list[float]
    plan_at_end: np.ndarray  # (HORIZON_BLOCKS, ACTION_BLOCK, 2) nominal plan at the final state


def run_nominal(env, planner, ctx: RootContext, k: int, *, seed: int, record_frames: bool = True) -> NominalRun:
    """Execute k receding-horizon nominal blocks from the current env state."""
    from helpers.imagination import next_warm_start

    frames, hist, state = list(ctx.frames), ctx.history_actions.copy(), ctx.state.copy()
    blocks, costs, times, feas, warm = [], [], [], [], None
    states = [state.copy()]
    bvel, bang, ncon, fr = [np.zeros(2)], [0.0], [0], [frames[-1]]
    for b in range(k):
        pr = planner.plan(frames, hist, ctx.goal_frame, seed=seed * 1000 + b, pusher_xy=state[:2], init_future=warm)
        log = run_actions(env, pr.blocks[0].reshape(-1, 2), record_frames=True)
        frames = (frames + [log.frames[-1]])[-HISTORY_FRAMES:]
        hist = np.concatenate([hist[1:], pr.blocks[:1]], 0)
        warm = next_warm_start(pr.blocks)
        state = log.states[-1].copy()
        blocks.append(pr.blocks[0])
        costs.append(pr.cost)
        times.append(pr.solve_time)
        feas.append(pr.frac_feasible)
        states.extend(log.states[1:])
        bvel.extend(log.block_vel[1:])
        bang.extend(log.block_ang_vel[1:])
        ncon.extend(log.n_contacts[1:])
        fr.append(log.frames[-1])
    final_plan = planner.plan(frames, hist, ctx.goal_frame, seed=seed * 1000 + 999, pusher_xy=state[:2], init_future=warm).blocks
    dense = DenseLog(np.asarray(states), np.asarray(bvel), np.asarray(bang), np.asarray(ncon),
                     np.asarray(blocks).reshape(-1, 2) if blocks else np.zeros((0, 2)), fr if record_frames else None)
    return NominalRun(np.asarray(blocks).reshape(-1, ACTION_BLOCK, 2), dense, costs, times, feas, final_plan)


def build_root(env, planner, pair: dict, *, seed: int, k: int, root_id: str, ledger: StepLedger | None = None) -> tuple[Root, RootContext, NominalRun]:
    """Seeded reset + idle prefix + k nominal blocks -> a stored Root with its nominal plan."""
    base = Root(seed=seed, start_state=pair["start"], goal_state=pair["goal"], prefix=idle_prefix())
    ctx0 = reset_root(env, base)
    nom = run_nominal(env, planner, ctx0, k, seed=seed)
    prefix = np.concatenate([base.prefix, nom.blocks.reshape(-1, 2)], 0) if k else base.prefix
    meta = {
        "episode": int(pair["episode"]), "t0": int(pair["t0"]), "goal_offset": int(pair["goal_offset"]),
        "k": int(k), "nominal_plan": nom.plan_at_end.tolist(),
        "contact_steps_prefix": int((nom.log.n_contacts > 0).sum()),
        "in_contact_last_block": bool((nom.log.n_contacts[-ACTION_BLOCK:] > 0).any()) if k else False,
        "state_at_root": nom.log.states[-1].tolist(),
        "nominal_costs": [float(c) for c in nom.costs],
        "block_err_px": float(np.linalg.norm(nom.log.states[-1, 2:4] - pair["goal"][2:4])),
    }
    root = Root(seed=seed, start_state=pair["start"], goal_state=pair["goal"], prefix=prefix, root_id=root_id, meta=meta)
    if ledger is not None:
        ledger.add("root_prefix", len(prefix))
    # the env is now AT the root state; return a context consistent with reset_root
    ctx = reset_root(env, root)
    return root, ctx, nom


# --------------------------------------------------------------------------------------
# proposals
# --------------------------------------------------------------------------------------
def commanded_path(tape: np.ndarray, pusher_xy) -> np.ndarray:
    a = np.asarray(tape, dtype=np.float64).reshape(-1, 2)
    return np.asarray(pusher_xy, dtype=np.float64) + np.cumsum(a * ACTION_SCALE, axis=0)


def exits_arena(tape: np.ndarray, pusher_xy, lo=ARENA_LO, hi=ARENA_HI) -> bool:
    p = commanded_path(tape, pusher_xy)
    return bool((p < lo).any() or (p > hi).any())


def perturbed_tapes(rng, nominal: np.ndarray, n: int, sigma: float, *, clip: float = ACTION_CLIP) -> np.ndarray:
    """n perturbations of a (K, 5, 2) plan: Gaussian noise on every action, clipped."""
    noise = rng.normal(0.0, sigma, size=(n, *nominal.shape))
    return np.clip(nominal[None] + noise, -clip, clip)


def aimed_tape(pusher_xy, target_xy, speed: float, n_blocks: int = HORIZON_BLOCKS, *, overshoot: float = 1.0) -> np.ndarray:
    """A straight push from the pusher through `target_xy` at `speed` raw units/step.

    The commanded step is `speed*100` px; the tape keeps pushing past the target by
    `overshoot` times the initial distance, then holds still.
    """
    p = np.asarray(pusher_xy, float)
    d = np.asarray(target_xy, float) - p
    dist = np.linalg.norm(d) + 1e-9
    u = d / dist
    total = dist * (1.0 + overshoot)
    steps = n_blocks * ACTION_BLOCK
    per = speed * ACTION_SCALE
    tape = np.zeros((steps, 2))
    travelled = 0.0
    for t in range(steps):
        if travelled >= total:
            break
        step = min(per, total - travelled)
        tape[t] = u * step / ACTION_SCALE
        travelled += step
    return tape.reshape(n_blocks, ACTION_BLOCK, 2)


def t_targets(state) -> dict[str, np.ndarray]:
    """Named aim points on the T footprint: its vertices and edge midpoints."""
    polys = t_polygons(state[2:5])
    pts = {}
    for name, poly in zip(("bar", "stem"), polys):
        for i, v in enumerate(poly):
            pts[f"{name}_v{i}"] = v
            pts[f"{name}_e{i}"] = 0.5 * (v + poly[(i + 1) % 4])
    pts["body"] = np.asarray(state[2:4], float)
    return pts


@dataclass
class Proposal:
    tape: np.ndarray  # (K, 5, 2)
    kind: str
    params: dict = field(default_factory=dict)


def propose(rng, root: Root, ctx: RootContext, *, n_random: int, sigmas=(0.05, 0.1, 0.2),
            n_stress: int = 0, stress_speeds=(0.1, 0.2, 0.3), extra_targets: dict | None = None,
            max_tries: int = 20) -> tuple[list[Proposal], int]:
    """The common candidate pool for one root. Returns (proposals, n_rejected_by_arena)."""
    nominal = np.asarray(root.meta["nominal_plan"], float)
    state = np.asarray(ctx.state, float)
    out: list[Proposal] = []
    rejected = 0
    if not exits_arena(nominal, state[:2]):
        out.append(Proposal(nominal.copy(), "nominal", {"sigma": 0.0}))
    per_sigma = max(1, n_random // len(sigmas))
    for sig in sigmas:
        got = 0
        tries = 0
        while got < per_sigma and tries < max_tries * per_sigma:
            cand = perturbed_tapes(rng, nominal, per_sigma, sig)
            for t in cand:
                tries += 1
                if exits_arena(t, state[:2]):
                    rejected += 1
                    continue
                out.append(Proposal(t, "random", {"sigma": float(sig)}))
                got += 1
                if got >= per_sigma:
                    break
    if n_stress:
        targets = t_targets(state)
        if extra_targets:
            targets.update({k: np.asarray(v, float) for k, v in extra_targets.items()})
        names = list(targets)
        got = 0
        tries = 0
        while got < n_stress and tries < max_tries * n_stress:
            tries += 1
            name = names[int(rng.integers(len(names)))]
            speed = float(rng.choice(stress_speeds))
            jitter = rng.normal(0.0, 8.0, size=2)
            t = aimed_tape(state[:2], targets[name] + jitter, speed, overshoot=float(rng.uniform(0.3, 1.5)))
            t = np.clip(t + rng.normal(0.0, 0.03, size=t.shape), -ACTION_CLIP, ACTION_CLIP)
            if exits_arena(t, state[:2]):
                rejected += 1
                continue
            out.append(Proposal(t, "stress", {"target": name, "speed": speed}))
            got += 1
    return out, rejected


# --------------------------------------------------------------------------------------
# execution and storage
# --------------------------------------------------------------------------------------
@dataclass
class Branch:
    root_id: str
    tape: np.ndarray  # (K, 5, 2)
    kind: str
    params: dict
    log: DenseLog


def execute_proposals(env, root: Root, proposals: list[Proposal], *, ledger: StepLedger | None = None,
                      record_frames: bool = True, category: str = "branch") -> list[Branch]:
    """Execute every proposal from the root (reset + prefix each time; all charged)."""
    out = []
    for p in proposals:
        reset_root(env, root, record_frames=False)
        log = execute_tape(env, p.tape.reshape(-1, 2), record_frames=record_frames)
        if ledger is not None:
            ledger.add(category, len(root.prefix) + p.tape.size // 2, branches=1)
        out.append(Branch(root.root_id, p.tape, p.kind, p.params, log))
    return out


class BankWriter:
    """Append branches to `<dir>/branches.h5`; roots and ledger to `<dir>/roots.json`."""

    def __init__(self, bank_dir: Path, *, n_blocks: int = HORIZON_BLOCKS, with_frames: bool = True):
        self.dir = Path(bank_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(self.dir / "branches.h5", "a")
        self.n_blocks = n_blocks
        self.with_frames = with_frames
        T = n_blocks * ACTION_BLOCK
        spec = {
            "tape": ((0, n_blocks, ACTION_BLOCK, 2), np.float32),
            "states": ((0, T + 1, 7), np.float64),
            "block_vel": ((0, T + 1, 2), np.float64),
            "block_ang_vel": ((0, T + 1), np.float64),
            "n_contacts": ((0, T + 1), np.int32),
            "root_index": ((0,), np.int64),
            "kind": ((0,), h5py.string_dtype()),
            "params": ((0,), h5py.string_dtype()),
        }
        if with_frames:
            spec["frames"] = ((0, n_blocks + 1, 224, 224, 3), np.uint8)
        for k, (shape, dt) in spec.items():
            if k not in self.h5:
                maxshape = (None, *shape[1:])
                chunks = (1, *shape[1:]) if k == "frames" else (64, *shape[1:])
                kw = {"compression": "gzip", "compression_opts": 1} if k == "frames" else {}
                self.h5.create_dataset(k, shape=shape, maxshape=maxshape, dtype=dt, chunks=chunks, **kw)
        self.roots: list[dict] = []
        self.root_index: dict[str, int] = {}
        if (self.dir / "roots.json").is_file():
            blob = json.loads((self.dir / "roots.json").read_text())
            self.roots = blob["roots"]
            self.root_index = {r["root_id"]: i for i, r in enumerate(self.roots)}

    def add_root(self, root: Root) -> int:
        if root.root_id in self.root_index:
            return self.root_index[root.root_id]
        self.roots.append(root.to_dict())
        self.root_index[root.root_id] = len(self.roots) - 1
        return self.root_index[root.root_id]

    def add_branches(self, branches: list[Branch]) -> None:
        n0 = self.h5["tape"].shape[0]
        n = len(branches)
        for k in self.h5:
            self.h5[k].resize(n0 + n, axis=0)
        for i, b in enumerate(branches):
            j = n0 + i
            self.h5["tape"][j] = b.tape.astype(np.float32)
            self.h5["states"][j] = b.log.states
            self.h5["block_vel"][j] = b.log.block_vel
            self.h5["block_ang_vel"][j] = b.log.block_ang_vel
            self.h5["n_contacts"][j] = b.log.n_contacts
            self.h5["root_index"][j] = self.root_index[b.root_id]
            self.h5["kind"][j] = b.kind
            self.h5["params"][j] = json.dumps(b.params)
            if self.with_frames:
                self.h5["frames"][j] = np.stack(b.log.frames)
        self.h5.flush()

    def finish(self, ledger: StepLedger, manifest: dict | None = None) -> None:
        blob = {"roots": self.roots, "ledger": ledger.to_dict(), "manifest": manifest or {},
                "n_branches": int(self.h5["tape"].shape[0]), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (self.dir / "roots.json").write_text(json.dumps(blob) + "\n")
        self.h5.close()


class Bank:
    """Read-only view of a bank."""

    def __init__(self, bank_dir: Path):
        self.dir = Path(bank_dir)
        blob = json.loads((self.dir / "roots.json").read_text())
        self.roots = [Root.from_dict(r) for r in blob["roots"]]
        self.ledger = blob["ledger"]
        self.manifest = blob.get("manifest", {})
        self.h5 = h5py.File(self.dir / "branches.h5", "r")

    def __len__(self) -> int:
        return int(self.h5["tape"].shape[0])

    def branch(self, j: int, *, frames: bool = False) -> dict:
        d = {k: self.h5[k][j] for k in ("tape", "states", "block_vel", "block_ang_vel", "n_contacts", "root_index")}
        d["kind"] = self.h5["kind"][j].decode() if isinstance(self.h5["kind"][j], bytes) else str(self.h5["kind"][j])
        d["params"] = json.loads(self.h5["params"][j])
        if frames and "frames" in self.h5:
            d["frames"] = self.h5["frames"][j]
        return d

    def root_of(self, j: int) -> Root:
        return self.roots[int(self.h5["root_index"][j])]

    def indices_for_root(self, i: int) -> np.ndarray:
        return np.where(self.h5["root_index"][:] == i)[0]
