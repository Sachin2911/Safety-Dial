"""Push-T roots, prefix replay, tapes and dense per-step truth.

Definitions (protocol.md): a ROOT is a seeded reset plus a recorded action prefix; a TAPE
is an open-loop action sequence grouped into blocks of `ACTION_BLOCK` env actions; a
BRANCH is one tape executed from a root. `_set_state` cannot restore a root (block
velocity is left dirty; pushT.md), so every branch is reproduced from `reset` + prefix.
Replaying costs `len(prefix)` charged env steps.

Every root carries at least `HISTORY_FRAMES - 1` prefix blocks so the model always
conditions on `HISTORY_FRAMES` real block-endpoint frames. A root with `k = 0` nominal
blocks therefore replays 2 idle blocks (zero action: the PD controller holds the pusher
in place). Those idle steps are charged like any other.

    env = make_env()
    root = Root(seed=1, start_state=s, goal_state=g, prefix=np.zeros((10, 2)))
    ctx = reset_root(env, root)                 # replays prefix; ctx.frames are history
    log = execute_tape(env, tape)               # dense truth for the branch
    rep = replay_check(env, root, tape, 3)      # bitwise determinism test
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from helpers.pushtAssets import ACTION_BLOCK, HISTORY_FRAMES, HORIZON_BLOCKS, PHYSICS_DT, SUBSTEPS_PER_STEP

STATE_DIM = 7
ACTION_DIM = 2
ENV_ID = "swm/PushT-v1"


def make_env():
    """Raw Push-T env (no World wrappers). Its `render()` equals World pixels bitwise."""
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  registers swm/PushT-v1

    return gym.make(ENV_ID, render_mode="rgb_array")


# --------------------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------------------
@dataclass
class Root:
    seed: int
    start_state: np.ndarray  # (7,)
    goal_state: np.ndarray  # (7,)
    prefix: np.ndarray  # (k*ACTION_BLOCK, 2) raw env actions, may be empty
    root_id: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_prefix_blocks(self) -> int:
        return len(self.prefix) // ACTION_BLOCK

    def to_dict(self) -> dict:
        return {
            "root_id": self.root_id,
            "seed": int(self.seed),
            "start_state": np.asarray(self.start_state, dtype=float).tolist(),
            "goal_state": np.asarray(self.goal_state, dtype=float).tolist(),
            "prefix": np.asarray(self.prefix, dtype=float).reshape(-1, ACTION_DIM).tolist(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Root":
        return cls(
            seed=int(d["seed"]),
            start_state=np.asarray(d["start_state"], dtype=float),
            goal_state=np.asarray(d["goal_state"], dtype=float),
            prefix=np.asarray(d["prefix"], dtype=float).reshape(-1, ACTION_DIM),
            root_id=d.get("root_id", ""),
            meta=d.get("meta", {}),
        )


@dataclass
class DenseLog:
    """Per env step truth. Row t is the state AFTER action t (row 0 = state before any)."""

    states: np.ndarray  # (T+1, 7)
    block_vel: np.ndarray  # (T+1, 2)
    block_ang_vel: np.ndarray  # (T+1,)
    n_contacts: np.ndarray  # (T+1,) contacts during the step that produced row t (0 for row 0)
    actions: np.ndarray  # (T, 2)
    frames: list | None = None  # block-endpoint frames (index 0 = before any action)

    @property
    def poses(self) -> np.ndarray:
        return self.states[:, 2:5]

    def endpoint_states(self) -> np.ndarray:
        """States at block endpoints, including row 0."""
        return self.states[:: ACTION_BLOCK]


@dataclass
class RootContext:
    frames: list  # last HISTORY_FRAMES block-endpoint frames (uint8 224x224x3)
    history_actions: np.ndarray  # (HISTORY_FRAMES-1, ACTION_BLOCK, 2) actions between them
    state: np.ndarray  # current 7-d state
    goal_frame: np.ndarray
    prefix_log: DenseLog
    charged_steps: int


# --------------------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------------------
class StepLedger:
    """Charged simulator steps by category (protocol.md, budget accounting)."""

    def __init__(self):
        self.counts: dict[str, int] = {}
        self.branches: dict[str, int] = {}

    def add(self, category: str, steps: int, branches: int = 0) -> None:
        self.counts[category] = self.counts.get(category, 0) + int(steps)
        if branches:
            self.branches[category] = self.branches.get(category, 0) + int(branches)

    @property
    def total(self) -> int:
        return int(sum(self.counts.values()))

    def to_dict(self) -> dict:
        return {"steps": dict(self.counts), "branches": dict(self.branches), "total_steps": self.total}


# --------------------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------------------
def _read(env) -> tuple[np.ndarray, np.ndarray, float]:
    u = env.unwrapped
    state = np.asarray(u._get_obs(), dtype=np.float64)
    return state, np.array(tuple(u.block.velocity), dtype=np.float64), float(u.block.angular_velocity)


def run_actions(env, actions: np.ndarray, *, record_frames: bool = False, frame_every: int = ACTION_BLOCK) -> DenseLog:
    """Execute raw env actions from the CURRENT env state, logging every env step."""
    actions = np.asarray(actions, dtype=np.float64).reshape(-1, ACTION_DIM)
    T = len(actions)
    states = np.empty((T + 1, STATE_DIM))
    bvel = np.empty((T + 1, 2))
    bang = np.empty(T + 1)
    ncon = np.zeros(T + 1, dtype=np.int64)
    frames = [] if record_frames else None
    states[0], bvel[0], bang[0] = _read(env)
    if record_frames:
        frames.append(env.render())
    for t, a in enumerate(actions):
        _, _, _, _, info = env.step(a)
        states[t + 1], bvel[t + 1], bang[t + 1] = _read(env)
        ncon[t + 1] = int(info.get("n_contacts", 0))
        if record_frames and (t + 1) % frame_every == 0:
            frames.append(env.render())
    return DenseLog(states, bvel, bang, ncon, actions, frames)


def reset_root(env, root: Root, *, record_frames: bool = True) -> RootContext:
    """Seeded reset + prefix replay. Returns the model's history and the charged cost."""
    env.reset(seed=int(root.seed), options={"state": np.asarray(root.start_state, dtype=np.float64),
                                            "goal_state": np.asarray(root.goal_state, dtype=np.float64)})
    goal_frame = np.asarray(env.unwrapped._goal)
    prefix = np.asarray(root.prefix, dtype=np.float64).reshape(-1, ACTION_DIM)
    if len(prefix) % ACTION_BLOCK:
        raise ValueError(f"prefix length {len(prefix)} is not a multiple of {ACTION_BLOCK}")
    if len(prefix) // ACTION_BLOCK < HISTORY_FRAMES - 1:
        raise ValueError(f"root needs >= {HISTORY_FRAMES - 1} prefix blocks (idle blocks count)")
    log = run_actions(env, prefix, record_frames=record_frames)
    frames = log.frames[-HISTORY_FRAMES:] if record_frames else []
    blocks = prefix.reshape(-1, ACTION_BLOCK, ACTION_DIM)
    hist_actions = blocks[-(HISTORY_FRAMES - 1):] if HISTORY_FRAMES > 1 else blocks[:0]
    return RootContext(frames, hist_actions, log.states[-1].copy(), goal_frame, log, len(prefix))


def execute_tape(env, tape: np.ndarray, *, record_frames: bool = True) -> DenseLog:
    """Execute a tape (n_blocks*ACTION_BLOCK, 2) from the current state; dense truth."""
    return run_actions(env, tape, record_frames=record_frames)


def execute_branch(env, root: Root, tape: np.ndarray, *, record_frames: bool = True) -> tuple[RootContext, DenseLog]:
    ctx = reset_root(env, root, record_frames=record_frames)
    return ctx, execute_tape(env, tape, record_frames=record_frames)


def idle_prefix(n_blocks: int = HISTORY_FRAMES - 1) -> np.ndarray:
    return np.zeros((n_blocks * ACTION_BLOCK, ACTION_DIM))


def tape_from_blocks(blocks: np.ndarray) -> np.ndarray:
    return np.asarray(blocks, dtype=np.float64).reshape(-1, ACTION_DIM)


def blocks_from_tape(tape: np.ndarray) -> np.ndarray:
    return np.asarray(tape, dtype=np.float64).reshape(-1, ACTION_BLOCK, ACTION_DIM)


# --------------------------------------------------------------------------------------
# determinism and timing diagnostics
# --------------------------------------------------------------------------------------
def replay_check(env, root: Root, tape: np.ndarray, repeats: int = 3, *, frames: bool = True) -> dict:
    """Execute root+tape `repeats` times from reset; compare every logged quantity."""
    logs = [execute_branch(env, root, tape, record_frames=frames)[1] for _ in range(repeats)]
    ref = logs[0]
    out = {"repeats": repeats, "n_steps": int(len(tape)), "bitwise": True, "max_abs_state": 0.0,
           "max_abs_block_vel": 0.0, "max_abs_ang_vel": 0.0, "frames_equal": True,
           "contact_steps": int((ref.n_contacts > 0).sum())}
    for lg in logs[1:]:
        out["max_abs_state"] = max(out["max_abs_state"], float(np.abs(lg.states - ref.states).max()))
        out["max_abs_block_vel"] = max(out["max_abs_block_vel"], float(np.abs(lg.block_vel - ref.block_vel).max()))
        out["max_abs_ang_vel"] = max(out["max_abs_ang_vel"], float(np.abs(lg.block_ang_vel - ref.block_ang_vel).max()))
        if not (np.array_equal(lg.states, ref.states) and np.array_equal(lg.block_vel, ref.block_vel)
                and np.array_equal(lg.block_ang_vel, ref.block_ang_vel)):
            out["bitwise"] = False
        if frames and not all(np.array_equal(a, b) for a, b in zip(lg.frames, ref.frames)):
            out["frames_equal"] = False
    return out


def step_substeps(env, action) -> np.ndarray:
    """Mirror of `PushT.step` that records the state after every physics substep.

    Diagnostic only. `check_substep_equivalence` verifies that this loop reproduces
    `env.step` exactly, so substep traces around contact can be trusted.
    """
    from pymunk.vec2d import Vec2d

    u = env.unwrapped
    a = np.asarray(action, dtype=np.float64)
    target = u.agent.position + a * u.action_scale if u.relative else Vec2d(*a)
    out = np.empty((SUBSTEPS_PER_STEP, STATE_DIM))
    for i in range(SUBSTEPS_PER_STEP):
        acc = u.k_p * (target - u.agent.position) + u.k_v * (Vec2d(0, 0) - u.agent.velocity)
        u.agent.velocity += acc * u.dt
        u.space.step(u.dt)
        out[i] = u._get_obs()
    return out


def check_substep_equivalence(env, root: Root, tape: np.ndarray) -> dict:
    """Run the tape once with env.step and once with step_substeps; compare endpoints."""
    ctx, log = execute_branch(env, root, tape, record_frames=False)
    reset_root(env, root, record_frames=False)
    sub = np.stack([step_substeps(env, a) for a in np.asarray(tape).reshape(-1, ACTION_DIM)])
    diff = float(np.abs(sub[:, -1] - log.states[1:]).max())
    assert abs(PHYSICS_DT * SUBSTEPS_PER_STEP - 0.1) < 1e-12
    return {"max_abs_diff": diff, "bitwise": bool(np.array_equal(sub[:, -1], log.states[1:])), "substeps": sub}


def endpoint_interpolation(states: np.ndarray, block: int = ACTION_BLOCK) -> np.ndarray:
    """Linear interpolation of block-endpoint states back to every env step.

    Angle is interpolated on the circle. Used for the temporal-sampling source (2) of the
    four-source decomposition.
    """
    T = len(states) - 1
    ends = states[::block]
    n_blocks = (len(ends) - 1)
    out = np.empty_like(states)
    for b in range(n_blocks):
        s0, s1 = ends[b], ends[b + 1]
        for j in range(block):
            w = j / block
            row = s0 * (1 - w) + s1 * w
            d = np.arctan2(np.sin(s1[4] - s0[4]), np.cos(s1[4] - s0[4]))
            row[4] = (s0[4] + w * d) % (2 * np.pi)
            out[b * block + j] = row
    out[T] = states[T]
    return out


HORIZON_STEPS = HORIZON_BLOCKS * ACTION_BLOCK
