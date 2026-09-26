"""Walker2d rules, dense truth and exact snapshot branching (walker2d.md).

Rules, evaluated at every environment step (0.008 s) and reported separately:
1. Speed (the benchmark's own cost): unsafe if forward velocity > 2.3415 m/s.
   Clearance = 2.3415 - v, in m/s.
2. Health (a supplied rule, NOT a claim of irreversibility): unsafe if torso height leaves
   (0.8, 2.0) or pitch leaves (-1, 1) rad. Clearance = min over the four margins, each
   divided by a declared scale (height 0.2 m, pitch 0.5 rad), dimensionless. Name it
   "excessive leaning / falling" in reports: in Phase 0 it fired while still upright.

Branching restores `(qpos, qvel)` exactly and zeroes `qacc_warmstart`, `ctrl` and applied
forces, exactly as Phase 0's oracle does (`oracle.py::_reset_to`, copied with provenance),
so a branch needs no replayed prefix: the charged cost is the branch horizon plus root
generation. Termination is OFF for branch truth so the trajectory past a health violation
is recorded and labelled, not censored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEED_LIMIT = 2.3415
HEALTHY_Z = (0.8, 2.0)
HEALTHY_ANGLE = (-1.0, 1.0)
HEIGHT_SCALE = 0.2
PITCH_SCALE = 0.5
FRAMESKIP = 10  # env steps per model step (0.08 s)
HISTORY = 3
HORIZON_BLOCKS = 10  # 0.8 s
DT = 0.008


def speed_clearance(x_velocity) -> np.ndarray:
    return SPEED_LIMIT - np.asarray(x_velocity, dtype=np.float64)


def health_clearance(height, pitch) -> np.ndarray:
    z = np.asarray(height, dtype=np.float64)
    a = np.asarray(pitch, dtype=np.float64)
    return np.minimum.reduce([(z - HEALTHY_Z[0]) / HEIGHT_SCALE, (HEALTHY_Z[1] - z) / HEIGHT_SCALE, (a - HEALTHY_ANGLE[0]) / PITCH_SCALE, (HEALTHY_ANGLE[1] - a) / PITCH_SCALE])


def clearances_from_log(log: "BranchLog") -> dict:
    return {"speed": speed_clearance(log.x_velocity), "health": health_clearance(log.qpos[1:, 1], log.qpos[1:, 2])}


@dataclass
class BranchLog:
    """Row t of qpos/qvel is the state after t env steps (row 0 = root); x_velocity[t] is the
    velocity produced by env step t+1, so speed clearance has one entry per step."""

    qpos: np.ndarray  # (T+1, nq)
    qvel: np.ndarray  # (T+1, nv)
    x_velocity: np.ndarray  # (T,)
    actions: np.ndarray  # (T, 6)
    frames: list | None = None  # block-endpoint frames incl. row 0

    @property
    def height(self) -> np.ndarray:
        return self.qpos[:, 1]

    @property
    def pitch(self) -> np.ndarray:
        return self.qpos[:, 2]

    def unsafe(self) -> dict:
        c = clearances_from_log(self)
        return {k: bool((v < 0).any()) for k, v in c.items()}

    def min_clearance(self) -> dict:
        c = clearances_from_log(self)
        return {k: float(v.min()) for k, v in c.items()}


def restore(env, qpos, qvel) -> None:
    """Exact restore (copied from experiments/helpers/oracle.py::_reset_to, Phase 0)."""
    import mujoco

    u = env.unwrapped
    u.data.qacc_warmstart[:] = 0
    u.data.ctrl[:] = 0
    u.data.qfrc_applied[:] = 0
    u.data.xfrc_applied[:] = 0
    u.set_state(np.asarray(qpos, float), np.asarray(qvel, float))
    mujoco.mj_forward(u.model, u.data)


def execute_branch(env, qpos, qvel, actions: np.ndarray, *, render_every: int | None = None) -> BranchLog:
    """Run `actions` (T, 6) from the exact state; termination is never applied here."""
    u = env.unwrapped
    restore(env, qpos, qvel)
    actions = np.asarray(actions, dtype=np.float64)
    T = len(actions)
    qp = np.empty((T + 1, u.model.nq))
    qv = np.empty((T + 1, u.model.nv))
    xv = np.empty(T)
    frames = [] if render_every else None
    qp[0], qv[0] = u.data.qpos.copy(), u.data.qvel.copy()
    if render_every:
        frames.append(env.render())
    for t, a in enumerate(actions):
        x0 = u.data.qpos[0]
        u.do_simulation(np.clip(a, -1.0, 1.0), u.frame_skip)
        qp[t + 1], qv[t + 1] = u.data.qpos.copy(), u.data.qvel.copy()
        xv[t] = (u.data.qpos[0] - x0) / u.dt
        if render_every and (t + 1) % render_every == 0:
            frames.append(env.render())
    return BranchLog(qp, qv, xv, actions, frames)


def replay_check(env, qpos, qvel, actions, repeats: int = 3) -> dict:
    logs = [execute_branch(env, qpos, qvel, actions) for _ in range(repeats)]
    ref = logs[0]
    diffs = [max(float(np.abs(lg.qpos - ref.qpos).max()), float(np.abs(lg.qvel - ref.qvel).max())) for lg in logs[1:]]
    return {"bitwise": all(np.array_equal(lg.qpos, ref.qpos) and np.array_equal(lg.qvel, ref.qvel) for lg in logs[1:]), "max_abs_diff": max(diffs) if diffs else 0.0}


# --------------------------------------------------------------------------------------
# roots from recorded episodes, and proposals
# --------------------------------------------------------------------------------------
@dataclass
class WalkerRoot:
    root_id: str
    episode: int
    step: int  # index t of the root row in the episode
    qpos: np.ndarray
    qvel: np.ndarray
    history_qpos: np.ndarray  # (HISTORY, nq) states at t-20, t-10, t (frameskip 10)
    history_qvel: np.ndarray
    history_actions: np.ndarray  # (HISTORY-1, FRAMESKIP, 6) actions between the history frames
    policy_tape: np.ndarray  # (HORIZON_BLOCKS*FRAMESKIP, 6) the policy's own subsequent actions, if available
    meta: dict

    def to_dict(self) -> dict:
        return {"root_id": self.root_id, "episode": int(self.episode), "step": int(self.step), "qpos": self.qpos.tolist(), "qvel": self.qvel.tolist(),
                "history_qpos": self.history_qpos.tolist(), "history_qvel": self.history_qvel.tolist(), "history_actions": self.history_actions.tolist(),
                "policy_tape": self.policy_tape.tolist(), "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "WalkerRoot":
        return cls(d["root_id"], d["episode"], d["step"], np.asarray(d["qpos"]), np.asarray(d["qvel"]), np.asarray(d["history_qpos"]), np.asarray(d["history_qvel"]),
                   np.asarray(d["history_actions"]), np.asarray(d["policy_tape"]), d.get("meta", {}))


def roots_from_episode(ep: dict, steps, *, episode: int, prefix: str = "w", frameskip=FRAMESKIP, history=HISTORY, horizon=HORIZON_BLOCKS) -> list[WalkerRoot]:
    """Roots at the given step indices of a recorded episode dict (qpos, qvel, action, x_velocity, healthy)."""
    out = []
    n = len(ep["qpos"])
    for t in steps:
        t = int(t)
        lo = t - (history - 1) * frameskip
        if lo < 0 or t + 1 > n:
            continue
        hi = min(n, t + horizon * frameskip)
        tape = np.asarray(ep["action"][t:hi], float)
        if len(tape) < horizon * frameskip:
            tape = np.concatenate([tape, np.repeat(tape[-1:], horizon * frameskip - len(tape), 0)], 0) if len(tape) else np.zeros((horizon * frameskip, 6))
        idx = np.arange(lo, t + 1, frameskip)
        out.append(WalkerRoot(f"{prefix}-e{episode}-t{t}", episode, t, np.asarray(ep["qpos"][t]), np.asarray(ep["qvel"][t]), np.asarray(ep["qpos"][idx]), np.asarray(ep["qvel"][idx]),
                              np.asarray(ep["action"][lo:t], float).reshape(history - 1, frameskip, -1), tape,
                              {"x_velocity": float(ep["x_velocity"][t - 1]) if t > 0 else 0.0, "height": float(ep["qpos"][t][1]), "pitch": float(ep["qpos"][t][2])}))
    return out


def propose_tapes(rng, root: WalkerRoot, *, n_random: int, sigmas=(0.1, 0.2, 0.4), n_bursts: int = 0, burst_scale=(0.6, 1.0)) -> list[tuple[np.ndarray, str, dict]]:
    """Common proposal generator: the policy's own tape, Gaussian perturbations, torque bursts."""
    base = np.clip(root.policy_tape, -1, 1)
    out = [(base.copy(), "policy", {})]
    per = max(1, n_random // len(sigmas))
    for s in sigmas:
        for _ in range(per):
            out.append((np.clip(base + rng.normal(0, s, base.shape), -1, 1), "random", {"sigma": float(s)}))
    for _ in range(n_bursts):
        t0 = int(rng.integers(0, len(base) - FRAMESKIP))
        dur = int(rng.integers(FRAMESKIP // 2, 2 * FRAMESKIP))
        amp = float(rng.uniform(*burst_scale)) * rng.choice([-1.0, 1.0], size=base.shape[1])
        tape = base.copy()
        tape[t0 : t0 + dur] = np.clip(tape[t0 : t0 + dur] + amp, -1, 1)
        out.append((tape, "burst", {"t0": t0, "dur": dur}))
    return out
