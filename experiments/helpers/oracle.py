"""Simulator-verified recoverability oracle.

Why this exists
---------------
The obvious ground truth for "this state is irreversible" is the environment's `terminated`
flag. That will not do. `terminated` is a threshold on torso height and pitch angle, both of
which decode from a raw 32x32 grayscale frame at R^2 = 0.99. Scoring a detector against it
measures whether the detector can see the torso, not whether it can see irreversibility, and any
result would be circular: the project's whole claim is that a failure set should not come from a
hand-set constant.

So recoverability is measured directly. From a saved MuJoCo state, with termination disabled,
search for an action sequence that returns the robot to a settled upright pose. If none is
found, the state is irrecoverable. This is a one-sided test: any success proves recoverability,
so more search can only shrink the irrecoverable set, never grow it. That monotonicity is what
makes the tiered search below safe.

What is actually being estimated is the H-step viability kernel of the health set under bounded
controls, which is the complement of the Hamilton-Jacobi backward reachable tube this project is
porting into a latent space. "Can it get back on its feet" is the same object, stated plainly.

The recovery predicate sits strictly INSIDE the environment's own health set (z >= 1.0 against a
healthy floor of 0.8; |pitch| <= 0.5 against a healthy bound of 1.0). That is deliberate and it
is the point: a state the oracle calls recoverable must reach a pose that is comfortably healthy,
not one that merely grazes the constant. It is what stops the oracle being a restatement of
`terminated`.

EVALUATION ONLY
---------------
Oracle labels are ground truth for scoring. They must never reach an estimator, and no method
hyperparameter may be selected on oracle AUROC. The structural guarantee is that the oracle needs
capabilities the method does not have: `set_state` on the true simulator, and rollouts of true
dynamics. The method sees pixels and rolls forward only through a learned predictor.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RecoveryPredicate:
    """What counts as recovered, and for how long.

    `dwell` is what stops a robot that tumbles through an upright pose from counting as
    recovered. `max_qvel` is what makes "upright" mean "upright and settled" rather than
    "upright and about to fall over again". The z bound is two-sided: without an upper limit a
    ballistic arc satisfies the predicate, since 25 steps is only 0.2 s of flight and CEM is
    perfectly capable of launching the robot rather than standing it up.

    The velocity cap applies to the ROOT degrees of freedom only (`qvel[0:3]`: torso x velocity,
    torso z velocity, torso pitch rate), not to the joint velocities. This matters more than it
    looks. A Walker2d in normal gait has joint velocities that routinely exceed 10 rad/s, and
    measured on a benign upright pool, only 91% of states satisfy an all-DOF cap of 5.0 against
    98.9% for a root-only cap. Requiring every joint to come to rest would turn "recovered" into
    "recovered and then stood perfectly still", which is a far harder control problem than the
    one being asked about, and it would misclassify roughly one benign walking state in twelve as
    irrecoverable. The physical notion wanted here is that the torso is upright and is neither
    falling nor tumbling; fast-moving legs are what walking looks like.
    """

    min_z: float
    max_z: float
    max_pitch: float
    max_qvel: float = 5.0
    dwell: int = 25
    qvel_slice: tuple[int, int] = (0, 3)  # root dofs: x velocity, z velocity, pitch rate

    def margin(self, qpos: np.ndarray, qvel: np.ndarray) -> float:
        """Signed slack, positive when satisfied. Used as the CEM objective.

        Written with Python scalars rather than numpy vector ops. This is called once per
        simulator step, and a full oracle labelling run is on the order of 10^9 steps, so the
        temporary arrays that `np.abs(qvel[lo:hi]).max()` allocates dominate the inner loop.
        The arithmetic is identical.
        """
        lo, hi = self.qvel_slice
        v = 0.0
        for k in range(lo, hi):
            a = qvel[k]
            if a < 0.0:
                a = -a
            if a > v:
                v = a
        pitch = qpos[2]
        if pitch < 0.0:
            pitch = -pitch
        z = qpos[1]
        m = z - self.min_z
        # Upper bound too. Without it the predicate is one-sided in z and a ballistic arc
        # counts as recovery: CEM can launch the robot, and 25 steps is only 0.2 s of flight.
        # It also makes the docstring's claim true, that every constant sits strictly inside
        # the environment's own health set (0.8 < z < 2.0 for Walker2d).
        m_hi = self.max_z - z
        if m_hi < m:
            m = m_hi
        m2 = self.max_pitch - pitch
        if m2 < m:
            m = m2
        m3 = (self.max_qvel - v) / self.max_qvel
        return m3 if m3 < m else m


#: Per-robot predicates. Every constant is strictly inside that robot's gymnasium health set.
#: Walker2d healthy: z in (0.8, 2.0), |pitch| < 1.0.  Hopper healthy: z > 0.7, |pitch| < 0.2.
PREDICATES = {
    # Walker2d healthy: 0.8 < z < 2.0, |pitch| < 1.0.  Hopper healthy: z > 0.7, |pitch| < 0.2.
    # Every constant below sits strictly inside its robot's health set.
    "Walker2d": RecoveryPredicate(min_z=1.00, max_z=1.80, max_pitch=0.50),
    "Hopper": RecoveryPredicate(min_z=0.95, max_z=1.60, max_pitch=0.10),
}

#: A state is hopeless when the robot is lying still: low and barely moving. Cuts the cost of
#: genuinely fallen states by roughly 3x.
#:
#: This is a search truncation, so strictly it is the one place where the oracle's one-sidedness
#: is an assumption rather than a guarantee: a rollout cut short here might, in principle, have
#: recovered later, which would make the label falsely irrecoverable. The thresholds are set
#: deliberately far inside the fallen regime (torso below 0.35 m, so on the ground, with root
#: velocities under 1.0 for a fifth of a second) precisely so that assumption is safe. Report it
#: as a stated approximation, and note that the V4 budget-saturation check would detect it: if
#: the exit were mislabelling states, raising the budget would flip labels and V4 would fail.
HOPELESS_Z = 0.35
HOPELESS_QVEL = 1.0
HOPELESS_DWELL = 25


@dataclass
class OracleConfig:
    robot: str = "Walker2d"
    version: str = "v1"
    horizon: int = 250  # control steps; 2.0 s at dt = 0.008
    n_policy: int = 8  # Tpi: trained-controller rollouts, when a policy is supplied
    policy_sigmas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
    n_constant: int = 13  # T1: zero torque plus +/-1 on each actuator
    n_random: int = 256  # T2, split across the betas below
    betas: tuple[float, ...] = (0.0, 0.5, 0.7, 0.9)
    use_cem: bool = True  # T3
    cem_knots: int = 10
    cem_pop: int = 64
    cem_elites: int = 8
    cem_iters: int = 8
    cem_restarts: int = 2
    cem_sigma0: float = 0.5
    cem_sigma_decay: float = 0.9
    budget_scale: float = 1.0  # for the V4 saturation check
    predicate: RecoveryPredicate | None = None
    extra: dict = field(default_factory=dict)

    def resolved_predicate(self) -> RecoveryPredicate:
        return self.predicate or PREDICATES[self.robot]


def state_seed(episode_idx: int, step_idx: int, salt: str = "safetydial") -> int:
    """Deterministic per-state RNG seed, so labels are exactly reproducible.

    Python's `hash()` is salted per process and would silently produce different labels on
    different runs, so this uses blake2b instead.
    """
    key = f"{salt}:{int(episode_idx)}:{int(step_idx)}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % (2**32)


class RecoverySimulator:
    """Owns one termination-disabled MuJoCo env and replays candidate action sequences.

    Built lazily so the object pickles cleanly into multiprocessing workers.
    """

    def __init__(self, cfg: OracleConfig):
        self.cfg = cfg
        self.pred = cfg.resolved_predicate()
        self._env = None

    @property
    def env(self):
        if self._env is None:
            from helpers.locoEnv import make_loco_env

            self._env = make_loco_env(
                self.cfg.robot,
                self.cfg.version,
                terminate_when_unhealthy=False,
                time_limit=False,
                six_tuple=True,
            )
            self._env.reset(seed=0)
        return self._env

    @property
    def action_dim(self) -> int:
        return int(self.env.action_space.shape[0])

    @staticmethod
    def _reset_to(u, qpos, qvel) -> None:
        """Restore state so a rollout depends only on `(qpos, qvel, actions)`.

        `MujocoEnv.set_state` writes qpos/qvel and calls `mj_forward`, but leaves
        `data.qacc_warmstart` and `data.ctrl` holding values from whatever ran previously.
        `qacc_warmstart` seeds the constraint solver, so the same rollout replayed after
        different predecessors produces answers that differ in the last couple of digits
        (measured: margin -1.4618128503645678 vs -1.4618128503645664).

        That is harmless for a state well inside or well outside the recovery set, and indeed
        full labels agreed across simulators. But the oracle's labels are the paper's ground
        truth and are claimed to be exactly reproducible, and for a state sitting exactly on the
        boundary a 1e-14 difference can flip `margin > 0` and therefore the label. Zeroing the
        warmstart and the controls costs one extra `mj_forward` per rollout, which is nothing
        against 250 simulation steps.
        """
        import mujoco

        u.data.qacc_warmstart[:] = 0
        u.data.ctrl[:] = 0
        u.data.qfrc_applied[:] = 0
        u.data.xfrc_applied[:] = 0
        u.set_state(np.asarray(qpos, float), np.asarray(qvel, float))
        mujoco.mj_forward(u.model, u.data)

    def rollout(self, qpos, qvel, actions) -> tuple[bool, float]:
        """Replay `actions` from `(qpos, qvel)`.

        Returns `(recovered, best_windowed_margin)`. The margin is the best value over all
        `dwell`-length windows of the per-step margin, i.e. exactly the quantity whose crossing
        zero means the predicate held for `dwell` consecutive steps. That makes it a smooth
        surrogate for the hard predicate, which is what CEM needs to climb.
        """
        u = self.env.unwrapped
        self._reset_to(u, qpos, qvel)

        pred = self.pred
        margins = np.empty(len(actions), dtype=np.float64)
        run = 0
        hopeless_run = 0

        for i, a in enumerate(actions):
            u.do_simulation(a, u.frame_skip)
            qp, qv = u.data.qpos, u.data.qvel
            m = pred.margin(qp, qv)
            margins[i] = m

            if m > 0:
                run += 1
                if run >= pred.dwell:
                    return True, float(m)
            else:
                run = 0

            if qp[1] < HOPELESS_Z and max(abs(qv[0]), abs(qv[1]), abs(qv[2])) < HOPELESS_QVEL:
                hopeless_run += 1
                if hopeless_run >= HOPELESS_DWELL:
                    margins = margins[: i + 1]
                    break
            else:
                hopeless_run = 0

        if len(margins) < pred.dwell:
            # No dwell-length window exists, so the "best windowed margin" is undefined. Return
            # the minimum rather than the maximum: the maximum is a different and strictly
            # optimistic quantity, and can be positive for a rollout that never came close to
            # satisfying the predicate, which would mislead CEM into climbing toward it.
            return False, float(margins.min()) if len(margins) else -np.inf
        # Sliding minimum over dwell-length windows, then the best such window.
        windows = np.lib.stride_tricks.sliding_window_view(margins, pred.dwell)
        return False, float(windows.min(axis=1).max())


def _constant_sequences(action_dim: int, horizon: int, n: int) -> list[np.ndarray]:
    """T1: zero torque, then +/-1 on each actuator. Resolves most benign states immediately."""
    seqs = [np.zeros((horizon, action_dim), dtype=np.float64)]
    for j in range(action_dim):
        for sign in (1.0, -1.0):
            a = np.zeros(action_dim)
            a[j] = sign
            seqs.append(np.tile(a, (horizon, 1)))
    return seqs[:n]


def _random_sequences(rng, action_dim, horizon, n, betas) -> list[np.ndarray]:
    """T2: low-pass filtered random torque.

    White noise at 125 Hz is a hopeless controller: it averages to nothing and the robot just
    falls. Filtering is what turns it into a sequence of coherent pushes. Several bandwidths are
    spanned so the search does not commit to one timescale.
    """
    seqs = []
    # Distribute the remainder so exactly `n` sequences come back. Integer division alone
    # silently returned fewer, which quietly shrank the search budget the labels were produced
    # under and would have made the V4 saturation check compare unequal budgets.
    counts = [n // len(betas)] * len(betas)
    for i in range(n % len(betas)):
        counts[i] += 1

    for beta, count in zip(betas, counts):
        # An exponential filter `a_t = beta a_{t-1} + (1-beta) u_t` on zero-mean noise has
        # stationary std reduced by sqrt((1-beta)/(1+beta)): at beta=0.9 that is 0.23, so the
        # smooth bandwidths would command less than a quarter of the available torque and the
        # tier would be testing "gentle nudges" rather than "coherent pushes". Rescaling makes
        # every bandwidth exercise comparable control authority.
        gain = np.sqrt((1.0 + beta) / (1.0 - beta)) if beta < 1.0 else 1.0
        for _ in range(count):
            u = rng.uniform(-1.0, 1.0, size=(horizon, action_dim))
            a = np.empty_like(u)
            prev = np.zeros(action_dim)
            for t in range(horizon):
                prev = beta * prev + (1.0 - beta) * u[t]
                a[t] = prev
            seqs.append(np.clip(a * gain, -1.0, 1.0))
    return seqs[:n]


def _knots_to_sequence(knots: np.ndarray, horizon: int) -> np.ndarray:
    """Linearly interpolate `(K, A)` control knots up to `(horizon, A)`.

    This is the single most important implementation detail in the oracle. Searching the raw
    `horizon x action_dim` space is 1500-dimensional for Walker2d at H = 250, and CEM simply does
    not work there. Ten knots makes it 60-dimensional and tractable, at no real cost: recovery
    manoeuvres are smooth, so a piecewise-linear torque profile is an ample parameterisation.
    """
    k = len(knots)
    src = np.linspace(0.0, 1.0, k)
    dst = np.linspace(0.0, 1.0, horizon)
    return np.clip(
        np.stack([np.interp(dst, src, knots[:, j]) for j in range(knots.shape[1])], axis=1),
        -1.0,
        1.0,
    )


def _cem_search(
    sim: RecoverySimulator, qpos, qvel, rng, cfg: OracleConfig
) -> tuple[bool, float, int]:
    """T3: cross-entropy method over control knots, on the true simulator.

    This tier is what makes the result an oracle rather than a random-search baseline. A pilot
    put T1+T2 at only 92% on obviously-benign states, so without CEM the oracle would mislabel
    roughly one benign state in twelve as irrecoverable.
    """
    a_dim, horizon = sim.action_dim, cfg.horizon
    k = cfg.cem_knots
    # Keep the population above the elite count: with a small budget_scale an unclamped pop
    # can fall below cem_elites, at which point "take the top k" selects everything and CEM
    # degenerates into unselected random search while still reporting itself as CEM.
    pop = max(cfg.cem_elites * 2, int(cfg.cem_pop * cfg.budget_scale))
    iters = max(1, int(cfg.cem_iters * cfg.budget_scale))
    best = -np.inf
    n_roll = 0

    for _ in range(cfg.cem_restarts):
        mean = np.zeros((k, a_dim))
        sigma = np.full((k, a_dim), cfg.cem_sigma0)
        for _ in range(iters):
            samples = np.clip(
                rng.normal(mean, sigma, size=(pop, k, a_dim)), -1.0, 1.0
            )
            scores = np.empty(pop)
            for i, s in enumerate(samples):
                ok, score = sim.rollout(qpos, qvel, _knots_to_sequence(s, horizon))
                n_roll += 1
                if ok:
                    return True, score, n_roll
                scores[i] = score
            best = max(best, scores.max())
            elite = samples[np.argsort(-scores)[: cfg.cem_elites]]
            mean = elite.mean(axis=0)
            sigma = np.maximum(elite.std(axis=0), 1e-3) * cfg.cem_sigma_decay
    return False, float(best), n_roll


def _policy_sequences(sim, qpos, qvel, policy, rng, horizon, n, sigma_grid) -> list[np.ndarray]:
    """Closed-loop rollouts of a trained controller, recorded as open-loop tapes.

    A competent walking policy is by far the best recovery attempt available, because staying
    upright is exactly the problem it was trained to solve. Random search has no such skill, so
    without this tier the oracle systematically under-detects recoverability for dynamic states:
    a robot in mid-gait is perfectly fine, but it needs an actual balancing controller to stay
    that way, and zero torque or a constant torque simply drops it.

    Noise is swept so the tier is a small search around the policy rather than a single attempt.
    """
    tapes = []
    for k in range(n):
        sigma = sigma_grid[k % len(sigma_grid)]
        u = sim.env.unwrapped
        sim._reset_to(u, qpos, qvel)
        tape = np.empty((horizon, sim.action_dim))
        for t in range(horizon):
            a = np.asarray(policy.act(u._get_obs()), float)
            if sigma:
                a = np.clip(a + rng.normal(0.0, sigma, sim.action_dim), -1.0, 1.0)
            tape[t] = a
            u.do_simulation(a, u.frame_skip)
        tapes.append(tape)
    return tapes


def label_state(
    sim: RecoverySimulator,
    qpos,
    qvel,
    seed: int,
    cfg: OracleConfig,
    continuation: np.ndarray | None = None,
    policy=None,
) -> dict:
    """Label one state. Short-circuits on the first success.

    Tiers, cheapest first. Because any success proves recoverability, ordering only affects cost,
    never the label, and adding a tier can only shrink the irrecoverable set.

      T0  the actions that actually followed this state in the recorded episode. This is a
          *certificate*, not a search: if the real trajectory reached a settled upright pose,
          the state was recoverable, and one rollout proves it. Free, and it resolves most
          benign states immediately.
      Tpi a trained controller, with a small noise sweep. The tier that makes the question
          meaningful for dynamic states: "could a competent controller have saved this".
      T1  constant torques. T2  low-pass random torques. T3  CEM on the true simulator.

    Returns the label, the tier that resolved it, the best margin found and the rollout count.
    `tier` is what the V1 check reads to show which tiers are load-bearing; the rollout count is
    what the budget-saturation check (V4) sweeps.
    """
    rng = np.random.default_rng(seed)
    a_dim, horizon = sim.action_dim, cfg.horizon
    best = -np.inf
    n_rollouts = 0

    if continuation is not None and len(continuation):
        ok, score = sim.rollout(qpos, qvel, np.asarray(continuation, float)[:horizon])
        n_rollouts += 1
        best = max(best, score)
        if ok:
            return {"recoverable": True, "tier": "T0", "margin": score,
                    "n_rollouts": n_rollouts}

    # Tiers are built lazily. Most states resolve at T0 or T1, and generating T2's 256
    # sequences of length H eagerly costs more than the rollouts that actually run.
    tiers = []
    if policy is not None and cfg.n_policy:
        tiers.append(
            ("Tpi", lambda: _policy_sequences(sim, qpos, qvel, policy, rng, horizon,
                                              cfg.n_policy, cfg.policy_sigmas))
        )
    tiers += [
        ("T1", lambda: _constant_sequences(a_dim, horizon, cfg.n_constant)),
        ("T2", lambda: _random_sequences(rng, a_dim, horizon,
                                         int(cfg.n_random * cfg.budget_scale), cfg.betas)),
    ]
    for tier_name, make_seqs in tiers:
        for seq in make_seqs():
            ok, score = sim.rollout(qpos, qvel, seq)
            n_rollouts += 1
            best = max(best, score)
            if ok:
                return {"recoverable": True, "tier": tier_name, "margin": score,
                        "n_rollouts": n_rollouts}

    if cfg.use_cem:
        # CEM rollouts must be counted: `n_rollouts` is what the V4 budget-saturation check
        # sweeps, and CEM is by far the largest part of the budget for a hard state.
        ok, score, cem_rolls = _cem_search(sim, qpos, qvel, rng, cfg)
        n_rollouts += cem_rolls
        best = max(best, score)
        if ok:
            return {"recoverable": True, "tier": "T3", "margin": score,
                    "n_rollouts": n_rollouts}

    return {"recoverable": False, "tier": "none", "margin": float(best),
            "n_rollouts": n_rollouts}


def label_batch(qpos_arr, qvel_arr, seeds, cfg: OracleConfig, progress=None,
                continuations=None, policy=None) -> dict:
    """Label a batch of states in one process. The multiprocessing entry point."""
    sim = RecoverySimulator(cfg)
    n = len(qpos_arr)
    out = {
        "recoverable": np.zeros(n, dtype=bool),
        "tier": np.zeros(n, dtype="S4"),
        "margin": np.zeros(n, dtype="f8"),
        "n_rollouts": np.zeros(n, dtype="i4"),
    }
    for i in range(n):
        cont = None if continuations is None else continuations[i]
        r = label_state(sim, qpos_arr[i], qvel_arr[i], int(seeds[i]), cfg,
                        continuation=cont, policy=policy)
        out["recoverable"][i] = r["recoverable"]
        out["tier"][i] = r["tier"].encode()
        out["margin"][i] = r["margin"]
        out["n_rollouts"][i] = r["n_rollouts"]
        if progress and (i + 1) % progress == 0:
            print(f"  [oracle] {i + 1}/{n} recoverable so far "
                  f"{out['recoverable'][: i + 1].mean():.3f}", flush=True)
    return out
