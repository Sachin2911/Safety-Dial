"""Closed-loop real executor: ground truth for policies evolved in the Walker LeWM (docs/evoPlan).

Every selected policy is replayed from the same saved MuJoCo snapshots it was scored on in
imagination. A task (candidate p, root r) restores the root exactly (`walkerRules.restore`); then,
for each of the 10 blocks of a segment, the policy acts on the latest two real latents (the root's
history latents first), the action is held for the block's 10 environment steps, and the block-end
state is rendered (`RenderContext.render_state`, a pure function of qpos and qvel) and encoded
(`WalkerImaginer.encode` itself) into the next latent. Dense truth (health clearance at every
step), true block progress and the probe readout of the real frames come back together, so the
imagined-minus-truth gap splits into model-dynamics error and probe error.

Exactness. The physics repeats `execute_branch` operation for operation (restore, clip,
`do_simulation`, finite check, copies, x velocity) and every task in a lockstep group owns its env
for the whole segment, so the solver warm start carries across blocks as in one branch: a fixed
tape reproduces `execute_branch` bitwise and repeated runs are identical. Frames come from a
separate render env, so rendering never touches the physics. (S4's stored bank frames were taken
with `env.render()` straight after `mj_step`, whose kinematics lag one MuJoCo substep: about 0.1% of
pixels differ, and the probe readout by at most 0.002 m, 0.008 rad, 0.015 m/s on healthy blocks of
the development bank.)

A task's outcome is a function of (theta[p], root r) alone, never of the tasks that share its
lockstep group, so `max_envs`, `pairs`, chunking and the worker count cannot change it. Batched
kernels would break that: with the real model on CUDA one frame's latent falls into 17 bitwise
classes over batch sizes 1 to 64 (batch 1 takes a TF32 cuDNN path, 6e-4 off) and a batched act
into 3, and on CPU a batched act also changes with a task's slot (memory alignment of its
weights). The closed loop amplifies those last bits into different falls: changing only
`max_envs` moved the first unsafe step in 10 of 96 development-bank tasks and progress by up to
0.3 m. So every block-end encoder call holds exactly `encode_batch` frames (black frames pad the
last one; at a fixed shape a frame's latent was identical at every slot and with every partner
set tried, real model on CUDA and tiny model on CPU), and the action (`policy.act(theta[[p]], feats)` on
freshly allocated inputs) and the probe readout (one segment's 10 latents) run one task per call.
Results still depend on `encode_batch` (default 64, the same for every stage), the device and the
software stack. History latents are encoded one root per call, bitwise the S4 and
`evoRoots.encode_histories` recipe. `RealPool` (behind `evaluate_parallel`) runs every executor on
one CPU thread, in process as in its spawned workers.

    ex = RealExecutor(model, scaler, probe, policy, device="cuda")
    out = ex.run(theta, roots)            # every (candidate, root), candidate-major
    out["dense_violated"], out["block_progress"], out["readout"], ex.counters
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import multiprocessing  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from helpers.evoImagine import Counters  # noqa: E402
from helpers.evoPolicy import ACTION_DIM, LATENT_DIM, N_PARAMS, LinearLatentPolicy  # noqa: E402
from helpers.locoData import RenderContext  # noqa: E402
from helpers.locoEnv import make_loco_env  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.walkerBank import endpoint_targets  # noqa: E402
from helpers.walkerLewm import WalkerImaginer, load_walker_model  # noqa: E402
from helpers.walkerRules import FRAMESKIP, HISTORY, HORIZON_BLOCKS, BranchLog, health_clearance, restore, rule_unsafe  # noqa: E402

HORIZON_STEPS = HORIZON_BLOCKS * FRAMESKIP
NQ = 9  # Walker2d: nq == nv
ENCODE_BATCH = 64  # block-end frames per encoder call, padded (module docstring)
_THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
_WORKER: dict = {}


def _check_sizes(max_envs: int, encode_batch: int) -> None:
    if int(max_envs) < 1:
        raise ValueError("max_envs must be at least 1")
    if not 1 <= int(encode_batch) <= 256:  # WalkerImaginer.encode cuts larger calls into chunks of 256
        raise ValueError("encode_batch must be in [1, 256]")


def _physics_env():
    """Physics only, as the S4 branches ran it: no rendering, termination off, reset once."""
    env = make_loco_env("Walker2d", "v1", render=False, terminate_when_unhealthy=False, six_tuple=True)
    env.reset(seed=0)
    return env


def _start(env, root, qp, qv) -> None:
    """Exact restore of the root and row 0, as `execute_branch` begins."""
    restore(env, root.qpos, root.qvel)
    u = env.unwrapped
    qp[0], qv[0] = u.data.qpos.copy(), u.data.qvel.copy()


def _step(u, a, qp, qv, xv, t: int) -> None:
    """Environment step t, operation for operation as in `execute_branch`."""
    x0 = u.data.qpos[0]
    u.do_simulation(np.clip(a, -1.0, 1.0), u.frame_skip)
    if not (np.isfinite(u.data.qpos).all() and np.isfinite(u.data.qvel).all()):
        raise FloatingPointError("Nonfinite Walker branch truth; do not label the unseen future safe")
    qp[t + 1], qv[t + 1] = u.data.qpos.copy(), u.data.qvel.copy()
    xv[t] = (u.data.qpos[0] - x0) / u.dt


class RealExecutor:
    """Real segments for (candidate, root) tasks in lockstep groups of up to `max_envs` envs, block
    ends encoded `encode_batch` frames per call (padded).

    `model`, `scaler`, `probe` and `policy` may all be None for physics-only use (`run_tapes`).
    """

    def __init__(self, model, scaler, probe, policy: LinearLatentPolicy | None, *, device: str = "cuda", ctx: RenderContext | None = None,
                 max_envs: int = 64, encode_batch: int = ENCODE_BATCH):
        _check_sizes(max_envs, encode_batch)
        self.device, self.probe, self.policy, self.max_envs, self.encode_batch = device, probe, policy, int(max_envs), int(encode_batch)
        self.imaginer = None if model is None else WalkerImaginer(model, scaler, device)
        self.ctx, self._own_ctx = (RenderContext(), True) if ctx is None else (ctx, False)
        self.counters = Counters()
        self._envs: list = []
        self._hist: dict = {}

    def close(self) -> None:
        for env in self._envs:
            env.close()
        self._envs = []
        if self._own_ctx:
            self.ctx.close()

    # ---- rendering, encoding, physics ------------------------------------------------------
    def encode(self, frames) -> torch.Tensor:
        """uint8 (N, H, W, 3) -> (N, D) float32 on the device: `WalkerImaginer.encode` itself."""
        if self.imaginer is None:
            raise ValueError("this executor has no model (physics only)")
        z = self.imaginer.encode(frames)
        self.counters.encodes += int(z.shape[0])
        return z

    def _encode_ends(self, frames: np.ndarray) -> torch.Tensor:
        """Block-end frames (n, H, W, 3) -> (n, D) in encoder calls of exactly `encode_batch`
        frames, black frames padding the last (not counted), so no latent depends on its group."""
        if self.imaginer is None:
            raise ValueError("this executor has no model (physics only)")
        n, B = len(frames), self.encode_batch
        if n % B:
            frames = np.concatenate([frames, np.zeros((B - n % B, *frames.shape[1:]), frames.dtype)])
        z = torch.cat([self.imaginer.encode(frames[i : i + B]) for i in range(0, len(frames), B)])[:n]
        self.counters.encodes += n
        return z

    def _act(self, th: list, z_t: torch.Tensor, z_prev: torch.Tensor) -> np.ndarray:
        """`policy.act(theta[[p]], feats)` one task per call (`th[i]` is task i's theta[[p]]), on
        freshly allocated inputs: batched, an action would depend on the batch size and the slot."""
        a = [self.policy.act(t, self.policy.features(z_t[i : i + 1], z_prev[i : i + 1])[:, None]) for i, t in enumerate(th)]
        return torch.cat(a)[:, 0].double().cpu().numpy()

    def _render(self, qpos, qvel) -> np.ndarray:
        frames = self.ctx.render_many(qpos, qvel)
        self.counters.renders += len(frames)
        return frames

    def _physics(self, n: int) -> list:
        while len(self._envs) < n:
            self._envs.append(_physics_env())
        return self._envs[:n]

    def history_latents(self, roots) -> torch.Tensor:
        """(R, 3, D) latents of each root's history states (t-20, t-10, t), rendered with
        `ctx.render_many` and encoded one root per call, cached by root_id."""
        t0 = time.perf_counter()
        z = self._history(roots)
        self.counters.wall_s += time.perf_counter() - t0
        return z

    def _history(self, roots) -> torch.Tensor:
        for root in roots:
            hq, hv = np.asarray(root.history_qpos, np.float64), np.asarray(root.history_qvel, np.float64)
            hit = self._hist.get(root.root_id)
            if hit is None:
                self._hist[root.root_id] = (hq.copy(), hv.copy(), self.encode(self._render(hq, hv)))
            elif not (np.array_equal(hit[0], hq) and np.array_equal(hit[1], hv)):
                raise ValueError(f"root id {root.root_id} reused for a different history")
        if not len(roots):
            return torch.empty((0, HISTORY, LATENT_DIM), device=self.device)
        return torch.stack([self._hist[root.root_id][2] for root in roots])

    # ---- fixed tapes -----------------------------------------------------------------------------
    def run_tapes(self, roots, tapes) -> list[BranchLog]:
        """Physics only: tape i (T, 6) from root i, bitwise `execute_branch` (frames None)."""
        t0 = time.perf_counter()
        tapes = np.asarray(tapes, dtype=np.float64)
        if tapes.ndim != 3 or len(tapes) != len(roots) or tapes.shape[2] != ACTION_DIM:
            raise ValueError(f"tapes must be ({len(roots)}, T, {ACTION_DIM}), got {tapes.shape}")
        T, logs = tapes.shape[1], []
        for lo in range(0, len(roots), self.max_envs):
            group = range(lo, min(len(roots), lo + self.max_envs))
            envs = self._physics(len(group))
            bufs = [(np.empty((T + 1, NQ)), np.empty((T + 1, NQ)), np.empty(T)) for _ in group]
            for env, i, (qp, qv, _) in zip(envs, group, bufs):
                _start(env, roots[i], qp, qv)
            for t in range(T):
                for env, i, (qp, qv, xv) in zip(envs, group, bufs):
                    _step(env.unwrapped, tapes[i, t], qp, qv, xv, t)
            logs += [BranchLog(qp, qv, xv, tapes[i].copy()) for i, (qp, qv, xv) in zip(group, bufs)]
        self.counters.real_steps += len(roots) * T
        self.counters.wall_s += time.perf_counter() - t0
        return logs

    # ---- closed loop -----------------------------------------------------------------------------
    def run(self, theta, roots, pairs=None, *, record_qpos: bool = False, z_hist=None) -> dict:
        """theta (P, 2310); tasks are `pairs` [(p, r), ...] or, if None, every (p, r), p major.

        `z_hist` (R, 3, D), e.g. `RootSet.z_hist`, replaces `history_latents(roots)` so a real
        segment can start from exactly the latents imagination started from. Each task's outputs
        depend on its own (theta[p], root r) only, not on the other tasks or `max_envs` (module
        docstring). Returns numpy arrays in task order: pairs (N, 2); actions (N, 10, 6) as
        executed; x_velocity (N, 100);
        dense_clearance (N, 100), the health clearance of rows 1..100; dense_violated (N,);
        dense_first_step (N,), the first unsafe step index or 100; endpoint_targets (N, 10, 3), the
        true (height, pitch, speed) at block ends; endpoint_clearance (N, 10); block_progress
        (N, 10) and progress (N,), torso x displacements; readout (N, 10, 3), the probe on the real
        block-end latents z (N, 10, D) float32; qpos and qvel (N, 101, 9) only with `record_qpos`.
        """
        t0 = time.perf_counter()
        theta = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
        if theta.ndim != 2 or theta.shape[1] != N_PARAMS:
            raise ValueError(f"theta must be (P, {N_PARAMS}), got {tuple(theta.shape)}")
        tasks = _tasks(len(theta), len(roots), pairs)
        zh = self._history(roots) if z_hist is None else torch.as_tensor(z_hist, dtype=torch.float32, device=self.device)
        if tuple(zh.shape) != (len(roots), HISTORY, LATENT_DIM):
            raise ValueError(f"z_hist must be ({len(roots)}, {HISTORY}, {LATENT_DIM}), got {tuple(zh.shape)}")
        buf = _alloc(len(tasks))
        with torch.inference_mode():
            for lo in range(0, len(tasks), self.max_envs):
                hi = lo + self.max_envs
                self._group(theta, roots, zh, tasks[lo:hi], {k: v[lo:hi] for k, v in buf.items()})
        self.counters.real_steps += len(tasks) * HORIZON_STEPS
        self.counters.wall_s += time.perf_counter() - t0
        return _result(buf, tasks, record_qpos)

    def _group(self, theta, roots, zh, tasks, out) -> None:
        """One lockstep group: each task restored into its own env, then per block act on
        (z_t, z_prev), hold for 10 steps, render every block-end state and encode them."""
        envs = self._physics(len(tasks))
        qp, qv, xv = out["qpos"], out["qvel"], out["x_velocity"]
        for i, (env, r) in enumerate(zip(envs, tasks[:, 1])):
            _start(env, roots[r], qp[i], qv[i])
        rows = torch.as_tensor(tasks[:, 1], device=self.device)
        th = [theta[[p]] for p in tasks[:, 0].tolist()]  # a fresh (1, 2310) tensor per task
        z_t, z_prev, zs = zh[rows, HISTORY - 1], zh[rows, HISTORY - 2], []
        for b in range(HORIZON_BLOCKS):
            a = self._act(th, z_t, z_prev)
            out["actions"][:, b] = a
            for i, env in enumerate(envs):
                for t in range(b * FRAMESKIP, (b + 1) * FRAMESKIP):
                    _step(env.unwrapped, a[i], qp[i], qv[i], xv[i], t)
            end = (b + 1) * FRAMESKIP
            z_prev, z_t = z_t, self._encode_ends(self._render(qp[:, end], qv[:, end]))
            zs.append(z_t)
        z = torch.stack(zs, 1)
        out["z"][:] = z.cpu().numpy()
        out["readout"][:] = np.stack([self.probe.predict(seg) for seg in z])  # one segment (10, D) per call


def _tasks(n_theta: int, n_roots: int, pairs) -> np.ndarray:
    """(N, 2) int64 (candidate, root) tasks; None is the full grid in (p, r) order."""
    if pairs is None:
        grid = np.meshgrid(np.arange(n_theta, dtype=np.int64), np.arange(n_roots, dtype=np.int64), indexing="ij")
        return np.stack(grid, -1).reshape(-1, 2)
    tasks = np.asarray(pairs, dtype=np.int64)
    tasks = tasks.reshape(0, 2) if tasks.size == 0 else tasks
    if tasks.ndim != 2 or tasks.shape[1] != 2:
        raise ValueError("pairs must be a list of (candidate, root) index pairs")
    if ((tasks < 0).any(1) | (tasks[:, 0] >= n_theta) | (tasks[:, 1] >= n_roots)).any():
        raise ValueError(f"pairs must index into theta ({n_theta}) and roots ({n_roots})")
    return tasks


def _alloc(n: int) -> dict:
    T, K = HORIZON_STEPS, HORIZON_BLOCKS
    return {"qpos": np.empty((n, T + 1, NQ)), "qvel": np.empty((n, T + 1, NQ)), "x_velocity": np.empty((n, T)),
            "actions": np.empty((n, K, ACTION_DIM)), "z": np.empty((n, K, LATENT_DIM), np.float32), "readout": np.empty((n, K, 3))}


def _result(buf: dict, tasks: np.ndarray, record_qpos: bool) -> dict:
    """Dense truth from the recorded states, next to what the policy saw and did."""
    qp, xv = buf["qpos"], buf["x_velocity"]
    clearance = health_clearance(qp[:, 1:, 1], qp[:, 1:, 2])
    unsafe = rule_unsafe("health", clearance)
    violated = unsafe.any(1)
    ends = np.array([endpoint_targets(q, v) for q, v in zip(qp, xv)], dtype=np.float64).reshape(-1, HORIZON_BLOCKS, 3)
    out = {"pairs": tasks, "actions": buf["actions"], "x_velocity": xv, "dense_clearance": clearance, "dense_violated": violated,
           "dense_first_step": np.where(violated, unsafe.argmax(1), HORIZON_STEPS), "endpoint_targets": ends,
           "endpoint_clearance": health_clearance(ends[..., 0], ends[..., 1]),
           "block_progress": qp[:, FRAMESKIP::FRAMESKIP, 0] - qp[:, :-1:FRAMESKIP, 0], "progress": qp[:, -1, 0] - qp[:, 0, 0],
           "readout": buf["readout"], "z": buf["z"]}
    if record_qpos:
        out.update(qpos=qp, qvel=buf["qvel"])
    return out


# ---- spawned workers ------------------------------------------------------------------------------
def _build(model_dir: str, probe_path: str, policy_state: dict, device: str, max_envs: int, encode_batch: int) -> RealExecutor:
    model, scaler = load_walker_model(Path(model_dir), device)
    probe, _ = load_probe(Path(probe_path), device)
    policy = LinearLatentPolicy.from_state(policy_state, device=device)
    return RealExecutor(model, scaler, probe, policy, device=device, max_envs=max_envs, encode_batch=encode_batch)


def _init_worker(*args) -> None:
    """One CPU thread per worker (helpers.threads), then the model, probe, policy, its own
    RenderContext and executor, once."""
    for key in _THREAD_KEYS:
        os.environ[key] = "1"
    torch.set_num_threads(1)
    _WORKER["executor"] = _build(*args)


def _worker_threads() -> dict:
    """The thread settings a process runs with (submitted to live workers by the tests)."""
    return {"torch": torch.get_num_threads(), **{key: os.environ.get(key) for key in _THREAD_KEYS}}


def _run_chunk(theta, roots, pairs, z_hist, record_qpos, *, executor=None):
    ex = executor if executor is not None else _WORKER["executor"]
    ex.counters = Counters()
    return ex.run(theta, roots, pairs, record_qpos=record_qpos, z_hist=z_hist), ex.counters


def _chunk(theta, roots, tasks, z_hist) -> tuple:
    """Only the candidates and roots a chunk uses, its tasks renumbered into them (same values,
    same order)."""
    ps, p_local = np.unique(tasks[:, 0], return_inverse=True)
    rs, r_local = np.unique(tasks[:, 1], return_inverse=True)
    return theta[ps], [roots[r] for r in rs], np.stack([p_local, r_local], 1), None if z_hist is None else z_hist[rs]


class RealPool:
    """Spawned workers (never fork: an inherited EGL context renders garbage without raising) that
    stay alive across `evaluate` calls. Each worker loads the model (`load_walker_model`), probe and
    policy (`LinearLatentPolicy.from_state(policy_state)`) and builds its own RenderContext and
    executor once, on one CPU thread so W workers fit the cgroup quota. `n_workers == 0` keeps one
    executor in process and runs it on one torch thread too (restored after each call), so even
    CPU kernels see the same thread count. Every input is checked here, so a bad one fails before
    any worker starts instead of as a broken pool. Use it as a context manager or `close()` it;
    create it under `if __name__ == "__main__":` in scripts.
    """

    def __init__(self, *, model_dir, probe_path, policy_state, n_workers: int, device: str = "cuda", max_envs: int = 64, chunk_tasks: int = 256,
                 encode_batch: int = ENCODE_BATCH):
        if n_workers < 0 or chunk_tasks < 1:
            raise ValueError("need n_workers >= 0 and chunk_tasks >= 1")
        _check_sizes(max_envs, encode_batch)
        needed = [Path(model_dir) / name for name in ("config.json", "weights.pt", "scalers.npz")] + [Path(probe_path)]  # load_walker_model, load_probe
        missing = [str(p) for p in needed if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"real executor inputs not found: {missing}")
        LinearLatentPolicy.from_state(policy_state)  # a bad state fails here, not as a broken worker pool
        self.chunk_tasks = int(chunk_tasks)
        args = (str(model_dir), str(probe_path), policy_state, device, int(max_envs), int(encode_batch))
        self._ex = _build(*args) if n_workers == 0 else None
        self._pool = None if n_workers == 0 else ProcessPoolExecutor(n_workers, mp_context=multiprocessing.get_context("spawn"),
                                                                     initializer=_init_worker, initargs=args)

    def evaluate(self, theta, roots, pairs=None, *, record_qpos: bool = False, z_hist=None, chunk_tasks: int | None = None) -> dict:
        """`RealExecutor.run` over chunks of `chunk_tasks` tasks (this call's, else the pool's):
        results in task order, every counter summed over the chunks under "counters" (wall_s is
        executor seconds, so with W workers it can exceed the elapsed time) and this call's
        elapsed seconds under "elapsed_s"."""
        t0 = time.perf_counter()
        if self._ex is None and self._pool is None:
            raise ValueError("this RealPool is closed")
        n = self.chunk_tasks if chunk_tasks is None else int(chunk_tasks)
        if n < 1:
            raise ValueError("chunk_tasks must be at least 1")
        theta = theta.detach().cpu().numpy() if isinstance(theta, torch.Tensor) else np.asarray(theta)
        if z_hist is not None:
            z_hist = z_hist.detach().cpu().numpy() if isinstance(z_hist, torch.Tensor) else np.asarray(z_hist)
        tasks = _tasks(len(theta), len(roots), pairs)
        jobs = [(*_chunk(theta, roots, tasks[i : i + n], z_hist), record_qpos) for i in range(0, len(tasks), n)]
        if self._pool is not None:
            futures = [self._pool.submit(_run_chunk, *job) for job in jobs]
            results = [f.result() for f in futures]
        else:
            n_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                results = [_run_chunk(*job, executor=self._ex) for job in jobs]
            finally:
                torch.set_num_threads(n_threads)
        out = {k: np.concatenate([r[k] for r, _ in results]) for k in results[0][0]} if results else _result(_alloc(0), tasks, record_qpos)
        out["pairs"] = tasks
        counters = Counters()
        for _, c in results:
            counters.add(c)
        out["counters"], out["elapsed_s"] = counters, time.perf_counter() - t0
        return out

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
        if self._ex is not None:
            self._ex.close()
        self._pool = self._ex = None

    def __enter__(self) -> "RealPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def evaluate_parallel(theta, roots, pairs, *, model_dir, probe_path, policy_state, n_workers: int, device: str = "cuda",
                      max_envs: int = 64, chunk_tasks: int = 256, encode_batch: int = ENCODE_BATCH, record_qpos: bool = False,
                      z_hist=None) -> dict:
    """One `RealPool.evaluate` with workers started and stopped around it ("elapsed_s" includes both).

    Tasks as in `RealExecutor.run` (`pairs` None for the full grid). The result does not depend on
    `n_workers`, `chunk_tasks` or `max_envs`.
    """
    t0 = time.perf_counter()
    with RealPool(model_dir=model_dir, probe_path=probe_path, policy_state=policy_state, n_workers=n_workers, device=device,
                  max_envs=max_envs, chunk_tasks=chunk_tasks, encode_batch=encode_batch) as pool:
        out = pool.evaluate(theta, roots, pairs, record_qpos=record_qpos, z_hist=z_hist)
    out["elapsed_s"] = time.perf_counter() - t0
    return out
