"""Sequential-block real executor, adapted from evoReal (commit 10e8d16).

Uses the exact same snapshot restoration, physics, padded encoder calls, task-local
policy calls, readouts and accounting. Only the action interface changes: execute
all ten distinct actions, in order. Inherited tape execution stays the reference.
The pool implementation is copied to use this executor in spawned workers without
patching globals in the historical module.
"""
from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from helpers.evoBlockPolicy import BLOCK_DIM, N_PARAMS, LinearBlockPolicy
from helpers.evoImagine import Counters
from helpers.evoPolicy import LATENT_DIM
from helpers.evoReal import (
    ENCODE_BATCH, HORIZON_STEPS, _THREAD_KEYS, RealExecutor, _alloc, _check_sizes,
    _chunk, _result, _start, _step, _tasks,
)
from helpers.poseProbes import load_probe
from helpers.walkerLewm import load_walker_model
from helpers.walkerRules import FRAMESKIP, HISTORY, HORIZON_BLOCKS

_WORKER = {}


class BlockRealExecutor(RealExecutor):
    def run(self, theta, roots, pairs=None, *, record_qpos: bool = False, z_hist=None) -> dict:
        """theta (P, 23100); tasks are `pairs` [(p, r), ...] or, if None, every (p, r), p major.

        `z_hist` (R, 3, D), e.g. `RootSet.z_hist`, replaces `history_latents(roots)` so a real
        segment can start from exactly the latents imagination started from. Each task's outputs
        depend on its own (theta[p], root r) only, not on the other tasks or `max_envs` (module
        docstring). Returns numpy arrays in task order: pairs (N, 2); actions (N, 10, 60) as
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
        buf = _block_alloc(len(tasks))
        with torch.inference_mode():
            for lo in range(0, len(tasks), self.max_envs):
                hi = lo + self.max_envs
                self._group(theta, roots, zh, tasks[lo:hi], {k: v[lo:hi] for k, v in buf.items()})
        self.counters.real_steps += len(tasks) * HORIZON_STEPS
        self.counters.wall_s += time.perf_counter() - t0
        return _result(buf, tasks, record_qpos)

    def _group(self, theta, roots, zh, tasks, out) -> None:
        """One lockstep group: each task restored into its own env, then per block act on
        (z_t, z_prev), execute 10 sequential actions, render every block-end state and encode them."""
        envs = self._physics(len(tasks))
        qp, qv, xv = out["qpos"], out["qvel"], out["x_velocity"]
        for i, (env, r) in enumerate(zip(envs, tasks[:, 1])):
            _start(env, roots[r], qp[i], qv[i])
        rows = torch.as_tensor(tasks[:, 1], device=self.device)
        th = [theta[[p]] for p in tasks[:, 0].tolist()]  # a fresh (1, 23100) tensor per task
        z_t, z_prev, zs = zh[rows, HISTORY - 1], zh[rows, HISTORY - 2], []
        for b in range(HORIZON_BLOCKS):
            a = self._act(th, z_t, z_prev)
            out["actions"][:, b] = a
            blocks = a.reshape(len(tasks), FRAMESKIP, 6)
            for i, env in enumerate(envs):
                for t in range(b * FRAMESKIP, (b + 1) * FRAMESKIP):
                    _step(env.unwrapped, blocks[i, t - b * FRAMESKIP], qp[i], qv[i], xv[i], t)
            end = (b + 1) * FRAMESKIP
            z_prev, z_t = z_t, self._encode_ends(self._render(qp[:, end], qv[:, end]))
            zs.append(z_t)
        z = torch.stack(zs, 1)
        out["z"][:] = z.cpu().numpy()
        out["readout"][:] = np.stack([self.probe.predict(seg) for seg in z])  # one segment (10, D) per call


def _block_alloc(n):
    buf = _alloc(n)
    buf["actions"] = np.empty((n, HORIZON_BLOCKS, BLOCK_DIM))
    return buf


def _build(model_dir: str, probe_path: str, policy_state: dict, device: str, max_envs: int, encode_batch: int) -> BlockRealExecutor:
    model, scaler = load_walker_model(Path(model_dir), device)
    probe, _ = load_probe(Path(probe_path), device)
    policy = LinearBlockPolicy.from_state(policy_state, device=device)
    return BlockRealExecutor(model, scaler, probe, policy, device=device, max_envs=max_envs, encode_batch=encode_batch)


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


class BlockRealPool:
    """Spawned workers (never fork: an inherited EGL context renders garbage without raising) that
    stay alive across `evaluate` calls. Each worker loads the model (`load_walker_model`), probe and
    policy (`LinearBlockPolicy.from_state(policy_state)`) and builds its own RenderContext and
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
        LinearBlockPolicy.from_state(policy_state)  # a bad state fails here, not as a broken worker pool
        self.chunk_tasks = int(chunk_tasks)
        args = (str(model_dir), str(probe_path), policy_state, device, int(max_envs), int(encode_batch))
        self._ex = _build(*args) if n_workers == 0 else None
        self._pool = None if n_workers == 0 else ProcessPoolExecutor(n_workers, mp_context=multiprocessing.get_context("spawn"),
                                                                     initializer=_init_worker, initargs=args)

    def evaluate(self, theta, roots, pairs=None, *, record_qpos: bool = False, z_hist=None, chunk_tasks: int | None = None) -> dict:
        """`BlockRealExecutor.run` over chunks of `chunk_tasks` tasks (this call's, else the pool's):
        results in task order, every counter summed over the chunks under "counters" (wall_s is
        executor seconds, so with W workers it can exceed the elapsed time) and this call's
        elapsed seconds under "elapsed_s"."""
        t0 = time.perf_counter()
        if self._ex is None and self._pool is None:
            raise ValueError("this BlockRealPool is closed")
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
        out = {k: np.concatenate([r[k] for r, _ in results]) for k in results[0][0]} if results else _result(_block_alloc(0), tasks, record_qpos)
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

    def __enter__(self) -> "BlockRealPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def evaluate_blocks_parallel(theta, roots, pairs, *, model_dir, probe_path, policy_state, n_workers: int, device: str = "cuda",
                      max_envs: int = 64, chunk_tasks: int = 256, encode_batch: int = ENCODE_BATCH, record_qpos: bool = False,
                      z_hist=None) -> dict:
    """One `BlockRealPool.evaluate` with workers started and stopped around it ("elapsed_s" includes both).

    Tasks as in `BlockRealExecutor.run` (`pairs` None for the full grid). The result does not depend on
    `n_workers`, `chunk_tasks` or `max_envs`.
    """
    t0 = time.perf_counter()
    with BlockRealPool(model_dir=model_dir, probe_path=probe_path, policy_state=policy_state, n_workers=n_workers, device=device,
                  max_envs=max_envs, chunk_tasks=chunk_tasks, encode_batch=encode_batch) as pool:
        out = pool.evaluate(theta, roots, pairs, record_qpos=record_qpos, z_hist=z_hist)
    out["elapsed_s"] = time.perf_counter() - t0
    return out
