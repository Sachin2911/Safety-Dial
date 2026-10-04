"""MLP real execution with unchanged sequential physics and encoding protocol.

The run method is adapted from evoBlockReal at e62c3f9 solely to validate the
policy's variable parameter count. _group, _act, history encoding, padded block
encoding, rendering, physics and result statistics are inherited unchanged.
This development diagnostic uses one process; no new multiprocessing path.
"""
import time

import torch

from helpers.evoBlockReal import BlockRealExecutor, _block_alloc
from helpers.evoPolicy import LATENT_DIM
from helpers.evoReal import HORIZON_STEPS, _result, _tasks
from helpers.walkerRules import HISTORY


class MLPRealExecutor(BlockRealExecutor):
    def run(self, theta, roots, pairs=None, *, record_qpos: bool = False, z_hist=None) -> dict:
        """theta (P, n_params); tasks are `pairs` [(p, r), ...] or, if None, every (p, r), p major.

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
        if theta.ndim != 2 or theta.shape[1] != self.policy.n_params:
            raise ValueError(f"theta must be (P, {self.policy.n_params}), got {tuple(theta.shape)}")
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

