"""Batched LeWM rollouts from a root, and the nominal CEM controller on the raw env.

Interfaces checked against the installed `stable_worldmodel` 0.1.1 LeWM (`rollout`,
`get_cost`) and `CEMSolver` on 26 September 2026:

- `rollout(info, actions)` splits the action sequence into the first `H` entries (one per
  history frame) and the rest, then predicts `T - H + 1` steps. For `H` real frames and a
  tape of `K` future blocks the sequence is `(H-1)` known history blocks followed by the
  `K` tape blocks, so `T = H + K - 1` and `predicted_emb[:, :, H:]` holds the `K` imagined
  latents; entries `< H` are the (encoded) real history.
- `get_cost` encodes the goal from `info['goal']` and pops `info['action']`, so a planning
  info dict must carry both keys; it also caches `emb`/`goal_emb` across CEM iterations.
- The criterion compares the LAST predicted latent with the goal, and `goal_emb` only
  broadcasts when the batch dimension is 1, so planning runs one root at a time.
- CEM samples every slot of the sequence, including the history slots. `HistoryCostModel`
  overwrites those slots with the known history blocks before each cost call, so the
  candidates the model scores are always consistent with what really happened.

Actions are normalised with the fitted action scaler and flattened per block
(5 consecutive 2-d actions -> 10-d), the layout the dataset loader uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from gymnasium.spaces import Box as GymBox

from helpers.probes import preprocess_pixels
from helpers.pushtAssets import (
    ACTION_BLOCK,
    HISTORY_FRAMES,
    HORIZON_BLOCKS,
    denormalise_actions,
    normalise_actions,
)

LATENT_DIM = 192


def encode(model, frames, device: str, batch_size: int = 256) -> torch.Tensor:
    """uint8 frames (N, 224, 224, 3) -> latents (N, D), float32 on `device`."""
    frames = np.asarray(frames)
    out = []
    with torch.inference_mode():
        for i in range(0, len(frames), batch_size):
            x = preprocess_pixels(frames[i : i + batch_size], device)
            out.append(model.encode({"pixels": x.unsqueeze(0)})["emb"][0].float())
    return torch.cat(out, 0)


def blocks_to_model(process, blocks: np.ndarray) -> torch.Tensor:
    """Raw action blocks (..., ACTION_BLOCK, 2) -> normalised flat (..., ACTION_BLOCK*2)."""
    blocks = np.asarray(blocks, dtype=np.float64)
    lead = blocks.shape[:-2]
    if blocks.size == 0:
        return torch.zeros((*lead, ACTION_BLOCK * 2), dtype=torch.float32)
    flat = normalise_actions(process, blocks.reshape(-1, 2)).reshape(*lead, ACTION_BLOCK * 2)
    return torch.from_numpy(np.ascontiguousarray(flat)).float()


def model_to_blocks(process, flat: torch.Tensor) -> np.ndarray:
    """Normalised flat (..., ACTION_BLOCK*2) -> raw action blocks (..., ACTION_BLOCK, 2)."""
    arr = flat.detach().cpu().numpy().astype(np.float64)
    lead = arr.shape[:-1]
    raw = denormalise_actions(process, arr.reshape(-1, 2)).reshape(*lead, ACTION_BLOCK, 2)
    return raw.astype(np.float64)


class Imaginer:
    """Holds the frozen model and scalers; rolls out many tapes from one root."""

    def __init__(self, model, process, device: str = "cuda"):
        self.model = model
        self.process = process
        self.device = device

    def encode(self, frames) -> torch.Tensor:
        return encode(self.model, frames, self.device)

    @torch.inference_mode()
    def rollout(
        self,
        real_emb: torch.Tensor,
        hist_blocks: np.ndarray,
        tapes: np.ndarray,
        *,
        chunk: int = 1024,
    ) -> torch.Tensor:
        """Imagined latents for every tape.

        real_emb: (H, D) latents of the history frames (last = current frame).
        hist_blocks: (H-1, ACTION_BLOCK, 2) raw actions between the history frames.
        tapes: (S, K, ACTION_BLOCK, 2) raw future blocks.
        Returns (S, K, D): the imagined latent after each of the K blocks.
        """
        H = real_emb.shape[0]
        tapes = np.asarray(tapes, dtype=np.float64)
        S, K = tapes.shape[:2]
        hist = blocks_to_model(self.process, hist_blocks).to(self.device)  # (H-1, 10)
        assert hist.shape[0] == H - 1, (hist.shape, H)
        fut = blocks_to_model(self.process, tapes).to(self.device)  # (S, K, 10)
        out = torch.empty((S, K, real_emb.shape[1]), device=self.device)
        emb = real_emb.to(self.device).float()
        for i in range(0, S, chunk):
            f = fut[i : i + chunk]
            s = f.shape[0]
            seq = torch.cat([hist.unsqueeze(0).expand(s, -1, -1), f], dim=1)  # (s, H-1+K, 10)
            info = {"emb": emb.unsqueeze(0).unsqueeze(0).expand(1, s, H, -1),
                    "pixels": torch.empty((1, s, H, 0), device=self.device)}
            pred = self.model.rollout(info, seq.unsqueeze(0))["predicted_emb"]  # (1, s, H+K, D)
            assert pred.shape[2] == H + K, pred.shape
            out[i : i + s] = pred[0, :, H:, :]
        return out

    def real_and_imagined(self, frames, hist_blocks, tapes, **kw):
        real = self.encode(frames)
        return real, self.rollout(real, hist_blocks, tapes, **kw)


# --------------------------------------------------------------------------------------
# nominal controller
# --------------------------------------------------------------------------------------
class HistoryCostModel(torch.nn.Module):
    """LeWM goal cost with pinned history slots and the action-space arena guard.

    The arena guard is the engineering control every arm shares (pushT.md): commanded
    pusher positions must stay inside the arena. It is computed in ACTION space with the
    historical `commanded_positions` (imported from safeCEM.py, never edited) and applied
    by constraint-priority ranking, exactly as the historical Safe CEM ranked feasibility:
    feasible candidates sort by goal cost, infeasible ones after them by exit depth.
    Without it the released planner drives the kinematic pusher out of the arena on
    long-horizon goals (observed on 26 September 2026 with the historical World pipeline).
    """

    def __init__(self, base, hist_flat: torch.Tensor, *, action_scaler=None, pusher_xy=None,
                 arena_guard: bool = True):
        super().__init__()
        self.base = base
        self.register_buffer("hist", hist_flat.clone())  # (H-1, 10)
        self.action_scaler = action_scaler
        self.pusher_xy = None if pusher_xy is None else np.asarray(pusher_xy, dtype=float)
        self.arena_guard = arena_guard and action_scaler is not None and pusher_xy is not None
        self.last_frac_feasible = 1.0

    def get_cost(self, info, candidates):
        n = self.hist.shape[0]
        if n:
            candidates = candidates.clone()
            candidates[:, :, :n] = self.hist.to(candidates.dtype)
        goal_cost = self.base.get_cost(info, candidates)
        if not self.arena_guard:
            return goal_cost
        from helpers.safeCEM import arena_exit_depth, commanded_positions

        with torch.no_grad():
            pos = commanded_positions(candidates[:, :, n:], self.pusher_xy, self.action_scaler)
            exit_depth = arena_exit_depth(pos).sum(dim=-1)  # (B, S)
        feasible = exit_depth <= 0.0
        self.last_frac_feasible = float(feasible.float().mean())
        big = (goal_cost[feasible].max() if feasible.any() else goal_cost.max()) + 1.0
        return torch.where(feasible, goal_cost, big + exit_depth)


@dataclass
class PlanResult:
    blocks: np.ndarray  # (HORIZON_BLOCKS, ACTION_BLOCK, 2) raw future plan
    flat: torch.Tensor  # (H-1+HORIZON_BLOCKS, 10) normalised, history slots included
    cost: float
    solve_time: float
    frac_feasible: float = 1.0


class NominalPlanner:
    """LeWM + CEM toward the goal image, historical settings (300 samples, 30 iters)."""

    def __init__(self, model, process, device="cuda", *, num_samples=300, n_steps=30,
                 topk=30, var_scale=1.0, horizon=HORIZON_BLOCKS, warm_start=True,
                 arena_guard=True):
        self.model, self.process, self.device = model, process, device
        self.arena_guard = arena_guard
        self.num_samples, self.n_steps, self.topk, self.var_scale = num_samples, n_steps, topk, var_scale
        self.horizon, self.warm_start = horizon, warm_start
        self.goal_emb_cache = None

    def _info(self, frames, goal_frame):
        px = preprocess_pixels(np.asarray(frames), self.device)  # (H, C, 224, 224)
        goal = preprocess_pixels(np.asarray(goal_frame)[None], self.device)  # (1, C, 224, 224)
        return {
            "pixels": px.unsqueeze(0),  # (1, H, C, 224, 224)
            "goal": goal.unsqueeze(0),  # (1, 1, C, 224, 224)
            "action": torch.zeros((1, 1, ACTION_BLOCK * 2), device=self.device),
        }

    def plan(self, frames, hist_blocks, goal_frame, *, seed: int, pusher_xy=None,
             init_future=None) -> PlanResult:
        """One CEM solve. `pusher_xy` anchors the arena guard; `init_future` is a raw
        (horizon, 5, 2) warm start."""
        import time

        from stable_worldmodel.policy import PlanConfig
        from stable_worldmodel.solver import CEMSolver

        H = len(frames)
        n_hist = H - 1
        hist = blocks_to_model(self.process, hist_blocks).to(self.device)
        assert hist.shape[0] == n_hist
        T = n_hist + self.horizon
        torch.manual_seed(seed)
        np.random.seed(seed)
        cost_model = HistoryCostModel(
            self.model, hist, action_scaler=self.process["action"], pusher_xy=pusher_xy,
            arena_guard=self.arena_guard,
        ).to(self.device).eval()
        solver = CEMSolver(model=cost_model, batch_size=1, num_samples=self.num_samples,
                           var_scale=self.var_scale, n_steps=self.n_steps, topk=self.topk,
                           device=self.device, seed=seed)
        solver.configure(action_space=GymBox(-1, 1, shape=(1, 2), dtype=np.float32), n_envs=1,
                         config=PlanConfig(horizon=T, receding_horizon=1, action_block=ACTION_BLOCK))
        init = torch.zeros((1, T, ACTION_BLOCK * 2), device=self.device)
        init[0, :n_hist] = hist
        if init_future is not None:
            f = blocks_to_model(self.process, init_future).to(self.device)
            init[0, n_hist : n_hist + f.shape[0]] = f
        t0 = time.time()
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):  # CEMSolver prints its solve time
            out = solver.solve(self._info(frames, goal_frame), init_action=init)
        flat = out["actions"][0].clone()  # (T, 10) normalised
        flat[:n_hist] = hist.cpu()
        blocks = model_to_blocks(self.process, flat[n_hist:])
        res = PlanResult(blocks, flat, float(out["costs"][0]), time.time() - t0)
        res.frac_feasible = cost_model.last_frac_feasible
        return res


def next_warm_start(plan_blocks: np.ndarray) -> np.ndarray:
    """Shift the previous future plan by one executed block, zero-padding the tail."""
    nxt = np.zeros_like(plan_blocks)
    nxt[:-1] = plan_blocks[1:]
    return nxt


__all__ = [
    "Imaginer", "NominalPlanner", "PlanResult", "HistoryCostModel", "encode",
    "blocks_to_model", "model_to_blocks", "next_warm_start", "LATENT_DIM", "HISTORY_FRAMES",
]
