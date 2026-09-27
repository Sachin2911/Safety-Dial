"""Safe CEM with the WHOLE-T constraint on imagined futures (E5, pushT.md).

The historical Safe CEM (`safeCEM.py`, read-only) constrained the pusher through a
pusher probe. Here the block-pose probe reads imagined latents, the T polygons are placed
from the predicted pose, and the signed clearance to the virtual hazard is computed on
the endpoint-interpolated path. Ranking is constraint-priority as in the historical
module: feasible candidates (min clearance >= dial and inside the arena) sort by goal
cost, infeasible ones after them by violation. `mode="penalty"` gives the matched
penalty-CEM reference `goal + lam * violation`. The arena guard is always on.
"""

from __future__ import annotations

import torch

from helpers.decomposition import interp_poses_batch
from helpers.imagination import HistoryCostModel, NominalPlanner
from helpers.pushtGeometry import Hazard, clearance_trace
from helpers.safeCEM import arena_exit_depth, commanded_positions


class TConstraintCostModel(HistoryCostModel):
    def __init__(self, base, hist_flat, *, probe, hazard: Hazard, dial: float = 0.0, mode: str = "safe", lam: float = 1.0,
                 action_scaler=None, pusher_xy=None):
        super().__init__(base, hist_flat, action_scaler=action_scaler, pusher_xy=pusher_xy, arena_guard=True)
        self.probe, self.hazard, self.dial, self.mode, self.lam = probe, hazard, float(dial), mode, float(lam)
        self.last_diag = {}

    def get_cost(self, info, candidates):
        n = self.hist.shape[0]
        if n:
            candidates = candidates.clone()
            candidates[:, :, :n] = self.hist.to(candidates.dtype)
        goal_cost = self.base.get_cost(info, candidates)  # populates predicted_emb
        H = info["pixels"].shape[2]
        with torch.no_grad():
            emb = info["predicted_emb"][:, :, H - 1 :, :]  # current real frame + K imagined
            B, S, T, D = emb.shape
            pose = self.probe.predict_pose(emb.reshape(-1, D).float()).reshape(B * S, T, 3)
            path = interp_poses_batch(pose)  # (B*S, L, 3)
            L = path.shape[1]
            cmin = clearance_trace(path.reshape(-1, 3), self.hazard).reshape(B * S, L).min(1)
            cmin_t = torch.as_tensor(cmin, device=goal_cost.device, dtype=goal_cost.dtype).reshape(B, S)
            pos = commanded_positions(candidates[:, :, n:], self.pusher_xy, self.action_scaler)
            arena = arena_exit_depth(pos).sum(dim=-1)
            hazard_violation = torch.clamp(self.dial - cmin_t, min=0.0)
        arena_ok = arena <= 0.0
        if self.mode == "penalty":
            # penalty on the hazard; the arena guard stays a hard, common control
            scored = goal_cost + self.lam * hazard_violation
            feasible = arena_ok
            violation = arena
        else:
            feasible = arena_ok & (hazard_violation <= 0.0)
            scored = goal_cost
            violation = hazard_violation + arena
        self.last_frac_feasible = float(feasible.float().mean())
        big = (scored[feasible].max() if feasible.any() else scored.max()) + 1.0
        total = torch.where(feasible, scored, big + violation)
        elite = int(total.argmin(dim=1)[0])
        self.last_diag = {"frac_feasible": self.last_frac_feasible, "elite_feasible": bool(feasible[0, elite]), "elite_cmin": float(cmin_t[0, elite]),
                          "elite_hazard_violation": float(hazard_violation[0, elite]), "elite_arena": float(arena[0, elite]), "cmin_median": float(cmin_t.median())}
        return total


class SafePlannerT(NominalPlanner):
    def __init__(self, model, process, device="cuda", *, probe, hazard: Hazard, dial=0.0, mode="safe", lam=1.0, **kw):
        super().__init__(model, process, device, **kw)
        self.probe, self.hazard, self.dial, self.mode, self.lam = probe, hazard, dial, mode, lam

    def _cost_model(self, hist, pusher_xy):
        return TConstraintCostModel(self.model, hist, probe=self.probe, hazard=self.hazard, dial=self.dial, mode=self.mode, lam=self.lam,
                                    action_scaler=self.process["action"], pusher_xy=pusher_xy)
