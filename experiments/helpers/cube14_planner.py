"""SafeCEM on Cube (task 14): the task-10 constraint-priority cost model with a check on the
commanded end-effector path.

OGBench's arm is driven by relative end-effector commands: each env step sets the target
pinch-point position to the current one plus 0.05 m times the first three action components
(the action is not clipped), clips that target to the workspace box, solves inverse
kinematics in Python and lets the joint PD controllers chase the result for 25 physics
substeps. Simulating that per candidate is too slow for CEM, so the check predicts the
effector path from the commands alone:

    p[k+1] = p[k] + gain * (clip(p[k] + 0.05 * a[k, :3]) - p[k]) + carry * (p[k] - p[k-1])

from the true effector position at every replan. gain = 1, carry = 0 is perfect tracking;
the diagnosis measures the real tracking and fits both. Unlike tasks 10, 12 and 13 the
check is therefore an approximation, and the clearance dial is also a margin for its error.
Points between steps are interpolated so a candidate cannot hop over the band.

Execution modes (solver wrappers live in cube14_runtime):
    mean       the upstream elite mean, which the check never evaluates
    best       the best checked candidate of the solve (iCEM style)
    mean_safe  the elite mean if it passes the same check, else the best checked candidate
"""

from __future__ import annotations

import numpy as np
import torch

from helpers.safecem10_planner import DialCostModel, rank_costs, substep_violation

EXECUTE_MODES = ("mean", "best", "mean_safe")
ACTION_DIM = 5
STEP_M = 0.05
WORKSPACE = ((0.25, -0.35, 0.02), (0.6, 0.35, 0.35))


def raw_actions(action_candidates, action_mean, action_scale):
    """(B, S, T, action_block * 5) normalized -> (B, S, T * action_block, 5) env actions."""
    B, S, T, D = action_candidates.shape
    a = action_candidates.reshape(B, S, T * (D // ACTION_DIM), ACTION_DIM)
    dev, dtype = a.device, a.dtype
    shape = (1, 1, 1, ACTION_DIM)
    mean = torch.as_tensor(np.asarray(action_mean), device=dev, dtype=dtype).view(shape)
    scale = torch.as_tensor(np.asarray(action_scale), device=dev, dtype=dtype).view(shape)
    return a * scale + mean


def commanded_path(effector, raw, *, gain=(1.0, 1.0, 1.0), carry=0.0, prev_step=None):
    """Effector after every env step, (B, S, M, 3), for env actions raw (B, S, M, 5)."""
    B, S, M, _ = raw.shape
    dev, dtype = raw.device, raw.dtype
    lo = torch.as_tensor(WORKSPACE[0], device=dev, dtype=dtype)
    hi = torch.as_tensor(WORKSPACE[1], device=dev, dtype=dtype)
    g = torch.as_tensor(np.broadcast_to(gain, (3,)).copy(), device=dev, dtype=dtype)
    p = torch.as_tensor(np.asarray(effector), device=dev, dtype=dtype).view(1, 1, 3)
    p = p.expand(B, S, 3)
    v = torch.zeros_like(p)
    if prev_step is not None:
        v = v + torch.as_tensor(np.asarray(prev_step), device=dev, dtype=dtype).view(1, 1, 3)
    out = []
    for k in range(M):
        target = torch.maximum(torch.minimum(p + STEP_M * raw[:, :, k, :3], hi), lo)
        nxt = p + g * (target - p) + carry * v
        v, p = nxt - p, nxt
        out.append(p)
    return torch.stack(out, dim=2)


def densify(start, path, n):
    """(B, S, M, k) waypoints after a start point -> (B, S, M * n, k), n points per step."""
    B, S, M, K = path.shape
    prev = torch.cat([start.expand(B, S, 1, K), path[:, :, :-1]], dim=2)
    w = torch.arange(1, n + 1, device=path.device, dtype=path.dtype) / n
    w = w.view(1, 1, 1, n, 1)
    dense = prev.unsqueeze(3) * (1.0 - w) + path.unsqueeze(3) * w
    return dense.reshape(B, S, M * n, K)


def tracking_errors(
    states, actions, start_state, *, gain=(1.0, 1.0, 1.0), carry=0.0, block=5, prev0=None
):
    """How well the commanded model predicts the executed effector positions.

    states (N, 7) after each env step, actions (N, 5) executed, start_state (7,), prev0 the
    effector displacement just before the start. Returns the one-step error and the
    open-loop error over each executed block from its true start."""
    states = np.asarray(states, dtype=np.float64)
    acts = np.asarray(actions, dtype=np.float64)
    ee = np.vstack([np.asarray(start_state, dtype=np.float64)[None, :3], states[:, :3]])
    n = min(len(acts), len(states))
    first = np.zeros(3) if prev0 is None else np.asarray(prev0, dtype=np.float64)
    one = []
    for k in range(n):
        prev = ee[k] - ee[k - 1] if k else first
        raw = torch.as_tensor(acts[k : k + 1]).view(1, 1, 1, ACTION_DIM)
        pred = commanded_path(ee[k], raw, gain=gain, carry=carry, prev_step=prev)[0, 0, 0]
        one.append(np.linalg.norm(pred.numpy()[:2] - ee[k + 1][:2]))
    blocks = []
    for k in range(0, n - block + 1, block):
        prev = ee[k] - ee[k - 1] if k else first
        raw = torch.as_tensor(acts[k : k + block]).view(1, 1, block, ACTION_DIM)
        pred = commanded_path(ee[k], raw, gain=gain, carry=carry, prev_step=prev)[0, 0].numpy()
        blocks.append(np.linalg.norm(pred[:, :2] - ee[k + 1 : k + block + 1, :2], axis=1).max())

    def stats(x):
        return {
            "median": float(np.median(x)) if x else None,
            "p95": float(np.percentile(x, 95)) if x else None,
            "max": float(np.max(x)) if x else None,
            "n": len(x),
        }

    return {"one_step_xy_m": stats(one), "block_xy_m": stats(blocks)}


class CubeCostModel(DialCostModel):
    """World-model goal cost plus a clearance constraint on the commanded effector path."""

    def __init__(
        self,
        base_model,
        boxes,
        clearances,
        *,
        action_mean,
        action_scale,
        mode="safe",
        lam=0.0,
        tol=0.0,
        track_best=False,
        gain=(1.0, 1.0, 1.0),
        carry=0.0,
        interp=4,
    ):
        super().__init__(
            base_model,
            None,
            boxes,
            clearances,
            mode=mode,
            lam=lam,
            tol=tol,
            arena_constraint=False,
            action_mean=action_mean,
            action_scale=action_scale,
            check="exact",
            track_best=track_best,
            controller={"substeps": int(interp)},
        )
        self.gain = tuple(float(g) for g in np.broadcast_to(gain, (3,)))
        self.carry = float(carry)
        self.interp = int(interp)
        self.effector = None  # true pinch-point position now
        self.prev_step = None  # its displacement over the last env step

    def controller_path(self, action_candidates):
        if self.effector is None:
            raise RuntimeError("The check needs the true effector position")
        raw = raw_actions(action_candidates, self.action_mean, self.action_scale)
        return commanded_path(
            self.effector, raw, gain=self.gain, carry=self.carry, prev_step=self.prev_step
        )

    def hazard(self, action_candidates):
        """Summed penetration per candidate (B, S) and its effector at block ends (B, S, T+1, 3)."""
        horizon = int(action_candidates.shape[2])
        clearances = self.clearances + [self.clearances[-1]] * max(
            0, horizon - len(self.clearances)
        )
        path = self.controller_path(action_candidates)
        B, S = path.shape[:2]
        start = torch.as_tensor(np.asarray(self.effector), device=path.device, dtype=path.dtype)
        start = start.view(1, 1, 1, 3)
        steps = action_candidates.shape[3] // ACTION_DIM
        dense = densify(start, path, self.interp)[..., :2]
        hazard = substep_violation(dense, self.boxes, clearances, steps * self.interp)
        ends = torch.cat([start.expand(B, S, 1, 3), path[:, :, steps - 1 :: steps]], dim=2)
        return hazard, ends

    def exact_violation(self, action_candidates):
        """Same violation as get_cost, (B, S), for the execute-time check of the elite mean."""
        return self.hazard(action_candidates)[0]

    def record_choice(self, kind, violation=None):
        """Annotate the solve's last iteration record with the executed plan's provenance."""
        if not self.history:
            return
        entry = {"executed": kind}
        if violation is not None:
            entry["executed_viol"] = float(violation)
        if kind == "best" and self._best and self._best[0] is not None:
            best = self._best[0]
            entry.update(
                {
                    "executed_feasible": best["feasible"],
                    "executed_viol": best["violation"],
                    "executed_goal": best["goal"],
                }
            )
        self.history[-1].update(entry)

    def get_cost(self, info_dict, action_candidates):
        goal_cost = self.base.get_cost(info_dict, action_candidates)
        self.queries += int(action_candidates.shape[0] * action_candidates.shape[1])
        with torch.no_grad():
            violation, ends = self.hazard(action_candidates)
            total, feasible = rank_costs(goal_cost, violation, self.mode, self.lam, self.tol)
            elite = total.argmin(dim=1)
            if self.track_best:
                self._track(goal_cost, violation, feasible, action_candidates, ends)
            self.history.append(
                {
                    "goal_min": float(goal_cost.min()),
                    "goal_median": float(goal_cost.median()),
                    "viol_median": float(violation.median()),
                    "viol_max": float(violation.max()),
                    "hazard_median": float(violation.median()),
                    "arena_median": 0.0,
                    "frac_feasible": float(feasible.float().mean()),
                    "elite_feasible": bool(feasible[0, elite[0]]),
                    "elite_viol": float(violation[0, elite[0]]),
                    "elite_goal": float(goal_cost[0, elite[0]]),
                    "elite_xy": ends[0, elite[0]].detach().cpu().tolist(),
                }
            )
        return total


def summarize_execution(history, n_steps):
    """How often the elite mean was executed versus the best checked candidate."""
    finals = [h for h in history[n_steps - 1 :: n_steps] if "executed" in h]
    if not finals:
        return {}
    kinds = [h["executed"] for h in finals]
    return {
        "replans_with_choice": len(finals),
        "mean_executed_frac": float(np.mean([k == "mean" for k in kinds])),
        "best_executed_frac": float(np.mean([k == "best" for k in kinds])),
        "executed_viol_mean": float(np.mean([h.get("executed_viol", 0.0) for h in finals])),
    }
