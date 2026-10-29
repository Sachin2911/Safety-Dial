"""SafeCEM on TwoRoom (task 12): the task-10 constraint-priority cost model with an exact
check built from TwoRoom's own kinematics.

TwoRoom moves the agent by speed * clip(action, -1, 1) per env step and then applies the
simulator's collision rule: clamp to the border (allowing for the agent radius) and, when
the move would enter the central wall outside the door, clamp to the wall face. Both are
deterministic, so the agent centre after every env step of a candidate plan is known
exactly from the current true position; `tworoom_path` reproduces
stable_worldmodel.envs.two_room.env.TwoRoomEnv._apply_collisions step for step, vectorized
over candidates. No probe and no world-model imagination enter the check.

Execution modes (solver wrappers live in tworoom12_runtime):
    mean       the upstream elite mean, which the check never evaluates
    best       the best checked candidate of the solve (iCEM style; hurt in task-10 B2)
    mean_safe  the elite mean if it passes the same exact check, else the best checked
               candidate; the mean hedges world-model error and the check verifies it
"""

from __future__ import annotations

import numpy as np
import torch

from helpers.safecem10_planner import DialCostModel, substep_violation

EXECUTE_MODES = ("mean", "best", "mean_safe")


def raw_actions(action_candidates, action_mean, action_scale):
    """(B, S, T, action_block * 2) normalized -> (B, S, T * action_block, 2) env actions."""
    B, S, T, D = action_candidates.shape
    a = action_candidates.reshape(B, S, T * (D // 2), 2)
    dev, dtype = a.device, a.dtype
    mean = torch.as_tensor(np.asarray(action_mean), device=dev, dtype=dtype).view(1, 1, 1, 2)
    scale = torch.as_tensor(np.asarray(action_scale), device=dev, dtype=dtype).view(1, 1, 1, 2)
    return a * scale + mean


def tworoom_path(start_xy, raw, geometry):
    """Agent centre after each env step: (B, S, M, 2) from raw actions (B, S, M, 2).

    Mirrors TwoRoomEnv.step for a vertical wall: pos2 = pos + clip(a) * speed; border clamp
    with the agent radius; if the move started on one side of the wall centre and would
    cross that side's effective face (wall face plus radius) while the centre is outside
    the door span plus margin, the x coordinate is set to the face minus (plus) 0.5.
    """
    g = geometry
    speed = float(g["speed"])
    radius = float(g["agent_radius"])
    border = float(g["border"])
    size = float(g["image_size"])
    centre = float(g["wall_center"])
    half = int(g["wall_thickness"]) // 2
    lo, hi = border + radius, size - border - radius
    face_left = centre - half - radius
    face_right = centre + half + radius
    door_lo = float(g["door_center"]) - float(g["door_half"]) - float(g["door_margin"])
    door_hi = float(g["door_center"]) + float(g["door_half"]) + float(g["door_margin"])
    a = raw.clamp(-1.0, 1.0)
    B, S, M, _ = a.shape
    p = torch.as_tensor(np.asarray(start_xy, dtype=float), device=a.device, dtype=a.dtype)
    x = p[0].expand(B, S).clone()
    y = p[1].expand(B, S).clone()
    out = torch.empty(B, S, M, 2, device=a.device, dtype=a.dtype)
    for m in range(M):
        x2 = (x + a[:, :, m, 0] * speed).clamp(lo, hi)
        y2 = (y + a[:, :, m, 1] * speed).clamp(lo, hi)
        started_left = x < centre
        in_door = (y2 >= door_lo) & (y2 <= door_hi)
        hit_left = started_left & (x2 > face_left) & ~in_door
        hit_right = (~started_left) & (x2 < face_right) & ~in_door
        x2 = torch.where(hit_left, torch.full_like(x2, face_left - 0.5), x2)
        x2 = torch.where(hit_right, torch.full_like(x2, face_right + 0.5), x2)
        x, y = x2, y2
        out[:, :, m, 0] = x
        out[:, :, m, 1] = y
    return out


def replay_error(states, start_xy, actions, geometry):
    """Gap between saved agent positions and the exact model replaying executed actions."""
    states = np.asarray(states, dtype=np.float64)[:, :2]
    acts = torch.as_tensor(np.asarray(actions, dtype=np.float64)).view(1, 1, -1, 2)
    path = tworoom_path(start_xy, acts, geometry)[0, 0].numpy()
    n = min(len(path), len(states))
    err = np.linalg.norm(path[:n] - states[:n], axis=1)
    return {
        "median": float(np.median(err)) if n else None,
        "max": float(err.max()) if n else None,
        "steps": int(n),
    }


class TwoRoomCostModel(DialCostModel):
    """World-model goal cost plus a clearance constraint checked on the exact agent path."""

    def __init__(
        self,
        base_model,
        boxes,
        clearances,
        *,
        geometry,
        action_mean,
        action_scale,
        mode="safe",
        lam=0.0,
        tol=0.0,
        track_best=False,
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
            controller={"substeps": 1},
        )
        self.geometry = dict(geometry)

    def controller_path(self, action_candidates):
        """Agent centre after every env step of every candidate, (B, S, T * block, 2)."""
        if self.pusher_xy is None:
            raise RuntimeError("The exact check needs the true agent position")
        raw = raw_actions(action_candidates, self.action_mean, self.action_scale)
        return tworoom_path(self.pusher_xy, raw, self.geometry)

    def exact_violation(self, action_candidates):
        """Summed penetration of the scheduled inflated boxes, (B, S); as in get_cost."""
        horizon = int(action_candidates.shape[2])
        clearances = self.clearances + [self.clearances[-1]] * max(
            0, horizon - len(self.clearances)
        )
        path = self.controller_path(action_candidates)
        per_block = action_candidates.shape[3] // 2
        return substep_violation(path, self.boxes, clearances, per_block)

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
