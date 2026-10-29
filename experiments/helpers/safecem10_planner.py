"""Constraint-priority CEM around the frozen LeWM Push-T world model, task 10.

Ported from the archived September pilot (archive/legacy/experiments/helpers/safeCEM.py)
with three extensions: several true hazard boxes, a clearance schedule d(t) over the
predicted blocks, and a feasibility tolerance. The upstream CEMSolver ranks candidates by
one scalar with torch.topk(largest=False), so constraint priority is encoded in the scalar:

    feasible:   cost = goal_cost
    infeasible: cost = (max feasible goal cost + 1) + violation

The arena constraint is computed in action space from the commanded PD targets, because a
probe trained on in-arena frames cannot see a pusher that has left the frame. The targets
overstate the travel (the PD controller lags them by about 100 px over a 25-step plan), so
this constraint is conservative; it is kept unchanged in every stage for comparability.

Three hazard checks (task 10, after the stage B diagnosis):
    probe     the September check: a probe reads the pusher from each imagined latent
    anchored  the same probe readings shifted so the current reading equals the true
              current position (only imagined displacements come from the world model)
    exact     the pusher path at every physics substep through the known linear PD
              controller of Push-T, from the true position and velocity; no perception
Optionally the cost model also tracks the best checked candidate of a whole CEM solve, so
the planner can execute it instead of the unchecked mean of the final elite set (iCEM).

Every function here is unit-testable without the world model; the LeWM dependency enters
only through `base_model.get_cost`, which must populate info_dict["predicted_emb"].
"""

from __future__ import annotations

import numpy as np
import torch

ARENA = (0.0, 512.0)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MLPProbe(torch.nn.Module):
    """LeWM latent (192) to pusher (x, y) in arena px; a generic physical probe."""

    def __init__(self, in_dim=192, hidden=(256, 128), out_dim=2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
            d = h
        layers.append(torch.nn.Linear(d, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def expand_box(box, margin):
    x0, x1, y0, y1 = box
    return (x0 - margin, x1 + margin, y0 - margin, y1 + margin)


def box_penetration(xy, box):
    """Depth inside an axis-aligned box, zero outside. xy (..., 2) -> (...)."""
    x0, x1, y0, y1 = box
    dx = torch.minimum(xy[..., 0] - x0, x1 - xy[..., 0])
    dy = torch.minimum(xy[..., 1] - y0, y1 - xy[..., 1])
    return torch.clamp(torch.minimum(dx, dy), min=0.0)


def arena_exit_depth(xy, arena=ARENA):
    lo, hi = arena
    below = torch.clamp(lo - xy, min=0.0)
    above = torch.clamp(xy - hi, min=0.0)
    return (below + above).sum(dim=-1)


def interpolate_path(xy, n_substeps):
    """xy (B, S, T, 2) -> (B, S, T-1, n_substeps, 2) straight segments between waypoints."""
    starts = xy[:, :, :-1, :].unsqueeze(-2)
    ends = xy[:, :, 1:, :].unsqueeze(-2)
    w = torch.linspace(0.0, 1.0, n_substeps, device=xy.device, dtype=xy.dtype)
    w = w.view(1, 1, 1, n_substeps, 1)
    return starts * (1.0 - w) + ends * w


def scheduled_boxes(boxes, clearances):
    """Per predicted block t, the true boxes inflated by clearances[t]."""
    return [[expand_box(b, d) for b in boxes] for d in clearances]


def hazard_violation(xy, boxes, clearances, n_substeps=5):
    """Summed penetration of the inflated boxes along the imagined pusher path.

    xy: (B, S, T, 2) probe positions; index 0 is the current frame and 1..T-1 are the
    predicted blocks. Segment t runs from block t to block t+1 and is checked against the
    boxes inflated by clearances[t], so a schedule has one entry per predicted block and
    a later, more uncertain block can carry a larger clearance. Returns (B, S).
    """
    B, S, T, _ = xy.shape
    segments = max(T - 1, 1)
    if len(clearances) < segments:
        raise ValueError("A clearance is needed for every predicted block")
    total = torch.zeros(B, S, device=xy.device, dtype=xy.dtype)
    if T == 1:
        for box in boxes:
            total = total + box_penetration(xy[:, :, 0], expand_box(box, clearances[0]))
        return total
    path = interpolate_path(xy, n_substeps)
    for t in range(T - 1):
        for box in boxes:
            inflated = expand_box(box, clearances[t])
            total = total + box_penetration(path[:, :, t], inflated).sum(dim=-1)
    return total


def commanded_positions(action_candidates, pusher_xy, action_mean, action_scale, gain=100.0):
    """Pusher positions implied by normalized action candidates, in arena px.

    Push-T sets the PD target to agent.position + action * 100 each env step and runs
    enough substeps to reach it, so the trajectory is exact in action space.
    action_candidates: (B, S, T, action_block * 2) normalized actions.
    Returns (B, S, T * action_block, 2).
    """
    B, S, T, D = action_candidates.shape
    a = action_candidates.reshape(B, S, T * (D // 2), 2)
    dev, dtype = a.device, a.dtype
    mean = torch.as_tensor(np.asarray(action_mean), device=dev, dtype=dtype).view(1, 1, 1, 2)
    scale = torch.as_tensor(np.asarray(action_scale), device=dev, dtype=dtype).view(1, 1, 1, 2)
    raw = a * scale + mean
    p0 = torch.as_tensor(np.asarray(pusher_xy), device=dev, dtype=dtype).view(1, 1, 1, 2)
    return p0 + torch.cumsum(raw * gain, dim=2)


def pd_response(n_env_steps, k_p=100.0, k_v=20.0, dt=0.01, substeps=10, action_scale=100.0):
    """Exact linear response of the Push-T pusher to relative actions, per axis.

    Each env step sets target = position + action_scale * action and runs `substeps`
    semi-implicit Euler steps of v += dt * (k_p * (target - x) - k_v * v); x += dt * v on a
    kinematic body, so collisions never move the pusher. The map is linear and shift
    invariant: x[m] = x0 + c[m] * v0 + G[m] @ a_raw for the M = n_env_steps * substeps
    physics substeps. Returns (c (M,), G (M, n_env_steps)) as float64 arrays.
    """
    M = n_env_steps * substeps

    def simulate(v0, u):
        x, v = 0.0, float(v0)
        out = np.zeros(M)
        for j in range(n_env_steps):
            target = x + action_scale * u[j]
            for k in range(substeps):
                v = v + dt * (k_p * (target - x) - k_v * v)
                x = x + dt * v
                out[j * substeps + k] = x
        return out

    c = simulate(1.0, np.zeros(n_env_steps))
    eye = np.eye(n_env_steps)
    G = np.stack([simulate(0.0, eye[j]) for j in range(n_env_steps)], axis=1)
    return c, G


def substep_violation(path, boxes, clearances, per_block):
    """Summed penetration along a substep path (B, S, M, 2); block t covers substeps
    [t * per_block, (t + 1) * per_block) and is checked against clearances[t]. Returns (B, S)."""
    B, S, M, _ = path.shape
    blocks = M // per_block
    if len(clearances) < blocks:
        raise ValueError("A clearance is needed for every predicted block")
    total = torch.zeros(B, S, device=path.device, dtype=path.dtype)
    for t in range(blocks):
        part = path[:, :, t * per_block : (t + 1) * per_block]
        for box in boxes:
            total = total + box_penetration(part, expand_box(box, clearances[t])).sum(dim=-1)
    return total


def rank_costs(goal_cost, violation, mode, lam=0.0, tol=0.0):
    """The scalar the CEM solver minimizes, for either ranking rule."""
    if mode == "penalty":
        return goal_cost + lam * violation, violation <= tol
    feasible = violation <= tol
    big = (goal_cost[feasible].max() if feasible.any() else goal_cost.max()) + 1.0
    return torch.where(feasible, goal_cost, big + violation), feasible


CHECKS = ("probe", "anchored", "exact")
CONTROLLER = {"k_p": 100.0, "k_v": 20.0, "dt": 0.01, "substeps": 10, "action_scale": 100.0}


class DialCostModel(torch.nn.Module):
    """Goal cost from the frozen world model plus a clearance constraint on the pusher."""

    def __init__(
        self,
        base_model,
        probe,
        boxes,
        clearances,
        *,
        mode="safe",
        lam=0.0,
        tol=0.0,
        n_substeps=5,
        arena_constraint=True,
        action_mean=None,
        action_scale=None,
        arena=ARENA,
        check="probe",
        track_best=False,
        controller=None,
    ):
        super().__init__()
        if mode not in ("safe", "penalty"):
            raise ValueError("mode must be safe or penalty")
        if check not in CHECKS:
            raise ValueError(f"check must be one of {CHECKS}")
        self.base = base_model
        self.probe = probe
        self.boxes = [tuple(float(v) for v in b) for b in boxes]
        self.clearances = [float(d) for d in clearances]
        self.mode = mode
        self.lam = float(lam)
        self.tol = float(tol)
        self.n_substeps = int(n_substeps)
        self.arena_constraint = bool(arena_constraint)
        self.action_mean = action_mean
        self.action_scale = action_scale
        self.arena = tuple(arena)
        self.check = check
        self.track_best = bool(track_best)
        self.controller = dict(CONTROLLER if controller is None else controller)
        self.pusher_xy = None
        self.pusher_vel = None
        self.history = []
        self.queries = 0
        self._best = None
        self._pd_cache = {}

    def _pd(self, n_env_steps, device, dtype):
        key = (n_env_steps, str(device), dtype)
        if key not in self._pd_cache:
            c, G = pd_response(n_env_steps, **self.controller)
            self._pd_cache[key] = (
                torch.as_tensor(c, device=device, dtype=dtype),
                torch.as_tensor(G, device=device, dtype=dtype),
            )
        return self._pd_cache[key]

    def controller_path(self, action_candidates):
        """Pusher centre at every physics substep, (B, S, T * action_block * substeps, 2)."""
        if self.action_mean is None or self.pusher_xy is None:
            raise RuntimeError("The exact check needs the action scaler and the true position")
        B, S, T, D = action_candidates.shape
        steps = T * (D // 2)
        dev, dtype = action_candidates.device, action_candidates.dtype
        c, G = self._pd(steps, dev, dtype)
        a = action_candidates.reshape(B, S, steps, 2)
        mean = torch.as_tensor(np.asarray(self.action_mean), device=dev, dtype=dtype)
        scale = torch.as_tensor(np.asarray(self.action_scale), device=dev, dtype=dtype)
        raw = a * scale.view(1, 1, 1, 2) + mean.view(1, 1, 1, 2)
        p0 = torch.as_tensor(np.asarray(self.pusher_xy), device=dev, dtype=dtype).view(1, 1, 1, 2)
        vel = np.zeros(2) if self.pusher_vel is None else np.asarray(self.pusher_vel)
        v0 = torch.as_tensor(vel, device=dev, dtype=dtype).view(1, 1, 1, 2)
        return p0 + c.view(1, 1, -1, 1) * v0 + torch.einsum("mn,bsnk->bsmk", G, raw)

    def begin_solve(self):
        """Forget the best candidate; called by the solver at the start of every replan."""
        self._best = None

    def _track(self, goal_cost, violation, feasible, action_candidates, xy):
        """Best candidate of the whole solve under the planner's own ranking: feasible before
        infeasible, then lower goal cost (feasible) or lower violation (infeasible). Goal
        costs of one solve share the observation, so they compare across CEM iterations."""
        B = goal_cost.shape[0]
        if self._best is None:
            self._best = [None] * B
        for b in range(B):
            if self.mode == "penalty":
                score = goal_cost[b] + self.lam * violation[b]
                i = int(torch.argmin(score))
                key = (0, float(score[i]))
            elif bool(feasible[b].any()):
                inf = torch.full_like(goal_cost[b], float("inf"))
                i = int(torch.argmin(torch.where(feasible[b], goal_cost[b], inf)))
                key = (0, float(goal_cost[b, i]))
            else:
                i = int(torch.argmin(violation[b]))
                key = (1, float(violation[b, i]))
            if self._best[b] is None or key < self._best[b]["key"]:
                self._best[b] = {
                    "key": key,
                    "actions": action_candidates[b, i].detach().clone(),
                    "xy": xy[b, i].detach().cpu().tolist(),
                    "feasible": bool(feasible[b, i]),
                    "violation": float(violation[b, i]),
                    "goal": float(goal_cost[b, i]),
                }

    def best_actions(self):
        """(B, T, D) best candidates of the current solve on the CPU, or None."""
        if not self._best or any(entry is None for entry in self._best):
            return None
        return torch.stack([entry["actions"] for entry in self._best]).detach().cpu()

    def record_execution(self):
        """Annotate the solve's last iteration record with the plan that will be executed."""
        if self._best and self._best[0] is not None and self.history:
            best = self._best[0]
            self.history[-1].update(
                {
                    "executed": "best",
                    "executed_feasible": best["feasible"],
                    "executed_viol": best["violation"],
                    "executed_goal": best["goal"],
                    "executed_xy": best["xy"],
                }
            )

    def get_cost(self, info_dict, action_candidates):
        goal_cost = self.base.get_cost(info_dict, action_candidates)
        self.queries += int(action_candidates.shape[0] * action_candidates.shape[1])
        hist_len = info_dict["pixels"].shape[2]
        with torch.no_grad():
            if self.check == "exact":
                horizon = int(action_candidates.shape[2])
                clearances = self.clearances + [self.clearances[-1]] * max(
                    0, horizon - len(self.clearances)
                )
                path = self.controller_path(action_candidates)
                per_block = (action_candidates.shape[3] // 2) * int(self.controller["substeps"])
                hazard = substep_violation(path, self.boxes, clearances, per_block)
                B, S = path.shape[:2]
                start = torch.as_tensor(
                    np.asarray(self.pusher_xy), device=path.device, dtype=path.dtype
                ).view(1, 1, 1, 2)
                ends = path[:, :, per_block - 1 :: per_block]
                xy = torch.cat([start.expand(B, S, 1, 2), ends], dim=2)
            else:
                embs = info_dict["predicted_emb"][:, :, max(hist_len - 1, 0) :, :]
                B, S, T, D = embs.shape
                probe = self.probe
                flat = embs.reshape(B * S * T, D).float()
                xy = probe(flat).reshape(B, S, T, 2)
                if self.check == "anchored":
                    if self.pusher_xy is None:
                        raise RuntimeError("The anchored check needs the true position")
                    true = torch.as_tensor(
                        np.asarray(self.pusher_xy), device=xy.device, dtype=xy.dtype
                    ).view(1, 1, 1, 2)
                    xy = xy - xy[:, :, :1, :] + true
                needed = max(T - 1, 1)
                clearances = self.clearances + [self.clearances[-1]] * max(
                    0, needed - len(self.clearances)
                )
                hazard = hazard_violation(xy, self.boxes, clearances, self.n_substeps)
            if self.arena_constraint:
                if self.action_mean is None or self.pusher_xy is None:
                    raise RuntimeError("Action-space arena constraint needs scaler and position")
                pos = commanded_positions(
                    action_candidates, self.pusher_xy, self.action_mean, self.action_scale
                )
                arena_pen = arena_exit_depth(pos, self.arena).sum(dim=-1)
            else:
                arena_pen = torch.zeros_like(hazard)
            violation = hazard + arena_pen
            total, feasible = rank_costs(goal_cost, violation, self.mode, self.lam, self.tol)
            elite = total.argmin(dim=1)
            if self.track_best:
                self._track(goal_cost, violation, feasible, action_candidates, xy)
            self.history.append(
                {
                    "goal_min": float(goal_cost.min()),
                    "goal_median": float(goal_cost.median()),
                    "viol_median": float(violation.median()),
                    "viol_max": float(violation.max()),
                    "hazard_median": float(hazard.median()),
                    "arena_median": float(arena_pen.median()),
                    "frac_feasible": float(feasible.float().mean()),
                    "elite_feasible": bool(feasible[0, elite[0]]),
                    "elite_viol": float(violation[0, elite[0]]),
                    "elite_goal": float(goal_cost[0, elite[0]]),
                    "elite_xy": xy[0, elite[0]].detach().cpu().tolist(),
                }
            )
        return total


def image_transform():
    """ImageNet normalization at 224 px, matching the released LeWM preprocessing."""
    from torchvision.transforms import v2

    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )


def make_state(pusher_xy, block_xy, block_theta=0.0, velocity=(0.0, 0.0)):
    px, py = pusher_xy
    bx, by = block_xy
    vx, vy = velocity
    return np.array([px, py, bx, by, block_theta, vx, vy], dtype=np.float64)


def summarize_history(history, n_steps):
    if not history:
        return {}
    finals = history[n_steps - 1 :: n_steps]
    return {
        "cem_iterations": len(history),
        "replans": len(finals),
        "frac_feasible_mean": float(np.mean([h["frac_feasible"] for h in history])),
        "final_frac_feasible": float(np.mean([h["frac_feasible"] for h in finals])),
        "final_elite_feasible": float(np.mean([h["elite_feasible"] for h in finals])),
        "final_elite_viol_mean": float(np.mean([h["elite_viol"] for h in finals])),
        "final_elite_goal_mean": float(np.mean([h["elite_goal"] for h in finals])),
    }
