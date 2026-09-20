"""Safe CEM and the Safety Dial for Push-T.

Why this module exists
----------------------
`hazardSweep.py` sweeps a penalty weight lambda in a scalarised cost
`C = goal + lambda * hazard`. That sweep collapsed: every non-zero lambda abandoned the
task and drove the pusher out of the arena. Three separate causes, all fixed here.

1. THE START WAS INSIDE THE INFLATED HAZARD. `calibrate_box` picks a box on the
   baseline path and never checks that the start clears it once the probe margin is
   added. The box it chose, (2, 82, 83, 163), inflates to (-33.5, 117.5, 47.5, 198.5),
   which contains the pusher start (70, 70). Every candidate plan was therefore in
   violation from step zero, so the penalty was a near-constant offset that the planner
   could only reduce by leaving. `place_hazard_between` guarantees clearance instead.

2. THE PLANNER COULD ESCAPE. The Push-T pusher is a `pymunk.Body(KINEMATIC)` driven by
   a PD controller toward `agent.position + action * 100`, and the arena clip in
   `env.step` is commented out. Kinematic bodies ignore collisions, so the pusher passes
   straight through the walls: fleeing the arena drives any hazard penalty to zero at no
   cost. Verified directly, the pusher reaches (-1381, -1313) and returns. So "safety"
   can be bought by escape rather than avoidance.

   Measured correction, worth stating plainly: with cause 1 fixed the escape stops being
   the dominant failure. Across dials 0 to 40 and every lambda tested, out-of-bounds was
   exactly 0.000, so a penalty is NOT inherently fragile here. The escape reappears only
   at dial 60, where the constraint becomes tight enough that leaving is the cheapest way
   to satisfy it. Treat the original "penalty fragility" sweep as confounded by cause 1.

   The constraint that stops it must be computed in ACTION space, not from `probe(z_hat)`.
   See `commanded_positions`: a probe trained on in-arena frames cannot see a pusher that
   has left the frame, so it cannot police its own validity domain.

3. A WEIGHT IS NOT A THRESHOLD. With a scalarised cost there is no notion of a feasible
   plan, so lambda has no units and no operating-point meaning. Safe CEM ranks
   feasibility first and only then goal progress, which makes the dial a clearance `d`
   in pixels that an operator sets at deployment.

Safe CEM
--------
`CEMSolver` picks elites with `torch.topk(costs, largest=False)`, a pure scalar ranking,
so constraint-priority ranking can be encoded in the scalar itself and the upstream
solver needs no modification:

    feasible:   cost = goal_cost
    infeasible: cost = (max feasible goal cost + 1) + violation

Every feasible candidate then sorts ahead of every infeasible one, feasible candidates
sort by goal progress, and infeasible candidates sort by violation magnitude so the
distribution still has a gradient to descend toward feasibility when nothing is safe
yet. This is the constraint-priority ranking of MPC-RCE (Liu et al., Constrained
Model-based RL with Robust Cross-Entropy Method), which is already in the reading list
as `docs/papers/myPapers/MPC-RCE.pdf`.

Keeping the violation term graded rather than binary matters: CEM sets
`var = topk_candidates.std(dim=1)`, so a constant cost over all candidates makes the
elite set arbitrary and the search collapses.
"""

from __future__ import annotations

import numpy as np
import torch

from helpers.linProbeHelpers import box_penetration, expand_box, interpolate_path

ARENA_LO = 0.0
ARENA_HI = 512.0


# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------


def arena_exit_depth(xy, lo: float = ARENA_LO, hi: float = ARENA_HI):
    """How far outside the arena a point is, in px. Zero inside. xy: (..., 2)."""
    below = torch.clamp(lo - xy, min=0.0)
    above = torch.clamp(xy - hi, min=0.0)
    return (below + above).sum(dim=-1)


def commanded_positions(action_candidates, pusher_xy, action_scaler, *, action_scale=100.0):
    """Pusher positions implied by a candidate action sequence, computed in ACTION space.

    Push-T's pusher is position-controlled: `env.step` sets the PD target to
    `agent.position + action * 100` (`action_scale = 100`) and runs 10 substeps, which is
    enough to reach it. So the commanded trajectory is an exact function of the action
    sequence and the current position. No world model, no probe, no learned error.

    This exists because an arena constraint read off `probe(z_hat)` CANNOT WORK. The probe
    is trained on in-arena frames, and once the pusher truly leaves the 512 px arena it is
    not visible in the 224 px frame at all, so no encoder output carries its position and
    the probe's estimate is uninformative. Measured: at dial 60 the planner reported 80.6%
    of candidates feasible while driving the pusher to x = -276, y = -562 on every seed,
    with the block untouched. A conservative dial actively pushes the planner toward that
    blind spot, because escaping is the cheapest way to satisfy a tight hazard constraint.

    action_candidates: (B, S, T, action_block * 2), normalised by the planner's
        StandardScaler for the action column.
    pusher_xy: (2,) current pusher position in arena px (unnormalised).
    action_scaler: the fitted sklearn StandardScaler in `process["action"]`.

    Returns (B, S, T * action_block, 2) of commanded positions in arena px.
    """
    B, S, T, D = action_candidates.shape
    n_env_steps = D // 2
    a = action_candidates.reshape(B, S, T * n_env_steps, 2)

    dev, dtype = a.device, a.dtype
    mean = torch.as_tensor(action_scaler.mean_, device=dev, dtype=dtype).view(1, 1, 1, 2)
    scale = torch.as_tensor(action_scaler.scale_, device=dev, dtype=dtype).view(1, 1, 1, 2)
    a_raw = a * scale + mean

    p0 = torch.as_tensor(pusher_xy, device=dev, dtype=dtype).view(1, 1, 1, 2)
    return p0 + torch.cumsum(a_raw * action_scale, dim=2)


def path_violation(
    predicted_emb,
    hist_len,
    probe,
    hazard_box,
    *,
    dial: float = 0.0,
    n_substeps: int = 5,
    arena_constraint: bool = True,
    arena=(ARENA_LO, ARENA_HI),
):
    """Constraint violation of each action candidate, in px.

    Returns (violation, hazard_part, arena_part, xy) with the three scalars shaped
    (B, S) and xy shaped (B, S, T, 2).

    The hazard box is inflated by `dial`, the deployment-time clearance. Violation is
    summed over interpolated path substeps so a plan that steps over the box between two
    waypoints is still caught.
    """
    start = max(hist_len - 1, 0)
    embs = predicted_emb[:, :, start:, :]
    B, S, T, D = embs.shape

    flat = embs.reshape(B * S * T, D).float()
    if hasattr(probe, "parameters"):
        p_dev = next(probe.parameters()).device
        if p_dev != flat.device:
            probe = probe.to(flat.device)
    xy = probe(flat).reshape(B, S, T, 2)

    box = expand_box(hazard_box, dial)

    if T == 1:
        hazard = box_penetration(xy, box).sum(dim=-1)
        arena_pen = arena_exit_depth(xy, *arena).sum(dim=-1)
    else:
        path = interpolate_path(xy, n_substeps)  # (B, S, T-1, n_substeps, 2)
        hazard = box_penetration(path, box).sum(dim=(2, 3))
        arena_pen = arena_exit_depth(path, *arena).sum(dim=(2, 3))

    violation = hazard + arena_pen if arena_constraint else hazard
    return violation, hazard, arena_pen, xy


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------


class DialCostModel(torch.nn.Module):
    """One cost model, two ranking rules, so the arms are directly comparable.

    mode="safe"    : constraint-priority ranking (Safe CEM). `dial` is the clearance.
    mode="penalty" : legacy scalarised cost `goal + lam * violation`.

    Both arms compute the identical constraint, so any difference between them is the
    ranking rule and not the constraint definition.
    """

    def __init__(
        self,
        base_model,
        probe,
        hazard_box,
        *,
        mode: str = "safe",
        dial: float = 0.0,
        lam: float = 0.0,
        n_substeps: int = 5,
        arena_constraint: bool = True,
        feasible_tol: float = 0.0,
        action_scaler=None,
        action_scale: float = 100.0,
    ):
        super().__init__()
        if mode not in ("safe", "penalty"):
            raise ValueError(f"mode must be 'safe' or 'penalty', got {mode!r}")
        self.base = base_model
        self.probe = probe
        self.hazard_box = tuple(map(float, hazard_box))
        self.mode = mode
        self.dial = float(dial)
        self.lam = float(lam)
        self.n_substeps = int(n_substeps)
        self.arena_constraint = bool(arena_constraint)
        self.feasible_tol = float(feasible_tol)
        # When given, the arena constraint is computed in action space rather than from
        # probe(z_hat). See commanded_positions for why the probe cannot do this job.
        self.action_scaler = action_scaler
        self.action_scale = float(action_scale)
        self.cost_history = []
        self.last_xy = None
        self.pusher_xy = None

    def get_cost(self, info_dict, action_candidates):
        # base.get_cost runs the rollout and populates info_dict["predicted_emb"],
        # so it must be called before the constraint is read.
        goal_cost = self.base.get_cost(info_dict, action_candidates)

        hist_len = info_dict["pixels"].shape[2]
        with torch.no_grad():
            violation, hazard, arena_pen, xy = path_violation(
                info_dict["predicted_emb"],
                hist_len,
                self.probe,
                self.hazard_box,
                dial=self.dial,
                n_substeps=self.n_substeps,
                arena_constraint=False,  # replaced below by the action-space constraint
            )
            if self.arena_constraint:
                if self.action_scaler is not None and self.pusher_xy is not None:
                    pos = commanded_positions(
                        action_candidates,
                        self.pusher_xy,
                        self.action_scaler,
                        action_scale=self.action_scale,
                    )
                    arena_pen = arena_exit_depth(pos).sum(dim=-1)
                else:
                    # Fallback: the probe-based check. Known to be unable to detect a
                    # genuine arena exit; kept only so the old behaviour is reproducible.
                    arena_pen = arena_exit_depth(xy).sum(dim=-1)
                violation = violation + arena_pen

        feasible = violation <= self.feasible_tol

        if self.mode == "penalty":
            total = goal_cost + self.lam * violation
        else:
            # Any offset above the largest feasible goal cost separates the two tiers.
            # Fall back to the global max when nothing is feasible this iteration.
            if feasible.any():
                big = goal_cost[feasible].max()
            else:
                big = goal_cost.max()
            big = big + 1.0
            total = torch.where(feasible, goal_cost, big + violation)

        elite = total.argmin(dim=1)
        self.last_xy = xy[0, elite[0]].detach().cpu()

        self.cost_history.append(
            {
                "goal_min": float(goal_cost.min()),
                "goal_median": float(goal_cost.median()),
                "viol_min": float(violation.min()),
                "viol_median": float(violation.median()),
                "viol_max": float(violation.max()),
                "hazard_median": float(hazard.median()),
                "arena_median": float(arena_pen.median()),
                "frac_feasible": float(feasible.float().mean()),
                "elite_feasible": bool(feasible[0, elite[0]]),
                "elite_viol": float(violation[0, elite[0]]),
                "elite_goal": float(goal_cost[0, elite[0]]),
            }
        )
        return total


# ---------------------------------------------------------------------------
# hazard geometry: a box genuinely between start and goal, with guaranteed clearance
# ---------------------------------------------------------------------------


def place_hazard_between(
    start_pusher_xy,
    goal_pusher_xy,
    *,
    half: float = 40.0,
    frac: float = 0.5,
    max_dial: float = 0.0,
    clearance: float = 5.0,
):
    """A square hazard on the segment between start and goal pusher positions.

    Guarantees the start and the goal both sit at least `clearance` px outside the box
    once it is inflated by `max_dial`. This is the check `calibrate_box` lacks and whose
    absence made the previous sweep start every episode in violation.

    Raises ValueError if no box of this size can separate the endpoints, which is the
    honest failure: pick a smaller `half` or a smaller `max_dial`.
    """
    s = np.asarray(start_pusher_xy, dtype=float)
    g = np.asarray(goal_pusher_xy, dtype=float)
    centre = s + frac * (g - s)

    box = (centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half)
    need = max_dial + clearance
    inflated = expand_box(box, need)

    def inside(p, b):
        return b[0] <= p[0] <= b[1] and b[2] <= p[1] <= b[3]

    for name, p in (("start", s), ("goal", g)):
        if inside(p, inflated):
            raise ValueError(
                f"{name} pusher {tuple(p)} lies inside the hazard box inflated by "
                f"max_dial+clearance={need:.1f} -> {tuple(round(v, 1) for v in inflated)}. "
                f"Reduce half (now {half}) or max_dial (now {max_dial})."
            )
    return box


def calibrate_box_on_path(
    states,
    start_pusher_xy,
    goal_pusher_xy,
    *,
    max_dial: float,
    clearance: float = 5.0,
    halves=(25.0, 30.0, 35.0, 40.0, 50.0),
    fracs=(0.3, 0.4, 0.5, 0.6, 0.7),
    target=(0.10, 0.40),
    entity: str = "pusher",
    verbose: bool = True,
):
    """Place the hazard on the route the unconstrained planner ACTUALLY takes, while
    keeping start and goal clear of the box inflated by the widest dial.

    Both existing helpers get half of this right and the other half wrong.
    `hazardSweep.calibrate_box` places the box on the observed baseline path, so there is
    genuinely something to avoid, but never checks that the start clears the inflated box,
    which is how the earlier sweep ended up starting every episode in violation.
    `place_hazard_between` guarantees that clearance but places the box on the straight
    line between start and goal, which the planner need not follow: measured directly, a
    box placed that way had a baseline violation fraction of 0.00, so the dial sweep had
    nothing to measure.

    This function requires both conditions at once and returns the candidate whose
    baseline violation sits closest to the middle of `target`.
    """
    from helpers.linProbeHelpers import real_violation_stats

    s = np.asarray(start_pusher_xy, float)
    g = np.asarray(goal_pusher_xy, float)
    col = slice(0, 2) if entity == "pusher" else slice(2, 4)
    path = states[:, col]
    need = max_dial + clearance

    def inside(p, b):
        return b[0] <= p[0] <= b[1] and b[2] <= p[1] <= b[3]

    cands = []
    for frac in fracs:
        centre = path[int(np.clip(frac, 0.0, 0.999) * len(path))]
        for half in halves:
            box = (centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half)
            infl = expand_box(box, need)
            clear = not inside(s, infl) and not inside(g, infl)
            viol = real_violation_stats(states, box, entity)["frac_violating"]
            cands.append(
                {
                    "frac": frac,
                    "half": half,
                    "box": tuple(float(v) for v in box),
                    "baseline_viol": float(viol),
                    "clear": bool(clear),
                    "in_target": bool(target[0] <= viol <= target[1]),
                }
            )

    if verbose:
        print(f"  {'frac':>5} {'half':>5} {'base viol':>10} {'clear':>6} {'usable':>7}  box")
        for c in sorted(cands, key=lambda d: -d["baseline_viol"]):
            usable = c["clear"] and c["in_target"]
            print(
                f"  {c['frac']:>5.2f} {c['half']:>5.0f} {c['baseline_viol']:>10.2f} "
                f"{str(c['clear']):>6} {str(usable):>7}  "
                f"{tuple(round(v) for v in c['box'])}"
            )

    usable = [c for c in cands if c["clear"] and c["in_target"]]
    pool = usable or [c for c in cands if c["clear"]]
    if not pool:
        raise ValueError(
            "no hazard placement both clears start/goal at the widest dial and lies on "
            "the baseline path. Reduce max_dial or the half-sizes."
        )
    mid = 0.5 * (target[0] + target[1])
    best = min(pool, key=lambda c: abs(c["baseline_viol"] - mid))
    if verbose:
        print(
            f"  chosen: box={tuple(round(v) for v in best['box'])} "
            f"baseline_viol={best['baseline_viol']:.2f} "
            f"(from {'target range' if usable else 'clear-only fallback'})"
        )
    return best["box"], cands


def hazard_clearance_report(start_pusher_xy, goal_pusher_xy, hazard_box, dials):
    """Per-dial clearance of start and goal from the inflated box. Diagnostic only."""
    s = np.asarray(start_pusher_xy, float)
    g = np.asarray(goal_pusher_xy, float)
    rows = []
    for d in dials:
        b = expand_box(hazard_box, d)

        def depth(p):
            dx = min(p[0] - b[0], b[1] - p[0])
            dy = min(p[1] - b[2], b[3] - p[1])
            return min(dx, dy)

        rows.append(
            {
                "dial": d,
                "inflated": tuple(round(v, 1) for v in b),
                "start_depth": round(float(depth(s)), 1),
                "goal_depth": round(float(depth(g)), 1),
                "start_inside": bool(depth(s) > 0),
                "goal_inside": bool(depth(g) > 0),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# episode runner
# ---------------------------------------------------------------------------


def run_episode_dial(
    base_model,
    probe,
    hazard_box,
    *,
    swm,
    process,
    transform,
    start_state,
    goal_state,
    mode="safe",
    dial=0.0,
    lam=0.0,
    n_substeps=5,
    arena_constraint=True,
    action_space_arena=True,
    seed=0,
    num_samples=300,
    n_steps=30,
    topk=30,
    var_scale=1.0,
    horizon=5,
    receding_horizon=1,
    action_block=5,
    eval_budget=50,
    keep_frames=False,
    device="cuda",
):
    """One episode under either ranking rule. Returns metrics plus diagnostics.

    `base_model` must be loaded once by the caller; reloading LeWM per episode
    dominated runtime in the earlier sweep.
    """
    from stable_worldmodel.solver import CEMSolver

    from helpers.hazardSweep import episode_metrics, make_on_step_v2

    # CEMSolver(seed=) does not control every RNG, so seed globally too.
    torch.manual_seed(seed)
    np.random.seed(seed)

    cost_model = (
        DialCostModel(
            base_model,
            probe,
            hazard_box,
            mode=mode,
            dial=dial,
            lam=lam,
            n_substeps=n_substeps,
            arena_constraint=arena_constraint,
            action_scaler=process.get("action") if action_space_arena else None,
        )
        .to(device)
        .eval()
    )

    solver = CEMSolver(
        model=cost_model,
        batch_size=1,
        num_samples=num_samples,
        var_scale=var_scale,
        n_steps=n_steps,
        topk=topk,
        device=device,
        seed=seed,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(
            horizon=horizon,
            receding_horizon=receding_horizon,
            action_block=action_block,
        ),
        process=process,
        transform=transform,
    )
    world = swm.World(
        "swm/PushT-v1",
        num_envs=1,
        image_shape=(224, 224),
        max_episode_steps=2 * eval_budget,
    )
    world.set_policy(policy)
    world.reset(seed=seed, options={"state": start_state, "goal_state": goal_state})

    frames, states, rewards = [], [], []

    # The action-space arena constraint needs the CURRENT pusher position to anchor the
    # commanded trajectory. Refresh it from the simulator each step (planning-time
    # proprio would do equally well; this is the same quantity, unnormalised).
    base_on_step = make_on_step_v2(frames, states, rewards)

    def on_step(w):
        base_on_step(w)
        cost_model.pusher_xy = np.asarray(states[-1][:2], dtype=float)

    cost_model.pusher_xy = np.asarray(start_state[:2], dtype=float)
    world._run(max_steps=eval_budget, mode="wait", on_step=on_step)
    states = np.stack(states)

    out = {"mode": mode, "dial": dial, "lam": lam, "seed": seed}
    # Violations are always measured against the TRUE box, never the inflated one.
    # episode_metrics already folds in real_violation_stats and out_of_bounds_stats.
    out.update(
        episode_metrics(states, rewards, goal_state, hazard_box, bool(world.terminateds[0]))
    )

    hist = cost_model.cost_history
    if hist:
        out["frac_feasible_mean"] = float(np.mean([h["frac_feasible"] for h in hist]))
        out["elite_feasible_frac"] = float(np.mean([h["elite_feasible"] for h in hist]))
        out["elite_viol_mean"] = float(np.mean([h["elite_viol"] for h in hist]))
        # Last CEM iteration of each replan is what actually gets executed.
        finals = hist[n_steps - 1 :: n_steps]
        out["final_frac_feasible"] = float(np.mean([h["frac_feasible"] for h in finals])) if finals else np.nan
        out["final_elite_feasible"] = float(np.mean([h["elite_feasible"] for h in finals])) if finals else np.nan
    out["states"] = states
    out["cost_history"] = hist
    if keep_frames:
        out["frames"] = frames
    return out


def summarize_runs(rows, key="dial"):
    """Mean +- std per dial (or lambda) for the metrics the trade-off curve needs."""
    vals = sorted({r[key] for r in rows})
    header = (
        f"{key:>8}  {'viol frac':>13}  {'block err px':>14}  "
        f"{'reward':>13}  {'oob frac':>12}  {'feasible':>9}"
    )
    lines = [header, "-" * len(header)]
    table = []
    for v in vals:
        g = [r for r in rows if r[key] == v]
        row = {
            key: v,
            "viol_mean": float(np.mean([r["frac_violating"] for r in g])),
            "viol_std": float(np.std([r["frac_violating"] for r in g])),
            "block_mean": float(np.mean([r["final_block_err_px"] for r in g])),
            "block_std": float(np.std([r["final_block_err_px"] for r in g])),
            "reward_mean": float(np.nanmean([r["final_coverage"] for r in g])),
            "reward_std": float(np.nanstd([r["final_coverage"] for r in g])),
            "oob_mean": float(np.mean([r["oob_frac"] for r in g])),
            "feasible": float(np.nanmean([r.get("final_frac_feasible", np.nan) for r in g])),
            "n": len(g),
        }
        table.append(row)
        lines.append(
            f"{v:>8g}  {row['viol_mean']:>6.3f}+-{row['viol_std']:<5.3f}  "
            f"{row['block_mean']:>7.1f}+-{row['block_std']:<5.1f}  "
            f"{row['reward_mean']:>7.1f}+-{row['reward_std']:<5.1f}  "
            f"{row['oob_mean']:>12.3f}  {row['feasible']:>9.3f}"
        )
    text = "\n".join(lines)
    return table, text


def monotonicity(table, key="dial", metric="viol_mean"):
    """Spearman correlation of the dial against a metric, plus strict-monotone check."""
    from scipy.stats import spearmanr

    xs = [r[key] for r in table]
    ys = [r[metric] for r in table]
    rho = float(spearmanr(xs, ys).statistic) if len(set(ys)) > 1 else float("nan")
    diffs = np.diff(ys)
    return {
        "spearman": rho,
        "non_increasing": bool(np.all(diffs <= 1e-9)),
        "non_decreasing": bool(np.all(diffs >= -1e-9)),
    }
