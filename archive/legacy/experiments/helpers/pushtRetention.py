"""Fixed source-episode goal-reaching checks with dense outcomes and exact T coverage."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from helpers.pushtGeometry import T_LOCAL, polygons_from_env, t_polygons

RETENTION_MIN_BLOCKS = 50
RETENTION_MIN_EPISODES = 20
RETENTION_PROTOCOL = "paired-source-fixed-maximum-horizon-v2"
RETENTION_GOAL_COVERAGE = .95
TERMINATION_POLICY = "maximum horizon; early completion requires environment terminal and verified whole-T coverage >=0.95"


def _signed_area(vertices):
    p = np.asarray(vertices, dtype=float)
    if len(p) < 3:
        return 0.0
    return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)) / 2)


def _intersection_area(subject, clip):
    """Sutherland-Hodgman clipping of convex polygons, independent of vertex winding."""
    out = list(np.asarray(subject, dtype=float))
    orientation = 1 if _signed_area(clip) >= 0 else -1
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        if not out:
            return 0.0
        edge = b - a

        def side(p):
            v = p - a
            return orientation * (edge[0] * v[1] - edge[1] * v[0])

        source, out = out, []
        previous = source[-1]
        dp = side(previous)
        for current in source:
            dc = side(current)
            if (dc >= -1e-9) != (dp >= -1e-9) and abs(dp - dc) > 1e-12:
                out.append(previous + dp / (dp - dc) * (current - previous))
            if dc >= -1e-9:
                out.append(current)
            previous, dp = current, dc
    return abs(_signed_area(out))


def block_coverage(pose, goal_pose, local=T_LOCAL):
    """Area of current whole-T intersection with the goal, divided by goal area."""
    current, target = t_polygons(pose, local), t_polygons(goal_pose, local)
    area = sum(abs(_signed_area(p)) for p in target)
    intersection = sum(_intersection_area(a, b) for a in current for b in target)
    return float(np.clip(intersection / area, 0.0, 1.0))


def make_cases(splits, seed: int, n: int = 20):
    from helpers.branchBank import expert_pairs
    from helpers.pushtAssets import H5_PATH
    from helpers.pushtReplay import Root, idle_prefix

    if n < 1 or len(splits["roles"]["retention"]) < n:
        raise ValueError("not enough independent retention source episodes")
    pairs = expert_pairs(H5_PATH, splits["roles"]["retention"], np.random.default_rng(seed), n)
    return [Root(seed + i, pair["start"], pair["goal"], idle_prefix(),
                 root_id=f"retention-{i:03d}",
                 meta={"episode": pair["episode"], "t0": pair["t0"], "role": "retention"})
            for i, pair in enumerate(pairs)]


def case_identity(root) -> str:
    return hashlib.sha256(json.dumps(root.to_dict(), sort_keys=True).encode()).hexdigest()


def load_fixed_cases(path, splits):
    """Load the exact frozen E2 case file, preserving bytes and independent source roles."""
    from pathlib import Path
    from helpers.pushtReplay import Root

    data = Path(path).read_bytes()
    cases = [Root.from_dict(row) for row in json.loads(data)]
    episodes, ids = [r.meta.get("episode") for r in cases], [r.root_id for r in cases]
    allowed = set(splits["roles"]["retention"])
    if (len(cases) < RETENTION_MIN_EPISODES or len(set(episodes)) != len(episodes)
            or len(set(ids)) != len(ids) or not set(episodes) <= allowed
            or any(r.meta.get("role") != "retention" for r in cases)):
        raise ValueError("Frozen cases require at least 20 distinct retention source episodes and roots")
    return cases, hashlib.sha256(data).hexdigest()


def evaluate_retention(model, process, cases, *, n_blocks=50, device="cuda"):
    from helpers.acquisitionSafety import MeteredEnv
    from helpers.imagination import next_warm_start
    from helpers.plannerAudit import AuditedNominalPlanner
    from helpers.pushtReplay import StepLedger, make_env, reset_root, run_actions

    if n_blocks < 1:
        raise ValueError("retention horizon must be positive")
    ledger = StepLedger()
    env = MeteredEnv(make_env(), ledger, "goal_retention")
    planner = AuditedNominalPlanner(model, process, device)
    rows = []
    for root in cases:
        before = ledger.total
        case = {"root_id": root.root_id, "episode": int(root.meta["episode"]),
                "case_sha256": case_identity(root),
                "requested_steps": n_blocks * 5}
        try:
            context = reset_root(env, root)
        except ValueError as exc:
            if "censored or out-of-domain prefix" not in str(exc):
                raise
            rows.append({**case, "valid": False, "reason": str(exc), "arena_exit": True,
                         "censored": True, "executed_steps": 0,
                         "charged_steps": ledger.total - before})
            continue
        local = polygons_from_env(env)
        frames, history = list(context.frames), context.history_actions.copy()
        state = context.state.copy()
        initial = state.copy()
        warm = None
        latency, diagnostics, states = [], [], [state.copy()]
        arena_exit, censored, terminated, truncated = False, False, False, False
        for block in range(n_blocks):
            plan = planner.plan(frames, history, context.goal_frame,
                                seed=root.seed * 1000 + block, pusher_xy=state[:2],
                                init_future=warm)
            latency.append(plan.solve_time)
            diagnostics.append(plan.diag or {})
            log = run_actions(env, plan.blocks[0], record_frames=True)
            count = log.executed_steps
            states.extend(log.states[1:count + 1])
            state = log.states[count].copy()
            arena_exit |= not bool(log.observation_valid[:count + 1].all())
            censored |= log.censored
            terminated |= bool(log.terminated[count])
            truncated |= bool(log.truncated[count])
            if arena_exit or censored or terminated or truncated:
                break
            frames = (frames + [log.frames[-1]])[-3:]
            history = np.concatenate([history[1:], plan.blocks[:1]], axis=0)
            warm = next_warm_start(plan.blocks)
        goal = root.goal_state[2:5]
        angle = np.arctan2(np.sin(state[4] - goal[2]), np.cos(state[4] - goal[2]))
        censored |= len(states) - 1 < case["requested_steps"]
        row = {**case, "valid": True, "terminated": terminated, "truncated": truncated,
               "censored_future": censored, "final_state": state.tolist(),
               "goal_state": np.asarray(root.goal_state).tolist(),
               "local_polygons": [p.tolist() for p in local],
               "initial_coverage": block_coverage(initial[2:5], goal, local),
               "final_coverage": block_coverage(state[2:5], goal, local),
               "final_pose_error_px": float(np.linalg.norm(state[2:4] - goal[:2])),
               "final_angle_error_deg": float(np.degrees(abs(angle))),
               "arena_exit": arena_exit, "censored": censored,
               "executed_steps": len(states) - 1, "charged_steps": ledger.total - before,
               "mean_latency_s": float(np.mean(latency)),
               "all_infeasible_solves": sum(d.get("frac_feasible", 1) == 0 for d in diagnostics),
               "infeasible_returned_solves": sum(not d.get("executed_feasible", True) for d in diagnostics),
               "states": np.asarray(states).tolist(), "diagnostics": diagnostics}
        row["completed_on_verified_goal"] = verified_goal_completion(row)
        row["outcome_complete"] = (not censored or row["completed_on_verified_goal"]) and not arena_exit
        rows.append(row)
        print(f"[retention] {root.root_id}: coverage={row['final_coverage']:.3f}, error={row['final_pose_error_px']:.1f}px, exit={arena_exit}", flush=True)
    env.close()
    valid = [r for r in rows if r["valid"]]
    summary = {"n": len(rows), "n_valid": len(valid), "n_blocks": n_blocks,
               "protocol": RETENTION_PROTOCOL, "steps_per_block": 5,
               "termination_policy": TERMINATION_POLICY,
               "n_completed_on_verified_goal": sum(r.get("completed_on_verified_goal", False) for r in rows),
               "arena_exits": sum(r["arena_exit"] for r in rows),
               "mean_final_coverage": float(np.mean([r["final_coverage"] for r in valid])) if valid else None,
               "mean_final_pose_error_px": float(np.mean([r["final_pose_error_px"] for r in valid])) if valid else None}
    return {"rows": rows, "summary": summary, "ledger": ledger.to_dict()}


def verified_goal_completion(row: dict) -> bool:
    """Environment terminal alone is weaker than actual 95% whole-T coverage.

    Installed stable_worldmodel PushT.eval_state uses combined pusher/block xy error
    <20 and periodic angle error <pi/9. Recheck this predicate and geometric coverage;
    the unused success_threshold attribute is not the environment's terminal rule.
    """
    if (row.get("terminated") is not True or row.get("truncated") is not False
            or row.get("arena_exit") is not False or row.get("valid") is not True):
        return False
    state, goal = np.asarray(row.get("final_state", [])), np.asarray(row.get("goal_state", []))
    local = tuple(np.asarray(p) for p in row.get("local_polygons", []))
    if (state.shape != (7,) or goal.shape != (7,) or len(local) != 2
            or any(p.shape != (4, 2) for p in local)
            or not all(np.isfinite(p).all() for p in (state, goal, *local))):
        return False
    position_error = np.linalg.norm(goal[:4] - state[:4])
    angle_error = abs(np.arctan2(np.sin(goal[4] - state[4]), np.cos(goal[4] - state[4])))
    coverage = block_coverage(state[2:5], goal[2:5], local)
    return bool(position_error < 20 and angle_error < np.pi / 9
                and coverage >= RETENTION_GOAL_COVERAGE
                and np.isclose(row.get("final_coverage", np.nan), coverage, atol=1e-6, rtol=0))


def retention_integrity(base, adapted) -> dict:
    """Validate fixed source cases and fully observed equal evaluation horizons."""
    a, b = base.get("rows", []), adapted.get("rows", [])
    reasons = []
    sa, sb = base.get("summary", {}), adapted.get("summary", {})
    n_blocks = sa.get("n_blocks")
    if (type(n_blocks) is not int or n_blocks < RETENTION_MIN_BLOCKS
            or sb.get("n_blocks") != n_blocks
            or any(s.get("protocol") != RETENTION_PROTOCOL or s.get("steps_per_block") != 5
                       or s.get("termination_policy") != TERMINATION_POLICY
                   for s in (sa, sb))):
        reasons.append("A declared equal horizon of at least 50 five-step blocks is required")
    if len(a) < RETENTION_MIN_EPISODES or len(a) != len(b):
        reasons.append("At least 20 paired retention episodes are required")
    for field in ("root_id", "episode", "case_sha256"):
        left, right = [r.get(field) for r in a], [r.get(field) for r in b]
        if left != right or None in left or len(set(left)) != len(left):
            reasons.append(f"Retention {field} must identify distinct, exactly paired source cases")
    requested = n_blocks * 5 if type(n_blocks) is int else None
    invalid_outcomes = []
    for row in a + b:
        executed = row.get("executed_steps")
        complete_horizon = executed == requested
        verified = verified_goal_completion(row)
        censored = type(executed) is int and requested is not None and executed < requested
        valid = (row.get("valid") is True and row.get("arena_exit") is False
                 and row.get("truncated") is False and type(executed) is int
                 and 0 < executed <= (requested or 0) and row.get("requested_steps") == requested
                 and row.get("censored") is censored and row.get("censored_future") is censored
                 and row.get("completed_on_verified_goal") is verified
                 and (row.get("terminated") is False or verified)
                 and (complete_horizon or verified))
        if not valid:
            invalid_outcomes.append(row.get("root_id"))
    if invalid_outcomes:
        reasons.append("Unresolved censored, invalid, timeout, arena-exit or unverified terminal outcomes cannot pass")
    for key in ("final_coverage", "final_pose_error_px"):
        if any(not isinstance(r.get(key), (int, float)) or not np.isfinite(r[key]) for r in a + b):
            reasons.append(f"Retention {key} must be finite for every paired case")
    return {"complete": not reasons, "reasons": reasons, "n_paired_cases": len(a),
            "n_blocks": n_blocks, "protocol": RETENTION_PROTOCOL}


def compare_retention(base, adapted, *, n_boot=2000, seed=903):
    integrity = retention_integrity(base, adapted)
    out = {**integrity, "criterion": "no detected paired degradation; not an equivalence guarantee"}
    if not integrity["complete"]:
        return {**out, "passes": False}
    a, b = base["rows"], adapted["rows"]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(n_boot, len(a)))
    for key in ("final_coverage", "final_pose_error_px"):
        differences = np.array([y[key] - x[key] for x, y in zip(a, b)])
        lo, hi = np.quantile(differences[indices].mean(axis=1), [0.025, 0.975])
        out[key] = {"adapted_minus_base": float(differences.mean()), "lo": float(lo), "hi": float(hi)}
    out["arena_exits_increased"] = sum(r["arena_exit"] for r in b) > sum(r["arena_exit"] for r in a)
    out["passes"] = bool(out["final_coverage"]["hi"] >= 0 and out["final_pose_error_px"]["lo"] <= 0
                         and not out["arena_exits_increased"])
    return out
