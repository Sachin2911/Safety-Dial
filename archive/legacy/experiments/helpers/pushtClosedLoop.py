"""Predeclared cases and observed-only metrics for the gated E5 runner."""
from __future__ import annotations

import numpy as np

from helpers.pushtGeometry import clearance_trace, hazard_from_dict, t_polygons
from helpers.pushtContactReplay import reset_root, run_actions


def fixed_cases(bank, layouts, *, episodes_per_layout=10, seed=0, margin=0.0):
    """Two fixed hazards, paired distinct roots, selected without rollout outcomes.

    Start/goal footprint validity is a layout-definition condition. Candidate order and
    hazard selection depend only on stored root identifiers and a predeclared RNG seed.
    Failure to fill the design aborts instead of silently reducing the sample size.
    """
    rng = np.random.default_rng(seed)
    root_map = {r.root_id: r for r in bank.roots}
    cases, used = [], set()
    for family in ("familiar", "heldout"):
        eligible_layouts = sorted([layout for layout in layouts if layout.family == family], key=lambda layout: layout.root_id)
        if not eligible_layouts:
            raise ValueError(f"No predeclared {family} hazard layouts")
        layout = eligible_layouts[0]
        order = rng.permutation(sorted(root_map))
        count = 0
        for root_id in order:
            if root_id in used:
                continue
            root = root_map[root_id]
            endpoints = np.asarray([root.meta["state_at_root"][2:5], root.goal_state[2:5]])
            if clearance_trace(endpoints, layout.shape).min() <= max(0.0, margin):
                continue
            case_id = len(cases)
            cases.append({"case_id": case_id, "root_id": str(root_id), "hazard": layout.hazard,
                          "hazard_family": family, "layout_source_root": layout.root_id,
                          "planning_seed": seed * 100 + case_id})
            used.add(root_id)
            count += 1
            if count == episodes_per_layout:
                break
        if count != episodes_per_layout:
            raise ValueError(f"Only {count} valid roots for the fixed {family} hazard; preserve design and revise on development")
    return cases


def closedloop_arm_specs(base, repaired, *, dial, fixed_margin):
    """Pair original/repaired models inside each declared constrained planner."""
    if not np.isfinite([dial, fixed_margin]).all() or min(dial, fixed_margin) < 0:
        raise ValueError("Closed-loop margins must be finite and nonnegative")
    return {
        "nominal": (base, None, dial),
        "safe_original": (base, "safe", dial),
        "safe_repaired": (repaired, "safe", dial),
        "safe_original_fixed_margin": (base, "safe", fixed_margin),
        "safe_repaired_fixed_margin": (repaired, "safe", fixed_margin),
        "penalty_original": (base, "penalty", dial),
        "penalty_repaired": (repaired, "penalty", dial),
    }


def penalty_reference_protocol(lam):
    """Do not imply behavioral calibration from equal CEM candidate budgets."""
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("The predeclared penalty weight must be finite and nonnegative")
    return {"lambda": float(lam), "lambda_selection": "predeclared_before_test",
            "matching": "equal_candidate_compute_only",
            "behavioral_matching": False,
            "shared_controls": ["case_ids", "planning_seeds", "CEM_samples", "CEM_iterations",
                                "horizon", "action_arena_guard", "pose_probe", "hazard_dial"],
            "interpretation": "No claim of matched acceptance, goal progress, or violation rate"}


def block_coverage(pose, goal_pose):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    block = unary_union([Polygon(poly) for poly in t_polygons(pose)])
    goal = unary_union([Polygon(poly) for poly in t_polygons(goal_pose)])
    return float(block.intersection(goal).area / goal.area)


def run_observed_episode(env, planner, root, case, *, n_blocks):
    """Stop before conditioning on unobserved/invalid frames and retain censored outcomes."""
    from helpers.imagination import next_warm_start

    hazard = hazard_from_dict(case["hazard"])
    ctx = reset_root(env, root)
    frames, hist, state, warm = list(ctx.frames), ctx.history_actions.copy(), ctx.state.copy(), None
    states, diags, latency, executed_blocks = [state.copy()], [], [], []
    terminated, truncated = False, False
    contact_rows = []
    for block in range(n_blocks):
        plan = planner.plan(frames, hist, ctx.goal_frame, seed=case["planning_seed"] * 1000 + block,
                            pusher_xy=state[:2], init_future=warm)
        if not plan.diag or "executed_feasible" not in plan.diag:
            raise ValueError("Planner did not provide an audit of the exact returned action sequence")
        diags.append(dict(plan.diag))
        latency.append(plan.solve_time)
        actions = plan.blocks[0].reshape(-1, 2)
        log = run_actions(env, actions, record_frames=True)
        n = log.executed_steps
        contact_rows.append({field: None if getattr(log, field, None) is None else int((getattr(log, field)[1:n + 1] > 0).sum())
                             for field in ("pusher_block_contacts", "block_wall_contacts")})
        states.extend(log.states[1:n + 1])
        executed_blocks.append(actions[:n].tolist())
        state = log.states[n].copy()
        terminated = bool(log.terminated is not None and log.terminated.any())
        truncated = bool(log.truncated is not None and log.truncated.any())
        if not log.valid_for_training or terminated or truncated:
            break
        frames = (frames + [log.frames[-1]])[-3:]
        hist = np.concatenate([hist[1:], plan.blocks[:1]], 0)
        warm = next_warm_start(plan.blocks)
    st = np.asarray(states)
    clearance = clearance_trace(st[:, 2:5], hazard)
    exits = ((st[:, :2] < 0) | (st[:, :2] > 512)).any(1)
    known_hazard = bool((clearance <= 0).any())
    known_unsafe = known_hazard or bool(exits.any())
    censored = len(st) - 1 < n_blocks * 5
    goal = np.asarray(root.goal_state)
    return {**case, "observed_env_steps": len(st) - 1, "requested_env_steps": n_blocks * 5,
            "root_prefix_steps": len(root.prefix), "censored": censored,
            "contact_kind": "pusher_block" if all(row["pusher_block_contacts"] is not None for row in contact_rows) else "any_collision",
            "pusher_block_contact_steps_observed": None if any(row["pusher_block_contacts"] is None for row in contact_rows) else sum(row["pusher_block_contacts"] for row in contact_rows),
            "block_wall_contact_steps_observed": None if any(row["block_wall_contacts"] is None for row in contact_rows) else sum(row["block_wall_contacts"] for row in contact_rows),
            "hazard_only_unsafe": True if known_hazard else (None if censored else False),
            "unsafe_composite": known_unsafe, "terminated": terminated, "truncated": truncated,
            "violation_steps_observed": int((clearance[1:] <= 0).sum()), "min_clearance_observed": float(clearance.min()),
            "arena_exit_steps": int(exits[1:].sum()), "final_coverage": block_coverage(st[-1, 2:5], goal[2:5]),
            "final_block_err_px": float(np.linalg.norm(st[-1, 2:4] - goal[2:4])),
            "final_angle_err_deg": float(np.degrees(abs(np.arctan2(np.sin(st[-1, 4] - goal[4]), np.cos(st[-1, 4] - goal[4]))))),
            "all_infeasible_solves": sum(d.get("frac_feasible", 1) == 0 for d in diags),
            "executed_infeasible_solves": sum(not d["executed_feasible"] for d in diags),
            "fallback_solves": sum(bool(d.get("fallback_used", False) or d.get("audit_replacement_used", False)) for d in diags),
            "unsafe_mean_replacements": sum(bool(d.get("audit_replacement_used", False)) for d in diags),
            "latency_mean_s": float(np.mean(latency)), "planning_diags": diags,
            "executed_action_blocks": executed_blocks, "dense_states": st.tolist()}
