"""Frozen small-study panels, random configurations and outcome-only selection."""

from __future__ import annotations

import numpy as np
import torch

from helpers.pusht_smoke import arena_clearance, physical_clearance


def render_goal_image(env, goal):
    """Render an exact target pose without physics, refreshing Pymunk shape caches."""
    goal = np.asarray(goal)
    env.agent.velocity = (0, 0)
    env.agent.position = tuple(goal[:2])
    env.block.angle = float(goal[4])
    env.block.position = tuple(goal[2:4])
    # debug_draw reads transformed shape geometry, not just the body's pose fields.
    env.space.reindex_shapes_for_body(env.agent)
    env.space.reindex_shapes_for_body(env.block)
    image = np.ascontiguousarray(env.render())
    blue = (image[..., 2] > 170) & (image[..., 0] < 100) & (image[..., 1] < 220)
    yy, xx = np.nonzero(blue)
    if not len(xx):
        raise ValueError("Rendered target pusher is not visible")
    centre = np.array([xx.mean(), yy.mean()]) * 512 / image.shape[0]
    if np.linalg.norm(centre - goal[:2]) > 10:
        raise ValueError("Rendered pusher does not match the requested target pose")
    return image


def configurations(design):
    current = dict(design["current"])
    result = [{"id": "current", "params": current}]
    rng = np.random.default_rng(design["random_seed"])
    keys = tuple(design["search_space"])
    seen = {tuple(current[k] for k in keys)}
    while len(result) <= design["random_configurations"]:
        values = tuple(rng.choice(design["search_space"][k]).item() for k in keys)
        if values in seen:
            continue
        seen.add(values)
        params = {**current, **dict(zip(keys, values))}
        result.append({"id": f"random-{len(result):02d}", "params": params})
    return result


def layout(initial, goal, kind, design):
    start, end = np.asarray(initial)[:2], np.asarray(goal)[:2]
    forward = end - start
    distance = np.linalg.norm(forward)
    if distance < 1e-6:
        raise ValueError("A distinct pusher goal is required")
    forward /= distance
    side = np.array([forward[1], -forward[0]])
    midpoint = (start + end) / 2

    def room(direction):
        return min(
            ((491 if direction[i] > 0 else 21) - midpoint[i]) / direction[i]
            for i in range(2)
            if abs(direction[i]) > 1e-8
        )

    if room(-side) > room(side):
        side = -side
    endpoints = torch.tensor(np.stack([start, end]), dtype=torch.float64)
    if not bool((arena_clearance(endpoints) >= design["margin_px"] + 0.001).all()):
        raise ValueError("A full-pusher endpoint violates the arena margin")
    for offset in range(0, 33, 4):
        case = {
            "kind": kind,
            "origin": (midpoint + side * offset).tolist(),
            "basis": np.stack([side, forward]).tolist(),
            **design["shapes"][kind],
        }
        if bool((physical_clearance(endpoints, case) >= design["margin_px"] + 0.001).all()):
            return case
    raise ValueError("No endpoint-clear layout under the frozen rule")


def violating(row):
    return any(row[key] for key in ("hazard_violation", "margin_breach", "arena_margin_breach"))


def summarize(rows):
    return {
        "episodes": len(rows),
        "safe_completions": sum(r["safe_completion"] for r in rows),
        "violating_episodes": sum(violating(r) for r in rows),
        "hazard_violation_episodes": sum(r["hazard_violation"] for r in rows),
        "unfinished": sum(r["unfinished"] for r in rows),
        "engineering_errors": sum(r["error"] is not None for r in rows),
        "engaged": sum(r["actions"] > 0 and r["pusher_path_length_px"] > 1 for r in rows),
        "object_moved_5px": sum(r["object_displacement_px"] >= 5 for r in rows),
        "J_sum": sum(r["actions"] if r["safe_completion"] else 50 for r in rows),
        "actions": sum(r["actions"] for r in rows),
        "planning_seconds": sum(r["planning_seconds"] for r in rows),
        "episode_seconds": sum(r["episode_seconds"] for r in rows),
        "resources": {
            key: sum(r["resources"].get(key, 0) for r in rows)
            for key in sorted(set().union(*(r["resources"] for r in rows)))
        },
    }


def select_configuration(configs, results, expected_case_ids):
    """No goal distances, reserved outcomes or low-work unfinished rewards enter selection."""
    ranked = []
    summaries = {}
    for order, config in enumerate(configs):
        rows = results[config["id"]]
        ids = [row["study_case_id"] for row in rows]
        if sorted(ids) != sorted(expected_case_ids):
            raise ValueError("Selection requires the entire frozen development panel exactly once")
        if any(row["study_split"] != "development" for row in rows):
            raise ValueError("Reserved evaluation outcomes cannot enter selection")
        summary = summarize(rows)
        summaries[config["id"]] = summary
        if summary["violating_episodes"] or summary["engineering_errors"]:
            continue
        key = (-summary["safe_completions"], summary["J_sum"], order)
        ranked.append((key, config))
    if not ranked:
        raise ValueError("No eligible development configuration; evaluation is not authorized")
    selected = min(ranked, key=lambda item: item[0])[1]
    return {
        "selected": selected,
        "development_summaries": summaries,
        "selection_information": "Development safe completion and J only; current-first ties",
    }
