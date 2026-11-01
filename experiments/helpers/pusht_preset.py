"""Preset-versus-tuning protocol primitives; no provider or simulator side effects."""

import numpy as np
import torch

from helpers.pusht_smoke import arena_clearance, physical_clearance

KINDS = (
    ("island", "mild"),
    ("island", "strong"),
    ("bar", "mild"),
    ("bar", "strong"),
    ("gate", "mild"),
    ("gate", "strong"),
)


def configurations(design):
    base = {
        "horizon": 5,
        "replan_blocks": 5,
        "iterations": 30,
        "budget_scale": 1.0,
        "variance": 1.0,
        "elite": 0.1,
        "retain": 0.0,
        "correlate": 0.0,
        "adaptive": False,
        "objective": "learned",
        "checked": True,
    }
    anchor = {
        **base,
        "horizon": 3,
        "replan_blocks": 1,
        "budget_scale": 0.2,
        "retain": 0.5,
        "correlate": 0.5,
    }
    result = [{"id": "authors", "params": base}, {"id": "working-anchor", "params": anchor}]
    rng = np.random.default_rng(design["search_seed"])
    seen = {tuple(c["params"].values()) for c in result}
    while len(result) < 2 + design["random_configurations"]:
        h = int(rng.choice([1, 2, 3, 5]))
        r = int(rng.choice([value for value in (1, 2, 5) if value <= h]))
        params = {
            **base,
            "horizon": h,
            "replan_blocks": r,
            "budget_scale": r / 5,
            "variance": float(rng.choice([0.5, 1.0, 2.0])),
            "elite": float(rng.choice([0.05, 0.1, 0.2])),
            "retain": float(rng.choice([0.0, 0.25, 0.5])),
            "correlate": float(rng.choice([0.0, 0.5, 0.8])),
        }
        if tuple(params.values()) in seen:
            continue
        seen.add(tuple(params.values()))
        result.append({"id": f"random-{len(result) - 1:02d}", "params": params})
    return result


def geometry(initial, goal, route, kind, strength):
    """Use only demonstrated path and settled endpoints, never controller outcomes."""
    start, end = np.asarray(initial, dtype=float)[:2], np.asarray(goal, dtype=float)[:2]
    forward = end - start
    forward /= np.linalg.norm(forward)
    side = np.array([forward[1], -forward[0]])
    route = np.asarray(route)
    length = np.r_[0, np.linalg.norm(np.diff(route, axis=0), axis=1).cumsum()]
    order = np.argsort(abs(length - length[-1] / 2), kind="stable")
    endpoints = torch.tensor(np.stack([start, end]), dtype=torch.float64)
    if not bool((arena_clearance(endpoints) >= 10).all()):
        return None
    for index in order:
        centre = route[index]
        if not ((centre >= 70) & (centre <= 442)).all():
            continue
        if min(np.linalg.norm(centre - start), np.linalg.norm(centre - end)) < 50:
            continue
        direction = side if np.dot(256 - centre, side) >= 0 else -side
        shape = {
            "kind": kind,
            "origin": centre.tolist(),
            "basis": np.stack([direction, forward]).tolist(),
        }
        if kind == "island":
            shape["radius_px"] = 12 if strength == "mild" else 20
        elif kind == "bar":
            shape.update(segment_half_length_px=30 if strength == "mild" else 50, radius_px=7)
        else:
            shape.update(
                gap_width_px=100 if strength == "mild" else 70,
                gap_offset_px=65,
                extent_px=180,
                band_half_width_px=7,
            )
        if bool((physical_clearance(endpoints, shape) >= 10).all()):
            return shape
    return None


def summary(rows):
    return {
        "episodes": len(rows),
        "safe_completions": sum(r["safe_completion"] for r in rows),
        "violations": sum(r["margin_breach"] or r["arena_margin_breach"] for r in rows),
        "contacts": sum(r["hazard_violation"] for r in rows),
        "unfinished": sum(r["unfinished"] for r in rows),
        "engaged": sum(r["actions"] > 0 and r["pusher_path_length_px"] > 1 for r in rows),
        "J": sum(r["actions"] if r["safe_completion"] else 50 for r in rows),
        "actions": sum(r["actions"] for r in rows),
        "planning_seconds": sum(r["planning_seconds"] for r in rows),
        "episode_seconds": sum(r["episode_seconds"] for r in rows),
        "resources": {
            k: sum(r["resources"].get(k, 0) for r in rows)
            for k in set().union(*(r["resources"] for r in rows))
        },
    }


def select(configs, results):
    expected = {(index, condition) for index in range(6) for condition in ("hazard", "control")}
    base_control = summary([r for r in results["authors"] if r["condition"] == "control"])
    scores, eligible = {}, []
    for order, config in enumerate(configs):
        rows = results[config["id"]]
        if {(r["case_index"], r["condition"]) for r in rows} != expected or len(rows) != 12:
            raise ValueError("Selection needs the entire development panel")
        if any(r["split"] != "development" or r["error"] for r in rows):
            raise ValueError("Only error-free development results may select parameters")
        hazard = summary([r for r in rows if r["condition"] == "hazard"])
        control = summary([r for r in rows if r["condition"] == "control"])
        scores[config["id"]] = {"hazard": hazard, "control": control}
        if hazard["violations"] or control["violations"]:
            continue
        if control["safe_completions"] < base_control["safe_completions"]:
            continue
        eligible.append(
            (
                (
                    -hazard["safe_completions"],
                    -control["safe_completions"],
                    hazard["J"] + control["J"],
                    order,
                ),
                config,
            )
        )
    if not eligible:
        raise ValueError("No eligible configuration; stop before reserved evaluation")
    return {"selected": min(eligible, key=lambda item: item[0])[1], "development": scores}
