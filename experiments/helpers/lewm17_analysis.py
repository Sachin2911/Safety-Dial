"""Paired crossed run/case bootstrap and honest resource summaries for task 17."""

import numpy as np

FIELDS = (
    "requested_safe_completion",
    "benchmark_safe_completion",
    "any_violation",
    "margin_breach",
    "unfinished",
    "engaged",
    "censored",
)


def summarize(rows):
    result = {"n": len(rows)}
    for field in FIELDS:
        result[field] = sum(bool(r[field]) for r in rows)
    result["J_mean"] = float(np.mean([r["J"] for r in rows])) if rows else None
    result["seconds"] = sum(r["seconds"] for r in rows)
    for field in (
        "hazard_samples",
        "margin_breach_samples",
        "path_length",
        "command_norm_sum",
        "bounded_control_norm_sum",
        "action_coordinates_clipped",
        "action_coordinates_total",
    ):
        result[field] = sum(r.get(field, 0) for r in rows)
    result["resources"] = {}
    for row in rows:
        for key, value in row["counters"].items():
            result["resources"][key] = result["resources"].get(key, 0) + value
    return result


def crossed_interval(matrix, seed=170990, draws=10000, confidence=0.95):
    """Resample whole run and case indices; planner seeds already averaged within cells."""
    matrix = np.asarray(matrix, dtype=float)
    nr, nc = matrix.shape
    rng = np.random.default_rng(seed)
    run_indices = rng.integers(0, nr, (draws, nr))
    case_indices = rng.integers(0, nc, (draws, nc))
    sampled = matrix[run_indices[:, :, None], case_indices[:, None, :]].mean((1, 2))
    tail = (1 - confidence) / 2
    return {
        "estimate": float(matrix.mean()),
        "confidence": confidence,
        "interval": np.quantile(sampled, [tail, 1 - tail]).tolist(),
        "run_means": matrix.mean(1).tolist(),
        "n_runs": nr,
        "n_cases": nc,
        "draws": draws,
        "seed": seed,
    }


def paired_contrast(left, right, confidence=0.95):
    left, right = np.asarray(left), np.asarray(right)
    if right.shape[0] == 1:
        right = np.broadcast_to(right, left.shape)
    if left.shape != right.shape:
        raise ValueError("Contrasts must preserve paired run and case blocks")
    return crossed_interval(left - right, confidence=confidence)


def matrix_for(rows, method, case_ids, field="requested_safe_completion"):
    n_runs = 1 if method in ("historical", "tuned") else 5
    matrix = np.zeros((n_runs, len(case_ids)))
    for replicate in range(n_runs):
        condition = method if n_runs == 1 else f"{method}-r{replicate}"
        for index, case_id in enumerate(case_ids):
            pair = [
                r
                for r in rows
                if r["job"]["condition"] == condition and r["job"]["case"]["id"] == case_id
            ]
            if len(pair) != 2 or {r["job"]["seed"] for r in pair} != {179710, 179711}:
                raise ValueError("Incomplete final case/planner-seed block")
            matrix[replicate, index] = np.mean([r[field] for r in pair])
    return matrix
