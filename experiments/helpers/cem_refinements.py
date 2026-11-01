"""Feasible-only CEM fitting, bounded exploration and checked final selection."""

import math

import torch


def fit_distribution(candidates, costs, feasible, mean, spread, count, momentum, minimum):
    """Fit one planning batch to safe elites, or least violation if none are safe."""
    if candidates.shape[0] != 1:
        raise ValueError("The planner currently supports one environment at a time")
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError("CEM momentum must lie in [0, 1)")
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("Minimum sampling spread must be finite and positive")
    safe = torch.where(feasible[0])[0]
    pool = safe if len(safe) else torch.arange(costs.shape[1], device=costs.device)
    indices = pool[costs[0, pool].argsort()[: min(count, len(pool))]]
    if not len(indices):
        raise ValueError("At least one candidate is required")
    elite = candidates[:, indices]
    fitted_mean = elite.mean(1)
    # A singleton contains no estimate of dispersion. Retain the existing search spread.
    fitted_spread = elite.std(1) if len(indices) > 1 else spread
    next_mean = momentum * mean + (1 - momentum) * fitted_mean
    next_spread = (momentum * spread + (1 - momentum) * fitted_spread).clamp_min(minimum)
    return (
        next_mean,
        next_spread,
        {
            "feasible_candidates": len(safe),
            "fitted_elites": len(indices),
            "feasible_only": bool(len(safe)),
            "spread_min": float(next_spread.min()),
            "spread_max": float(next_spread.max()),
        },
    )


def choose_checked_mean(mean, score, feasible, best):
    """Prefer the mean on ties, otherwise the best feasible task cost discovered."""
    if feasible and (best is None or score <= best[0]):
        return mean, "mean"
    if best is not None:
        return best[1].to(mean.device, mean.dtype), "best_checked"
    raise ValueError("No feasible mean or sampled plan exists")
