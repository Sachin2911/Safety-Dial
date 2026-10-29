"""Uncertainty for paired starts. Selection and confirmation are kept separate."""

import math

import numpy as np
from scipy.stats import beta, norm


def exact_binomial(k, n, alpha):
    if not 0 <= k <= n or n < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid binomial count or alpha")
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper


def wilson(k, n, alpha=0.05):
    if not 0 <= k <= n or n < 1:
        raise ValueError("Invalid rate denominator")
    z = float(norm.ppf(1 - alpha / 2))
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def paired_binary(candidate, reference, alpha=0.05):
    a, b = np.asarray(candidate, dtype=bool), np.asarray(reference, dtype=bool)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("A complete, nonempty paired panel is required")
    wins, losses = int(np.sum(a & ~b)), int(np.sum(~a & b))
    # Two exact marginal intervals at error alpha/2 each yield joint error <=alpha.
    wl, wu = exact_binomial(wins, len(a), alpha / 2)
    ll, lu = exact_binomial(losses, len(a), alpha / 2)
    return {
        "n": len(a),
        "candidate_only": wins,
        "reference_only": losses,
        "both": int(np.sum(a & b)),
        "neither": int(np.sum(~a & ~b)),
        "difference": (wins - losses) / len(a),
        "interval": [wl - lu, wu - ll],
        "alpha": alpha,
        "method": "Bonferroni combination of exact binomial win and loss intervals",
    }


def paired_mean(candidate, reference, draws, alpha=0.05):
    a, b = np.asarray(candidate, float), np.asarray(reference, float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("A complete paired panel is required")
    difference = a - b
    resampled = difference[draws].mean(axis=1)
    interval = np.quantile(resampled, [alpha / 2, 1 - alpha / 2]).tolist()
    return {
        "n": len(a),
        "difference": float(difference.mean()),
        "interval": interval,
        "alpha": alpha,
        "degenerate": interval[0] == interval[1],
        "method": "paired seed-block percentile bootstrap; 10000 shared resamples",
    }
