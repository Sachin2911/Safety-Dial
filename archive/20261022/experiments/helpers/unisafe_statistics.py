"""Paired binary contrasts with conservative finite-sample uncertainty bounds."""

import math

import numpy as np
from scipy.stats import beta


def clopper_pearson(k, n, alpha=0.05):
    if not 0 <= k <= n or n < 1:
        raise ValueError((k, n))
    return [
        float(beta.ppf(alpha / 2, k, n - k + 1)) if k else 0.0,
        float(beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0,
    ]


def paired_binary(candidate, reference):
    candidate, reference = np.asarray(candidate, dtype=bool), np.asarray(reference, dtype=bool)
    if candidate.shape != reference.shape or candidate.ndim != 1 or not len(candidate):
        raise ValueError("Paired nonempty one-dimensional outcomes are required")
    wins = int((candidate & ~reference).sum())
    losses = int((reference & ~candidate).sum())
    n, discordant = len(candidate), wins + losses
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant)
    # Each marginal interval has coverage >= 97.5%; Bonferroni gives joint
    # coverage >= 95%, hence a conservative interval for p(win)-p(loss).
    w = clopper_pearson(wins, n, alpha=.025)
    loss = clopper_pearson(losses, n, alpha=.025)
    return {
        "n": n, "candidate_only_positive": wins, "reference_only_positive": losses,
        "both_positive": int((candidate & reference).sum()),
        "neither_positive": int((~candidate & ~reference).sum()),
        "rate_difference": (wins - losses) / n,
        "conservative_exact_difference95": [w[0] - loss[1], w[1] - loss[0]],
        "mcnemar_exact_two_sided_p": p,
        "interval_method": "Bonferroni combination of exact 97.5% binomial intervals for paired win and loss probabilities",
    }


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    result, previous = {}, 0.0
    for i, key in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - i) * pvalues[key]))
        result[key] = previous
    return result
