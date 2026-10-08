"""Statistics for the evolution study (docs/evoPlan/README.md, "Statistics").

Every interval that pools start states resamples SOURCE EPISODES (clusters), never single
roots: the 256 evaluation roots are four per test episode, so roots from one episode are not
independent. Violation rates get Wilson intervals; zero events in n bounds a rate only at
about 3/n. Across seeds: interquartile mean with a bootstrap interval; rule comparisons use
Mann-Whitney U with Holm's correction.

    lo, hi = wilson(k, n)
    rho = clustered_spearman(imag, real, clusters)       # policies x roots matrices
    gap = paired_gap_difference(real_hi, imag_hi, real_lo, imag_lo, clusters)
"""

from __future__ import annotations

import numpy as np


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for k events in n trials (95% by default)."""
    if n <= 0:
        return float("nan"), float("nan")
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def iqm(values) -> float:
    """Interquartile mean: the mean of the middle 50% (scipy's 25% trimmed mean)."""
    from scipy.stats import trim_mean

    v = np.asarray(values, dtype=np.float64)
    return float(trim_mean(v, 0.25)) if len(v) else float("nan")


def bootstrap_ci(values, stat=iqm, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Percentile 95% interval of `stat` over independent units (for example seeds)."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    return {"point": stat(v), "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5)),
            "n": int(len(v)), "n_boot": n_boot}


def _cluster_draws(clusters, n_boot: int, seed: int):
    clusters = np.asarray(clusters)
    ids = np.unique(clusters)
    members = [np.where(clusters == c)[0] for c in ids]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        yield np.concatenate([members[i] for i in pick])


def spearman(x, y) -> float:
    from scipy.stats import spearmanr

    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def clustered_spearman(imagined, real, clusters, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Spearman correlation across policies between imagined and real per-policy means.

    imagined, real: (n_policies, n_roots) per-root scores on identical start states;
    clusters: (n_roots,) source episode of each root. Each bootstrap draw resamples source
    episodes with replacement, recomputes every policy's mean over the drawn roots and
    correlates the two vectors of means.
    """
    a, b = np.asarray(imagined, np.float64), np.asarray(real, np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != len(clusters):
        raise ValueError("imagined and real must be (policies, roots) matching the clusters")
    point = spearman(a.mean(1), b.mean(1))
    vals = np.array([spearman(a[:, s].mean(1), b[:, s].mean(1)) for s in _cluster_draws(clusters, n_boot, seed)])
    ok = np.isfinite(vals)
    return {"point": point, "lo": float(np.percentile(vals[ok], 2.5)) if ok.any() else float("nan"),
            "hi": float(np.percentile(vals[ok], 97.5)) if ok.any() else float("nan"),
            "n_boot": int(ok.sum()), "n_policies": int(a.shape[0]), "n_clusters": int(len(np.unique(clusters)))}


def clustered_mean_ci(values, clusters, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean over roots (any trailing unit axis first) with a source-episode bootstrap.

    values: (n_roots,) or (n_units, n_roots); the statistic is the grand mean.
    """
    v = np.asarray(values, np.float64)
    v2 = v[None] if v.ndim == 1 else v
    draws = np.array([v2[:, s].mean() for s in _cluster_draws(clusters, n_boot, seed)])
    return {"point": float(v2.mean()), "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5)),
            "n_boot": n_boot, "n_clusters": int(len(np.unique(clusters)))}


def paired_gap_difference(real_hi, imag_hi, real_lo, imag_lo, clusters, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Gate 2 estimand: mean over seeds of [gap(hi) - gap(lo)], gap = real - imagined rate.

    Each array is (n_seeds, n_roots) of 0/1 violation indicators on identical evaluation
    roots (the same roots for both settings, so the difference is paired). Bootstrap draws
    resample source episodes; seeds stay fixed within a draw.
    """
    arrays = [np.asarray(x, np.float64) for x in (real_hi, imag_hi, real_lo, imag_lo)]
    if len({x.shape for x in arrays}) != 1 or arrays[0].ndim != 2:
        raise ValueError("all four arrays must share one (seeds, roots) shape")
    rh, ih, rl, il = arrays

    def stat(s):
        return float(((rh[:, s] - ih[:, s]).mean(1) - (rl[:, s] - il[:, s]).mean(1)).mean())

    full = np.arange(rh.shape[1])
    draws = np.array([stat(s) for s in _cluster_draws(clusters, n_boot, seed)])
    return {"point": stat(full), "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5)),
            "gap_hi": float((rh - ih).mean()), "gap_lo": float((rl - il).mean()),
            "n_boot": n_boot, "n_seeds": int(rh.shape[0]), "n_clusters": int(len(np.unique(clusters)))}


def mann_whitney(a, b) -> dict:
    """Two-sided Mann-Whitney U test of two independent samples (for example seeds)."""
    from scipy.stats import mannwhitneyu

    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    res = mannwhitneyu(a, b, alternative="two-sided")
    return {"U": float(res.statistic), "p": float(res.pvalue), "n_a": int(len(a)), "n_b": int(len(b))}


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error), in the input order."""
    p = np.asarray(pvalues, np.float64)
    order = np.argsort(p, kind="stable")
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj
