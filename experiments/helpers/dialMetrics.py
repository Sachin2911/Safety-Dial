"""Dial metrics: false-safe acceptance, acceptance rate, dial curves, matched acceptance,
root-clustered bootstrap (protocol.md).

    A_m = 1  when  c_hat_min >= m       (accept the tape at margin m)
    FSA(m) = #(A_m=1 and U=1) / #(A_m=1)   undefined (NaN) when nothing is accepted
    AR(m)  = #(A_m=1) / N
    FRR(m) = #(A_m=0 and U=0) / #(U=0)

All functions take per-tape arrays: `c_hat` (predicted minimum clearance), `u` (true
unsafe flag, 0/1), `root` (root index per tape, for clustering).
"""

from __future__ import annotations

import numpy as np


def _clearances(c_hat, *, allow_empty=True):
    c = np.asarray(c_hat, dtype=float)
    if c.ndim != 1 or not np.isfinite(c).all():
        raise ValueError("Clearances must be a finite one-dimensional array; handle censored rows explicitly")
    if not allow_empty and not len(c):
        raise ValueError("At least one clearance is required")
    return c


def decisions(c_hat: np.ndarray, m: float) -> np.ndarray:
    if np.isnan(m):
        raise ValueError("The decision margin must not be NaN")
    return _clearances(c_hat) >= m


def fsa(c_hat, u, m: float, *, censored=None) -> dict:
    a = decisions(c_hat, m)
    truth = np.asarray(u)
    if truth.shape != a.shape or not np.isin(truth, [False, True]).all():
        raise ValueError("Unsafe labels must be observed 0/1 values matching the clearance shape")
    u = truth.astype(bool)
    unresolved = np.zeros_like(u) if censored is None else np.asarray(censored, dtype=bool)
    if unresolved.shape != a.shape:
        raise ValueError("Censoring mask must match the clearance shape")
    unresolved = unresolved & ~u  # a known violation resolves the binary unsafe event
    n_acc = int(a.sum())
    out = {"m": float(m), "n": int(len(a)), "n_accepted": n_acc, "acceptance_rate": n_acc / len(a) if len(a) else float("nan")}
    out["n_false_safe"] = int((a & u).sum())
    out["n_censored"] = int(unresolved.sum())
    out["n_accepted_censored"] = int((a & unresolved).sum())
    out["fsa_lower"] = out["n_false_safe"] / n_acc if n_acc else float("nan")
    out["fsa_upper"] = (out["n_false_safe"] + out["n_accepted_censored"]) / n_acc if n_acc else float("nan")
    out["fsa"] = out["fsa_lower"] if not out["n_accepted_censored"] else float("nan")
    safe = ~u & ~unresolved
    n_safe = int(safe.sum())
    out["false_reject_rate"] = int(((~a) & safe).sum()) / n_safe if n_safe else float("nan")
    return out


def dial_curve(c_hat, u, margins) -> list[dict]:
    return [fsa(c_hat, u, m) for m in margins]


def margin_for_acceptance(c_hat, target_ar: float) -> float:
    """The margin m on THIS set such that AR(m) is closest to target_ar (from above)."""
    if not np.isfinite(target_ar) or not 0 <= target_ar <= 1:
        raise ValueError("target_ar must lie in [0, 1]")
    c = np.sort(_clearances(c_hat, allow_empty=False))
    n = len(c)
    # A deterministic threshold cannot split ties; choose the smallest reachable
    # acceptance at or above the requested target and report its achieved rate.
    k = int(np.ceil(np.nextafter(target_ar * n, -np.inf))) if target_ar else 0
    k = min(max(k, 0), n)
    if k == 0:
        return float(np.nextafter(c[-1], np.inf))
    return float(c[n - k])


def matched_fsa(c_dev, u_dev, c_test, u_test, target_ar: float) -> dict:
    """Choose m on development to hit target_ar; apply unchanged on test."""
    m = margin_for_acceptance(c_dev, target_ar)
    r = fsa(c_test, u_test, m)
    r["target_acceptance_rate"] = float(target_ar)
    r["dev"] = fsa(c_dev, u_dev, m)
    return r


def auc_dial(c_hat, u, ar_lo: float = 0.2, ar_hi: float = 0.9, n_grid: int = 50, *, censored=None) -> float:
    """Normalised trapezoid area over actual reachable acceptance rates.

    n_grid remains a compatible argument; integration uses all distinct dial points.
    """
    if not 0 <= ar_lo < ar_hi <= 1:
        raise ValueError("AUC acceptance range must satisfy 0 <= ar_lo < ar_hi <= 1")
    c = _clearances(c_hat, allow_empty=False)
    truth = np.asarray(u)
    # Validate truth as well as predictions before computing the actual dial steps.
    validated = fsa(c, truth, float("-inf"), censored=censored)
    if validated["n_censored"]:
        return float("nan")
    order = np.argsort(-c, kind="stable")
    sorted_c, sorted_u = c[order], truth[order].astype(bool)
    endpoints = np.r_[np.flatnonzero(sorted_c[:-1] != sorted_c[1:]), len(c) - 1]
    counts = endpoints + 1
    ars = counts / len(c)
    vals = np.cumsum(sorted_u)[endpoints] / counts
    # Zero accepted plans have undefined FSA. Do not extrapolate from an unattainable
    # acceptance (for example, a model whose predictions are all tied).
    if ar_lo < ars[0] or ar_hi > ars[-1]:
        return float("nan")
    interior = (ars > ar_lo) & (ars < ar_hi)
    x = np.r_[ar_lo, ars[interior], ar_hi]
    y = np.interp(x, ars, vals)
    return float(np.trapezoid(y, x) / (ar_hi - ar_lo))


def clearance_error_stats(c_hat, c_true) -> dict:
    """Signed clearance error e = c_hat_min - c_min (positive = optimistic)."""
    e = np.asarray(c_hat) - np.asarray(c_true)
    return {"mean": float(e.mean()), "p50": float(np.percentile(e, 50)), "p90": float(np.percentile(e, 90)),
            "p95": float(np.percentile(e, 95)), "p99": float(np.percentile(e, 99)), "frac_optimistic": float((e > 0).mean()), "n": int(len(e))}


def cluster_bootstrap(stat_fn, root, n_boot: int = 1000, seed: int = 0, **arrays) -> dict:
    """95% interval of `stat_fn(**arrays)` resampling ROOTS (clusters), not tapes."""
    root = np.asarray(root)
    if root.ndim != 1 or not len(root):
        raise ValueError("Bootstrap requires at least one root cluster")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if any(len(np.asarray(v)) != len(root) for v in arrays.values()):
        raise ValueError("Bootstrap arrays must match the root array length")
    ids = np.unique(root)
    idx_by_root = {r: np.where(root == r)[0] for r in ids}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        sel = np.concatenate([idx_by_root[r] for r in pick])
        vals.append(stat_fn(**{k: np.asarray(v)[sel] for k, v in arrays.items()}))
    vals = np.array(vals, float)
    ok = ~np.isnan(vals)
    point = stat_fn(**arrays)
    return {"point": float(point), "lo": float(np.percentile(vals[ok], 2.5)) if ok.any() else float("nan"),
            "hi": float(np.percentile(vals[ok], 97.5)) if ok.any() else float("nan"), "n_boot": int(ok.sum())}


def paired_difference(stat_fn, root, n_boot: int = 1000, seed: int = 0, **pairs) -> dict:
    """Paired (same tapes) difference stat_fn(B) - stat_fn(A) with a root bootstrap.

    `pairs` holds arrays with suffixes `_a` and `_b`, e.g. c_hat_a, c_hat_b, u.
    """
    a = {k[:-2]: v for k, v in pairs.items() if k.endswith("_a")}
    b = {k[:-2]: v for k, v in pairs.items() if k.endswith("_b")}
    shared = {k: v for k, v in pairs.items() if not (k.endswith("_a") or k.endswith("_b"))}

    def diff(**arr):
        aa = {k: arr[f"{k}_a"] for k in a} | {k: arr[k] for k in shared}
        bb = {k: arr[f"{k}_b"] for k in b} | {k: arr[k] for k in shared}
        return stat_fn(**bb) - stat_fn(**aa)

    return cluster_bootstrap(diff, root, n_boot=n_boot, seed=seed, **pairs)
