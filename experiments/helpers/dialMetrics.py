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


def decisions(c_hat: np.ndarray, m: float) -> np.ndarray:
    return np.asarray(c_hat) >= m


def fsa(c_hat, u, m: float) -> dict:
    a = decisions(c_hat, m)
    u = np.asarray(u).astype(bool)
    n_acc = int(a.sum())
    out = {"m": float(m), "n": int(len(a)), "n_accepted": n_acc, "acceptance_rate": n_acc / max(len(a), 1)}
    out["n_false_safe"] = int((a & u).sum())
    out["fsa"] = out["n_false_safe"] / n_acc if n_acc else float("nan")
    n_safe = int((~u).sum())
    out["false_reject_rate"] = int(((~a) & (~u)).sum()) / n_safe if n_safe else float("nan")
    return out


def dial_curve(c_hat, u, margins) -> list[dict]:
    return [fsa(c_hat, u, m) for m in margins]


def margin_for_acceptance(c_hat, target_ar: float) -> float:
    """The margin m on THIS set such that AR(m) is closest to target_ar (from above)."""
    c = np.sort(np.asarray(c_hat))
    n = len(c)
    k = int(round(target_ar * n))  # number to accept
    k = min(max(k, 0), n)
    if k == 0:
        return float(c[-1]) + 1e-9
    return float(c[n - k])


def matched_fsa(c_dev, u_dev, c_test, u_test, target_ar: float) -> dict:
    """Choose m on development to hit target_ar; apply unchanged on test."""
    m = margin_for_acceptance(c_dev, target_ar)
    r = fsa(c_test, u_test, m)
    r["target_acceptance_rate"] = float(target_ar)
    r["dev"] = fsa(c_dev, u_dev, m)
    return r


def auc_dial(c_hat, u, ar_lo: float = 0.2, ar_hi: float = 0.9, n_grid: int = 50) -> float:
    """Area under FSA against acceptance rate over [ar_lo, ar_hi] (trapezoid, NaN-safe)."""
    ars = np.linspace(ar_lo, ar_hi, n_grid)
    vals = []
    for ar in ars:
        m = margin_for_acceptance(c_hat, ar)
        vals.append(fsa(c_hat, u, m)["fsa"])
    vals = np.array(vals, float)
    ok = ~np.isnan(vals)
    if ok.sum() < 2:
        return float("nan")
    return float(np.trapezoid(vals[ok], ars[ok]) / (ars[ok][-1] - ars[ok][0]))


def clearance_error_stats(c_hat, c_true) -> dict:
    """Signed clearance error e = c_hat_min - c_min (positive = optimistic)."""
    e = np.asarray(c_hat) - np.asarray(c_true)
    return {"mean": float(e.mean()), "p50": float(np.percentile(e, 50)), "p90": float(np.percentile(e, 90)),
            "p95": float(np.percentile(e, 95)), "p99": float(np.percentile(e, 99)), "frac_optimistic": float((e > 0).mean()), "n": int(len(e))}


def cluster_bootstrap(stat_fn, root, n_boot: int = 1000, seed: int = 0, **arrays) -> dict:
    """95% interval of `stat_fn(**arrays)` resampling ROOTS (clusters), not tapes."""
    root = np.asarray(root)
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
