"""Walker decomposition by elapsed horizon and declared physical regimes.

All horizon decisions are cumulative from the root. Speed regimes use the root's
signed velocity and a 0.3 m/s band around the supplied limit. Health regimes use
the root's normalized clearance, with 0.5 as the near-boundary cutoff.
"""
from __future__ import annotations

import numpy as np

from helpers.dialMetrics import auc_dial, clearance_error_stats, cluster_bootstrap, fsa
from helpers.walkerRules import DT, FRAMESKIP, SPEED_LIMIT, health_clearance, rule_unsafe

SOURCES = ("dense", "endpoint", "real_readout", "imagined")
REGIME_FIELDS = ("kind", "speed_regime", "health_regime")
REGIME_DEFINITIONS = {"speed_band_m_per_s": 0.3, "health_near_clearance": 0.5,
                      "definition": "strata use root state; horizons are cumulative"}


def cumulative_clearance(clearance, frameskip=FRAMESKIP):
    values = np.asarray(clearance, dtype=np.float64)
    if values.ndim != 1 or not len(values) or len(values) % frameskip:
        raise ValueError("Dense clearance must contain complete action blocks")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite clearance must not become a safe outcome")
    return np.minimum.accumulate(values)[frameskip - 1::frameskip].tolist()


def root_regimes(root):
    speed = float(root.meta["x_velocity"])
    if speed < 0:
        speed_regime = "reverse"
    elif abs(speed - SPEED_LIMIT) <= 0.3:
        speed_regime = "near_limit"
    elif speed < SPEED_LIMIT:
        speed_regime = "below_limit"
    else:
        speed_regime = "above_limit"
    health = float(health_clearance(root.qpos[1], root.qpos[2]))
    health_regime = "unhealthy" if health <= 0 else "near_band" if health <= 0.5 else "interior"
    return {"speed_regime": speed_regime, "health_regime": health_regime,
            "root_speed": speed, "root_health_clearance": health,
            "policy_tape_padding_steps": int(root.meta.get("policy_tape_padding_steps", 0))}


def at_horizon(rows, block):
    if block < 1:
        raise ValueError("Horizon blocks are one-indexed")
    sliced = []
    for row in rows:
        item = dict(row)
        for key, values in row.items():
            if key.startswith("cmin_by_block_"):
                if block > len(values):
                    raise ValueError("Requested horizon exceeds stored trajectory")
                item[key.replace("cmin_by_block_", "cmin_", 1)] = values[block - 1]
        sliced.append(item)
    return sliced


def source_summary(rows, rule, margin=0., *, n_boot=0):
    if not rows:
        return {"n": 0, "n_source_episodes": 0}
    true = np.array([r[f"cmin_dense_{rule}"] for r in rows])
    unsafe = rule_unsafe(rule, true)
    roots = np.array([r["root"] for r in rows])
    out = {"n": len(rows), "n_source_episodes": len(np.unique(roots)),
           "n_unsafe": int(unsafe.sum()), "margin": float(margin)}
    present = [s for s in SOURCES if all(f"cmin_{s}_{rule}" in r for r in rows)]
    accepted = {}
    for source in present:
        clearance = np.array([r[f"cmin_{source}_{rule}"] for r in rows])
        out[source] = fsa(clearance, unsafe, margin)
        accepted[source] = clearance >= margin
        if source != "dense":
            out[source]["clearance_error"] = clearance_error_stats(clearance, true)
        if n_boot:
            out[source]["fsa_ci"] = cluster_bootstrap(
                lambda c, u: fsa(c, u, margin)["fsa"], roots, n_boot=n_boot,
                c=clearance, u=unsafe)
    if all(s in accepted for s in ("endpoint", "real_readout", "imagined")):
        fs = unsafe & accepted["imagined"]
        out["attribution"] = {
            "n_false_safe_imagined": int(fs.sum()),
            "temporal": int((fs & accepted["endpoint"]).sum()),
            "readout": int((fs & ~accepted["endpoint"] & accepted["real_readout"]).sum()),
            "imagination": int((fs & ~accepted["endpoint"] & ~accepted["real_readout"]).sum())}
    return out


def decomposition_report(rows, rule, margin=0., *, n_boot=300):
    if not rows:
        raise ValueError("Decomposition needs at least one branch")
    key = f"cmin_by_block_dense_{rule}"
    if any(key not in row for row in rows):
        raise ValueError("Horizon attribution requires per-block source clearances")
    blocks = len(rows[0][key])
    horizons = {str(k): {"seconds": k * FRAMESKIP * DT,
                        **source_summary(at_horizon(rows, k), rule, margin)}
                for k in range(1, blocks + 1)}
    by_regime = {}
    for field in REGIME_FIELDS:
        by_regime[field] = {}
        for value in sorted({r[field] for r in rows}):
            subset = [r for r in rows if r[field] == value]
            by_regime[field][value] = {"overall": source_summary(subset, rule, margin),
                "by_horizon": {str(k): source_summary(at_horizon(subset, k), rule, margin)
                               for k in range(1, blocks + 1)}}
    return {"overall": source_summary(rows, rule, margin, n_boot=n_boot),
            "by_horizon": horizons, "by_regime": by_regime,
            "regime_definitions": REGIME_DEFINITIONS,
            "uncertainty": "overall 95% bootstrap intervals resample source episodes; stratum tables report counts"}


def merge_frozen_sources(rows, reference_rows):
    """Keep the physical readout fixed when only predictor-side weights change."""
    def identity(row):
        return row["root_id"], row["branch"]
    if [identity(r) for r in rows] != [identity(r) for r in reference_rows]:
        raise ValueError("Frozen-source reference is not paired with evaluated branches")
    result = []
    for row, reference in zip(rows, reference_rows, strict=True):
        merged = dict(row)
        for key, value in reference.items():
            if key.startswith(("cmin_real_readout_", "cmin_by_block_real_readout_")):
                merged[key] = value
        result.append(merged)
    return result


def imagined_summary(rows, rule, margin, *, n_boot=1000):
    true = np.array([r[f"cmin_dense_{rule}"] for r in rows])
    clearance = np.array([r[f"cmin_imagined_{rule}"] for r in rows])
    unsafe = rule_unsafe(rule, true)
    roots = np.array([r["root"] for r in rows])
    ordinary = [row for row in rows if row["kind"] == "policy"]
    return {"margin": float(margin), "at_matched": fsa(clearance, unsafe, margin),
            "fsa_ci": cluster_bootstrap(lambda c, u: fsa(c, u, margin)["fsa"], roots,
                                         n_boot=n_boot, c=clearance, u=unsafe),
            "auc_dial": auc_dial(clearance, unsafe),
            "auc_acceptance_range": [0.2, 0.9],
            "clearance_error": clearance_error_stats(clearance, true),
            "state_mae_by_block": np.mean([r["state_abs_error_by_block"] for r in rows], axis=0).tolist(),
            "ordinary_retention": {"n": len(ordinary),
                "state_mae_by_block": np.mean([r["state_abs_error_by_block"] for r in ordinary], axis=0).tolist() if ordinary else None,
                "displacement_mae": float(np.mean([abs(r["predicted_displacement"] - r["true_displacement"]) for r in ordinary])) if ordinary else None}}
