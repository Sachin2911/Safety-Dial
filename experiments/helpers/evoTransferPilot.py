"""Finite-sample ranking diagnostics and conditional pilot effect precision."""
import math

import numpy as np
from scipy.stats import norm

from helpers.evoStats import clustered_mean_ci, clustered_spearman


def rank_audit(imagined, real, episodes, *, n_boot, seed):
    out = clustered_spearman(imagined, real, episodes, n_boot=n_boot, seed=seed)
    out['requested_bootstraps'] = n_boot
    out['valid_bootstrap_fraction'] = out['n_boot']/n_boot
    out['defined'] = bool(np.isfinite(out['point']))
    out['transfer_screen_pass'] = bool(out['defined'] and out['point'] >= .5
        and np.isfinite(out['lo']) and out['lo'] > 0 and out['valid_bootstrap_fraction'] >= .9)
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in out.items()}


def selected_effect(real, imagined, progress, *, selected, assessment_indices, episodes,
                    n_boot, seed, target_effect=.1):
    """All policy arrays are (candidates, roots); selected is fixed before real outcomes.

    Root-only normal-approximation sizing is conditional on this one selected policy
    and is explicitly not power for a study with random search seeds.
    """
    r, i, p = [np.asarray(a) for a in [real, imagined, progress]]
    if r.ndim != 2 or r.shape != i.shape or r.shape != p.shape:
        raise ValueError('matching candidate-by-root arrays required')
    idx = np.asarray(assessment_indices, dtype=int)
    if len(np.unique(idx)) != len(idx) or len(idx) < 2:
        raise ValueError('distinct assessment roots required')
    gap_delta = (r[selected, idx].astype(float)-i[selected, idx])-(r[0, idx].astype(float)-i[0, idx])
    clusters = np.asarray(episodes)[idx]
    ci = clustered_mean_ci(gap_delta, clusters, n_boot=n_boot, seed=seed)
    delta_real = clustered_mean_ci(r[selected, idx].astype(float)-r[0, idx], clusters, n_boot=n_boot, seed=seed)
    delta_progress = clustered_mean_ci(p[selected, idx]-p[0, idx], clusters, n_boot=n_boot, seed=seed)
    variance = float(np.var(gap_delta, ddof=1))
    multiplier = float((norm.ppf(.975)+norm.ppf(.8))**2/target_effect**2)
    return dict(selected_index=int(selected), n_assessment_roots=len(idx), gap_change=ci,
        real_violation_change=delta_real, progress_change=delta_progress,
        root_only_power_planning=dict(target_absolute_effect=target_effect, power=.8, alpha=.05,
            observed_paired_variance=variance,
            plug_in_n=math.ceil(multiplier*variance) if variance > 0 else None,
            range_bound_variance=4., worst_variance_n=math.ceil(multiplier*4.),
            warning='conditional single-winner normal approximation; search-seed variance unmeasured'))
