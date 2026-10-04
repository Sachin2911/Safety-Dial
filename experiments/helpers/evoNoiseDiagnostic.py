"""Disjoint episode roles and time-aligned cached windows for a bounded noise audit."""
import numpy as np


def calibration_split(n_roots, excluded, *, n_fit=64, n_check=64, seed=20261022):
    excluded = np.asarray(excluded, dtype=np.int64)
    if len(np.unique(excluded)) != len(excluded) or (excluded < 0).any() or (excluded >= n_roots).any():
        raise ValueError('unique valid excluded roots required')
    available = np.setdiff1d(np.arange(n_roots), excluded)
    if len(available) < n_fit+n_check or min(n_fit, n_check) < 1:
        raise ValueError('not enough disjoint calibration episodes')
    order = np.random.default_rng(seed).permutation(available)
    return order[:n_fit], order[n_fit:n_fit+n_check]


def trace_windows(history_z, history_actions, future_z, future_actions):
    """Root-major one-step windows, each ending at the next recorded real latent.

    Inputs are real encoder histories (R,3,D), raw action histories (R,2,10,6),
    and real future latents/actions (R,K,D)/(R,K,60). No imaginary states enter.
    Actions are returned raw; the caller applies the frozen model scaler once.
    """
    hz, ha, fz, fa = map(np.asarray, [history_z, history_actions, future_z, future_actions])
    if hz.ndim != 3 or fz.ndim != 3 or hz.shape[1] != 3:
        raise ValueError('three history frames and block-end future latents required')
    r, k, d = fz.shape
    if hz.shape != (r, 3, d) or ha.shape != (r, 2, 10, 6) or fa.shape != (r, k, 60) or k < 1:
        raise ValueError('incompatible recorded trace shapes')
    if not all(np.isfinite(a).all() for a in [hz, ha, fz, fa]):
        raise ValueError('nonfinite recorded trace')
    zs = np.concatenate([hz, fz], axis=1)
    acts = np.concatenate([ha.reshape(r, 2, 60), fa], axis=1)
    zw = np.stack([zs[:, j:j+4] for j in range(k)], axis=1).reshape(r*k, 4, d)
    aw = np.stack([acts[:, j:j+3] for j in range(k)], axis=1).reshape(r*k, 3, 60)
    return zw, aw, zs, acts


def residual_summary(residuals):
    r = np.asarray(residuals, np.float64)
    if r.ndim != 2 or len(r) < 2 or not np.isfinite(r).all():
        raise ValueError('at least two finite residual rows required')
    mean = r.mean(0)
    total = np.mean(r*r)
    return dict(rows=len(r), latent_dimensions=r.shape[1], rms=float(np.sqrt(total)),
                bias_rms=float(np.sqrt(np.mean(mean*mean))),
                bias_energy_fraction=None if total == 0 else float(np.mean(mean*mean)/total),
                std_rms=float(np.sqrt(np.mean(r.var(0, ddof=1)))))
