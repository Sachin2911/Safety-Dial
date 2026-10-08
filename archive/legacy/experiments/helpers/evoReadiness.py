"""Prospective whole-episode roots and paired baseline measurement readiness."""
from __future__ import annotations

import numpy as np

from helpers.evoReal import _step
from helpers.evoStats import wilson
from helpers.walkerRules import health_clearance, rule_unsafe, roots_from_episode


def generate_episode(env, actor, *, seed, max_steps, counter):
    """Record a complete new teacher episode, stopping at dense health failure."""
    env.reset(seed=int(seed))
    actor.reset()
    u = env.unwrapped
    qp, qv, xv = np.empty((max_steps+1, 9)), np.empty((max_steps+1, 9)), np.empty(max_steps)
    actions = np.empty((max_steps, 6), dtype=np.float32)
    qp[0], qv[0] = u.data.qpos.copy(), u.data.qvel.copy()
    n = 0
    for t in range(max_steps):
        counter['teacher_queries'] += 1
        action = np.asarray(actor.act(u._get_obs()), dtype=np.float32)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise FloatingPointError('invalid source actor action')
        actions[t] = np.clip(action, -1, 1)
        counter['real_steps'] += 1
        _step(u, actions[t], qp, qv, xv, t)
        n = t+1
        if bool(rule_unsafe('health', health_clearance(qp[n, 1], qp[n, 2]))):
            break
    return dict(qpos=qp[:n+1], qvel=qv[:n+1], action=actions[:n], x_velocity=xv[:n])


def representative_root(episode, *, episode_id, rng):
    """Same surviving-future conditioning as the original representative root bank."""
    n = len(episode['action'])
    if n <= 120:
        return None
    t = int(rng.integers(20, n-100))
    # roots_from_episode expects pre-action states, not the terminal extra state.
    rows = dict(episode, qpos=episode['qpos'][:-1], qvel=episode['qvel'][:-1])
    return roots_from_episode(rows, [t], episode=int(episode_id), prefix='fresh')[0]


def baseline_audit(real, imagined, progress, reference_progress, *, n_boot, seed,
                   min_progress_ratio=.5, max_failure_upper=.9, max_gap_width=.2):
    """One independent root per episode; paired resampling for gap and progress ratio.

    Failure probability uses Wilson bounds to avoid a zero-width interval when all
    observations agree. No upper failure bound here is a safe-control guarantee.
    """
    real, imagined, progress, reference_progress = [np.asarray(x) for x in
        [real, imagined, progress, reference_progress]]
    n = len(real)
    if n < 2 or any(x.shape != (n,) for x in [imagined, progress, reference_progress]):
        raise ValueError('matching one-dimensional episode arrays required')
    if not all(np.isfinite(x).all() for x in [real, imagined, progress, reference_progress]):
        raise ValueError('finite complete outcomes required')
    if not (np.isin(real, [0, 1]).all() and np.isin(imagined, [0, 1]).all()):
        raise ValueError('violations must be binary')
    rng = np.random.default_rng(seed)
    idx = rng.integers(n, size=(n_boot, n))
    gap = real.astype(float)-imagined.astype(float)
    gap_ci = np.quantile(gap[idx].mean(1), [.025, .975])
    denominator = reference_progress[idx].mean(1)
    valid_reference = bool(reference_progress.mean() > 0 and np.all(denominator > 0))
    if valid_reference:
        ratios = progress[idx].mean(1)/denominator
        ratio_ci = np.quantile(ratios, [.025, .975])
        ratio = dict(point=float(progress.mean()/reference_progress.mean()),
                     lo=float(ratio_ci[0]), hi=float(ratio_ci[1]))
    else:
        ratio = dict(point=None, lo=None, hi=None)
    failure_ci = wilson(int(real.sum()), n)
    checks = dict(valid_reference=valid_reference,
        progress_retained=bool(valid_reference and ratio['lo'] >= min_progress_ratio),
        failure_headroom=bool(failure_ci[1] < max_failure_upper),
        baseline_gap_precision=bool(gap_ci[1]-gap_ci[0] <= max_gap_width))
    return dict(n_independent_episodes=n, checks=checks, research_pilot_ready=all(checks.values()),
        progress_ratio=ratio, failure=dict(count=int(real.sum()), point=float(real.mean()),
                                           lo=float(failure_ci[0]), hi=float(failure_ci[1]), method='Wilson'),
        gap=dict(point=float(gap.mean()), lo=float(gap_ci[0]), hi=float(gap_ci[1]),
                 width=float(gap_ci[1]-gap_ci[0]), method='paired episode bootstrap'),
        n_boot=n_boot, seed=seed, original_gate0_passed=False, safe_controller_claim=False)


def reference_arrays(logs):
    """Convert the validated physics-only tape interface to audit arrays."""
    qp = np.stack([log.qpos for log in logs])
    qv = np.stack([log.qvel for log in logs])
    unsafe = rule_unsafe('health', health_clearance(qp[:, 1:, 1], qp[:, 1:, 2]))
    return dict(qpos=qp, qvel=qv, actions=np.stack([log.actions for log in logs]),
                dense_violated=unsafe.any(axis=1), progress=qp[:, -1, 0]-qp[:, 0, 0])
