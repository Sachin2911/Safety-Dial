"""Clean source-teacher block targets at the original cached visual inputs."""
from __future__ import annotations

from types import SimpleNamespace

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from helpers.evoReal import _start, _step
from helpers.walkerRules import health_clearance, rule_unsafe


def cached_steps(path, episodes, targets, *, stride=5):
    """Recover cache times and verify against every recorded forward action block."""
    episodes, targets = np.asarray(episodes), np.asarray(targets)
    if targets.shape != (len(episodes), 60):
        raise ValueError('expected one 60-action target per cached row')
    times = np.empty(len(episodes), dtype=np.int64)
    with h5py.File(path, 'r') as f:
        offsets, lengths = f['ep_offset'][:], f['ep_len'][:]
        for e in np.unique(episodes):
            lo, n = int(offsets[e]), int(lengths[e])
            actions = f['action'][lo:lo+n]
            t = np.arange(0, n, stride)
            t = t[(t >= 10) & (t+10 <= n)]
            idx = np.flatnonzero(episodes == e)
            if len(idx) != len(t):
                raise ValueError('cached episode row count differs from reconstruction')
            wanted = np.stack([actions[s:s+10].reshape(60) for s in t]).astype(np.float32)
            if not np.array_equal(wanted, targets[idx].astype(np.float32)):
                raise ValueError('cached targets disagree with temporal alignment')
            times[idx] = t
    return times


def check_fit_indices(episodes, indices, allowed_fit, validation):
    episodes, indices = np.asarray(episodes), np.asarray(indices)
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        raise ValueError('fit indices must be one-dimensional and unique')
    if len(indices) == 0 or indices.min() < 0 or indices.max() >= len(episodes):
        raise ValueError('fit indices out of range or empty')
    fit, val = set(map(int, allowed_fit)), set(map(int, validation))
    if fit.intersection(val) or not set(map(int, episodes[indices])).issubset(fit):
        raise ValueError('teacher relabelling must use fit episodes only')


def clean_teacher_block(env, qpos, qvel, teacher, counter):
    """New independent root, then ten exact feedback steps without solver resets.

    This is not a takeover in an existing rollout. Each cached state starts a new
    branch with the same reset convention as the real evaluation harness.
    Counter is updated before each attempted simulator step, including exceptions.
    """
    qp, qv, xv = np.empty((11, 9)), np.empty((11, 9)), np.empty(10)
    _start(env, SimpleNamespace(qpos=qpos, qvel=qvel), qp, qv)
    teacher.reset()
    actions = np.empty((10, 6))
    for t in range(10):
        counter['teacher_queries'] += 1
        action = np.asarray(teacher.act(env.unwrapped._get_obs()))
        if action.shape != (6,) or not np.isfinite(action).all():
            raise FloatingPointError('invalid teacher action')
        actions[t] = np.clip(action, -1, 1)
        counter['real_steps'] += 1
        _step(env.unwrapped, actions[t], qp, qv, xv, t)
    clearance = health_clearance(qp[1:, 1], qp[1:, 2])
    unsafe = rule_unsafe('health', clearance)
    return dict(actions=actions, qpos=qp, qvel=qv, clearance=clearance,
                violated=bool(unsafe.any()), progress=float(qp[-1, 0]-qp[0, 0]))
