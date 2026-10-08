"""Paired teacher-labelled augmentation, preserving episode roles and solver prefixes."""
from __future__ import annotations

import numpy as np

from helpers.evoReal import _start, _step


def select_training_starts(fit_episodes, validation_episodes, lengths, *, count, seed):
    """One root per sampled fit episode, never use validation or health outcomes."""
    fit = np.asarray(fit_episodes, dtype=int)
    if len(np.unique(fit)) != len(fit) or set(fit).intersection(validation_episodes):
        raise ValueError('fit episodes must be unique and disjoint from validation')
    lengths = np.asarray(lengths)
    eligible = fit[lengths[fit] >= 120]
    if len(eligible) < count or count < 1:
        raise ValueError('insufficient full-horizon fit episodes')
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(eligible, count, replace=False))
    return [(int(e), int(rng.integers(20, lengths[e] - 100 + 1))) for e in selected]


def augmentation_indices(original_count, new_count, *, fraction, seed):
    """Keep the old epoch length; repeat new data nearly uniformly, not outcome-weighted."""
    if original_count < 1 or new_count < 1 or not 0 < fraction < 1:
        raise ValueError('need nonempty pools and a mixture fraction strictly between zero and one')
    n_new = int(original_count * fraction)
    if n_new < 1:
        raise ValueError('mixture contains no new examples')
    rng = np.random.default_rng(seed)
    old = np.sort(rng.choice(original_count, original_count - n_new, replace=False))
    new = np.concatenate([rng.permutation(new_count)
                          for _ in range((n_new + new_count - 1) // new_count)])[:n_new]
    return old, new


def teacher_label(env, root, tape, teacher, *, boundary):
    """Replay prefix from the root, then query a ten-step teacher branch.

    Return the label, full prefix/branch states, and charged attempted simulation calls.
    The caller verifies the entire prefix against the trajectory supplying its features.
    """
    tape = np.asarray(tape, dtype=np.float64)
    if tape.shape != (100, 6) or boundary not in range(0, 100, 10):
        raise ValueError('expected (100, 6) tape and a ten-step boundary')
    steps = int(boundary) + 10
    qp, qv, xv = np.empty((steps + 1, 9)), np.empty((steps + 1, 9)), np.empty(steps)
    _start(env, root, qp, qv)
    teacher.reset()
    target = []
    for t in range(steps):
        action = tape[t] if t < boundary else teacher.act(env.unwrapped._get_obs())
        if not np.isfinite(action).all():
            raise FloatingPointError('nonfinite teacher branch action')
        if t >= boundary:
            target.append(np.clip(action, -1, 1))
        _step(env.unwrapped, action, qp, qv, xv, t)
    return np.asarray(target).reshape(60), qp, qv, steps
