"""Teacher queries and takeovers with the original branch's solver state intact.

Every intervention replays its entire prefix from the root. Restoring only qpos/qvel
at an intermediate boundary would reset the solver warm start and confound the change.
Teacher actions use privileged observations and are diagnostic references, not latent
controllers or a claim that a failed teacher recovery is impossible recovery.
"""
from __future__ import annotations

import numpy as np

from helpers.evoReal import _start, _step
from helpers.walkerRules import health_clearance, rule_unsafe


def teacher_branch(env, root, tape, teacher, *, takeover_step=100, feedback_interval=1):
    tape = np.asarray(tape, dtype=np.float64)
    if tape.shape != (100, 6) or not np.isfinite(tape).all():
        raise ValueError('expected a finite (100, 6) tape')
    if takeover_step not in range(101) or feedback_interval not in range(1, 101):
        raise ValueError('invalid takeover or feedback interval')
    qp, qv, xv = np.empty((101, 9)), np.empty((101, 9)), np.empty(100)
    actions, targets = np.empty((100, 6)), np.empty((100, 6))
    _start(env, root, qp, qv)
    teacher.reset()
    for t in range(100):
        targets[t] = teacher.act(env.unwrapped._get_obs())
        if not np.isfinite(targets[t]).all():
            raise FloatingPointError('nonfinite teacher action')
        if t < takeover_step:
            action = tape[t]
        else:
            if (t - takeover_step) % feedback_interval == 0:
                held = targets[t].copy()
            action = held
        actions[t] = np.clip(action, -1., 1.)
        _step(env.unwrapped, actions[t], qp, qv, xv, t)
    clearance = health_clearance(qp[1:, 1], qp[1:, 2])
    unsafe = rule_unsafe('health', clearance)
    first = int(np.flatnonzero(unsafe)[0]) if unsafe.any() else 100
    return dict(qpos=qp, qvel=qv, actions=actions, teacher_actions=targets,
                clearance=clearance, violated=bool(unsafe.any()), first_unsafe_step=first,
                progress=float(qp[-1, 0] - qp[0, 0]))


def action_error_summary(actions, targets, first_unsafe):
    """Exclude actions at/after an already unsafe state; include the action causing failure.

    first_unsafe is indexed by transition, so the pre-action state at that index is
    still healthy. Position groups can have different counts, which are reported.
    """
    actions, targets = np.asarray(actions), np.asarray(targets)
    if actions.shape != targets.shape or actions.ndim != 3 or actions.shape[1:] != (100, 6):
        raise ValueError('expected matching (N, 100, 6) actions')
    first = np.asarray(first_unsafe)
    if first.shape != (len(actions),):
        raise ValueError('one first-unsafe transition per trajectory required')
    mse = np.square(actions - targets).mean(axis=-1)
    mask = np.arange(100)[None] <= first[:, None]
    def group(selected):
        return {'n_actions': int(selected.sum()),
                'mse': float(mse[selected].mean()) if selected.any() else None}
    return {'all': group(mask), 'block_position': [
        group(mask & (np.arange(100)[None] % 10 == j)) for j in range(10)]}
