"""Body-specific Push-T contacts observed without changing the physics stepping loop.

The upstream n_contacts is a rounded average over all collision types, including walls.
This observer preserves that callback/counter and separately sums contact points between
specific bodies across the physics substeps of each env.step. A positive sum establishes
contact during that environment step, even if bodies have separated by its endpoint.

Use make_env and these replay functions for new collections. Legacy unwrapped environments
remain supported and return the original logs with no inferred body-specific contact data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from helpers import pushtReplay as legacy

CONTACT_KIND = "pusher_block"
CONTACT_COUNTER = "post_solve_contact_points_summed_over_physics_substeps_v1"
CONTACT_FIELDS = ("pusher_block_contacts", "block_wall_contacts")


@dataclass
class ContactDenseLog(legacy.DenseLog):
    pusher_block_contacts: np.ndarray | None = None
    block_wall_contacts: np.ndarray | None = None
    contact_kind: str = CONTACT_KIND
    contact_counter: str = CONTACT_COUNTER


class ContactObserverEnv:
    """Add a passive post-solve observer while retaining the original callback exactly."""

    def __init__(self, env):
        self.env = env
        self.contact_history = []
        self._recording = False
        self._counts = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self.contact_history = []
        self._recording = False
        u = self.env.unwrapped
        original = u._handle_collision

        def post_solve(arbiter, space, data):
            original(arbiter, space, data)
            if not self._recording:
                return
            a, b = (shape.body for shape in arbiter.shapes)
            points = len(arbiter.contact_point_set.points)
            if ((a is u.agent and b is u.block) or (b is u.agent and a is u.block)):
                self._counts["pusher_block_contacts"] += points
            elif ((a is u.block and b is space.static_body) or (b is u.block and a is space.static_body)):
                self._counts["block_wall_contacts"] += points
            else:
                self._counts["other_contacts"] += points

        # PushT uses collision type zero for all shapes. This adds no callback that
        # changes collision decisions, impulses, positions, velocities or step order.
        u.space.on_collision(0, 0, post_solve=post_solve)
        self._post_solve = post_solve
        return result

    def step(self, action):
        self._counts = {name: 0 for name in (*CONTACT_FIELDS, "other_contacts")}
        self._recording = True
        try:
            observation, reward, terminated, truncated, info = self.env.step(action)
        finally:
            self._recording = False
        counts = dict(self._counts)
        counts["all_contact_points"] = sum(counts.values())
        if counts["all_contact_points"] != self.env.unwrapped.n_contact_points:
            raise RuntimeError("Body-specific contact observer did not preserve the upstream collision count")
        self.contact_history.append(counts)
        info = {**info, **counts, "contact_kind": CONTACT_KIND, "contact_counter": CONTACT_COUNTER}
        return observation, reward, terminated, truncated, info


def make_env():
    return ContactObserverEnv(legacy.make_env())


def _history(env):
    # MeteredEnv delegates these attributes, so accounting wrappers remain effective.
    return getattr(env, "contact_history", None)


def _attach(log, counts):
    if counts is None:
        return log
    if len(counts) != log.executed_steps:
        raise RuntimeError("Contact history does not align with the observed transition count")
    data = {}
    for field in CONTACT_FIELDS:
        value = np.zeros(len(log.states), dtype=np.int64)
        value[1:log.executed_steps + 1] = [row[field] for row in counts]
        data[field] = value
    return ContactDenseLog(**vars(log), **data)


def run_actions(env, actions, **kwargs):
    history = _history(env)
    start = len(history) if history is not None else None
    log = legacy.run_actions(env, actions, **kwargs)
    return _attach(log, None if start is None else _history(env)[start:])


def reset_root(env, root, *, record_frames=True):
    ctx = legacy.reset_root(env, root, record_frames=record_frames)
    ctx.prefix_log = _attach(ctx.prefix_log, _history(env))
    return ctx


def execute_tape(env, tape, *, record_frames=True):
    return run_actions(env, tape, record_frames=record_frames)


def execute_branch(env, root, tape, *, record_frames=True):
    ctx = reset_root(env, root, record_frames=record_frames)
    return ctx, execute_tape(env, tape, record_frames=record_frames)
