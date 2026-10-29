"""DMC Reacher (qpos_match) with a goal frame and a state record in `info` (task 13).

The planning policy reads the goal image from `info["goal"]`; this subclass renders the arm
at the target pose once per reset (the target sphere stays hidden, as in the default
rendering) and then restores the start state. `info["state"]` holds the joint positions,
joint velocities and the fingertip position after every env step, which the runtime saves.
Dynamics, the success rule and rendering are untouched.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper

ENV_ID = "reacher13/ReacherGoal-v0"


class GoalReacherEnv(ReacherDMControlWrapper):
    def __init__(self, task="qpos_match", **kwargs):
        self._goal_image = None  # before the parent's setup, which may read `info`
        super().__init__(task=task, **kwargs)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        target = (options or {}).get("target_qpos")
        self._goal_image = None if target is None else self.render_pose(target)
        return obs, self.info

    def render_pose(self, qpos):
        """Frame of the arm at `qpos` at rest, leaving the simulator state unchanged."""
        physics = self.env.physics
        saved_qpos = physics.data.qpos.copy()
        saved_qvel = physics.data.qvel.copy()
        self.set_state(np.asarray(qpos, dtype=np.float64), np.zeros_like(saved_qvel))
        frame = np.asarray(self.render()).copy()
        self.set_state(saved_qpos, saved_qvel)
        return frame

    @property
    def info(self):
        info = super().info
        physics = self.env.physics
        finger = np.asarray(physics.named.data.geom_xpos["finger"][:2], dtype=np.float64)
        info["state"] = np.concatenate([info["qpos"], info["qvel"], finger])
        if self._goal_image is not None:
            info["goal"] = self._goal_image
        return info


def register():
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="helpers.reacher13_env:GoalReacherEnv")
    return ENV_ID
