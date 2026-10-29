"""OGBench Cube (single) with a dataset goal and a state record in `info` (task 14).

LeWM evaluates Cube on dataset pairs: start from a recorded simulator state, plan toward
the frame recorded 25 steps later, and succeed once the cube is within 0.04 m of where it
was then. This subclass takes that goal frame and goal cube position as reset options. The
default success rule is the environment's own 0.04 m test, measured against the goal cube
position. With `goal_effector_pos` (and `effector_radius_m`) the episode instead succeeds
when the gripper's pinch point is that close to the goal frame's, the reach task of the
dial study (the free planner cannot complete carries; see the task-14 diagnosis). The
translucent target marker, which only matters for rendering, can be drawn elsewhere (the
dataset frames show the data-collection target there). `info["goal"]` holds the goal
frame and `info["state"]` the effector position, cube position and gripper opening after
every env step, which the runtime saves. The arm is drawn opaque, as in the dataset's frames
(OGBench's pixel default draws it translucent). Dynamics and the controller are untouched.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

ENV_ID = "cube14/CubeGoal-v0"
SUCCESS_RADIUS_M = 0.04


class GoalCubeEnv(CubeEnv):
    def __init__(self, env_type="single", ob_type="pixels", pixel_transparent_arm=False, **kwargs):
        self._goal_image = None  # before the parent's setup, which may read `info`
        self._goal_cube_pos = None
        self._goal_effector = None
        self._effector_radius = None
        super().__init__(
            env_type=env_type,
            ob_type=ob_type,
            pixel_transparent_arm=pixel_transparent_arm,
            **kwargs,
        )

    def reset(self, seed=None, options=None):
        options = dict(options or {})
        goal_image = options.pop("goal_image", None)
        goal_cube = options.pop("goal_cube_pos", None)
        marker = options.pop("marker_pos", None)
        goal_effector = options.pop("goal_effector_pos", None)
        self._effector_radius = float(options.pop("effector_radius_m", SUCCESS_RADIUS_M))
        self._goal_cube_pos = None if goal_cube is None else np.asarray(goal_cube, dtype=float)
        self._goal_effector = None
        if goal_effector is not None:
            self._goal_effector = np.asarray(goal_effector, dtype=float)
        if "state" in options and options["state"] is not None:
            options["state"] = np.asarray(options["state"], dtype=np.float64)
        super().reset(seed=seed, options=options)
        if marker is None:
            marker = goal_cube
        if marker is not None:
            self.set_target_pos(0, np.asarray(marker, dtype=np.float64))
            mujoco.mj_forward(self._model, self._data)
        self.post_step()  # success against the goal cube position, marker visibility
        self._success = bool(self._compute_successes()[0])
        self._goal_image = None if goal_image is None else np.asarray(goal_image).copy()
        info = self.get_reset_info()
        return self.compute_observation(), self._decorate(info)

    def _compute_successes(self):
        if self._goal_effector is not None:
            effector = self._data.site_xpos[self._pinch_site_id]
            return [bool(np.linalg.norm(effector - self._goal_effector) <= self._effector_radius)]
        if self._goal_cube_pos is None:
            return super()._compute_successes()
        cube = self._data.joint("object_joint_0").qpos[:3]
        return [bool(np.linalg.norm(cube - self._goal_cube_pos) <= SUCCESS_RADIUS_M)]

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, self._decorate(info)

    def _decorate(self, info):
        effector = self._data.site_xpos[self._pinch_site_id].copy()
        cube = self._data.joint("object_joint_0").qpos[:3].copy()
        opening = np.clip(self._data.qpos[self._gripper_opening_joint_id] / 0.8, 0, 1)
        info["state"] = np.concatenate([effector, cube, [opening]]).astype(np.float64)
        if self._goal_image is not None:
            info["goal"] = self._goal_image
        return info


def register():
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="helpers.cube14_env:GoalCubeEnv")
    return ENV_ID
