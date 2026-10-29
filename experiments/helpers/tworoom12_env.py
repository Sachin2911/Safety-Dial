"""TwoRoom with the goal frame in `info`, registered under its own id (task 12).

stable_worldmodel's Push-T puts a rendered goal image in `info["goal"]`; TwoRoom keeps it
private (`_target_img`, the agent drawn at the target) because LeWM evaluated TwoRoom from
dataset goals. The planning policy needs `info["goal"]`, so this subclass exposes the same
frame. Dynamics, rendering and the default geometry are untouched.
"""

from __future__ import annotations

import gymnasium as gym
from stable_worldmodel.envs.two_room.env import TwoRoomEnv

ENV_ID = "tworoom12/TwoRoomGoal-v1"


class GoalTwoRoomEnv(TwoRoomEnv):
    def _get_info(self):
        info = super()._get_info()
        if self._target_img is not None:
            info["goal"] = self._target_img.permute(1, 2, 0).cpu().numpy()
        return info


def register():
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="helpers.tworoom12_env:GoalTwoRoomEnv")
    return ENV_ID
