"""Native environments with explicitly counted label collection and execution.

The protected body is the navigation disk or a conservative disk about the fingertip
or pinch site, projected into world XY. Cube success remains object manipulation.
"""

import json

import numpy as np


class NativeEnv:
    def __init__(self, name, counters, journal=None):
        self.name, self.counters, self.phase = name, counters, "reset"
        self.journal = None if journal is None else open(journal, "a", buffering=1)
        self.dense = []
        if name == "tworoom":
            from helpers.tworoom12_env import GoalTwoRoomEnv

            self.env = GoalTwoRoomEnv()
        elif name == "reacher":
            from helpers.reacher13_env import GoalReacherEnv
            from dm_control.mujoco.engine import Physics

            self.original_step = Physics.step
            owner = self

            def step(physics, nstep=1):
                if int(nstep) != 1:
                    counters[owner.phase + "_physics_steps"] += int(nstep)
                    raise RuntimeError("Dense instrumentation requires nstep=1")
                result = owner.original_step(physics, nstep=nstep)
                counters[owner.phase + "_physics_steps"] += int(nstep)
                if owner.phase == "executed":
                    owner.dense.append(physics.named.data.geom_xpos["finger"][:2].copy())
                return result

            Physics.step = step
            self.env = GoalReacherEnv()
        else:
            from helpers.lewm18_runtime import DenseCube

            owner = self

            class CountedCube(DenseCube):
                def integrate(env, action):
                    before = counters["executed_physics_steps"]
                    result = super().integrate(action)
                    if owner.phase == "reset":
                        delta = counters["executed_physics_steps"] - before
                        counters["executed_physics_steps"] -= delta
                        counters["reset_physics_steps"] += delta
                        counters["reset_control_actions"] += 1
                    return result

                def preview(env, actions):
                    raise RuntimeError("Simulator candidate previews are forbidden")

            self.env = CountedCube(ob_type="states", width=224, height=224)
            self.env.work_counters = counters

    def checkpoint(self, status):
        if self.journal is not None:
            self.journal.write(
                json.dumps({"status": status, "resources": dict(self.counters)}) + "\n"
            )

    def reset(self, seed, state=None, goal=None, goal_image=None):
        self.phase = "reset"
        self.counters["environment_resets"] += 1
        self.checkpoint("reset_started")
        options = {}
        if state is not None:
            options["state"] = np.asarray(state).copy()
        self.goal = goal
        if self.name == "tworoom":
            if goal is not None:
                options["target_state"] = np.asarray(goal)
        elif self.name == "reacher":
            if goal is not None:
                options["target_qpos"] = np.asarray(goal)
        elif goal is not None:
            options.update(goal_cube_pos=np.asarray(goal), goal_image=goal_image)
        _, self.info = self.env.reset(seed=seed, options=options)
        self.phase = "executed"
        self.checkpoint("reset_complete")
        return self.observe()

    def observe(self):
        if self.name == "tworoom":
            return self.env.agent_position.numpy().copy()
        if self.name == "reacher":
            p = self.env.env.physics
            return np.r_[p.data.qpos, p.data.qvel, p.named.data.geom_xpos["finger"][:2]]
        info = self.env.compute_ob_info()
        yaw = info["proprio/effector_yaw"][0]
        return np.r_[
            info["proprio/joint_pos"],
            info["proprio/joint_vel"],
            info["proprio/gripper_opening"],
            info["proprio/gripper_vel"],
            info["proprio/gripper_contact"],
            info["proprio/effector_pos"],
            np.sin(yaw),
            np.cos(yaw),
            info["proprio/effector_pos"][:2],
        ]

    def step(self, action):
        self.dense = []
        self.counters["environment_actions"] += 1
        self.checkpoint("action_started")
        before = self.observe()[-2:]
        _, _, _, _, self.info = self.env.step(np.asarray(action))
        observed = self.observe()
        if self.name == "tworoom":
            self.dense = [observed[-2:].copy()]
            self.counters["executed_discrete_steps"] += 1
        elif self.name == "cube":
            self.dense = self.env.last_dense[:, :2].copy()
        self.last_dense = np.asarray(self.dense)
        if not len(self.last_dense):
            raise ValueError("No recorded native integration points")
        self.last_segment_start = before
        self.checkpoint("action_complete")
        return observed, self.last_dense

    def success(self):
        if self.name == "tworoom":
            return np.linalg.norm(self.observe()[-2:] - self.goal) < 16
        if self.name == "reacher":
            return np.abs(self.observe()[:2] - self.goal).max() < 0.05
        return bool(self.env._compute_successes()[0])

    def render(self):
        self.counters["render_frames"] += 1
        return np.asarray(self.env.render()).copy()

    def pose(self, state, render=False):
        """Dataset/preparation labels only. Never called by the planner."""
        self.counters["preparation_kinematic_queries"] += 1
        if self.name == "tworoom":
            return np.asarray(state)[:2].copy(), None
        if self.name == "reacher":
            p = self.env.env.physics
            q, v = p.data.qpos.copy(), p.data.qvel.copy()
            self.env.set_state(np.asarray(state)[:2], np.asarray(state)[2:4])
            point, image = self.observe()[-2:].copy(), self.render() if render else None
            self.env.set_state(q, v)
            return point, image
        import mujoco

        snapshot = mujoco.MjData(self.env._model)
        mujoco.mj_copyData(snapshot, self.env._model, self.env._data)
        nq = self.env._model.nq
        self.env._data.qpos[:] = state[:nq]
        self.env._data.qvel[:] = state[nq:]
        mujoco.mj_forward(self.env._model, self.env._data)
        point, image = self.observe()[-2:].copy(), self.render() if render else None
        cube = self.env._data.joint("object_joint_0").qpos[:3].copy()
        mujoco.mj_copyData(self.env._data, self.env._model, snapshot)
        return point, (image, cube)

    def close(self):
        self.env.close()
        if self.journal is not None:
            self.journal.close()
        if self.name == "reacher":
            from dm_control.mujoco.engine import Physics

            Physics.step = self.original_step
