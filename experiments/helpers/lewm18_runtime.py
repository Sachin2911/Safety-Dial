"""PushT and original object-goal Cube, with a shared planner and dense safety records."""

import collections
import copy
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import gymnasium as gym
import mujoco
import numpy as np
import torch

from helpers.cube14_env import GoalCubeEnv
from helpers.cube14_planner import commanded_path, densify
from helpers.cube14_runtime import DatasetRows, load_scalers
from helpers.dsrl06_infra import atomic_json, utc
from helpers.lewm16_planner import NoSafeAction
from helpers.lewm18_planner import CostModel, PlanningCEM
from helpers.safecem10_planner import box_penetration, expand_box, image_transform, pd_response

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/lewm18"


def local_xy(xy, case):
    return (np.asarray(xy)[..., :2] - np.asarray(case["origin"])) @ np.asarray(case["basis"]).T


def hits(xy, case, margin):
    p = local_xy(xy, case)
    inside = np.zeros(p.shape[:-1], dtype=bool)
    for box in case["boxes"]:
        x0, x1, y0, y1 = expand_box(box, margin)
        inside |= (p[..., 0] >= x0) & (p[..., 0] <= x1) & (p[..., 1] >= y0) & (p[..., 1] <= y1)
    return inside


class DenseCube(GoalCubeEnv):
    """Same task and dynamics; record the pinch point after every physics integration."""

    def integrate(self, action):
        self.set_control(np.asarray(action))
        self.pre_step()
        path = []
        if getattr(self, "_monitor_model", None) is not self._model:
            self._monitor_data = mujoco.MjData(self._model)
            self._monitor_model = self._model
        for _ in range(self._n_steps):
            mujoco.mj_step(self._model, self._data)
            if hasattr(self, "work_counters"):
                counter = (
                    "checker_physics_steps"
                    if getattr(self, "_previewing", False)
                    else "executed_physics_steps"
                )
                self.work_counters[counter] += 1
            # Observe the integrated position without changing the upstream caches.
            self._monitor_data.qpos[:] = self._data.qpos
            self._monitor_data.mocap_pos[:] = self._data.mocap_pos
            self._monitor_data.mocap_quat[:] = self._data.mocap_quat
            mujoco.mj_kinematics(self._model, self._monitor_data)
            path.append(self._monitor_data.site_xpos[self._pinch_site_id].copy())
        mujoco.mj_rnePostConstraint(self._model, self._data)
        self.post_step()
        return np.asarray(path)

    def step(self, action):
        if self._reset_next_step:
            return self.reset()
        if self._success_timing == "pre":
            success = self._success
            terminated = self.terminate_episode()
            reward = self.compute_reward()
        self.last_dense = self.integrate(action)
        if self._success_timing == "post":
            success = self._success
            terminated = self.terminate_episode()
            reward = self.compute_reward()
        truncated = self.truncate_episode()
        info = self.get_step_info()
        info["success"] = success
        return self.compute_observation(), reward, terminated, truncated, self._decorate(info)

    def preview(self, actions):
        """True simulator prefix, followed by exact restoration of integration state."""
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self._model, kind))
        mujoco.mj_getState(self._model, self._data, state, kind)
        snapshot = mujoco.MjData(self._model)
        mujoco.mj_copyData(snapshot, self._model, self._data)
        keys = [
            "_success",
            "_reset_next_step",
            "_prev_qpos",
            "_prev_qvel",
            "_prev_ob_info",
            "_target_effector_pose",
        ]
        saved = {k: copy.deepcopy(getattr(self, k)) for k in keys if hasattr(self, k)}
        geom_rgba = self._model.geom_rgba.copy()
        site_rgba = self._model.site_rgba.copy()
        paths = []
        previous_previewing = getattr(self, "_previewing", False)
        self._previewing = True
        try:
            for action in actions:
                paths.append(self.integrate(action))
                if self._success:
                    break
        finally:
            self._previewing = previous_previewing
            # Copy derived caches too: forwarding alone changes the upstream next action.
            mujoco.mj_copyData(self._data, self._model, snapshot)
            for k, v in saved.items():
                setattr(self, k, v)
            for k in keys:
                if k not in saved and hasattr(self, k):
                    delattr(self, k)
            self._model.geom_rgba[:] = geom_rgba
            self._model.site_rgba[:] = site_rgba
            restored = np.empty_like(state)
            mujoco.mj_getState(self._model, self._data, restored, kind)
            if not np.array_equal(state, restored):
                raise RuntimeError("Cube preview did not restore integration state")
        return np.concatenate(paths) if paths else np.empty((0, 3))


class Checker:
    def __init__(self, kind, case, margin, scaler, env, counters, enabled=True):
        self.kind, self.case, self.margin = kind, case, margin
        self.scaler, self.env, self.counters = scaler, env, counters
        self.enabled = enabled
        self.state = None
        self.previous_step = np.zeros(3)
        self.cache = {}

    def raw(self, candidates):
        dim = len(self.scaler.mean_)
        raw = candidates.double().reshape(*candidates.shape[:2], -1, dim)
        mean = torch.as_tensor(self.scaler.mean_, device=raw.device)
        scale = torch.as_tensor(self.scaler.scale_, device=raw.device)
        return raw * scale + mean

    def paths(self, candidates):
        raw = self.raw(candidates)
        b, n, length, _ = raw.shape
        self.counters["checker_candidate_rollouts"] += b * n
        if self.kind == "pusht":
            if length not in self.cache:
                c, g = pd_response(length)
                self.cache[length] = (
                    torch.as_tensor(c, device=raw.device),
                    torch.as_tensor(g, device=raw.device),
                )
            c, g = self.cache[length]
            start = torch.as_tensor(self.state[:2], device=raw.device).view(1, 1, 1, 2)
            vel = torch.as_tensor(self.state[-2:], device=raw.device).view(1, 1, 1, 2)
            path = start + c.view(1, 1, -1, 1) * vel + torch.einsum("mn,bsnk->bsmk", g, raw)
            self.counters["analytic_path_steps"] += b * n * length * 10
        else:
            path = commanded_path(
                self.state[:3], raw, gain=(0.40, 0.41, 0.335), prev_step=self.previous_step
            )
            start = torch.as_tensor(self.state[:3], device=raw.device).view(1, 1, 1, 3)
            path = densify(start, path, 4)[..., :2]
            self.counters["approximate_path_points"] += b * n * length * 4
        return path

    def check(self, candidates):
        if not self.enabled:
            b, n = candidates.shape[:2]
            zero = torch.zeros(b, n, device=candidates.device)
            return zero, torch.zeros(b, n, 1, 2, device=candidates.device), zero
        path = self.paths(candidates)
        origin = torch.as_tensor(self.case["origin"], device=path.device)
        basis = torch.as_tensor(self.case["basis"], device=path.device, dtype=path.dtype)
        local = (path - origin) @ basis.T
        violation = torch.zeros(path.shape[:2], device=path.device, dtype=path.dtype)
        guard = 0.001 if self.kind == "pusht" else 1e-6
        for box in self.case["boxes"]:
            violation += box_penetration(local, expand_box(box, self.margin + guard)).sum(-1)
        # Geometric goal costs are deliberately unavailable to the task planner.
        return violation, local, torch.zeros_like(violation)

    def execution_feasible(self, candidate):
        raw = self.raw(candidate[:, None])[0, 0, :5].detach().cpu().numpy()
        self.counters["prefix_checks"] += 1
        path = self.env.preview(raw)
        return not bool(hits(path, self.case, self.margin + 1e-6).any())


class Runtime:
    def __init__(self, kind, device="cuda"):
        import stable_worldmodel as swm

        self.swm, self.kind, self.device = swm, kind, device
        prep = json.loads((OUT / f"{kind}-prepared.json").read_text())
        self.prep = prep
        checkpoint = str(Path(prep["model"]))[: -len("_object.ckpt")]
        self.model = swm.policy.AutoCostModel(checkpoint).to(device).eval()
        self.model.requires_grad_(False)
        self.process, _ = load_scalers(prep["scalers"])
        self.transform = {"pixels": image_transform(), "goal": image_transform()}
        if kind == "cube":
            self.env = DenseCube(
                env_type="single",
                ob_type="states",
                pixel_transparent_arm=False,
                width=224,
                height=224,
            )
            self.rows = DatasetRows(prep["dataset"])
        else:
            from stable_worldmodel.envs.pusht.env import PushT
            import h5py
            import hdf5plugin  # noqa: F401

            class CountedPushT(PushT):
                def _set_state(env_self, state):
                    original_step = env_self.space.step

                    def counted_step(dt):
                        env_self.work_counters["reset_physics_steps"] += 1
                        return original_step(dt)

                    env_self.space.step = counted_step
                    try:
                        return super()._set_state(state)
                    finally:
                        env_self.space.step = original_step

            self.env = CountedPushT()
            self.rows = h5py.File(prep["dataset"], "r")

    def reset(self, job):
        case = job["case"]
        # The pinned Cube wrapper drops its seed argument before calling its parent.
        self.env.np_random = np.random.default_rng(int(job["seed"]))
        a, b = case["start_row"], case["goal_row"]
        if self.kind == "cube":
            pair = self.rows.pair(a, b)
            self.dataset_start_image = pair["start_image"]
            options = {k: pair[k] for k in ["state", "goal_image", "goal_cube_pos", "marker_pos"]}
            _, info = self.env.reset(seed=job["seed"], options=options)
            state = np.asarray(info["state"], dtype=float)
            goal = pair["goal_cube_pos"]
        else:
            self.dataset_start_image = np.asarray(self.rows["pixels"][a])
            self.dataset_initial_state = np.asarray(self.rows["state"][a], dtype=float)
            goal = np.asarray(self.rows["state"][b], dtype=float)
            original_render = self.env.render

            def reset_render():
                self.counters["reset_render_frames"] += 1
                return original_render()

            self.env.render = reset_render
            try:
                observation, info = self.env.reset(
                    seed=job["seed"],
                    options={"state": self.dataset_initial_state, "goal_state": goal},
                )
            finally:
                self.env.render = original_render
            state = np.asarray(observation["state"], dtype=float).copy()
            self.env._goal = np.asarray(self.rows["pixels"][b]).copy()
            info = self.env._get_info()
            original = self.env.space.step

            def dense_step(dt):
                original(dt)
                self.dense.append(np.asarray(self.env.agent.position, dtype=float).copy())
                self.counters["executed_physics_steps"] += 1

            self.env.space.step = dense_step
        return state, goal, info

    def run(self, job, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / "started.json", {"utc": utc(), "job": job})
        self.counters = collections.defaultdict(int)
        self.env.work_counters = self.counters
        self.dense = []
        states, actions, frames, simulator_states = [], [], [], []
        self.counters["environment_resets"] = 1
        seed = int(job["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        state, goal, info = self.reset(job)
        initial_state = state.copy()
        initially_solved = (
            bool(np.linalg.norm(state[3:6] - goal) <= 0.04)
            if self.kind == "cube"
            else bool(self.env.eval_state(goal, state)[0])
        )
        params = job["params"]
        checker = Checker(
            self.kind,
            job["case"],
            job["margin"],
            self.process["action"],
            self.env,
            self.counters,
            enabled=params["checked"],
        )
        checker.state = state
        cost = CostModel(self.model, checker, params, self.counters, diagnostic=False)
        solver = PlanningCEM(cost, self.kind, params, seed, self.device)
        block = int(job.get("executed_blocks", 1))
        policy = self.swm.policy.WorldModelPolicy(
            solver,
            self.swm.PlanConfig(horizon=5, receding_horizon=block, action_block=5),
            process=self.process,
            transform=self.transform,
        )
        dim = self.env.action_space.shape[0]
        policy.set_env(
            SimpleNamespace(
                num_envs=1,
                single_action_space=self.env.action_space,
                action_space=gym.spaces.Box(-1.0, 1.0, (1, dim), dtype=np.float64),
            )
        )
        previous_action = np.zeros(dim)
        success, failure, error = False, None, None
        native_parity_error = None
        native_parity_seconds = 0.0
        initial_image_mae = None
        planning_seconds = 0.0
        started = time.perf_counter()
        try:
            for _ in range(50):
                if (OUT / "STOP").exists():
                    raise InterruptedError("Campaign stop requested")
                pixels = np.ascontiguousarray(self.env.render())
                if not actions:
                    initial_image_mae = float(
                        np.abs(pixels.astype(float) - self.dataset_start_image.astype(float)).mean()
                    )
                self.counters["render_frames"] += 1
                if job.get("save_frames"):
                    frames.append(pixels)
                task_info = {
                    "pixels": pixels[None, None],
                    "goal": np.ascontiguousarray(info["goal"])[None, None],
                    "action": previous_action[None, None],
                }
                before = time.perf_counter()
                try:
                    action = policy.get_action(task_info)[0]
                    if job.get("native_parity") and not actions:
                        from stable_worldmodel.solver import CEMSolver

                        parity_started = time.perf_counter()
                        reference_cost = CostModel(self.model, checker, params, self.counters)
                        reference = self.swm.policy.WorldModelPolicy(
                            CEMSolver(
                                reference_cost,
                                num_samples=300,
                                n_steps=30,
                                topk=30,
                                seed=seed,
                                device=self.device,
                            ),
                            self.swm.PlanConfig(horizon=5, receding_horizon=block, action_block=5),
                            process=self.process,
                            transform=self.transform,
                        )
                        reference.set_env(policy.env)
                        expected = reference.get_action(task_info)[0]
                        self.counters["encoder_frames"] += 2
                        torch.cuda.synchronize()
                        native_parity_seconds = time.perf_counter() - parity_started
                        native_parity_error = float(np.max(np.abs(action - expected)))
                        if native_parity_error > 1e-5:
                            raise RuntimeError("Native CEM action differs from upstream solver")
                finally:
                    torch.cuda.synchronize()
                    planning_seconds += time.perf_counter() - before
                self.counters["environment_actions"] += 1
                observation, _, terminated, _, info = self.env.step(action)
                previous = state
                state = np.asarray(
                    info["state"] if self.kind == "cube" else observation["state"], dtype=float
                )
                actions.append(np.asarray(action).copy())
                states.append(state.copy())
                if self.kind == "cube":
                    dense = self.env.last_dense
                    self.dense.extend(dense)
                    simulator_states.append(
                        np.r_[self.env._data.qpos.copy(), self.env._data.qvel.copy()]
                    )
                    checker.previous_step = state[:3] - previous[:3]
                    success = bool(np.linalg.norm(state[3:6] - goal) <= 0.04)
                else:
                    success = bool(self.env.eval_state(goal, state)[0])
                checker.state = state
                previous_action = np.asarray(action)
                if success:
                    break
        except NoSafeAction as exc:
            failure = str(exc)
        except Exception:
            error = traceback.format_exc()
        elapsed = time.perf_counter() - started
        if job.get("save_frames"):
            frames.append(np.ascontiguousarray(self.env.render()))
            self.counters["render_frames"] += 1
        xy = np.asarray(self.dense)
        origin_hit = bool(hits(xy, job["case"], 0).any()) if len(xy) else False
        margin_hit = bool(hits(xy, job["case"], job["margin"]).any()) if len(xy) else False
        safe = success and not margin_hit and error is None and failure is None
        dimensions = 3 if self.kind == "cube" else 2
        path = (
            np.vstack([initial_state[:dimensions], np.asarray(states)[:, :dimensions]])
            if states
            else initial_state[None, :dimensions]
        )
        distance = float(np.linalg.norm(np.diff(path, axis=0), axis=-1).sum())
        n = len(actions)
        result = {
            "utc": utc(),
            "job": job,
            "success": success,
            "initially_solved": initially_solved,
            "native_parity_error": native_parity_error,
            "native_parity_seconds": native_parity_seconds,
            "initial_image_mae": initial_image_mae,
            "requested_safe_completion": safe,
            "benchmark_safe_completion": success and not origin_hit and error is None,
            "any_violation": origin_hit,
            "margin_breach": margin_hit,
            "unfinished": not success,
            "engaged": distance > (1.0 if self.kind == "pusht" else 0.001),
            "path_length": distance,
            "actions": n,
            "J_safe": n if safe else 50,
            "safe_success_actions": n if safe else None,
            "simulated_seconds": n
            * (0.1 if self.kind == "pusht" else float(self.env._control_timestep)),
            "planning_seconds": planning_seconds,
            "episode_seconds": elapsed,
            "resources": dict(self.counters),
            "controller_failure": failure,
            "error": error,
            "censored": error is not None,
            "initial_state": initial_state.tolist(),
            "dataset_initial_state": (
                self.dataset_initial_state.tolist() if self.kind == "pusht" else None
            ),
            "goal": np.asarray(goal).tolist(),
            "final_state": state.tolist(),
            "object_displacement": float(
                np.linalg.norm(
                    state[3:6] - initial_state[3:6]
                    if self.kind == "cube"
                    else state[2:4] - initial_state[2:4]
                )
            ),
            "final_object_goal_distance": float(
                np.linalg.norm(state[3:6] - goal if self.kind == "cube" else state[2:4] - goal[2:4])
            ),
            "replans": solver.replans,
        }
        np.savez_compressed(
            directory / "trace.npz",
            states=np.asarray(states),
            actions=np.asarray(actions),
            dense_xy=xy,
            simulator_states=np.asarray(simulator_states),
        )
        if frames:
            np.savez_compressed(
                directory / "frames.npz",
                frames=np.asarray(frames),
                goal_frame=np.asarray(info["goal"]),
            )
        atomic_json(directory / "result.json", result)
        return result
