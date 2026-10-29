"""Remote-only runtime (task 13): load the frozen LeWM Reacher model once, run fixed episodes.

Imports stable_worldmodel lazily so the unit tests and the local manager never need the
checkpoint. Everything that touches the simulator counts its interactions, and every
episode replays its executed actions through the exact arm model to validate it. A job may
carry its own start state and target pose (the diagnosis does) and may have no hazard.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from helpers.reacher13_env import register
from helpers.reacher13_geometry import fingertip, hazard_boxes, layout, realized_metrics
from helpers.reacher13_planner import (
    EXECUTE_MODES,
    ArmModel,
    ReacherCostModel,
    replay_error,
    summarize_execution,
)
from helpers.safecem10_planner import image_transform, summarize_history

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/reacher13"


class Scaler:
    """A StandardScaler stand-in with the transform/inverse_transform the policy calls."""

    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)

    def transform(self, x):
        return (np.asarray(x, dtype=np.float64) - self.mean_) / self.scale_

    def inverse_transform(self, x):
        return np.asarray(x, dtype=np.float64) * self.scale_ + self.mean_


def load_scalers(path=DATA / "scalers.json"):
    blob = json.loads(Path(path).read_text())
    process = {}
    for key, item in blob["columns"].items():
        process[key] = Scaler(item["mean"], item["scale"])
        if key != "action":
            process[f"goal_{key}"] = process[key]
    return process, blob


def load_world_model(device, name="reacher/lewm"):
    os.environ.setdefault("STABLEWM_HOME", str(DATA / "stablewm"))
    import stable_worldmodel as swm

    model = swm.policy.AutoCostModel(name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def solver_class(execute):
    """CEMSolver for the elite mean; a wrapper that may swap in the best checked candidate."""
    from stable_worldmodel.solver import CEMSolver

    if execute == "mean":
        return CEMSolver

    class CheckedCEM(CEMSolver):
        def solve(self, info_dict, init_action=None):
            self.model.begin_solve()
            outputs = super().solve(info_dict, init_action)
            best = self.model.best_actions()
            if execute == "mean_safe":
                mean = outputs["actions"]
                with torch.no_grad():
                    viol = self.model.exact_violation(mean.detach().cpu().double().unsqueeze(1))
                if bool((viol[:, 0] <= self.model.tol).all()) or best is None:
                    self.model.record_choice("mean", float(viol[0, 0]))
                    return outputs
            if best is not None:
                outputs["elite_mean_actions"] = outputs["actions"]
                outputs["actions"] = best.to(outputs["actions"].dtype)
                self.model.record_choice("best")
            return outputs

    return CheckedCEM


class Runtime:
    probe_stats = None

    def __init__(self, device="cuda", spec=None, threads=4):
        import stable_worldmodel as swm

        self.swm = swm
        self.device = device
        self.spec = spec
        self.env_id = register()
        self.model = load_world_model(device, spec["pins"]["checkpoint_name"])
        self.process, self.scaler_blob = load_scalers()
        transform = image_transform()
        self.transform = {"pixels": transform, "goal": transform}
        self.arm = ArmModel(threads=threads)

    def episode(self, job):
        """Start state (q1, q2, v1, v2) and target pose of a job, from the job or the design."""
        env = self.spec["environment"]
        start = job.get("start_state")
        if start is None:
            start = [*env["start_qpos"], 0.0, 0.0]
        target = job.get("target_qpos", env.get("target_qpos"))
        return np.asarray(start, dtype=np.float64), np.asarray(target, dtype=np.float64)

    def run(self, job, keep_frames=False):
        """One fixed episode; returns metrics, saved states and planner diagnostics."""
        spec = self.spec
        env = spec["environment"]
        params = job["planner"]
        seed = int(job["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        lay = layout(job["layout"], spec) if job.get("layout") else None
        boxes = hazard_boxes(lay, spec) if lay else []
        start, target = self.episode(job)
        action = self.process["action"]
        execute = params.get("execute", "mean")
        if execute not in EXECUTE_MODES:
            raise ValueError(f"execute must be one of {EXECUTE_MODES}")
        cost_model = ReacherCostModel(
            self.model,
            boxes,
            job["clearances"],
            arm=self.arm,
            action_mean=action.mean_,
            action_scale=action.scale_,
            mode=job["mode"],
            lam=job["lam"],
            tol=params["feasible_tol"],
            track_best=execute != "mean",
        ).eval()
        solver = solver_class(execute)(
            model=cost_model,
            batch_size=1,
            num_samples=params["num_samples"],
            var_scale=params["var_scale"],
            n_steps=params["n_steps"],
            topk=params["topk"],
            device=self.device,
            seed=seed,
        )
        policy = self.swm.policy.WorldModelPolicy(
            solver=solver,
            config=self.swm.PlanConfig(
                horizon=params["horizon"],
                receding_horizon=params["receding_horizon"],
                action_block=params["action_block"],
            ),
            process=self.process,
            transform=self.transform,
        )
        budget = int(job.get("budget", env["eval_budget_steps"]))
        world = self.swm.World(
            self.env_id,
            num_envs=1,
            image_shape=(env["image_size"], env["image_size"]),
            max_episode_steps=int(env["max_episode_steps"]),
        )
        world.set_policy(policy)
        world.reset(seed=seed, options={"state": start, "target_qpos": target})
        states, rewards, frames, executed = [], [], [], []
        plan_action = policy.get_action

        def recording_action(info_dict, **kwargs):
            chosen = plan_action(info_dict, **kwargs)
            executed.append(np.asarray(chosen, dtype=float).reshape(-1)[:2].copy())
            return chosen

        policy.get_action = recording_action
        counter = {"env_steps": 0}

        def on_step(w):
            counter["env_steps"] += 1
            states.append(np.asarray(w.infos["state"][0, 0], dtype=float).reshape(-1)[:6].copy())
            r = getattr(w, "rewards", None)
            rewards.append(float(np.asarray(r).ravel()[0]) if r is not None else float("nan"))
            cost_model.arm_state = states[-1][:4]
            cost_model.pusher_xy = states[-1][4:6]
            if keep_frames:
                frame = np.asarray(w.infos["pixels"][0, -1]).copy()
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                frames.append(frame)

        cost_model.arm_state = start[:4].copy()
        cost_model.pusher_xy = fingertip(start[:2], self.arm.links)[0]
        started = time.perf_counter()
        torch.cuda.synchronize() if self.device.startswith("cuda") else None
        world._run(max_steps=budget, mode="wait", on_step=on_step)
        torch.cuda.synchronize() if self.device.startswith("cuda") else None
        elapsed = time.perf_counter() - started
        terminated = bool(world.terminateds[0])
        states_arr = np.stack(states)
        actions_arr = np.stack(executed)[: len(states_arr)]
        replay = replay_error(states_arr, start, actions_arr, self.arm, terminated=terminated)
        if lay:
            metrics = realized_metrics(states_arr, lay, spec, terminated=terminated)
        else:
            metrics = free_metrics(states_arr, start, target, self.arm.links, budget, terminated)
        metrics.update(
            {
                "env_steps": counter["env_steps"],
                "model_candidate_rollouts": cost_model.queries,
                "planner_seconds": elapsed,
                "planner": summarize_history(cost_model.history, params["n_steps"]),
                "execution": summarize_execution(cost_model.history, params["n_steps"]),
                "check": "exact",
                "execute": execute,
                "exact_replay_error_m": replay,
            }
        )
        if counter["env_steps"] != len(states_arr) or counter["env_steps"] > budget:
            raise RuntimeError("Simulator interaction count disagrees with saved states")
        return {
            "metrics": metrics,
            "states": states_arr,
            "rewards": np.asarray(rewards, dtype=float),
            "actions": actions_arr,
            "history": cost_model.history,
            "frames": np.stack(frames) if frames else None,
            "terminated": terminated,
        }


def free_metrics(states, start, target, links, budget, terminated):
    """Outcome of an episode without a hazard (the diagnosis)."""
    qpos, xy = states[:, 0:2], states[:, 4:6]
    target_xy = fingertip(target, links)[0]
    start_xy = fingertip(start[:2], links)[0]
    joint_error = np.abs(qpos - target).max(axis=1)
    path = np.linalg.norm(np.diff(np.vstack([start_xy, xy]), axis=0), axis=1)
    n = len(states)
    return {
        "n_states": int(n),
        "success": bool(terminated),
        "steps_to_success": int(n) if terminated else None,
        "J": float(n if terminated else budget),
        "start_target_distance_m": float(np.linalg.norm(target_xy - start_xy)),
        "start_joint_error_rad": float(np.abs(start[:2] - target).max()),
        "final_joint_error_rad": float(joint_error[-1]),
        "min_joint_error_rad": float(joint_error.min()),
        "final_target_distance_m": float(np.linalg.norm(xy[-1] - target_xy)),
        "min_target_distance_m": float(np.linalg.norm(xy - target_xy, axis=1).min()),
        "path_length_m": float(path.sum()),
        "max_step_m": float(path.max()),
        "median_step_m": float(np.median(path)),
    }
