"""Remote-only runtime (task 12): load the frozen LeWM TwoRoom model once, run fixed episodes.

Imports stable_worldmodel lazily so the unit tests and the local manager never need the
checkpoint. Everything that touches the simulator counts its interactions, and every
episode replays its executed actions through the exact check to validate it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from helpers.safecem10_planner import image_transform, summarize_history
from helpers.tworoom12_env import register
from helpers.tworoom12_geometry import hazard_boxes, layout, realized_metrics
from helpers.tworoom12_planner import (
    EXECUTE_MODES,
    TwoRoomCostModel,
    replay_error,
    summarize_execution,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/tworoom12"


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


def load_world_model(device, name="tworoom/lewm"):
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

    def __init__(self, device="cuda", spec=None):
        import stable_worldmodel as swm

        self.swm = swm
        self.device = device
        self.spec = spec
        self.env_id = register()
        self.model = load_world_model(device, spec["pins"]["checkpoint_name"])
        self.process, self.scaler_blob = load_scalers()
        transform = image_transform()
        self.transform = {"pixels": transform, "goal": transform}
        env = spec["environment"]
        self.start = np.asarray(env["start_agent"], dtype=np.float64)
        self.target = np.asarray(env["target"], dtype=np.float64)

    def run(self, job, keep_frames=False):
        """One fixed episode; returns metrics, saved states and planner diagnostics."""
        spec = self.spec
        env = spec["environment"]
        params = job["planner"]
        seed = int(job["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        lay = layout(job["layout"], spec)
        boxes = hazard_boxes(lay, spec)
        action = self.process["action"]
        execute = params.get("execute", "mean")
        if execute not in EXECUTE_MODES:
            raise ValueError(f"execute must be one of {EXECUTE_MODES}")
        cost_model = TwoRoomCostModel(
            self.model,
            boxes,
            job["clearances"],
            geometry=env["geometry"],
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
        budget = int(env["eval_budget_steps"])
        world = self.swm.World(
            self.env_id,
            num_envs=1,
            image_shape=(env["image_size"], env["image_size"]),
            max_episode_steps=int(env["max_episode_steps"]),
        )
        world.set_policy(policy)
        world.reset(seed=seed, options={"state": self.start, "target_state": self.target})
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
            states.append(np.asarray(w.infos["state"][0, 0], dtype=float).reshape(-1)[:2].copy())
            r = getattr(w, "rewards", None)
            rewards.append(float(np.asarray(r).ravel()[0]) if r is not None else float("nan"))
            cost_model.pusher_xy = states[-1]
            if keep_frames:
                frame = np.asarray(w.infos["pixels"][0, -1]).copy()
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                frames.append(frame)

        cost_model.pusher_xy = self.start.copy()
        started = time.perf_counter()
        torch.cuda.synchronize() if self.device.startswith("cuda") else None
        world._run(max_steps=budget, mode="wait", on_step=on_step)
        torch.cuda.synchronize() if self.device.startswith("cuda") else None
        elapsed = time.perf_counter() - started
        terminated = bool(world.terminateds[0])
        states_arr = np.stack(states)
        actions_arr = np.stack(executed)[: len(states_arr)]
        metrics = realized_metrics(states_arr, lay, spec, terminated=terminated)
        metrics.update(
            {
                "env_steps": counter["env_steps"],
                "model_candidate_rollouts": cost_model.queries,
                "planner_seconds": elapsed,
                "planner": summarize_history(cost_model.history, params["n_steps"]),
                "execution": summarize_execution(cost_model.history, params["n_steps"]),
                "check": "exact",
                "execute": execute,
                "exact_replay_error_px": replay_error(
                    states_arr, self.start, actions_arr, env["geometry"]
                ),
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
