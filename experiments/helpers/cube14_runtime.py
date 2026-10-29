"""Remote-only runtime (task 14): load the frozen LeWM Cube model once, run fixed episodes.

Imports stable_worldmodel lazily so the unit tests and the local manager never need the
checkpoint. Every episode is a dataset pair, as in LeWM's own evaluation: the simulator
starts from the recorded state of one row and the planner aims at the frame recorded 25 rows
later. Everything that touches the simulator counts its interactions, and every episode
reports how well the commanded-path model predicted the effector it actually moved. A job
may carry its own pair (the diagnosis does) and may have no hazard.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from helpers.cube14_env import register
from helpers.cube14_geometry import hazard_boxes, layout, outcome_metrics, realized_metrics
from helpers.cube14_planner import (
    EXECUTE_MODES,
    CubeCostModel,
    summarize_execution,
    tracking_errors,
)
from helpers.safecem10_planner import image_transform, summarize_history

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/cube14"


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


def load_world_model(device, name="cube/lewm"):
    os.environ.setdefault("STABLEWM_HOME", str(DATA / "stablewm"))
    import stable_worldmodel as swm

    model = swm.policy.AutoCostModel(name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def dataset_file():
    files = sorted((DATA / "processed").rglob("*.h5"))
    if len(files) != 1:
        raise ValueError(f"Expected one dataset file, found {len(files)}")
    return files[0]


class DatasetRows:
    """Read-only access to the pinned Cube dataset, one pair of rows at a time."""

    def __init__(self, path=None):
        import h5py
        import hdf5plugin  # noqa: F401  (the frames are Blosc-compressed)

        self.f = h5py.File(path or dataset_file(), "r")

    def pair(self, start_row, goal_row):
        f = self.f
        start_row, goal_row = int(start_row), int(goal_row)
        if int(f["ep_idx"][start_row]) != int(f["ep_idx"][goal_row]) or goal_row <= start_row:
            raise ValueError("A pair must be two increasing rows of one episode")
        effector = np.asarray(f["proprio_effector_pos"][start_row], dtype=np.float64)
        first = int(f["step_idx"][start_row]) == 0
        before = effector if first else np.asarray(f["proprio_effector_pos"][start_row - 1])
        return {
            "start_row": start_row,
            "goal_row": goal_row,
            "state": np.concatenate([f["qpos"][start_row], f["qvel"][start_row]]).astype(float),
            "goal_image": np.asarray(f["pixels"][goal_row]),
            "start_image": np.asarray(f["pixels"][start_row]),
            "goal_cube_pos": np.asarray(f["privileged_block_0_pos"][goal_row], dtype=float),
            "goal_effector_pos": np.asarray(f["proprio_effector_pos"][goal_row], dtype=float),
            "marker_pos": np.asarray(f["privileged_target_block_pos"][start_row], dtype=float),
            "start_effector": effector,
            "prev_step": effector - np.asarray(before, dtype=np.float64),
            "start_cube": np.asarray(f["privileged_block_0_pos"][start_row], dtype=float),
            "actions": np.asarray(f["action"][start_row:goal_row], dtype=np.float64),
            "effector_path": np.asarray(
                f["proprio_effector_pos"][start_row + 1 : goal_row + 1], dtype=np.float64
            ),
            "cube_path": np.asarray(
                f["privileged_block_0_pos"][start_row + 1 : goal_row + 1], dtype=np.float64
            ),
        }


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

    def __init__(self, device="cuda", spec=None, rows=None):
        import stable_worldmodel as swm

        self.swm = swm
        self.device = device
        self.spec = spec
        self.env_id = register()
        self.model = load_world_model(device, spec["pins"]["checkpoint_name"])
        self.process, self.scaler_blob = load_scalers()
        transform = image_transform()
        self.transform = {"pixels": transform, "goal": transform}
        self.rows = rows or DatasetRows()

    def episode(self, job, lay=None):
        """The dataset pair of a job: its own, else the scene of its layout."""
        pair = job.get("pair") or (lay or {}).get("pair")
        if pair is None:
            raise ValueError("A job needs a dataset pair")
        return self.rows.pair(pair["start_row"], pair["goal_row"])

    def run(self, job, keep_frames=False):
        """One fixed episode; returns metrics, saved states and planner diagnostics."""
        spec = self.spec
        env = spec["environment"]
        params = job["planner"]
        model_cfg = spec.get("check_model", {})
        seed = int(job["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        lay = layout(job["layout"], spec) if job.get("layout") else None
        boxes = hazard_boxes(lay, spec) if lay else []
        pair = self.episode(job, lay)
        action = self.process["action"]
        execute = params.get("execute", "mean")
        if execute not in EXECUTE_MODES:
            raise ValueError(f"execute must be one of {EXECUTE_MODES}")
        gain = model_cfg.get("gain", [1.0, 1.0, 1.0])
        carry = float(model_cfg.get("carry", 0.0))
        cost_model = CubeCostModel(
            self.model,
            boxes,
            job["clearances"],
            action_mean=action.mean_,
            action_scale=action.scale_,
            mode=job["mode"],
            lam=job["lam"],
            tol=params["feasible_tol"],
            track_best=execute != "mean",
            gain=gain,
            carry=carry,
            interp=int(model_cfg.get("interp", 4)),
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
        options = {
            "state": pair["state"],
            "goal_image": pair["goal_image"],
            "goal_cube_pos": pair["goal_cube_pos"],
            "marker_pos": pair["marker_pos"],
        }
        rule = job.get("success_rule", env.get("success", {}).get("rule", "cube"))
        if rule == "effector":
            options["goal_effector_pos"] = pair["goal_effector_pos"]
            options["effector_radius_m"] = float(env["success"]["effector_radius_m"])
        elif rule != "cube":
            raise ValueError("success rule must be cube or effector")
        world.reset(seed=seed, options=options)
        start = np.asarray(world.infos["state"][0, 0], dtype=float).reshape(-1)[:7].copy()
        states, rewards, frames, executed = [], [], [], []
        plan_action = policy.get_action

        def recording_action(info_dict, **kwargs):
            chosen = plan_action(info_dict, **kwargs)
            executed.append(np.asarray(chosen, dtype=float).reshape(-1)[:5].copy())
            return chosen

        policy.get_action = recording_action
        counter = {"env_steps": 0}

        def on_step(w):
            counter["env_steps"] += 1
            states.append(np.asarray(w.infos["state"][0, 0], dtype=float).reshape(-1)[:7].copy())
            r = getattr(w, "rewards", None)
            rewards.append(float(np.asarray(r).ravel()[0]) if r is not None else float("nan"))
            last = states[-2] if len(states) > 1 else start
            cost_model.prev_step = states[-1][:3] - last[:3]
            cost_model.effector = states[-1][:3]
            if keep_frames:
                frame = np.asarray(w.infos["pixels"][0, -1]).copy()
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                frames.append(frame)

        cost_model.effector = start[:3].copy()
        cost_model.prev_step = pair["prev_step"].copy()
        started = time.perf_counter()
        torch.cuda.synchronize() if self.device.startswith("cuda") else None
        world._run(max_steps=budget, mode="wait", on_step=on_step)
        torch.cuda.synchronize() if self.device.startswith("cuda") else None
        elapsed = time.perf_counter() - started
        terminated = bool(world.terminateds[0])
        states_arr = np.stack(states)
        actions_arr = np.stack(executed)[: len(states_arr)]
        goals = (pair["goal_cube_pos"], pair["goal_effector_pos"])
        if lay:
            metrics = realized_metrics(states_arr, lay, start, *goals, spec, terminated=terminated)
        else:
            metrics = outcome_metrics(states_arr, start, *goals, budget, terminated)
        metrics["success_rule"] = rule
        metrics.update(
            {
                "env_steps": counter["env_steps"],
                "model_candidate_rollouts": cost_model.queries,
                "planner_seconds": elapsed,
                "planner": summarize_history(cost_model.history, params["n_steps"]),
                "execution": summarize_execution(cost_model.history, params["n_steps"]),
                "check": "commanded",
                "execute": execute,
                "tracking": tracking_errors(
                    states_arr, actions_arr, start, gain=gain, carry=carry, prev0=pair["prev_step"]
                ),
                "start_row": pair["start_row"],
                "goal_row": pair["goal_row"],
            }
        )
        if counter["env_steps"] != len(states_arr) or counter["env_steps"] > budget:
            raise RuntimeError("Simulator interaction count disagrees with saved states")
        return {
            "metrics": metrics,
            "start": start,
            "states": states_arr,
            "rewards": np.asarray(rewards, dtype=float),
            "actions": actions_arr,
            "history": cost_model.history,
            "frames": np.stack(frames) if frames else None,
            "terminated": terminated,
        }
