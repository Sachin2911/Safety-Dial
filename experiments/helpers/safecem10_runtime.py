"""Remote-only runtime: load the frozen world model once, run fixed Push-T episodes.

Imports stable_worldmodel lazily so the unit tests and the local manager never need the
checkpoint. Everything that touches the simulator counts its interactions.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from helpers.safecem10_geometry import gate_boxes, layout, realized_metrics
from helpers.safecem10_planner import (
    CONTROLLER,
    DialCostModel,
    MLPProbe,
    image_transform,
    make_state,
    pd_response,
    summarize_history,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/safecem10"


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


PROBE_FILES = {"v1": "probe.pt", "v2": "probe-v2.pt"}


def load_probe(device, path=DATA / "probe.pt"):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    probe = MLPProbe(in_dim=blob["in_dim"], hidden=tuple(blob["hidden"]))
    probe.load_state_dict(blob["state_dict"])
    return probe.to(device).eval(), blob["stats"]


def file_sha256(path):
    import hashlib

    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def best_candidate_solver():
    """CEMSolver that executes the best checked candidate of the solve, not the elite mean.

    The upstream solver returns the mean of the final elite set, which the constraint never
    evaluated. iCEM (Pinneri et al. 2020) executes the best elite instead; here the cost
    model tracks the best candidate of all iterations under the planner's own ranking.
    """
    from stable_worldmodel.solver import CEMSolver

    class BestCandidateCEM(CEMSolver):
        def solve(self, info_dict, init_action=None):
            self.model.begin_solve()
            outputs = super().solve(info_dict, init_action)
            best = self.model.best_actions()
            if best is not None:
                outputs["elite_mean_actions"] = outputs["actions"]
                outputs["actions"] = best.to(outputs["actions"].dtype)
                self.model.record_execution()
            return outputs

    return BestCandidateCEM


def step_tracking_error(states, start_state, actions, controller=CONTROLLER):
    """Per-step gap between the saved pusher positions and the linear PD model's prediction
    from the previous saved state and the executed raw action (validates the exact check)."""
    c, G = pd_response(1, **controller)
    prev = np.vstack([np.asarray(start_state, dtype=float)[None], states[:-1]])
    pred = prev[:, :2] + c[-1] * prev[:, -2:] + G[-1, 0] * actions
    err = np.linalg.norm(pred - states[:, :2], axis=1)
    return {
        "median": float(np.median(err)),
        "p95": float(np.percentile(err, 95)),
        "max": float(err.max()),
    }


def load_world_model(device):
    os.environ.setdefault("STABLEWM_HOME", str(DATA / "stablewm"))
    import stable_worldmodel as swm

    model = swm.policy.AutoCostModel("pusht/lewm").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class Runtime:
    def __init__(self, device="cuda", spec=None):
        import stable_worldmodel as swm

        self.swm = swm
        self.device = device
        self.spec = spec
        self.model = load_world_model(device)
        self.probe, self.probe_stats = load_probe(device)
        self.probes = {"v1": (self.probe, self.probe_stats)}
        v2 = DATA / PROBE_FILES["v2"]
        expected = (spec or {}).get("probe_v2", {}).get("sha256")
        if v2.exists():
            if expected and file_sha256(v2) != expected:
                raise ValueError("probe-v2.pt differs from the frozen probe receipt")
            self.probes["v2"] = load_probe(device, v2)
        self.process, self.scaler_blob = load_scalers()
        transform = image_transform()
        self.transform = {"pixels": transform, "goal": transform}
        env = spec["environment"]
        self.start_state = make_state(env["start_pusher"], env["start_block"], env["block_theta"])
        self.goal_state = make_state(env["goal_pusher"], env["goal_block"], env["block_theta"])

    def run(self, job, keep_frames=False):
        """One fixed episode; returns metrics, saved states and planner diagnostics."""
        from stable_worldmodel.solver import CEMSolver

        spec = self.spec
        env = spec["environment"]
        params = job["planner"]
        seed = int(job["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        lay = layout(job["layout"], spec)
        boxes = gate_boxes(lay, spec)
        action = self.process["action"]
        check = params.get("check", "probe")
        execute = params.get("execute", "mean")
        version = params.get("probe_version", "v1")
        if version not in self.probes:
            raise RuntimeError(f"Probe {version} is not prepared on this machine")
        if execute not in ("mean", "best"):
            raise ValueError("execute must be mean or best")
        controller = env.get("controller", CONTROLLER)
        cost_model = DialCostModel(
            self.model,
            self.probes[version][0],
            boxes,
            job["clearances"],
            mode=job["mode"],
            lam=job["lam"],
            tol=params["feasible_tol"],
            n_substeps=params["n_substeps"],
            arena_constraint=params["arena_constraint"] == "action_space",
            action_mean=action.mean_,
            action_scale=action.scale_,
            arena=tuple(env["arena"]),
            check=check,
            track_best=execute == "best",
            controller=controller,
        ).eval()
        solver_class = best_candidate_solver() if execute == "best" else CEMSolver
        solver = solver_class(
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
            env["env_name"],
            num_envs=1,
            image_shape=(env["image_size"], env["image_size"]),
            max_episode_steps=int(env["max_episode_steps"]),
        )
        world.set_policy(policy)
        world.reset(seed=seed, options={"state": self.start_state, "goal_state": self.goal_state})
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
            states.append(np.asarray(w.infos["state"][0, 0], dtype=float).copy())
            r = getattr(w, "rewards", None)
            rewards.append(float(np.asarray(r).ravel()[0]) if r is not None else float("nan"))
            cost_model.pusher_xy = states[-1][:2]
            cost_model.pusher_vel = states[-1][-2:]
            if keep_frames:
                frame = np.asarray(w.infos["pixels"][0, -1]).copy()
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                frames.append(frame)

        cost_model.pusher_xy = np.asarray(self.start_state[:2], dtype=float)
        cost_model.pusher_vel = np.asarray(self.start_state[-2:], dtype=float)
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
                "final_reward": rewards[-1] if rewards else None,
                "max_reward": float(np.nanmax(rewards)) if rewards else None,
                "model_candidate_rollouts": cost_model.queries,
                "planner_seconds": elapsed,
                "planner": summarize_history(cost_model.history, params["n_steps"]),
                "check": check,
                "execute": execute,
                "probe_version": version,
                "controller_step_error_px": step_tracking_error(
                    states_arr, self.start_state, actions_arr, controller
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
