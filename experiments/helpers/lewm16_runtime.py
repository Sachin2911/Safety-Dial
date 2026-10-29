"""One-worker runtime with actual-step receipts, dense monitors, and frozen LeWM."""

from __future__ import annotations

import collections
import json
import os
import time
import traceback
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from helpers.lewm16_planner import CampaignCEM, CostModel, ExactCheck, NoSafeAction
from helpers.lewm16_protocol import CONFIG, ROOT, boxes, hits, load_spec, write_json
from helpers.safecem10_planner import image_transform


class Scaler:
    def __init__(self, mean, scale):
        self.mean_, self.scale_ = np.asarray(mean), np.asarray(scale)

    def transform(self, a):
        return (a - self.mean_) / self.scale_

    def inverse_transform(self, a):
        return a * self.scale_ + self.mean_


class Runtime:
    def __init__(self, kind, device="cuda", threads=2):
        import stable_worldmodel as swm

        self.swm, self.kind, self.device = swm, kind, device
        self.spec = load_spec(kind)
        name = ROOT / f"data/lewm16/stablewm/checkpoints/{kind}/lewm"
        self.model = swm.policy.AutoCostModel(str(name)).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        scalers = json.loads((CONFIG / f"{kind}-scalers.json").read_text())
        self.process = {
            key: Scaler(value["mean"], value["scale"]) for key, value in scalers["columns"].items()
        }
        self.transform = {"pixels": image_transform(), "goal": image_transform()}
        self.arm = None
        self._hooked_env = None
        if kind == "tworoom":
            from helpers.tworoom12_env import GoalTwoRoomEnv

            self.env = GoalTwoRoomEnv()
        else:
            from helpers.reacher13_env import GoalReacherEnv
            from helpers.reacher13_planner import ArmModel

            self.env = GoalReacherEnv(task="qpos_match")
            self.arm = ArmModel(threads=threads)

    def install_physics_recorder(self):
        # Reset may rebuild DMC's underlying environment, invalidating an earlier hook.
        if self.kind != "reacher" or self._hooked_env is self.env.env:
            return
        self._hooked_env = self.env.env
        original_step = self.env.env.step

        def recorded_physics_step(action):
            step = original_step(action)
            self.counters["executed_physics_steps"] += 1
            physics = self.env.env.physics
            self.fine_states.append(
                np.r_[
                    physics.data.qpos.copy(),
                    physics.data.qvel.copy(),
                    physics.named.data.geom_xpos["finger"][:2].copy(),
                ]
            )
            return step

        self.env.env.step = recorded_physics_step

    def frame_info(self, info, action):
        self.counters["render_frames"] += 1
        # Explicit task-only interface. State and true target never reach LeWM/the adapter.
        return {
            "pixels": np.ascontiguousarray(self.env.render())[None, None],
            "goal": np.ascontiguousarray(info["goal"])[None, None],
            "action": np.asarray(action)[None, None],
        }

    def run(self, job, attempt):
        seed, case = job["seed"], job["case"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.counters = collections.defaultdict(int)
        self.fine_states = []
        self.counters["environment_resets"] = 1
        if self.kind == "tworoom":
            options = {"state": np.asarray(case["start"]), "target_state": np.asarray(case["goal"])}
        else:
            options = {"state": np.asarray(case["start"]), "target_qpos": np.asarray(case["goal"])}
        _, info = self.env.reset(seed=seed, options=options)
        self.install_physics_recorder()
        self.counters["goal_render_frames"] += 1
        check = ExactCheck(
            self.kind,
            self.spec,
            self.process["action"],
            boxes(self.kind, case),
            job["margin"],
            self.counters,
            arm=self.arm,
        )
        check.state, check.goal = np.asarray(case["start"]), np.asarray(case["goal"])
        diagnostic = job.get("diagnostic", job["stage"] in ("E", "Q") and seed % 3 == 0)
        model = CostModel(self.model, check, job["params"], self.counters, diagnostic)
        solver = CampaignCEM(model, self.kind, job["params"], seed, self.device)
        policy = self.swm.policy.WorldModelPolicy(
            solver,
            self.swm.PlanConfig(horizon=5, receding_horizon=1, action_block=5),
            process=self.process,
            transform=self.transform,
        )
        proxy = SimpleNamespace(
            num_envs=1,
            single_action_space=self.env.action_space,
            action_space=gym.spaces.Box(-1.0, 1.0, (1, 2), dtype=np.float64),
        )
        policy.set_env(proxy)
        states, actions, replay_errors, frames = [], [], [], []
        previous_action = np.zeros(2)
        terminated, controller_failure, error = False, None, None
        native_parity = None
        started = time.perf_counter()
        try:
            for _ in range(int(job.get("max_steps", 50))):
                if (ROOT / "outputs/lewm16/STOP").exists():
                    raise InterruptedError("Campaign stop requested")
                info_dict = self.frame_info(info, previous_action)
                if job.get("save_frames", False):
                    frames.append(info_dict["pixels"][0, 0].copy())
                chosen = policy.get_action(info_dict)[0]
                if job.get("native_parity") and not actions:
                    from stable_worldmodel.solver import CEMSolver

                    from helpers.lewm16_protocol import CANONICAL

                    settings = CANONICAL[self.kind]
                    reference = CEMSolver(
                        model=self.model,
                        num_samples=settings["samples"],
                        n_steps=10,
                        topk=round(settings["samples"] * settings["elite"]),
                        seed=seed,
                        device=self.device,
                    )
                    reference.configure(
                        action_space=solver._action_space, n_envs=1, config=solver._config
                    )
                    # These extra model queries are engineering costs, with no extra actions.
                    self.counters["model_candidate_rollouts"] += settings["samples"] * 10
                    self.counters["predictor_steps"] += settings["samples"] * 50
                    self.counters["encoder_frames"] += 2
                    expected = reference.solve(policy._prepare_info(info_dict))["actions"]
                    native_parity = float(
                        np.abs(expected[0, 0].numpy() - solver.replans[0]["chosen_block"]).max()
                    )
                    write_json(attempt / "native-parity.json", {"max_abs": native_parity})
                    if native_parity > 1e-5:
                        raise ValueError(f"Live native action parity failed: {native_parity}")
                if not np.isfinite(chosen).all():
                    raise ValueError("Nonfinite proposed executable action")
                old_state = check.state.copy()
                fine_before = len(self.fine_states)
                _, _, terminated, truncated, info = self.env.step(chosen)
                self.counters["environment_actions"] += 1
                if self.kind == "tworoom":
                    self.counters["executed_physics_steps"] += 1
                state = np.asarray(info["state"], dtype=float).copy()
                states.append(state)
                actions.append(chosen.copy())
                if self.kind == "tworoom":
                    self.fine_states.append(state)
                # Independent execution replay. This additional model work is counted.
                normalized = self.process["action"].transform(chosen)
                check.state = old_state
                path, _ = check.paths(torch.as_tensor(normalized).view(1, 1, 1, 2))
                observed_fine = np.asarray(self.fine_states[fine_before:])
                observed_xy = (
                    observed_fine[:, :2] if self.kind == "tworoom" else observed_fine[:, 4:6]
                )
                predicted_xy = path[0, 0, : len(observed_xy)].numpy()
                replay_errors.append(
                    float(np.linalg.norm(observed_xy - predicted_xy, axis=-1).max())
                )
                check.state = state[:2] if self.kind == "tworoom" else state[:4]
                previous_action = chosen
                write_json(
                    attempt / "progress.json",
                    {
                        "steps": len(states),
                        "counters": dict(self.counters),
                        "updated_epoch": time.time(),
                    },
                )
                if terminated or truncated:
                    break
        except NoSafeAction as exc:
            controller_failure = str(exc)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            (attempt / "traceback.txt").write_text(traceback.format_exc())
        elapsed = time.perf_counter() - started
        dim = 2 if self.kind == "tworoom" else 6
        a = np.asarray(actions).reshape(-1, 2)
        s = np.asarray(states).reshape(-1, dim)
        fine = np.asarray(self.fine_states).reshape(-1, dim)
        xy = s[:, :2] if self.kind == "tworoom" else s[:, 4:6]
        fine_xy = fine[:, :2] if self.kind == "tworoom" else fine[:, 4:6]
        hazard = boxes(self.kind, case)
        legacy_hit = hits(xy, hazard)
        dense_hit = hits(fine_xy, hazard)
        margin_hit = hits(fine_xy, hazard, job["margin"])
        if self.kind == "tworoom":
            start_xy = np.asarray(case["start"])
            goal_distance = (
                float(np.linalg.norm(s[-1] - case["goal"]))
                if len(s)
                else float(np.linalg.norm(start_xy - case["goal"]))
            )
        else:
            from helpers.reacher13_geometry import fingertip

            start_xy = fingertip(case["start"][:2], self.arm.links)[0]
            last_q = s[-1, :2] if len(s) else np.asarray(case["start"][:2])
            goal_distance = float(np.abs(last_q - case["goal"]).max())
        np.savez_compressed(
            attempt / "states.npz", states=s, actions=a, fine_states=fine, frames=np.asarray(frames)
        )
        write_json(attempt / "history.json", model.history)
        write_json(attempt / "replans.json", solver.replans)
        if model.snapshots:
            write_json(attempt / "candidate-snapshots.json", model.snapshots)
        metrics = {
            "status": "error" if error else "complete",
            "error": error,
            "success": bool(terminated),
            "benchmark_safe_completion": bool(terminated and not dense_hit.any()),
            "requested_safe_completion": bool(terminated and not margin_hit.any()),
            "any_violation": bool(dense_hit.any()),
            "legacy_violation": bool(legacy_hit.any()),
            "margin_breach": bool(margin_hit.any()),
            "hazard_samples": int(dense_hit.sum()),
            "margin_breach_samples": int(margin_hit.sum()),
            "unfinished": not bool(terminated),
            "exposure": float(dense_hit.mean()) if len(dense_hit) else 0.0,
            "controller_failure": controller_failure,
            "censored": controller_failure is not None,
            "env_steps": len(s),
            "J": len(s) if terminated else 50,
            "path_length": float(
                np.linalg.norm(np.diff(np.vstack([start_xy, xy]), axis=0), axis=1).sum()
            ),
            "action_norm_mean": float(np.linalg.norm(a, axis=1).mean()) if len(a) else 0.0,
            "final_goal_distance": goal_distance,
            "replay_error_max": max(replay_errors, default=0.0),
            "native_parity_max": native_parity,
            "seconds": elapsed,
            "counters": dict(self.counters),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "job": job,
        }
        write_json(attempt / "result.json", metrics)
        return metrics
