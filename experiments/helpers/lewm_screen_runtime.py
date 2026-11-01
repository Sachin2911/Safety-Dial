"""Counted runtimes for the declared PushT/Cube fixed-controller screen."""

import collections
import json
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from helpers.lewm16_planner import NoSafeAction
from helpers.lewm18_planner import CostModel, PlanningCEM
from helpers.lewm_proposal_adapter import ProposalSpec
from helpers.lewm_proposal_pilot import write_json
from helpers.lewm_screen import (
    CubeProposalAdapter,
    CubeProposalSpec,
    ScreenPushTAdapter,
    distance,
    image_stats,
)
from helpers.learned_motion import LearnedMotionChecker, load_motion
from helpers.portable_env import NativeEnv
from helpers.portable_motion import MotionChecker, load_model, shape_constraints
from helpers.pusht_smoke import Scaler
from helpers.safety_model import pusht_constraints


class ScreenCubeEnv(NativeEnv):
    """Retain one native renderer per stage and count constructor/reset physics too."""

    def __init__(self, counters):
        from helpers.lewm18_runtime import DenseCube

        self.name, self.counters, self.phase = "cube", counters, "reset"
        self.journal, self.dense = None, []
        owner = self
        import mujoco

        original_step = mujoco.mj_step

        def counted_step(*args, **kwargs):
            count = int(kwargs.get("nstep", args[2] if len(args) > 2 else 1))
            counters["native_" + owner.phase + "_physics_steps"] += count
            return original_step(*args, **kwargs)

        self._step_patch = patch("mujoco.mj_step", side_effect=counted_step)
        self._step_patch.start()

        class CountedCube(DenseCube):
            work_counters = counters

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

        try:
            self.env = CountedCube(ob_type="states", width=224, height=224)
        except BaseException:
            self._step_patch.stop()
            raise
        self.env.work_counters = counters
        if abs(self.env.control_timestep() - 0.05) > 1e-12 or self.env._n_steps != 25:
            raise RuntimeError("Cube control/physics rate differs from the frozen protocol")

    def close(self):
        try:
            return super().close()
        finally:
            self._step_patch.stop()


class BodyMotionChecker(MotionChecker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_point = None

    def observe(self, observed):
        point = np.asarray(observed)[-2:].copy()
        velocity = (
            np.zeros(2) if self.previous_point is None else (point - self.previous_point) / 0.05
        )
        self.body_proprio = torch.as_tensor(np.r_[point, velocity][None], dtype=torch.float64)
        self.previous_point = point
        return super().observe(observed)


def load_runtime(root, design, environment, constructor_receipt=None):
    from stable_worldmodel.policy import AutoCostModel

    cfg = design["environments"][environment]
    checkpoint = root / cfg["world_model"]
    wm = AutoCostModel(str(checkpoint)[: -len("_object.ckpt")], cache_dir=checkpoint.parent)
    wm = wm.cuda().eval().requires_grad_(False)
    columns = json.loads((root / cfg["scalers"]).read_text())["columns"]
    process = {k: Scaler(v["mean"], v["scale"]) for k, v in columns.items()}
    if environment == "pusht":
        head, _ = load_motion(root / cfg["motion_model"], device="cuda")
        artifact = json.loads((root / cfg["controller"]).read_text())
        weights = np.asarray(artifact["parameters"])
        rt = SimpleNamespace(
            environment=environment,
            wm=wm,
            head=head,
            process=process,
            weights=weights,
            spec=ProposalSpec(**artifact["spec"]),
            env=None,
        )
    else:
        head = load_model(root / cfg["motion_model"])
        spec = CubeProposalSpec(projection_seed=design["cube_adapter_projection_seed"])
        rt = SimpleNamespace(
            environment=environment,
            wm=wm,
            head=head,
            process=process,
            weights=spec.initial_parameters(design["cube_adapter_seed"]),
            spec=spec,
            counters=collections.Counter(),
        )
        try:
            rt.env = ScreenCubeEnv(rt.counters)
        finally:
            if constructor_receipt is not None:
                write_json(constructor_receipt, dict(rt.counters))
    if not rt.head.calibration or not rt.head.calibration["ready"]:
        raise RuntimeError("Frozen learned motion model is not released")
    return rt


def adapter_for(rt, role, shape):
    if role == "native":
        return None
    weights = rt.weights.copy()
    if role == "zero":
        weights[rt.spec.hidden_size * 30 :] = 0
    adapter = ScreenPushTAdapter if rt.environment == "pusht" else CubeProposalAdapter
    return adapter(rt.spec, weights, shape)


def push_trial(rt, case, pair, role, shape, seed, config, folder, cap):
    from contextlib import ExitStack
    from scripts.run_pusht_smoke import run_episode

    adapter = adapter_for(rt, role, shape)
    counters = collections.Counter()

    def checker(shape, margin, scaler, resources, enabled):
        if not enabled:
            raise RuntimeError("Learned checking must remain enabled")
        return LearnedMotionChecker(rt.head, pusht_constraints(shape), margin, resources)

    def solver(model, env, params, planner_seed, device):
        return PlanningCEM(model, env, params, planner_seed, device, proposal_adapter=adapter)

    def forbidden(*args, **kwargs):
        raise RuntimeError("Analytic/reference prediction is forbidden")

    protocol = {
        "planner": config["planner"],
        "replan_blocks": config["planner"]["replan_blocks"],
        "seed": seed,
        "margin_px": config["margin"],
        "max_episode_actions": cap,
        "max_runtime_minutes": 5,
        "verify_analytic_dynamics": False,
    }
    with ExitStack() as guard:
        for target in (
            "scripts.run_pusht_smoke.pd_response",
            "helpers.safecem10_planner.pd_response",
            "helpers.pusht_smoke.ShapeChecker.check",
            "helpers.pusht_smoke.ShapeChecker.paths",
        ):
            guard.enter_context(patch(target, side_effect=forbidden))
        result, _ = run_episode(
            role,
            shape,
            rt.wm,
            rt.process,
            pair,
            protocol,
            folder / "episode",
            time.time(),
            checker_factory=checker,
            planner_type=solver,
            device="cuda",
            resource_counters=counters,
        )
    result["engaged"] = result["actions"] > 0 and result["pusher_path_length_px"] > 1
    return result, adapter


def cube_trial(rt, case, pair, role, shape, seed, config, folder, cap):
    import gymnasium as gym
    import stable_worldmodel as swm
    from helpers.safecem10_planner import image_transform

    adapter = adapter_for(rt, role, shape)
    counters, env = rt.counters, rt.env
    counters.clear()
    env.journal = (folder / "interactions.jsonl").open("a", buffering=1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    observed = env.reset(seed, pair["initial"], pair["goal"], pair["goal_image"])
    if not np.allclose(observed, case["settled_initial"], atol=1e-8, rtol=0):
        raise RuntimeError("Cube settled initial state differs from preparation")
    checker = BodyMotionChecker(rt.head, shape_constraints(shape), config["margin"], counters)
    checker.observe(observed)
    params = config["planner"]
    cost = CostModel(rt.wm, checker, params, counters)
    solver = PlanningCEM(cost, "cube", params, seed, "cuda", proposal_adapter=adapter)
    policy = swm.policy.WorldModelPolicy(
        solver,
        swm.PlanConfig(horizon=5, receding_horizon=params["replan_blocks"], action_block=5),
        process={"action": rt.process["action"]},
        transform={"pixels": image_transform(), "goal": image_transform()},
    )
    policy.set_env(
        SimpleNamespace(
            num_envs=1,
            single_action_space=env.env.action_space,
            action_space=gym.spaces.Box(-1, 1, (1, 5), dtype=np.float64),
        )
    )
    states, cubes, actions, proposed, frames = (
        [observed.copy()],
        [env.env._data.joint("object_joint_0").qpos[:3].copy()],
        [],
        [],
        [],
    )
    dense = [observed[-2:].copy()]
    fresh_frames = []
    frame_actions = []
    cached_frame = None
    previous = np.zeros(5)
    success, failure, error = False, None, None
    started = time.perf_counter()
    planning = 0.0
    try:
        if env.success():
            raise RuntimeError("Already successful start")
        for _ in range(cap):
            fresh = not len(policy._action_buffer[0])
            if fresh or cached_frame is None:
                cached_frame = env.render()
            frame = cached_frame
            image_stats(frame)
            frames.append(frame)
            fresh_frames.append(fresh)
            frame_actions.append(len(actions))
            before = time.perf_counter()
            try:
                command = policy.get_action(
                    {
                        "pixels": frame[None, None],
                        "goal": pair["goal_image"][None, None],
                        "action": previous[None, None],
                    }
                )[0]
            finally:
                torch.cuda.synchronize()
                planning += time.perf_counter() - before
            executed = np.clip(np.asarray(command), -1, 1)
            observed, path = env.step(executed)
            checker.observe(observed)
            dense.extend(path)
            states.append(observed.copy())
            cubes.append(env.env._data.joint("object_joint_0").qpos[:3].copy())
            actions.append(executed.copy())
            proposed.append(np.asarray(command).copy())
            previous = executed
            success = env.success()
            write_json(folder / "live.json", {"actions": len(actions), "resources": dict(counters)})
            if success:
                break
            if time.perf_counter() - started > 300:
                raise TimeoutError("Episode runtime cap")
    except NoSafeAction as exc:
        failure = str(exc)
    except Exception:
        error = traceback.format_exc()
    elapsed = time.perf_counter() - started
    frames.append(env.render())
    fresh_frames.append(True)
    frame_actions.append(len(actions))
    dense = np.asarray(dense)
    clear = distance(dense, shape, config["body_radius"])
    unsafe = bool((clear < config["margin"] - 1e-6).any())
    pathlength = float(np.linalg.norm(np.diff(np.asarray(states)[:, -2:], axis=0), axis=1).sum())
    result = {
        "success": bool(success),
        "safe_completion": bool(success and not unsafe and not error and not failure),
        "margin_breach": unsafe,
        "arena_margin_breach": False,
        "hazard_violation": bool((clear <= 0).any()),
        "unfinished": not bool(success),
        "engaged": len(actions) > 0 and pathlength > 0.001,
        "protected_body_path_length": pathlength,
        "minimum_clearance": float(clear.min()),
        "actions": len(actions),
        "episode_seconds": elapsed,
        "planning_seconds": planning,
        "resources": dict(counters),
        "error": error,
        "controller_failure": failure,
        "goal_object_xyz": pair["goal"].tolist(),
        "final_object_xyz": cubes[-1].tolist(),
    }
    np.savez_compressed(
        folder / "trace.npz",
        states=states,
        actions=actions,
        proposed=proposed,
        dense=dense,
        frames=frames,
        goal_image=pair["goal_image"],
        cube_xyz=cubes,
        frame_fresh=fresh_frames,
        frame_actions=frame_actions,
    )
    write_json(folder / "replans.json", solver.replans)
    env.journal.close()
    env.journal = None
    return result, adapter


def trial(rt, case, role, kind, seed, config, folder, *, cap=150):
    folder.mkdir(parents=True, exist_ok=False)
    shape = case["shapes"][kind]
    write_json(
        folder / "started.json",
        {
            "environment": rt.environment,
            "pair": case["index"],
            "role": role,
            "kind": kind,
            "seed": seed,
            "cap": cap,
        },
    )
    with np.load(case["pair_path"], allow_pickle=False) as data:
        pair = dict(data)
    result, adapter = (push_trial if rt.environment == "pusht" else cube_trial)(
        rt, case, pair, role, shape, seed, config, folder, cap
    )
    prefix = folder / "episode" if rt.environment == "pusht" else folder
    with np.load(prefix / "trace.npz", allow_pickle=False) as data:
        flags = [i for i, frame in enumerate(data["frames"]) if image_stats(frame)["flagged"]]
    if adapter is not None:
        write_json(folder / "controller.json", adapter.artifact())
        write_json(folder / "proposals.json", adapter.history)
    result.update(
        environment=rt.environment,
        pair_index=case["index"],
        role=role,
        kind=kind,
        seed=seed,
        shape=shape,
        frame_flags=flags,
        maximum_predictor_steps=config["maximum_predictor_steps"],
        proposal_observation="causal native protected-body proprioception and frozen visual goal delta",
    )
    write_json(folder / "result.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "environment",
                    "pair_index",
                    "role",
                    "kind",
                    "seed",
                    "safe_completion",
                    "actions",
                    "frame_flags",
                    "error",
                )
            }
        ),
        flush=True,
    )
    if result["resources"].get("predictor_steps", 0) > config["maximum_predictor_steps"]:
        raise RuntimeError("Frozen per-episode predictor ceiling exceeded")
    if result["error"]:
        raise RuntimeError("Technical failure retained; no automatic retry")
    return result
