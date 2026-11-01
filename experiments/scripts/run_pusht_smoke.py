"""One-scene frozen-LeWM engineering demo with three virtual pusher-only hazards."""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.lewm16_planner import NoSafeAction
from helpers.lewm18_planner import CostModel, PlanningCEM
from helpers.pusht_smoke import (
    ShapeChecker,
    Scaler,
    arena_clearance,
    case_geometry,
    physical_clearance,
)
from helpers.safecem10_planner import image_transform, pd_response


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".part")
    staging.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    staging.replace(path)


def dense_metrics(xy, case, margin):
    xy = torch.as_tensor(xy, dtype=torch.float64)
    clearance = physical_clearance(xy, case).numpy()
    arena = arena_clearance(xy).numpy()
    return {
        "hazard_violation": bool((clearance <= 0).any()),
        "margin_breach": bool((clearance < margin - 1e-6).any()),
        "arena_margin_breach": bool((arena < margin - 1e-6).any()),
        "min_hazard_clearance_px": float(clearance.min()),
        "min_arena_clearance_px": float(arena.min()),
    }


def run_episode(
    role,
    case,
    model,
    process,
    pair,
    protocol,
    directory,
    campaign_started,
    *,
    checker_factory=None,
    planner_type=PlanningCEM,
    device="cuda",
    resource_counters=None,
):
    import gymnasium as gym
    import stable_worldmodel as swm
    from stable_worldmodel.envs.pusht.env import PushT

    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "started.json", {"role": role, "seed": protocol["seed"], "case": case})
    counters = collections.defaultdict(int) if resource_counters is None else resource_counters

    class CountedPushT(PushT):
        def _set_state(self, state):
            original = self.space.step

            def reset_step(dt):
                counters["reset_physics_steps"] += 1
                return original(dt)

            self.space.step = reset_step
            try:
                return super()._set_state(state)
            finally:
                self.space.step = original

        def render(self):
            counters["render_frames"] += 1
            return super().render()

    seed = protocol["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = CountedPushT()
    counters["environment_resets"] += 1
    observation, info = env.reset(
        seed=seed,
        options={"state": pair["initial_state"], "goal_state": pair["goal_state"]},
    )
    radius = next(iter(env.agent.shapes)).radius
    if abs(radius - 15) > 1e-8:
        raise RuntimeError("Pusher radius differs from frozen protocol")
    initial = np.asarray(observation["state"], dtype=float).copy()
    goal = pair["goal_state"]
    env._goal = pair["goal_image"].copy()
    info = env._get_info()
    dense = [initial.copy()]
    original_step = env.space.step

    def dense_step(dt):
        original_step(dt)
        counters["executed_physics_steps"] += 1
        dense.append(np.asarray(env._get_obs(), dtype=float).copy())

    env.space.step = dense_step
    params = {**protocol["planner"], "checked": role != "reference"}
    checker_type = ShapeChecker if checker_factory is None else checker_factory
    checker = checker_type(
        case, protocol["margin_px"], process["action"], counters, enabled=params["checked"]
    )
    if hasattr(checker, "observe"):
        checker.observe(observation["proprio"])
    else:
        checker.state = initial.copy()
    cost = CostModel(model, checker, params, counters)
    solver = planner_type(cost, "pusht", params, seed, device)
    transforms = {"pixels": image_transform(), "goal": image_transform()}
    if checker_factory is not None and not hasattr(checker, "observe"):
        transforms["pixels_previous"] = image_transform()
    policy = swm.policy.WorldModelPolicy(
        solver,
        swm.PlanConfig(
            horizon=5, receding_horizon=protocol.get("replan_blocks", 1), action_block=5
        ),
        process=process,
        transform=transforms,
    )
    policy.set_env(
        SimpleNamespace(
            num_envs=1,
            single_action_space=env.action_space,
            action_space=gym.spaces.Box(-1, 1, (1, 2), dtype=np.float64),
        )
    )
    states, actions, frames = [initial.copy()], [], []
    state, previous_action, previous_pixels = initial.copy(), np.zeros(2), None
    success, failure, error = False, None, None
    pd_error, planning_seconds = 0.0, 0.0
    started = time.perf_counter()
    verify_analytic = protocol.get("verify_analytic_dynamics", True)
    c, g = pd_response(1) if verify_analytic else (None, None)
    try:
        if env.eval_state(goal, initial)[0]:
            raise RuntimeError("Metadata-selected initial state is already successful")
        for _ in range(protocol["max_episode_actions"]):
            if time.time() - campaign_started > protocol["max_runtime_minutes"] * 60:
                raise TimeoutError("Smoke-test runtime allowance exhausted")
            pixels = np.ascontiguousarray(env.render())
            frames.append(pixels)
            task_info = {
                "pixels": pixels[None, None],
                "goal": pair["goal_image"][None, None],
                "action": previous_action[None, None],
            }
            if checker_factory is not None and not hasattr(checker, "observe"):
                task_info["pixels_previous"] = (
                    pixels if previous_pixels is None else previous_pixels
                )[None, None]
                task_info["safety_history_valid"] = np.array([previous_pixels is not None])
            before = time.perf_counter()
            try:
                action = policy.get_action(task_info)[0]
            finally:
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                planning_seconds += time.perf_counter() - before
            if verify_analytic:
                expected = state[:2] + c[:, None] * state[-2:] + g @ np.asarray(action)[None]
            previous_dense = len(dense)
            counters["environment_actions"] += 1
            observation, _, _, _, info = env.step(action)
            state = np.asarray(observation["state"], dtype=float).copy()
            states.append(state)
            actions.append(np.asarray(action).copy())
            realized = np.asarray(dense[previous_dense:])[:, :2]
            if verify_analytic:
                pd_error = max(pd_error, float(np.max(np.abs(expected - realized))))
                if pd_error > 1e-4:
                    raise RuntimeError("Analytic pusher path does not match physics")
            if hasattr(checker, "observe"):
                checker.observe(observation["proprio"])
            else:
                checker.state = state.copy()
            previous_action = np.asarray(action)
            previous_pixels = pixels
            success = bool(env.eval_state(goal, state)[0])
            write_json(directory / "live.json", {"actions": len(actions), "resources": counters})
            if success:
                break
    except NoSafeAction as exc:
        failure = str(exc)
    except Exception:
        error = traceback.format_exc()
    elapsed = time.perf_counter() - started
    frames.append(np.ascontiguousarray(env.render()))
    dense = np.asarray(dense)
    metrics = dense_metrics(dense[:, :2], case, protocol["margin_px"])
    safe = (
        success
        and not any(metrics[k] for k in ("margin_breach", "arena_margin_breach"))
        and failure is None
        and error is None
    )
    result = {
        "role": role,
        "case": case,
        "seed": seed,
        "success": success,
        "safe_completion": safe,
        "unfinished": not success,
        **metrics,
        "actions": len(actions),
        "simulated_seconds": len(actions) * 0.1,
        "episode_seconds": elapsed,
        "planning_seconds": planning_seconds,
        "pd_max_error_px": pd_error if verify_analytic else None,
        "resources": dict(counters),
        "controller_failure": failure,
        "error": error,
        "initial_state": initial.tolist(),
        "goal_state": goal.tolist(),
        "final_state": state.tolist(),
        "joint_position_error_px": float(np.linalg.norm(goal[:4] - state[:4])),
        "object_displacement_px": float(np.linalg.norm(state[2:4] - initial[2:4])),
        "pusher_path_length_px": float(
            np.linalg.norm(np.diff(np.asarray(states)[:, :2], axis=0), axis=1).sum()
        ),
    }
    if checker_factory is not None:
        result["safety_model"] = {
            "mode": checker.mode,
            "calibration": checker.model.calibration,
            "reference_controls_execution": checker.mode == "shadow",
        }
    np.savez_compressed(
        directory / "trace.npz",
        states=states,
        actions=actions,
        dense=dense,
        frames=np.asarray(frames),
        goal_image=pair["goal_image"],
    )
    write_json(directory / "replans.json", solver.replans)
    write_json(directory / "result.json", result)
    env.close()
    print(json.dumps(result), flush=True)
    return result, dense[:, :2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--role", choices=["reference", "island", "bar", "gate"])
    parser.add_argument("--label")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = ROOT / "configs/pusht_smoke_20261029/protocol.json"
    protocol = json.loads(config.read_text())
    pair = dict(np.load(args.data / "pair.npz"))
    manifest = args.out / "source-freeze.json"
    files = [config, Path(__file__), ROOT / "experiments/helpers/pusht_smoke.py"]
    files += [
        ROOT / f"experiments/helpers/{name}.py"
        for name in (
            "lewm18_planner",
            "cem_refinements",
            "lewm18_runtime",
            "lewm16_planner",
            "safecem10_planner",
            "cube14_runtime",
        )
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    hashes["scalers.json"] = hashlib.sha256((args.data / "scalers.json").read_bytes()).hexdigest()
    if not manifest.exists():
        write_json(
            manifest,
            {
                "frozen_epoch": time.time(),
                "protocol": protocol,
                "source_sha256": hashes,
                "pair": json.loads((args.data / "pair.json").read_text()),
            },
        )
    else:
        original = json.loads(manifest.read_text())
        if hashes != original["source_sha256"]:
            if not args.label:
                raise RuntimeError("Changed source requires an explicit preserved repair label")
            write_json(args.out / f"amendment-{args.label}.json", {"source_sha256": hashes})
    campaign_started = json.loads(manifest.read_text())["frozen_epoch"]
    import stable_worldmodel as swm

    model = swm.policy.AutoCostModel("pusht/lewm").to("cuda").eval()
    model.requires_grad_(False)
    normalizers = json.loads((args.data / "scalers.json").read_text())["columns"]
    process = {key: Scaler(value["mean"], value["scale"]) for key, value in normalizers.items()}
    placeholder = {
        "kind": "island",
        "radius_px": 12,
        "origin": ((pair["initial_state"][:2] + pair["goal_state"][:2]) / 2).tolist(),
        "basis": [[1, 0], [0, 1]],
    }
    roles = [args.role] if args.role else protocol["order"]
    for role in roles:
        previous = list(args.out.glob("runs/*/started.json"))
        if len(previous) >= protocol["max_rollouts"]:
            raise RuntimeError("Frozen maximum number of smoke-test rollouts reached")
        prior_actions = sum(
            json.loads(p.read_text())["actions"] for p in args.out.glob("runs/*/result.json")
        )
        if (
            prior_actions + protocol["max_episode_actions"]
            > protocol["max_total_environment_actions"]
        ):
            raise RuntimeError("Frozen interaction allowance exhausted")
        if role == "reference":
            case = placeholder
        else:
            case = json.loads((args.out / "cases.json").read_text())[role]
        result, path = run_episode(
            role,
            case,
            model,
            process,
            pair,
            protocol,
            args.out / "runs" / (args.label or role),
            campaign_started,
        )
        if result["error"]:
            raise RuntimeError(result["error"])
        if role == "reference":
            cases = case_geometry(result["initial_state"], pair["goal_state"], path, protocol)
            write_json(args.out / "cases.json", cases)
            for kind, geometry in cases.items():
                rescored = dense_metrics(path, geometry, protocol["margin_px"])
                write_json(
                    args.out / f"reference-rescored-{kind}.json",
                    {
                        **rescored,
                        "success": result["success"],
                        "actions": result["actions"],
                        "note": "Same reference trajectory rescored, not another rollout.",
                    },
                )
    results = [json.loads(p.read_text()) for p in sorted(args.out.glob("runs/*/result.json"))]
    write_json(
        args.out / "summary.json",
        {
            "engineering_only": True,
            "results": results,
            "total_environment_actions": sum(
                r["resources"].get("environment_actions", 0) for r in results
            ),
            "total_physics_steps": sum(
                r["resources"].get("executed_physics_steps", 0)
                + r["resources"].get("reset_physics_steps", 0)
                for r in results
            ),
        },
    )


if __name__ == "__main__":
    main()
