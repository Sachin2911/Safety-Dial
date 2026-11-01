"""Prospectively frozen PushT action/horizon/commitment diagnosis on one new pod."""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import importlib

importlib.import_module("triton")
import argparse
import collections
from contextlib import ExitStack
import datetime as dt
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm16_planner import NoSafeAction
from helpers.lewm18_planner import PlanningCEM
from helpers.lewm_navigation_evolution import ActionContract, ProjectedCostModel
from helpers.learned_motion import LearnedMotionChecker, load_motion
from helpers.lewm_proposal_pilot import write_json
from helpers.lewm_screen import image_stats
from helpers.pusht_smoke import ShapeChecker, Scaler, arena_clearance, physical_clearance
from helpers.safety_model import pusht_constraints, scaler_digest
from helpers.safecem10_planner import image_transform, pd_response
from scripts.run_pusht_smoke import dense_metrics


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
    action_contract=None,
    project_candidates=False,
    clip_execution=False,
    proposal_adapter=None,
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
    cost = ProjectedCostModel(
        model, checker, params, counters, action_contract, project=project_candidates
    )
    solver = planner_type(cost, "pusht", params, seed, device, proposal_adapter=proposal_adapter)
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
    states, actions, proposed_actions, frames = [initial.copy()], [], [], []
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
                raw_action = policy.get_action(task_info)[0]
                action = (
                    action_contract.execute(raw_action)
                    if clip_execution
                    else np.asarray(raw_action).copy()
                )
                proposed_actions.append(np.asarray(raw_action).copy())
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
        proposed=np.asarray(proposed_actions[: len(actions)]),
        proposed_all=np.asarray(proposed_actions),
        dense=dense,
        frames=np.asarray(frames),
        goal_image=pair["goal_image"],
    )
    write_json(directory / "replans.json", solver.replans)
    write_json(directory / "projection.json", cost.projection_history)
    write_json(directory / "result.json", result)
    if proposal_adapter is not None:
        write_json(directory / "controller.json", proposal_adapter.artifact())
        write_json(directory / "proposals.json", proposal_adapter.history)
    env.close()
    result["engaged"] = result["actions"] > 0 and result["pusher_path_length_px"] > 1
    result["native_action_out_of_bounds"] = (
        int(np.any(np.abs(np.asarray(actions)) > 1 + 1e-6, axis=-1).sum()) if actions else 0
    )
    result["object_displacement_after_action25_px"] = float(
        np.linalg.norm(
            np.asarray(states)[-1, 2:4] - np.asarray(states)[min(25, len(states) - 1), 2:4]
        )
    )
    result["goal_latent_cost_first_last"] = (
        [
            solver.replans[0]["features"]["current_cost"],
            solver.replans[-1]["features"]["current_cost"],
        ]
        if solver.replans
        else []
    )
    result["raw_proposal_count"] = len(proposed_actions)
    write_json(directory / "result.json", result)
    return result, dense[:, :2]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identities():
    names = [
        "experiments/scripts/run_lewm_push_diagnostic.py",
        "experiments/scripts/run_pusht_smoke.py",
        "experiments/helpers/lewm_push_evolution.py",
        "experiments/helpers/lewm_navigation_evolution.py",
        "experiments/helpers/lewm18_planner.py",
        "experiments/helpers/lewm16_planner.py",
        "experiments/helpers/cem_refinements.py",
        "experiments/helpers/lewm_proposal_adapter.py",
        "experiments/helpers/lewm_residual.py",
        "experiments/helpers/learned_motion.py",
        "experiments/helpers/pusht_smoke.py",
        "experiments/helpers/safety_model.py",
        "experiments/helpers/safecem10_planner.py",
    ]
    return {name: digest(ROOT / name) for name in names}


def reserve(design, started):
    now = dt.datetime.now(dt.timezone.utc)
    if now + dt.timedelta(minutes=20) >= dt.datetime.fromisoformat(design["deadline_utc"]):
        raise RuntimeError("Shared deadline preservation reserve reached")
    if time.time() - started >= design["maximum_stage_minutes"] * 60:
        raise RuntimeError("Frozen diagnostic elapsed-time limit reached")


def summarize(rows):
    return {
        "episodes": len(rows),
        "safe_completion": sum(r["safe_completion"] for r in rows),
        "violation": sum(r["margin_breach"] or r["arena_margin_breach"] for r in rows),
        "unfinished": sum(r["unfinished"] for r in rows),
        "engaged": sum(r["engaged"] for r in rows),
        "actions": sum(r["actions"] for r in rows),
        "predictor_steps": sum(r["resources"].get("predictor_steps", 0) for r in rows),
    }


def quality_gallery(out, rows, recipe):
    height, width = 248, 224
    canvas = Image.new("RGB", (5 * width, len(rows) * height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    for index, row in enumerate(rows):
        with np.load(out / "episodes" / row["key"] / "trace.npz", allow_pickle=False) as data:
            frames = data["frames"]
            for col, frame in enumerate(np.linspace(0, len(frames) - 1, 5).round().astype(int)):
                x, y = col * width, index * height
                canvas.paste(Image.fromarray(frames[frame]), (x, y + 24))
                draw.text(
                    (x + 4, y + 3),
                    f"{row['case_id']} s{row['seed']} a{frame}",
                    fill="black",
                    font=font,
                )
    folder = out / "quality"
    folder.mkdir(exist_ok=True)
    canvas.save(folder / (recipe + ".png"))


def run(design, out):
    if out.exists():
        raise FileExistsError("Use a new prospective namespace; never overwrite outcomes")
    out.mkdir(parents=True)
    started = time.time()
    frozen = source_identities()
    write_json(
        out / "source-freeze.json",
        {
            "protocol_sha256": digest(ROOT / design["protocol_path"]),
            "sources": frozen,
            "pod_id": design["execution_pod_id"],
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    for path, expected in design["asset_sha256"].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError("Frozen asset differs: " + path)
    from stable_worldmodel.policy import AutoCostModel

    checkpoint = ROOT / design["world_model"]
    model = AutoCostModel(str(checkpoint)[: -len("_object.ckpt")], cache_dir=checkpoint.parent)
    model = model.cuda().eval().requires_grad_(False)
    head, provenance = load_motion(ROOT / design["motion_model"], device="cuda")
    if head.spec.max_horizon < 5 or head.spec.action_block != 5 or head.spec.action_dim != 2:
        raise RuntimeError("The frozen motion head does not cover all diagnostic horizons")
    if head.spec.world_model_sha256 and head.spec.world_model_sha256 != digest(checkpoint):
        raise RuntimeError("Safety head and world model identity differ")
    if head.spec.body_radius != design["body_radius_px"]:
        raise RuntimeError("Protected body contract differs")
    if not head.calibration or not head.calibration["ready"]:
        raise RuntimeError("Frozen motion calibration is not ready")
    columns = json.loads((ROOT / design["scalers"]).read_text())["columns"]
    process = {name: Scaler(value["mean"], value["scale"]) for name, value in columns.items()}
    action = process["action"]
    if (
        head.spec.action_scaler_sha256
        and scaler_digest(action.mean_, action.scale_) != head.spec.action_scaler_sha256
    ):
        raise RuntimeError("Safety head and frozen action scaler identity differ")
    contract = ActionContract(tuple(action.mean_), tuple(action.scale_), (-1, -1), (1, 1))
    write_json(
        out / "runtime-contract.json",
        {
            "motion_spec": vars(head.spec),
            "motion_provenance": provenance,
            "action_contract": contract.record(),
            "motion_calibration": head.calibration,
            "new_simulator_preparation_resets": 0,
        },
    )
    state = {
        "phase": "running",
        "jobs": {},
        "active": None,
        "recipes": {},
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_json(out / "state.json", state)

    class AdmittedChecker(LearnedMotionChecker):
        def observe(self, observed):
            if self.observed is None:
                point = torch.as_tensor(np.asarray(observed)[:2][None], dtype=torch.float64)
                margin = design["margin_px"] + float(head.error_allowance)
                if (
                    min(
                        float(physical_clearance(point, self.case).min()),
                        float(arena_clearance(point).min()),
                    )
                    < margin
                ):
                    raise RuntimeError(
                        "Settled initial pusher violates the frozen admission margin"
                    )
            return super().observe(observed)

    def checker_factory(shape, margin, scaler, counters, enabled):
        if not enabled:
            raise RuntimeError("Learned checking remains enabled in every diagnostic")
        checker = AdmittedChecker(head, pusht_constraints(shape), margin, counters)
        checker.case = shape
        return checker

    def forbidden(*args, **kwargs):
        raise RuntimeError("Analytic/reference dynamics are forbidden in the diagnostic")

    try:
        for recipe in design["recipes"]:
            rows = []
            for case in design["cases"]:
                with np.load(ROOT / case["pair_path"], allow_pickle=False) as data:
                    pair = dict(data)
                for seed in case["seeds"]:
                    reserve(design, started)
                    if source_identities() != frozen:
                        raise RuntimeError("Frozen diagnostic source changed mid-stage")
                    key = f"{recipe['name']}_{case['id']}_s{seed}"
                    folder = out / "episodes" / key
                    starts = list((out / "episodes").glob("*/started.json"))
                    if len(starts) >= design["maximum_episode_starts"]:
                        raise RuntimeError("Frozen start cap reached")
                    folder.mkdir(parents=True, exist_ok=False)
                    write_json(
                        folder / "started.json",
                        {
                            "key": key,
                            "case_id": case["id"],
                            "seed": seed,
                            "recipe": recipe,
                            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        },
                    )
                    counters = collections.Counter()
                    state["active"] = key
                    write_json(out / "state.json", state)
                    protocol = {
                        "planner": recipe["planner"],
                        "replan_blocks": recipe["planner"]["replan_blocks"],
                        "seed": seed,
                        "margin_px": design["margin_px"],
                        "max_episode_actions": design["maximum_episode_actions"],
                        "max_runtime_minutes": 5,
                        "verify_analytic_dynamics": False,
                    }
                    try:
                        # The legacy function creates its own directory; this fresh inner
                        # directory retains raw frames while the outer folder journals starts.
                        with ExitStack() as guard:
                            for target in (
                                __name__ + ".pd_response",
                                "helpers.safecem10_planner.pd_response",
                                "helpers.pusht_smoke.ShapeChecker.check",
                                "helpers.pusht_smoke.ShapeChecker.paths",
                            ):
                                guard.enter_context(patch(target, side_effect=forbidden))
                            row, _ = run_episode(
                                "native",
                                case["shape"],
                                model,
                                process,
                                pair,
                                protocol,
                                folder / "episode",
                                time.time(),
                                checker_factory=checker_factory,
                                device="cuda",
                                resource_counters=counters,
                                action_contract=contract,
                                project_candidates=recipe["project_candidates"],
                                clip_execution=recipe["clip_execution"],
                            )
                        row.update(
                            key=key,
                            case_id=case["id"],
                            recipe=recipe["name"],
                            project_candidates=recipe["project_candidates"],
                            clip_execution=recipe["clip_execution"],
                            pair_index=case["pair_index"],
                            source_episode=case["source_episode"],
                            source_row=case["source_row"],
                        )
                        with np.load(folder / "episode/trace.npz", allow_pickle=False) as trace:
                            row["frame_flags"] = [
                                i
                                for i, frame in enumerate(trace["frames"])
                                if image_stats(frame)["flagged"]
                            ]
                        write_json(folder / "result.json", row)
                        state["jobs"][key] = row
                        rows.append(row)
                        if (
                            row["resources"].get("predictor_steps", 0)
                            > design["maximum_predictor_steps_per_episode"]
                        ):
                            raise RuntimeError("Per-episode predictor cap exceeded")
                        if row["error"]:
                            raise RuntimeError(
                                "Technical episode error retained; diagnose before a new stage"
                            )
                    except BaseException:
                        write_json(
                            folder / "technical-error.json",
                            {
                                "key": key,
                                "error": traceback.format_exc(),
                                "resources": dict(counters),
                            },
                        )
                        raise
                    state["active"] = None
                    write_json(out / "state.json", state)
                    print(
                        json.dumps(
                            {
                                "key": key,
                                "safe_completion": row["safe_completion"],
                                "actions": row["actions"],
                                "error": row["error"],
                            }
                        ),
                        flush=True,
                    )
            state["recipes"][recipe["name"]] = summarize(rows)
            quality_gallery(
                out, [{**row, "key": row["key"] + "/episode"} for row in rows], recipe["name"]
            )
            write_json(out / "state.json", state)
        if len(state["jobs"]) != design["maximum_scored_episodes"]:
            raise RuntimeError("Frozen diagnostic schedule is incomplete")
        state["phase"] = "complete"
        state["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    except BaseException:
        state["phase"] = "technical_failure"
        state["error"] = traceback.format_exc()
        raise
    finally:
        write_json(out / "state.json", state)
    print(
        json.dumps(
            {"phase": state["phase"], "recipes": state["recipes"], "episodes": len(state["jobs"])}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.protocol.read_text())
    if os.environ.get("RUNPOD_POD_ID") != design["execution_pod_id"]:
        raise RuntimeError("Only the declared new rented pod may run this stage")
    if not torch.cuda.is_available():
        raise RuntimeError("The declared rented GPU is required")
    torch.set_num_threads(4)

    def stop(signum, frame):
        raise InterruptedError("Shared campaign process stop requested")

    signal.signal(signal.SIGTERM, stop)
    try:
        run(design, args.out)
    except BaseException:
        if args.out.exists():
            write_json(
                args.out / "worker-error.json",
                {"error": traceback.format_exc(), "episode_logs_are_authoritative": True},
            )
        raise


if __name__ == "__main__":
    main()
