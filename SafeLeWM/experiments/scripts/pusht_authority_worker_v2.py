"""PushT adapter execution, derived from the qualified manipulation episode runner.

The original manipulation runner remains unchanged. Every correction is recorded.
"""

import collections
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import manip as M  # noqa: E402
from manip_pipeline import (  # noqa: E402
    _near_waypoint,
    witness_plan,
    witness_role,
    witness_check,
)


def clip_native_action(action):
    """Execute the existing native-box contract, preserving the caller's proposal."""
    action = np.asarray(action, dtype=np.float32)
    if not np.isfinite(action).all():
        raise ValueError("Nonfinite action")
    return np.clip(action, -1, 1).astype(np.float32)


def run_episode(
    task,
    folder,
    case,
    arm,
    seed,
    data,
    native,
    bank,
    totals,
    deadline,
    hazard=None,
    witness=None,
    parameters=None,
):
    """One episode of any arm. Witness arms use the recording and scripted waypoints only."""
    import torch

    from helpers.pusht_authority import features
    from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter
    from anchored_planning.planners import TargetGaussian

    spec = M.TASKS[task]
    name = f"{case['id']}_{arm}_{seed}"
    out = folder / name
    out.mkdir()
    actor = None
    if parameters is not None:
        actor = RecurrentActionAdapter(AdapterSpec(402, 2, initialization_seed=13101)).eval()
        actor.load_vector(parameters)
        hidden = actor.initial_state()
    previous = np.zeros(2, np.float32)
    displacement = previous.copy()
    counts = collections.Counter()
    arrays = M.case_arrays(task, data, case["start_row"], case["goal_row"])
    goal_image = np.asarray(data["pixels"][case["goal_row"]])
    trace = dict(
        states=[],
        points=[],
        actions=[],
        proposed_actions=[],
        frames=[],
        hit=[],
        rewards=[],
        terminated=[],
        truncated=[],
        success=[],
        phase=[],
        base_actions=[],
        inputs=[],
        hidden=[],
        gains=[],
        residuals=[],
        current_latents=[],
        goal_latents=[],
    )
    decisions = []
    env = M.make_env(task)
    handle = None
    original_cost = None
    success = False
    witness_invalid = None
    status = "incomplete"
    started = time.monotonic()
    budget = spec["budget"]
    monitored = arm in {"monitored", "replay", "scripted"} and hazard is not None
    centre = np.asarray(hazard["hazard_centre"]) if hazard is not None else None
    try:
        counts["environment_resets"] += 1
        M.save(out / "live.json", dict(resources=dict(counts)))
        _, info = M.reset_case(env, task, arrays, seed)
        phys = M.physical(env, task)
        frame = np.ascontiguousarray(env.render())
        discrepancy = float(
            np.abs(
                frame.astype(float) - np.asarray(data["pixels"][case["start_row"]]).astype(float)
            ).mean()
        )
        trace["states"].append(phys["state"])
        trace["points"].append(np.asarray(phys["points"]))
        trace["frames"].append(frame)
        success = M.is_success(env, task, info, arrays)
        initial_success = success
        pending = []
        checked = False
        if arm in {"final", "ap", "clean", "monitored"}:

            def encoder_hook(_module, args):
                counts["encoder_frames"] += int(args[0].shape[0])

            handle = native.model.encoder.register_forward_pre_hook(encoder_hook)
            original_cost = native.latent_cost.get_cost

            def counted_cost(info_, actions):
                counts["model_batch_calls"] += 1
                counts["candidate_model_steps"] += int(np.prod(actions.shape[:3]))
                return original_cost(info_, actions)

            native.latent_cost.get_cost = counted_cost
            goal_encoding = native.encode_rgb(goal_image).cpu().numpy()

            class Images:
                def __getitem__(self, index):
                    return goal_image if int(index) == -1 else data["pixels"][int(index)]

            planner = TargetGaussian(native, Images())
            planner.set_action_bounds(env.action_space.low, env.action_space.high)
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            recorded = np.asarray(data["action"][case["start_row"] : case["goal_row"]], np.float32)
            plan = witness_plan(task, arm, case, hazard, recorded, spec)
            plan_index = 0
            phase_steps = 0
            waypoint_steps = 0
            release_steps = 0
        step = 0
        while step < budget and not success:
            if time.monotonic() >= deadline:
                raise TimeoutError("Frozen job deadline reached")
            before_points = trace["points"][-1]
            inputs, state_log = np.empty(0, np.float32), np.empty(0, np.float32)
            gain, residual = np.ones(2, np.float32), np.zeros(2, np.float32)
            if arm in {"final", "ap", "clean", "monitored"}:
                phase = "planner"
                if not pending:
                    if arm == "final":
                        sources = [-1]
                        retrieved = None
                    else:
                        current = native.encode_rgb(trace["frames"][-1]).cpu().numpy()
                        remaining = max(5, case["horizon"] - step)
                        sources_array, distances = bank.nearest_topk(
                            current, goal_encoding, remaining, 1
                        )
                        sources = sources_array[0].tolist()
                        retrieved = distances[0].tolist()
                        assert bank.train_mask[bank.episode_idx[sources[0]]]
                        counts["retrieval_queries"] += 1
                    calls_before, blocks_before = (
                        counts["model_batch_calls"],
                        counts["candidate_model_steps"],
                    )
                    selected, detail = planner.choose(
                        "gaussian_final" if arm == "final" else "gaussian_observed",
                        trace["frames"][-1],
                        sources,
                        generator,
                        -1,
                    )
                    assert (
                        counts["model_batch_calls"] - calls_before
                        == detail["predictor_batch_calls"]
                    )
                    assert (
                        counts["candidate_model_steps"] - blocks_before
                        == detail["predictor_candidate_transitions"]
                    )
                    counts["replans"] += 1
                    detail.update(step=step, retrieval_distances=retrieved, sources=sources)
                    decisions.append(detail)
                    pending = list(selected)
                action = np.asarray(pending.pop(0), np.float32)
                base = action.copy()
                proposed = action.copy()
                if actor is not None:
                    inputs = features(
                        current,
                        goal_encoding,
                        trace["states"][-1],
                        centre,
                        base,
                        previous,
                        displacement,
                        step,
                    )
                    with torch.inference_mode():
                        output, hidden, details = actor(
                            torch.from_numpy(inputs)[None], hidden, torch.from_numpy(base)[None]
                        )
                    action = output[0].numpy().copy()
                    proposed = details["proposed"][0].numpy().copy()
                    gain = details["gain"][0].numpy().copy()
                    residual = details["residual"][0].numpy().copy()
                    state_log = hidden[0].numpy().copy()
                    counts["adapter_calls"] += 1
            else:
                if plan_index >= len(plan):
                    break
                kind, payload = plan[plan_index]
                if kind == "replay":
                    phase = "replay"
                    if phase_steps >= len(payload):
                        plan_index += 1
                        phase_steps = 0
                        continue
                    action = np.asarray(payload[phase_steps], np.float32)
                    phase_steps += 1
                elif kind == "goto":
                    phase = "scripted"
                    target, settle, checked = payload
                    reached = (
                        M.at_waypoint(task, env, target, spec)
                        if settle
                        else _near_waypoint(task, env, target, spec)
                    )
                    if reached:
                        plan_index += 1
                        waypoint_steps = 0
                        continue
                    waypoint_steps += 1
                    if waypoint_steps > spec["settle_steps"] + 40:
                        witness_invalid = f"waypoint {plan_index} not reached"
                        break
                    action = M.witness_action(task, env, target, hold_closed=True)
                elif kind == "release":
                    phase = "release"
                    if release_steps >= payload:
                        plan_index += 1
                        continue
                    release_steps += 1
                    target = np.asarray(M.physical(env, task)["points"][0])
                    action = M.witness_action(task, env, target, hold_closed=False)
                elif kind == "hold":
                    phase = "hold"
                    if phase_steps >= payload:
                        plan_index += 1
                        phase_steps = 0
                        continue
                    phase_steps += 1
                    action = np.zeros(spec["action_dim"], np.float32)
                proposed = action.copy()
            if actor is None:
                base = action.copy()
            counts["clipped_actions"] += int(np.any(np.abs(proposed) > 1))
            action = clip_native_action(action)
            if actor is None:
                base = clip_native_action(base)
            counts["environment_actions"] += 1
            if counts["environment_actions"]:
                M.save(out / "live.json", dict(resources=dict(counts)))
            _, reward, terminated, truncated, info = env.step(action)
            phys = M.physical(env, task)
            after_points = np.asarray(phys["points"])
            hit = bool(monitored and M.monitor_hit(task, before_points, after_points, centre))
            success = M.is_success(env, task, info, arrays)
            trace["states"].append(phys["state"])
            trace["points"].append(after_points)
            trace["frames"].append(np.ascontiguousarray(env.render()))
            trace["base_actions"].append(base)
            trace["current_latents"].append(
                np.asarray(current).reshape(-1) if actor is not None else np.empty(0)
            )
            trace["goal_latents"].append(
                np.asarray(goal_encoding).reshape(-1) if actor is not None else np.empty(0)
            )
            trace["inputs"].append(inputs)
            trace["hidden"].append(state_log)
            trace["gains"].append(gain)
            trace["residuals"].append(residual)
            displacement = after_points[0] - before_points[0]
            previous = action.copy()
            trace["actions"].append(action)
            trace["proposed_actions"].append(proposed)
            trace["hit"].append(hit)
            trace["rewards"].append(float(reward))
            trace["terminated"].append(bool(terminated))
            trace["truncated"].append(bool(truncated))
            trace["success"].append(bool(success))
            trace["phase"].append(phase)
            step += 1
            if arm == "scripted" and phase == "scripted" and checked:
                witness_invalid = witness_invalid or witness_check(
                    task, env, trace, case, hazard, spec
                )
            if task == "pusht" and bool(terminated):
                success = True
            if (terminated or truncated) and not success:
                break
        status = "complete"
    except BaseException:
        status = "failed"
        (out / "traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        if handle is not None:
            handle.remove()
        if original_cost is not None:
            native.latent_cost.get_cost = original_cost
        env.close()
        np.savez_compressed(
            out / "trace.npz",
            **{k: np.asarray(v) for k, v in trace.items()},
            hazard_centre=centre if centre is not None else np.zeros(0),
            goal_state=arrays.get("goal_state", arrays.get("goal_pos")),
        )
        M.save(out / "planning.json", decisions)
        totals.update(counts)
        M.save(out / "execution.json", dict(status=status, resources=dict(counts)))
    points = np.asarray(trace["points"])
    path = M.polyline_length(points[:, 0])
    violation = bool(any(trace["hit"])) if monitored else None
    clearance = None
    if centre is not None and len(points) > 1:
        clearance = min(
            M.segment_distance(a[i][:2], b[i][:2], centre)
            for a, b in zip(points[:-1], points[1:])
            for i in range(len(a))
        ) - M.exclusion(task)
    result = dict(
        id=name,
        case=case["id"],
        family=(hazard or {}).get("family"),
        arm=arm,
        seed=seed,
        role=witness_role((hazard or {}).get("family"), arm),
        success=bool(success),
        initial_success=bool(initial_success),
        violation=violation,
        safe_completion=(bool(success) and not violation) if monitored else None,
        unfinished=not bool(success),
        engaged=path > spec["engaged_path"],
        path_length=path,
        minimum_clearance=None if clearance is None else float(clearance),
        witness_invalid=witness_invalid,
        start_image_discrepancy=discrepancy,
        steps=step,
        resources=dict(counts),
        seconds=time.monotonic() - started,
    )
    M.save(out / "result.json", result)
    print(json.dumps(result), flush=True)
    return result
