"""Qualified environment execution with optional sampled PPO logits."""

import collections
import sys
import time


def run_episode(
    spec,
    parameters,
    native,
    bank,
    data,
    output,
    deadline,
    device,
    action_limit=200,
    policy=None,
    sampling_seed=0,
):
    import numpy as np
    import torch
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    import anchored_pipeline as ap
    from helpers.replication_budget import episode_status
    from helpers.pilot_common import features
    from helpers.mechanism import actor_step, mask_inputs
    from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter

    assert 1 <= action_limit <= 200
    case, arm, seed = spec["case"], spec["arm"], spec["seed"]
    folder = output / "episodes" / spec["id"]
    folder.mkdir(parents=True, exist_ok=False)
    ap.save(folder / "spec.json", {**spec, "device": device, "action_limit": action_limit})
    counts = collections.Counter()
    keys = [
        "states",
        "frames",
        "actions",
        "base_actions",
        "proposed_actions",
        "gains",
        "residuals",
        "inputs",
        "hidden",
        "hit",
        "rewards",
        "terminated",
        "truncated",
    ]
    keys += ["raw_logits", "log_probability", "values"]
    trace = {key: [] for key in keys}
    decisions = []
    started = time.monotonic()
    env = TwoRoomEnv()
    handle, original_cost = None, None
    success, status = False, "incomplete"
    witness = arm in ("safe", "risky")
    actor = None
    if parameters is not None:
        actor = RecurrentActionAdapter(AdapterSpec(396, 2)).eval()
        actor.load_vector(parameters)
        hidden = actor.initial_state()
    if policy is not None:
        assert parameters is None
        actor = policy
        hidden = policy.initial_state()
        sampling_generator = torch.Generator().manual_seed(sampling_seed)
    try:
        with torch.inference_mode():
            counts["environment_resets"] += 1
            ap.save(folder / "live.json", dict(resources=dict(counts)))
            env.reset(
                seed=seed,
                options={
                    "state": np.asarray(case["initial"], dtype=np.float32),
                    "target_state": np.asarray(case["goal"], dtype=np.float32),
                },
            )
            np.testing.assert_array_equal(env.agent_position.numpy(), case["initial"])
            assert env.wall_axis == 1 and env.wall_thickness == 10 and env.num_doors == 1
            np.testing.assert_array_equal(env.door_positions.numpy(), [49])
            np.testing.assert_array_equal(env.door_sizes.numpy(), [14])
            assert float(env.variation_space["agent"]["radius"].value.item()) == 7
            assert float(env.variation_space["agent"]["speed"].value.item()) == 5
            trace["states"].append(env.agent_position.numpy().copy())
            trace["frames"].append(env.render())
            if witness:
                path = np.asarray(case[arm + "_path"], dtype=np.float32)
                waypoint = 0
            else:

                def encoder_hook(_module, args):
                    counts["encoder_frames"] += int(args[0].shape[0])

                handle = native.model.encoder.register_forward_pre_hook(encoder_hook)
                original_cost = native.latent_cost.get_cost

                def counted_cost(info, actions):
                    counts["model_batch_calls"] += 1
                    counts["candidate_model_steps"] += int(np.prod(actions.shape[:3]))
                    return original_cost(info, actions)

                native.latent_cost.get_cost = counted_cost
                goal_image = env._target_img.permute(1, 2, 0).numpy().copy()
                goal_encoding = native.encode_rgb(goal_image).cpu().numpy()
                planner = ap.TargetGaussian(native, data["pixels"])
                planner.set_action_bounds(env.action_space.low, env.action_space.high)
                generator = torch.Generator(device="cuda").manual_seed(seed)
            pending = []
            previous = np.zeros(2, dtype=np.float32)
            displacement = previous.copy()
            for step in range(action_limit):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Frozen job deadline reached")
                before = trace["states"][-1]
                inputs, state_log = np.empty(0), np.empty(0)
                gain, residual = np.ones(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
                if witness:
                    while waypoint < len(path) - 1 and np.linalg.norm(before - path[waypoint]) <= 1:
                        waypoint += 1
                    proposed = ((path[waypoint] - before) / 5).astype(np.float32)
                    action = np.clip(proposed, -1, 1)
                    base = action.copy()
                else:
                    if not pending:
                        current = native.encode_rgb(trace["frames"][-1]).cpu().numpy()
                        sources, distances = bank.nearest_topk(
                            current,
                            goal_encoding,
                            max(5, 100 - step),
                            1,
                        )
                        source = int(sources[0, 0])
                        assert bank.train_mask[bank.episode_idx[source]]
                        counts["retrieval_queries"] += 1
                        calls_before = counts["model_batch_calls"]
                        blocks_before = counts["candidate_model_steps"]
                        selected, detail = planner.choose(
                            "gaussian_observed",
                            trace["frames"][-1],
                            [source],
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
                        detail.update(
                            step=step, source=source, retrieval_distances=distances[0].tolist()
                        )
                        decisions.append(detail)
                        pending = list(selected)
                    base = np.asarray(pending.pop(0), dtype=np.float32)
                    proposed, action = base.copy(), base.copy()
                    if actor is not None:
                        inputs = features(
                            current,
                            goal_encoding,
                            before,
                            case["hazard_centre"],
                            base,
                            previous,
                            displacement,
                            step,
                        )
                        inputs = mask_inputs(inputs, arm)
                        if policy is None:
                            result, hidden, detail = actor_step(
                                actor,
                                torch.from_numpy(inputs)[None],
                                hidden,
                                torch.from_numpy(base)[None],
                                arm,
                            )
                        else:
                            result, hidden, detail = policy.sample(
                                torch.from_numpy(inputs)[None],
                                hidden,
                                torch.from_numpy(base)[None],
                                sampling_generator,
                            )
                            for key in ["raw_logits", "log_probability", "values"]:
                                trace[key].append(detail[key][0].numpy().copy())
                        counts["adapter_calls"] += 1
                        action = result[0].numpy().copy()
                        proposed = detail["proposed"][0].numpy().copy()
                        gain = detail["gain"][0].numpy().copy()
                        residual = detail["residual"][0].numpy().copy()
                        state_log = hidden[0].numpy().copy()
                assert np.isfinite(action).all() and (np.abs(action) <= 1).all()
                counts["environment_actions"] += 1
                counts["clipped_actions"] += int(np.any(np.abs(proposed) > 1))
                ap.save(folder / "live.json", dict(resources=dict(counts)))
                _, reward, terminated, truncated, _ = env.step(action)
                after = env.agent_position.numpy().copy()
                hit = ap.segment_distance(before, after, case["hazard_centre"]) <= 18
                values = dict(
                    actions=action,
                    base_actions=base,
                    proposed_actions=proposed,
                    gains=gain,
                    residuals=residual,
                    inputs=inputs,
                    hidden=state_log,
                    hit=hit,
                    rewards=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                for key, value in values.items():
                    trace[key].append(value)
                trace["states"].append(after)
                trace["frames"].append(env.render())
                displacement = after - before
                previous = action.copy()
                success = bool(terminated)
                if terminated or truncated:
                    break
            status = episode_status(
                counts["environment_actions"], success or truncated, action_limit
            )
    finally:
        if handle is not None:
            handle.remove()
        if original_cost is not None:
            native.latent_cost.get_cost = original_cost
        env.close()
        np.savez_compressed(folder / "trace.npz", **trace, goal=case["goal"])
        ap.save(folder / "planning.json", decisions)
        ap.save(folder / "execution.json", dict(status=status, resources=dict(counts)))
    positions = np.asarray(trace["states"])
    length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    violation = bool(any(trace["hit"]))
    result = dict(
        id=spec["id"],
        status=status,
        action_limit=action_limit,
        repeat=spec.get("repeat"),
        candidate=spec.get("candidate"),
        stage=spec["stage"],
        representation=spec.get("representation"),
        search=spec.get("search"),
        case=case["id"],
        family=case["family"],
        arm=arm,
        seed=seed,
        device=device,
        success=success,
        violation=violation,
        safe_completion=success and not violation,
        unfinished=not success,
        engaged=length > 1,
        path_length=length,
        resources=dict(counts),
        correction_norm_mean=float(
            np.linalg.norm(np.asarray(trace["actions"]) - trace["base_actions"], axis=1).mean()
        ),
        seconds=time.monotonic() - started,
    )
    ap.save(folder / "result.json", result)
    return result


if __name__ == "__main__":
    sys.exit("Use run_matched_ppo.py to launch the prospectively frozen worker schedule.")
