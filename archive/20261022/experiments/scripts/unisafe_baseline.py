"""Run paired frozen UNISafe controllers and preserve physical rollout evidence."""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe_source import ROOT, prepare_source, save_implementation, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/unisafe/baseline.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--modes", choices=("unfiltered", "filtered"), nargs="+")
    parser.add_argument("--backup-postprocess", choices=("released", "arm_clip"))
    parser.add_argument("--episode-length", type=float)
    parser.add_argument("--policy-seed-offset", type=int)
    parser.add_argument("--continuation-policy-seed-offset", type=int)
    parser.add_argument("--continuation-at-step", type=int, default=300)
    parser.add_argument("--uncertainty-threshold", type=float)
    parser.add_argument("--reachability-threshold", type=float)
    parser.add_argument(
        "--diagnostic-probe",
        choices=("none", "no_filter", "arm_clip", "modal_next", "hold_gripper"),
        default="none",
    )
    parser.add_argument("--probe-seed", type=int, default=1)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    for key in (
        "seeds",
        "modes",
        "backup_postprocess",
        "policy_seed_offset",
        "uncertainty_threshold",
        "reachability_threshold",
    ):
        if (value := getattr(args, key)) is not None:
            settings[key] = value
    if args.episode_length is not None:
        settings["episode_length_s"] = args.episode_length
    settings["continuation_policy_seed_offset"] = args.continuation_policy_seed_offset
    settings["continuation_at_step"] = args.continuation_at_step
    settings["diagnostic_probe"] = args.diagnostic_probe
    settings["probe_seed"] = args.probe_seed
    output = args.output or ROOT / "runs/unisafe" / datetime.now(UTC).strftime(
        "baseline-%Y%m%dT%H%M%SZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    started = time.perf_counter()
    script_hash = sha256(__file__)
    runtime = prepare_source()
    save_implementation(output)
    sys.path.insert(0, str(runtime))
    from isaaclab.app import AppLauncher

    launcher = AppLauncher({"headless": True, "enable_cameras": True})
    exit_code = 1
    try:
        import random

        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        from takeoff.config.franka.takeoff_ik_rel_env_cfg import FrankaTakeoffEnvCfg

        from helpers.unisafe_controller import FrozenController, model_observation
        from helpers.unisafe_environment import make_environment
        from helpers.unisafe_model import load_frozen

        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg = FrankaTakeoffEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = "cuda:0"
        cfg.seed = settings["seeds"][0]
        cfg.episode_length_s = settings["episode_length_s"]
        cfg.compute_final_obs = True
        env = make_environment(cfg)
        agent, safety, _, provenance = load_frozen(filter_step=settings["filter_step"])
        controller = FrozenController(agent, safety, settings)
        torch.cuda.reset_peak_memory_stats()
        episodes, paired_states, pairs = [], {}, []
        simulator_steps = 0
        np.set_printoptions(precision=6)

        def physical_state():
            result = {
                name: env.scene[name].data.root_link_state_w.torch.detach().cpu().numpy().copy()
                for name in ("base", "cube_1", "cube_2", "cube_3")
            }
            robot = env.scene["robot"]
            result["joint_pos"] = robot.data.joint_pos.torch.detach().cpu().numpy().copy()
            result["joint_vel"] = robot.data.joint_vel.torch.detach().cpu().numpy().copy()
            return result

        def physical_observation(obs):
            cubes = obs["policy"]["cube_positions"][0].reshape(3, 3)
            eef = obs["policy"]["eef_pos"][0]
            return {
                "eef_pos": eef.tolist(),
                "eef_quat_wxyz": obs["policy"]["eef_quat"][0].tolist(),
                "gripper_pos": obs["policy"]["gripper_pos"][0].tolist(),
                "cube_positions": cubes.tolist(),
                "cube_quaternions_wxyz": obs["policy"]["cube_orientations"][0].tolist(),
                "middle_base_xy_distance_m": (cubes[1, :2] - cubes[0, :2]).norm().item(),
                "top_base_xy_distance_m": (cubes[2, :2] - cubes[0, :2]).norm().item(),
                "top_base_z_distance_m": abs((cubes[2, 2] - cubes[0, 2]).item()),
                "eef_middle_distance_m": (eef - cubes[1]).norm().item(),
                "physical_failure": bool(obs["subtask_terms"]["failure"].item()),
                "physical_success": bool(obs["subtask_terms"]["success"].item()),
            }

        def frame(obs, text):
            panel = Image.new("RGB", (256, 152), "white")
            draw = ImageDraw.Draw(panel)
            draw.text((3, 4), text, fill="black")
            for i, key in enumerate(("front_cam", "wrist_cam")):
                panel.paste(
                    Image.fromarray(obs["rgb_camera"][key][0].cpu().numpy().astype(np.uint8)),
                    (i * 128, 24),
                )
            return panel

        for seed in settings["seeds"]:
            # Alternate arm order across seeds to reduce systematic order effects.
            modes = settings["modes"] if seed % 2 == 0 else list(reversed(settings["modes"]))
            for mode in modes:
                episode_start = time.perf_counter()
                destination = output / f"seed-{seed:03d}-{mode}"
                destination.mkdir()
                obs, _ = env.reset(seed=seed)
                initial = physical_state()
                np.savez_compressed(destination / "initial_state.npz", **initial)
                initial_images = {k: v.cpu().numpy().copy() for k, v in obs["rgb_camera"].items()}
                if seed in paired_states:
                    reference, reference_images = paired_states[seed]
                    error = max(float(np.max(np.abs(initial[k] - reference[k]))) for k in initial)
                    assert error < 1e-6, (
                        f"Mismatched initial physical states for seed {seed}: {error}"
                    )
                    pairs.append(
                        {
                            "seed": seed,
                            "max_initial_state_error": error,
                            "initial_camera_mae": {
                                key: float(
                                    np.abs(
                                        initial_images[key].astype(float) - reference_images[key]
                                    ).mean()
                                )
                                for key in initial_images
                            },
                        }
                    )
                else:
                    paired_states[seed] = initial, initial_images
                policy_seed = seed + settings["policy_seed_offset"]
                random.seed(policy_seed)
                np.random.seed(policy_seed)
                torch.manual_seed(policy_seed)
                controller.settings = dict(settings)
                if seed != settings["probe_seed"] or mode != "filtered":
                    controller.settings["diagnostic_probe"] = "none"
                controller.reset()
                frames = [frame(obs, f"{mode}, seed {seed}, reset")]
                frames[0].save(destination / "reset.png")
                rows, cameras, state_observations = [], [], []
                for t in range(env.max_episode_length):
                    continuation_seed = None
                    if (
                        settings["continuation_policy_seed_offset"] is not None
                        and t == settings["continuation_at_step"]
                    ):
                        # Change only future stochastic draws. Preserve physics,
                        # observations and both recurrent controller states.
                        continuation_seed = seed + settings["continuation_policy_seed_offset"]
                        random.seed(continuation_seed)
                        np.random.seed(continuation_seed)
                        torch.manual_seed(continuation_seed)
                    current = model_observation(obs, first=t == 0)
                    cameras.append(
                        np.stack(
                            [
                                initial_images[k][0]
                                if t == 0
                                else obs["rgb_camera"][k][0].cpu().numpy().copy()
                                for k in ("front_cam", "wrist_cam")
                            ]
                        )
                    )
                    state_observations.append(physical_observation(obs))
                    torch.cuda.synchronize()
                    inference_start = time.perf_counter()
                    action, row = controller.act(
                        current, mode, diagnostic=t % settings["diagnostic_interval"] == 0
                    )
                    torch.cuda.synchronize()
                    row["inference_seconds"] = time.perf_counter() - inference_start
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    simulator_steps += 1
                    done = bool((terminated | truncated).item())
                    terminal_obs = info["final_obs"] if done else next_obs
                    conditions = {
                        name: bool(env.termination_manager.get_term(name).item())
                        for name in env.termination_manager.active_terms
                    }
                    row.update(
                        step=t,
                        simulation_time_s=(t + 1) * env.step_dt,
                        reward=reward.item(),
                        terminated=bool(terminated.item()),
                        truncated=bool(truncated.item()),
                        termination_terms=conditions,
                        continuation_seed=continuation_seed,
                        before=state_observations[-1],
                        after=physical_observation(terminal_obs),
                    )
                    row.update(controller.prediction_error(model_observation(terminal_obs)))
                    assert np.isfinite(row["reward"])
                    rows.append(row)
                    with (destination / "steps.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    if t % 2 == 0 or done:
                        frames.append(
                            frame(
                                terminal_obs,
                                f"{mode} s{seed} t{t + 1} filter={int(row['intervened'])}",
                            )
                        )
                    obs = next_obs
                    if done:
                        cameras.append(
                            np.stack(
                                [
                                    terminal_obs["rgb_camera"][k][0].cpu().numpy().copy()
                                    for k in ("front_cam", "wrist_cam")
                                ]
                            )
                        )
                        state_observations.append(physical_observation(terminal_obs))
                        break
                else:
                    raise RuntimeError(
                        "Environment did not signal termination at its configured horizon"
                    )
                physical_failure = any(r["after"]["physical_failure"] for r in rows)
                boundary = any(r["termination_terms"]["ee_position"] for r in rows)
                success = rows[-1]["termination_terms"]["success"]
                outcome = (
                    "success"
                    if success and not physical_failure and not boundary
                    else "physical_failure"
                    if physical_failure
                    else "boundary_exit"
                    if boundary
                    else "timeout"
                    if rows[-1]["truncated"]
                    else "other_termination"
                )
                summary = {
                    "seed": seed,
                    "policy_seed": policy_seed,
                    "mode": mode,
                    "outcome": outcome,
                    "success": success,
                    "safe_success": outcome == "success",
                    "physical_failure": physical_failure,
                    "boundary_exit": boundary,
                    "timeout": rows[-1]["truncated"],
                    "steps": len(rows),
                    "simulation_time_s": len(rows) * env.step_dt,
                    "completion_time_s": len(rows) * env.step_dt if outcome == "success" else None,
                    "interventions": sum(r["intervened"] for r in rows),
                    "shadow_interventions": sum(r["would_intervene"] for r in rows),
                    "uncertainty_triggers": sum(r["uncertainty_trigger"] for r in rows),
                    "value_triggers": sum(r["value_trigger"] for r in rows),
                    "final_terms": rows[-1]["termination_terms"],
                    "final_state": rows[-1]["after"],
                    "query_counts": dict(controller.queries),
                    "inference_median_s": float(np.median([r["inference_seconds"] for r in rows])),
                    "elapsed_seconds": time.perf_counter() - episode_start,
                }
                np.savez_compressed(
                    destination / "observations.npz",
                    cameras=np.stack(cameras),
                    eef_pos=np.asarray([o["eef_pos"] for o in state_observations]),
                    eef_quat_wxyz=np.asarray([o["eef_quat_wxyz"] for o in state_observations]),
                )
                frames[0].save(
                    destination / "rollout.gif",
                    save_all=True,
                    append_images=frames[1:],
                    duration=100,
                    loop=0,
                )
                (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                episodes.append(summary)
                print(
                    json.dumps(
                        {
                            k: summary[k]
                            for k in ("seed", "mode", "outcome", "steps", "interventions")
                        }
                    ),
                    flush=True,
                )
                (output / "progress.json").write_text(json.dumps(episodes, indent=2) + "\n")
        summary = {
            "status": "complete",
            "settings": settings,
            "model_provenance": provenance,
            "episodes": episodes,
            "paired_initial_conditions": pairs,
            "environment_steps": simulator_steps,
            "physics_steps": env.sim.get_physics_step_count(),
            "explicit_resets": len(episodes),
            "automatic_resets": len(episodes),
            "render_calls": env.sim.render_generation,
            "training_updates": 0,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "elapsed_seconds": time.perf_counter() - started,
            "platform": {name: version(name) for name in ("isaacsim", "isaaclab", "torch")},
            "gpu": torch.cuda.get_device_name(),
            "script_sha256": script_hash,
            "scope": "Migrated single-environment pilot; independent episodes, not mid-episode counterfactual branches",
            "history": "Both recurrent tracks receive clipped executed actions in normalized coordinates",
            "sampling": "Sampled task posterior, modal filter posterior, sampled hypothetical proposal next state",
            "shadow_filter": "Unfiltered arm computes identical diagnostic queries but does not apply replacement",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (output / "environment_config.json").write_text(
            json.dumps(cfg.to_dict(), indent=2, default=str) + "\n"
        )
        env.close()
        env = None
        exit_code = 0
        print("Results:", output, flush=True)
    except BaseException:
        (output / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        launcher.app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
