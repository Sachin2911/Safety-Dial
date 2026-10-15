"""Validate migrated UNISafe reset, cameras, controls and outcome logic.

Controlled state injections test outcome wiring, not autonomous task competence.
No learned controller is run and no training occurs.
"""

import argparse
import json
import sys
import time
import traceback
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe_source import ROOT, prepare_source, save_implementation, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = args.output or ROOT / "runs/unisafe" / datetime.now(UTC).strftime(
        "environment-%Y%m%dT%H%M%SZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    runtime = prepare_source()
    save_implementation(output)
    sys.path.insert(0, str(runtime))
    from isaaclab.app import AppLauncher

    launcher = AppLauncher({"headless": True, "enable_cameras": True})
    app = launcher.app
    exit_code = 1
    try:
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        from takeoff.config.franka.takeoff_ik_rel_env_cfg import FrankaTakeoffEnvCfg

        from helpers.unisafe_environment import make_environment, normalized_to_environment

        cfg = FrankaTakeoffEnvCfg()
        cfg.scene.num_envs = 1
        cfg.seed = args.seed
        cfg.sim.device = "cuda:0"
        cfg.compute_final_obs = True  # Preserve actual terminal observations for the test log.
        env = make_environment(cfg)
        init_physics_calls = env.sim.get_physics_step_count()
        checks, trace, frames = {}, [], []
        reset_calls = 0
        automatic_resets = 0

        def state():
            values = {
                name: env.scene[name].data.root_link_state_w.torch.clone()
                for name in ("base", "cube_1", "cube_2", "cube_3")
            }
            values["joint_pos"] = env.scene["robot"].data.joint_pos.torch.clone()
            values["eef_pos"] = env.scene["ee_frame"].data.target_pos_source.torch[:, 0].clone()
            return values

        def reset():
            nonlocal reset_calls
            reset_calls += 1
            obs, _ = env.reset(seed=args.seed)
            return obs

        def camera_panel(obs, caption):
            panel = Image.new("RGB", (256, 152), "white")
            draw = ImageDraw.Draw(panel)
            draw.text((3, 5), caption, fill="black")
            for i, name in enumerate(("front_cam", "wrist_cam")):
                pixels = obs["rgb_camera"][name][0].detach().cpu().numpy()
                assert pixels.shape == (128, 128, 3), (name, pixels.shape)
                assert np.isfinite(pixels).all() and 0 <= pixels.min() <= pixels.max() <= 255
                assert pixels.std() > 2, f"Blank camera: {name}"
                panel.paste(Image.fromarray(pixels.astype(np.uint8)), (i * 128, 24))
            return panel

        def terms():
            return {
                name: bool(env.termination_manager.get_term(name).item())
                for name in env.termination_manager.active_terms
            }

        def step(normalized, phase, allow_done=False):
            nonlocal automatic_resets
            action = torch.tensor([normalized], device=env.device, dtype=torch.float32)
            executed = normalized_to_environment(action)
            obs, reward, terminated, truncated, info = env.step(executed)
            done = bool((terminated | truncated).item())
            automatic_resets += int(done)
            terminal_obs = info["final_obs"] if done else obs
            if done:
                for name in ("front_cam", "wrist_cam"):
                    assert (
                        terminal_obs["rgb_camera"][name].data_ptr()
                        != obs["rgb_camera"][name].data_ptr()
                    ), "Terminal camera aliases the reset buffer"
                terminal_positions = terminal_obs["policy"]["cube_positions"].clone()
                env.observation_manager.compute()
                torch.testing.assert_close(
                    terminal_obs["policy"]["cube_positions"], terminal_positions
                )
            event = {
                "step": len(trace),
                "phase": phase,
                "normalized_action": action[0].tolist(),
                "environment_action": executed[0].tolist(),
                "eef_pos": terminal_obs["policy"]["eef_pos"][0].tolist(),
                "eef_quat_wxyz": terminal_obs["policy"]["eef_quat"][0].tolist(),
                "cube_positions": terminal_obs["policy"]["cube_positions"][0].tolist(),
                "gripper_pos": terminal_obs["policy"]["gripper_pos"][0].tolist(),
                "reward": reward.item(),
                "terminated": bool(terminated.item()),
                "truncated": bool(truncated.item()),
                "terms": terms(),
            }
            assert np.isfinite(event["eef_pos"]).all() and np.isfinite(event["reward"])
            trace.append(event)
            # Flush evidence even if a later assertion fails.
            with (output / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            frames.append(camera_panel(terminal_obs, f"{phase}, step {event['step']}"))
            assert allow_done or not done, event
            return obs, event

        obs = reset()
        first = state()
        np.savez_compressed(
            output / "reset_state.npz",
            **{name: value.cpu().numpy() for name, value in first.items()},
        )
        # Detect the stale wrist startup frame seen during migration: both reset
        # views should contain the blue support block at this task's initial pose.
        for name, value in obs["rgb_camera"].items():
            rgb = value.float()
            blue = (rgb[..., 2] > rgb[..., 0] + 30) & (rgb[..., 2] > rgb[..., 1] + 30)
            assert blue.sum().item() > 20, f"Support block absent in reset view: {name}"
        camera_panel(obs, "Reset: front | wrist").save(output / "reset.png")
        camera_stats = {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "std": value.float().std().item(),
            }
            for name, value in obs["rgb_camera"].items()
        }
        # The model-facing orientation must remain wxyz despite the simulator's xyzw API.
        actual = env.scene["ee_frame"].data.target_quat_source.torch[:, 0]
        torch.testing.assert_close(obs["policy"]["eef_quat"], actual[:, [3, 0, 1, 2]])
        reset()
        second = state()
        reset_error = max((first[k] - second[k]).abs().max().item() for k in first)
        assert reset_error < 1e-6, reset_error
        checks["seeded_reset"] = {"max_absolute_state_error": reset_error}
        checks["cameras"] = camera_stats
        checks["reset_support_block_visible"] = True
        checks["quaternion_observation_order"] = "wxyz verified against simulator xyzw"
        print("UNISAFE_RESET_AND_CAMERAS_PASSED", flush=True)

        opened = [0, 0, 0, 0, 0, 0, 1]
        for _ in range(10):
            obs, _ = step(opened, "hold")
        before_arm = state()["eef_pos"]
        for _ in range(5):
            obs, _ = step([0, 0, 0.1, 0, 0, 0, 1], "arm_up")
        after_arm = state()["eef_pos"]
        arm_dz = (after_arm - before_arm)[0, 2].item()
        assert arm_dz > 0.005, arm_dz
        # Inspect the actual IK-scaled command as well as the observed physical displacement.
        processed = env.action_manager.get_term("arm_action").processed_actions
        torch.testing.assert_close(
            processed, torch.tensor([[0, 0, 0.0075, 0, 0, 0]], device=env.device)
        )
        checks["arm_command"] = {"measured_dz_m": arm_dz, "ik_command_dz_m": 0.0075}
        before_close = env.scene["robot"].data.joint_pos.torch[0, -2:].clone()
        for _ in range(20):
            obs, _ = step([0, 0, 0, 0, 0, 0, -1], "gripper_close")
        closed = env.scene["robot"].data.joint_pos.torch[0, -2:].clone()
        for _ in range(20):
            obs, _ = step(opened, "gripper_open")
        reopened = env.scene["robot"].data.joint_pos.torch[0, -2:].clone()
        assert (before_close - closed).min() > 0.02, (before_close, closed)
        assert (reopened - closed).min() > 0.02, (reopened, closed)
        checks["gripper"] = {
            "initial": before_close.tolist(),
            "closed": closed.tolist(),
            "reopened": reopened.tolist(),
            "units": "m per finger",
        }
        print("UNISAFE_CONTROLS_PASSED", flush=True)

        def inject(name, position):
            asset = env.scene[name]
            pose = torch.tensor([[*position, 0, 0, 0, 1]], device=env.device)
            asset.write_root_pose_to_sim_index(root_pose=pose)
            asset.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), device=env.device)
            )
            env.scene.update(0.0)

        # Each scenario begins from the same seeded reset. These are deliberately
        # injected test fixtures, never counted as policy successes or failures.
        outcome_checks = {}
        for scenario in ("cube_2_drop", "cube_3_drop", "cube_3_displaced", "success", "timeout"):
            reset()
            base_pos = env.scene["cube_1"].data.root_link_pos_w.torch[0].tolist()
            if scenario in ("cube_2_drop", "cube_3_drop"):
                name = scenario.removesuffix("_drop")
                inject(name, [0.68, 0.18, 0.12])
            elif scenario == "cube_3_displaced":
                inject("cube_3", [base_pos[0] + 0.15, base_pos[1], 0.20])
            elif scenario == "success":
                inject("cube_2", [base_pos[0] + 0.12, base_pos[1], 0.20])
                inject("cube_3", [base_pos[0], base_pos[1], base_pos[2] + 0.03])
            else:
                env.episode_length_buf[:] = env.max_episode_length - 1
            env.termination_manager.compute()
            immediate = terms()
            if scenario == "success":
                assert immediate["success"] and not immediate["failure"], immediate
            elif scenario == "timeout":
                assert not immediate["time_out"], immediate
            else:
                assert immediate["failure"], immediate
            _, event = step(opened, f"injected_{scenario}", allow_done=True)
            if scenario == "timeout":
                assert event["truncated"] and not event["terminated"], event
                assert event["terms"]["time_out"], event
            elif scenario == "success":
                assert event["terminated"] and event["terms"]["success"], event
                assert not event["terms"]["failure"], event
            else:
                assert event["terminated"] and event["terms"]["failure"], event
            positions = np.asarray(event["cube_positions"]).reshape(3, 3)
            if scenario in ("cube_2_drop", "cube_3_drop"):
                index = 1 if scenario == "cube_2_drop" else 2
                assert positions[index, 2] < 0.13, positions
            elif scenario == "cube_3_displaced":
                assert np.linalg.norm(positions[2, :2] - positions[0, :2]) > 0.10, positions
            elif scenario == "success":
                assert np.linalg.norm(positions[1, :2] - positions[0, :2]) > 0.07, positions
            outcome_checks[scenario] = {"before_step": immediate, "after_step": event["terms"]}
        checks["injected_outcome_tests"] = outcome_checks
        checks["terminal_buffers_preserved_across_auto_reset"] = True
        frames[0].save(
            output / "controls_and_outcomes.gif",
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0,
        )
        print("UNISAFE_OUTCOME_TESTS_PASSED", flush=True)

        scene_cfg = cfg.scene.to_dict()
        config_snapshot = {
            "scene": scene_cfg,
            "actions": cfg.actions.to_dict(),
            "events": cfg.events.to_dict(),
            "simulation": cfg.sim.to_dict(),
            "decimation": cfg.decimation,
            "episode_length_s": cfg.episode_length_s,
            "max_episode_length": env.max_episode_length,
            "observations": cfg.observations.to_dict(),
            "terminations": cfg.terminations.to_dict(),
        }
        (output / "environment_config.json").write_text(
            json.dumps(config_snapshot, indent=2, default=str) + "\n"
        )
        summary = {
            "status": "passed",
            "seed": args.seed,
            "checks": checks,
            "source_revision": json.loads((runtime / "compatibility.json").read_text())["revision"],
            "compatibility_sha256": sha256(runtime / "compatibility.json"),
            "script_sha256": sha256(__file__),
            "adapter_sha256": sha256(ROOT / "experiments/helpers/unisafe_environment.py"),
            "platform": {name: version(name) for name in ("isaacsim", "isaaclab", "torch")},
            "gpu": torch.cuda.get_device_name(),
            "environment_steps": len(trace),
            "physics_steps": env.sim.get_physics_step_count(),
            "initialization_physics_steps": init_physics_calls,
            "render_calls": env.sim.render_generation,
            "injected_object_poses": 5,
            "injected_episode_counters": 1,
            "explicit_resets": reset_calls,
            "automatic_resets": automatic_resets,
            "training_updates": 0,
            "learned_model_queries": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "scope": "one-env engineering acceptance; injected outcomes are not autonomous performance",
            "camera_pose_metadata": "not consumed; native camera pose metadata is not validated on this renderer path",
            "replication_limit": "Isaac Sim 6.2 / Isaac Lab 3 migration, not validated against original Isaac Sim 4.2 physics or pixels",
            "action_transform": "clip(normalized * 0.15, -0.15, 0.15), then IK scale 0.5; negative gripper closes",
            "physics_accounting": "SimulationContext physics step counter includes initialization; resets and state injections reported separately",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("Results:", output, flush=True)
        env.close()
        # Release sensor-backed tensor views before Kit unloads its plugins.
        del obs, actual, processed, first, second
        env = None
        import gc

        gc.collect()
        exit_code = 0
    except BaseException as exc:
        (output / "error.txt").write_text(traceback.format_exc())
        print(f"UNISAFE_ENVIRONMENT_FAILED: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
