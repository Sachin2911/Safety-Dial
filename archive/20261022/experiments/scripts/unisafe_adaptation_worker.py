"""Evaluate a declared sequence of task adapters under one frozen UNISafe filter."""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import Counter
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe_source import ROOT, prepare_source, save_implementation, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    settings = json.loads((ROOT / "configs/unisafe/baseline.json").read_text())
    settings.update(plan.get("controller_settings", {}))
    if settings.get("diagnostic_probe", "none") != "none":
        raise ValueError("Physical diagnostic overrides must be off for adaptation experiments")
    settings["diagnostic_interval"] = 0
    runtime = prepare_source()
    save_implementation(args.output)
    sys.path.insert(0, str(runtime))
    from isaaclab.app import AppLauncher

    script_hash = sha256(__file__)
    started = time.perf_counter()
    launcher = AppLauncher({"headless": True, "enable_cameras": True})
    exit_code = 1
    try:
        import random

        import numpy as np
        import torch
        from takeoff.config.franka.takeoff_ik_rel_env_cfg import FrankaTakeoffEnvCfg

        from helpers.unisafe_adaptation import GraspReleaseDelay, ReleaseDelay
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
        cfg.seed = plan["episodes"][0]["seed"]
        cfg.episode_length_s = settings["episode_length_s"]
        cfg.compute_final_obs = True
        env = make_environment(cfg)
        agent, safety, _, provenance = load_frozen(filter_step=settings["filter_step"])
        controller = FrozenController(agent, safety, settings)
        torch.cuda.reset_peak_memory_stats()
        episodes, queries, total_steps = [], Counter(), 0

        def state(obs):
            c = obs["policy"]["cube_positions"][0].reshape(3, 3)
            e = obs["policy"]["eef_pos"][0]
            return {
                "eef_pos": e.tolist(),
                "eef_quat_wxyz": obs["policy"]["eef_quat"][0].tolist(),
                "gripper_pos": obs["policy"]["gripper_pos"][0].tolist(),
                "cube_positions": c.tolist(),
                "cube_quaternions_wxyz": obs["policy"]["cube_orientations"][0].tolist(),
                "middle_base_xy_distance_m": (c[1, :2] - c[0, :2]).norm().item(),
                "top_base_xy_distance_m": (c[2, :2] - c[0, :2]).norm().item(),
                "top_base_z_distance_m": abs((c[2, 2] - c[0, 2]).item()),
                "eef_middle_distance_m": (e - c[1]).norm().item(),
                "physical_failure": bool(obs["subtask_terms"]["failure"].item()),
                "physical_success": bool(obs["subtask_terms"]["success"].item()),
            }

        def cameras(obs):
            return np.stack([
                obs["rgb_camera"][k][0].cpu().numpy().copy()
                for k in ("front_cam", "wrist_cam")
            ])

        for index, job in enumerate(plan["episodes"]):
            begin = time.perf_counter()
            destination = args.output / f"episode-{index:04d}"
            destination.mkdir()
            seed, mode = job["seed"], job.get("mode", "filtered")
            if mode not in ("filtered", "unfiltered"):
                raise ValueError(mode)
            adapter_type = job.get("adapter_kind", plan.get("adapter_kind", "release_delay"))
            adapter_class = {"release_delay": ReleaseDelay, "grasp_release_delay": GraspReleaseDelay}[adapter_type]
            controller.proposal_adapter = adapter_class(**job.get("adapter", {}))
            obs, _ = env.reset(seed=seed)
            initial = {
                k: env.scene[k].data.root_link_state_w.torch.cpu().numpy().copy()
                for k in ("base", "cube_1", "cube_2", "cube_3")
            }
            initial.update({
                k: getattr(env.scene["robot"].data, k).torch.cpu().numpy().copy()
                for k in ("joint_pos", "joint_vel")
            })
            np.savez_compressed(destination / "initial_state.npz", **initial)
            first_cameras = cameras(obs)
            np.savez_compressed(destination / "reset_cameras.npz", cameras=first_cameras)
            policy_seed = job.get("policy_seed", seed)
            random.seed(policy_seed)
            np.random.seed(policy_seed)
            torch.manual_seed(policy_seed)
            controller.reset()
            rows, saved_cameras, poses = [], [], []
            record = job.get("record_cameras", plan.get("record_cameras", False))
            with (destination / "steps.jsonl").open("w") as stream:
                for t in range(env.max_episode_length):
                    before = state(obs)
                    image = first_cameras if t == 0 else cameras(obs)
                    if record:
                        saved_cameras.append(image)
                    poses.append(before)
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    action, row = controller.act(
                        model_observation(obs, first=t == 0), mode,
                        adapter_observation={"gripper_pos": obs["policy"]["gripper_pos"]},
                    )
                    torch.cuda.synchronize()
                    row["inference_seconds"] = time.perf_counter() - tick
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    total_steps += 1
                    done = bool((terminated | truncated).item())
                    terminal = info["final_obs"] if done else next_obs
                    terms = {
                        name: bool(env.termination_manager.get_term(name).item())
                        for name in env.termination_manager.active_terms
                    }
                    row.update(
                        step=t, simulation_time_s=(t + 1) * env.step_dt,
                        reward=reward.item(), terminated=bool(terminated.item()),
                        truncated=bool(truncated.item()), termination_terms=terms,
                        before=before, after=state(terminal),
                        camera_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    )
                    stream.write(json.dumps(row) + "\n")
                    rows.append(row)
                    obs = next_obs
                    if done:
                        last_cameras = cameras(terminal)
                        if record:
                            saved_cameras.append(last_cameras)
                        poses.append(row["after"])
                        break
                else:
                    raise RuntimeError("Episode did not terminate within the configured horizon")
            np.savez_compressed(destination / "terminal_cameras.npz", cameras=last_cameras)
            if record:
                np.savez_compressed(
                    destination / "observations.npz", cameras=np.stack(saved_cameras),
                    eef_pos=np.asarray([s["eef_pos"] for s in poses]),
                    eef_quat_wxyz=np.asarray([s["eef_quat_wxyz"] for s in poses]),
                )
            failed = any(r["after"]["physical_failure"] for r in rows)
            boundary = any(r["termination_terms"]["ee_position"] for r in rows)
            success = rows[-1]["termination_terms"]["success"]
            outcome = (
                "physical_failure" if failed else "boundary_exit" if boundary
                else "success" if success else "timeout" if rows[-1]["truncated"]
                else "other_termination"
            )
            utility = (
                1.0 - 0.02 * len(rows) / env.max_episode_length if outcome == "success"
                else -1.0 if failed or boundary else 0.0
            )
            summary = {
                "index": index, "job": job, "seed": seed, "policy_seed": policy_seed,
                "mode": mode, "outcome": outcome, "success": success,
                "safe_success": outcome == "success", "physical_failure": failed,
                "boundary_exit": boundary, "timeout": rows[-1]["truncated"],
                "steps": len(rows), "simulation_time_s": len(rows) * env.step_dt,
                "completion_time_s": len(rows) * env.step_dt if outcome == "success" else None,
                "interventions": sum(r["intervened"] for r in rows),
                "adapter_override_steps": sum(r["adapter_gripper_override"] for r in rows),
                "adapter_events": controller.proposal_adapter.events,
                "utility": utility, "reward_sum": sum(r["reward"] for r in rows),
                "query_counts": dict(controller.queries), "final_terms": rows[-1]["termination_terms"],
                "final_state": rows[-1]["after"],
                "elapsed_seconds": time.perf_counter() - begin,
            }
            queries.update(controller.queries)
            (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            episodes.append(summary)
            (args.output / "progress.json").write_text(json.dumps(episodes, indent=2) + "\n")
            print(json.dumps({k: summary[k] for k in (
                "index", "job", "outcome", "steps", "interventions", "utility"
            )}), flush=True)
        result = {
            "status": "complete", "settings": settings, "model_provenance": provenance,
            "episodes": episodes, "environment_steps": total_steps,
            "physics_steps": env.sim.get_physics_step_count(), "query_counts": dict(queries),
            "explicit_resets": len(episodes), "automatic_resets": len(episodes),
            "render_calls": env.sim.render_generation, "training_updates": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "platform": {k: version(k) for k in ("torch", "isaacsim", "isaaclab")},
            "gpu": torch.cuda.get_device_name(), "script_sha256": script_hash,
            "scope": "Declared adapter evaluations; randomized order does not imply identical rendered prefixes",
            "utility": "success: 1 - 0.02*steps/horizon; failure or boundary exit: -1; unfinished: 0",
        }
        assert result["physics_steps"] == total_steps * cfg.decimation
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        env.close()
        exit_code = 0
    except BaseException:
        (args.output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        launcher.app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
