"""One fresh simulator process, one fixed task-9 episode, durable per-step evidence."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.unisafe09_outcomes import EpisodeScore
from helpers.unisafe09_protocol import verify_freeze
from helpers.unisafe09_rendering import configure_renderer, validate_cameras
from helpers.unisafe09_source import prepare_source, save_implementation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    freeze = verify_freeze(args.freeze)
    job = json.loads(args.job.read_text())
    if job not in freeze["jobs"]:
        raise ValueError("Job is absent from the frozen schedule")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "worker-start.json").exists():
        raise ValueError("Every invocation needs a fresh attempt directory")
    atomic_json(output / "worker-start.json", {"utc": utc(), "pid": os.getpid(), "job": job})
    atomic_json(output / "job.json", job)
    settings = json.loads((ROOT / "configs/unisafe09/controller.json").read_text())
    assert settings.get("diagnostic_probe", "none") == "none"
    configure_renderer(ROOT)
    runtime = prepare_source()
    save_implementation(output)
    sys.path.insert(0, str(runtime))
    from isaaclab.app import AppLauncher

    started = time.perf_counter()
    launcher = None
    env = controller = shadow = None
    account = {
        "completed_controls": 0,
        "started_controls": 0,
        "phase": "app_startup",
        "physics_counter": None,
        "controller_queries": {},
        "shadow_queries": {},
        "query_bundles_started": 0,
        "shadow_bundles_started": 0,
    }

    def checkpoint(phase):
        account.update(phase=phase, utc=utc(), elapsed_seconds=time.perf_counter() - started)
        if controller is not None:
            account["controller_queries"] = dict(controller.queries)
        if shadow is not None:
            account["shadow_queries"] = dict(shadow.queries)
        if env is not None:
            account["physics_counter"] = int(env.sim.get_physics_step_count())
        atomic_json(output / "accounting.json", account)

    exit_code = 1
    try:
        checkpoint("app_startup")
        launcher = AppLauncher({"headless": True, "enable_cameras": True})
        import numpy as np
        import torch
        from takeoff.config.franka.takeoff_ik_rel_env_cfg import FrankaTakeoffEnvCfg

        from helpers.unisafe09_adapter import Adapter
        from helpers.unisafe09_controller import FrozenController, model_observation
        from helpers.unisafe09_environment import make_environment, physical_arrays
        from helpers.unisafe09_model import load_frozen
        from helpers.unisafe09_protocol import condition

        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg = FrankaTakeoffEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = "cuda:0"
        cfg.seed = job["seed"]
        cfg.episode_length_s = 15.0
        cfg.compute_final_obs = True
        assert cfg.decimation == 5 and cfg.sim.dt == 0.01
        env = make_environment(cfg, continuation=job["continuation"])
        account["physics_after_construction"] = int(env.sim.get_physics_step_count())
        checkpoint("model_loading")
        agent, safety, _, provenance = load_frozen(filter_step=settings["filter_step"])
        adapter = Adapter(job["condition_spec"]) if job["condition_spec"] else None
        controller = FrozenController(agent, safety, settings, adapter)
        if job["condition"] == "native_direct":
            shadow = FrozenController(agent, safety, settings, Adapter(condition("d00_g100")))
        torch.cuda.reset_peak_memory_stats()
        obs, _ = env.reset(seed=job["seed"])
        assert env.max_episode_length == 300 and abs(env.step_dt - 0.05) < 1e-12
        account["physics_after_explicit_reset"] = int(env.sim.get_physics_step_count())
        np.savez_compressed(output / "initial_state.npz", **physical_arrays(env))
        random.seed(job["policy_seed"])
        np.random.seed(job["policy_seed"])
        torch.manual_seed(job["policy_seed"])
        controller.reset()
        if shadow:
            shadow.reset()

        def state(observation):
            c = observation["policy"]["cube_positions"][0].reshape(3, 3)
            e = observation["policy"]["eef_pos"][0]
            terms = {}
            for label, upper, lower in (
                ("top_on_base", 2, 0),
                ("middle_on_base", 1, 0),
                ("top_on_middle", 2, 1),
            ):
                diff = c[upper] - c[lower]
                height = diff[2:].norm()
                terms[label] = bool(
                    (diff[:2].norm() < 0.05) & ((height - 0.03).abs() < 0.01) & (height > 0.014)
                )
            terms["middle_not_dropped"] = bool(c[1, 2] >= 0.13)
            terms["middle_separated"] = bool((c[1, :2] - c[0, :2]).norm() > 0.07)
            reduced = (
                terms["top_on_base"]
                and not terms["middle_on_base"]
                and not terms["top_on_middle"]
                and terms["middle_not_dropped"]
            )
            full = reduced and terms["middle_separated"]
            observed = bool(observation["subtask_terms"]["success"].item())
            if observed != full:
                raise RuntimeError("Independent geometry disagrees with the released success term")
            return {
                "eef_pos": e.tolist(),
                "eef_quat_wxyz": observation["policy"]["eef_quat"][0].tolist(),
                "gripper_pos": observation["policy"]["gripper_pos"][0].tolist(),
                "cube_positions": c.tolist(),
                "cube_quaternions_wxyz": observation["policy"]["cube_orientations"][0].tolist(),
                "middle_base_xy_distance_m": (c[1, :2] - c[0, :2]).norm().item(),
                "top_base_xy_distance_m": (c[2, :2] - c[0, :2]).norm().item(),
                "top_base_z_distance_m": abs((c[2, 2] - c[0, 2]).item()),
                "eef_middle_distance_m": (e - c[1]).norm().item(),
                "physical_failure": bool(observation["subtask_terms"]["failure"].item()),
                "physical_success": observed,
                "success_geometry": terms,
                "reduced_predicate": reduced,
            }

        def cameras(observation):
            result = np.stack(
                [
                    observation["rgb_camera"][k][0].cpu().numpy().copy()
                    for k in ("front_cam", "wrist_cam")
                ]
            )
            validate_cameras(result)
            return result

        score = EpisodeScore(job["continuation"])
        rows = []
        (output / "cameras").mkdir()
        np.savez_compressed(output / "reset_cameras.npz", cameras=cameras(obs))
        checkpoint("ready")
        startup_seconds = time.perf_counter() - started
        with (output / "steps.jsonl").open("w", buffering=1) as stream:
            for t in range(340 if job["continuation"] else 300):
                before = state(obs)
                image = cameras(obs)
                np.savez_compressed(output / "cameras" / f"{t:04d}.npz", cameras=image)
                model_obs = model_observation(obs, first=t == 0)
                allowed = {"gripper_pos": obs["policy"]["gripper_pos"]}
                if shadow:
                    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                torch.cuda.synchronize()
                tick = time.perf_counter()
                account["query_bundles_started"] += 1
                checkpoint("controller_started")
                action, row = controller.act(model_obs, "filtered", adapter_observation=allowed)
                torch.cuda.synchronize()
                row["inference_seconds"] = time.perf_counter() - tick
                if shadow:
                    after_cpu, after_gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
                    torch.set_rng_state(cpu_rng)
                    torch.cuda.set_rng_state(gpu_rng)
                    account["shadow_bundles_started"] += 1
                    checkpoint("shadow_started")
                    identical, identity_row = shadow.act(
                        model_obs, "filtered", adapter_observation=allowed
                    )
                    checks = [
                        torch.equal(action, identical),
                        controller.queries == shadow.queries,
                        torch.equal(after_cpu, torch.get_rng_state()),
                        torch.equal(after_gpu, torch.cuda.get_rng_state()),
                    ]
                    for key in (
                        "task_proposed_action",
                        "proposed_action",
                        "filtered_action",
                        "executed_normalized_action",
                        "uncertainty",
                        "next_q1",
                        "next_q2",
                    ):
                        checks.append(row[key] == identity_row[key])
                    torch.set_rng_state(after_cpu)
                    torch.cuda.set_rng_state(after_gpu)
                    row["recorded_input_identity_passed"] = all(checks)
                    if not all(checks):
                        raise RuntimeError("No-op controller changes action, RNG or query schedule")
                account["started_controls"] += 1
                checkpoint("physics_started")
                next_obs, reward, terminated, truncated, info = env.step(action)
                account["completed_controls"] += 1
                done = bool((terminated | truncated).item())
                if done and env.terminal_snapshot is None:
                    raise RuntimeError("Terminal observation was not captured")
                observed = info["final_obs"] if done else next_obs
                terms = (
                    env.terminal_terms
                    if done
                    else {
                        name: bool(env.termination_manager.get_term(name).item())
                        for name in env.termination_manager.active_terms
                    }
                )
                after = state(observed)
                physical_failure = (
                    after["physical_failure"]
                    or terms["cube_2_dropping"]
                    or terms["cube_3_dropping"]
                )
                if terms["success"] != after["physical_success"]:
                    raise RuntimeError("Termination and observation success disagree")
                finished = score.advance(
                    success=terms["success"],
                    physical_failure=physical_failure,
                    boundary=terms["ee_position"],
                )
                if done and score.benchmark is None:
                    raise RuntimeError("Premature timeout or an unscored termination")
                row.update(
                    step=t,
                    simulation_time_s=(t + 1) * env.step_dt,
                    reward=reward.item(),
                    terminated=bool(terminated.item()),
                    truncated=bool(truncated.item()),
                    termination_terms=terms,
                    before=before,
                    after=after,
                    camera_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    episode_length=int(env.episode_length_buf.item()),
                    suppressed_resets=env.suppressed_resets,
                    automatic_resets=env.automatic_resets,
                    query_counts=dict(controller.queries),
                )
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                checkpoint("control_completed")
                if score.benchmark and not finished and env.automatic_resets:
                    raise RuntimeError("Continuation suffered a premature automatic reset")
                obs = next_obs
                if finished:
                    np.savez_compressed(output / "terminal_cameras.npz", cameras=cameras(observed))
                    arrays = env.terminal_arrays if done else physical_arrays(env)
                    np.savez_compressed(output / "terminal_state.npz", **arrays)
                    break
            else:
                raise RuntimeError("Scoring did not finish within its declared maximum controls")
        measured = int(env.sim.get_physics_step_count())
        rollout_physics = measured - account["physics_after_explicit_reset"]
        if rollout_physics != 5 * score.controls:
            raise RuntimeError("Physics counter disagrees with explicit controls")
        result = {
            "status": "complete",
            "job": job,
            "condition": job["condition"],
            "seed": job["seed"],
            "policy_seed": job["policy_seed"],
            **score.summary(),
            "interventions": sum(r["intervened"] for r in rows),
            "gripper_override_steps": sum(r.get("adapter_gripper_override", False) for r in rows),
            "translation_active_steps": sum(
                r.get("adapter_translation_active", False) for r in rows
            ),
            "grasp_evidence_steps": sum(r.get("adapter_grasp_evidence", False) for r in rows),
            "native_closing_proposals": sum(r["task_proposed_action"][-1] < 0 for r in rows),
            "eef_path_length_m": sum(
                float(np.linalg.norm(np.array(r["after"]["eef_pos"]) - r["before"]["eef_pos"]))
                for r in rows
            ),
            "max_middle_displacement_m": max(
                float(
                    np.linalg.norm(
                        np.array(r["after"]["cube_positions"][1])
                        - rows[0]["before"]["cube_positions"][1]
                    )
                )
                for r in rows
            ),
            "query_counts": dict(controller.queries),
            "shadow_query_counts": dict(shadow.queries) if shadow else {},
            "identity_checks": len(rows) if shadow else 0,
            "physics_steps": measured,
            "rollout_physics_steps": rollout_physics,
            "reset_internal_physics_steps": account["physics_after_explicit_reset"],
            "explicit_resets": 1,
            "automatic_resets": env.automatic_resets,
            "suppressed_resets": env.suppressed_resets,
            "startup_seconds": startup_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "settings": settings,
            "model_provenance": provenance,
            "training_updates": 0,
            "platform": {k: version(k) for k in ("torch", "isaacsim", "isaaclab")},
            "gpu": torch.cuda.get_device_name(),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "final_state": rows[-1]["after"],
            "final_terms": rows[-1]["termination_terms"],
        }
        atomic_json(output / "summary.json", result)
        checkpoint("complete")
        print(
            json.dumps(
                {
                    "job": job["id"],
                    "outcome": result["outcome"],
                    "controls": score.controls,
                    "elapsed_seconds": result["elapsed_seconds"],
                }
            ),
            flush=True,
        )
        exit_code = 0
    except BaseException:
        (output / "error.txt").write_text(traceback.format_exc())
        checkpoint("invalid")
        raise
    finally:
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
