"""Render recorded states on the rented host; never invoke a planner or evaluation step."""

import argparse
import base64
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_io import publish_artifact, report_progress


def read(path):
    return json.loads(Path(path).read_text())


def publish_json(name, value):
    def encode_extra(item):
        if isinstance(item, bytes):
            return {"bytes_base64": base64.b64encode(item).decode("ascii")}
        raise TypeError(f"Unsupported metadata type: {type(item).__name__}")

    publish_artifact(name, data=json.dumps(value, indent=2, allow_nan=False,
                                          default=encode_extra) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(config, task):
    root = Path(config["asset_root"])
    assert str(root).startswith("/workspace/LeWM_Experiment1/")
    sys.path.insert(0, str(root / "media-matplotlib"))
    if task in ("cube", "reacher"):
        sys.path.insert(0, str(root / "runtime-mujoco3.11.0-dmcontrol1.0.44"))


def package(config):
    root = Path(config["asset_root"])
    target = root / "media-matplotlib"
    if not target.exists():
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target),
            "--report", str(Path(os.environ["LEWM_RUN_DIR"]) / ".control/media-pip.json"),
            "matplotlib==3.10.7", "contourpy==1.3.3", "cycler==0.12.1",
            "fonttools==4.60.1", "kiwisolver==1.4.9", "pyparsing==3.2.5",
            "python-dateutil==2.9.0.post0", "six==1.17.0",
        ], check=True, timeout=180)
    pip_report = Path(os.environ["LEWM_RUN_DIR"]) / ".control/media-pip.json"
    if pip_report.exists():
        publish_artifact("artifacts/media-pip.json", source=pip_report)
    for i, task in enumerate(("tworoom", "reacher", "pusht", "cube")):
        reuse = config.get("reuse_completed_tasks", {}).get(task)
        if reuse:
            prior = Path("/workspace/LeWM_Experiment1/runs") / reuse["run"]
            assert sha(prior / "manifest.json") == reuse["manifest_sha256"]
            manifest = read(prior / "manifest.json")
            assert len(read(prior / "artifacts" / task / "audit.json")["examples"]) == 2
            for path in sorted((prior / "artifacts" / task).iterdir()):
                name = path.relative_to(prior).as_posix()
                assert sha(path) == manifest["files"][name]["sha256"]
                publish_artifact(name, source=path)
        else:
            subprocess.run([sys.executable, __file__, "--task", task], check=True, timeout=300)
        report_progress(i + 1, 4, unit="environments rendered")
    reports = [read(Path(os.environ["LEWM_RUN_DIR"]) / "artifacts" / task / "audit.json")
               for task in config["tasks"]]
    assert sum(len(r["examples"]) for r in reports) == 8
    files = list((Path(os.environ["LEWM_RUN_DIR"]) / "artifacts").rglob("*"))
    total = sum(p.stat().st_size for p in files if p.is_file())
    assert total < 100_000_000
    accounting = read(root / "media-accounting-20261104.json")
    assert accounting["environments_reserved"] <= 24
    assert accounting["setup_physics_steps"] <= 2000
    original_audit = read(Path("/workspace/LeWM_Experiment1/runs") /
                          "repro-20261104-final-audit/artifacts/audit.json")
    assert read(root / "interaction-ledger.json") == original_audit["interaction_ledger"]
    publish_json("artifacts/media-audit.json", {
        "complete": True, "videos": 8, "trajectory_images": 8,
        "published_bytes_before_final_index": total, "reports": reports,
        "model_queries": 0, "new_evaluation_actions": 0,
        "selection": config["selection"],
        "reused_task_sources": config.get("reuse_completed_tasks", {}),
        "scientific_results_changed": False,
        "scientific_ledger_unchanged": True,
        "all_attempts_accounting": accounting,
    })
    lines = ["# LeWM reproduction examples", "", config["selection"] + ".", "",
             "These are reconstructed views of the saved post-action states, not original "
             "video recordings. Playback is 10 displayed actions per second, not real time. "
             "Success labels come from the original audited run. Examples are illustrative, "
             "not a representative estimate of performance.", "",
             "| Environment | Outcome | Video | Trajectory |", "| --- | --- | --- | --- |"]
    for task in ("tworoom", "reacher", "pusht", "cube"):
        for outcome in ("success", "failure"):
            stem = f"{task}/{outcome}"
            lines.append(f"| {task} | {outcome} | [MP4]({stem}.mp4) | "
                         f"[PNG]({stem}-trajectory.png) |")
    lines += ["", "PushT block poses are recorded; agent motion is reconstructed from saved "
              "actions and checked against every saved reward and termination label. "
              "Reacher's selected success has no retained goal image; its video says so. "
              "Cube's selected success first met the native condition at action 1. "
              "No evaluation actions or model queries were added. Setup work is counted "
              "separately in media-audit.json."]
    publish_artifact("artifacts/README.md", data="\n".join(lines) + "\n")


def render_task(config, task):
    configure(config, task)
    import gymnasium as gym
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import stable_worldmodel  # noqa: F401  registers environments

    started = time.time()
    specification = config["tasks"][task]
    run = Path("/workspace/LeWM_Experiment1/runs") / specification["run"]
    assert sha(run / "manifest.json") == specification["source_manifest_sha256"]
    panel = specification["panel"]
    cfg = read(run / "artifacts" / panel / "configuration.json")
    summary = read(run / "metrics" / panel / "summary.json")
    episodes = read(run / "metrics" / panel / "episodes.json")
    assert specification["chosen"] == [next(e for e in episodes if e["success"] == outcome)
                                       for outcome in (True, False)]
    trace_files = sorted((run / "metrics" / panel).glob("steps-*.jsonl"))
    traces = [json.loads(line) for p in trace_files for line in p.read_text().splitlines()]
    scratch = Path(os.environ["LEWM_RUN_DIR"]) / ".control" / f"media-{task}"
    scratch.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    budget_file = Path(config["asset_root"]) / "media-accounting-20261104.json"
    ledger = read(budget_file) if budget_file.exists() else {
        "environments_reserved": 0, "setup_physics_steps": 0, "attempts": []}
    assert ledger["environments_reserved"] + 2 <= 24
    ledger["environments_reserved"] += 2
    ledger["attempts"].append({"run": Path(os.environ["LEWM_RUN_DIR"]).name, "task": task})
    budget_file.write_text(json.dumps(ledger, indent=2) + "\n")

    # Observe every physics advance, including construction and reset internals.
    if task in ("cube", "reacher"):
        import mujoco
        original_mj_step = mujoco.mj_step

        def mj_step(*args, **kwargs):
            counts["setup_physics_steps"] += int(kwargs.get("nstep", 1))
            assert ledger["setup_physics_steps"] + counts["setup_physics_steps"] <= 2000
            return original_mj_step(*args, **kwargs)

        mujoco.mj_step = mj_step
    if task == "pusht":
        import pymunk
        original_space_step = pymunk.Space.step

        def space_step(self, dt):
            counts["setup_physics_steps"] += 1
            assert ledger["setup_physics_steps"] + counts["setup_physics_steps"] <= 2000
            return original_space_step(self, dt)

        pymunk.Space.step = space_step

    world = dict(cfg["world"])
    world.pop("num_envs")
    env_name = world.pop("env_name")
    # Patch the class before construction to count reset-internal control calls too.
    entry = gym.spec(env_name).entry_point
    if isinstance(entry, str):
        module, name = entry.split(":")
        cls = getattr(importlib.import_module(module), name)
    else:
        cls = entry
    original_step = cls.step

    def counted_step(self, *args, **kwargs):
        counts["setup_controller_calls"] += 1
        return original_step(self, *args, **kwargs)

    cls.step = counted_step
    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    font = ImageFont.truetype(str(font_path), 22)
    small = ImageFont.truetype(str(font_path), 17)
    outputs = []
    env = None
    try:
        for episode in specification["chosen"]:
            outcome = "success" if episode["success"] else "failure"
            rows = sorted([r for r in traces if r["case"] == episode["case"]],
                          key=lambda r: r["step"])
            assert [r["step"] for r in rows] == list(range(1, 51))
            env = gym.make(env_name, **world)
            base = env.unwrapped
            seed_values = summary["native_metrics"].get("seeds")
            seed = int(seed_values[episode["case"]]) if seed_values is not None else None
            inputs, goal_image = {}, None
            if task != "reacher":
                path = Path(config["asset_root"]) / "datasets" / (
                    specification["dataset"]["dataset_name"] + ".h5")
                with h5py.File(path, "r") as f:
                    start_row = int(f["ep_offset"][episode["episode"]]) + episode["start_step"]
                    goal_row = start_row + 25
                    assert start_row == episode["row"]
                    inputs = {k: np.asarray(f[k][start_row]) for k in f
                              if k not in ("pixels", "ep_len", "ep_offset")}
                    inputs.update({"goal_" + k: np.asarray(f[k][goal_row]) for k in f
                                   if k not in ("pixels", "ep_len", "ep_offset")})
                    goal_image = np.asarray(f["pixels"][goal_row])
            elif episode["case"] == 0:
                goal_image = np.asarray(Image.open(run / "artifacts" / panel / "goal.png"))
            variations = {k.removeprefix("variation."): v for k, v in inputs.items()
                          if k.startswith("variation.")}
            options = {"variation": list(variations), "variation_values": variations}
            counts["resets"] += 1
            base.reset(seed=seed, options=options if variations else {})
            if task != "reacher":
                for call in cfg["original_task_config"]["eval"]["callables"]:
                    args = {k: inputs[v["value"]] if v.get("in_dataset", True) else v["value"]
                            for k, v in call["args"].items()}
                    getattr(base, call["method"])(**args)
            setup_counts = dict(counts)

            def forbidden_step(*args, **kwargs):
                raise AssertionError("No new evaluation action is authorized for media")

            base.step = forbidden_step
            frames, coordinates, agent_coordinates, errors = [], [], [], []
            if task == "pusht":
                # Reconstruct the recorded initialization, independent of the fresh
                # renderer's random reset/contact history. _set_state advances dt once.
                agent_vel = inputs["state"][-2:].astype(float).copy()
                agent_pos = inputs["state"][:2].astype(float) + agent_vel * base.dt
                base.agent.position = tuple(agent_pos)
                base.agent.velocity = tuple(agent_vel)
            for row in rows:
                info = row["info"]
                if task == "tworoom":
                    base._set_state(np.asarray(info["state"]))
                    coordinates.append(info["state"])
                elif task in ("reacher", "cube"):
                    base.set_state(np.asarray(info["qpos"]), np.asarray(info["qvel"]))
                    coordinates.append(info["qpos"] if task == "reacher" else
                                       base._data.joint("object_joint_0").qpos[:2].copy())
                    actual = base.env.physics.data.qpos if task == "reacher" else base._data.qpos
                    assert np.array_equal(actual, np.asarray(info["qpos"]))
                else:
                    # Match native float32 action scaling, Vec2d arithmetic and C-double
                    # body setters. A NumPy float64 recurrence accumulates rounding drift.
                    action = np.asarray(row["action"], dtype=np.float32)
                    target = (base.agent.position + action * base.action_scale
                              if base.relative else action)
                    for _ in range(int(1 / (base.dt * base.control_hz))):
                        acceleration = (base.k_p * (target - base.agent.position)
                                        + base.k_v * (pymunk.Vec2d(0, 0) - base.agent.velocity))
                        base.agent.velocity += acceleration * base.dt
                        base.agent.position += base.agent.velocity * base.dt
                        counts["kinematic_updates"] += 1
                    agent_pos = np.asarray(base.agent.position, dtype=float)
                    base.block.angle = info["block_pose"][2]
                    # Rotation changes the body origin around its centre of gravity.
                    # Restore angle first, then the recorded position, as _set_state does.
                    base.block.position = tuple(info["block_pose"][:2])
                    assert np.allclose(base._get_obs()[2:4], info["block_pose"][:2], atol=1e-10)
                    state = base._get_obs()
                    reached, distance = base.eval_state(base.goal_state, state)
                    errors.append(abs(float(row["reward"]) + float(distance)))
                    assert bool(reached) == row["terminated"]
                    assert errors[-1] < 1e-4, (episode["case"], row["step"], errors[-1])
                    # Pymunk debug drawing reads cached shape geometry. Refresh it
                    # after direct body restoration, without advancing physics.
                    base.space.reindex_shapes_for_body(base.agent)
                    base.space.reindex_shapes_for_body(base.block)
                    coordinates.append(info["block_pose"][:2])
                    agent_coordinates.append(agent_pos.copy())
                frames.append(np.asarray(base.render()).copy())
            assert counts["setup_physics_steps"] == setup_counts.get("setup_physics_steps", 0)
            assert counts["setup_controller_calls"] == setup_counts.get("setup_controller_calls", 0)
            assert all(frame.shape == frames[0].shape for frame in frames)
            images = scratch / outcome
            images.mkdir(exist_ok=True)
            goal_display = Image.fromarray(goal_image) if goal_image is not None else None
            for i, frame in enumerate(frames):
                canvas = Image.new("RGB", (960, 640), "#f4f6f8")
                draw = ImageDraw.Draw(canvas)
                title = f"{task.upper()} | recorded {outcome} | case {episode['case']}"
                draw.text((24, 16), title, font=font, fill="#183747")
                draw.text((24, 50), "Reconstructed states | seed 42 | original main CEM setting",
                          font=small, fill="#415463")
                canvas.paste(Image.fromarray(frame).resize((448, 448)), (24, 110))
                draw.text((24, 84), f"After action {i + 1}/50", font=small, fill="#183747")
                draw.text((488, 84), "Dataset goal image", font=small, fill="#183747")
                if goal_display:
                    canvas.paste(goal_display.resize((448, 448)), (488, 110))
                else:
                    draw.rectangle((488, 110, 936, 558), fill="#e3e9ed")
                    draw.multiline_text((515, 275), "Goal image not retained\nfor this selected case",
                                        font=font, fill="#415463", spacing=9)
                reached = episode["first_success_step"]
                label = (f"Original monitor first succeeded at action {reached}." if reached else
                         "Original monitor did not succeed within 50 actions.")
                draw.text((24, 577), label, font=small, fill="#183747")
                draw.text((24, 607), "10 displayed actions/second; reconstructed, not live footage",
                          font=small, fill="#415463")
                canvas.save(images / f"{i:03d}.png")
            video = scratch / f"{outcome}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "10", "-i",
                            str(images / "%03d.png"), "-c:v", "libx264", "-crf", "20",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)],
                           check=True, timeout=90)
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,nb_read_frames,duration", "-of", "json",
                str(video)], text=True))["streams"][0]
            assert int(probe["nb_read_frames"]) == 50
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
                           check=True, timeout=60)
            xy = np.asarray(coordinates)
            fig = plt.figure(figsize=(12, 8), layout="constrained")
            grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.5])
            for col, index in enumerate((0, 24, 49)):
                ax = fig.add_subplot(grid[0, col])
                ax.imshow(frames[index])
                ax.set_title(f"After action {index + 1}")
                ax.axis("off")
            ax = fig.add_subplot(grid[1, :2])
            ax.plot(xy[:, 0], xy[:, 1], color="#8aa1ac", linewidth=1.4)
            scatter = ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(1, 51), cmap="viridis", s=22)
            ax.scatter(*xy[0], color="#e88b26", s=85, marker="o", label="Action 1")
            ax.scatter(*xy[-1], color="#bc3059", s=90, marker="X", label="Action 50")
            if task == "pusht":
                agents = np.asarray(agent_coordinates)
                ax.plot(agents[:, 0], agents[:, 1], "--", color="#555555", label="Agent")
                target_xy = inputs["goal_state"][2:4]
            elif task == "tworoom":
                target_xy = inputs["goal_proprio"]
            elif task == "cube":
                target_xy = inputs["goal_privileged_block_0_pos"][:2]
            else:
                target_xy = None
            if target_xy is not None:
                ax.scatter(*target_xy, marker="*", s=180, color="#1f8b58", label="Goal position")
            if task in ("tworoom", "pusht"):
                ax.invert_yaxis()
            ax.set_xlabel("Joint 1 (rad)" if task == "reacher" else
                          ("x (m)" if task == "cube" else "x (arena units)"))
            ax.set_ylabel("Joint 2 (rad)" if task == "reacher" else
                          ("y (m)" if task == "cube" else "y (arena units)"))
            ax.set_title("Recorded joint trajectory" if task == "reacher" else
                         ("Recorded block path (XY projection)" if task == "cube" else
                          "Recorded block trajectory" if task == "pusht" else
                          "Recorded agent trajectory"))
            if task == "cube":
                points = np.vstack([xy, target_xy])
                centre = (points.min(axis=0) + points.max(axis=0)) / 2
                span = np.maximum(np.ptp(points, axis=0) * 1.15, 0.1)
                ax.set_xlim(centre[0] - span[0] / 2, centre[0] + span[0] / 2)
                ax.set_ylim(centre[1] - span[1] / 2, centre[1] + span[1] / 2)
                ax.ticklabel_format(useOffset=False, style="plain")
                if np.max(np.ptp(xy, axis=0)) < 1e-6:
                    ax.text(0.03, 0.04, "Cube position unchanged to 1 micrometre",
                            transform=ax.transAxes, fontsize=9)
            ax.legend(loc="best", fontsize=8)
            ax.grid(alpha=0.2)
            fig.colorbar(scatter, ax=ax, label="Action", shrink=0.85)
            ax = fig.add_subplot(grid[1, 2])
            if goal_image is not None:
                ax.imshow(goal_image)
                ax.set_title("Dataset goal image")
            else:
                ax.text(0.5, 0.5, "Goal image not retained\nfor this selected case",
                        ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            fig.suptitle(f"{task.upper()} | recorded {outcome} | primary seed 42, "
                         f"case {episode['case']}\n{label}", fontsize=14)
            picture = scratch / f"{outcome}-trajectory.png"
            fig.savefig(picture, dpi=160)
            plt.close(fig)
            inputs_serial = {k: v.tolist() for k, v in inputs.items()}
            record = {"episode": episode, "source_run": specification["run"],
                      "source_hf_revision": specification["source_revision"],
                      "source_manifest_sha256": specification["source_manifest_sha256"],
                      "trace_hashes": {p.name: sha(p) for p in trace_files},
                      "rows": rows, "restoration_inputs": inputs_serial,
                      "coordinates": xy.tolist(),
                      "agent_coordinates": [x.tolist() for x in agent_coordinates],
                      "frame_range": [1, 50], "goal_image_available": goal_image is not None,
                      "initial_image_available": False if task == "reacher" else True,
                      "pusht_max_reward_error": max(errors, default=0), "video_probe": probe,
                      "render_source_sha256": sha(__file__)}
            prefix = f"artifacts/{task}/{outcome}"
            publish_artifact(prefix + ".mp4", source=video)
            publish_artifact(prefix + "-trajectory.png", source=picture)
            publish_json(prefix + "-reconstruction.json", record)
            if goal_image is not None:
                goal_path = scratch / f"{outcome}-goal.png"
                Image.fromarray(goal_image).save(goal_path)
                publish_artifact(prefix + "-goal.png", source=goal_path)
            outputs.append({k: record[k] for k in (
                "episode", "goal_image_available", "pusht_max_reward_error", "video_probe")})
            env.close()
            env = None
    finally:
        if env is not None:
            env.close()
        ledger["setup_physics_steps"] += counts["setup_physics_steps"]
        ledger["attempts"][-1]["counts"] = dict(counts)
        budget_file.write_text(json.dumps(ledger, indent=2) + "\n")
        publish_json(f"artifacts/{task}/accounting.json", {"counts": dict(counts), "ledger": ledger})
    publish_json(f"artifacts/{task}/audit.json", {
        "task": task, "examples": outputs, "counts": dict(counts),
        "elapsed_seconds": time.time() - started,
        "model_queries": 0, "new_evaluation_actions": 0,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("tworoom", "reacher", "pusht", "cube"))
    args = parser.parse_args()
    config = read(os.environ["LEWM_CONFIG"])
    if args.task:
        render_task(config, args.task)
    else:
        package(config)
