#!/usr/bin/env python3
"""Compose matched AP/CEM videos from verified saved evidence, without replay.

The main view blends the two observation crops after separately smoothing each
saved video with motion-compensated interpolation. Synthetic in-between images
are presentation only, never scored evidence. Paths use the saved primitive-step
trace. No simulator, checkpoint or model package is used. Both arms share a
constant action-time clock and hold their last available image after recording.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile

import ap_media

REPO = ap_media.REPO
ARMS = ("gaussian_observed", "gaussian_final")
NAMES = {"gaussian_observed": "AP-CEM", "gaussian_final": "Final-goal CEM"}
COLORS = {"gaussian_observed": "#197a91", "gaussian_final": "#d36a32"}
GHOST_TINTS = {"gaussian_observed": (36, 184, 208), "gaussian_final": (245, 129, 50)}
TEXT, MUTED, BACKGROUND = "#172c3b", "#586c78", "#f4f7fa"
SIZE, FPS, CROP_SIZE = (1600, 900), 30, (448, 448)
ACTIONS_PER_SECOND, FINAL_HOLD_SECONDS = 20, 1.5
PROJECT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT / "LeWM_Experiment1/runs/AP"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_pairs(index, environments, outcomes):
    require(index.get("passed") is True and not index.get("errors"),
            "Paired-evidence verification has not passed")
    selected = []
    for task in environments:
        pairs = [pair for pair in index["pairs"] if pair["task"] == task]
        require(len(pairs) == 128 and {pair["query_ordinal"] for pair in pairs} == set(range(128)),
                f"Selection requires all 128 paired queries for {task}")
        for outcome in outcomes:
            eligible = [pair for pair in pairs if pair["arms"]["gaussian_observed"]["success"]
                        is (outcome == "success")]
            require(bool(eligible), f"No AP {outcome} exists for {task}")
            selected.append((outcome, min(eligible, key=lambda pair: pair["query_ordinal"])))
    return selected


def source_record(arm, kind):
    item = arm[kind]
    return {"source_path": f"runs/{arm['run_id']}/{item['path']}",
            "source_revision": arm["revision"],
            "source_manifest_sha256": arm["manifest_sha256"],
            "bytes": item["bytes"], "sha256": item["sha256"]}


def load_pair(pair, require_videos=True):
    require(pair["condition"] == "clean" and all(pair["equality"].values()),
            "The pair does not have identical initial conditions")
    require(set(pair["arms"]) == set(ARMS), "Both original planners are required")
    loaded, sources = {}, []
    for name in ARMS:
        arm = pair["arms"][name]
        require(arm["repository"] == REPO, "Wrong source repository")
        require(re.fullmatch(r"[0-9a-f]{40}", arm["revision"]), "Unpinned source revision")
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", arm["run_id"]),
                "Unsafe source run identity")
        root = RUN_ROOT / arm["run_id"]
        require(not root.is_symlink() and root.resolve().parent == RUN_ROOT.resolve(),
                "Source run escapes the downloaded evidence root")
        manifest_path = ap_media.source_file(root, "manifest.json")
        require(ap_media.digest(manifest_path) == arm["manifest_sha256"],
                "The local source manifest differs from its verified pin")
        manifest = ap_media.read_json(manifest_path)
        require(manifest["run_id"] == arm["run_id"], "Source manifest run differs")
        paths = {}
        for kind in ("case", "trajectory", "video"):
            item = arm[kind]
            expected = {"bytes": item["bytes"], "sha256": item["sha256"]}
            require(manifest["files"].get(item["path"]) == expected,
                    f"Pair metadata differs from the pinned {kind} manifest entry")
            path = ap_media.source_file(root, item["path"])
            require(path.resolve() == Path(item["local_path"]).resolve(),
                    "Pair local path differs from the original run path")
            if kind != "video" or require_videos:
                ap_media.verify_file(root, item["path"], manifest["files"])
            paths[kind] = path
            sources.append(source_record(arm, kind))
        case = ap_media.read_json(paths["case"])
        trajectory = ap_media.read_json(paths["trajectory"])
        require(case["compact"] == arm["compact"], "Saved native outcome differs from pair")
        compact = case["compact"]
        require(compact["identity"] == pair["identity"] and compact["arm"] == name
                and compact["query_ordinal"] == pair["query_ordinal"]
                and compact["optimizer_seed"] == pair["optimizer_seed"]
                and compact["condition"] == pair["condition"] == "clean"
                and compact["task"] == pair["task"] and compact["completed"] is True,
                "Native case identity differs")
        require(len(trajectory["steps"]) == compact["primitive_steps"] == arm["stop_action"],
                "Trace length differs from the original stopping action")
        require([step["action"] for step in trajectory["steps"]] == case["actions"],
                "Primitive actions differ between native and physical traces")
        video = trajectory["video"]
        require(video["capture"] == "native planner render calls only"
                and video["frame_steps"] == arm["frame_steps"]
                and video["frames"] == len(arm["frame_steps"]), "Native frame index differs")
        require(arm["frame_steps"] == list(range(0, max(1, arm["stop_action"]), 5)),
                "Unexpected native observation spacing")
        require([entry["action_index"] for entry in trajectory["render_hashes"]]
                == arm["frame_steps"], "Original render hashes have different action indices")
        physical = ap_media.coordinates(pair["task"], trajectory)
        # Earlier PushT records contain a repeated full goal image at every step.
        # Validate original bytes first, then retain only fields used by this view.
        light = {"initial": trajectory["initial"], "render_hashes": trajectory["render_hashes"],
                 "steps": [{"action": step["action"]} for step in trajectory["steps"]]}
        loaded[name] = {"source": arm, "case": compact, "trajectory": light,
                        "paths": paths, "plot": physical}
        del trajectory, case
    left, right = (loaded[name] for name in ARMS)
    for key in ("identity", "observation", "snapshot", "true_goal", "goal_rgb_sha256"):
        require(left["trajectory"]["initial"][key] == right["trajectory"]["initial"][key],
                f"Paired initial {key} differs")
    require(left["trajectory"]["render_hashes"][0]["sha256"]
            == right["trajectory"]["render_hashes"][0]["sha256"],
            "Paired initial raw rendered images differ")
    return loaded, sources


def ffmpeg_executable():
    path = shutil.which("ffmpeg")
    if path is None:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
    require(Path(path).is_file(), "An existing ffmpeg executable is required")
    return path


def decoded_frames(ffmpeg, path, expected):
    from PIL import Image

    command = [ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-vf",
               "crop=448:448:0:50", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    result = subprocess.run(command, capture_output=True, check=True, timeout=120)
    frame_bytes = math.prod(CROP_SIZE) * 3
    require(len(result.stdout) == expected * frame_bytes, "Decoded original frame count differs")
    frames = [Image.frombytes("RGB", CROP_SIZE, result.stdout[i:i + frame_bytes])
              for i in range(0, len(result.stdout), frame_bytes)]
    goal = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", str(path),
                           "-frames:v", "1", "-vf", "crop=448:448:448:50", "-f",
                           "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                          capture_output=True, check=True, timeout=120)
    require(len(goal.stdout) == frame_bytes, "Fixed-goal crop has an unexpected size")
    return frames, Image.frombytes("RGB", CROP_SIZE, goal.stdout)


def fonts():
    from PIL import ImageFont

    base = Path("/usr/share/fonts/truetype/dejavu")
    require((base / "DejaVuSans.ttf").is_file(), "DejaVu Sans is required for readable labels")
    return {size: ImageFont.truetype(str(base / "DejaVuSans.ttf"), size)
            for size in (14, 15, 16, 17, 18, 20, 22, 24, 28, 32, 36)}


def smoothed_frames(ffmpeg, arm, destination):
    """Interpolate each original crop before blending; retain an exact time map."""
    import numpy as np

    last = arm["source"]["last_frame_action"]
    count = math.ceil(last * FPS / ACTIONS_PER_SECOND) + 1
    if count == 1:
        destination.write_bytes(arm["frames"][0].tobytes())
        return np.memmap(destination, dtype="uint8", mode="r", shape=(1, 448, 448, 3))
    # Two cloned source frames let the bidirectional filter reach the final
    # observation. Trim them away; later holds are handled by the shared clock.
    filters = ("crop=448:448:0:50,tpad=stop_mode=clone:stop_duration=0.5,"
               f"minterpolate=fps={FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:"
               "me=epzs:mb_size=8:search_param=32:vsbmc=1,"
               f"trim=end_frame={count},setpts=PTS-STARTPTS")
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-n", "-threads", "2",
                    "-r", str(ACTIONS_PER_SECOND / 5), "-i", str(arm["paths"]["video"]),
                    "-vf", filters, "-an", "-fps_mode", "passthrough",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    str(destination)], capture_output=True, check=True, timeout=600)
    require(destination.stat().st_size == count * math.prod(CROP_SIZE) * 3,
            "Motion interpolation produced an unexpected number of frames")
    return np.memmap(destination, dtype="uint8", mode="r", shape=(count, 448, 448, 3))


class PathPlot:
    def __init__(self, pair, loaded):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.loaded, self.lines, self.points = loaded, {}, {}
        self.figure = Figure(figsize=(7.84, 3.55), dpi=100, facecolor="white")
        self.canvas = FigureCanvasAgg(self.figure)
        cube = pair["task"] == "cube"
        pusht = pair["task"] == "pusht"
        self.cube, self.bounds = cube, {}
        if pusht:
            axes = [self.figure.add_axes((0.07, 0.14, 0.39, 0.65)),
                    self.figure.add_axes((0.56, 0.14, 0.40, 0.65))]
            axes[0].set_title("Full paths", fontsize=9, pad=6)
            axes[1].set_title("Goal region (zoom)", fontsize=9, pad=6)
        else:
            axes = [self.figure.add_subplot(111, projection="3d" if cube else None)]
        all_points, detail_points = [], []
        for name in ARMS:
            plot = loaded[name]["plot"]
            for label, points in plot["series"].items():
                all_points.extend(points)
                detail_points.extend(points if label == "Block" else points[:1])
        goals = loaded[ARMS[0]]["plot"]["goals"]
        all_points.extend(goals.values())
        detail_points.extend(goals.values())
        labels = loaded[ARMS[0]]["plot"]["axis_labels"]
        for view, axis in enumerate(axes):
            for name in ARMS:
                series = loaded[name]["plot"]["series"]
                for label in series:
                    style = "--" if label == "Block" else "-"
                    line, = axis.plot([], [], *([[]] if cube else []), color=COLORS[name],
                                      linestyle=style, linewidth=2,
                                      label=NAMES[name] + (" " + label if len(series) > 1 else ""))
                    marker, = axis.plot([], [], *([[]] if cube else []), marker="o",
                                        color=COLORS[name], markersize=5, linestyle="none")
                    self.lines[view, name, label] = line
                    self.points[view, name, label] = marker
            for index, goal in enumerate(goals.values()):
                axis.scatter(*([goal[i]] for i in range(len(goal))), marker="*", s=110,
                             color="#8e700e", label="Goal" if index == 0 else None, zorder=8)
            limits = []
            limit_points = all_points if view == 0 else detail_points
            setters = (axis.set_xlim, axis.set_ylim, axis.set_zlim) if cube else \
                (axis.set_xlim, axis.set_ylim)
            for dimension, setter in enumerate(setters):
                values = [point[dimension] for point in limit_points]
                low, high = min(values), max(values)
                margin = max((high - low) * 0.1, 0.02 if cube else 0.1)
                limit = [low - margin, high + margin]
                limits.append(limit)
                setter(*limit)
            self.bounds["full_paths" if view == 0 else "goal_region"] = limits
            axis.set_xlabel(labels[0], fontsize=8 if pusht else 9, labelpad=2)
            axis.set_ylabel(labels[1], fontsize=8 if pusht else 9, labelpad=2)
            if cube:
                axis.set_zlabel(labels[2], fontsize=9, labelpad=2)
                axis.view_init(elev=23, azim=-57)
            else:
                axis.set_aspect("equal", adjustable="box")
                axis.grid(alpha=0.18)
            axis.tick_params(labelsize=7 if pusht else 8)
        if pusht:
            handles, names = axes[0].get_legend_handles_labels()
            self.figure.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                               fontsize=7.5, frameon=False, ncol=3)
            self.bounds["goal_region_rule"] = (
                "Shared initial agent/block positions and goals plus both complete block paths; "
                "fixed 10% margin; full overview retains both complete agent paths")
        else:
            axes[0].legend(loc="upper left", fontsize=8, frameon=False)
            self.figure.subplots_adjust(left=0.11, right=0.95, bottom=0.16, top=0.96)
        self.axes = axes

    def frame(self, action):
        from PIL import Image

        for name in ARMS:
            stop = self.loaded[name]["source"]["stop_action"]
            for label, series in self.loaded[name]["plot"]["series"].items():
                position = min(action, stop)
                whole = int(position)
                points = list(series[:whole + 1])
                if whole < stop and position > whole:
                    fraction = position - whole
                    points.append([a + fraction * (b - a)
                                   for a, b in zip(series[whole], series[whole + 1])])
                x, y = [point[0] for point in points], [point[1] for point in points]
                for view in range(len(self.axes)):
                    line, marker = self.lines[view, name, label], self.points[view, name, label]
                    line.set_data(x, y)
                    marker.set_data([x[-1]], [y[-1]])
                    if self.cube:
                        z = [point[2] for point in points]
                        line.set_3d_properties(z)
                        marker.set_3d_properties([z[-1]])
        self.canvas.draw()
        return Image.frombuffer("RGBA", self.canvas.get_width_height(),
                                self.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")


def tinted_ghost(image, task, name):
    """Recolor visible agents/objects by their existing pixel colors, without warping.

    The task-specific soft masks retain static walls, goals and dark backgrounds.
    This is a visualization mask, not a recovered object/physics annotation.
    """
    import numpy as np
    from PIL import Image

    pixels = np.asarray(image, dtype=np.float32)
    red, green, blue = pixels.transpose(2, 0, 1)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    tint = np.array(GHOST_TINTS[name], dtype=np.float32)
    if task == "tworoom":
        # Replace only the red excess, including faint glow over black walls.
        # Dividing by the local red value would amplify dark codec noise.
        excess = np.maximum(red - np.maximum(green, blue), 0)
        neutral = pixels.copy()
        neutral[..., 0] -= excess
        result = neutral + tint * (excess / 255)[..., None]
    else:
        if task == "pusht":
            mask = np.clip(np.minimum((blue - red - 3) / 12, (blue - green - 2) / 7), 0, 1)
        elif task == "reacher":
            yellow = np.minimum((red - blue - 10) / 40, (red - 75) / 75)
            cyan = np.minimum((green - red - 30) / 70, (green - 120) / 80)
            mask = np.clip(np.maximum(yellow, cyan), 0, 1)
        else:
            require(task == "cube", "Unknown tint palette")
            metal = (luminance - 75) / 35
            purple = np.minimum((red - green - 10) / 25, (blue - green - 20) / 30)
            cube = np.minimum((red - green - 30) / 40, (red - blue - 20) / 40)
            mask = np.clip(np.maximum.reduce([metal, purple, cube]), 0, 1)
        shade = 0.55 + 0.45 * luminance / 255
        colored = shade[..., None] * tint
        result = pixels * (1 - mask[..., None]) + colored * mask[..., None]
    return Image.fromarray(np.rint(np.clip(result, 0, 255)).astype(np.uint8))


def compose_frame(pair, loaded, action, plot, font):
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    task = ap_media.TASKS[pair["task"]]
    draw.text((32, 21), f"{task}  /  AP vs CEM", font=font[36], fill=TEXT)
    draw.text((32, 70), f"Query {pair['query_ordinal']:03d}  ·  Same start, goal and seeds",
              font=font[20], fill=MUTED)
    maximum = max(arm["source"]["stop_action"] for arm in loaded.values())
    draw.text((1192, 29), f"Step {int(action):03d} / {maximum:03d}", font=font[28], fill=TEXT)
    draw.text((1192, 70), "20 actions / playback second", font=font[18], fill=MUTED)
    observations, metadata = {}, {}
    for name in ARMS:
        arm = loaded[name]
        source_steps = arm["source"]["frame_steps"]
        index = max(0, bisect_right(source_steps, action) - 1)
        observed = source_steps[index]
        next_observed = source_steps[min(index + 1, len(source_steps) - 1)]
        smooth_index = min(round(action * FPS / ACTIONS_PER_SECOND), len(arm["smooth"]) - 1)
        observations[name] = Image.fromarray(arm["smooth"][smooth_index])
        # Use the original decoded pixels at exact observed timestamps and after
        # the last capture. This also makes the shared initial image exact.
        if math.isclose(action, observed, abs_tol=1e-8) or action >= source_steps[-1]:
            observations[name] = arm["frames"][index]
        stop = arm["source"]["stop_action"]
        metadata[name] = {"observation_bracket": [observed, next_observed],
                          "interpolated": observed < action < next_observed,
                          "image_held": action > source_steps[-1],
                          "stopped": action >= stop, "stop": stop,
                          "executed_actions": min(int(action), stop)}
    overlay = Image.blend(tinted_ghost(observations[ARMS[0]], pair["task"], ARMS[0]),
                          tinted_ghost(observations[ARMS[1]], pair["task"], ARMS[1]), 0.5)
    draw.text((32, 115), "GHOST COMPARISON", font=font[18], fill=MUTED)
    draw.ellipse((548, 119, 560, 131), fill=COLORS[ARMS[0]])
    draw.text((570, 114), "AP", font=font[18], fill=COLORS[ARMS[0]])
    draw.ellipse((630, 119, 642, 131), fill=COLORS[ARMS[1]])
    draw.text((652, 114), "CEM", font=font[18], fill=COLORS[ARMS[1]])
    canvas.paste(overlay.resize((712, 712), Image.Resampling.LANCZOS), (32, 146))
    draw.rectangle((31, 145, 745, 859), outline="#d9e2e8", width=2)
    for index, (name, label) in enumerate(((ARMS[0], NAMES[ARMS[0]]),
                                         (ARMS[1], NAMES[ARMS[1]]), (None, "Goal"))):
        x = 784 + index * 272
        color = COLORS[name] if name else "#8e700e"
        draw.text((x, 112), label, font=font[22], fill=color)
        image = observations[name] if name else loaded[ARMS[0]]["goal_image"]
        canvas.paste(image.resize((240, 240), Image.Resampling.LANCZOS), (x, 146))
        draw.rectangle((x - 1, 145, x + 240, 386), outline=color, width=3)
        if name:
            info = metadata[name]
            if info["stopped"]:
                outcome = "Reached goal" if loaded[name]["source"]["success"] else "Not reached"
                status = f"{outcome} · {info['stop']} steps"
            else:
                status = "Moving" if not info["image_held"] else "Last recorded view"
            draw.text((x, 397), status, font=font[16], fill=color)
            if info["image_held"]:
                draw.text((x, 423), f"View held from step {arm_last(loaded, name)}",
                          font=font[15], fill=MUTED)
        else:
            draw.text((x, 397), "Shared target", font=font[16], fill=MUTED)
    label = "Joint-space trajectories" if pair["task"] == "reacher" else \
        "Cube trajectories · XYZ" if pair["task"] == "cube" else "Recorded trajectories"
    draw.text((784, 469), label, font=font[24], fill=TEXT)
    canvas.paste(plot.frame(action), (784, 508))
    draw.text((32, 875), "Motion-smoothed playback · Original captured frames are linked below",
              font=font[15], fill=MUTED)
    return canvas, metadata


def arm_last(loaded, name):
    return loaded[name]["source"]["last_frame_action"]


def render_pair(outcome, pair, loaded, sources, output, relative, ffmpeg, generator_hash, qa_root):
    from PIL import Image

    for name in ARMS:
        arm = loaded[name]
        arm["frames"], arm["goal_image"] = decoded_frames(
            ffmpeg, arm["paths"]["video"], len(arm["source"]["frame_steps"]))
    maximum = max(arm["source"]["stop_action"] for arm in loaded.values())
    timeline = [min(index * ACTIONS_PER_SECOND / FPS, maximum)
                for index in range(math.ceil(maximum * FPS / ACTIONS_PER_SECOND) + 1)]
    timeline.extend([maximum] * round(FPS * FINAL_HOLD_SECONDS))
    output.parent.mkdir(parents=True, exist_ok=True)
    qa_root.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), f"Derived destination already exists: {output}")
    poster = output.with_name(output.stem + "-poster.png")
    require(not poster.exists(), "Derived poster already exists")
    font, plot = fonts(), PathPlot(pair, loaded)
    temporary = tempfile.TemporaryDirectory(prefix="ap-smoothed-frames-")
    for name in ARMS:
        loaded[name]["smooth"] = smoothed_frames(
            ffmpeg, loaded[name], Path(temporary.name) / (name + ".rgb"))
    process = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}",
                                "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
                                "-preset", "medium", "-crf", "18", "-threads", "4", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", str(output)], stdin=subprocess.PIPE)
    frames = []
    try:
        for index, action in enumerate(timeline):
            frame, metadata = compose_frame(pair, loaded, action, plot, font)
            process.stdin.write(frame.tobytes())
            frames.append({"frame": index, "playback_seconds": index / FPS,
                           "action_index": action, "arms": metadata})
            if index in {0, len(timeline) // 2, len(timeline) - 1}:
                frame.save(qa_root / (output.stem + f"-frame-{index:03d}.png"))
            if index == len(timeline) // 2:
                frame.save(poster)
        process.stdin.close()
        require(process.wait(timeout=120) == 0, "Derived video encoding failed")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for name in ARMS:
            del loaded[name]["smooth"]
        temporary.cleanup()
    require(ap_media.digest(Path(__file__)) == generator_hash, "Generator changed during rendering")
    # Decode the result to a null sink. This checks the complete encoded bitstream.
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", str(output), "-f", "null", "-"],
                   capture_output=True, check=True, timeout=120)
    with Image.open(poster) as image:
        require(image.size == SIZE, "Derived preview dimensions differ")
    stops = {name: {"stopping_action": loaded[name]["source"]["stop_action"],
                    "last_observed_action": loaded[name]["source"]["last_frame_action"],
                    "success": loaded[name]["source"]["success"],
                    "stop_reason": loaded[name]["source"]["stop_reason"],
                    "held_frames_after_stop": sum(frame["arms"][name]["stopped"] for frame in frames)}
             for name in ARMS}
    return {"environment": pair["task"], "outcome": outcome,
            "query_ordinal": pair["query_ordinal"], "record_id": pair["record_id"],
            "identity": pair["identity"], "optimizer_seed": pair["optimizer_seed"],
            "output": {"path": relative, "bytes": output.stat().st_size,
                       "sha256": ap_media.digest(output)}, "sources": sources,
            "poster": {"path": str(Path(relative).with_name(poster.name)),
                       "bytes": poster.stat().st_size, "sha256": ap_media.digest(poster)},
            "poster_position": {"frame": len(timeline) // 2,
                                "action_index": timeline[len(timeline) // 2]},
            "comparison_checks": {"same_seed": True, "same_initial_state": True, "same_goal": True},
            "simulator_actions": 0, "model_queries": 0, "stopping": stops,
            "frames": frames, "fps": FPS, "resolution": list(SIZE),
            "playback_actions_per_second": ACTIONS_PER_SECOND,
            "motion_interpolation": "FFmpeg minterpolate; per arm before ghost blending",
            "trajectory_coordinates": loaded[ARMS[0]]["plot"][
                "coordinate_note"], "trajectory_views": plot.bounds,
            "pair_raw_render_sha256": pair["initial_render_sha256"],
            "pair_goal_rgb_sha256": pair["goal_rgb_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--qa-root", type=Path,
                        help="QA-only frames, default a separate sibling of --output-root")
    parser.add_argument("--environment", action="append", choices=ap_media.TASKS)
    parser.add_argument("--outcome", action="append", choices=("success", "failure"))
    parser.add_argument("--inputs-only", action="store_true",
                        help="Validate paired metadata and list required original video downloads")
    args = parser.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.report_id), "Unsafe report ID")
    environments = args.environment or list(ap_media.TASKS)
    outcomes = args.outcome or ["success", "failure"]
    require(len(set(environments)) == len(environments) and len(set(outcomes)) == len(outcomes),
            "Duplicate environment or outcome selection")
    helper_hash = ap_media.digest(Path(ap_media.__file__))
    generator_hash = ap_media.digest(Path(__file__))
    raw_index = args.pair_index.read_bytes()
    pair_index_hash = hashlib.sha256(raw_index).hexdigest()
    index = json.loads(raw_index)
    selected = selected_pairs(index, environments, outcomes)
    loaded = [(outcome, pair, *load_pair(pair, require_videos=not args.inputs_only))
              for outcome, pair in selected]
    require(not any(path.is_symlink() for path in (args.output_root, *args.output_root.parents)),
            "Derived output cannot traverse a symlink")
    require(not any((path / "run.json").exists()
                    for path in (args.output_root, *args.output_root.parents)),
            "Derived output must not be written into an original run")
    qa_root = args.qa_root or args.output_root.with_name(args.output_root.name + "-qa")
    require(not qa_root.resolve().is_relative_to(args.output_root.resolve()),
            "QA frames must stay outside the publication content directory")
    require(not any(path.is_symlink() or (path / "run.json").exists()
                    for path in (qa_root, *qa_root.parents)), "Unsafe QA output directory")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.inputs_only:
        downloads = [dict(source, run_id=arm["source"]["run_id"],
                          local_path=str(arm["paths"]["video"]),
                          exists=arm["paths"]["video"].is_file())
                     for _, _, arms, _ in loaded for arm in arms.values()
                     for source in [source_record(arm["source"], "video")]]
        destination = args.output_root / "required-original-videos.json"
        destination.write_text(json.dumps(downloads, indent=2) + "\n")
        print(json.dumps({"required_videos": len(downloads), "path": str(destination)}))
        return
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="ap-ghost-matplotlib-"))
    import PIL
    import matplotlib
    matplotlib.use("Agg")
    ffmpeg = ffmpeg_executable()
    ffmpeg_version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True,
                                    check=True, timeout=10).stdout.splitlines()[0]
    prefix = f"media/{args.report_id}"
    comparisons = []
    for outcome, pair, arms, sources in loaded:
        relative = f"{prefix}/comparisons/{pair['task']}-{outcome}.mp4"
        comparison = render_pair(outcome, pair, arms, sources, args.output_root / relative,
                                 relative, ffmpeg, generator_hash, qa_root)
        comparisons.append(comparison)
        print(json.dumps({"completed": f"{pair['task']}/{outcome}",
                          "query_ordinal": pair["query_ordinal"], "output": comparison["output"]}),
              flush=True)
    require(ap_media.digest(Path(ap_media.__file__)) == helper_hash,
            "Coordinate helper changed during verification/rendering")
    result = {"schema": "lewm-ap-paired-ghost-videos-v2", "repo_id": REPO,
              "derived_comparisons_verified": True, "created_utc": datetime.now(timezone.utc).isoformat(),
              "generator": {"source_sha256": generator_hash, "source_path": str(Path(__file__)),
                            "coordinate_helper_sha256": helper_hash,
                            "python": platform.python_version(), "pillow": PIL.__version__,
                            "matplotlib": matplotlib.__version__, "ffmpeg": ffmpeg_version,
                            "ffmpeg_executable_sha256": ap_media.digest(Path(ffmpeg))},
              "pair_index": {"path": str(args.pair_index), "sha256": pair_index_hash},
              "settings": {"size": list(SIZE), "fps": FPS, "overlay_alpha": 0.5,
                           "observation_crop": [0, 50, 448, 498], "goal_crop": [448, 50, 896, 498],
                           "selection": "First AP success/failure by original ordinal in a complete cohort",
                           "timeline": "Constant shared action clock; shorter arm stays at its stop",
                           "playback_actions_per_second": ACTIONS_PER_SECOND,
                           "frame_holds": "Hold original last observation after each arm's final capture",
                           "final_state_hold_seconds": FINAL_HOLD_SECONDS,
                           "action_bars": False,
                           "motion_interpolation": {
                               "method": "FFmpeg minterpolate",
                               "settings": "mci/aobmc/bidir/epzs, block8, search32, vsbmc1",
                               "scope": "Each original observation crop separately, before ghost blending",
                               "evidence_status": "Synthetic in-between images for presentation only",
                               "native_capture_spacing_actions": 5},
                           "trajectory_interpolation": "Linear marker motion between recorded primitive states",
                           "ghost_method": "50/50 blend of separately smoothed and tinted crops",
                           "ghost_tints": {name: list(color) for name, color in GHOST_TINTS.items()},
                           "tint_mask": "Task-specific soft pixel-color masks; visualization only",
                           "individual_previews": "Original colors, with motion interpolation"},
              "selected_environments": environments, "selected_outcomes": outcomes,
              "complete_eight_comparison_set": len(comparisons) == 8,
              "comparisons": comparisons, "simulator_actions": 0, "model_queries": 0}
    destination = args.output_root / prefix / "comparisons.json"
    require(not destination.exists(), "Comparison index already exists")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"index": str(destination), "sha256": ap_media.digest(destination),
                      "comparisons": len(comparisons)}), flush=True)


if __name__ == "__main__":
    main()
