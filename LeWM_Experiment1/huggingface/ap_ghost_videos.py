#!/usr/bin/env python3
"""Compose matched AP/CEM videos from verified saved evidence, without replay.

The main view is an honest 50/50 blend of original observation crops. Physical
paths are plotted separately in native coordinates; no image-plane projection,
segmentation, simulator, checkpoint or model package is used. Every displayed
state/action comes from the saved primitive-step trace. Images are held between
recorded observations and after stopping, with explicit action-index labels.
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
TEXT, MUTED, BACKGROUND = "#172c3b", "#586c78", "#f4f7fa"
SIZE, FPS, CROP_SIZE = (1280, 960), 2, (448, 448)
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
            for size in (14, 15, 16, 17, 18, 20, 22, 28, 32)}


class PathPlot:
    def __init__(self, pair, loaded):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.loaded, self.lines, self.points = loaded, {}, {}
        self.figure = Figure(figsize=(6.64, 3.10), dpi=100, facecolor="white")
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
                points = series[:min(action, stop) + 1]
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


def action_scales(loaded):
    actions = [step["action"] for arm in loaded.values() for step in arm["trajectory"]["steps"]]
    if not actions:
        return []
    dimensions = len(actions[0])
    require(all(len(action) == dimensions for action in actions), "Action dimensions differ")
    require(all(math.isfinite(float(value)) for action in actions for value in action),
            "Nonfinite executed action")
    scales = [max(max(abs(float(action[i])) for action in actions), 1e-6)
              for i in range(dimensions)]
    require(all(math.isfinite(value) for value in scales), "Nonfinite executed action")
    return scales


def action_panel(draw, font, box, name, arm, action, scales):
    x, y, width, height = box
    stop = arm["source"]["stop_action"]
    completed = min(action, stop)
    stopped = action >= stop
    color = MUTED if stopped else COLORS[name]
    draw.rounded_rectangle((x, y, x + width, y + height), radius=12, fill="white")
    draw.text((x + 16, y + 10), NAMES[name], font=font[20], fill=COLORS[name])
    if completed == 0:
        caption = "No action executed" + ("; stopped at reset" if stopped else "")
    elif stopped:
        caption = f"STOPPED at {stop}; last action held"
    else:
        caption = f"Last executed action: {completed}"
    draw.text((x + 16, y + 38), caption, font=font[15], fill=color)
    values = arm["trajectory"]["steps"][completed - 1]["action"] if completed else [0] * len(scales)
    row_height = min(29, (height - 72) / max(1, len(scales)))
    center, half = x + 300, 185
    for index, (value, scale) in enumerate(zip(values, scales)):
        row_y = y + 69 + index * row_height
        draw.text((x + 16, row_y - 8), f"a{index}", font=font[15], fill=TEXT)
        draw.line((center - half, row_y, center + half, row_y), fill="#dce4e9", width=3)
        draw.line((center, row_y - 6, center, row_y + 6), fill="#9aa9b3", width=1)
        end = center + half * float(value) / scale
        if completed and abs(end - center) > 0.5:
            draw.rectangle((min(center, end), row_y - 4, max(center, end), row_y + 4), fill=color)
        draw.text((x + width - 95, row_y - 9), f"{value:+.3f}" if completed else "--",
                  font=font[15], fill=color)


def compose_frame(pair, loaded, action, plot, font, scales):
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    task = ap_media.TASKS[pair["task"]]
    draw.text((24, 17), f"{task}  |  Matched planner comparison", font=font[32], fill=TEXT)
    draw.text((24, 60), f"Query {pair['query_ordinal']:03d}  |  Same start, goal and seeds  |  "
              f"Action timeline: {action}", font=font[20], fill=MUTED)
    observations, metadata = {}, {}
    for name in ARMS:
        arm = loaded[name]
        index = bisect_right(arm["source"]["frame_steps"], action) - 1
        observed = arm["source"]["frame_steps"][index]
        observations[name] = arm["frames"][index]
        stop = arm["source"]["stop_action"]
        completed = min(action, stop)
        metadata[name] = {"observed": observed, "stopped": action >= stop, "stop": stop,
                          "executed_actions": completed,
                          "last_executed_action_index": completed - 1 if completed else None}
    overlay = Image.blend(observations[ARMS[0]], observations[ARMS[1]], 0.5)
    draw.text((24, 98), "Merged original observations · 50/50", font=font[22], fill=TEXT)
    canvas.paste(overlay.resize((520, 520)), (24, 130))
    draw.rounded_rectangle((23, 129, 545, 651), radius=2, outline="#cfdae2", width=2)
    for index, name in enumerate(ARMS):
        info = metadata[name]
        label = f"{NAMES[name]}: observation after {info['observed']} actions"
        if info["stopped"]:
            label += f"; STOPPED at {info['stop']}"
        elif info["observed"] < action:
            label += " (held)"
        draw.text((24, 665 + index * 25), label, font=font[16], fill=COLORS[name])
    for index, (name, label) in enumerate(((ARMS[0], NAMES[ARMS[0]]),
                                         (ARMS[1], NAMES[ARMS[1]]), (None, "Fixed goal"))):
        x = 580 + index * 232
        color = COLORS[name] if name else "#8e700e"
        draw.text((x, 99), label, font=font[20], fill=color)
        image = observations[name] if name else loaded[ARMS[0]]["goal_image"]
        canvas.paste(image.resize((200, 200)), (x, 130))
        draw.rectangle((x - 1, 129, x + 200, 330), outline=color, width=3)
        if name:
            info = metadata[name]
            status = ("Stopped: " + ("success" if loaded[name]["source"]["success"] else "failure")) \
                if info["stopped"] else "Planning"
            draw.text((x, 337), status, font=font[16], fill=color)
            draw.text((x, 359), f"Observed at action {info['observed']}", font=font[14], fill=MUTED)
        else:
            draw.text((x, 337), "Same dataset goal", font=font[16], fill=MUTED)
    label = "Recorded joint-space paths" if pair["task"] == "reacher" else \
        "Recorded cube XYZ paths" if pair["task"] == "cube" else "Recorded native XY paths"
    draw.text((580, 391), label, font=font[20], fill=TEXT)
    canvas.paste(plot.frame(action), (580, 420))
    action_panel(draw, font, (24, 743, 596, 177), ARMS[0], loaded[ARMS[0]], action, scales)
    action_panel(draw, font, (660, 743, 596, 177), ARMS[1], loaded[ARMS[1]], action, scales)
    draw.text((24, 935), "Saved frames only · No rerun · Paths use recorded states · "
              "Signed action bars use shared per-channel scales", font=font[15], fill=MUTED)
    return canvas, metadata


def render_pair(outcome, pair, loaded, sources, output, relative, ffmpeg, generator_hash, qa_root):
    from PIL import Image

    for name in ARMS:
        arm = loaded[name]
        arm["frames"], arm["goal_image"] = decoded_frames(
            ffmpeg, arm["paths"]["video"], len(arm["source"]["frame_steps"]))
    timeline = sorted({point for arm in loaded.values() for point in arm["source"]["frame_steps"]}
                      | {arm["source"]["stop_action"] for arm in loaded.values()})
    timeline.extend([timeline[-1]] * 3)  # Two seconds to read the labelled final held state.
    output.parent.mkdir(parents=True, exist_ok=True)
    qa_root.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), f"Derived destination already exists: {output}")
    poster = output.with_name(output.stem + "-poster.png")
    require(not poster.exists(), "Derived poster already exists")
    font, plot, scales = fonts(), PathPlot(pair, loaded), action_scales(loaded)
    process = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x960",
                                "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
                                "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", str(output)], stdin=subprocess.PIPE)
    frames = []
    try:
        for index, action in enumerate(timeline):
            frame, metadata = compose_frame(pair, loaded, action, plot, font, scales)
            process.stdin.write(frame.tobytes())
            frames.append({"frame": index, "action_index": action, "arms": metadata})
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
            "action_scales": scales, "trajectory_coordinates": loaded[ARMS[0]]["plot"][
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
    result = {"schema": "lewm-ap-paired-ghost-videos-v1", "repo_id": REPO,
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
                           "timeline": "Union of native observation action indices and both true stopping actions",
                           "frame_holds": "Use the last original observation at or before each action index",
                           "final_state_hold_seconds": 2,
                           "action_bars": "Human action number; zero-based index retained in metadata; inactive after stop; shared channel scales",
                           "ghost_method": "50/50 original decoded pixels; no segmentation or image-plane projection"},
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
