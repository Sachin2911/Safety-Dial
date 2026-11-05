#!/usr/bin/env python3
"""Select original AP videos and plot saved physical trajectories, without replay.

Example:
  python3 ap_media.py --run-root /download/runs/eval-shard-0 \
    --run-root /download/runs/eval-shard-1 --status final-status.json \
    --report-id ap-reproduction-20261105 --output content-ap

Supply every downloaded evaluation run root. Each needs its final manifest.json
and all metrics/cases/*.json listed in that manifest. The selected cases also
need metrics/trajectories/<case>.json.gz. The status JSON is the compute wrapper's
saved status with upload.verified, upload.revision and upload.manifest_sha256 for
each run. Manifest and trace bytes are checked locally before selection.

Both arms must contain all 128 original query ordinals in all four environments.
Selection is the lowest original query ordinal among AP-CEM successes/failures;
an absent outcome is recorded explicitly. Native MP4s need not be downloaded:
downloads.json lists only the required pinned original paths, hashes and sizes.
No network calls or video modifications are performed here.

Default output includes one trajectory PNG per selected case, using every saved
physical state including the stopping state. Matplotlib is imported only for
these plots; --no-plots performs standard-library-only selection. Generated plots
are marked as derived trace analysis, separate from the original video bytes.
No model or simulator package is imported, and no environment is stepped.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import re
from urllib.parse import quote


REPO = "Sachioster/LeWMExp1_AP_Reproduction"
TASKS = {"tworoom": "TwoRoom", "reacher": "Reacher", "pusht": "PushT", "cube": "OGBench Cube"}
ARMS = ("gaussian_final", "gaussian_observed")
COLORS = {"agent": "#286B9E", "block": "#8063A7", "start": "#23806B",
          "stop": "#C5573D", "goal": "#C28D17", "text": "#213547"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def safe_relative(value):
    require(isinstance(value, str) and value, "Missing relative source path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in value
            and not any(ord(character) < 32 for character in value), "Unsafe relative path")
    return path


def source_file(root, name):
    path = root / safe_relative(name)
    require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
            f"Source escapes downloaded run: {name}")
    return path


def verify_file(root, name, files):
    require(name in files, f"File absent from final manifest: {root.name}/{name}")
    expected = files[name]
    require(type(expected.get("bytes")) is int and expected["bytes"] >= 0
            and re.fullmatch(r"[0-9a-f]{64}", expected.get("sha256", "")),
            f"Invalid manifest entry: {name}")
    path = source_file(root, name)
    require(path.is_file(), f"Download required: {root.name}/{name}")
    require(path.stat().st_size == expected["bytes"] and digest(path) == expected["sha256"],
            f"Downloaded bytes differ from final manifest: {root.name}/{name}")
    return path


def load_runs(run_roots, status):
    require(status.get("storage_repo") == REPO, "Status does not identify the AP dataset")
    by_id = {row["run_id"]: row for row in status["runs"]}
    require(len(by_id) == len(status["runs"]), "Status contains duplicate run identifiers")
    runs, seen = [], set()
    for root in run_roots:
        root = root.resolve()
        manifest_path = source_file(root, "manifest.json")
        require(manifest_path.is_file(), f"Final manifest missing from {root}")
        manifest = read_json(manifest_path)
        run_id = manifest["run_id"]
        require(re.fullmatch(r"[A-Za-z0-9_-]+", run_id), "Unsafe run identifier")
        require(run_id not in seen, f"Run supplied twice: {run_id}")
        seen.add(run_id)
        require(run_id in by_id, f"Run missing from saved status: {run_id}")
        row = by_id[run_id]
        upload = row.get("upload") or {}
        require(upload.get("verified") is True and re.fullmatch(r"[0-9a-f]{40}", upload.get("revision", "")),
                f"Run needs a verified pinned upload: {run_id}")
        require(not row.get("job_alive") and not row.get("supervisor_alive"),
                f"Run is still active: {run_id}")
        require(digest(manifest_path) == upload.get("manifest_sha256"),
                f"Manifest differs from the verified upload receipt: {run_id}")
        require(manifest.get("schema_version") == 1, f"Unexpected manifest schema: {run_id}")
        require(manifest["result"] == row["result"], f"Run result differs from status: {run_id}")
        runs.append({"root": root, "run_id": run_id, "revision": upload["revision"],
                     "manifest_sha256": upload["manifest_sha256"], "files": manifest["files"]})
    return runs


def collect_cases(runs):
    cases = {}
    for run in runs:
        names = sorted(name for name in run["files"]
                       if re.fullmatch(r"metrics/cases/[^/]+\.json", name))
        for name in names:
            record = read_json(verify_file(run["root"], name, run["files"]))
            case = record["compact"]
            task, arm, ordinal = case["task"], case["arm"], case["query_ordinal"]
            require(task in TASKS and arm in ARMS, f"Unexpected study case: {name}")
            require(type(ordinal) is int and 0 <= ordinal < 128, f"Invalid original ordinal: {name}")
            require(case["completed"] is True and case["condition"] == "clean",
                    f"Case is incomplete or outside the clean cohort: {name}")
            require(type(case["success"]) is bool, f"Invalid recorded outcome: {name}")
            require(type(case["primitive_steps"]) is int
                    and 0 <= case["primitive_steps"] <= case["budget"], f"Invalid action count: {name}")
            require(len(record["actions"]) == case["primitive_steps"], f"Native action count differs: {name}")
            key = task, arm, ordinal
            require(key not in cases, f"Duplicate original case: {key}")
            cases[key] = {"case": case, "trace": record, "run": run, "trace_name": name}
    for task in TASKS:
        for arm in ARMS:
            ordinals = {ordinal for selected_task, selected_arm, ordinal in cases
                        if selected_task == task and selected_arm == arm}
            require(ordinals == set(range(128)),
                    f"Complete cohort required before selecting media: {task}/{arm} has {len(ordinals)}/128")
        for ordinal in range(128):
            pair = [cases[task, arm, ordinal]["case"] for arm in ARMS]
            require(pair[0]["identity"] == pair[1]["identity"],
                    f"Paired original case identities differ: {task}/{ordinal}")
    return cases


def vector(values, count, label):
    require(isinstance(values, (list, tuple)) and len(values) >= count, f"Missing physical values: {label}")
    result = [float(value) for value in values[:count]]
    require(all(math.isfinite(value) for value in result), f"Non-finite physical values: {label}")
    return result


def state_vector(value):
    return value["state"] if isinstance(value, dict) else value


def coordinates(task, trajectory):
    states = [trajectory["initial"], *trajectory["steps"]]
    goal = trajectory["initial"]["true_goal"]
    if task == "tworoom":
        series = {"Agent": [vector(state_vector(state["snapshot"]["state"]), 2, "TwoRoom state")
                             for state in states]}
        goals = {"Agent": vector(goal, 2, "TwoRoom goal")}
        labels = ("x (native units)", "y (native units)")
        note = "Agent XY in native environment coordinates; walls are not reconstructed."
    elif task == "pusht":
        physical = [vector(state_vector(state["observation"]), 5, "PushT state") for state in states]
        series = {"Agent": [row[:2] for row in physical], "Block": [row[2:4] for row in physical]}
        target = vector(goal, 5, "PushT goal")
        goals = {"Agent": target[:2], "Block": target[2:4]}
        labels = ("x (native units)", "y (native units)")
        note = "Agent and block XY only. Block orientation is omitted; the native outcome includes it."
    elif task == "reacher":
        series = {"Arm": [vector(state["info"]["qpos"], 2, "Reacher info.qpos") for state in states]}
        goals = {"Arm": vector(goal, 2, "Reacher joint goal")}
        labels = ("Joint 1 angle (rad)", "Joint 2 angle (rad)")
        note = "Joint space from recorded info.qpos. This is not the fingertip's Cartesian path."
    elif task == "cube":
        series = {"Cube": [vector(state["snapshot"]["qpos"][14:17], 3, "Cube qpos[14:17]")
                            for state in states]}
        target = vector(goal, 22, "Cube normalized goal")
        goals = {"Cube": [target[19] / 10 + 0.425, target[20] / 10, target[21] / 10]}
        labels = ("x (m)", "y (m)", "z (m)")
        note = "Cube XYZ from saved qpos[14:17]; goal converted using the native normalization."
    else:
        raise ValueError(f"Unknown environment: {task}")
    return {"series": series, "goals": goals, "axis_labels": labels,
            "coordinate_note": note, "recorded_states": len(states)}


def verify_selected(selected):
    case, run = selected["case"], selected["run"]
    stem = PurePosixPath(selected["trace_name"]).stem
    trajectory_name = f"metrics/trajectories/{stem}.json.gz"
    trajectory_path = verify_file(run["root"], trajectory_name, run["files"])
    trajectory = read_json(trajectory_path)
    require(trajectory["task"] == case["task"] and trajectory["case"] == stem,
            "Selected trajectory identity differs")
    require(trajectory["initial"]["identity"] == case["identity"]
            and trajectory["initial"]["query_ordinal"] == case["query_ordinal"]
            and trajectory["initial"]["arm"] == case["arm"], "Trajectory original case differs")
    require(len(trajectory["steps"]) == case["primitive_steps"], "Selected physical action count differs")
    require([step["action"] for step in trajectory["steps"]] == selected["trace"]["actions"],
            "Selected physical and native actions differ")
    video = trajectory["video"]
    steps = video["frame_steps"]
    require(video["capture"] == "native planner render calls only", "Video capture is not native")
    require(steps and steps[0] == 0 and video["frames"] == len(steps)
            and all(type(step) is int and 0 <= step <= case["primitive_steps"] for step in steps)
            and all(b - a == 5 for a, b in zip(steps, steps[1:])), "Invalid native video frame index")
    require([frame["action_index"] for frame in trajectory["render_hashes"]] == steps,
            "Video frames differ from the saved render index")
    video_name = f"artifacts/videos/{stem}.mp4"
    require(video_name in run["files"], "Selected original video is absent from final manifest")
    video_file = run["files"][video_name]
    require(type(video_file.get("bytes")) is int and video_file["bytes"] > 0
            and re.fullmatch(r"[0-9a-f]{64}", video_file.get("sha256", "")), "Invalid original video metadata")
    if source_file(run["root"], video_name).exists():
        verify_file(run["root"], video_name, run["files"])
    return trajectory, trajectory_name, video_name


def limits(points, dimensions, minimum_span):
    lows = [min(point[index] for point in points) for index in range(dimensions)]
    highs = [max(point[index] for point in points) for index in range(dimensions)]
    span = max([high - low for low, high in zip(lows, highs)] + [minimum_span]) * 1.2
    return [(0.5 * (low + high) - span / 2, 0.5 * (low + high) + span / 2)
            for low, high in zip(lows, highs)]


def plot_trajectory(task, outcome, case, plot_data, destination):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    dimensions = len(plot_data["axis_labels"])
    figure = plt.figure(figsize=(8.8, 7.2), facecolor="white")
    axes = figure.add_subplot(111, projection="3d" if dimensions == 3 else None)
    all_points, handles = [], []
    for number, (name, values) in enumerate(plot_data["series"].items()):
        color = COLORS["agent"] if number == 0 else COLORS["block"]
        goal = plot_data["goals"][name]
        components = list(zip(*values))
        axes.plot(*components, color=color, linewidth=2.0, alpha=0.9)
        axes.scatter(*values[0], marker="o", color=COLORS["start"], s=75,
                     edgecolors="white", linewidths=0.9, zorder=5)
        axes.scatter(*values[-1], marker="s", facecolors="none", edgecolors=COLORS["stop"],
                     s=105, linewidths=2, zorder=6)
        axes.scatter(*goal, marker="*", color=COLORS["goal"], edgecolors=color,
                     linewidths=0.8, s=190, zorder=7)
        handles.append(Line2D([], [], color=color, linewidth=2, label=name + " path"))
        all_points.extend(values)
        all_points.append(goal)
    handles.extend([
        Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["start"],
               markeredgecolor="white", markersize=8, label="Start"),
        Line2D([], [], marker="s", color="none", markeredgecolor=COLORS["stop"],
               markeredgewidth=2, markersize=8, label="Stop"),
        Line2D([], [], marker="*", color="none", markerfacecolor=COLORS["goal"],
               markeredgecolor=COLORS["goal"], markersize=12, label="Goal"),
    ])
    bounds = limits(all_points, dimensions, 0.02 if task in ("cube", "reacher") else 1.0)
    axes.set_xlim(*bounds[0])
    axes.set_ylim(*bounds[1])
    axes.set_xlabel(plot_data["axis_labels"][0], labelpad=9)
    axes.set_ylabel(plot_data["axis_labels"][1], labelpad=9)
    if dimensions == 3:
        axes.set_zlim(*bounds[2])
        axes.set_zlabel(plot_data["axis_labels"][2], labelpad=9)
        axes.set_box_aspect((1, 1, 1))
        axes.view_init(elev=23, azim=-58)
    else:
        axes.set_aspect("equal", adjustable="box")
        axes.spines[["top", "right"]].set_visible(False)
    axes.grid(True, color="#DDE3E7", alpha=0.65)
    axes.tick_params(labelsize=9)
    axes.legend(handles=handles, loc="best", frameon=False, fontsize=9)
    figure.suptitle(f"{TASKS[task]} | AP-CEM {outcome}", fontsize=17, color=COLORS["text"], y=0.965)
    axes.set_title(f"Original query {case['query_ordinal']} | stopped after {case['primitive_steps']} actions",
                   fontsize=11, color=COLORS["text"], pad=15)
    figure.text(0.5, 0.067, plot_data["coordinate_note"], ha="center", fontsize=9, color=COLORS["text"])
    figure.text(0.5, 0.039,
                f"Derived from all {plot_data['recorded_states']} saved physical states, including the stop. "
                "No simulation or replay.", ha="center", fontsize=9, color=COLORS["text"])
    figure.subplots_adjust(left=0.11, right=0.91, bottom=0.18, top=0.84)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, facecolor="white",
                   metadata={"Software": "AP saved-trajectory analysis; matplotlib " + matplotlib.__version__})
    plt.close(figure)
    return matplotlib.__version__


def build(run_roots, status_path, report_id, output, *, plots=True):
    require(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", report_id), "Invalid report identifier")
    runs = load_runs(run_roots, read_json(status_path))
    cases = collect_cases(runs)
    prefix = f"media/{report_id}"
    examples, downloads, selections = [], [], []
    counts = Counter((task, arm, row["case"]["success"]) for (task, arm, _), row in cases.items())
    for task in TASKS:
        for outcome in ("success", "failure"):
            candidates = [cases[task, "gaussian_observed", ordinal] for ordinal in range(128)
                          if cases[task, "gaussian_observed", ordinal]["case"]["success"] is (outcome == "success")]
            if not candidates:
                examples.append({"environment": task, "outcome": outcome, "case": None,
                                 "reason": f"No AP-CEM {outcome} occurred in the 128 original queries."})
                continue
            selected = candidates[0]
            trajectory, trajectory_name, video_name = verify_selected(selected)
            plot_data = coordinates(task, trajectory)
            run = selected["run"]
            destination = f"{prefix}/{task}/{outcome}.mp4"
            source = f"runs/{run['run_id']}/{video_name}"
            expected = run["files"][video_name]
            example = {
                "environment": task, "outcome": outcome, "case": selected["case"],
                "source_run": run["run_id"], "source_revision": run["revision"],
                "source_trace": f"runs/{run['run_id']}/{selected['trace_name']}",
                "source_trace_sha256": run["files"][selected["trace_name"]]["sha256"],
                "source_trajectory": f"runs/{run['run_id']}/{trajectory_name}",
                "source_trajectory_sha256": run["files"][trajectory_name]["sha256"],
                "source_manifest_sha256": run["manifest_sha256"],
                "video": dict(trajectory["video"], fps=2),
                "files": [{"path": destination, "source_path": source, **expected}],
                "note": "Original planner observations at five-action intervals. "
                        "The final displayed observation may precede the recorded stop.",
            }
            downloads.append({"repo_id": REPO, "revision": run["revision"], "source_path": source,
                              "destination_path": destination, **expected,
                              "url": f"https://huggingface.co/datasets/{REPO}/resolve/{run['revision']}/"
                                     + quote(source, safe="/")})
            examples.append(example)
            selections.append((example, plot_data))

    # Validate every cohort and selected source before writing any final media.
    output = output.resolve()
    plotting_version = None
    for example, plot_data in selections:
        if not plots:
            continue
        plot_name = f"{prefix}/{example['environment']}/{example['outcome']}-trajectory.png"
        destination = output / plot_name
        require(destination.resolve().is_relative_to(output), "Plot destination escapes output")
        plotting_version = plot_trajectory(example["environment"], example["outcome"],
                                           example["case"], plot_data, destination)
        example["trajectory_plot"] = {
            "path": plot_name, "bytes": destination.stat().st_size, "sha256": digest(destination),
            "representation": "Derived analysis of saved physical states; not a video frame",
            "coordinate_note": plot_data["coordinate_note"],
            "axis_labels": plot_data["axis_labels"], "recorded_states": plot_data["recorded_states"],
            "start": {name: values[0] for name, values in plot_data["series"].items()},
            "stop": {name: values[-1] for name, values in plot_data["series"].items()},
            "goal": plot_data["goals"],
            "source_trajectory": example["source_trajectory"],
            "source_trajectory_sha256": example["source_trajectory_sha256"],
        }
    index = {
        "schema_version": 1, "experiment": report_id, "repo_id": REPO,
        "selection_verified": True,
        "selection": "First AP-CEM success and first failure in each complete 128-query cohort, "
                     "ordered by original global query ordinal across all shards.",
        "representation": "Original evaluation planner observations; not replayed or reconstructed",
        "fps": 2, "frame_action_spacing": 5,
        "stopping_frame_note": "The final video frame may precede the stopping action. "
                               "Derived trajectory plots include the recorded stopping state.",
        "cohorts": {task: {arm: {"episodes": 128, "successes": counts[task, arm, True],
                                   "unfinished": counts[task, arm, False]} for arm in ARMS} for task in TASKS},
        "source_manifests": [{key: run[key] for key in ("run_id", "revision", "manifest_sha256")} for run in runs],
        "generator": {"source_sha256": digest(Path(__file__)), "python": platform.python_version(),
                      "matplotlib": plotting_version, "simulator_actions": 0, "model_queries": 0},
        "examples": examples,
    }
    directory = output / prefix
    require(directory.resolve().is_relative_to(output), "Index destination escapes output")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
    (directory / "downloads.json").write_text(json.dumps({"repo_id": REPO, "files": downloads}, indent=2) + "\n")
    return {"index": str(directory / "index.json"), "downloads": str(directory / "downloads.json"),
            "selected_clips": len(downloads), "trajectory_plots": len(selections) if plots else 0,
            "downloaded": False, "published": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    try:
        result = build(args.run_root, args.status, args.report_id, args.output, plots=not args.no_plots)
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(2, f"Invalid saved media evidence: {error}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
