"""Post-completion scientific trajectory media from immutable saved evidence only.

Default showcase selection is repeat0 and the first admitted case of each condition
for BOTH frozen endpoints. Overlays are analysis-only and never enter policy inputs.
No model, simulator, renderer environment or fitting operation is invoked.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
import analyze_lewm_navigation_campaign_20261031_v2 as analysis

ARMS = analysis.ARMS
LABELS = analysis.LABELS
CONDITIONS = analysis.CONDITIONS
FAMILIES = ("primary50", "secondary30")
MODEL_FIELDS = (
    "actual_encoder_call_attempts",
    "actual_encoder_frames_attempted",
    "actual_cost_call_attempts",
    "actual_candidate_rollouts_attempted",
    "actual_candidate_block_steps_attempted",
    "actual_safety_call_attempts",
    "actual_safety_transition_rows_attempted",
    "goal_readout_call_attempts",
    "predictor_steps",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def choose_cases(protocols, repeat=0):
    """Metadata order only. No success, violation, action or progress field is read."""
    chosen = {}
    for family in FAMILIES:
        rows = [p for p in protocols if p["family"] == family and p["repeat_id"] == repeat]
        require(len(rows) == 1, "One fixed showcase repeat required per endpoint")
        cases = {job["case"]["case_id"]: job["case"] for job in rows[0]["protocol"]["jobs"]}
        chosen[family] = {}
        for condition in CONDITIONS:
            eligible = sorted(
                (case for case in cases.values() if case["condition"] == condition),
                key=lambda case: (case["source_episode"], case["source_step"], case["case_id"]),
            )
            require(len(eligible) == 4, "Every final condition requires four admitted cases")
            chosen[family][condition] = eligible[0]["case_id"]
    require(
        chosen[FAMILIES[0]] == chosen[FAMILIES[1]],
        "Both endpoint showcases must use the same first admitted source cases",
    )
    return chosen


def complete_inputs(args, media_plan):
    """Validate ALL completed final profiles before loading any trace/frame arrays."""
    require(
        os.environ.get("RUNPOD_POD_ID") in media_plan["allowed_pod_ids"],
        "Media computation belongs on a declared rented host",
    )
    require(
        media_plan["source_sha256"][str(Path(__file__).relative_to(ROOT))]
        == digest(Path(__file__)),
        "Media source differs from its freeze",
    )
    for name, expected in media_plan["source_sha256"].items():
        require(digest(ROOT / name) == expected, "Frozen media dependency changed: " + name)
    analysis_ref = media_plan["analysis_plan"]
    analysis_path = ROOT / analysis_ref["path"]
    require(digest(analysis_path) == analysis_ref["sha256"], "Frozen analysis contract changed")
    analysis_plan = analysis.verify_plan(analysis_path)
    root = args.artifact_root.resolve()
    inputs = load(args.inputs)
    require(
        inputs["plan"]["sha256"] == analysis_ref["sha256"]
        and inputs["all_final_output_namespaces_absent_at_binding"] is True,
        "Media requires prospectively bound analysis inputs",
    )
    readiness = analysis.joint_readiness(root, inputs["joint_readiness"]["joint_opening"])
    _, manifest = analysis.load_reference(root, inputs["final_manifest"])
    manifest_files = {
        str(analysis.resolve(root, ref["path"])): ref["sha256"] for ref in manifest["files"]
    }
    profiles, protocols = [], []
    for entry in inputs["profiles"]:
        path, protocol = analysis.load_reference(root, entry["protocol"])
        require(
            protocol["evaluation"]["body_radius"] == media_plan["plotting"]["body_radius"]
            and protocol["evaluation"]["margin"] == media_plan["plotting"]["margin"]
            and protocol["evaluation"]["motion_allowance"]
            == media_plan["plotting"]["empirical_planning_allowance"],
            "Plot geometry differs from the frozen safety/planning contract",
        )
        require(
            manifest_files.get(str(path)) == entry["protocol"]["sha256"],
            "Media profile differs from joint final manifest",
        )
        profiles.append(analysis.saved_profile(root, entry, analysis_plan))
        protocols.append({**entry, "protocol": protocol})
    require(
        len(profiles) == len(manifest["schedule"]),
        "Every frozen final profile must be complete, including optional auxiliary controls",
    )
    references = {
        str(args.inputs.resolve()): digest(args.inputs),
        str(analysis_path): analysis_ref["sha256"],
        inputs["final_manifest"]["path"]: inputs["final_manifest"]["sha256"],
    }
    _, joint = analysis.load_reference(root, readiness["joint_opening"])
    joint_refs = [readiness["joint_opening"], joint["secondary_activation"]]
    for family_key in ("primary_opening", "secondary_opening"):
        _, opening = analysis.load_reference(root, joint[family_key])
        joint_refs.extend([joint[family_key], *opening["selection_locks"]])
    references.update({ref["path"]: ref["sha256"] for ref in joint_refs})
    for entry in inputs["profiles"]:
        references[entry["protocol"]["path"]] = entry["protocol"]["sha256"]
        state_path = analysis.resolve(root, entry["state_path"])
        references[entry["state_path"]] = digest(state_path)
        references[str((state_path.parent / "source-freeze.json").relative_to(root))] = digest(
            state_path.parent / "source-freeze.json"
        )
        state = load(state_path)
        for job_id, row in state["jobs"].items():
            inner = analysis.resolve(root, row["extra"]["episode_result"])
            references[str(inner.relative_to(root))] = row["extra"]["episode_result_sha256"]
            for name in ("started.json", "result.json"):
                path = state_path.parent / "jobs" / job_id / name
                references[str(path.relative_to(root))] = digest(path)
    for family in FAMILIES:
        require(
            sorted(p["repeat_id"] for p in profiles if p["family"] == family) == [0, 1, 2],
            "Both complete three-repeat endpoint profiles required",
        )
    return root, inputs, profiles, protocols, readiness, references


def showcase_data(np, root, profiles, protocols, chosen):
    panels, metrics, references = {}, [], {}
    for family in FAMILIES:
        profile = next(p for p in profiles if p["family"] == family and p["repeat_id"] == 0)
        metadata = next(p for p in protocols if p["family"] == family and p["repeat_id"] == 0)
        state_path = analysis.resolve(root, metadata["state_path"])
        state = load(state_path)
        for condition in CONDITIONS:
            case_id = chosen[family][condition]
            for arm in ARMS:
                jobs = [
                    job
                    for job in metadata["protocol"]["jobs"]
                    if job["arm"] == arm and job["case"]["case_id"] == case_id
                ]
                require(len(jobs) == 1, "Exactly one showcase job per arm/case required")
                job = jobs[0]
                row = state["jobs"][job["job_id"]]
                inner_path = analysis.resolve(root, row["extra"]["episode_result"])
                inner = load(inner_path)
                folder = inner_path.parent
                trace_path, selector_path = folder / "trace.npz", folder / "selector-history.json"
                selector = load(selector_path)
                require(
                    digest(selector_path) == row["extra"]["decision_reconstruction_sha256"],
                    "Selected-target history differs from the retained episode",
                )
                with np.load(trace_path, allow_pickle=False) as archive:
                    data = {
                        key: archive[key].copy()
                        for key in ("states", "actions", "proposed", "original_goal")
                    }
                require(
                    data["states"].ndim == 2
                    and data["states"].shape[1] == 2
                    and len(data["states"]) == len(data["actions"]) + 1,
                    "Invalid executed trajectory shape",
                )
                require(
                    data["actions"].shape == data["proposed"].shape == (len(data["actions"]), 2),
                    "Invalid proposed/executed native command shape",
                )
                require(
                    all(np.isfinite(value).all() for value in data.values()),
                    "Nonfinite saved trajectory",
                )
                np.testing.assert_allclose(
                    data["states"][0], job["case"]["initial_xy"], atol=1e-5, rtol=0
                )
                np.testing.assert_allclose(
                    data["original_goal"], job["case"]["goal_xy"], atol=1e-5, rtol=0
                )
                np.testing.assert_array_equal(data["actions"], np.clip(data["proposed"], -1, 1))
                require(
                    inner["shape"] == job["case"]["shape"],
                    "Virtual geometry differs from frozen task",
                )
                require(
                    len(data["actions"])
                    == row["environment_actions"]
                    == inner["recorded_successful_actions"],
                    "Executed action count differs",
                )
                panel = {
                    **data,
                    "shape": job["case"]["shape"],
                    "selector": selector,
                    "case_id": case_id,
                    "source_episode": job["case"]["source_episode"],
                    "trace_path": trace_path,
                    "arm": arm,
                    "condition": condition,
                    "metrics": {
                        key: row[key]
                        for key in (
                            "safe_completion",
                            "violation",
                            "unfinished",
                            "engaged",
                            "environment_actions",
                        )
                    },
                    "model_work": {key: row["resources"].get(key, 0) for key in MODEL_FIELDS},
                    "episode_seconds": inner["episode_seconds"],
                    "planning_seconds": inner["planning_seconds"],
                    "controller_failure": inner["controller_failure"],
                }
                panels[(family, condition, arm)] = panel
                metrics.append(
                    {
                        "family": family,
                        "repeat_id": 0,
                        "condition": condition,
                        "arm": arm,
                        "case_id": case_id,
                        "source_episode": panel["source_episode"],
                        **panel["metrics"],
                        **panel["model_work"],
                        "episode_seconds": panel["episode_seconds"],
                        "planning_seconds": panel["planning_seconds"],
                        "controller_failure": panel["controller_failure"],
                    }
                )
                for path in (
                    trace_path,
                    selector_path,
                    inner_path,
                    folder.parent / "started.json",
                    folder.parent / "result.json",
                ):
                    references[str(path.relative_to(root))] = digest(path)
                references[job["case"]["input_path"]] = job["case"]["input_sha256"]
        for key in ("state_sha256", "source_freeze_sha256"):
            references[family + "/" + key] = profile[key]
    return panels, metrics, references


def render_profile(plt, np, panels, family, plotting, output):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    body, margin, allowance = [
        plotting[key] for key in ("body_radius", "margin", "empirical_planning_allowance")
    ]
    fig, axes = plt.subplots(4, 6, figsize=(16.5, 10.8), sharex=True, sharey=True)
    evidence = []
    for r, condition in enumerate(CONDITIONS):
        for c, arm in enumerate(ARMS):
            axis, panel = axes[r, c], panels[(family, condition, arm)]
            states, goal = panel["states"], panel["original_goal"]
            shape = panel["shape"]
            if shape["kind"] == "island":
                centre, radius = shape["centre"], shape["radius"]
                for rad, style, color, fill in (
                    (radius, "-", "#bd3535", True),
                    (radius + body + margin, "--", "#d58b1b", False),
                    (radius + body + margin + allowance, ":", "#7d6b91", False),
                ):
                    axis.add_patch(
                        Circle(
                            centre,
                            rad,
                            facecolor=color if fill else "none",
                            edgecolor=color,
                            alpha=0.23 if fill else 0.8,
                            linestyle=style,
                            linewidth=0.8,
                        )
                    )
            axis.add_patch(
                Circle(
                    goal,
                    plotting["original_goal_radius"],
                    facecolor="#73bf7a",
                    edgecolor="#358843",
                    alpha=0.18,
                    linewidth=0.8,
                )
            )
            axis.plot(states[:, 0], states[:, 1], color="#202020", linewidth=1.2, zorder=4)
            axis.scatter(*states[0], marker="^", color="#202020", s=18, zorder=5)
            axis.scatter(*goal, marker="*", color="#358843", s=35, zorder=5)
            axis.add_patch(
                Circle(
                    states[-1],
                    body,
                    facecolor="#4d91be",
                    edgecolor="#26658d",
                    alpha=0.35,
                    linewidth=0.8,
                    zorder=5,
                )
            )
            for query in panel["selector"]:
                point = query.get("selected_target_xy_estimate")
                if point is None:
                    continue
                observed = query["target_kind"] == "observed_successor"
                axis.scatter(
                    *point,
                    marker="x" if observed else "D",
                    color="#cc597c" if observed else "#9374ac",
                    s=12 if observed else 9,
                    linewidths=0.7,
                    alpha=0.75,
                    zorder=3,
                )
            commands = panel["proposed"]
            for index in range(0, len(commands), max(1, (len(commands) + 3) // 4)):
                direction = commands[index] * plotting["proposed_command_glyph_scale"]
                axis.annotate(
                    "",
                    xy=states[index] + direction,
                    xytext=states[index],
                    arrowprops={"arrowstyle": "->", "color": "#959595", "linewidth": 0.7},
                    zorder=2,
                )
            score = panel["metrics"]
            axis.text(
                0.02,
                0.98,
                f"SC{int(score['safe_completion'])} V{int(score['violation'])} U{int(score['unfinished'])}\nA{score['environment_actions']} WM{panel['model_work'].get('actual_cost_call_attempts', 0)}",
                transform=axis.transAxes,
                va="top",
                fontsize=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )
            axis.set(xlim=(0, 224), ylim=(0, 224), aspect="equal")
            axis.set_xticks([0, 112, 224])
            axis.set_yticks([0, 112, 224])
            axis.tick_params(labelsize=6)
            if r == 0:
                axis.set_title(LABELS[c], fontsize=9)
            if c == 0:
                axis.set_ylabel(
                    condition + f"\ne{panel['source_episode']}\nNative Y (px)", fontsize=7
                )
            if r == 3:
                axis.set_xlabel("Native X (px)", fontsize=7)
            evidence.append(
                {
                    "condition": condition,
                    "arm": arm,
                    "body_disk_radius": body,
                    "goal_radius": plotting["original_goal_radius"],
                    "margin": margin,
                    "empirical_allowance": allowance,
                    "hazard_radius": shape.get("radius"),
                    "overlay_is_analysis_only": True,
                }
            )
    handles = [
        Line2D([0], [0], color="#202020", label="Executed states/path"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="#bd3535",
            alpha=0.4,
            label="Actual virtual hazard (recorded radius)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="none",
            color="#cc597c",
            label="Selected observed target",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            color="#9374ac",
            label="Original-goal readout estimate",
        ),
        Line2D(
            [0],
            [0],
            color="#959595",
            marker=">",
            label="Proposed command glyph (8px/unit, not a state prediction)",
        ),
        Line2D([0], [0], color="#d58b1b", linestyle="--", label="Body-centre hazard margin: r+7+3"),
        Line2D(
            [0],
            [0],
            color="#7d6b91",
            linestyle=":",
            label="Predicted-centre buffer: r+7+3+12.405 (empirical)",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 0.015))
    endpoint = (
        "Primary 50-action endpoint" if family == "primary50" else "Secondary 30-action endpoint"
    )
    fig.suptitle(
        f"{endpoint}, fixed repeat 0: first admitted case in each condition. Analysis-only overlays.",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.09,
        "Original goal disc: 16 px. Blue endpoint disc: protected body 7 px. SC/V/U are frozen episode labels; A is executed actions; WM is cost-call attempts.\nLines join saved native states. The dotted empirical checker buffer applies to predicted body centres, separately from the observed 3 px margin; it is not a certified bound.",
        ha="center",
        fontsize=7,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.16, hspace=0.15, wspace=0.08)
    for extension in ("png", "svg"):
        fig.savefig(
            output / f"{family}-showcase-trajectories.{extension}", dpi=240, bbox_inches="tight"
        )
    plt.close(fig)
    return evidence


def raw_pair_video(np, left, right, path, fps):
    """Raw images are copied unchanged into an outside-labelled analysis canvas."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        return {"skipped": "ffmpeg not installed; no dependency installation attempted"}
    codecs = subprocess.run(
        [executable, "-hide_banner", "-encoders"], check=True, capture_output=True, text=True
    )
    if "libx264rgb" not in codecs.stdout:
        return {"skipped": "Installed ffmpeg lacks lossless RGB x264 encoder"}
    from PIL import Image, ImageDraw

    arrays = []
    for panel in (left, right):
        with np.load(panel["trace_path"], allow_pickle=False) as archive:
            frames = archive["frames"].copy()
        require(
            frames.dtype == np.uint8
            and frames.ndim == 4
            and frames.shape[1:] == (224, 224, 3)
            and len(frames),
            "Invalid saved raw frames",
        )
        arrays.append(frames)
    process = subprocess.Popen(
        [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "448x256",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "ultrafast",
            "-y",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for index in range(max(map(len, arrays))):
            header = Image.new("RGB", (448, 32), "black")
            text = ImageDraw.Draw(header)
            for side, panel in enumerate((left, right)):
                text.text((224 * side + 4, 2), panel["arm"] + f" | playback {fps}fps", fill="white")
                text.text(
                    (224 * side + 4, 16), "Raw frames; hold endpoint on termination", fill="white"
                )
            canvas = np.empty((256, 448, 3), dtype=np.uint8)
            canvas[:32] = np.asarray(header)
            for side in (0, 1):
                canvas[32:, 224 * side : 224 * (side + 1)] = arrays[side][
                    min(index, len(arrays[side]) - 1)
                ]
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
    require(process.wait() == 0, "Raw paired video encoder failed")
    return {
        "path": str(path),
        "raw_frame_counts": [len(frames) for frames in arrays],
        "fps": fps,
        "playback_is_not_simulation_time": True,
        "labels_outside_unmodified224x224_frames": True,
        "encoder": "libx264rgb lossless RGB",
        "last_frame_held_after_termination": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-video", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    plan = load(args.plan)
    require(
        plan["schema"] == "lewm-navigation-scientific-media-plan-v1",
        "Unknown scientific media plan",
    )
    require(
        plan["showcase"]["repeat_id"] == 0
        and plan["showcase"]["families"] == list(FAMILIES)
        and plan["showcase"]["condition_order"] == list(CONDITIONS),
        "Showcase repeat/profiles/condition ordering is prospectively fixed",
    )
    require(
        plan["plotting"]
        == {
            "body_radius": 7.0,
            "margin": 3.0,
            "empirical_planning_allowance": 12.405431747436523,
            "original_goal_radius": 16.0,
            "proposed_command_glyph_scale": 8.0,
        },
        "Media uses the unchanged goal/body/margin/allowance and explicit command-glyph scale",
    )
    require(not args.out.exists(), "Use a new scientific media output namespace")
    root, inputs, profiles, protocols, readiness, all_references = complete_inputs(args, plan)
    chosen = choose_cases(protocols)
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels, metrics, references = showcase_data(np, root, profiles, protocols, chosen)
    references.update(all_references)
    args.out.mkdir(parents=True)
    geometry, videos = {}, []
    for family in FAMILIES:
        geometry[family] = render_profile(plt, np, panels, family, plan["plotting"], args.out)
        if args.raw_video:
            for condition in CONDITIONS:
                first, second = plan["raw_video"]["fixed_pair"]
                videos.append(
                    raw_pair_video(
                        np,
                        panels[(family, condition, first)],
                        panels[(family, condition, second)],
                        args.out / f"{family}-{condition}-raw-{first}-{second}.mp4",
                        plan["raw_video"]["playback_fps"],
                    )
                )
    with (args.out / "showcase-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    report = {
        "schema": "lewm-navigation-scientific-media-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "plan_sha256": digest(args.plan),
        "analysis_inputs_sha256": digest(args.inputs),
        "joint_manifest": inputs["final_manifest"],
        "joint_readiness": readiness,
        "all_final_profiles_verified_complete_before_trace_load": True,
        "showcase_rule": plan["showcase"],
        "chosen_cases": chosen,
        "metrics": metrics,
        "geometry_annotations": geometry,
        "input_sha256": references,
        "videos": videos,
        "overlays_are_analysis_only_never_policy_inputs": True,
        "model_work_aliases_not_added_together": True,
        "new_environment_resets": 0,
        "new_environment_actions": 0,
        "new_model_queries": 0,
        "source_sha256": plan["source_sha256"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.out / "showcase-evidence.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (args.out / "artifact-manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    path.name: digest(path) for path in sorted(args.out.iterdir()) if path.is_file()
                },
                "plan_sha256": digest(args.plan),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.out),
                "showcase_cases_each_endpoint": 4,
                "arms": 6,
                "new_model_simulator_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
