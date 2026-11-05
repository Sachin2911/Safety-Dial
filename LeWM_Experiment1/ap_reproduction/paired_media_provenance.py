#!/usr/bin/env python3
"""Describe verified AP/control video pairs using saved evidence only."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from audit import ARMS, MAIN_SHA256, TASKS, digest, optimizer_seed, read_json


def build(base, status_path, protocol_path, final=False):
    errors, cases = [], defaultdict(dict)

    def require(condition, message):
        if not condition:
            errors.append(message)

    status, protocol = read_json(status_path), read_json(protocol_path)
    require(digest(protocol_path) == MAIN_SHA256, "Original query protocol hash differs")
    proofs = {row["run_id"]: row for row in status["runs"]}
    expected = {
        (task, ordinal): row
        for task in TASKS
        for ordinal, row in enumerate(protocol["studies"][task]["confirm"])
    }
    for root in sorted(path.parent for path in base.glob("*/run.json")):
        paths = sorted((root / "metrics/cases").glob("*.json"))
        if not paths:
            continue
        record = proofs.get(root.name, {})
        upload = record.get("upload") or {}
        require(
            record.get("phase") == "finished"
            and upload.get("verified") is True
            and not upload.get("error")
            and not record.get("job_alive")
            and not record.get("supervisor_alive"),
            root.name + ": unverified/live source",
        )
        manifest = read_json(root / "manifest.json")
        require(
            digest(root / "manifest.json") == upload.get("manifest_sha256"),
            root.name + ": pinned manifest differs",
        )
        require(manifest.get("run_id") == root.name, root.name + ": manifest identity differs")

        def artifact(relative, local_required):
            metadata = manifest["files"].get(relative)
            require(
                isinstance(metadata, dict) and metadata.get("bytes", 0) > 0,
                root.name + ": missing/empty artifact " + relative,
            )
            if metadata is None:
                return {"path": relative, "local_path": str(root / relative)}
            path = root / relative
            if local_required or path.exists():
                require(
                    path.is_file()
                    and path.stat().st_size == metadata["bytes"]
                    and digest(path) == metadata["sha256"],
                    root.name + ": local artifact differs " + relative,
                )
            return {"path": relative, "local_path": str(path), **metadata}

        for path in paths:
            native = read_json(path)
            compact = native["compact"]
            task, ordinal, arm = compact["task"], compact["query_ordinal"], compact["arm"]
            key = task, ordinal
            require(key in expected and arm in ARMS, path.name + ": unknown original case")
            if key not in expected or arm not in ARMS:
                continue
            require(arm not in cases[key], path.name + ": repeated scientific identity")
            if arm in cases[key]:
                continue
            require(
                compact["identity"] == expected[key] and compact["condition"] == "clean",
                path.name + ": original identity/condition differs",
            )
            require(
                compact["optimizer_seed"] == optimizer_seed(task, ordinal),
                path.name + ": optimizer seed differs",
            )
            require(compact.get("completed") is True, path.name + ": native case incomplete")
            case_file = artifact("metrics/cases/" + path.name, True)
            trajectory_file = artifact("metrics/trajectories/" + path.stem + ".json.gz", True)
            video_file = artifact("artifacts/videos/" + path.stem + ".mp4", False)
            trajectory = read_json(Path(trajectory_file["local_path"]))
            initial, steps = trajectory["initial"], trajectory["steps"]
            require(
                initial["identity"] == compact["identity"]
                and initial["query_ordinal"] == ordinal
                and initial["arm"] == arm,
                path.name + ": trajectory identity differs",
            )
            require(
                [step["action"] for step in steps] == native["actions"]
                and len(steps) == compact["primitive_steps"],
                path.name + ": stopping/action sequence differs",
            )
            rendered = trajectory["render_hashes"]
            frame_steps = [frame["action_index"] for frame in rendered]
            expected_frames = list(range(0, len(steps), 5)) or [0]
            video = trajectory["video"]
            require(
                frame_steps == expected_frames == video["frame_steps"]
                and len(frame_steps) == video["frames"],
                path.name + ": native frame/action alignment differs",
            )
            require(
                video["capture"] == "native planner render calls only",
                path.name + ": unexpected video capture policy",
            )
            initial_state = {
                name: initial[name] for name in ("observation", "snapshot", "true_goal")
            }
            initial_state_hash = hashlib.sha256(
                json.dumps(
                    initial_state, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
            cases[key][arm] = {
                "initial": initial_state,
                "initial_render_sha256": rendered[0]["sha256"] if rendered else None,
                "goal_rgb_sha256": initial["goal_rgb_sha256"],
                "record": {
                    "run_id": root.name,
                    "repository": "Sachioster/LeWMExp1_AP_Reproduction",
                    "revision": upload.get("revision"),
                    "manifest_sha256": upload.get("manifest_sha256"),
                    "case": case_file,
                    "trajectory": trajectory_file,
                    "video": video_file,
                    "compact": compact,
                    "frame_steps": frame_steps,
                    "video_frames": video["frames"],
                    "video_fps": 2,
                    "stop_action": len(steps),
                    "stop_reason": compact["stop_reason"],
                    "success": bool(compact["success"]),
                    "last_frame_action": frame_steps[-1] if frame_steps else None,
                    "last_frame_precedes_stop": bool(frame_steps) and frame_steps[-1] < len(steps),
                    "initial_state_sha256": initial_state_hash,
                },
            }
    pairs, incomplete = [], []
    for (task, ordinal), row in sorted(expected.items()):
        pair = cases.get((task, ordinal), {})
        if set(pair) != set(ARMS):
            incomplete.append(
                {
                    "task": task,
                    "query_ordinal": ordinal,
                    "record_id": row["record_id"],
                    "available_arms": sorted(pair),
                }
            )
            continue
        control, ap = (pair[arm] for arm in ARMS)
        equality = {
            name: control["initial"][name] == ap["initial"][name]
            for name in ("observation", "snapshot", "true_goal")
        }
        equality["initial_observation"] = equality.pop("observation")
        equality["initial_snapshot"] = equality.pop("snapshot")
        for name in ("identity", "optimizer_seed"):
            equality[name] = control["record"]["compact"][name] == ap["record"]["compact"][name]
        for name in ("initial_render_sha256", "goal_rgb_sha256"):
            equality[name] = control[name] == ap[name]
        require(all(equality.values()), f"{task}/{ordinal}: paired starting evidence differs")
        arms = {arm: pair[arm]["record"] for arm in ARMS}
        pairs.append(
            {
                "task": task,
                "query_ordinal": ordinal,
                "record_id": row["record_id"],
                "identity": row,
                "condition": "clean",
                "optimizer_seed": optimizer_seed(task, ordinal),
                "equality": equality,
                "same_seed": equality["optimizer_seed"],
                "same_state": equality["initial_observation"] and equality["initial_snapshot"],
                "same_goal": equality["true_goal"] and equality["goal_rgb_sha256"],
                "same_start_image": equality["initial_render_sha256"],
                "initial_render_sha256": ap["initial_render_sha256"],
                "goal_rgb_sha256": ap["goal_rgb_sha256"],
                "arms": arms,
                "aligned_frame_action_indices": sorted(
                    set(control["record"]["frame_steps"] + ap["record"]["frame_steps"])
                ),
                "maximum_stop_action": max(arm["stop_action"] for arm in arms.values()),
            }
        )
    if final:
        require(len(pairs) == 512 and not incomplete, "Final paired cohort is not all 512 queries")
    return {
        "schema": "lewm-ap-paired-media-provenance-v1",
        "passed": not errors,
        "errors": errors,
        "final_requested": final,
        "expected_pairs": 512,
        "pairs": pairs,
        "missing_pairs": incomplete,
        "frame_alignment": "Shared executed action index; hold the latest recorded frame "
        "without interpolation and label each arm's stopping action.",
        "limitations": [
            "Starting image equality refers to raw planner-render hashes, not "
            "the labelled/lossy video pixels.",
            "The final saved observation can precede the stopping action; "
            "no terminal image is inferred or rendered.",
        ],
        "verifier": {
            "path": "LeWM_Experiment1/ap_reproduction/paired_media_provenance.py",
            "sha256": digest(Path(__file__)),
        },
        "inputs": {
            "run_root": str(base),
            "status": str(status_path),
            "status_sha256": digest(status_path),
            "protocol": str(protocol_path),
            "protocol_sha256": digest(protocol_path),
        },
    }


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=project / "runs/AP")
    parser.add_argument(
        "--status",
        type=Path,
        default=project / "computeConfigs/records/ap_reproduction/combined-session-status.json",
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    report = build(args.run_root, args.status, args.protocol, args.final)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "pairs": len(report["pairs"]),
                "errors": report["errors"],
                "missing_pairs": len(report["missing_pairs"]),
            }
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
