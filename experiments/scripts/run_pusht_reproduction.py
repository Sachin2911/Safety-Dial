"""Execute and audit the frozen manual-preset reproduction on its declared rented pod."""

import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def identities(root, design):
    for key in ("asset_sha256", "source_sha256"):
        for name, expected in design[key].items():
            if digest(root / name) != expected:
                raise RuntimeError("Frozen input/source changed: " + name)


def report(root, design):
    import numpy as np

    rows, checks = [], []
    resources = collections.Counter()
    for arm in design["arms"]:
        for case in design["case_order"]:
            folder = root / "results" / arm["id"] / case
            result = json.loads((folder / "result.json").read_text())
            with np.load(folder / "episode/trace.npz", allow_pickle=False) as data:
                states, actions, dense = data["states"], data["actions"], data["dense"]
            goal, shape = np.asarray(result["goal_state"]), result["case"]
            xy = (dense[:, :2] - shape["origin"]) @ np.asarray(shape["basis"]).T
            if shape["kind"] == "island":
                distance = np.linalg.norm(xy, axis=-1) - shape["radius_px"]
            elif shape["kind"] == "bar":
                nearest = np.c_[
                    np.clip(
                        xy[:, 0], -shape["segment_half_length_px"], shape["segment_half_length_px"]
                    ),
                    np.zeros(len(xy)),
                ]
                distance = np.linalg.norm(xy - nearest, axis=-1) - shape["radius_px"]
            else:
                offset, half, extent = (
                    shape["gap_offset_px"],
                    shape["gap_width_px"] / 2,
                    shape["extent_px"],
                )
                distances = []
                for lo, hi in ((-extent, offset - half), (offset + half, extent)):
                    q = np.abs(xy - [(lo + hi) / 2, 0]) - [
                        (hi - lo) / 2,
                        shape["band_half_width_px"],
                    ]
                    distances.append(
                        np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
                    )
                distance = np.minimum(*distances)
            clearance = distance - 15
            arena = np.minimum(dense[:, :2], 512 - dense[:, :2]).min(-1) - 15
            violation = bool((clearance < 6 - 1e-6).any() or (arena < 6 - 1e-6).any())
            angular = abs(goal[4] - states[-1, 4])
            angular = min(angular, 2 * np.pi - angular)
            success = bool(np.linalg.norm(goal[:4] - states[-1, :4]) < 20 and angular < np.pi / 9)
            safe = (
                success
                and not violation
                and not result["error"]
                and not result["controller_failure"]
            )
            if safe != result["safe_completion"] or success != result["success"]:
                raise RuntimeError("Independent native-goal outcome mismatch")
            if violation != bool(result["margin_breach"] or result["arena_margin_breach"]):
                raise RuntimeError("Independent dense-monitor mismatch")
            if len(actions) != result["actions"] or len(actions) > 50:
                raise RuntimeError("Action accounting mismatch")
            if len(dense) != 10 * len(actions) + 1:
                raise RuntimeError("Dense physics accounting mismatch")
            if result["resources"]["executed_physics_steps"] != 10 * len(actions):
                raise RuntimeError("Physics counter mismatch")
            if result["resources"]["predictor_steps"] > 10 * (11250 + 3):
                raise RuntimeError("Manual planning budget exceeded")
            plans = json.loads((folder / "episode/replans.json").read_text())
            if any(p["horizon"] != 3 or p["samples"] != 250 for p in plans):
                raise RuntimeError("Manual horizon/population differs")
            original_name = "island-r1" if case == "island" else case
            with np.load(root / "assets/historical-traces" / original_name / "trace.npz") as old:
                n = min(len(states), len(old["states"]))
                comparison = {
                    "shared_state_samples": n,
                    "max_state_difference": float(np.abs(states[:n] - old["states"][:n]).max()),
                    "exact_state_prefix": bool(np.array_equal(states[:n], old["states"][:n])),
                    "same_completion_actions": len(actions)
                    == design["historical_results"][case]["actions"],
                }
            resources.update(result["resources"])
            rows.append(
                {
                    "arm": arm["id"],
                    "case": case,
                    "safe_completion": bool(safe),
                    "success": success,
                    "violation": violation,
                    "contact": bool((clearance <= 0).any()),
                    "unfinished": not success,
                    "engaged": result["pusher_path_length_px"] > 1,
                    "actions": len(actions),
                    "error": result["error"],
                    "controller_failure": result["controller_failure"],
                    "minimum_clearance_px": float(clearance.min()),
                    "minimum_arena_clearance_px": float(arena.min()),
                    "historical_trace_comparison": comparison,
                    "episode_seconds": result["episode_seconds"],
                    "planning_seconds": result["planning_seconds"],
                    "result_sha256": digest(folder / "result.json"),
                }
            )
            checks.append({"arm": arm["id"], "case": case, "audited": True})
    summaries = {}
    for arm in design["arm_order"]:
        selected = [r for r in rows if r["arm"] == arm]
        summaries[arm] = {
            "episodes": len(selected),
            **{
                key: sum(bool(r[key]) for r in selected)
                for key in ("safe_completion", "violation", "contact", "unfinished", "engaged")
            },
            "actions": [r["actions"] for r in selected],
            "errors": sum(bool(r["error"]) for r in selected),
        }
    identities(root, design)
    value = {
        "verified": True,
        "summaries": summaries,
        "rows": rows,
        "checks": checks,
        "resources": dict(resources),
        "limitations": design["limits"],
        "scope": "Frozen known-scene reproduction; no tuning, additional training or held-out claim.",
    }
    write(root / "results/report.json", value)
    print(
        json.dumps({"verified": True, "summaries": summaries, "resources": dict(resources)}),
        flush=True,
    )


def main():
    root = Path("/workspace/Safety-Dial")
    design = json.loads((root / "design.json").read_text())
    if os.environ.get("RUNPOD_POD_ID") != design["pod_id"]:
        raise RuntimeError("This experiment may only execute on its declared RunPod")
    if not (root / "environment/ready.json").exists():
        raise RuntimeError("Pinned environment setup has not completed")
    identities(root, design)
    write(
        root / "results/driver-freeze.json",
        {
            "design_sha256": digest(root / "design.json"),
            "driver_sha256": digest(Path(__file__)),
            "epoch": time.time(),
        },
    )
    steps = []
    for arm in design["arms"]:
        for case in design["case_order"]:
            folder = root / "results" / arm["id"] / case
            command = [
                sys.executable,
                "-u",
                str(root / arm["source_root"] / "experiments/scripts/run_safety_pusht.py"),
                "--protocol",
                str(root / "protocols" / (arm["id"] + "-" + case + ".json")),
                "--out",
                str(folder),
                "--device",
                arm["device"],
                "--threads",
                "4",
            ]
            log = root / "results/logs" / (arm["id"] + "-" + case + ".log")
            log.parent.mkdir(exist_ok=True)
            row = {"arm": arm["id"], "case": case, "status": "running", "epoch": time.time()}
            steps.append(row)
            write(root / "results/pipeline.json", steps)
            with log.open("wb") as output:
                completed = subprocess.run(
                    command,
                    cwd=root / arm["source_root"],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=420,
                    check=False,
                )
            row.update(
                status="complete" if completed.returncode == 0 else "failed",
                exit_code=completed.returncode,
                elapsed_seconds=time.time() - row["epoch"],
            )
            write(root / "results/pipeline.json", steps)
            print(json.dumps(row), flush=True)
            if not (folder / "result.json").exists():
                raise RuntimeError(
                    "Episode terminated without final accounting; preserve and inspect"
                )
    report(root, design)


if __name__ == "__main__":
    main()
