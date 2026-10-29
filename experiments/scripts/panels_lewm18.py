"""Select disjoint dataset episodes using geometry only, before any task-18 outcomes."""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc

OUT = ROOT / "outputs/lewm18"
CONFIG = ROOT / "configs/lewm18"


def local_hits(p, origin, basis, boxes, margin):
    xy = (np.asarray(p)[:2] - origin) @ basis.T
    return any(
        x0 - margin <= xy[0] <= x1 + margin and y0 - margin <= xy[1] <= y1 + margin
        for x0, x1, y0, y1 in boxes
    )


def make_panels(kind, protocol):
    prep = json.loads((OUT / f"{kind}-prepared.json").read_text())
    rng = np.random.default_rng(protocol["panel_seed"] + (kind == "cube"))
    rejected = collections.Counter()
    with h5py.File(prep["dataset"], "r") as f:
        episode_key = "ep_idx" if kind == "cube" else "episode_idx"
        ep = np.asarray(f[episode_key][:]).reshape(-1)
        valid = np.flatnonzero(ep[:-25] == ep[25:])
        if kind == "cube":
            cube = np.asarray(f["privileged_block_0_pos"][:])
            effector = np.asarray(f["proprio_effector_pos"][:])
            distances = np.linalg.norm(cube[valid + 25, :2] - cube[valid, :2], axis=1)
            valid = valid[distances >= 0.08]
        else:
            states = np.asarray(f["state"][:])
            distances = np.linalg.norm(states[valid + 25, 2:4] - states[valid, 2:4], axis=1)
            valid = valid[distances >= 20]
        chosen = []
        episodes = set()
        geometry = protocol["geometry"][kind]
        for a in rng.permutation(valid):
            b = a + 25
            if int(ep[a]) in episodes:
                rejected["episode_already_selected"] += 1
                continue
            if kind == "cube":
                start, goal = cube[a, :2], cube[b, :2]
                monitored_start, monitored_goal = effector[a, :2], effector[b, :2]
            else:
                start, goal = states[a, :2], states[b, :2]
                monitored_start, monitored_goal = start, goal
                if np.linalg.norm(states[a, :4] - states[b, :4]) < 20:
                    rejected["initially_solved_position"] += 1
                    continue
            direction = goal - start
            length = float(np.linalg.norm(direction))
            if length < (0.06 if kind == "cube" else 50):
                rejected["short_monitored_path"] += 1
                continue
            direction = direction / length
            basis = np.stack([np.array([direction[1], -direction[0]]), direction])
            origin = (start + goal) / 2
            offset = geometry["gap_offset_abs"] * (-1 if rng.random() < 0.5 else 1)
            half = geometry["gap_width"] / 2
            width, extent = geometry["band_half_width"], geometry["pillar_extent"]
            boxes = [
                [-extent, offset - half, -width, width],
                [offset + half, extent, -width, width],
            ]
            margin = max(protocol["dials"][kind])
            if any(
                local_hits(p, origin, basis, boxes, margin)
                for p in (monitored_start, monitored_goal, start, goal)
            ):
                rejected["endpoint_in_inflated_hazard"] += 1
                continue
            chosen.append(
                {
                    "id": f"{kind}-{len(chosen):02d}",
                    "start_row": int(a),
                    "goal_row": int(b),
                    "episode": int(ep[a]),
                    "origin": origin.tolist(),
                    "basis": basis.tolist(),
                    "boxes": boxes,
                    "object_displacement": float(
                        np.linalg.norm(
                            cube[b] - cube[a] if kind == "cube" else states[b, 2:4] - states[a, 2:4]
                        )
                    ),
                }
            )
            episodes.add(int(ep[a]))
            if len(chosen) == 24:
                break
        if len(chosen) != 24:
            raise RuntimeError(f"Not enough geometry-valid cases for {kind}")
        context = []
        # An unfiltered dataset-pair reference, independent of moving-object admission.
        for a in rng.permutation(np.flatnonzero(ep[:-25] == ep[25:])):
            if int(ep[a]) in episodes:
                continue
            context.append(
                {
                    "id": f"{kind}-context-{len(context):02d}",
                    "start_row": int(a),
                    "goal_row": int(a + 25),
                    "episode": int(ep[a]),
                    "origin": [0, 0],
                    "basis": [[1, 0], [0, 1]],
                    "boxes": [],
                }
            )
            episodes.add(int(ep[a]))
            if len(context) == 12:
                break
    return {
        "engineering": chosen[:6],
        "development": chosen[6:12],
        "final": chosen[12:],
        "benchmark_context": context,
        "rejections": dict(rejected),
        "valid_moving_pairs": int(len(valid)),
    }


def freeze(name, jobs, extra=None):
    files = sorted((ROOT / "experiments/helpers").glob("lewm18*.py"))
    files += sorted((ROOT / "experiments/scripts").glob("*lewm18.py"))
    files += [
        ROOT / "experiments/helpers/lewm16_planner.py",
        ROOT / "experiments/helpers/cube14_env.py",
        ROOT / "experiments/helpers/cube14_planner.py",
        ROOT / "experiments/helpers/cube14_runtime.py",
        ROOT / "experiments/helpers/safecem10_planner.py",
        CONFIG / "protocol.json",
    ]
    source = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    obj = {
        "schema": "lewm18-stage-v1",
        "utc": utc(),
        "name": name,
        "jobs": jobs,
        "source_sha256": source,
        "extra": extra or {},
    }
    path = OUT / f"{name}-freeze.json"
    if path.exists():
        raise FileExistsError(path)
    archive = OUT / "source" / f"{name}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as bundle:
        for file in files:
            bundle.add(file, arcname=str(file.relative_to(ROOT)))
    obj["source_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    atomic_json(path, obj)
    return path


def params(mode=None, checked=True, adaptive=False, **extra):
    h, spread, retain, correlate = mode or [5, 1, 0, 0]
    return {
        "checked": checked,
        "objective": "learned",
        "adaptive": adaptive,
        "horizon": h,
        "variance": spread,
        "retain": retain,
        "correlate": correlate,
        "iterations": 30,
        **extra,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=("cube", "pusht"))
    args = parser.parse_args()
    protocol = json.loads((CONFIG / "protocol.json").read_text())
    for kind in [args.env] if args.env else ("cube", "pusht"):
        path = OUT / f"{kind}-panels.json"
        if path.exists():
            raise FileExistsError("Panels must not be regenerated")
        panel = make_panels(kind, protocol)
        atomic_json(path, panel)
        jobs = []
        for i, case in enumerate(panel["engineering"][:4]):
            for method, mode in [("native", None), ("canonical", None)]:
                jobs.append(
                    {
                        "id": f"E-{kind}-{i}-{method}",
                        "stage": "E",
                        "env": kind,
                        "case": case,
                        "seed": 180000 + i,
                        "method": method,
                        "margin": protocol["dials"][kind][1],
                        "params": params(mode, checked=method != "native"),
                        "save_frames": i == 0,
                        "native_parity": i == 0 and method == "native",
                    }
                )
        freeze(f"engineering-{kind}", jobs)
        print(
            json.dumps({"environment": kind, "cases": 24, "engineering_jobs": len(jobs)}),
            flush=True,
        )


if __name__ == "__main__":
    main()
