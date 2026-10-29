"""Task-17 environment identities and job construction. No historical launches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/lewm17"
CONFIG = ROOT / "configs/lewm17"
ENVIRONMENTS = ("tworoom", "reacher")
CANONICAL = {"tworoom": {"samples": 1000, "elite": 0.01}, "reacher": {"samples": 300, "elite": 0.1}}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_spec(env):
    return json.loads((CONFIG / f"{env}.json").read_text())


def boxes(env, case, spec=None):
    spec = spec or load_spec(env)
    hz = spec["hazard"]
    x = case["gap_center_x"]
    half = case["gap_width"] / 2
    y0, y1 = hz["y_band"]
    return [(hz["left_edge"], x - half, y0, y1), (x + half, hz["right_end"], y0, y1)]


def hits(xy, hazard_boxes, margin=0):
    xy = np.asarray(xy).reshape(-1, 2)
    result = np.zeros(len(xy), dtype=bool)
    for x0, x1, y0, y1 in hazard_boxes:
        result |= (
            (xy[:, 0] >= x0 - margin)
            & (xy[:, 0] <= x1 + margin)
            & (xy[:, 1] >= y0 - margin)
            & (xy[:, 1] <= y1 + margin)
        )
    return result


def job(env, stage, case, margin, seed, condition, params=None):
    p = {
        "horizon": 5,
        "variance": 1.0,
        "retain": 0.0,
        "correlate": 0.0,
        "budget_scale": 1.0,
        "adaptive": False,
        "objective": "latent",
        "checked": True,
    }
    if condition == "native":
        p["checked"] = False
    elif condition == "short":
        p["horizon"] = 3
    elif condition == "memory":
        p.update(retain=0.25, correlate=0.5)
    elif condition == "larger":
        p["budget_scale"] = 2.0
    elif condition == "geometric":
        p["objective"] = "geometric"
    if params:
        p.update(params)
    prefix = f"{stage}-{env}-{case['id']}-d{margin:g}-s{seed}-{condition}"
    return {
        "id": prefix,
        "env": env,
        "stage": stage,
        "case": case,
        "margin": margin,
        "seed": int(seed),
        "condition": condition,
        "params": p,
        "block": f"{env}-{case['id']}-d{margin:g}-s{seed}",
    }


def freeze(name, jobs=None, extra=None):
    import datetime as dt
    import tarfile

    path = ROOT / f"progress/receipts/17-{name}.json"
    if path.exists():
        raise FileExistsError(path)
    paths = sorted(
        set(ROOT.glob("experiments/helpers/lewm17*.py"))
        | set(ROOT.glob("experiments/scripts/*lewm17*"))
        | set(ROOT.glob("experiments/tests/test_lewm17*.py"))
        | set(CONFIG.rglob("*.json"))
    )
    borrowed = [
        "safecem10_planner.py",
        "tworoom12_planner.py",
        "tworoom12_env.py",
        "reacher13_planner.py",
        "reacher13_env.py",
        "reacher13_geometry.py",
    ]
    paths += [ROOT / "experiments/helpers" / x for x in borrowed]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths if p.is_file()}
    archive = OUT / "source" / f"{name}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for relative in hashes:
            tar.add(ROOT / relative, arcname=relative)
    write_json(
        path,
        {
            "schema": "lewm17-freeze-v1",
            "name": name,
            "frozen_utc": dt.datetime.now(dt.UTC).isoformat(),
            "jobs": jobs or [],
            "source_sha256": hashes,
            "archive_sha256": digest(archive),
            "extra": extra or {},
        },
    )
    return path


def initialize_panels():
    """Geometry-only acceptance, disjoint RNG streams, no controller outcomes."""
    from helpers.reacher13_geometry import fingertip

    if (CONFIG / "panels.json").exists():
        raise FileExistsError("Do not regenerate task-17 panels")
    all_panels = {}
    for phase, size, offset in [("development", 12, 0), ("validation", 6, 1), ("final", 20, 2)]:
        panels = {}
        rng = np.random.default_rng(202610261700 + offset)
        for env in ENVIRONMENTS:
            spec = load_spec(env)
            cases = []
            while len(cases) < size:
                if env == "tworoom":
                    start = [float(rng.uniform(150, 180)), float(rng.uniform(112, 124))]
                    goal = [float(rng.uniform(148, 180)), float(rng.uniform(165, 181))]
                    route = start[0] + (145 - start[1]) / (goal[1] - start[1]) * (
                        goal[0] - start[0]
                    )
                    center, width, margin = float(route + rng.uniform(-12, 12)), 24.0, 6.0
                    endpoints = [start, goal]
                else:
                    h, elbow = float(rng.uniform(0.6, 0.8)), float(rng.uniform(0.85, 0.95))
                    start, goal = [h, elbow, 0.0, 0.0], [-h, elbow]
                    center, width, margin = float(rng.uniform(0.182, 0.207)), 0.05, 0.015
                    endpoints = fingertip(
                        np.array([start[:2], goal]), spec["environment"]["link_lengths"]
                    )
                case = {
                    "id": f"t17-{phase}-{env}-{len(cases):02d}",
                    "start": start,
                    "goal": goal,
                    "gap_center_x": center,
                    "gap_width": width,
                }
                if not hits(endpoints, boxes(env, case), margin).any():
                    cases.append(case)
            panels[env] = cases
        target = CONFIG / ("sealed/final.json" if phase == "final" else f"{phase}.json")
        write_json(target, panels)
        all_panels[phase] = {
            "path": str(target.relative_to(ROOT)),
            "sha256": digest(target),
            "generator_seed": 202610261700 + offset,
            "cases_per_environment": size,
        }
    write_json(
        CONFIG / "panels.json",
        {
            "schema": "lewm17-panels-v1",
            "panels": all_panels,
            "acceptance": "Geometry only, at maximum clearance",
        },
    )
