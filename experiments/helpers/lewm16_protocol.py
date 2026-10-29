"""Task-16 immutable plans and cases. No simulator or provider calls."""

from __future__ import annotations

import hashlib
import json
import datetime as dt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/lewm16"
CONFIG = ROOT / "configs/lewm16"
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


def initialize():
    """Freeze models/scalers and reserve fresh cases before any task-16 outcomes."""
    CONFIG.mkdir(parents=True, exist_ok=True)
    if (CONFIG / "panels.json").exists():
        raise FileExistsError("Panels already exist; do not regenerate reserved cases")
    panels = {
        "schema": "lewm16-panels-v1",
        "generator_seed": 2026102616,
        "familiar": {},
        "fresh": {},
    }
    rng = np.random.default_rng(panels["generator_seed"])
    for env, old in [("tworoom", "tworoom12"), ("reacher", "reacher13")]:
        spec = json.loads((ROOT / f"configs/{old}/design.json").read_text())
        prep = json.loads((ROOT / f"outputs/{old}/remote/prepare-complete.json").read_text())
        # Exact prior serialized scaler bytes reconstruct the recorded hash.
        scaler_bytes = (json.dumps(prep["scalers"], indent=2) + "\n").encode()
        assert hashlib.sha256(scaler_bytes).hexdigest() == prep["scalers_sha256"]
        (CONFIG / f"{env}-scalers.json").write_bytes(scaler_bytes)
        current = {k: spec[k] for k in ("pins", "environment", "hazard")}
        current.update(
            schema="lewm16-environment-v1",
            environment_name=env,
            scaler_sha256=prep["scalers_sha256"],
            provenance=f"configs/{old}/design.json",
        )
        write_json(CONFIG / f"{env}.json", current)
        e = spec["environment"]
        base_start = e["start_agent"] if env == "tworoom" else e["start_qpos"] + [0.0, 0.0]
        base_goal = e["target"] if env == "tworoom" else e["target_qpos"]
        panels["familiar"][env] = [
            dict(
                id=x["id"],
                start=base_start,
                goal=base_goal,
                gap_center_x=x["gap_center_x"],
                gap_width=x["gap_width"],
            )
            for x in spec["hazard"]["layouts"]
        ]
        fresh = []
        while len(fresh) < 20:
            if env == "tworoom":
                start = [float(rng.uniform(150, 180)), float(rng.uniform(112, 124))]
                goal = [float(rng.uniform(148, 180)), float(rng.uniform(165, 181))]
                route_x = start[0] + (145 - start[1]) / (goal[1] - start[1]) * (goal[0] - start[0])
                center, width, margin = float(route_x + rng.uniform(-12, 12)), 24.0, 6.0
                endpoints = [start, goal]
            else:
                from helpers.reacher13_geometry import fingertip

                h, elbow = float(rng.uniform(0.6, 0.8)), float(rng.uniform(0.85, 0.95))
                start, goal = [h, elbow, 0.0, 0.0], [-h, elbow]
                center, width, margin = float(rng.uniform(0.182, 0.207)), 0.05, 0.015
                endpoints = fingertip(np.asarray([start[:2], goal]), e["link_lengths"])
            case = dict(
                id=f"new{len(fresh):02d}",
                start=start,
                goal=goal,
                gap_center_x=center,
                gap_width=width,
            )
            if hits(endpoints, boxes(env, case, current), margin).any():
                continue
            fresh.append(case)
        panels["fresh"][env] = fresh
    write_json(CONFIG / "panels.json", panels)
    write_json(
        CONFIG / "design.json",
        {
            "schema": "lewm16-design-v1",
            "namespace": "lewm16",
            "max_env_actions": 160000,
            "episode_actions": 50,
            "executed_block": 5,
            "iterations": 10,
            "monitors": ["environment_samples", "all_executed_physics_samples"],
            "numerical_guard": {"tworoom": 0.001, "reacher": 0.000001},
            "canonical": CANONICAL,
            "panel_sha256": digest(CONFIG / "panels.json"),
            "fresh_cases_sealed": True,
            "readout": None,
            "controller_inputs": [
                "latent_goal_cost",
                "relative_cost_improvement",
                "observed_prediction_error",
                "previous_feasible_fraction",
            ],
        },
    )


def job(env, stage, case, margin, seed, condition, params=None):
    p = dict(
        horizon=5,
        variance=1.0,
        retain=0.0,
        correlate=0.0,
        budget_scale=1.0,
        adaptive=False,
        objective="latent",
        checked=True,
    )
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
    return dict(
        id=prefix,
        env=env,
        stage=stage,
        case=case,
        margin=margin,
        seed=int(seed),
        condition=condition,
        params=p,
        block=f"{env}-{case['id']}-d{margin:g}-s{seed}",
    )


def initial_jobs(stage="Q"):
    panels = json.loads((CONFIG / "panels.json").read_text())
    jobs = []
    for env in ENVIRONMENTS:
        margins = [0.0, 6.0] if env == "tworoom" else [0.0, 0.015]
        cases = panels["familiar"][env]
        for case in cases:
            for margin in margins:
                for seed in range(161000, 161003):
                    for condition in [
                        "native",
                        "canonical",
                        "short",
                        "memory",
                        "larger",
                        "geometric",
                    ]:
                        jobs.append(job(env, stage, case, margin, seed, condition))
    return jobs


def freeze(name, jobs, extra=None):
    """An immutable job manifest; its complete source set is recorded before launch."""
    path = ROOT / f"progress/receipts/16-{name}.json"
    if path.exists():
        raise FileExistsError(path)
    paths = sorted(
        set(ROOT.glob("experiments/helpers/lewm16*.py"))
        | set(ROOT.glob("experiments/scripts/*lewm16*"))
        | set(ROOT.glob("experiments/tests/test_lewm16*.py"))
        | set(CONFIG.glob("*"))
    )
    borrowed = [
        "safecem10_planner.py",
        "tworoom12_planner.py",
        "tworoom12_env.py",
        "reacher13_planner.py",
        "reacher13_env.py",
        "reacher13_geometry.py",
        "dsrl06_infra.py",
    ]
    paths += [ROOT / "experiments/helpers" / x for x in borrowed]
    write_json(
        path,
        {
            "schema": "lewm16-freeze-v1",
            "name": name,
            "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "jobs": jobs,
            "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths if p.is_file()},
            "extra": extra or {},
        },
    )
    return path
