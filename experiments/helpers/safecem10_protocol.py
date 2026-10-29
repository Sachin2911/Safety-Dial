"""Task-10 immutable schedules, job construction, admission and freeze verification."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from helpers.safecem10_evo import GENES, clearance_schedule, decode, encode, grid_points

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def design():
    return json.loads((ROOT / "configs/safecem10/design.json").read_text())


def job_id(stage, layout, seed, tag):
    return f"{stage}-{layout}-{seed}-{tag}"


def planner_params(spec, **overrides):
    base = dict(spec["planner"])
    base.update(overrides)
    return base


def make_job(stage, layout, seed, mode, *, spec, clearances=None, lam=0.0, tag=None, **extra):
    params = planner_params(spec, **{k: v for k, v in extra.items() if k in spec["planner"]})
    if clearances is None:
        clearances = [0.0] * int(params["horizon"])
    clearances = [float(x) for x in clearances]
    if len(clearances) != int(params["horizon"]):
        raise ValueError("One clearance per predicted block")
    if tag is None:
        if mode == "penalty":
            tag = f"lam{lam:g}"
        else:
            tag = f"d{clearances[0]:g}"
    job = {
        "id": job_id(stage, layout, seed, tag),
        "stage": stage,
        "layout": layout,
        "seed": int(seed),
        "mode": mode,
        "lam": float(lam),
        "clearances": clearances,
        "planner": params,
    }
    job.update({k: v for k, v in extra.items() if k not in spec["planner"]})
    return job


def quantile_dials(imagination, spec, stage="B"):
    """Dials in uncertainty units: radial probe error quantiles at the prediction depth."""
    out = []
    grid = spec["stages"][stage]["dials"]
    for q in spec["imagination"]["quantiles"]:
        value = round(float(imagination["radial_quantiles_px"][str(q)]), 0)
        if value > 80 or any(abs(value - g) < 2 for g in grid):
            continue
        out.append({"quantile": q, "dial": value})
    return out


def schedule_A(spec):
    jobs = []
    stage = spec["stages"]["A"]
    for layout in spec["gate"]["layouts"]:
        for seed in stage["seeds"]:
            jobs.append(make_job("A", layout["id"], seed, "penalty", spec=spec, lam=0.0))
    first = stage["seeds"][0]
    for d in (0.0, 40.0):
        jobs.append(
            make_job(
                "A",
                "G0",
                first,
                "safe",
                spec=spec,
                clearances=[d] * spec["planner"]["horizon"],
            )
        )
    return jobs


def schedule_B(spec, imagination):
    stage = spec["stages"]["B"]
    jobs = []
    horizon = spec["planner"]["horizon"]
    dials = [{"dial": float(d), "quantile": None} for d in stage["dials"]]
    dials += quantile_dials(imagination, spec)
    for layout in stage["layouts"]:
        for seed in stage["seeds"]:
            for entry in dials:
                job = make_job(
                    "B",
                    layout,
                    seed,
                    "safe",
                    spec=spec,
                    clearances=[entry["dial"]] * horizon,
                    tag=f"d{entry['dial']:g}"
                    if entry["quantile"] is None
                    else f"q{entry['quantile']}",
                )
                job["quantile"] = entry["quantile"]
                jobs.append(job)
            for lam in stage["penalty_lams"]:
                jobs.append(make_job("B", layout, seed, "penalty", spec=spec, lam=lam))
    rng = random.Random(2026102510)
    rng.shuffle(jobs)
    return jobs


def schedule_B2(spec, imagination_v2):
    """Stage B2: the grounded checks on the stage B seeds, paired with stage B."""
    stage = spec["stages"]["B2"]
    horizon = spec["planner"]["horizon"]
    quantiles = quantile_dials(imagination_v2, spec, stage="B2")
    jobs = []
    for cond in stage["conditions"]:
        pipeline = {
            "check": cond["check"],
            "probe_version": cond["probe_version"],
            "execute": cond["execute"],
        }
        for layout in cond["layouts"]:
            for seed in stage["seeds"]:
                if cond["mode"] == "penalty":
                    job = make_job(
                        "B2",
                        layout,
                        seed,
                        "penalty",
                        spec=spec,
                        lam=cond["lam"],
                        tag=f"{cond['name']}-lam{cond['lam']:g}",
                        **pipeline,
                    )
                    job.update({"condition": cond["name"], "quantile": None})
                    jobs.append(job)
                    continue
                dials = [{"dial": float(v), "quantile": None} for v in stage["dials"]]
                if cond["dials"] == "grid+quantiles":
                    dials += quantiles
                for entry in dials:
                    tag = (
                        f"{cond['name']}-d{entry['dial']:g}"
                        if entry["quantile"] is None
                        else f"{cond['name']}-q{entry['quantile']}"
                    )
                    job = make_job(
                        "B2",
                        layout,
                        seed,
                        "safe",
                        spec=spec,
                        clearances=[entry["dial"]] * horizon,
                        tag=tag,
                        **pipeline,
                    )
                    job.update({"condition": cond["name"], "quantile": entry["quantile"]})
                    jobs.append(job)
    rng = random.Random(2026102511)
    rng.shuffle(jobs)
    return jobs


def stage_pipeline(spec, stage):
    """Planner overrides that define a search stage's pipeline (empty for stage C)."""
    return dict(spec["stages"].get(stage, {}).get("pipeline", {}))


def candidate_job(stage, layout, seed, x, spec, tag, extra):
    """One episode for one genotype on one seed; the planner is fully determined by x."""
    genes = spec["stages"]["C"]["genes"]
    params = decode(x, genes)
    topk = max(2, int(round(params["topk_frac"] * spec["planner"]["num_samples"])))
    job = make_job(
        stage,
        layout,
        seed,
        "safe",
        spec=spec,
        clearances=clearance_schedule(params),
        tag=tag,
        horizon=params["horizon"],
        topk=topk,
        feasible_tol=params["tol"],
        **stage_pipeline(spec, stage),
    )
    job["genotype"] = [float(v) for v in np.asarray(x)]
    job["params"] = params
    job.update(extra)
    return job


def evaluation_seeds(spec, method, replicate, generation):
    """Seeds shared by every method for the same replicate and generation (common random
    numbers), disjoint across generations and replicates, and never a final seed."""
    if method not in spec["stages"]["C"]["methods"]:
        raise ValueError("Unknown search method")
    base = spec["stages"]["C"]["search_seed_base"]
    n = spec["stages"]["C"]["eval_seeds_per_candidate"]
    start = base + 30 * replicate + n * generation
    seeds = list(range(start, start + n))
    if set(seeds) & set(spec["stages"]["C"]["final_seeds"]):
        raise ValueError("Search seeds collide with final seeds")
    return seeds


def aggregate(results):
    """Objectives of one genotype from its episode results (lower is better for both)."""
    return [
        float(np.mean([r["exposure"] for r in results])),
        float(np.mean([r["J"] for r in results])),
    ]


def search_budget(spec, generations):
    population = spec["stages"]["C"]["population"]
    return population * generations * spec["stages"]["C"]["eval_seeds_per_candidate"]


def grid_for_budget(spec, candidates):
    """The frozen 96-point grid, subsampled deterministically to the candidate budget."""
    genes = spec["stages"]["C"]["genes"]
    points = grid_points(spec["stages"]["C"]["grid"], genes)
    if candidates >= len(points):
        return points
    idx = np.linspace(0, len(points) - 1, candidates).round().astype(int)
    return [points[i] for i in sorted(set(idx.tolist()))]


def verify_freeze(path):
    freeze = json.loads(Path(path).read_text())
    if not freeze["authorization"]["user_launch"]:
        raise ValueError("Execution is not authorized")
    for name, expected in freeze["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    return freeze


def encode_params(params, spec):
    return encode(params, spec["stages"]["C"]["genes"])


__all__ = [
    "GENES",
    "aggregate",
    "candidate_job",
    "design",
    "digest",
    "encode_params",
    "evaluation_seeds",
    "grid_for_budget",
    "make_job",
    "quantile_dials",
    "schedule_A",
    "schedule_B",
    "schedule_B2",
    "search_budget",
    "stage_pipeline",
    "verify_freeze",
]
