"""Task-12 immutable schedules, job construction and freeze verification (TwoRoom)."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from helpers.tworoom12_evo import GENES, clearance_schedule, decode, encode, grid_points

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def design():
    return json.loads((ROOT / "configs/tworoom12/design.json").read_text())


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
        tag = f"lam{lam:g}" if mode == "penalty" else f"d{clearances[0]:g}"
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


def schedule_A(spec):
    """Unconstrained planner on every layout and seed, plus two safe dials on T2."""
    jobs = []
    stage = spec["stages"]["A"]
    horizon = spec["planner"]["horizon"]
    for lay in spec["hazard"]["layouts"]:
        for seed in stage["seeds"]:
            jobs.append(
                make_job("A", lay["id"], seed, "penalty", spec=spec, lam=0.0, execute="mean")
            )
    first = stage["seeds"][0]
    for d in stage.get("safe_dials", (0.0, 12.0)):
        jobs.append(make_job("A", "T2", first, "safe", spec=spec, clearances=[d] * horizon))
    return jobs


def schedule_B(spec):
    """The dial sweep: free baseline, the checked pipeline on every layout, the unchecked
    elite mean on the search layout; seeds shared across conditions for pairing."""
    stage = spec["stages"]["B"]
    horizon = spec["planner"]["horizon"]
    jobs = []
    for cond in stage["conditions"]:
        for lay in cond["layouts"]:
            for seed in stage["seeds"]:
                if cond["mode"] == "penalty":
                    job = make_job(
                        "B",
                        lay,
                        seed,
                        "penalty",
                        spec=spec,
                        lam=cond["lam"],
                        tag=f"{cond['name']}-lam{cond['lam']:g}",
                        execute=cond["execute"],
                    )
                    job["condition"] = cond["name"]
                    jobs.append(job)
                    continue
                for d in stage["dials"]:
                    job = make_job(
                        "B",
                        lay,
                        seed,
                        "safe",
                        spec=spec,
                        clearances=[float(d)] * horizon,
                        tag=f"{cond['name']}-d{d:g}",
                        execute=cond["execute"],
                    )
                    job["condition"] = cond["name"]
                    jobs.append(job)
    rng = random.Random(2026102512)
    rng.shuffle(jobs)
    return jobs


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
        **spec["stages"]["C"]["pipeline"],
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
    "schedule_A",
    "schedule_B",
    "search_budget",
    "verify_freeze",
]
