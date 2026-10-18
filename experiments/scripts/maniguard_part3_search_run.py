"""Physically evaluate the frozen 64-episode Part 3 pilot only after its model gate."""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments/scripts"))

from maniguard_part2_run import launch, require_guard, save_new

from helpers.maniguard_followup_budget import atomic, clock
from helpers.maniguard_part2_controller import Config
from helpers.maniguard_part2_protocol import ENGINEERING, RUN, validate_frozen
from helpers.maniguard_part3_acquisition import screen
from helpers.maniguard_part3_protocol import implementation_files
from helpers.maniguard_part3_search import (
    METHODS,
    comparison_jobs,
    initial_population,
    jobs_for,
    mutation_pool,
    rank_population,
    select_descendants,
)
from helpers.maniguard_phase2_families import validate_scene_access
from helpers.maniguard_phase3_protocol import sha

MODEL = ROOT / "data/maniguard/models/phase3-parts2-3-20261018"


def stage(directory, jobs, base_protocol):
    directory.mkdir(parents=True, exist_ok=True)
    protocol = directory / "protocol.json"
    if not protocol.exists():
        save_new(protocol, base_protocol)
        save_new(directory / "jobs-plan.json", {"jobs": jobs, "protocol_sha256": sha(protocol)})
    else:
        if json.loads((directory / "jobs-plan.json").read_text())["jobs"] != jobs:
            raise ValueError("A resumed stage differs from its prospective jobs")
        validate_frozen(protocol)
    outcomes = []
    for job in jobs:
        existing = directory / "attempts" / (job["episode_id"] + "-a1") / "audit.json"
        record = json.loads(existing.read_text()) if existing.exists() else launch(directory, job, protocol)
        outcomes.append((job, record))
        atomic(directory / "progress.json", {"records": [r for _, r in outcomes], "clock": clock()})
        if not record["valid"]:
            raise RuntimeError("Missing required physical fitness; stop without retry or imputation")
    save_new(directory / "results.json", {"records": [r for _, r in outcomes], "clock": clock()})
    return outcomes


def reference():
    records = json.loads((RUN / "part2/results.json").read_text())["records"]
    ranked = []
    for condition in ("B", "C"):
        selected = [r for r in records if r["condition"] == condition]
        if len(selected) != 4 or any(not r["valid"] for r in selected):
            raise ValueError("Part 2 cannot supply its frozen fixed-reference rule")
        ranked.append((sum(r["U"] for r in selected) / 4,
                       -sum(r["costs"]["policy_queries"] for r in selected), condition))
    # Final key B wins a full tie, as frozen before Part 2.
    ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))
    winner = ranked[0][2]
    return {"condition": winner, "controller": next(r["controller"] for r in records
                                                      if r["condition"] == winner),
            "scores": ranked, "part2_result_sha256": sha(RUN / "part2/results.json")}


def main():
    require_guard()
    audit_path = MODEL / "forecast-audit.json"
    forecast = json.loads(audit_path.read_text())
    if not forecast["gate"]["eligible"]:
        raise RuntimeError("Forecast gate is not eligible; physical optimizer comparison stays unstarted")
    part = RUN / "part3/search"
    part.mkdir(parents=True, exist_ok=True)
    fixed = reference()
    save_new(part / "fixed-reference.json", fixed)
    # A conservative observed planning duration, without using outcomes to choose
    # allocation. The 64 slots are indivisible and never shrunk after this decision.
    timings = [r["process"]["elapsed_seconds"] for r in
               json.loads((RUN / "part2/results.json").read_text())["records"]]
    planning_seconds = max(180.0, float(max(timings)) * 1.2)
    decision = {"at": time.time(), "episodes": 64, "engineering_attempts": 3,
                "planning_seconds_each": planning_seconds,
                "required_seconds": 64 * planning_seconds + 3 * 120 + 300,
                "clock": clock(), "basis": "Part2 process timings only; no label-dependent allocation"}
    save_new(part / "resource-decision.json", decision)
    if decision["required_seconds"] > clock()["physical_remaining"]:
        atomic(part / "terminal.json", {"status": "incomplete_resource_limit_before_search",
                                         "decision": decision})
        return
    protocol = json.loads((RUN / "part2/protocol.json").read_text())
    protocol.update(schema="maniguard-part3-search-v1", part=3, full_episode_allowance=64,
                    model_gate_sha256=sha(audit_path), forecast_freeze_sha256=sha(MODEL / "forecast-freeze.json"),
                    initial_seed=20261024, mutation_seed=20261025, selection_seed=20261026,
                    allocation={"shared_initial": 8, "descendants": 24,
                                "confirmation": 12, "comparison": 20},
                    information="same four-parameter controller at deployment; model used for training-time cap selection only",
                    selection="top two actual mean-U initial parents; re-evaluated elite; one timing explorer; two cap proposals",
                    model_update="none during primary comparison",
                    model_files={p.name: sha(p) for p in MODEL.iterdir()
                                 if p.name in ("artifacts.npz", "ensemble.pt", "calibration.json", "context-bank.npz")},
                    code_hashes={str(p.relative_to(ROOT)): sha(p) for p in implementation_files()})
    scenes = {j["scene"] for j in jobs_for({"shared": initial_population()}, "initial")}
    scenes |= {"task_0048/base", "task_0050/base"}
    for scene in sorted(scenes):
        purpose = "validation" if scene in ("task_0048/base", "task_0050/base") else "train"
        validate_scene_access("clutter", scene, purpose)
        scene_path = ROOT / "data/maniguard/benchmark/clutter_pickup" / scene / "scene_ep1.json"
        protocol["scenes"] += [{"scene": scene, "purpose": purpose, "sha256": sha(scene_path)}]
    # Three attempts maximum, no retries. Exercise neutral, six-action timing and
    # a nonzero correction threshold with the already checked physical evaluator.
    engineering = part / "engineering"
    engineering.mkdir()
    checks = []
    for i, genome in enumerate(([0, 0, 4, 0], [3, 3, 0, 1], [2, 2, 3, 0])):
        job = {"episode_id": f"part3-engineering-{i}", "part": 3, "purpose": "engineering",
               "condition": "engineering", "scene": ENGINEERING, "seed": 5900,
               "action_horizon": 32, "genome": genome,
               "controller": asdict(Config.from_genome(genome))}
        record = launch(engineering, job)
        checks.append(record)
        if not record["engineering_valid"]:
            save_new(part / "engineering-incomplete.json", checks)
            raise RuntimeError("Bounded Part3 engineering failed")
    protocol["engineering_audits"] = checks
    save_new(part / "protocol.json", protocol)
    initial = initial_population()
    initial_records = stage(part / "initial", jobs_for({"shared": initial}, "initial"), protocol)
    ranking, scores = rank_population(initial, initial_records)
    shared = mutation_pool(initial, ranking)
    shared.update(initial_population=initial, initial_ranking=ranking, initial_scores=scores)
    save_new(part / "mutation-pool.json", shared)
    forecasts = screen(shared["pool"], MODEL, device="cpu")
    save_new(part / "acquisition.json", forecasts)
    descendants = select_descendants(shared, forecasts["records"])
    save_new(part / "descendant-lineage.json", descendants)
    populations = {method: descendants[method]["population"] for method in METHODS}
    descended = stage(part / "descendants", jobs_for(populations, "descendants"), protocol)
    shortlist, final_ranks = {}, {}
    for method in METHODS:
        ranking, scores = rank_population(populations[method], [(j, r) for j, r in descended
                                                                if j["method"] == method])
        shortlist[method] = [populations[method][slot] for slot in ranking[:2]]
        final_ranks[method] = {"ranking": ranking, "scores": scores}
    save_new(part / "shortlist.json", {"genomes": shortlist, "final_population": final_ranks})
    confirmation = stage(part / "confirmation", jobs_for(shortlist, "confirmation"), protocol)
    finalists, choices = {}, {}
    for method in METHODS:
        original_slots = final_ranks[method]["ranking"][:2]
        combined = []
        for slot, original_slot in enumerate(original_slots):
            values = [r["U"] for j, r in descended if j["method"] == method and j["slot"] == original_slot]
            values += [r["U"] for j, r in confirmation if j["method"] == method and j["slot"] == slot]
            if len(values) != 4:
                raise ValueError("Finalist needs four equally weighted full outcomes")
            combined.append(sum(values) / 4)
        winner = min(range(2), key=lambda slot: (-combined[slot], original_slots[slot]))
        finalists[method] = shortlist[method][winner]
        choices[method] = {"mean_U": combined, "original_slots": original_slots, "selected": winner}
    save_new(part / "finalists.json", {"genomes": finalists, "selection": choices,
                                        "frozen_at": time.time(), "protocol_sha256": sha(part / "protocol.json")})
    fresh = stage(part / "comparison", comparison_jobs(finalists, fixed["controller"]), protocol)
    save_new(part / "terminal.json", {"status": "completed", "fresh_episodes": len(fresh),
                                      "full_episodes": 64, "clock": clock()})


if __name__ == "__main__":
    main()
