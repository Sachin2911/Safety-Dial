"""One matched-budget evolutionary transition, with immutable shared pools/lineage."""

from dataclasses import asdict
from itertools import product

import numpy as np

from helpers.maniguard_part2_controller import DOMAIN, Config

GRID = list(product(*(range(len(values)) for values in DOMAIN.values())))
NEUTRAL = [0, 0, 4, 0]
METHODS = ("E", "M", "C")


def initial_population(seed=20261024):
    rng = np.random.default_rng(seed)
    population = [NEUTRAL.copy()]
    while len(population) < 4:
        genome = list(GRID[int(rng.integers(len(GRID)))])
        if genome not in population:
            population.append(genome)
    return population


def rank_population(population, records):
    scores = []
    for slot in range(len(population)):
        outcomes = [record for job, record in records if job["slot"] == slot]
        if len(outcomes) != 2 or any(not r.get("valid") for r in outcomes):
            raise ValueError("Two valid full physical evaluations required; missing fitness is not zero")
        scores.append(sum(r["U"] for r in outcomes) / 2)
    return sorted(range(len(population)), key=lambda i: (-scores[i], i)), scores


def mutation_pool(population, ranking, seed=20261025):
    rng = np.random.default_rng(seed)
    parents = ranking[:2]
    timing_parent = int(rng.choice(parents))
    timing = population[timing_parent].copy()
    coordinate = int(rng.choice([2, 3]))
    alternatives = [i for i in range(len(list(DOMAIN.values())[coordinate])) if i != timing[coordinate]]
    timing[coordinate] = int(rng.choice(alternatives))
    pool = []
    # Finite cap space avoids an unbounded rejection loop. Each parent retains its
    # timing genes; pool generation is shared exactly across all three methods.
    available = []
    for parent in parents:
        original = population[parent]
        for cap0, cap1 in product(range(4), repeat=2):
            child = [cap0, cap1, *original[2:]]
            if child != original:
                available.append({"genome": child, "parent": parent})
    for index in rng.permutation(len(available)):
        proposal = available[int(index)]
        if proposal["genome"] not in [p["genome"] for p in pool]:
            pool.append({**proposal, "birth": len(pool)})
        if len(pool) == 16:
            break
    return {"elite": population[ranking[0]].copy(), "elite_parent": ranking[0],
            "timing": timing, "timing_parent": timing_parent, "pool": pool,
            "seed": seed, "rng_state": rng.bit_generator.state}


def select_descendants(shared, rankings=None, seed=20261026):
    rankings = [] if rankings is None else rankings
    fallback_order = np.random.default_rng(seed).permutation(len(shared["pool"])).tolist()
    result = {}
    for method in METHODS:
        selected, log = [], []
        if method != "E":
            eligible = [r for r in rankings if r[method + "_trusted"]]
            eligible.sort(key=lambda r: tuple(r["ranking"]) + (r["birth"],))
            for row in eligible[:2]:
                selected.append(row["birth"])
                log.append({"birth": row["birth"], "selection": "model_rank"})
        for index in fallback_order:
            if len(selected) == 2:
                break
            if index not in selected:
                selected.append(index)
                log.append({"birth": index, "selection": "uniform" if method == "E" else "random_abstention_fallback"})
        if len(selected) != 2:
            raise ValueError("Cannot fill the frozen four-slot population")
        result[method] = {
            "population": [shared["elite"], shared["timing"],
                           *[shared["pool"][i]["genome"] for i in selected]],
            "slots": [{"role": "reevaluated_elite", "parent": shared["elite_parent"]},
                      {"role": "unassisted_timing_mutant", "parent": shared["timing_parent"]},
                      *[{"role": "motion_mutant", "parent": shared["pool"][i]["parent"], **l}
                        for i, l in zip(selected, log, strict=True)]],
            "selected_pool_indices": selected, "fallback_order": fallback_order,
        }
    return result


def jobs_for(populations, stage):
    panels = {
        "initial": [("task_0034/base", 6000), ("task_0002/base", 6001)],
        "descendants": [("task_0034/base", 6010), ("task_0002/base", 6011)],
        "confirmation": [("task_0034/base", 6020), ("task_0002/base", 6021)],
    }
    jobs = []
    for case, (scene, seed) in enumerate(panels[stage]):
        for slot in range(len(next(iter(populations.values())))):
            methods = list(populations)
            methods = methods[(case + slot) % len(methods):] + methods[:(case + slot) % len(methods)]
            for method in methods:
                genome = populations[method][slot]
                jobs.append({"episode_id": f"part3-{stage}-{method}-s{slot}-c{case}",
                             "part": 3, "stage": stage, "purpose": "train",
                             "condition": method, "method": method, "slot": slot,
                             "case": case, "scene": scene, "seed": seed,
                             "genome": genome, "controller": asdict(Config.from_genome(genome)),
                             "action_horizon": 1100})
    return jobs


def comparison_jobs(finalists, reference):
    configs = {"identity": asdict(Config()), "fixed": reference,
               **{method: asdict(Config.from_genome(finalists[method])) for method in METHODS}}
    names = list(configs)
    jobs = []
    for block, (scene, seed) in enumerate(
        (scene, 7000 + 10 * i + r)
        for i, scene in enumerate(("task_0048/base", "task_0050/base")) for r in range(2)
    ):
        order = names[block:] + names[:block]
        for name in order:
            jobs.append({"episode_id": f"part3-comparison-b{block}-{name}", "part": 3,
                         "stage": "comparison", "purpose": "validation", "block": block,
                         "condition": name, "scene": scene, "seed": seed,
                         "controller": configs[name], "action_horizon": 1100})
    return jobs
