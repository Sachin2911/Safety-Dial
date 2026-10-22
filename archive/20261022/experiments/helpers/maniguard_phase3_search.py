"""Part 1 search, exact schedules and a timing-only allocation selector."""

import itertools
import math
from dataclasses import asdict

import numpy as np

from helpers.maniguard_phase3_adapter import DOMAIN, NEUTRAL, Controller, genome_hash

TRAIN = ("task_0034/base", "task_0002/base")
COMPARISON = ("task_0029/base", "task_0032/base")
ENGINEERING = "task_0040/base"
VARIANTS = {48: (3, 4), 28: (2, 2)}  # populations, repetitions per validation scene
GRID = tuple(itertools.product(*(range(len(values)) for values in DOMAIN.values())))


def select_allocation(durations, remaining_seconds, overhead_seconds=300):
    """Inputs deliberately contain timing only; shared population zero counts once."""
    if len(durations) != 8 or any(
        type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in durations
    ):
        raise ValueError("Need eight completed spawn-through-audit process durations")
    t_plan = max(417.0, 1.25 * max(durations))
    estimates = {size: (size - 8) * t_plan + overhead_seconds for size in VARIANTS}
    choice = next((size for size in VARIANTS if estimates[size] <= remaining_seconds), None)
    return {
        "variant": choice,
        "t_plan": t_plan,
        "durations": list(durations),
        "remaining_seconds": remaining_seconds,
        "overhead_seconds": overhead_seconds,
        "remaining_estimates": estimates,
        "shared_initial_episodes": 8,
    }


class Evolution:
    def __init__(self, seed=20261018):
        self.rng = np.random.default_rng(seed)
        self.parents = []
        self.generation = 0

    def ask(self):
        records = []
        if not self.parents:
            unused = [g for g in GRID if g != NEUTRAL]
            initial = [NEUTRAL] + [
                unused[int(i)] for i in self.rng.choice(len(unused), size=3, replace=False)
            ]
            return [
                {
                    "genome": list(g),
                    "kind": "neutral" if i == 0 else "initial_random",
                    "parent_slot": None,
                    "draws": 0,
                    "fallback": False,
                }
                for i, g in enumerate(initial)
            ]
        records.append(
            {
                "genome": self.parents[0].copy(),
                "kind": "elite_reevaluation",
                "parent_slot": 0,
                "draws": 0,
                "fallback": False,
            }
        )
        for _ in range(3):
            fallback = False
            for draw in range(1, 101):
                parent_slot = int(self.rng.integers(2))
                parent = self.parents[parent_slot]
                child = parent.copy()
                changed = int(self.rng.integers(4))
                for j, values in enumerate(DOMAIN.values()):
                    if j == changed or self.rng.random() < 0.5:
                        if self.rng.random() < 0.2:
                            child[j] = int(self.rng.integers(len(values)))
                        else:
                            child[j] = int(
                                np.clip(
                                    child[j] + self.rng.choice([-2, -1, 1, 2]), 0, len(values) - 1
                                )
                            )
                if child != parent and child not in [r["genome"] for r in records]:
                    break
            else:
                unused = [
                    g
                    for g in GRID
                    if list(g) not in [r["genome"] for r in records] and list(g) != parent
                ]
                child = list(unused[int(self.rng.integers(len(unused)))])
                fallback = True
            records.append(
                {
                    "genome": child,
                    "kind": "offspring",
                    "parent_slot": parent_slot,
                    "draws": draw,
                    "fallback": fallback,
                }
            )
        return records

    def tell(self, population, outcomes):
        if len(population) != 4 or len(outcomes) != 4 or any(len(x) != 2 for x in outcomes):
            raise ValueError("Every slot needs two fresh audited full episodes")
        scores = []
        for pair in outcomes:
            if any(
                r.get("valid") is not True
                or r.get("full_episode") is not True
                or r.get("dial") != 1
                or r.get("U") not in (-2, 0, 1)
                for r in pair
            ):
                raise ValueError("Invalid or partial evidence has no fitness")
            scores.append(sum(r["U"] for r in pair) / 2)
        order = sorted(range(4), key=lambda i: (-scores[i], i))
        self.parents = [population[i]["genome"].copy() for i in order[:2]]
        self.generation += 1
        return {
            "scores": scores,
            "order": order,
            "tie_rule": "stable_slot_birth_order",
            "eligible_parents": self.parents,
            "rng_state": self.rng.bit_generator.state,
        }

    def finalist(self, population, ranking):
        slot = ranking["order"][0]
        genome = population[slot]["genome"]
        return {
            "genome": genome,
            "decoded": asdict(Controller.from_genome(genome)),
            "genome_sha256": genome_hash(genome),
            "slot": slot,
            "final_population": self.generation - 1,
            "fitness": ranking["scores"][slot],
            "identity_selected": tuple(genome) == NEUTRAL,
        }


def training_jobs(generation, population, schedule_rng):
    jobs = []
    for i, scene in enumerate(TRAIN):
        for slot in schedule_rng.permutation(4).tolist():
            genome = population[slot]["genome"]
            jobs.append(
                {
                    "episode_id": f"g{generation}-s{i}-c{slot}",
                    "generation": generation,
                    "slot": slot,
                    "scene": scene,
                    "seed": 1000 + 10 * generation + i,
                    "purpose": "train",
                    "genome": genome,
                    "dial": 1,
                    "action_horizon": 1100,
                }
            )
    return jobs


def comparison_jobs(size, genome, schedule_rng):
    repetitions = VARIANTS[size][1]
    permutations = list(itertools.permutations((0, 0.5, 1)))
    orders = []
    while len(orders) < 2 * repetitions:
        orders.extend([permutations[i] for i in schedule_rng.permutation(6).tolist()])
    jobs = []
    for j, scene in enumerate(COMPARISON):
        for r in range(repetitions):
            block = f"v{j}-r{r}"
            for d in orders[j * repetitions + r]:
                jobs.append(
                    {
                        "episode_id": f"{block}-d{d}",
                        "block": block,
                        "scene": scene,
                        "seed": 2000 + 10 * j + r,
                        "purpose": "validation",
                        "genome": list(genome),
                        "dial": d,
                        "action_horizon": 1100,
                    }
                )
    return jobs
