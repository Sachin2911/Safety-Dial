"""Four matched sequential acquisition kernels, independent of the simulator.

Only evaluated trajectories may update these states. A rollout adapter must charge
each proposal/validation/attempt before calling update. This module neither launches
physics nor authorizes a search when the frozen premise has failed.
"""

import copy
from dataclasses import asdict, dataclass

import numpy as np

METHODS = ("R", "U", "E", "D")


@dataclass(frozen=True)
class Scenario:
    proposal_id: str
    task: str
    genome: tuple[float, float]
    order: int
    valid: bool
    quality: float | None = 0.0
    uncertainty: float | None = 0.0
    descriptor: tuple[float, float] | None = None

    def __post_init__(self):
        if (len(self.genome) != 2 or not np.isfinite(self.genome).all()
                or np.any(np.abs(self.genome) > 1)):
            raise ValueError("Two finite normalized scene genes in [-1,1] required")
        if not self.valid:
            if self.quality is not None or self.uncertainty is not None:
                raise ValueError("Invalid scenes have no scientific acquisition scores")
            return
        if self.quality is None or self.uncertainty is None or not np.isfinite(
                [self.quality, self.uncertainty]).all():
            raise ValueError("Finite acquisition scores required")
        if not 0 <= self.quality <= 1 or self.uncertainty < 0:
            raise ValueError("Invalid quality or disagreement")


def archive_cell(descriptor):
    values = np.asarray(descriptor, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all() or np.any(np.abs(values) > 1):
        raise ValueError("Descriptor outside frozen normalized spatial bounds")
    return tuple(np.minimum(3, np.floor((values + 1) * 2).astype(int)))


def reflect_bounds(values):
    """Reflect rather than clip Gaussian offspring, avoiding boundary point masses."""
    values = np.asarray(values)
    return 1 - np.abs((values + 1) % 4 - 2)


class Acquisition:
    def __init__(self, method, shared, seed, slots=12, mutation_sigma=.25):
        if method not in METHODS or len(shared) != 8:
            raise ValueError("Four declared arms require exactly eight shared attempts")
        self.method, self.rng, self.slots = method, np.random.default_rng(seed), slots
        self.shared = copy.deepcopy(list(shared))
        self.tasks = sorted({s.task for s in shared})
        if len(self.tasks) != 2 or any(sum(s.task == task for s in shared) != 4 for task in self.tasks):
            raise ValueError("Balance eight shared seeds over two development base tasks")
        if any(not any(s.valid and s.task == task for s in shared) for task in self.tasks):
            raise ValueError("Both base tasks require at least one valid shared scene")
        self.sigma = mutation_sigma
        self.records, self.pending = [], None
        self.evaluated, self.archive = list(self.shared), {}
        for scenario in self.shared:
            self._archive(scenario)
        # R uses one stratified Latin hypercube per base task, independent of scores.
        self.random_schedule = {}
        for offset, task in enumerate(self.tasks):
            count = len(range(offset, slots, len(self.tasks)))
            self.random_schedule[task] = np.column_stack([
                (self.rng.permutation(count) + self.rng.random(count)) / count * 2 - 1
                for _ in range(2)])

    def _score(self, scenario):
        return scenario.uncertainty if self.method == "U" else scenario.quality

    def _archive(self, scenario):
        if not scenario.valid or scenario.descriptor is None:
            return
        cell = archive_cell(scenario.descriptor)
        previous = self.archive.get(cell)
        if previous is None or (scenario.quality, -scenario.order) > (previous.quality, -previous.order):
            self.archive[cell] = scenario

    def propose(self):
        if self.pending is not None:
            raise ValueError("Every evaluated slot must be recorded before another proposal")
        slot = len(self.records)
        if slot >= self.slots:
            raise StopIteration("Acquisition allowance exhausted")
        task = self.tasks[slot % len(self.tasks)]
        parent, fallback = None, None
        if self.method == "R":
            genome = self.random_schedule[task][slot // len(self.tasks)]
        else:
            eligible = [s for s in self.evaluated if s.valid and s.task == task]
            if self.method == "D":
                parents = [s for s in self.archive.values() if s.task == task]
            else:
                # Two retained parents per task, four in total. Ties use birth order.
                parents = sorted(eligible, key=lambda s: (-self._score(s), s.order))[:2]
            if not parents or not any(self._score(s) > 0 for s in parents):
                parents = [s for s in self.shared if s.valid and s.task == task]
                fallback = "uniform_shared_task_parent"
            parent = parents[int(self.rng.integers(len(parents)))]
            genome = reflect_bounds(np.asarray(parent.genome) + self.rng.normal(0, self.sigma, 2))
        self.pending = {"proposal_id": f"{self.method}-{slot:02d}", "method": self.method,
                        "slot": slot, "task": task, "genome": genome.tolist(),
                        "parent": parent.proposal_id if parent else None,
                        "fallback": fallback, "duplicate_policy": "charge and retain repeats; no free resampling"}
        return copy.deepcopy(self.pending)

    def update(self, scenario, cost):
        if self.pending is None or scenario.proposal_id != self.pending["proposal_id"]:
            raise ValueError("Evaluation does not match the one pending charged proposal")
        if scenario.task != self.pending["task"] or list(scenario.genome) != self.pending["genome"]:
            raise ValueError("Evaluation silently changed the proposed scene")
        required = ("controls", "physics_steps", "VLA_queries", "seconds", "validation_attempts")
        if any(k not in cost for k in required):
            raise ValueError("Every evaluated or invalid slot requires a complete cost record")
        if any(v is not None and (not np.isfinite(v) or v < 0) for v in cost.values()):
            raise ValueError("Invalid interaction accounting")
        self.evaluated.append(scenario)
        self._archive(scenario)
        self.records.append({**self.pending, "evaluation": asdict(scenario), "cost": dict(cost),
                             "archive": {f"{k[0]},{k[1]}": s.proposal_id
                                         for k, s in sorted(self.archive.items())}})
        self.pending = None

    def confirmation_scenario(self):
        discoveries = [s for s in self.evaluated[8:] if s.valid]
        if not discoveries:
            discoveries = [s for s in self.shared if s.valid]
        return min(discoveries, key=lambda s: (-s.quality, s.order))


def interleaved_schedule(seed=20261045, slots=12):
    rng = np.random.default_rng(seed)
    return [{"slot": slot, "method": str(method)} for slot in range(slots)
            for method in rng.permutation(METHODS)]
