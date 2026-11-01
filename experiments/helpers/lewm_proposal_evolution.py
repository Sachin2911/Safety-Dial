"""Resumable ES mechanics for proposal weights; no implicit simulator evaluation."""

import hashlib
import json
from pathlib import Path
import pickle

import cma
import numpy as np


class ProposalEvolution:
    """Each submitted fitness belongs to one complete declared development panel.

    Smoke tests use a synthetic evaluator on the rented host. Real episode training
    requires its own prospective panel and budget; creating this object runs none.
    Only restore checkpoints produced by this class in your own trusted run tree.
    """

    def __init__(self, initial, *, seed, population=4, sigma=0.05, checkpoint, resume=False):
        self.path = Path(checkpoint)
        initial = np.asarray(initial, dtype=float)
        config = {
            "initial": initial.tolist(),
            "seed": seed,
            "population": population,
            "sigma": sigma,
        }
        self.identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if resume:
            with self.path.open("rb") as stream:
                state = pickle.load(stream)
            if state["identity"] != self.identity:
                raise ValueError("Evolution resume configuration differs")
            self.es, self.pending, self.history = state["es"], state["pending"], state["history"]
            np.random.set_state(state["rng"])
        else:
            if self.path.exists() or initial.ndim != 1 or not np.isfinite(initial).all():
                raise ValueError("Fresh finite initialization and new checkpoint required")
            self.es = cma.CMAEvolutionStrategy(
                initial,
                sigma,
                {
                    "seed": seed,
                    "popsize": population,
                    "CMA_diagonal": True,
                    "verbose": -9,
                    "verb_disp": 0,
                    "verb_log": 0,
                },
            )
            self.pending, self.history = None, []
            self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        staging = self.path.with_suffix(".part")
        with staging.open("wb") as stream:
            pickle.dump(
                {
                    "identity": self.identity,
                    "es": self.es,
                    "pending": self.pending,
                    "history": self.history,
                    "rng": np.random.get_state(),
                },
                stream,
            )
        staging.replace(self.path)

    def candidates(self):
        if self.pending is None:
            self.pending = [np.asarray(x).tolist() for x in self.es.ask()]
            self.save()
        return np.asarray(self.pending).copy()

    def submit(self, fitness):
        scores = np.asarray(fitness, dtype=float)
        if self.pending is None or scores.shape != (len(self.pending),):
            raise ValueError("One fitness per pending candidate is required")
        if not np.isfinite(scores).all():
            raise ValueError("Invalid panels cannot update evolution")
        self.es.tell(self.pending, -scores)
        self.history.append({"parameters": self.pending, "fitness": scores.tolist()})
        self.pending = None
        self.save()
