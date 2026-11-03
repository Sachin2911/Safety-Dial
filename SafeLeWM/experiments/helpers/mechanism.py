"""Frozen representation maps and precisely scoped recurrent-adapter ablations."""

import numpy as np

from helpers.pilot_common import score


class SearchMap:
    def __init__(self, neutral, scales, representation, projection=None):
        self.neutral = np.asarray(neutral, dtype=np.float32)
        self.scales = np.asarray(scales, dtype=np.float32)
        assert self.neutral.shape == self.scales.shape == (19172,)
        assert representation in ["full", "head", "subspace"]
        self.representation = representation
        self.projection = projection
        self.dimension = {"full": 19172, "head": 132, "subspace": 256}[representation]
        if representation == "subspace":
            assert projection.shape == (19172, 256) and projection.dtype == np.float32

    def decode(self, genotype):
        genotype = np.asarray(genotype, dtype=np.float32)
        assert genotype.shape == (self.dimension,) and np.isfinite(genotype).all()
        if self.representation == "full":
            update = genotype
        elif self.representation == "head":
            update = np.zeros(19172, dtype=np.float32)
            update[-132:] = genotype
        else:
            update = self.projection @ genotype
        return self.neutral + self.scales * update


def projection_matrix(seed):
    matrix = np.random.default_rng(seed).normal(size=(19172, 256)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix


def mask_inputs(inputs, arm):
    value = np.asarray(inputs, dtype=np.float32).copy()
    if arm == "blind":
        value[386:389] = 0
    return value


def actor_step(actor, inputs, hidden, base, arm):
    if arm == "reset":
        hidden = actor.initial_state(batch_size=inputs.shape[0])
    return actor(inputs, hidden, base)


def objective(rows, arm):
    metrics = score(rows)
    if arm == "task":
        metrics["score"] = metrics["complete"]
    return metrics


def choose_representation(rows, representations=("full", "head", "subspace")):
    assert rows and all(r["stage"] == "development" for r in rows)
    metrics = {
        name: score([r for r in rows if r["representation"] == name]) for name in representations
    }
    assert all(v["n"] == 108 for v in metrics.values())
    # Python max retains the earliest representation for exact ties.
    winner = max(representations, key=lambda name: metrics[name]["score"])
    return winner, metrics
