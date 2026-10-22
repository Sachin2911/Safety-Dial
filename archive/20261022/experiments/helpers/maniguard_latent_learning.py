"""Unfunded episodic REINFORCE interface for the same conditional steering weights.

This module never launches a simulator or queries a policy. A future prospectively
budgeted runner must supply two independently verified full-episode scores per
sample. It does not join the frozen minimum EA/random-search comparison, and is
not a state-conditioned SAC implementation or evidence of a competent RL baseline.
"""

from copy import deepcopy

import numpy as np

from helpers.maniguard_latent import CODE_DIM, FEATURE_DIM, LatentController

SHAPE = (CODE_DIM, FEATURE_DIM)


def matrix(value):
    value = np.asarray(value)
    if (value.dtype.kind not in "fiu" or value.shape != SHAPE
            or not np.isfinite(value).all()):
        raise ValueError("Expected a finite 4 by 44 parameter matrix")
    return value.astype(np.float64)


def scores(value):
    if len(value) != 2 or any(type(v) is not int or v not in [-2, 0, 1] for v in value):
        raise ValueError("Need two valid physical episode scores per sample")
    return list(value)


def gaussian_score(raw, center, sigma):
    """Derivative of log N(raw; center, sigma**2 I) with respect to center.

    The sampled controller is tanh(raw). Its transform has no trainable Jacobian
    term in this score derivative at a fixed sample. No simulator derivative is
    used. Quantized float32 controller weights retain the same sampling estimator.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sampling sigma must be finite and positive")
    return (matrix(raw) - matrix(center)) / sigma**2


class ParameterReinforce:
    """Fixed-variance parameter policy gradient with leave-one-out baselines.

    Four sampled conditional controllers each require two case scores. A consumer
    must charge all attempts and failures in its physical ledger. Initialization
    may use the same four shared candidates as the other methods. Checkpoint state
    includes pending draws, partial scores and RNG, so observation can resume without
    resampling or treating an incomplete batch as a learning update.

    Defaults describe an unexecuted interface, not a funded experimental protocol.
    Selecting a finalist, learning hyperparameters and an independent audit still
    require a prospective allocation. The mean controller has no physical receipt.
    """

    def __init__(self, *, seed=20261020, sigma=.25, learning_rate=.05, batches=2,
                 shared_initialization=None, state=None):
        self.rng = np.random.default_rng(seed)
        if state is None:
            self.sigma, self.learning_rate, self.batches = sigma, learning_rate, batches
            self.center = np.zeros(SHAPE, dtype=np.float64)
            self.generation, self.pending, self.responses, self.history = 0, [], {}, []
            self.shared_initialization = deepcopy(shared_initialization or [])
            if self.shared_initialization:
                if len(self.shared_initialization) != 4:
                    raise ValueError("Shared initialization requires exactly four candidates")
                ids = [row["id"] for row in self.shared_initialization]
                if len(set(ids)) != 4:
                    raise ValueError("Shared candidate identifiers must be distinct")
                for row in self.shared_initialization:
                    LatentController(row["weights"])
                    scores(row["scores"])
                best = min(self.shared_initialization,
                           key=lambda row: (-sum(row["scores"]), row["id"]))
                # Endpoints need a finite inverse; the exported controller always
                # remains in the original closed weight domain [-1, 1].
                self.center = np.arctanh(np.clip(matrix(best["weights"]),
                                                 -1 + 1e-7, 1 - 1e-7))
        else:
            state = deepcopy(state)
            if state["schema"] != "maniguard-latent-parameter-reinforce-v1":
                raise ValueError("Unknown learning checkpoint schema")
            self.sigma, self.learning_rate, self.batches = (
                state["sigma"], state["learning_rate"], state["batches"])
            self.center = matrix(state["center"])
            self.generation = state["generation"]
            self.pending, self.responses = state["pending"], state["responses"]
            self.history = state["history"]
            self.shared_initialization = state["shared_initialization"]
            self.rng.bit_generator.state = state["rng"]
        if (not np.isfinite(self.sigma) or self.sigma <= 0
                or not np.isfinite(self.learning_rate) or self.learning_rate <= 0
                or type(self.batches) is not int or self.batches < 1):
            raise ValueError("Invalid fixed learning configuration")
        if (type(self.generation) is not int or not 0 <= self.generation <= self.batches
                or len(self.history) != self.generation):
            raise ValueError("Learning generation and history disagree")
        if self.pending:
            expected = {f"reinforce-g{self.generation + 1}-{i}" for i in range(4)}
            if (self.generation >= self.batches or len(self.pending) != 4
                    or {row["id"] for row in self.pending} != expected
                    or not set(self.responses).issubset(expected)):
                raise ValueError("Pending learning batch is incomplete or inconsistent")
            for row in self.pending:
                self._verify_sample(row)
            for value in self.responses.values():
                scores(value)
        elif self.responses:
            raise ValueError("Scores have no corresponding pending batch")

    @staticmethod
    def _verify_sample(row):
        controller = LatentController(row["weights"])
        if (not np.array_equal(controller.weights, np.tanh(matrix(row["raw"])).astype(np.float32))
                or controller.sha256() != row["sha256"]):
            raise ValueError("Learning sample differs from its recorded draw or controller hash")

    def propose(self):
        if not self.pending:
            if self.generation >= self.batches:
                raise ValueError("Declared learning batch allowance exhausted")
            for i in range(4):
                raw = self.center + self.sigma * self.rng.normal(size=SHAPE)
                controller = LatentController(np.tanh(raw))
                self.pending.append({"id": f"reinforce-g{self.generation + 1}-{i}",
                                     "raw": raw.tolist(), "weights": controller.weights.tolist(),
                                     "sha256": controller.sha256()})
        return deepcopy(self.pending)

    def record(self, candidate_id, episode_scores):
        if candidate_id not in {row["id"] for row in self.pending}:
            raise ValueError("Scores must name a pending sampled controller")
        value = scores(episode_scores)
        if candidate_id in self.responses and self.responses[candidate_id] != value:
            raise ValueError("A completed sample's physical scores cannot change")
        self.responses[candidate_id] = value

    def update(self):
        if len(self.pending) != 4 or len(self.responses) != 4:
            raise ValueError("Complete every case of every sample before learning")
        for row in self.pending:
            self._verify_sample(row)
        returns = np.asarray([sum(self.responses[row["id"]]) for row in self.pending], float)
        # Each baseline excludes its own sampled return. A constant batch has
        # exactly zero gradient; no privileged or additional reward enters here.
        baseline = (returns.sum() - returns) / 3
        gradient = np.mean([
            (value - reference) * gaussian_score(row["raw"], self.center, self.sigma)
            for row, value, reference in zip(self.pending, returns, baseline, strict=True)], axis=0)
        updated = self.center + self.learning_rate * gradient
        if not np.isfinite(updated).all():
            raise ValueError("Nonfinite learning update; retain the original pending checkpoint")
        self.history.append({"generation": self.generation + 1,
                             "samples": deepcopy(self.pending), "scores": deepcopy(self.responses),
                             "gradient_norm": float(np.linalg.norm(gradient)),
                             "valid_case_scores": 8})
        self.center = updated
        self.generation += 1
        self.pending, self.responses = [], {}
        return self.mean_controller()

    def mean_controller(self):
        return LatentController(np.tanh(self.center))

    def state(self):
        return deepcopy({"schema": "maniguard-latent-parameter-reinforce-v1",
                         "sigma": self.sigma, "learning_rate": self.learning_rate,
                         "batches": self.batches, "generation": self.generation,
                         "center": self.center.tolist(), "rng": self.rng.bit_generator.state,
                         "pending": self.pending, "responses": self.responses,
                         "history": self.history,
                         "shared_initialization": self.shared_initialization})
