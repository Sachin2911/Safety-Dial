"""Bounded shared native-noise steering, with no simulator or model dependency.

The feature contract contains robot/history and frozen visual/text embeddings only.
Physical task predicates and safety labels belong to evaluation, never this actor.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

FEATURE_DIM = 44
CODE_DIM = 4


def finite(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != shape or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be finite numeric data of shape {shape}")
    return raw.astype(np.float32)


def arm_basis(action_dim=32, seed=20261019):
    """Four orthonormal arm directions, repeated across native chunk positions.

    Only the first seven model coordinates are supported. Gripper and padding keep
    their native independent noise, although model coupling can affect all outputs.
    """
    if type(action_dim) is not int or action_dim < 8:
        raise ValueError("Model action width must include arm and gripper")
    basis = np.zeros((action_dim, CODE_DIM), dtype=np.float32)
    basis[:7] = np.linalg.qr(np.random.default_rng(seed).normal(size=(7, CODE_DIM)))[0]
    return basis


def feature_vector(joints, fingers, eef, overview, wrist, text, *,
                   previous_executed=None, previous_eef=None):
    """Observation before the next query; history ends at the last executed action."""
    joints = finite(joints, (7,), "arm positions")
    fingers = finite(fingers, (2,), "finger positions")
    eef = finite(eef, (3,), "end-effector position")
    executed = joints if previous_executed is None else finite(
        previous_executed, (8,), "previous executed action")[:7]
    old_eef = eef if previous_eef is None else finite(previous_eef, (3,), "previous eef")
    robot = np.clip(np.r_[joints / np.pi, fingers / .04,
                          (executed - joints) / .2, (eef - old_eef) / .05], -1, 1)
    semantic = [finite(v, (8,), n) for v, n in
                [(overview, "overview"), (wrist, "wrist"), (text, "instruction")]]
    if any(np.any(np.abs(v) > 1) for v in semantic):
        raise ValueError("Projected semantic features must be bounded in [-1,1]")
    return finite(np.concatenate([np.ones(1), robot, *semantic]),
                  (FEATURE_DIM,), "actor features")


@dataclass(frozen=True)
class SteeringSpec:
    alpha: float = .35
    weight_bound: float = 1.
    basis_seed: int = 20261019

    def __post_init__(self):
        if not np.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError("Steering magnitude must lie in (0,1]")
        if not np.isfinite(self.weight_bound) or self.weight_bound <= 0:
            raise ValueError("Weights need a finite positive bound")


class LatentController:
    def __init__(self, weights, spec=None):
        self.spec = SteeringSpec() if spec is None else spec
        self.weights = finite(weights, (CODE_DIM, FEATURE_DIM), "controller weights").copy()
        if np.any(np.abs(self.weights) > self.spec.weight_bound):
            raise ValueError("Controller exceeds frozen parameter domain")

    @property
    def identity(self):
        return not np.any(self.weights)

    def code(self, features):
        return np.tanh(self.weights @ finite(features, (FEATURE_DIM,), "actor features"))

    def offset(self, features, horizon, action_dim):
        if type(horizon) is not int or horizon < 1:
            raise ValueError("A positive native horizon is required")
        shift = self.spec.alpha * arm_basis(action_dim, self.spec.basis_seed) @ self.code(features)
        return np.broadcast_to(shift, (horizon, action_dim)).copy()

    def sha256(self):
        return hashlib.sha256(self.weights.astype("<f4").tobytes()).hexdigest()


def outcome_score(requested_completion, violated):
    if type(requested_completion) is not bool or type(violated) is not bool:
        raise ValueError("Missing physical completion or violation label")
    return int(requested_completion and not violated) - 2 * int(violated)


def initial_weights(seed=20261019):
    """The same four controllers and physical observations seed both optimizers."""
    zero = np.zeros((CODE_DIM, FEATURE_DIM), dtype=np.float32)
    positive, negative = zero.copy(), zero.copy()
    positive[0, 0], negative[0, 0] = np.arctanh(.5), -np.arctanh(.5)
    random = np.random.default_rng(seed).uniform(-1, 1, zero.shape).astype(np.float32)
    return [zero, positive, negative, random]


class CandidateSearch:
    """Two matched generations; resumable RNG state and explicit ancestry.

    Both methods retain and pay to reevaluate their current best. The other three
    slots use Gaussian mutation for EA and uniform domain draws for random search.
    Ties use candidate identifier, never an independent audit outcome.
    """
    def __init__(self, method, seed=20261020, state=None):
        if method not in ("ea", "random"):
            raise ValueError("Expected ea or random")
        self.method = method
        self.rng = np.random.default_rng(seed)
        self.candidates = []
        self.generation = 0
        if state is not None:
            if state["method"] != method:
                raise ValueError("Optimizer checkpoint method mismatch")
            self.rng.bit_generator.state = state["rng"]
            self.candidates = state["candidates"]
            self.generation = state["generation"]

    def record(self, candidate_id, weights, scores, parent=None):
        controller = LatentController(weights)
        if any(c["id"] == candidate_id for c in self.candidates):
            raise ValueError("Candidate identifiers cannot be overwritten")
        if len(scores) != 2 or any(type(s) is not int or s not in (-2, 0, 1) for s in scores):
            raise ValueError("Each candidate needs two valid physical case scores")
        self.candidates.append({"id": candidate_id, "weights": controller.weights.tolist(),
                                "scores": list(scores), "parent": parent,
                                "generation": self.generation, "sha256": controller.sha256()})

    def ranked(self):
        # A reevaluated elite's latest cohort replaces its earlier noisy score for
        # selection. No best-of-repeats promotion or accumulated extra information.
        current = [c for c in self.candidates if c["generation"] == self.generation]
        if len(current) != 4:
            raise ValueError("Complete all four matched candidate evaluations before selection")
        return sorted(current, key=lambda c: (-sum(c["scores"]), c["id"]))

    def propose(self):
        if self.generation >= 2:
            raise ValueError("Frozen two-generation allowance exhausted")
        ranked = self.ranked()
        self.generation += 1
        result = []
        for index in range(4):
            parent = ranked[0] if index == 0 else ranked[(index - 1) % 2]
            if index == 0:
                weights = np.asarray(parent["weights"], dtype=np.float32)
            elif self.method == "ea":
                weights = np.clip(np.asarray(parent["weights"])
                                  + self.rng.normal(0, .25, (CODE_DIM, FEATURE_DIM)), -1, 1)
            else:
                weights = self.rng.uniform(-1, 1, (CODE_DIM, FEATURE_DIM))
            result.append({"id": f"{self.method}-g{self.generation}-{index}",
                           "weights": weights.astype(np.float32).tolist(),
                           "parent": parent["id"] if index == 0 or self.method == "ea" else None})
        return result

    def state(self):
        return {"method": self.method, "rng": self.rng.bit_generator.state,
                "generation": self.generation, "candidates": self.candidates}
