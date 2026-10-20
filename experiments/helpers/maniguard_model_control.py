"""Prospective control components; no simulator, encoder or policy-service imports."""

from dataclasses import dataclass

import numpy as np

from helpers.maniguard_latent import arm_basis


def finite(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError("Invalid " + name)
    return value


class PersistentNoiseController:
    """Five bounded coefficients, including explicit gripper-noise support.

    Inputs are the 44 causal actor features and five learned task-state beliefs.
    Beliefs are predictions, not privileged monitor labels. Identity weights keep
    zero offset; the caller must retain the native Gaussian draw and RNG path.
    """

    def __init__(self, weights, *, arm_scale=.35, gripper_scale=2., persistence=.8):
        self.weights = finite(weights, (5, 49), "controller weights").copy()
        if np.any(np.abs(self.weights) > 1):
            raise ValueError("Controller weights exceed [-1,1]")
        if (not np.isfinite([arm_scale, gripper_scale, persistence]).all()
                or not 0 <= arm_scale <= 1 or not 0 <= gripper_scale <= 2
                or not 0 <= persistence < 1):
            raise ValueError("Invalid prospective control bounds")
        self.arm_scale, self.gripper_scale, self.persistence = arm_scale, gripper_scale, persistence
        self.previous = None

    def reset(self):
        self.previous = None

    def offset(self, actor_features, learned_beliefs, *, horizon=16, action_dim=32):
        if type(horizon) is not int or horizon < 1 or type(action_dim) is not int or action_dim < 8:
            raise ValueError("Invalid native noise dimensions")
        features = finite(actor_features, (44,), "causal actor features")
        beliefs = finite(learned_beliefs, (5,), "learned task-state beliefs")
        if np.any(np.abs(features) > 1) or np.any((beliefs < 0) | (beliefs > 1)):
            raise ValueError("Features or learned beliefs exceed their declared bounds")
        raw = np.tanh(self.weights @ np.r_[features, beliefs])
        code = raw if self.previous is None else (
            self.persistence * self.previous + (1 - self.persistence) * raw)
        self.previous = code.copy()
        shift = self.arm_scale * (arm_basis(action_dim).astype(float) @ code[:4])
        shift[7] += self.gripper_scale * code[4]
        return np.broadcast_to(shift, (horizon, action_dim)).copy()


class SequenceProgress:
    """Evaluation-only ordered milestones, separate from primary terminal utility."""

    def __init__(self):
        self.stage, self.lifted, self.violated = 0, False, False

    def observe(self, *, closed, held, supported, in_goal, violated):
        if any(type(v) is not bool for v in (closed, held, supported, in_goal, violated)):
            raise ValueError("Progress requires complete factual Boolean labels")
        self.violated |= violated
        old = self.stage
        if not self.violated:
            if not self.lifted and supported and closed:
                self.stage = max(self.stage, 1)
                if held:
                    self.stage = max(self.stage, 2)
            if not supported and not self.lifted:
                if self.stage >= 2 and closed and held:
                    self.stage = max(self.stage, 3)
                self.lifted = True
            if self.stage >= 3 and closed and held and in_goal:
                self.stage = 4
        return {"ordered_stage": self.stage, "new_milestones": self.stage - old,
                "secondary_fitness": 0 if self.violated else self.stage,
                "physical_violation": self.violated,
                "scope": "Training supervision only; not an actor input or completion verdict."}


class RosarlPenalty:
    """ROSARL Algorithm 1's practical value-extrema penalty interface.

    Values must come from a declared external value estimator. Model progress or
    predicted risk is not silently treated as a value function. Freeze one shared
    estimate before comparative candidate scoring. No theorem extension is claimed.
    """

    def __init__(self, *, value_source):
        if value_source not in ("factual_value_estimator", "synthetic_engineering_reference"):
            raise ValueError("Declare the source of value estimates")
        self.value_source = value_source
        self.rmin = self.rmax = self.vmin = self.vmax = 0.
        self.frozen = False

    def observe(self, reward, value):
        if self.frozen or not np.isfinite([reward, value]).all():
            raise ValueError("Cannot alter a frozen estimate or use nonfinite observations")
        self.rmin, self.rmax = min(self.rmin, reward), max(self.rmax, reward)
        self.vmin = min(self.vmin, self.rmin, value)
        self.vmax = max(self.vmax, self.rmax, value)

    def freeze(self):
        self.frozen = True
        return self.snapshot()

    @property
    def penalty(self):
        if not self.frozen:
            raise ValueError("Freeze the shared penalty before comparing candidates")
        return min(self.rmin, self.vmin - self.vmax)

    def reward(self, factual_reward, violated):
        if not self.frozen:
            raise ValueError("Freeze the shared penalty before comparing candidates")
        if type(violated) is not bool or not np.isfinite(factual_reward):
            raise ValueError("Factual reward and violation are required")
        return self.penalty if violated else float(factual_reward)

    def snapshot(self):
        return {"value_source": self.value_source, "reward_min": self.rmin,
                "reward_max": self.rmax, "value_min": self.vmin, "value_max": self.vmax,
                "frozen": self.frozen,
                "minmax_estimate": min(self.rmin, self.vmin - self.vmax),
                "scope": "Practical empirical estimator; no finite-budget POMDP safety guarantee."}


def predicted_progress(stages):
    stages = np.asarray(stages, dtype=float)
    if stages.ndim != 2 or stages.shape[1] != 5 or not np.isfinite(stages).all():
        raise ValueError("Expected five predicted stage scores per candidate")
    if np.any((stages < 0) | (stages > 1)):
        raise ValueError("Stage scores must lie in [0,1]")
    closed, held, goal = stages[:, 0], stages[:, 1], stages[:, 2]
    return closed + closed * held + closed * held * goal


@dataclass
class FilterDecision:
    status: str
    selected_index: int | None
    proposed: np.ndarray
    filtered: np.ndarray | None
    risk_upper_scores: list
    support_distances: list
    executed: np.ndarray | None = None

    def record_executed(self, actions):
        if self.filtered is None or self.executed is not None:
            raise ValueError("No selected action, or execution has already been recorded")
        actions = np.asarray(actions, dtype=float)
        if (actions.ndim != 2 or actions.shape[1] != 8 or not len(actions)
                or len(actions) > len(self.filtered) or not np.isfinite(actions).all()
                or not np.array_equal(actions, self.filtered[:len(actions)])):
            raise ValueError("Executed environment inputs must be a factual selected prefix")
        self.executed = actions.copy()


def filter_prefixes(prefixes, prediction, upper_scores, *, gate_pass, risk_limit=.05,
                    support_radius=2.):
    """Preserve an accepted proposal or select a supported alternative; else abstain.

    Index zero is the proposal. This function returns a decision, never executes
    it. Abstention is unfinished execution, not measured successful safety.
    """
    prefixes = np.asarray(prefixes, dtype=float)
    if (prefixes.ndim != 3 or prefixes.shape[2] != 8 or not len(prefixes)
            or prefixes.shape[1] < 1 or not np.isfinite(prefixes).all()):
        raise ValueError("Expected decoded physical candidate prefixes")
    n = len(prefixes)
    upper = finite(upper_scores, (n,), "upper risk scores")
    support = finite(prediction["support_distance"], (n,), "support distances")
    if (type(gate_pass) is not bool or not np.isfinite([risk_limit, support_radius]).all()
            or not 0 <= risk_limit <= 1 or support_radius <= 0
            or np.any((upper < 0) | (upper > 1)) or np.any(support < 0)):
        raise ValueError("Invalid filtering contract")
    progress = predicted_progress(prediction["stage_mean"])
    if len(progress) != n:
        raise ValueError("Stage predictions do not match candidate prefixes")
    eligible = (upper <= risk_limit) & (support <= support_radius) & gate_pass
    selected = None
    if eligible[0]:
        selected = 0
    elif eligible.any():
        scale = np.r_[np.full(7, np.pi), 1.]
        distance = np.mean(((prefixes - prefixes[0]) / scale)**2, axis=(1, 2))
        selected = min(np.flatnonzero(eligible), key=lambda i: (distance[i], -progress[i], i))
    status = "abstained" if selected is None else "accepted" if selected == 0 else "replaced"
    return FilterDecision(status, int(selected) if selected is not None else None,
                          prefixes[0].copy(),
                          None if selected is None else prefixes[selected].copy(),
                          upper.tolist(), support.tolist())


def select_candidates(candidate_ids, prediction, upper_scores, slots, ledger, request_id, *,
                      gate_pass, penalty, seed=6102001, support_radius=2.):
    """Allocate physical evaluations using a short-prefix surrogate, not fake fitness."""
    ids = list(candidate_ids)
    if (len(set(ids)) != len(ids) or type(slots) is not int or not 1 <= slots <= len(ids)
            or not np.isfinite(penalty) or penalty > 0 or type(gate_pass) is not bool):
        raise ValueError("Invalid prospective candidate allocation")
    upper = finite(upper_scores, (len(ids),), "candidate risk scores")
    support = finite(prediction["support_distance"], (len(ids),), "candidate support")
    progress = predicted_progress(prediction["stage_mean"])
    if len(progress) != len(ids) or np.any((upper < 0) | (upper > 1)):
        raise ValueError("Candidate predictions and identities differ")
    ledger.admit(request_id, "candidate_screenings", len(ids))
    try:
        utility = progress + penalty * upper
        if gate_pass and np.all(support <= support_radius):
            selected = sorted(range(len(ids)), key=lambda i: (-utility[i], ids[i]))[:slots]
            mode = "model_assisted_short_prefix_priority"
        else:
            selected = np.random.default_rng(seed).choice(len(ids), slots, replace=False).tolist()
            mode = "declared_model_free_allocation_fallback"
        result = {"selected_ids": [ids[i] for i in selected], "mode": mode,
                  "surrogate_priority": dict(zip(ids, utility.tolist(), strict=True)),
                  "physical_fitness": None,
                  "scope": "Priority only. Full factual episode outcomes remain required."}
    except Exception:
        ledger.finish(request_id, "failed")
        raise
    ledger.finish(request_id, "returned")
    return result


def mutate_candidates(parent_weights, pool_size, *, seed, sigma=.25):
    """Bounded alternating-parent proposal pool, also usable with a fixed parent."""
    parents = np.asarray(parent_weights, dtype=float)
    if (parents.ndim != 3 or not len(parents) or not np.isfinite(parents).all()
            or np.any(np.abs(parents) > 1) or type(pool_size) is not int or pool_size < 1
            or not np.isfinite(sigma) or sigma <= 0):
        raise ValueError("Invalid parent pool or mutation specification")
    rng = np.random.default_rng(seed)
    return [np.clip(parents[i % len(parents)] + rng.normal(0, sigma, parents.shape[1:]), -1, 1)
            for i in range(pool_size)]


class ModelAssistedController:
    """Wire one factual-anchor prediction to filtering and candidate allocation.

    Prefixes must already be decoded at the actual observation. This is not an
    imagined multi-query VLA rollout. Physical evaluation and execution remain
    external and cannot be inferred from this class's model predictions.
    """

    def __init__(self, model, gate, qualification, penalty):
        if not penalty.frozen:
            raise ValueError("A common frozen ROSARL estimate is required")
        self.model, self.gate, self.qualification, self.penalty = (
            model, gate, qualification, penalty)

    def screen(self, current, previous, history_valid, decoded_prefixes, candidate_ids, slots,
               ledger, request_id):
        prefixes = np.asarray(decoded_prefixes, dtype=float)
        if prefixes.ndim != 3 or prefixes.shape[2] != 8 or len(prefixes) != len(candidate_ids):
            raise ValueError("Candidate identities and decoded prefixes differ")
        n = len(prefixes)
        current = finite(current, (44,), "current factual features")
        previous = finite(previous, (44,), "causal previous features")
        if type(history_valid) is not bool:
            raise ValueError("History availability must be explicit")
        prediction = self.model.predict(np.repeat(current[None], n, axis=0),
                                         np.repeat(previous[None], n, axis=0),
                                         np.full(n, history_valid), prefixes, ledger,
                                         request_id + "/prediction")
        upper = self.model.risk_upper_score(prediction, self.gate)
        passed = (self.qualification.get("prediction_gate_pass") is True
                  and self.model.calibration["ready"] is True)
        decision = filter_prefixes(prefixes, prediction, upper, gate_pass=passed,
                                   risk_limit=self.gate["risk_score_limit"],
                                   support_radius=self.model.spec["support_radius"])
        allocation = select_candidates(candidate_ids, prediction, upper, slots, ledger,
                                        request_id + "/allocation", gate_pass=passed,
                                        penalty=self.penalty.penalty,
                                        support_radius=self.model.spec["support_radius"])
        return {"decision": decision, "allocation": allocation, "prediction": prediction,
                "rosarl": self.penalty.snapshot(), "physical_fitness": None,
                "new_executed_controls": 0}


def decode_with_budget(decode, request_id, ledger, *args, **kwargs):
    """Explicit caller-supplied decoder bridge; no service discovery or auto-launch."""
    ledger.admit(request_id, "decoder_requests", 1)
    try:
        result = decode(*args, **kwargs)
    except Exception:
        ledger.finish(request_id, "failed")
        raise
    ledger.finish(request_id, "returned")
    return result
