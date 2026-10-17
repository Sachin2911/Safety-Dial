"""Composable execution of shared actor choices, without simulator task predicates.

Decisions happen outside ``act`` at fixed four-action boundaries. This module
changes environment-input targets; its command caps do not certify physical
velocity or safety. The runtime handles exhaustion of short policy chunks.
"""

import numpy as np

from helpers.maniguard_phase2_model import DEFAULT_SPEC, decode, forward, log_prob, validate_genome

CAPS = (None, 0.03, 0.06)
SMOOTHING = (1.0, 0.75, 0.5)
RELEASE_DELAYS = (0, 2, 4)
QUERY_INTERVALS = (8, 4, 16)
DECISION_INTERVAL = 4


def _vector(value, size, name):
    value = np.asarray(value)
    if value.shape != (size,) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite real numeric vector of length {size}")
    return value


def _choices(value):
    value = np.asarray(value)
    if (value.shape != (4,) or value.dtype.kind not in "iu"
            or np.any(value < 0) or np.any(value > 2)):
        raise ValueError("choices must be four integer indices in [0, 2]")
    return value.astype(np.int64, copy=True)


def _validate_frame(frame):
    if frame is None:
        raise ValueError("A robot frame is required when executing actions")
    if (getattr(frame, "prerequisite", None) is not None
            or getattr(frame, "target_engaged", None) is not None):
        raise ValueError("The primary phase-two controller rejects privileged task labels")
    frame.validate()
    _vector(frame.joints, 7, "frame joints")
    _vector(frame.fingers, 2, "frame fingers")
    _vector(frame.eef_position, 3, "frame end-effector position")


class Phase2Adapter:
    """One actor and explicit episode-local memory for every task family.

    The contextual reference uses only the measured tracking feature already in
    the common input: features[12:19] = clip((last input - measured q)/0.2).
    A magnitude above 0.5 therefore means tracking error above 0.10 radians.
    Fixed/contextual decisions have no learned-policy likelihood.
    """

    def __init__(self, genome, stochastic=False, seed=0, mode="learned", fixed_choices=None):
        self.genome = validate_genome(genome).copy()
        if type(stochastic) is not bool:
            raise ValueError("stochastic must be boolean")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if mode not in ("learned", "fixed", "contextual"):
            raise ValueError("mode must be learned, fixed or contextual")
        if mode != "fixed" and fixed_choices is not None:
            raise ValueError("Only fixed mode accepts fixed_choices")
        if stochastic and mode != "learned":
            raise ValueError("Reference controllers have deterministic decisions")
        self.stochastic = stochastic
        self.seed = seed
        self.mode = mode
        self.fixed_choices = _choices([0, 0, 0, 0] if fixed_choices is None else fixed_choices)
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.hidden = np.zeros(DEFAULT_SPEC.hidden_dim, dtype=np.float64)
        self.choices = self.fixed_choices.copy() if self.mode == "fixed" else np.zeros(4, dtype=int)
        self.last_executed = None
        self.last_proposal = None
        self.last_eef = None
        self.last_decision_step = None
        self.last_decision = None
        self.last_event = {}
        self.since_query = 0
        self.control_steps = 0
        self.opening_steps = 0
        self.release_budget = 0

    def decide(self, features, step):
        """Choose once at action 0, 4, 8, ...; callers retain the returned record."""
        features = _vector(features, DEFAULT_SPEC.input_dim, "features").astype(np.float64)
        expected = 0 if self.last_decision_step is None else self.last_decision_step + DECISION_INTERVAL
        if type(step) is not int or step != expected:
            raise ValueError(f"Decision step must be {expected}; decisions occur every four actions")
        if step != self.control_steps:
            raise ValueError("Decision step must match the number of applied control actions")
        before = self.hidden.copy()
        logits = None
        likelihood = None
        if self.mode == "learned":
            logits, self.hidden = forward(self.genome, features, self.hidden)
            self.choices = decode(logits, self.rng if self.stochastic else None)
            likelihood = log_prob(logits, self.choices)
        elif self.mode == "fixed":
            self.choices = self.fixed_choices.copy()
        else:
            self.choices = np.array([int(np.max(np.abs(features[12:19])) > 0.5), 0, 0, 0])
        self.last_decision_step = step
        record = {"features": features.tolist(), "choices": self.choices.tolist(),
                  "log_prob": likelihood, "logits": None if logits is None else logits.tolist(),
                  "hidden_before": before.tolist(), "hidden_after": self.hidden.tolist(),
                  "step": step, "mode": self.mode, "stochastic": self.stochastic}
        self.last_decision = record
        return record

    def act(self, proposal, frame):
        """Apply smoothing, then a measured-joint cap, then bounded opening delay."""
        _validate_frame(frame)
        proposal = _vector(proposal, 8, "proposal")
        choices = _choices(self.choices)
        cap = CAPS[choices[1]]
        alpha = SMOOTHING[choices[2]]
        action = proposal.copy()
        if alpha != 1.0 or cap is not None:
            # Float proposals retain their dtype. Integer fixtures must not
            # silently truncate a noninteger transformed command.
            if action.dtype.kind != "f":
                action = action.astype(np.float64)
            joints = np.asarray(frame.joints, dtype=np.float64)
            desired = proposal[:7].astype(np.float64)
            if alpha != 1.0:
                reference = joints if self.last_executed is None else self.last_executed[:7]
                desired = reference + alpha * (desired - reference)
            if cap is not None:
                desired = joints + np.clip(desired - joints, -cap, cap)
            if not np.isfinite(desired).all():
                raise ValueError("Motion composition produced nonfinite targets")
            action[:7] = desired
        opening = bool(proposal[-1] >= 0)
        delayed = False
        if opening:
            if self.opening_steps == 0:
                preceding_close = self.last_executed is not None and self.last_executed[-1] < 0
                self.release_budget = RELEASE_DELAYS[choices[3]] if preceding_close else 0
            self.opening_steps += 1
            delayed = self.opening_steps <= self.release_budget
            if delayed:
                action[-1] = -1.0
        else:
            self.opening_steps = 0
            self.release_budget = 0
        if not np.isfinite(action).all():
            raise ValueError("Adapter produced nonfinite commands")
        self.last_proposal = proposal.copy()
        self.last_eef = np.asarray(frame.eef_position).copy()
        event = {"kind": "phase2", "mode": self.mode, "choices": choices.tolist(),
                 "guard_blocked": False, "release_delayed": bool(delayed),
                 "changed": not np.array_equal(action, proposal),
                 "step_limit": cap, "smoothing": alpha,
                 "release_budget": int(self.release_budget),
                 "opening_steps": self.opening_steps, "since_query": self.since_query,
                 "decision_step": self.last_decision_step}
        self.last_event = event
        return action, event

    def applied(self, executed):
        """Retain actual environment input, which may differ from the adapted target."""
        self.last_executed = _vector(executed, 8, "executed action").copy()
        self.since_query += 1
        self.control_steps += 1

    def on_query(self, frame=None):
        if frame is not None:
            _validate_frame(frame)
        self.since_query = 0

    def replan_reason(self, frame=None):
        if frame is not None:
            _validate_frame(frame)
        query_choice = _choices(self.choices)[0]
        if self.since_query >= QUERY_INTERVALS[query_choice]:
            return ("baseline_horizon", "requested_requery", "extended_horizon")[query_choice]
        return None
