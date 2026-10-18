"""Causal correction-triggered feedback around frozen joint-target proposals."""

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

DOMAIN = {
    "free_step_limit": [None, 0.12, 0.08, 0.04],
    "grasp_step_limit": [None, 0.08, 0.04, 0.02],
    "correction_threshold": [0, 0.5, 1, 2, None],
    "minimum_interval": [4, 6],
}
TOLERANCE = 1e-6
NORMALIZER = np.sqrt(7.0) * 0.04


@dataclass(frozen=True)
class Config:
    free_step_limit: float | None = None
    grasp_step_limit: float | None = None
    correction_threshold: float | None = None
    minimum_interval: int = 4

    def __post_init__(self):
        for name, choices in DOMAIN.items():
            value = getattr(self, name)
            if isinstance(value, bool) or value not in choices:
                raise ValueError(f"Unsupported {name}: {value}")
        if type(self.minimum_interval) is not int:
            raise ValueError("Integer interval required")

    @classmethod
    def from_genome(cls, genome):
        if len(genome) != 4:
            raise ValueError("Four categorical coordinates required")
        values = {}
        for (name, choices), index in zip(DOMAIN.items(), genome, strict=True):
            if type(index) is not int or not 0 <= index < len(choices):
                raise ValueError("Invalid coordinate: " + name)
            values[name] = choices[index]
        return cls(**values)


def config_hash(config):
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


def conditions():
    return {
        "A": Config(),
        "B": Config(0.04, 0.02, None, 4),
        "C": Config(0.04, 0.02, 0, 4),
        "D": Config(0.04, 0.02, 0.5, 4),
    }


class FeedbackController:
    def __init__(self, config):
        self.controller = config
        self.reset()

    @property
    def horizon(self):
        return self.controller.minimum_interval if self.controller.correction_threshold == 0 else 8

    def reset(self):
        self.last_executed = self.previous_fingers = None
        self.pending_proposal = self.pending_fingers = None
        self.since_query = 0
        self.correction = self.last_correction = 0.0

    def on_query(self, frame):
        if self.pending_proposal is not None:
            raise ValueError("Unapplied proposal before query")
        self.since_query = 0
        self.correction = 0.0

    def act(self, proposal, frame):
        if self.pending_proposal is not None:
            raise ValueError("Previous proposal has not been applied")
        p, q, fingers = map(np.asarray, (proposal, frame.joints, frame.fingers))
        for value, shape in ((p, (8,)), (q, (7,)), (fingers, (2,))):
            if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"Expected finite numeric {shape}")
        if p.dtype.kind != "f":
            raise ValueError("Floating point policy commands required")
        grasp = bool(
            np.all((np.abs(fingers) >= 0.003) & (np.abs(fingers) <= 0.035))
            and self.previous_fingers is not None
            and np.max(np.abs(fingers - self.previous_fingers)) < 0.002
            and p[-1] < 0
        )
        cfg = self.controller
        cap = cfg.grasp_step_limit if grasp else cfg.free_step_limit
        adapted = p.copy()
        if cap is not None:
            adapted[:7] = q + np.clip(p[:7] - q, -cap, cap)
        event = {
            "kind": "correction_feedback_v1",
            "proposal_dtype": str(p.dtype),
            "joint_dtype": str(q.dtype),
            "finger_dtype": str(fingers.dtype),
            "effective_horizon": self.horizon,
            "grasp_evidence": grasp,
            "step_limit": cap,
            "cap_active": not np.array_equal(p, adapted),
            "changed": not np.array_equal(p, adapted),
            "guard_blocked": False,
            "correction_before": self.correction,
            "executed_since_query": self.since_query,
        }
        self.pending_proposal, self.pending_fingers = p.copy(), fingers.copy()
        return adapted, event

    def applied(self, executed):
        executed = np.asarray(executed)
        if (self.pending_proposal is None or executed.shape != (8,)
                or not np.isfinite(executed).all()):
            raise ValueError("One pending proposal and finite final environment input required")
        difference = self.pending_proposal[:7].astype(np.float64) - executed[:7]
        difference[np.abs(difference) <= TOLERANCE] = 0.0
        self.last_correction = float(np.linalg.norm(difference) / NORMALIZER)
        self.correction += self.last_correction
        self.last_executed = executed.copy()
        self.previous_fingers = self.pending_fingers
        self.pending_proposal = self.pending_fingers = None
        self.since_query += 1

    def replan_reason(self, frame):
        if self.pending_proposal is not None:
            raise ValueError("Scheduling must follow execution")
        if self.since_query >= 8:
            return "maximum_interval"
        cfg = self.controller
        if (cfg.correction_threshold is not None and self.since_query >= cfg.minimum_interval
                and self.correction >= cfg.correction_threshold):
            return "fixed_minimum" if cfg.correction_threshold == 0 else "correction_threshold"
        return None
