"""Four categorical execution parameters and a fixed, external intervention dial.

Only command and proprioceptive inputs enter this controller. ``applied`` must receive
the final environment input, after dial blending and action-space clipping.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

DOMAIN = {
    "free_step_limit": [None, 0.12, 0.08, 0.04],
    "grasp_step_limit": [None, 0.08, 0.04, 0.02],
    "smoothing": [1.0, 0.75, 0.5],
    "horizon": [4, 8, 16],
}
NEUTRAL = (0, 0, 0, 1)


def genome_hash(genome):
    config = Controller.from_genome(genome)
    body = {"genome": list(genome), "decoded": asdict(config)}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Controller:
    free_step_limit: float | None = None
    grasp_step_limit: float | None = None
    smoothing: float = 1.0
    horizon: int = 8

    def __post_init__(self):
        for name, domain in DOMAIN.items():
            value = getattr(self, name)
            if isinstance(value, bool) or value not in domain:
                raise ValueError(f"Unsupported {name}: {value}")
        if type(self.horizon) is not int:
            raise ValueError("Horizon must be an integer")

    @classmethod
    def from_genome(cls, genome):
        if len(genome) != 4:
            raise ValueError("Exactly four categorical coordinates are required")
        values = {}
        for (name, domain), index in zip(DOMAIN.items(), genome, strict=True):
            if type(index) is not int or not 0 <= index < len(domain):
                raise ValueError(f"Invalid {name} coordinate: {index}")
            values[name] = domain[index]
        return cls(**values)


class SafetyDial:
    def __init__(self, controller, dial):
        if type(dial) not in (int, float) or not math.isfinite(dial) or not 0 <= dial <= 1:
            raise ValueError("Dial must be finite and in [0,1]")
        self.controller = controller
        self._dial = float(dial)
        self.reset()

    @property
    def dial(self):
        return self._dial

    @property
    def horizon(self):
        return math.floor(8 + self.dial * (self.controller.horizon - 8) + 0.5)

    def reset(self):
        self.last_executed = None
        self.previous_fingers = None
        self.pending_fingers = None
        self.since_query = 0

    def on_query(self, frame):
        if self.pending_fingers is not None:
            raise ValueError("Unapplied proposal before query")
        self.since_query = 0

    def act(self, proposal, frame):
        if self.pending_fingers is not None:
            raise ValueError("Previous proposal has not been applied")
        p, q, fingers = (np.asarray(proposal), np.asarray(frame.joints), np.asarray(frame.fingers))
        for value, shape in ((p, (8,)), (q, (7,)), (fingers, (2,))):
            if (
                value.shape != shape
                or value.dtype.kind not in "fiu"
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"Expected finite numeric {shape}")
        if p.dtype.kind != "f":
            raise ValueError("Policy commands must use a floating-point dtype")
        grasp = bool(
            np.all((np.abs(fingers) >= 0.003) & (np.abs(fingers) <= 0.035))
            and self.previous_fingers is not None
            and np.max(np.abs(fingers - self.previous_fingers)) < 0.002
            and p[-1] < 0
        )
        cfg = self.controller
        cap = cfg.grasp_step_limit if grasp else cfg.free_step_limit
        anchor = p.copy()
        reference = q if self.last_executed is None else self.last_executed[:7]
        # Both action-neutral configurations and the neutral genome have an exact path.
        if not (cfg.smoothing == 1 and cap is None):
            desired = reference + cfg.smoothing * (p[:7] - reference)
            anchor[:7] = desired if cap is None else q + np.clip(desired - q, -cap, cap)
        blended = p.copy()
        if self.dial == 1:
            blended = anchor.copy()
        elif self.dial != 0:
            blended[:7] = p[:7] + self.dial * (anchor[:7] - p[:7])
        if not np.isfinite(blended).all():
            raise ValueError("Nonfinite controller output")
        self.pending_fingers = fingers.copy()
        event = {
            "kind": "phase3_part1",
            "dial": self.dial,
            "proposal_dtype": str(p.dtype),
            "joint_dtype": str(q.dtype),
            "finger_dtype": str(fingers.dtype),
            "requested_horizon": cfg.horizon,
            "effective_horizon": self.horizon,
            "reference": reference.tolist(),
            "grasp_evidence": grasp,
            "step_limit": cap,
            "cap_active": cap is not None
            and not np.array_equal(anchor[:7], reference + cfg.smoothing * (p[:7] - reference)),
            "anchor": anchor.tolist(),
            "changed": not np.array_equal(p, blended),
            "guard_blocked": False,
        }
        return blended, event

    def applied(self, executed):
        executed = np.asarray(executed)
        if (
            self.pending_fingers is None
            or executed.shape != (8,)
            or not np.isfinite(executed).all()
        ):
            raise ValueError("Applied history requires one pending action and finite final input")
        self.last_executed = executed.copy()
        self.previous_fingers = self.pending_fingers
        self.pending_fingers = None
        self.since_query += 1

    def replan_reason(self, frame):
        return None  # Prefix consumption and chunk exhaustion belong to the executor.
