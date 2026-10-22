"""Prospective bounded residual on native absolute joint targets and gripper.

No simulator, decoder, model prediction, safety label or execution is used here.
The numerical defaults are prototype settings, not validated physical limits.
"""

import hashlib
import json

import numpy as np

from helpers.maniguard_model_control import finite


class BoundedActionResidual:
    """A direct-action comparator using the same 44 causal features and five beliefs.

    Runtime joint limits must be supplied. Input commands must already use the
    native seven absolute joint targets and binarized gripper (+1 open, -1 close).
    A neutral gripper residual preserves the native command in each prefix step.
    Zero weights preserve every valid native command exactly. No zero joint vector
    is substituted for holding position, and no blanket gripper-close rule is used.
    """

    def __init__(self, weights, joint_lower, joint_upper, *, max_arm_delta=.05,
                 persistence=.8, gripper_threshold=.5):
        self.weights = finite(weights, (8, 49), "residual weights").copy()
        self.lower = finite(joint_lower, (7,), "runtime joint lower bounds").copy()
        self.upper = finite(joint_upper, (7,), "runtime joint upper bounds").copy()
        self.delta = finite(
            np.broadcast_to(max_arm_delta, (7,)), (7,), "arm residual bounds").copy()
        if (np.any(np.abs(self.weights) > 1) or np.any(self.lower >= self.upper)
                or np.any(self.delta < 0) or np.any(self.delta > self.upper - self.lower)
                or not np.isfinite([persistence, gripper_threshold]).all()
                or not 0 <= persistence < 1 or not 0 < gripper_threshold < 1):
            raise ValueError("Invalid prospective residual contract")
        self.persistence, self.threshold = float(persistence), float(gripper_threshold)
        contract = {"weights": self.weights.tolist(), "joint_lower": self.lower.tolist(),
                    "joint_upper": self.upper.tolist(), "max_arm_delta": self.delta.tolist(),
                    "persistence": self.persistence, "gripper_threshold": self.threshold}
        self.contract_sha256 = hashlib.sha256(json.dumps(
            contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
        for value in (self.weights, self.lower, self.upper, self.delta):
            value.setflags(write=False)
        self.previous = None

    def reset(self):
        self.previous = None

    def state_dict(self):
        return {"schema": "maniguard-bounded-residual-state-v1",
                "contract_sha256": self.contract_sha256,
                "previous": None if self.previous is None else self.previous.tolist()}

    def load_state_dict(self, state):
        if (set(state) != {"schema", "contract_sha256", "previous"}
                or state["schema"] != "maniguard-bounded-residual-state-v1"
                or state["contract_sha256"] != self.contract_sha256):
            raise ValueError("Residual checkpoint belongs to a different controller contract")
        previous = None if state["previous"] is None else finite(
            state["previous"], (8,), "residual memory").copy()
        if previous is not None and np.any(np.abs(previous) > 1):
            raise ValueError("Residual checkpoint memory exceeds bounds")
        self.previous = previous

    def propose(self, native_prefix, actor_features, learned_beliefs):
        native = np.asarray(native_prefix, dtype=float)
        if (native.ndim != 2 or native.shape[1] != 8 or not len(native)
                or not np.isfinite(native).all()
                or np.any(native[:, :7] < self.lower) or np.any(native[:, :7] > self.upper)
                or not np.isin(native[:, 7], [-1., 1.]).all()):
            raise ValueError(
                "Expected native absolute targets within runtime bounds and +/-1 gripper")
        features = finite(actor_features, (44,), "causal actor features")
        beliefs = finite(learned_beliefs, (5,), "learned task-state beliefs")
        if np.any(np.abs(features) > 1) or np.any((beliefs < 0) | (beliefs > 1)):
            raise ValueError("Features or learned beliefs exceed their declared bounds")
        raw = np.tanh(self.weights @ np.r_[features, beliefs])
        code = raw if self.previous is None else (
            self.persistence * self.previous + (1 - self.persistence) * raw)
        proposed = native.copy()
        proposed[:, :7] = np.clip(native[:, :7] + self.delta * code[:7], self.lower, self.upper)
        if code[7] >= self.threshold:
            proposed[:, 7] = 1.
        elif code[7] <= -self.threshold:
            proposed[:, 7] = -1.
        self.previous = code.copy()
        return {"native_prefix": native.copy(), "proposed_prefix": proposed,
                "arm_delta": proposed[:, :7] - native[:, :7],
                "gripper_overrides": proposed[:, 7] != native[:, 7], "coefficients": code.copy(),
                "filtered_prefix": None, "executed_prefix": None,
                "controller_contract_sha256": self.contract_sha256,
                "new_executed_controls": 0, "physical_validation": False}
