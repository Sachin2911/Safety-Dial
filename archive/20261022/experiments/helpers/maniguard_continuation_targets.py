"""Source-matched raw-position target adapter; no simulator or runtime discovery.

The caller must supply and later validate a captured runtime controller contract.
CPU agreement with a declared snapshot does not establish physical installation.
"""

import copy
import hashlib
import json

import numpy as np

from helpers.maniguard_model_control import finite

ARM_PRESET = {"name": "JointController", "motor_type": "position",
              "command_input_limits": None, "command_output_limits": None,
              "use_delta_commands": False, "use_impedances": False}
SOURCE_KEYS = {"joint_controller", "controller_base", "controller_preset"}


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


class NativeJointResidualAdapter:
    """Apply a bounded residual after native limited-position target clipping.

    The supported source preset passes absolute radians through without command
    scaling, then BaseController.clip_control clips limited position joints.
    Correcting the raw command instead can produce no effective target change
    when that command is already beyond a joint limit. This adapter starts from
    the clipped target. It preserves original environment-command coordinates
    wherever the effective target does not change, including exact zero residual.

    All seven arm joints must have finite limits. Different controller modes,
    normalization, mappings and unlimited joints require a different adapter.
    Outputs are desired targets, never achieved positions or executed actions.
    """

    def __init__(self, snapshot, residual):
        if (snapshot.get("schema") != "maniguard-raw-joint-runtime-snapshot-v1"
                or snapshot.get("evidence_origin") not in (
                    "synthetic_engineering_fixture", "captured_live_controller")
                or snapshot.get("action_normalize") is not False
                or snapshot.get("arm_action_indices") != list(range(7))
                or snapshot.get("gripper_action_indices") != [7]):
            raise ValueError("Unsupported native action contract")
        arm = snapshot.get("arm_controller", {})
        for key, expected in ARM_PRESET.items():
            value = arm.get(key, "missing")
            if (value != expected or (expected is False and value is not False)
                    or (expected is None and value is not None)):
                raise ValueError("Unsupported native arm controller preset: " + key)
        gripper = snapshot.get("gripper_controller", {})
        if (gripper.get("name") != "MultiFingerGripperController"
                or gripper.get("mode") != "binary"
                or gripper.get("command_input_limits", "missing") is not None
                or snapshot.get("gripper_convention") != "+1_open_-1_close"):
            raise ValueError("Unsupported native gripper controller contract")
        limited = snapshot.get("joint_has_limits", [])
        if len(limited) != 7 or any(type(value) is not bool or not value for value in limited):
            raise ValueError("All seven arm joints must have confirmed finite position limits")
        self.lower = finite(snapshot.get("joint_lower_limits"), (7,), "runtime lower limits").copy()
        self.upper = finite(snapshot.get("joint_upper_limits"), (7,), "runtime upper limits").copy()
        sources = snapshot.get("source_sha256", {})
        if (np.any(self.lower >= self.upper) or set(sources) != SOURCE_KEYS
                or not all(_sha256(value) for value in sources.values())
                or not _sha256(snapshot.get("runtime_evidence_sha256"))):
            raise ValueError("Invalid runtime limits or source/evidence hashes")
        if not np.array_equal(residual.lower, self.lower) or not np.array_equal(
                residual.upper, self.upper):
            raise ValueError("Residual and native runtime joint bounds differ")
        self.snapshot = copy.deepcopy(snapshot)
        self.contract_sha256 = hashlib.sha256(json.dumps(
            self.snapshot, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.residual = residual
        self.lower.setflags(write=False)
        self.upper.setflags(write=False)

    def canonical_targets(self, native_environment_prefix):
        """Compute source-defined desired targets, without inferring robot motion."""
        native = np.asarray(native_environment_prefix, dtype=float)
        if (native.ndim != 2 or native.shape[1] != 8 or not len(native)
                or not np.isfinite(native).all() or not np.isin(native[:, 7], [-1., 1.]).all()):
            raise ValueError("Expected finite raw absolute commands with a +/-1 gripper")
        targets = native.copy()
        targets[:, :7] = np.clip(native[:, :7], self.lower, self.upper)
        return targets

    def propose(self, native_environment_prefix, actor_features, learned_beliefs):
        canonical = self.canonical_targets(native_environment_prefix)
        native = np.asarray(native_environment_prefix, dtype=float)
        residual = self.residual.propose(canonical, actor_features, learned_beliefs)
        target = residual["proposed_prefix"]
        changed = target[:, :7] != canonical[:, :7]
        proposed = native.copy()
        proposed[:, :7] = np.where(changed, target[:, :7], native[:, :7])
        proposed[:, 7] = target[:, 7]
        if not np.array_equal(self.canonical_targets(proposed), target):
            raise ValueError("Residual environment commands do not reproduce desired targets")
        return {
            "native_environment_prefix": native.copy(), "native_joint_target_prefix": canonical,
            "residual_joint_target_prefix": target.copy(), "proposed_environment_prefix": proposed,
            "effective_arm_target_delta": target[:, :7] - canonical[:, :7],
            "environment_input_arm_delta": proposed[:, :7] - native[:, :7],
            "native_clipped_coordinates": native[:, :7] != canonical[:, :7],
            "coefficients": residual["coefficients"],
            "target_contract_sha256": self.contract_sha256,
            "residual_contract_sha256": residual["controller_contract_sha256"],
            "runtime_evidence_origin": self.snapshot["evidence_origin"],
            "runtime_evidence_independently_verified": False,
            "filtered_environment_prefix": None, "executed_environment_prefix": None,
            "achieved_joint_positions": None, "dispatch_authorized": False,
            "new_executed_controls": 0,
            "scope": "Declared limited-position clipping only; runtime validation is external.",
        }

    def reset(self):
        self.residual.reset()

    def state_dict(self):
        return {"schema": "maniguard-native-target-adapter-state-v1",
                "target_contract_sha256": self.contract_sha256,
                "residual_state": self.residual.state_dict()}

    def load_state_dict(self, state):
        if (set(state) != {"schema", "target_contract_sha256", "residual_state"}
                or state["schema"] != "maniguard-native-target-adapter-state-v1"
                or state["target_contract_sha256"] != self.contract_sha256):
            raise ValueError("Adapter memory belongs to a different runtime contract")
        self.residual.load_state_dict(state["residual_state"])
