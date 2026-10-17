"""Compact execution adapters for the six fruits hypotheses.

Pure controller logic. Physical safety is measured by the unchanged benchmark,
not certified by these heuristics. Each experiment enables one hypothesis.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

KINDS = ("identity", "replan", "motion", "prerequisite", "release", "retry", "handback")


@dataclass(frozen=True)
class AdapterConfig:
    kind: str = "identity"
    horizon: int = 8
    min_query_interval: int = 1
    tracking_threshold: float = 0.10
    replan_on_gripper_change: bool = True
    free_step_limit: float = 0.12
    grasp_step_limit: float = 0.04
    smoothing: float = 1.0
    settle_steps: int = 0
    release_delay: int = 4
    min_closing_steps: int = 3
    min_aperture: float = 0.003
    max_aperture: float = 0.035
    aperture_stability: float = 0.002
    privileged: bool = False
    lift_margin: float = 0.001
    closure_confirm_steps: int = 1
    guard_requery_steps: int = 1
    handback_hold_steps: int = 0
    handback_requery: bool = True
    stall_window: int = 20
    stall_distance: float = 0.003
    demand_threshold: float = 0.03
    retry_distance: float = 0.015
    retry_steps: int = 5
    retry_limit: int = 2
    retry_cooldown: int = 40

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Unknown adapter: {self.kind}")
        positive_ints = ("horizon", "min_query_interval", "min_closing_steps",
                         "closure_confirm_steps", "guard_requery_steps", "stall_window",
                         "retry_steps")
        nonnegative_ints = ("settle_steps", "release_delay", "handback_hold_steps",
                            "retry_limit", "retry_cooldown")
        for key in positive_ints + nonnegative_ints:
            value = getattr(self, key)
            minimum = 1 if key in positive_ints else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if self.stall_window < 2 or self.min_query_interval > self.horizon:
            raise ValueError("Invalid stall window or query interval")
        positive = ("tracking_threshold", "free_step_limit", "grasp_step_limit",
                    "aperture_stability", "stall_distance", "demand_threshold", "retry_distance")
        for key in positive:
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not 0 < self.smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1]")
        if not 0 <= self.min_aperture < self.max_aperture or not np.isfinite(self.max_aperture):
            raise ValueError("Invalid aperture interval")
        if not np.isfinite(self.lift_margin) or self.lift_margin < 0:
            raise ValueError("lift_margin must be finite and nonnegative")
        if self.kind in ("prerequisite", "handback") and not self.privileged:
            raise ValueError("The initial prerequisite guard requires explicit privileged=True")


@dataclass(frozen=True)
class Frame:
    joints: np.ndarray
    fingers: np.ndarray
    eef_position: np.ndarray
    jacobian: np.ndarray | None = None  # world-frame translational Jacobian, shape (3, 7)
    prerequisite: bool | None = None
    target_engaged: bool | None = None

    def validate(self):
        for name, shape in (("joints", (7,)), ("fingers", (2,)), ("eef_position", (3,))):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name}: expected finite {shape}")
        if self.jacobian is not None and (
            self.jacobian.shape != (3, 7) or not np.isfinite(self.jacobian).all()
        ):
            raise ValueError("Invalid translational Jacobian")


class ExecutionAdapter:
    def __init__(self, config: AdapterConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.last_executed = None
        self.previous_fingers = None
        self.previous_proposed_gripper = None
        self.query_gripper = None
        self.since_query = 0
        self.force_replan = False
        self.closing_steps = 0
        self.opening_steps = 0
        self.delay_active = False
        self.closure_run = 0
        self.prerequisite_seen = False
        self.guard_run = 0
        self.settling = 0
        self.handback_remaining = 0
        self.was_blocked = False
        self.position_history = deque(maxlen=self.config.stall_window)
        self.demand_history = deque(maxlen=self.config.stall_window)
        self.retry_count = 0
        self.retry_remaining = 0
        self.retry_wait = 0
        self.retry_goal = None
        self.retry_gripper = None
        self.last_event = {}

    def on_query(self, frame):
        frame.validate()
        self.since_query = 0
        self.force_replan = False
        self.query_gripper = np.asarray(frame.fingers).copy()

    def grasp_evidence(self, frame):
        aperture = np.abs(frame.fingers)
        in_range = np.all((aperture >= self.config.min_aperture)
                          & (aperture <= self.config.max_aperture))
        stable = self.previous_fingers is not None and np.max(
            np.abs(frame.fingers - self.previous_fingers)
        ) < self.config.aperture_stability
        return bool(in_range and stable)

    def act(self, proposal, frame):
        frame.validate()
        proposal = np.asarray(proposal)
        if proposal.shape != (8,) or not np.isfinite(proposal).all():
            raise ValueError("Expected a finite eight-dimensional absolute-joint action")
        action = proposal.copy()
        cfg = self.config
        opening = bool(proposal[-1] >= 0)
        grasp = self.grasp_evidence(frame)
        event = {"kind": cfg.kind, "guard_blocked": False, "release_delayed": False,
                 "retry_started": False, "retry_active": False, "grasp_evidence": grasp}

        if cfg.kind == "motion":
            switched = (self.previous_proposed_gripper is not None
                        and opening != self.previous_proposed_gripper)
            if switched:
                self.settling = cfg.settle_steps
            reference = frame.joints if self.last_executed is None else self.last_executed[:7]
            desired = reference + cfg.smoothing * (proposal[:7] - reference)
            cap = cfg.grasp_step_limit if grasp and not opening else cfg.free_step_limit
            action[:7] = frame.joints + np.clip(desired - frame.joints, -cap, cap)
            if self.settling:
                action[:7] = frame.joints
                self.settling -= 1
            event["step_limit"] = cap

        elif cfg.kind == "release":
            if opening:
                if self.opening_steps == 0:
                    self.delay_active = self.closing_steps >= cfg.min_closing_steps and grasp
                self.opening_steps += 1
                if self.delay_active and self.opening_steps <= cfg.release_delay:
                    action[-1] = -1.0
                    event["release_delayed"] = True
                self.closing_steps = 0
            else:
                self.opening_steps = 0
                self.delay_active = False
                self.closing_steps = self.closing_steps + 1 if grasp else 0

        elif cfg.kind in ("prerequisite", "handback"):
            if frame.prerequisite is None or frame.target_engaged is None or frame.jacobian is None:
                raise ValueError("Prerequisite guard requires declared object state and kinematics")
            self.closure_run = self.closure_run + 1 if frame.prerequisite else 0
            # Weak-until prerequisite: once confirmed it remains discharged.
            self.prerequisite_seen |= self.closure_run >= cfg.closure_confirm_steps
            upward = float(frame.jacobian[2] @ (proposal[:7] - frame.joints))
            blocked = bool(frame.target_engaged and not self.prerequisite_seen
                           and upward > cfg.lift_margin)
            event.update(guard_blocked=blocked, predicted_lift_linearized=upward,
                         prerequisite_seen=self.prerequisite_seen)
            if blocked:
                action[:7] = frame.joints
                self.guard_run += 1
                self.was_blocked = True
                if self.guard_run % cfg.guard_requery_steps == 0:
                    self.force_replan = True
            else:
                self.guard_run = 0
                if self.was_blocked:
                    self.was_blocked = False
                    if cfg.kind == "handback":
                        self.handback_remaining = cfg.handback_hold_steps
                        self.force_replan |= cfg.handback_requery
                if self.handback_remaining:
                    action[:7] = frame.joints
                    self.handback_remaining -= 1
                    if self.handback_remaining == 0:
                        self.force_replan |= cfg.handback_requery
            event["handback_remaining"] = self.handback_remaining

        elif cfg.kind == "retry":
            if frame.jacobian is None:
                raise ValueError("Retry primitive requires proprioceptive kinematics")
            self.position_history.append(np.asarray(frame.eef_position).copy())
            self.demand_history.append(float(np.max(np.abs(proposal[:7] - frame.joints))))
            if self.retry_wait:
                self.retry_wait -= 1
            ready = (len(self.position_history) == cfg.stall_window
                     and self.retry_count < cfg.retry_limit and not self.retry_wait
                     and not self.retry_remaining)
            if ready:
                positions = np.asarray(self.position_history)
                excursion = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
                demand = float(np.mean(self.demand_history))
                desired_motion = frame.jacobian @ (proposal[:7] - frame.joints)
                norm = np.linalg.norm(desired_motion)
                if excursion < cfg.stall_distance and demand >= cfg.demand_threshold and norm > 1e-8:
                    retreat = -cfg.retry_distance * desired_motion / norm
                    jac = frame.jacobian
                    joint_delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.05**2 * np.eye(3), retreat)
                    self.retry_goal = frame.joints + joint_delta
                    self.retry_gripper = float(proposal[-1])
                    self.retry_remaining = cfg.retry_steps
                    self.retry_count += 1
                    event["retry_started"] = True
            if self.retry_remaining:
                action[:7] = frame.joints + np.clip(
                    self.retry_goal - frame.joints, -cfg.free_step_limit, cfg.free_step_limit,
                )
                action[-1] = self.retry_gripper
                self.retry_remaining -= 1
                event["retry_active"] = True
                if self.retry_remaining == 0:
                    self.force_replan = True
                    self.retry_wait = cfg.retry_cooldown
                    self.position_history.clear()
                    self.demand_history.clear()
            event["retry_count"] = self.retry_count

        self.previous_fingers = np.asarray(frame.fingers).copy()
        self.previous_proposed_gripper = opening
        event["changed"] = not np.array_equal(action, proposal)
        self.last_event = event
        return action, event

    def applied(self, executed):
        """Use the final action AFTER environment clipping, not the proposal."""
        self.last_executed = np.asarray(executed).copy()
        self.since_query += 1

    def should_replan(self, frame):
        return self.replan_reason(frame) is not None

    def replan_reason(self, frame):
        frame.validate()
        if self.since_query < self.config.min_query_interval:
            return None
        if self.force_replan:
            return "intervention_requery"
        if self.config.kind != "replan":
            return None
        error = 0.0 if self.last_executed is None else np.max(
            np.abs(self.last_executed[:7] - frame.joints)
        )
        changed = self.query_gripper is not None and np.max(
            np.abs(frame.fingers - self.query_gripper)
        ) >= self.config.aperture_stability
        if error >= self.config.tracking_threshold:
            return "joint_tracking_error"
        if self.config.replan_on_gripper_change and changed:
            return "gripper_state_change"
        return None
