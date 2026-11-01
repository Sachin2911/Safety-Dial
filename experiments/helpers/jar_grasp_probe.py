"""One-shot grasp initiation around frozen ManiGuard joint commands."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class GraspController:
    """Causal, single-attempt gripper rule. Arm coordinates are always preserved."""

    def __init__(self, settings, enabled):
        self.settings = settings
        self.enabled = enabled
        reference = settings["reference"]
        self.reference_position = np.asarray(reference["eef_position_in_jar_frame_m"])
        self.reference_rotation = Rotation.from_quat(
            reference["eef_quaternion_in_jar_frame_xyzw"]
        )
        self.mode = "idle"
        self.streak = 0
        self.age = 0
        self.true_streak = 0
        self.false_streak = 0
        self.attempted = False
        self.next_step = 0

    def apply(self, step, native, observation):
        native = np.asarray(native)
        if step != self.next_step or native.shape != (8,) or native.dtype.kind != "f":
            raise ValueError("Expected contiguous steps and eight floating joint/gripper targets")
        if not np.isfinite(native).all() or native[7] not in (-1, 1):
            raise ValueError("Expected finite native targets and a binarized gripper")
        self.next_step += 1
        output = native.copy()
        valid = bool(observation["valid"])
        distance = angle = None
        if valid:
            values = [observation[k] for k in ("eef_pos", "eef_quat", "jar_pos", "jar_quat")]
            valid = all(np.isfinite(np.asarray(value)).all() for value in values)
            valid &= observation["grasp"] in (-1, 1)
        if valid:
            jar_rotation = Rotation.from_quat(observation["jar_quat"])
            relative_position = jar_rotation.inv().apply(
                np.asarray(observation["eef_pos"]) - observation["jar_pos"]
            )
            relative_rotation = jar_rotation.inv() * Rotation.from_quat(observation["eef_quat"])
            distance = float(np.linalg.norm(relative_position - self.reference_position))
            angle = float((self.reference_rotation.inv() * relative_rotation).magnitude() * 180 / np.pi)
        ap = observation.get("ap", {})
        trigger = self.settings["trigger"]
        eligible = bool(
            valid and ap.get("jar_closed") and ap.get("jar_on_support")
            and ap.get("jar_upright") and not ap.get("jar_dropped")
            and not observation.get("previous_violation", True)
            and observation["grasp"] == -1 and native[7] == 1
            and distance <= trigger["position_tolerance_m"]
            and angle <= trigger["orientation_tolerance_deg"]
        )
        before = self.mode
        if self.mode == "idle":
            self.streak = self.streak + 1 if eligible else 0
            if self.enabled and self.streak >= trigger["consecutive_qualifying_controls"]:
                self.mode, self.attempted, self.age = "acquiring", True, 0
        if self.mode in ("acquiring", "holding"):
            if not valid:
                self.mode = "finished"
            elif self.mode == "acquiring":
                self.true_streak = self.true_streak + 1 if observation["grasp"] == 1 else 0
                if self.true_streak >= self.settings["action"]["grasp_confirmation_controls"]:
                    self.mode = "holding"
                elif self.age >= self.settings["action"]["max_acquisition_controls"]:
                    self.mode = "finished"
            elif self.mode == "holding":
                self.false_streak = self.false_streak + 1 if observation["grasp"] == -1 else 0
                if self.false_streak >= 2:
                    self.mode = "finished"
            if self.mode in ("acquiring", "holding"):
                output[7] = -1
                self.age += 1
        if not np.array_equal(output[:7], native[:7]):
            raise AssertionError("Probe changed an arm target")
        return output, {
            "step": step, "enabled": self.enabled, "valid": bool(valid),
            "eligible": eligible, "position_gap_m": distance, "orientation_gap_deg": angle,
            "mode_before": before, "mode_after": self.mode, "age": self.age,
            "attempted": self.attempted, "grasp": observation["grasp"],
            "native": native, "modified": output, "override": bool(output[7] != native[7]),
            "observation": observation,
        }


COUNTERS = {
    "environment_step_attempts": 0, "environment_step_returns": 0,
    "reset_attempts": 0, "reset_returns": 0,
    "simulator_step_attempts": 0, "simulator_step_returns": 0,
    "step_physics_attempts": 0, "step_physics_returns": 0,
    "physics_callbacks": 0, "physics_callback_registered": False,
    "physics_before_counter_installation": None,
    "physics_before_counter_installation_note": "Initial simulator bootstrap unmeasured, not zero",
    "warmup_attempts": 0, "warmup_returns": 0,
    "query_attempts": 0, "query_returns": 0, "query_seconds": 0.0,
}


def job_directory():
    return Path(os.environ["JAR_GRASP_PROBE_JOB"])


def persist_counters():
    directory = job_directory()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "counters.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(COUNTERS, indent=2) + "\n")
    temporary.replace(path)


def event(name, **fields):
    directory = job_directory()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": name, "time": time.time(), **fields}, default=encode) + "\n")


def mark(name, **fields):
    if name == "query_attempt":
        COUNTERS["query_attempts"] += 1
        COUNTERS["query_started"] = time.monotonic()
    elif name == "query_return":
        COUNTERS["query_returns"] += 1
        COUNTERS["query_seconds"] += time.monotonic() - COUNTERS.pop("query_started")
    elif name in ("warmup_attempt", "warmup_return"):
        COUNTERS[name + "s"] += 1
    event(name, **fields)
    persist_counters()


def install_accounting(og):
    """Wrap launch, then count subsequent environment/reset/simulator calls."""
    original_launch = og.launch

    def counted(owner, method, key):
        original = getattr(owner, method)

        def wrapper(*args, **kwargs):
            COUNTERS[key + "_attempts"] += 1
            persist_counters()
            result = original(*args, **kwargs)
            COUNTERS[key + "_returns"] += 1
            persist_counters()
            return result

        setattr(owner, method, wrapper)

    def launch(*args, **kwargs):
        event("simulator_launch_attempt")
        result = original_launch(*args, **kwargs)
        sim = og.sim
        counted(sim, "step", "simulator_step")
        counted(sim, "step_physics", "step_physics")
        from omnigibson.envs.env_base import Environment

        counted(Environment, "step", "environment_step")
        counted(Environment, "reset", "reset")
        try:
            def tick(_dt):
                COUNTERS["physics_callbacks"] += 1

            sim.add_physics_callback("jar_grasp_probe_counter", tick)
            COUNTERS["physics_callback_registered"] = True
        except (AttributeError, RuntimeError) as error:
            COUNTERS["physics_callback_error_type"] = type(error).__name__
        event("simulator_counter_installation")
        persist_counters()
        return result

    og.launch = launch
    atexit.register(persist_counters)
    mark("accounting_armed")


def create_observer(cfg, scene):
    from record_rollout import RolloutRecorder

    class ProbeRecorder(RolloutRecorder):
        def __init__(self, cfg, scene):
            super().__init__(cfg, scene)
            plan = json.loads(Path(os.environ["JAR_GRASP_PROBE_PLAN"]).read_text())
            self.controller = GraspController(
                plan["controller"], os.environ["JAR_GRASP_PROBE_ARM"] == "pose_triggered_grasp"
            )
            self.traces = (job_directory() / "decisions.jsonl").open("x", buffering=1)

        def start(self, env, robot, obs, monitor, episode_seed, scene):
            super().start(env, robot, obs, monitor, episode_seed, scene)
            if self.jar is None or monitor is None:
                raise ValueError("Probe requires the actual target and original safety monitor")
            if not (
                cfg.action_frequency == 20 and cfg.execute_horizon == 8
                and cfg.source_profile is None and cfg.camera_resolution == 256
                and cfg.controller_preset == "joint_position_raw" and cfg.gripper_binarize
                and cfg.success_hold_steps == 10 and not cfg.ik_eef_to_joint
                and cfg.grasping_mode == "assisted" and cfg.longfinger
            ):
                raise ValueError("Evaluation differs from the frozen native contract")
            event("episode_started", episode_seed=episode_seed, scene=scene["name"])

        def modify_action(self, step, command):
            snapshot = self.monitor.recording_state(step)
            position, quaternion = self.jar.get_position_orientation()
            observation = {
                "valid": bool(snapshot["valid"]),
                "eef_pos": array(self.robot.get_eef_position()),
                "eef_quat": array(self.robot.get_eef_orientation()),
                "jar_pos": array(position), "jar_quat": array(quaternion),
                "grasp": int(self.robot.is_grasping(candidate_obj=self.jar)),
                "ap": (snapshot["observation"] or {}).get("ap", {}),
                "previous_violation": bool(self.monitor.violated),
            }
            modified, trace = self.controller.apply(step, command, observation)
            self.traces.write(json.dumps(trace, default=encode) + "\n")
            return modified

        def proposal(self, step, chunk, planned_steps, action_space):
            super().proposal(step, chunk, planned_steps, action_space)
            self.meta["environment_action_space"] = {
                key: [float(value) if np.isfinite(value) else None for value in array(values)]
                for key, values in (("low", action_space.low), ("high", action_space.high))
            }

        def finish(self, result):
            super().finish(result)
            self.traces.close()
            event("episode_finished", result=result)
            persist_counters()

    return ProbeRecorder(cfg, scene)


def modify_action(recorder, step, command):
    if recorder is None:
        raise ValueError("Explicit action hook requires the probe recorder")
    return recorder.modify_action(step, command)
