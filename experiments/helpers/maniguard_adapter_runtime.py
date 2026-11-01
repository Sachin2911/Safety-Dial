"""Live adapter, passive recordings and crash-surviving interaction accounting."""

import atexit
import json
import os
from pathlib import Path
import time

import numpy as np

from maniguard_adapter import AdapterSpec, ResidualAdapter, observation_from_row


COUNTERS = {f"{name}_{suffix}": 0 for name in
            ("environment_step", "simulator_step", "reset", "step_physics",
             "warmup", "query") for suffix in ("attempts", "returns")}
COUNTERS.update(physics_callbacks=0, physics_callback_registered=False, query_seconds=0.0,
                physics_before_counter_installation=None)


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def job_directory():
    return Path(os.environ["MANIGUARD_ADAPTER_JOB"])


def persist():
    path = job_directory() / "counters.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(COUNTERS, indent=2) + "\n")
    temporary.replace(path)


def mark(event, **fields):
    now = time.time()
    if event.endswith("_attempt"):
        if now >= float(os.environ["MANIGUARD_ADAPTER_END_EPOCH"]):
            raise TimeoutError("Run deadline reached")
        if event == "query_attempt" and COUNTERS["query_attempts"] >= 250:
            raise RuntimeError("Episode query allowance exhausted")
        COUNTERS[event + "s"] += 1
    elif event.endswith("_return"):
        COUNTERS[event + "s"] += 1
    if event == "query_attempt":
        COUNTERS["query_started"] = time.monotonic()
    if event == "query_return":
        COUNTERS["query_seconds"] += time.monotonic() - COUNTERS.pop("query_started")
    persist()
    with (job_directory() / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": event, "time": now, **fields}) + "\n")


def install_accounting(og):
    original_launch = og.launch

    def counted(owner, method, name):
        original = getattr(owner, method)

        def call(*args, **kwargs):
            if name == "environment_step":
                if COUNTERS["environment_step_attempts"] >= 2010:
                    raise RuntimeError("Episode environment-step allowance exhausted")
                if time.time() >= float(os.environ["MANIGUARD_ADAPTER_END_EPOCH"]):
                    raise TimeoutError("Run deadline reached")
            COUNTERS[name + "_attempts"] += 1
            persist()
            result = original(*args, **kwargs)
            COUNTERS[name + "_returns"] += 1
            persist()
            return result

        setattr(owner, method, call)

    def launch(*args, **kwargs):
        result = original_launch(*args, **kwargs)
        from omnigibson.envs.env_base import Environment

        counted(Environment, "step", "environment_step")
        counted(Environment, "reset", "reset")
        counted(og.sim, "step", "simulator_step")
        counted(og.sim, "step_physics", "step_physics")

        def tick(_dt):
            COUNTERS["physics_callbacks"] += 1

        og.sim.add_physics_callback("maniguard_adapter_accounting", tick)
        COUNTERS["physics_callback_registered"] = True
        persist()
        return result

    og.launch = launch
    atexit.register(persist)
    persist()


def create_observer(cfg, scene):
    from accounted_recorder import AccountedRecorder

    class AdapterRecorder(AccountedRecorder):
        def __init__(self, cfg, scene):
            super().__init__(cfg, scene)
            artifact = json.loads((job_directory() / "controller.json").read_text())
            self.adapter = ResidualAdapter(AdapterSpec(**artifact["spec"]),
                                           artifact["parameters"], enabled=artifact["enabled"])
            self.decisions = (job_directory() / "decisions.jsonl").open("x", buffering=1)

        def start(self, env, robot, obs, monitor, episode_seed, scene):
            super().start(env, robot, obs, monitor, episode_seed, scene)
            if (cfg.action_frequency != 20 or cfg.execute_horizon != 8
                    or cfg.max_steps != 2000 or cfg.success_hold_steps != 10
                    or cfg.camera_resolution != 256 or cfg.source_profile is not None
                    or cfg.controller_preset != "joint_position_raw" or cfg.ik_eef_to_joint
                    or not cfg.gripper_binarize or not cfg.longfinger
                    or cfg.grasping_mode != "assisted" or monitor is None or self.jar is None):
                raise ValueError("Unexpected ManiGuard execution contract")
            self.arm_indices = np.asarray(self.meta["arm_control_idx"], dtype=int)
            if self.arm_indices.shape != (7,):
                raise ValueError("Unexpected arm joint mapping")

        def modify_action(self, step, command):
            if len(self.rows) != step + 1 or int(self.rows[-1]["step"]) != step:
                raise ValueError("Adapter observation is not the current pre-action state")
            observation = observation_from_row(self.rows[-1], self.ap_names, self.arm_indices)
            modified, trace = self.adapter.apply(step, command, observation)
            self.decisions.write(json.dumps(trace, default=encode) + "\n")
            return modified

        def proposal(self, step, chunk, planned_steps, action_space):
            super().proposal(step, chunk, planned_steps, action_space)
            self.meta["environment_action_space"] = {
                key: [float(v) if np.isfinite(v) else None for v in np.asarray(values)]
                for key, values in (("low", action_space.low), ("high", action_space.high))}

        def finish(self, result):
            super().finish(result)
            self.decisions.close()
            persist()

    return AdapterRecorder(cfg, scene)


def modify_action(recorder, step, command):
    if recorder is None:
        raise ValueError("Residual adapter requires its explicit recorder")
    return recorder.modify_action(step, command)
