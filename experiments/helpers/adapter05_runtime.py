"""Explicit evaluator hook. Only robot proprioception and reference history enter control."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from execution_adapter import AdapterConfig, ExecutionAdapter


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class Runtime:
    def __init__(self, cfg, robot):
        self.robot = robot
        settings = json.loads(Path(os.environ["SAFETY_DIAL_ADAPTER_CONFIG"]).read_text())
        self.adapter = ExecutionAdapter(AdapterConfig(**settings))
        self.indices = array(robot.arm_control_idx[robot.default_arm]).astype(int)
        if self.indices.shape != (7,) or len(set(self.indices)) != 7:
            raise ValueError("Unexpected arm joint mapping")
        if (
            cfg.action_frequency != 20
            or cfg.state_mode != "joint"
            or cfg.controller_preset != "joint_position_raw"
            or cfg.ik_eef_to_joint
        ):
            raise ValueError("Adapter requires the frozen raw joint-position contract")
        self.journal = Path(os.environ["SAFETY_DIAL_ADAPTER_JOURNAL"]).open("x", buffering=1)
        self.warmups = 0
        self.applied = 0
        self.emit("created", settings=asdict(self.adapter.config), arm_control_idx=self.indices)

    def emit(self, event, **fields):
        self.journal.write(json.dumps({"event": event, **fields}, default=encode) + "\n")

    def warmup_before(self, command):
        self.emit("warmup_attempt", step=self.warmups, command=array(command))

    def warmup_after(self):
        self.emit("warmup_applied", step=self.warmups)
        self.warmups += 1

    def apply(self, step, command):
        q = array(self.robot.get_joint_positions())[self.indices]
        dq = array(self.robot.get_joint_velocities())[self.indices]
        corrected, trace = self.adapter.apply(command, q, dq, step)
        self.emit("command", command_dtype=str(np.asarray(command).dtype), **trace)
        return corrected

    def after_step(self, step):
        self.emit("applied", step=step)
        self.applied += 1

    def finish(self, result):
        self.emit("finish", result=result, warmup_steps=self.warmups, policy_steps=self.applied)
        self.journal.close()


def create_adapter(cfg, robot):
    return Runtime(cfg, robot)
