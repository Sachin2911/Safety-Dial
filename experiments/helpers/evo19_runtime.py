"""Per-episode task 19 adapter inside the evaluator process.

Task 15's runtime (`regrasp15_runtime.Runtime`) restricted to native and projection
modes, with the evolvable check of `evo19_check`, plus the shield mode (amendment 3). State comes from the accounted
recorder's row for the current control (privileged simulator state, declared and shared
by every arm). Outside the gate, and always in native mode, the native chunk passes
through as the same object. Every decision is journaled for replay.
"""

import json
import os
from pathlib import Path

import numpy as np

from evo19_check import (EvoConfig, excess, features, project, retrace, tilt_deg, triggered,
                         window_contacts)
from regrasp15_check import EXECUTED, Gate
from regrasp15_runtime import Runtime as Regrasp15Runtime
from regrasp15_runtime import array


class Runtime(Regrasp15Runtime):
    def __init__(self, cfg, robot, recorder):
        settings = json.loads(Path(os.environ["SAFETY_DIAL_REGRASP_CONFIG"]).read_text())
        self.config = EvoConfig(**settings)
        if recorder is None or not hasattr(recorder, "rows"):
            raise ValueError("The adapter reads its state from the accounted recorder")
        if (cfg.action_frequency != 20 or cfg.state_mode != "joint"
                or cfg.controller_preset != "joint_position_raw" or cfg.ik_eef_to_joint
                or cfg.execute_horizon != EXECUTED or cfg.max_steps != 2000):
            raise ValueError("The adapter requires the frozen raw joint-position contract")
        self.robot, self.recorder = robot, recorder
        self.indices = array(robot.arm_control_idx[robot.default_arm]).astype(int)
        if self.indices.shape != (7,) or len(set(self.indices.tolist())) != 7:
            raise ValueError("Unexpected arm joint mapping")
        self.gate = Gate(self.config)
        self.base = None
        self.origin = None  # where the next consecutive retrace starts
        self.counts = {"queries": 0, "gated_queries": 0, "vetoes": 0, "candidate_requests": 0,
                       "candidate_decodes": 0, "projections": 0}
        self.journal = Path(os.environ["SAFETY_DIAL_REGRASP_JOURNAL"]).open("x", buffering=1)
        self.emit("created", settings=self.config.settings(), arm_control_idx=self.indices)

    def choose(self, step, chunk, policy_obs, client):
        self._consume()
        rows = self.recorder.rows
        if len(rows) != step + 1 or int(rows[-1]["step"]) != step:
            raise ValueError("Recorder state is not at the queried control")
        row = rows[-1]
        q = array(row["q"])[self.indices].astype(np.float64)
        eef = array(row["eef_pos"]).astype(np.float64)
        jar = array(row["jar_pos"]).astype(np.float64)
        if self.base is None:
            self._init_base(q, eef)
        yaw = self.base["yaw"]
        native = np.array(chunk, dtype=np.float32, copy=True)
        self.counts["queries"] += 1
        entry = {"step": step, "gate_open": self.gate.open, "native": native, "q": q,
                 "eef": eef, "jar": jar}
        if not self.gate.open:
            self.emit("decision", **entry, action="gate_shut")
            return chunk
        self.counts["gated_queries"] += 1
        if self.config.mode == "shield":
            tilt = tilt_deg(array(row["jar_quat"]))
            contacts = window_contacts([r["jar_robot_contacts"] for r in rows], step, self.config)
            entry.update(tilt_deg=tilt, contacts=contacts)
            if not triggered(tilt, contacts, self.config):
                self.origin = None
                self.emit("decision", **entry, action="native")
                return chunk
            self.counts["vetoes"] += 1
            origin = step if self.origin is None else self.origin
            joints = [array(r["q"])[self.indices].astype(np.float64) for r in rows]
            hands = [array(r["eef_pos"]).astype(np.float64) for r in rows]
            result, self.origin = retrace(native, joints, hands, origin, self.config.retreat_m)
            executed = result.astype(np.float32)
            self.counts["projections"] += 1
            self.emit("decision", **entry, action="rewound", origin=origin,
                      target=self.origin, executed=executed)
            return executed
        feat = features(native, q, eef, jar, yaw, self.config)
        entry["native_near_speed_mps"] = feat["max_near_speed_mps"]
        entry["native_min_horizontal_m"] = feat["min_horizontal_m"]
        if self.config.mode == "native" or excess(feat, self.config.speed_limit_mps) == 0.0:
            self.emit("decision", **entry, action="native")
            return chunk
        self.counts["vetoes"] += 1
        result, scale, check = project(native, q, eef, jar, yaw, self.config)
        executed = result.astype(np.float32)
        self.counts["projections"] += 1
        self.emit("decision", **entry, action="projected", scale=scale, executed=executed,
                  executed_near_speed_mps=check["max_near_speed_mps"])
        return executed


def create_adapter(cfg, robot, recorder):
    return Runtime(cfg, robot, recorder)
