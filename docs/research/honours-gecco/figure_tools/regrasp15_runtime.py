"""Per-episode task 15 regrasp adapter inside the evaluator process.

The gate propositions and the measured arm, hand and jar state come from the accounted
recorder's row for the current control. This is privileged simulator information,
declared in the plan and shared by every adapter arm. Outside the gate, and always in
native mode, the native chunk passes through as the same object. Inside the gate a
select arm asks the policy server for K candidates on the same observation only when
the native chunk fails the check; a project arm slows a failing native chunk. Every
decision and its inputs go to a journal from which validation replays it.
"""

import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

from regrasp15_check import EXECUTED, CheckConfig, Gate, excess, features, project, select
from regrasp15_kinematics import eef_local, yaw_from_quaternion, yaw_matrix

AP = {
    "closed": "jar_closed",
    "dropped": "jar_dropped",
    "on_support": "jar_on_support",
    "upright": "jar_upright",
}


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


def candidate_seed(episode_seed, step):
    """Recorded, deterministic latent seed for one candidate request."""
    return zlib.crc32(f"regrasp15:{int(episode_seed)}:{int(step)}".encode())


def gate_inputs(ap, names):
    """(closed, on_support, upright, dropped) from one row of monitor propositions."""
    ap = np.asarray(ap, dtype=bool)
    index = {key: list(names).index(name) for key, name in AP.items()}
    return (bool(ap[index["closed"]]), bool(ap[index["on_support"]]),
            bool(ap[index["upright"]]), bool(ap[index["dropped"]]))


class Runtime:
    def __init__(self, cfg, robot, recorder):
        settings = json.loads(Path(os.environ["SAFETY_DIAL_REGRASP_CONFIG"]).read_text())
        self.config = CheckConfig(**settings)
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
        self.counts = {"queries": 0, "gated_queries": 0, "vetoes": 0, "candidate_requests": 0,
                       "candidate_decodes": 0, "projections": 0}
        self.journal = Path(os.environ["SAFETY_DIAL_REGRASP_JOURNAL"]).open("x", buffering=1)
        self.emit("created", settings=self.config.settings(), arm_control_idx=self.indices)

    def emit(self, event, **fields):
        self.journal.write(json.dumps({"event": event, **fields}, default=encode) + "\n")

    def _consume(self):
        rows = self.recorder.rows
        while self.gate.seen < len(rows):
            row = rows[self.gate.seen]
            step = int(row["step"])
            if step != self.gate.seen:
                raise ValueError("Recorder rows are not contiguous control states")
            latched, closed = self.gate.latched_at, self.gate.closed_at
            self.gate.update(step, *gate_inputs(array(row["ap"]), self.recorder.ap_names))
            if latched is None and self.gate.latched_at is not None:
                self.emit("gate_latched", step=step)
            if closed is None and self.gate.closed_at is not None:
                self.emit("gate_closed", step=step, reason=self.gate.reason)

    def _init_base(self, q, eef):
        position, orientation = self.robot.get_position_orientation()
        position = array(position).astype(np.float64)
        orientation = array(orientation).astype(np.float64)
        yaw = yaw_from_quaternion(orientation)
        residual = float(np.linalg.norm(position + yaw_matrix(yaw) @ eef_local(q) - eef))
        self.base = {"yaw": yaw, "position": position, "quaternion": orientation,
                     "fk_residual_m": residual}
        self.emit("base", **self.base)

    def choose(self, step, chunk, policy_obs, client):
        """Return the chunk to execute for the query at this control."""
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
        feat = features(native, q, eef, jar, yaw, self.config.near_radius_m)
        entry["native_near_speed_mps"] = feat["max_near_speed_mps"]
        entry["native_min_horizontal_m"] = feat["min_horizontal_m"]
        if self.config.mode == "native" or excess(feat, self.config.speed_limit_mps) == 0.0:
            self.emit("decision", **entry, action="native")
            return chunk
        self.counts["vetoes"] += 1
        if self.config.mode == "project":
            result, scale, check = project(native, q, eef, jar, yaw, self.config)
            executed = result.astype(np.float32)
            self.counts["projections"] += 1
            self.emit("decision", **entry, action="projected", scale=scale, executed=executed,
                      executed_near_speed_mps=check["max_near_speed_mps"])
            return executed
        seed = candidate_seed(self.recorder.episode_seed, step)
        request = dict(policy_obs)
        request["regrasp15"] = {"k": self.config.candidates, "seed": seed}
        started = time.monotonic()
        reply = client.infer(request)
        request_s = time.monotonic() - started
        candidates = np.asarray(reply["candidates"], dtype=np.float32)
        expected = (self.config.candidates, 16, 8)
        if (candidates.shape != expected or not np.isfinite(candidates).all()
                or int(reply["noise_seed"]) != seed):
            raise ValueError("Candidate reply is not a finite (K, 16, 8) set for this seed")
        self.counts["candidate_requests"] += 1
        self.counts["candidate_decodes"] += self.config.candidates
        choice, rule, feats, over, distance = select(native, candidates, q, eef, jar, yaw,
                                                     self.config)
        executed = native if choice == 0 else candidates[choice - 1]
        self.emit("decision", **entry, action=rule, seed=seed, candidates=candidates,
                  choice=choice, over=over, distance=distance,
                  speeds=[f["max_near_speed_mps"] for f in feats], executed=executed,
                  request_s=request_s, decode_s=float(reply["decode_s"]))
        return np.array(executed, dtype=np.float32, copy=True)

    def finish(self, result):
        self._consume()
        self.emit("finish", result=result, counts=self.counts,
                  gate={"latched_at": self.gate.latched_at, "closed_at": self.gate.closed_at,
                        "reason": self.gate.reason})
        self.journal.close()


def create_adapter(cfg, robot, recorder):
    return Runtime(cfg, robot, recorder)
