"""Passive task-11 recorder: the DSRL06 scorer plus residual routing and R1 geometry.

Added reads are passive state getters: the jar world AABB (the quantity the native
goal checker already evaluates every control), the finger link positions and the
native goal-sphere specification. No action, step, render or reset is introduced.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

try:
    from accounted_recorder import AccountedRecorder
    from dsrl06_recording import DSRLRecorder
    from record_rollout import _np
except ImportError:
    from helpers.accounted_recorder import AccountedRecorder
    from helpers.dsrl06_recording import DSRLRecorder

    def _np(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x)


from helpers.dsrl11_server import Dsrl11QueryLedger


ENVELOPE_KEYS = {
    "mode",
    "job_id",
    "attempt_id",
    "purpose",
    "residual_source",
    "epsilon_seed",
    "uniform_seed",
    "actor_url",
    "actor_condition",
    "actor_version",
    "rng_seed",
}
GEOMETRY_FIELDS = ("eef_pos", "jar_pos", "jar_aabb_min", "jar_aabb_max", "finger_pos")


class ResidualRecorder(DSRLRecorder):
    def __init__(self, cfg, scene):
        AccountedRecorder.__init__(self, cfg, scene)
        path = Path(os.environ["SAFETY_DIAL_DSRL_CONFIG"])
        self.settings = json.loads(path.read_text())
        unknown = set(self.settings) - ENVELOPE_KEYS
        if unknown:
            raise ValueError(f"Unknown episode routing fields: {sorted(unknown)}")
        if not self.settings.get("job_id") or self.settings.get("mode") != "residual":
            raise ValueError("DSRL11 recording requires an explicit job and the residual mode")
        self.query_index = 0
        self.query_records = []
        self.last_obs = None
        self.warmup_attempts = 0
        self.warmup_returned = 0
        self.finger_names = None
        self.goal_region = None
        self.emit("dsrl_configuration", settings=self.settings)

    def _fingers(self):
        if self.finger_names is None:
            self.finger_names = sorted(n for n in self.robot.links if "finger" in n.lower())
            if not self.finger_names:
                raise ValueError("The pinned Franka finger links are unavailable")
        return self.finger_names

    def _physical(self, step, obs, monitor):
        row = super()._physical(step, obs, monitor)
        if self.jar is None:
            raise ValueError("The target jar is required for the R1 geometry")
        # Same call and simulator state as the native goal check that follows oracle();
        # float64 storage is lossless for either tensor precision.
        aabb_min, aabb_max = self.jar.aabb
        row["jar_aabb_min"] = _np(aabb_min).astype(np.float64).reshape(3)
        row["jar_aabb_max"] = _np(aabb_max).astype(np.float64).reshape(3)
        row["finger_pos"] = np.stack(
            [
                _np(self.robot.links[name].get_position_orientation()[0]).astype(np.float32)
                for name in self._fingers()
            ]
        )
        return row

    def start(self, env, robot, obs, monitor, episode_seed, scene):
        super().start(env, robot, obs, monitor, episode_seed, scene)
        self.meta.update(
            dsrl11_schema="dsrl11-recording-v1",
            finger_link_names=list(self._fingers()),
            geometry_fields=list(GEOMETRY_FIELDS),
            eef_link_names=dict(getattr(robot, "eef_link_names", {}) or {}),
        )

    def after_query(self, step, obs, chunk):
        # The parent confirms through the task-6 ledger class, whose fixed ceiling
        # differs from the task-11 ledger; this is the same check on the new ledger.
        response = obs.pop("_dsrl06_response", None)
        if not isinstance(response, dict) or not response.get("record_path"):
            raise ValueError("Policy server did not return its DSRL11 query receipt")
        if response.get("job_id") != self.settings["job_id"] or response.get("mode") != "residual":
            raise ValueError("Server query receipt belongs to another job or route")
        if int(response["query_index"]) != self.query_index or int(response["step"]) != step:
            raise ValueError("Server query receipt is not aligned with this proposal")
        with np.load(Path(response["record_path"]), allow_pickle=False) as saved:
            if not np.array_equal(np.asarray(chunk), saved["actions"]):
                raise ValueError("Executed proposal differs from the server's saved output")
            meta = json.loads(str(saved["meta"]))
            if meta["status"] != "returned" or meta["query_id"] != response["query_id"]:
                raise ValueError("Server's query artifact is incomplete")
        Dsrl11QueryLedger(response["query_ledger"]).confirm(response["query_id"])
        self.query_records.append(dict(response))
        self.emit(
            "query_returned",
            step=int(step),
            query_index=self.query_index,
            **{k: response[k] for k in ("query_id", "record_path", "mode", "actor_version")},
        )
        self.query_index += 1

    def initial_goal(self, checker, env):
        region = getattr(checker, "raw_region", None)
        if region is None:
            raise ValueError("The Jar goal must be the native goal-region checker")
        self.goal_region = {
            "mode": str(region.mode),
            "target_name": str(region.target_name),
            "marker_name": str(region.marker_name),
            "center_world": [float(x) for x in region.center_world],
            "radius_m": float(region.radius_m),
        }
        self.meta["goal_region"] = self.goal_region
        self.emit("goal_region", **self.goal_region)
        super().initial_goal(checker, env)


def create_observer(cfg, scene):
    return ResidualRecorder(cfg, scene)
