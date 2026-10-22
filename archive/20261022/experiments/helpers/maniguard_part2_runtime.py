"""Followup telemetry. Privileged measurements are targets, never controller inputs.

The episode entry point imports the legacy/base classes with the job environment
unset, suppressing their historical import-time singleton creation. This module alone
owns the active study singleton; the frozen base files remain byte-identical.
"""

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from helpers.maniguard_part2_controller import Config, FeedbackController, config_hash
from helpers.maniguard_phase3_runtime import Part1Study
from helpers.maniguard_runtime import Study


class FollowupStudy(Part1Study):
    def __init__(self):
        job = json.loads(Path(os.environ["SAFETY_DIAL_MANIGUARD_JOB"]).read_text())
        cfg = Config(**job["controller"])
        if config_hash(cfg) != job["controller_sha256"]:
            raise ValueError("Controller hash mismatch")
        adapter = FeedbackController(cfg)
        config = SimpleNamespace(
            kind="phase3_followup", privileged=False, guard_joint_limits=False,
            horizon=adapter.horizon,
        )
        Study.__init__(self, adapter=adapter, config=config)
        self.counts.update(
            adapter_seconds=0.0, policy_queries_completed=0, policy_queries_failed=0,
            initialization_seconds=None, passive_capture_seconds=0.0, passive_frames=0,
        )
        self.capture = self.output / "observations"
        self.capture.mkdir()
        self.env = self.robot = self.monitor = None

    def start(self, scene, obs, env, robot, monitor):
        self.env, self.robot, self.monitor = env, robot, monitor
        if monitor is None:
            raise ValueError("Physical monitor required")
        self.adapter.reset()
        Study.start(self, scene, obs)
        self.observed(0, obs, monitor)
        self.counts["initialization_seconds"] = time.monotonic() - self.started
        self.write(
            "controller", decoded=asdict(self.adapter.controller),
            controller_sha256=self.job["controller_sha256"],
            overview_shape=list(obs["overview_image"].shape),
            wrist_shape=list(obs["wrist_images"].shape),
            privileged_inputs=False,
        )
        self.ledger()

    def applied(self, step, executed):
        super().applied(step, executed)
        self.write(
            "correction", step=step, normalized=self.adapter.last_correction,
            accumulated=self.adapter.correction, since_query=self.adapter.since_query,
            scope="proposal minus actual environment input, arm targets only",
        )

    def measurements(self):
        # Exact implementation of pinned ManiGuard Upright._get_value, including
        # float32 and local +Z. These read-only calls do not step or render physics.
        import torch as th
        from omnigibson.object_states import Upright
        from omnigibson.utils import transform_utils as transform

        protected = []
        seen = set()
        for obj in self.monitor._resolver._active_objs.values():
            if obj is None or id(obj) in seen or Upright not in obj.states:
                continue
            seen.add(id(obj))
            position, quaternion = obj.get_position_orientation()
            world_up = th.tensor([0.0, 0.0, 1.0], dtype=th.float32)
            up = th.as_tensor(transform.quat_apply(
                th.as_tensor(quaternion, dtype=th.float32), world_up,
            ), dtype=th.float32).reshape(-1)[-3:]
            up = up / (th.norm(up) + 1e-8)
            angle = float(th.rad2deg(th.acos(th.clamp(th.dot(up, world_up), -1.0, 1.0))))
            threshold = float(obj.states[Upright].max_tilt_deg)
            protected.append({
                "name": obj.name, "angle_deg": angle, "threshold_deg": threshold,
                "margin_deg": threshold - angle,
                "position": position.detach().cpu().tolist(),
                "quaternion_xyzw": quaternion.detach().cpu().tolist(),
            })
        # Monitor resolver restricts active objects. Retain explicit IDs and compare
        # this scope with the native proposition bindings in the audit.
        target = self.env.scene.object_registry("name", self.scene["target_name"])
        region = self.scene.get("goal_region") or {}
        center = region.get("center_world")
        progress = None
        if target is not None and center is not None:
            position = target.get_position_orientation()[0].detach().cpu().numpy()
            progress = -float(np.linalg.norm(position - np.asarray(center)))
        return {"protected_objects": protected, "progress_negative_distance_m": progress,
                "goal_region": region, "scope": "evaluation_targets_only"}

    def observed(self, step, obs, monitor):
        super().observed(step, obs, monitor)
        started = time.perf_counter()
        measurements = self.measurements()
        destination = self.capture / f"{step:04d}.npz"
        np.savez_compressed(
            destination, state=obs["states"], overview=obs["overview_image"],
            wrist=obs["wrist_images"],
        )
        self.write("measurement", step=step, **measurements)
        self.counts["passive_frames"] += 1
        self.counts["passive_capture_seconds"] += time.perf_counter() - started


STUDY = FollowupStudy() if os.environ.get("SAFETY_DIAL_MANIGUARD_JOB") else None
