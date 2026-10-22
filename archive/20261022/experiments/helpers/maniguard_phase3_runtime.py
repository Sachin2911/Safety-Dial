"""Semantic Part 1 study, with physical labels confined to evaluation telemetry."""

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from helpers.maniguard_phase3_adapter import Controller, SafetyDial, genome_hash
from helpers.maniguard_runtime import Study


class Part1Study(Study):
    def __init__(self):
        job = json.loads(Path(os.environ["SAFETY_DIAL_MANIGUARD_JOB"]).read_text())
        controller = Controller.from_genome(job["genome"])
        if genome_hash(job["genome"]) != job["genome_sha256"]:
            raise ValueError("Controller hash mismatch")
        adapter = SafetyDial(controller, job["dial"])
        config = SimpleNamespace(
            kind="phase3_part1", privileged=False, guard_joint_limits=False, horizon=adapter.horizon
        )
        super().__init__(adapter=adapter, config=config)
        self.counts.update(
            adapter_seconds=0.0,
            policy_queries_completed=0,
            policy_queries_failed=0,
            initialization_seconds=None,
        )

    def ledger(self):
        super().ledger()
        # App startup precedes PhysX subscription. Preserve this unknown scope rather
        # than asserting that callbacks cover unobserved startup advances.
        path = self.output / "ledger.json"
        value = json.loads(path.read_text())
        value["physics_scope"] = "PhysX callbacks after app launch, including scene initialization"
        value["pre_subscription_physics_ticks"] = None
        from helpers.maniguard_phase3_budget import atomic

        atomic(path, value)

    def validate_policy(self, policy, client_type):
        super().validate_policy(policy, client_type)
        cfg = self.effective_cfg
        self.write("effective_config", values=json.loads(json.dumps(asdict(cfg), default=str)))

    def start(self, scene, obs, env, robot, monitor):
        self.adapter.reset()
        super().start(scene, obs)
        if monitor is None:
            raise ValueError("Physical monitor is required")
        self.observed(0, obs, monitor)
        self.counts["initialization_seconds"] = time.monotonic() - self.started
        from omnigibson.macros import gm

        self.write(
            "controller",
            genome=self.job["genome"],
            genome_sha256=self.job["genome_sha256"],
            decoded=asdict(self.adapter.controller),
            dial=self.adapter.dial,
            effective_horizon=self.adapter.horizon,
            overview_shape=list(obs["overview_image"].shape),
            wrist_shape=list(obs["wrist_images"].shape),
            rendering={
                key: getattr(gm, key)
                for key in ("ENABLE_HQ_RENDERING", "USE_GPU_DYNAMICS", "ENABLE_FLATCACHE")
            },
        )
        self.ledger()

    def connect_policy(self, cfg, upstream_connect):
        self.effective_cfg = cfg
        return super().connect_policy(cfg, upstream_connect)

    def after_query(self, step, chunk):
        super().after_query(step, chunk)
        self.counts["policy_queries_completed"] += 1
        values = np.asarray(chunk)
        if values.ndim != 2 or values.shape[1] != 8 or not len(values):
            raise ValueError("Policy returned an invalid chunk")
        if not np.isfinite(values).all():
            raise ValueError("Policy returned nonfinite actions")
        self.ledger()

    def action(self, step, ci, original, frame):
        began = time.perf_counter()
        try:
            return super().action(step, ci, original, frame)
        finally:
            self.counts["adapter_seconds"] += time.perf_counter() - began

    def clip_action(self, step, action, space):
        clipped = np.clip(action, space.low, space.high)
        if np.isnan(space.low).any() or np.isnan(space.high).any():
            raise ValueError("NaN action-space bounds")
        # Raw-joint action spaces legitimately contain unbounded coordinates.
        low = [float(x) if np.isfinite(x) else str(float(x)) for x in space.low]
        high = [float(x) if np.isfinite(x) else str(float(x)) for x in space.high]
        self.write(
            "action_space",
            step=step,
            low=low,
            high=high,
            bounds_encoding="finite numbers or explicit inf/-inf strings",
            dtype=str(space.low.dtype),
            clipped=clipped,
        )
        return clipped

    def close(self):
        if self.query_started is not None:
            self.counts["policy_queries_failed"] += 1
            self.counts["policy_query_seconds"] += time.monotonic() - self.query_started
            self.write("query_failed", elapsed_seconds=time.monotonic() - self.query_started)
            self.query_started = None
        super().close()

    def finish(self, result):
        if self.query_started is not None:
            self.counts["policy_queries_failed"] += 1
        super().finish(result)


STUDY = Part1Study() if os.environ.get("SAFETY_DIAL_MANIGUARD_JOB") else None
