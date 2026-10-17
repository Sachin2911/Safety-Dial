"""Instrumented ManiGuard integration; original monitor and goals remain unchanged."""

import atexit
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from helpers.maniguard_adaptation import AdapterConfig, ExecutionAdapter, Frame


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().tolist()
    raise TypeError(type(value).__name__)


def arm_position_limits(robot, indices):
    """Read the same position limits and mask supplied to the robot's controllers."""
    limits = robot.control_limits

    def select(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)[indices].copy()

    lower, upper = limits["position"]
    return select(lower), select(upper), select(limits["has_limit"])


class Study:
    def __init__(self, adapter=None, config=None):
        self.job = json.loads(Path(os.environ["SAFETY_DIAL_MANIGUARD_JOB"]).read_text())
        self.output = Path(self.job["output"])
        self.config = AdapterConfig(**self.job["adapter"]) if config is None else config
        self.adapter = ExecutionAdapter(self.config) if adapter is None else adapter
        self.trace = (self.output / "trace.jsonl").open("x", buffering=1)
        self.started = time.monotonic()
        self.query_started = None
        self.phase = "initialization"
        self.counts = {"physics_steps": 0, "physics_seconds": 0.0, "env_step_attempts": 0,
                       "env_steps_returned": 0, "policy_queries": 0, "policy_query_seconds": 0.0,
                       "policy_control_steps": 0, "guard_steps": 0, "changed_steps": 0}
        self.physics_subscription = None
        self.physics_accounting_attached = False
        self.finished = False
        self.write("job", job=self.job)
        self.ledger()
        atexit.register(self.close)

    def close(self):
        if self.trace.closed:
            return
        if not self.finished:
            self.write("incomplete", last_phase=self.phase)
            self.phase = "process_exited_without_result"
        self.ledger()
        self.trace.close()

    def write(self, kind, **values):
        self.trace.write(json.dumps({"type": kind, **values}, default=json_value,
                                    allow_nan=False) + "\n")

    def ledger(self):
        record = {**self.counts, "phase": self.phase,
                  "physics_accounting_attached": self.physics_accounting_attached,
                  "elapsed_seconds": time.monotonic() - self.started}
        temp = self.output / "ledger.tmp"
        temp.write_text(json.dumps(record, indent=2) + "\n")
        temp.replace(self.output / "ledger.json")

    def connect_policy(self, cfg, upstream_connect):
        if cfg.random_policy:
            return upstream_connect(cfg)
        if not cfg.use_openpi_client:
            raise ValueError("Study requires the OpenPI client")
        from helpers.maniguard_client import StudyClient
        return StudyClient(host=cfg.host, port=cfg.port), "openpi"

    def validate_policy(self, policy, client_type):
        if self.job["random_policy"]:
            if client_type != "random":
                raise ValueError("Expected the explicitly requested engineering random policy")
            return
        if client_type != "openpi":
            raise ValueError("Study requires the pinned OpenPI policy server")
        metadata = policy.get_server_metadata()
        spec = self.job["policy"]
        expected = {"serve_config": spec["config"], "checkpoint_revision": spec["revision"],
                    "episode_seed_reset": "study_episode_id"}
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Policy handshake mismatch for {key}: {metadata.get(key)!r}")
        self.write("policy_metadata", metadata=metadata)

    def install_accounting(self, og):
        from omnigibson import simulator

        original_launch = simulator._launch_app

        def launch(*args, **kwargs):
            self.phase = "app_startup"
            self.ledger()
            app = original_launch(*args, **kwargs)
            from omni.physx import get_physx_interface

            def physics_step(dt):
                self.counts["physics_steps"] += 1
                self.counts["physics_seconds"] += float(dt)
                if self.counts["physics_steps"] % 120 == 0:
                    self.ledger()

            self.physics_subscription = get_physx_interface().subscribe_physics_step_events(
                physics_step,
            )
            self.physics_accounting_attached = True
            self.phase = "scene_initialization"
            self.ledger()
            return app

        simulator._launch_app = launch
        original_step = og.Environment.step

        def step(env, *args, **kwargs):
            self.counts["env_step_attempts"] += 1
            result = original_step(env, *args, **kwargs)
            self.counts["env_steps_returned"] += 1
            return result

        og.Environment.step = step

    def frame(self, env, robot, monitor, step, jacobian_fn):
        joints = robot.get_joint_positions().detach().cpu().numpy()
        arm = robot.default_arm
        indices = robot.arm_control_idx[arm].cpu().numpy()
        fingers = robot.gripper_control_idx[arm].cpu().numpy()
        jac = None
        prerequisite = None
        engaged = None
        if self.config.kind in ("retry", "prerequisite", "handback"):
            from omnigibson.utils.transform_utils import quat2mat
            base_rotation = quat2mat(robot.get_position_orientation()[1]).cpu().numpy()
            jac = base_rotation @ jacobian_fn(robot, arm, indices)[:3]
        if self.config.privileged:
            if monitor is None:
                raise ValueError("Guard requested without an active physical safety monitor")
            state = monitor.recording_state(step)
            if not state["valid"]:
                raise ValueError("Guard requested a stale or unavailable prerequisite label")
            labels = state["observation"]["ap"]
            key = {"jar": "jar_closed", "lid": "lid_on_container"}.get(self.job["family"])
            if key not in labels:
                raise ValueError(f"Expected prerequisite {key!r}, received {list(labels)}")
            prerequisite = bool(labels[key])
            from maniguard.utils.goal_region import robot_holds_target
            from omnigibson.object_states import ContactBodies
            target = env.scene.object_registry("name", self.scene["target_name"])
            if target is None:
                raise ValueError("Guard target did not resolve")
            engaged = bool(robot_holds_target(env, target) or (
                target.states[ContactBodies].get_value() & set(robot.links.values())
            ))
        limits = arm_position_limits(robot, indices) if self.config.guard_joint_limits else (
            None, None, None,
        )
        return Frame(joints[indices].copy(), joints[fingers].copy(),
                     robot.get_eef_position().cpu().numpy().copy(), jac, prerequisite, engaged,
                     *limits)

    def start(self, scene, obs):
        self.scene = scene
        self.phase = "policy"
        self.write("initial", scene=scene, state=obs["states"])
        np.savez_compressed(self.output / "initial_observation.npz", state=obs["states"],
                            overview=obs["overview_image"], wrist=obs["wrist_images"])
        self.ledger()

    def before_query(self, step, obs, frame):
        self.adapter.on_query(frame)
        self.query_started = time.monotonic()
        self.counts["policy_queries"] += 1
        images = {key: hashlib.sha256(np.ascontiguousarray(obs[key]).tobytes()).hexdigest()
                  for key in ("overview_image", "wrist_images")}
        self.write("query", step=step, state=obs["states"], image_hashes=images)
        self.ledger()

    def after_query(self, step, chunk):
        elapsed = time.monotonic() - self.query_started
        self.counts["policy_query_seconds"] += elapsed
        self.write("chunk", step=step, actions=chunk, elapsed_seconds=elapsed)
        self.query_started = None

    def action(self, step, ci, original, frame):
        adapted, event = self.adapter.act(original, frame)
        self.counts["guard_steps"] += int(event["guard_blocked"])
        self.counts["changed_steps"] += int(event["changed"])
        self.write("proposal", step=step, chunk_index=ci, original=original,
                   adapted=adapted, frame=asdict(frame), event=event)
        return adapted

    def applied(self, step, executed):
        self.adapter.applied(executed)
        self.counts["policy_control_steps"] += 1
        self.write("executed", step=step, action=executed,
                   action_scope="environment_input_before_controller_output_clipping")

    def should_replan(self, step, frame):
        reason = self.adapter.replan_reason(frame)
        if reason is not None:
            self.write("replan", step=step, reason=reason, executed_prefix=self.adapter.since_query)
        return reason is not None

    def observed(self, step, obs, monitor):
        state = monitor.recording_state(step) if monitor is not None else None
        self.write("observed", step=step, state=obs["states"], monitor=state)

    def finish(self, result):
        if self.query_started is not None:
            elapsed = time.monotonic() - self.query_started
            self.counts["policy_query_seconds"] += elapsed
            self.write("query_failed", elapsed_seconds=elapsed)
            self.query_started = None
        self.finished = True
        self.phase = result["status"]
        self.write("result", result=result)
        self.ledger()


def _legacy_study():
    path = os.environ.get("SAFETY_DIAL_MANIGUARD_JOB")
    if not path or json.loads(Path(path).read_text()).get("phase") == "phase2":
        return None
    return Study()


STUDY = _legacy_study()
