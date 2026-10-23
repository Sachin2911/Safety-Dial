"""Per-step rollout recorder for maniguard.eval.benchmark at 6bf4c61, one .npz per episode.

Plugs into the evaluator's existing passive observer hook (`maniguard/eval/recording.py`,
`make_observer`), which the per-task loop calls at seven points: `start`, `proposal`,
`before_action`, `applied`, `transition`, `oracle`, `finish`. Nothing upstream is modified
and scoring is untouched: the recorder only reads simulator state after each executed
action and never steps, renders or alters an action. The upstream observer
(`maniguard.data.recording.observer`) needs the high-fidelity source capture and cannot run
under the paper's legacy 256 profile, which is why this one exists.

Use (behavior env, from the ManiGuard checkout):

    PYTHONPATH=/workspace/Safety-Dial/scripts/todo2 python -m maniguard.eval.benchmark \
        --config <eval yaml> ... \
        --recording-factory record_rollout:create_observer \
        --recording-output-dir /workspace/recordings/<run>

Per episode it writes `<output dir>/<scene name with '/' as '_'>.npz` with, for every step
0..N (N = executed steps; step 0 is the state after warm-up, before the first action):
  step, wall_s                        step index, time.monotonic at the record
  q, dq                               all robot joint positions and velocities
  eef_pos, eef_quat                   end effector pose in world (robot.get_eef_position/orientation)
  gripper_q                           the two finger joint positions
  policy_state                        the 8-D state the policy was sent at that step
  jar_pos, jar_quat                   target jar base pose in world
  jar_link_pose                       (links, 7) pose of every jar link (body and lid)
  jar_joint_q, jar_joint_dq           the jar's joints (the lid hinge)
  jar_lin_vel, jar_ang_vel            target jar velocities
  jar_robot_contacts                  number of robot links in contact with the jar
  jar_finger_contact                  any finger link in contact with the jar
  grasping                            robot.is_grasping(jar) as int (IsGraspingState)
  other_obj_pos                       (objects, 3) positions of every other movable task object
  ap, ap_valid, monitor_doomed        the monitor's propositions at that step (from recording_state)
  recorder_s                          seconds spent inside the recorder for that step
and per policy query and executed action:
  proposal_step, proposal_chunk       (P,), (P, T, A) every chunk the server returned, raw
  proposal_planned                    (P,) how many of its actions were to be executed
  exec_step, exec_proposal, exec_offset, exec_raw, exec_ctrl, exec_clipped
                                      (N,) ... (N, A) the executed actions at the three
                                      boundaries the evaluator distinguishes
Metadata (joint names, link names, object names, scene, seed, config, result row) is in the
`meta` entry as a JSON string.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class RolloutRecorder:
    def __init__(self, cfg, scene):
        self.cfg, self.scene = cfg, scene
        self.out_dir = Path(cfg.recording_output_dir)
        self.rows: list[dict] = []
        self.proposals: list[dict] = []
        self.execs: list[dict] = []
        self.pending = None
        self.proposal_id = -1
        self.pending_obs = None
        self.overhead = 0.0
        self.started = False

    # -- evaluator callbacks -------------------------------------------------------------
    def start(self, env, robot, obs, monitor, episode_seed, scene):
        t0 = time.monotonic()
        from omnigibson.object_states import ContactBodies
        self.ContactBodies = ContactBodies
        self.env, self.robot, self.monitor = env, robot, monitor
        self.episode_seed = episode_seed
        self.robot_links = set(robot.links.values())
        self.finger_links = {l for n, l in robot.links.items() if "finger" in n.lower()}
        target = scene.get("target_name")
        self.jar = env.scene.object_registry("name", target) if target else None
        if self.jar is None:
            jars = [o for o in env.scene.objects if "hinged_jar" in getattr(o, "name", "")]
            self.jar = jars[0] if jars else None
        self.jar_links = list(self.jar.links.items()) if self.jar is not None else []
        self.jar_joints = list(self.jar.joints.items()) if self.jar is not None else []
        self.others = [o for o in env.scene.objects
                       if o is not robot and o is not self.jar and not getattr(o, "fixed_base", False)
                       and not getattr(o, "name", "").startswith("goal_region")
                       and getattr(o, "category", "") not in {"walls", "floors", "ceilings", "door", "window"}]
        self.ap_names = sorted((scene.get("ltl_safety") or {}).get("propositions", {}))
        self.meta = {
            "scene_name": scene.get("name"), "target_name": target, "episode_seed": episode_seed,
            "prompt": scene.get("prompt"), "joint_names": list(robot.joints),
            "arm_control_idx": _np(robot.arm_control_idx[robot.default_arm]).tolist(),
            "gripper_control_idx": _np(robot.gripper_control_idx[robot.default_arm]).tolist(),
            "jar_name": getattr(self.jar, "name", None),
            "jar_link_names": [n for n, _ in self.jar_links],
            "jar_joint_names": [n for n, _ in self.jar_joints],
            "jar_joint_limits": [[float(j.lower_limit), float(j.upper_limit)] for _, j in self.jar_joints],
            "other_object_names": [o.name for o in self.others],
            "ap_names": self.ap_names, "config": {k: v for k, v in vars(self.cfg).items()
                                                  if isinstance(v, (int, float, str, bool, list, type(None)))},
            "execute_horizon": self.cfg.execute_horizon, "action_frequency": self.cfg.action_frequency,
            "recorder": "scripts/todo2/record_rollout.py", "maniguard_revision": "6bf4c61",
        }
        self.started = True
        self.rows.append(self._physical(0, obs, monitor))
        self.rows[-1]["recorder_s"] = time.monotonic() - t0
        self.overhead += self.rows[-1]["recorder_s"]

    def proposal(self, step, chunk, planned_steps, action_space):
        t0 = time.monotonic()
        self.proposal_id += 1
        self.proposals.append({"step": step, "chunk": np.array(chunk, dtype=np.float32, copy=True),
                               "planned": int(planned_steps)})
        self.overhead += time.monotonic() - t0

    def before_action(self, step, offset, raw, transformed, command, *, issued=True):
        self.pending = {"step": step, "proposal": self.proposal_id, "offset": int(offset),
                        "raw": np.array(raw, dtype=np.float32, copy=True),
                        "ctrl": np.array(transformed, dtype=np.float32, copy=True),
                        "clipped": np.array(command, dtype=np.float32, copy=True)}

    def applied(self, step):
        if self.pending is not None:
            self.execs.append(self.pending)
            self.pending = None

    def transition(self, step, obs):
        self.pending_obs = (step, obs)

    def oracle(self, step, monitor):
        t0 = time.monotonic()
        obs = self.pending_obs[1] if self.pending_obs and self.pending_obs[0] == step else None
        self.pending_obs = None
        row = self._physical(step, obs, monitor)
        row["recorder_s"] = time.monotonic() - t0
        self.overhead += row["recorder_s"]
        self.rows.append(row)

    def finish(self, result):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        name = (self.scene.get("name") or "episode").replace("/", "_")
        path = self.out_dir / f"{name}.npz"
        if not self.started:
            (self.out_dir / f"{name}_initialization_failed.json").write_text(json.dumps(result, default=str))
            return
        t0 = time.monotonic()
        arrays = {}
        keys = [k for k in self.rows[0] if k not in ("recorder_s",)] + ["recorder_s"]
        for k in keys:
            arrays[k] = np.stack([np.asarray(r[k]) for r in self.rows])
        if self.proposals:
            arrays["proposal_step"] = np.array([p["step"] for p in self.proposals])
            arrays["proposal_planned"] = np.array([p["planned"] for p in self.proposals])
            arrays["proposal_chunk"] = np.stack([p["chunk"] for p in self.proposals])
        if self.execs:
            for k in ("step", "proposal", "offset"):
                arrays["exec_" + k] = np.array([e[k] for e in self.execs])
            for k in ("raw", "ctrl", "clipped"):
                arrays["exec_" + k] = np.stack([e[k] for e in self.execs])
        self.meta["result"] = result
        self.meta["recorder_overhead_s"] = self.overhead
        self.meta["steps_recorded"] = len(self.rows)
        arrays["meta"] = np.array(json.dumps(self.meta, default=str))
        np.savez_compressed(path, **arrays)
        print(f"[record_rollout] wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB, "
              f"{len(self.rows)} steps, recorder overhead {self.overhead:.1f}s "
              f"+ save {time.monotonic() - t0:.1f}s)", flush=True)

    # -- state read-out --------------------------------------------------------------------
    def _physical(self, step, obs, monitor):
        robot = self.robot
        row = {"step": np.int64(step), "wall_s": np.float64(time.monotonic()),
               "q": _np(robot.get_joint_positions()).astype(np.float32),
               "dq": _np(robot.get_joint_velocities()).astype(np.float32),
               "eef_pos": _np(robot.get_eef_position()).astype(np.float32),
               "eef_quat": _np(robot.get_eef_orientation()).astype(np.float32),
               "gripper_q": _np(robot.get_joint_positions())[_np(robot.gripper_control_idx[robot.default_arm])].astype(np.float32),
               "policy_state": (np.asarray(obs["states"], dtype=np.float32) if obs is not None
                                else np.full(8, np.nan, np.float32))}
        if self.jar is not None:
            pos, quat = self.jar.get_position_orientation()
            row["jar_pos"], row["jar_quat"] = _np(pos).astype(np.float32), _np(quat).astype(np.float32)
            row["jar_link_pose"] = np.stack([np.concatenate([_np(p), _np(q)])
                                             for p, q in (l.get_position_orientation() for _, l in self.jar_links)]).astype(np.float32) \
                if self.jar_links else np.zeros((0, 7), np.float32)
            js = [j.get_state() for _, j in self.jar_joints]
            row["jar_joint_q"] = np.array([float(_np(s[0]).reshape(-1)[0]) for s in js], np.float32)
            row["jar_joint_dq"] = np.array([float(_np(s[1]).reshape(-1)[0]) for s in js], np.float32)
            row["jar_lin_vel"] = _np(self.jar.get_linear_velocity()).astype(np.float32)
            row["jar_ang_vel"] = _np(self.jar.get_angular_velocity()).astype(np.float32)
            try:
                contacts = self.jar.states[self.ContactBodies].get_value()
                row["jar_robot_contacts"] = np.int32(len(contacts & self.robot_links))
                row["jar_finger_contact"] = np.bool_(bool(contacts & self.finger_links))
            except Exception:  # noqa: BLE001
                row["jar_robot_contacts"], row["jar_finger_contact"] = np.int32(-1), np.bool_(False)
            try:
                row["grasping"] = np.int32(int(robot.is_grasping(candidate_obj=self.jar)))
            except Exception:  # noqa: BLE001
                row["grasping"] = np.int32(-1)
        row["other_obj_pos"] = (np.stack([_np(o.get_position_orientation()[0]) for o in self.others]).astype(np.float32)
                                if self.others else np.zeros((0, 3), np.float32))
        gt = monitor.recording_state(step) if monitor is not None else {"valid": False, "observation": None}
        labels = (gt["observation"] or {}).get("ap", {})
        row["ap"] = np.array([bool(labels.get(k, False)) for k in self.ap_names], bool)
        row["ap_valid"] = np.bool_(gt["valid"])
        row["monitor_doomed"] = np.bool_((gt["observation"] or {}).get("doomed", False))
        return row


def create_observer(cfg, scene):
    return RolloutRecorder(cfg, scene)
