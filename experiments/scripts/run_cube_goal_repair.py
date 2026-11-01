"""Repair Cube goal-marker consistency before the new, fully retained comparison."""

import os

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import importlib

importlib.import_module("triton")
import collections
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from scripts import run_portable_lewm as study
from helpers.portable_env import NativeEnv
from helpers.safety_data import write_json
from helpers.safety_model import file_digest

STRIDE = None
ACTION_CAP = None


class RecordedCube(NativeEnv):
    def checkpoint(self, status):
        super().checkpoint(status)
        if self.name == "cube" and status in ("reset_complete", "action_complete"):
            self.counters["task_monitor_queries"] += 1
            row = {
                "action": self.counters["environment_actions"],
                "cube_xyz": self.env._data.joint("object_joint_0").qpos[:3].tolist(),
                "goal_xyz": None if self.goal is None else np.asarray(self.goal).tolist(),
                "robot_proprioception": self.observe().tolist(),
            }
            with Path(self.journal.name).with_name("object-states.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")

    def step(self, action):
        if ACTION_CAP is not None and self.counters["environment_actions"] >= ACTION_CAP:
            raise RuntimeError("Bounded renderer-equivalence probe reached its six-action cap")
        return super().step(action)

    def render(self):
        count = self.counters["environment_actions"]
        previous = getattr(self, "_last_frame_action", None)
        refresh = STRIDE is None or count % STRIDE == 0 or self.success() or previous == count
        if refresh or not hasattr(self, "_last_frame"):
            self._last_frame = super().render()
        self._last_frame_action = count
        return self._last_frame.copy()


def main():
    global STRIDE, ACTION_CAP
    if os.environ.get("RUNPOD_POD_ID") != study.DESIGN["pod_id"]:
        raise RuntimeError("Remote pod only")
    previous = study.OUTPUT / "cube-v2"
    out = study.OUTPUT / "cube-v3"
    out.mkdir(parents=True, exist_ok=False)
    (out / "motion").mkdir()
    (out / "inputs").mkdir()
    freeze = {
        "epoch": time.time(),
        "reason": "Correct goal marker before rendering goal frame; original engineering trials retained and excluded.",
        "same_cases": True,
        "same_models": True,
        "same_settings_and_selection": True,
        "source": file_digest(__file__),
        "original_panel": file_digest(previous / "panel.json"),
        "renderer": "Cache observed frames only between policy replans, after a paired six-action, two-replan equivalence check. Always capture fresh at replans, completion and final frame. Native action and state logging remains every action.",
        "engineering_allowance": {"max_probes": 2, "max_probe_actions": 12, "max_probe_resets": 2},
    }
    write_json(out / "goal-repair-freeze.json", freeze)
    for name in ("model.pt", "report.json"):
        shutil.copy2(previous / "motion" / name, out / "motion" / name)
    write_json(out / "motion/resources.json", {})
    write_json(
        out / "motion/reused-from.json",
        {
            "path": str(previous / "motion"),
            "model_sha256": file_digest(out / "motion/model.pt"),
            "new_training_interactions": 0,
        },
    )
    study.freeze(out, "preparation")
    panel = json.loads((previous / "panel.json").read_text())
    dataset = study.cube_data("val")
    c = collections.Counter()
    env = RecordedCube("cube", c, out / "preparation-interactions.jsonl")
    changes = []
    try:
        for case in panel["cases"]:
            with np.load(previous / "inputs" / (case["id"] + ".npz")) as z:
                pair = dict(z)
            env.reset(case["seed"], pair["initial"], pair["goal"])
            pair["start_image"] = env.render()
            future = dataset["states"][case["source_row"] + 25]
            _, (image, goal) = env.pose(future, render=True)
            if not np.allclose(goal, pair["goal"], atol=1e-8):
                raise RuntimeError("Object goal changed")
            difference = float(
                np.abs(image.astype(float) - pair["goal_image"].astype(float)).mean()
            )
            pair["goal_image"] = image
            target = out / "inputs" / (case["id"] + ".npz")
            np.savez_compressed(target, **pair)
            changes.append(
                {
                    "case": case["id"],
                    "old_pair_sha256": case["sha256"],
                    "goal_frame_mean_pixel_change": difference,
                }
            )
            case["sha256"] = file_digest(target)
    finally:
        env.close()
        write_json(out / "preparation-resources.json", dict(c))
    panel.update(
        sources=json.loads((out / "preparation-freeze.json").read_text())["sources"],
        resources=dict(c),
        motion_model_sha256=file_digest(out / "motion/model.pt"),
        goal_frame_repair={"source": file_digest(__file__), "cases": changes},
    )
    write_json(out / "panel.json", panel)
    study.NativeEnv = RecordedCube
    rt = study.runtime("cube", out)
    case = {**panel["cases"][0], "split": "engineering-full-frames"}
    config = panel["configurations"][1]
    ACTION_CAP = 6
    STRIDE = 1
    study.trial("cube", out, rt, config, case, "control")
    case2 = {**case, "split": "engineering-plan-frames"}
    STRIDE = 5
    study.trial("cube", out, rt, config, case2, "control")

    def trace(case):
        p = out / case["split"] / config["id"] / case["id"] / "control/trace.npz"
        with np.load(p) as z:
            return z["actions"], z["states"]

    a, s = trace(case)
    b, t = trace(case2)
    equivalent = len(a) >= 6 and np.allclose(a, b, atol=1e-9) and np.allclose(s, t, atol=1e-9)
    write_json(
        out / "renderer-equivalence.json",
        {
            "passed": bool(equivalent),
            "actions_per_probe": len(a),
            "max_action_error": float(np.max(np.abs(a - b))),
            "max_state_error": float(np.max(np.abs(s - t))),
        },
    )
    probes = study.OUTPUT / "cube-renderer-probes-20261030"
    probes.mkdir(exist_ok=False)
    for folder in ("engineering-full-frames", "engineering-plan-frames"):
        shutil.move(out / folder, probes / folder)
    # Fall back to ordinary capture if equivalence fails, before any optimizer trials.
    capture_fast = bool(equivalent)
    ACTION_CAP = None
    STRIDE = None
    original = study.trial

    def recorded_trial(name, root, rt, config, case, condition):
        global STRIDE
        STRIDE = config["params"]["replan_blocks"] * 5 if capture_fast else 1
        return original(name, root, rt, config, case, condition)

    study.trial = recorded_trial
    del rt
    for stage in ("tune", "evaluate", "report"):
        start = time.time()
        p = out / (stage + "-timing.json")
        write_json(p, {"stage": stage, "status": "running", "epoch": start})
        if stage == "report":
            study.report("cube", out)
        else:
            study.compare("cube", out, stage)
        write_json(
            p, {"stage": stage, "status": "complete", "elapsed_seconds": time.time() - start}
        )


if __name__ == "__main__":
    main()
