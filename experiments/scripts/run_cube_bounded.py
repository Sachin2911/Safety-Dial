"""Cube action-contract repair: bound actual commands exactly as safety inference does."""

import os

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import importlib

importlib.import_module("triton")
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from scripts import run_portable_lewm as study
from scripts.run_cube_goal_repair import RecordedCube
from helpers.pusht_smoke import Scaler
from helpers.safety_data import write_json
from helpers.safety_model import file_digest

LAST_COMMAND = None


class BoundedScaler(Scaler):
    def inverse_transform(self, value):
        global LAST_COMMAND
        raw = super().inverse_transform(value)
        LAST_COMMAND = np.asarray(raw).copy()
        return np.clip(raw, -1.0, 1.0)


class BoundedCube(RecordedCube):
    def step(self, action):
        action = np.asarray(action)
        if not np.isfinite(action).all() or bool((np.abs(action) > 1 + 1e-8).any()):
            raise RuntimeError("The policy must supply a finite command in the native action Box")
        record = {
            "action_index": self.counters["environment_actions"],
            "commanded_before_bounds": np.asarray(LAST_COMMAND).reshape(-1).tolist(),
            "executed": action.tolist(),
        }
        with Path(self.journal.name).with_name("applied-actions.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return super().step(action)


def main():
    if os.environ.get("RUNPOD_POD_ID") != study.DESIGN["pod_id"]:
        raise RuntimeError("Remote pod only")
    prior = study.OUTPUT / "cube-v3"
    out = study.OUTPUT / "cube-v4"
    out.mkdir(parents=True, exist_ok=False)
    (out / "motion").mkdir()
    for file in ("model.pt", "report.json"):
        shutil.copy2(prior / "motion" / file, out / "motion" / file)
    write_json(out / "motion/resources.json", {})
    write_json(
        out / "motion/reused-from.json",
        {
            "path": str(study.OUTPUT / "cube-v2/motion"),
            "model_sha256": file_digest(out / "motion/model.pt"),
            "new_training_interactions": 0,
        },
    )
    shutil.copytree(prior / "inputs", out / "inputs")
    panel = json.loads((prior / "panel.json").read_text())
    protocol = {
        "epoch": time.time(),
        "source_sha256": file_digest(__file__),
        "reason": "Native OGBench unnormalize_action does not clip, but safety inference assumes bounded commands. Enforce native Box at policy output for both arms.",
        "action_bounds": [-1, 1],
        "unchanged": [
            "frozen world model",
            "learned safety model and calibration",
            "all twelve physical cases and goal images",
            "hazards/margins",
            "planning parameter configurations",
            "development-only selection",
            "150-action/270k-predictor-step caps",
        ],
        "previous_pass": "cube-v3; all completed commands and failures retained, no reserved outcomes used.",
        "candidate_task_cost": "Same upstream unbounded latent task-cost predictions as the other environment pilots. Safety predictions and actually applied commands use the declared Box bounds.",
        "maximum_new_trials": 120,
        "maximum_new_environment_actions": 18000,
    }
    write_json(out / "action-contract-freeze.json", protocol)
    study.freeze(out, "preparation")
    panel["sources"] = json.loads((out / "preparation-freeze.json").read_text())["sources"]
    panel["action_contract_repair"] = protocol
    write_json(out / "panel.json", panel)
    write_json(out / "preparation-resources.json", {})
    for path in (Path(__file__), ROOT / "experiments/scripts/run_cube_goal_repair.py"):
        dest = out / "source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    original = study.scaler

    def bounded_scaler(name):
        base = original(name)
        return BoundedScaler(base.mean_, base.scale_)

    study.scaler = bounded_scaler
    study.NativeEnv = BoundedCube
    for stage in ("tune", "evaluate", "report"):
        start = time.time()
        path = out / (stage + "-timing.json")
        write_json(path, {"stage": stage, "status": "running", "epoch": start})
        try:
            if stage == "report":
                study.report("cube", out)
            else:
                study.compare("cube", out, stage)
        except BaseException:
            write_json(
                path, {"stage": stage, "status": "failed", "elapsed_seconds": time.time() - start}
            )
            raise
        write_json(
            path, {"stage": stage, "status": "complete", "elapsed_seconds": time.time() - start}
        )


if __name__ == "__main__":
    main()
