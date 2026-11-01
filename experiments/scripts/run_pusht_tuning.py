"""Prepare, execute and close a frozen small PushT random-configuration study."""

import argparse
import collections
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from helpers.learned_motion import LearnedMotionChecker, load_motion
from helpers.lewm18_planner import PlanningCEM
from helpers.pusht_smoke import Scaler
from helpers.pusht_tuning import (
    configurations,
    layout,
    render_goal_image,
    select_configuration,
    summarize,
)
from helpers.safety_data import write_json
from helpers.safety_model import file_digest, pusht_constraints, scaler_digest
from scripts.run_pusht_smoke import run_episode

SOURCES = (
    "experiments/scripts/run_pusht_tuning.py",
    "experiments/helpers/pusht_tuning.py",
    "experiments/scripts/run_pusht_smoke.py",
    "experiments/helpers/lewm18_planner.py",
    "experiments/helpers/cem_refinements.py",
    "experiments/helpers/lewm16_planner.py",
    "experiments/helpers/learned_motion.py",
    "experiments/helpers/pusht_smoke.py",
    "experiments/helpers/safety_model.py",
    "experiments/helpers/safecem10_planner.py",
)


def hashes():
    return {name: file_digest(ROOT / name) for name in SOURCES}


def prepare(design_path, design, output):
    output.mkdir(parents=True, exist_ok=False)
    if file_digest(ROOT / design["safety_checkpoint"]) != design["safety_checkpoint_sha256"]:
        raise ValueError("Frozen safety checkpoint changed")
    if file_digest(ROOT / design["world_model_checkpoint"]) != design["world_model_sha256"]:
        raise ValueError("Frozen world model changed")
    historical = ROOT / design["historical_input_root"]
    old_panel = json.loads((historical / "pusht-panels.json").read_text())["development"]
    cases = []
    for index, entry in enumerate(old_panel):
        if entry["episode"] != design["source_episodes"][index]:
            raise ValueError("Input episode identity differs from design")
        source = historical / f"attempts/D-pusht-m0-c{index}-s180100-k0/result.json"
        record = json.loads(source.read_text())
        # No historical success, trajectory, action count or configuration score is consumed.
        initial, goal = record["dataset_initial_state"], record["goal"]
        split = "development" if index in design["development_indices"] else "evaluation"
        kind = design["layout_order"][index % 3]
        cases.append(
            {
                "id": f"case-{index:02d}-{kind}",
                "index": index,
                "split": split,
                "source_episode": entry["episode"],
                "source_file": str(source.relative_to(ROOT)),
                "source_sha256": file_digest(source),
                "initial_state": initial,
                "goal_state": goal,
                "case": layout(initial, goal, kind, design),
                "planner_seed": 29103100 + index,
            }
        )
    if len(cases) != 6 or len({c["source_episode"] for c in cases}) != 6:
        raise ValueError("Six distinct source episodes are required")
    frozen = {
        "design": design,
        "design_sha256": file_digest(design_path),
        "cases": cases,
        "configurations": configurations(design),
        "sources": hashes(),
        "information": "Original development input states only; historical final pool untouched",
    }
    write_json(output / "panel-freeze.json", frozen)
    source_root = output / "source"
    for name in SOURCES:
        dest = source_root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    from stable_worldmodel.envs.pusht.env import PushT

    counters = collections.Counter()

    class CountedPushT(PushT):
        def _set_state(self, state):
            original = self.space.step

            def step(dt):
                counters["reset_physics_steps"] += 1
                return original(dt)

            self.space.step = step
            try:
                return super()._set_state(state)
            finally:
                self.space.step = original

        def render(self):
            counters["render_frames"] += 1
            return super().render()

    inputs = output / "inputs"
    inputs.mkdir()
    try:
        for case in cases:
            env = CountedPushT()
            try:
                counters["environment_resets"] += 1
                env.reset(
                    seed=case["planner_seed"],
                    options={
                        "state": np.asarray(case["initial_state"]),
                        "goal_state": np.asarray(case["goal_state"]),
                    },
                )
                start_image = env.render().copy()
                goal = np.asarray(case["goal_state"])
                goal_image = render_goal_image(env, goal)
                if np.abs(start_image.astype(float) - goal_image.astype(float)).mean() < 0.5:
                    raise ValueError("Start and target images are unexpectedly indistinguishable")
                path = inputs / f"{case['id']}.npz"
                np.savez_compressed(
                    path,
                    initial_state=case["initial_state"],
                    goal_state=goal,
                    start_image=start_image,
                    goal_image=goal_image,
                )
                case["pair_sha256"] = file_digest(path)
            finally:
                env.close()
    finally:
        write_json(output / "preparation-resources.json", dict(counters))
    shutil.copy2(ROOT / design["scalers"], inputs / "scalers.json")
    write_json(
        output / "prepared-panel.json",
        {
            **frozen,
            "cases": cases,
            "scalers_sha256": file_digest(inputs / "scalers.json"),
            "preparation_resources": dict(counters),
        },
    )
    print(
        json.dumps(
            {
                "prepared_cases": len(cases),
                "configurations": len(frozen["configurations"]),
                "resources": dict(counters),
            }
        )
    )


def load_runtime(design, panel, output):
    if hashes() != panel["sources"]:
        raise ValueError("Study sources differ from the pre-outcome freeze")
    device = design.get("device", "cpu")
    head, _ = load_motion(ROOT / design["safety_checkpoint"], device=device)
    if file_digest(ROOT / design["safety_checkpoint"]) != design["safety_checkpoint_sha256"]:
        raise ValueError("Safety checkpoint changed")
    checkpoint = ROOT / design["world_model_checkpoint"]
    if file_digest(checkpoint) != design["world_model_sha256"]:
        raise ValueError("World-model checkpoint changed")
    if head.spec.world_model_sha256 != design["world_model_sha256"]:
        raise ValueError("Safety head belongs to a different frozen world model")
    from stable_worldmodel.policy import AutoCostModel

    model = AutoCostModel(str(checkpoint)[: -len("_object.ckpt")], cache_dir=checkpoint.parent)
    model.to(device).eval().requires_grad_(False)
    if file_digest(output / "inputs/scalers.json") != panel["scalers_sha256"]:
        raise ValueError("Study scalers changed")
    columns = json.loads((output / "inputs/scalers.json").read_text())["columns"]
    action = columns["action"]
    if scaler_digest(action["mean"], action["scale"]) != head.spec.action_scaler_sha256:
        raise ValueError("Safety head and task action normalization differ")
    process = {key: Scaler(value["mean"], value["scale"]) for key, value in columns.items()}
    return head, model, process


def trial(design, panel, output, runtime, config, case, stage):
    head, model, process = runtime
    path = output / "inputs" / f"{case['id']}.npz"
    if file_digest(path) != case["pair_sha256"]:
        raise ValueError("Frozen pair changed")
    with np.load(path, allow_pickle=False) as data:
        pair = dict(data)
    directory = output / stage / config["id"] / case["id"]
    if directory.exists():
        raise FileExistsError(directory)
    protocol = {
        "seed": case["planner_seed"],
        "planner": config["params"],
        "margin_px": design["margin_px"],
        "max_episode_actions": design["max_episode_actions"],
        "max_runtime_minutes": design["max_episode_runtime_minutes"],
        "verify_analytic_dynamics": False,
    }
    write_json(
        output / "active.json",
        {
            "stage": stage,
            "configuration": config["id"],
            "case": case["id"],
            "started_epoch": time.time(),
        },
    )
    counters = collections.Counter()

    def factory(shape, margin, scaler, resources, enabled):
        if not enabled:
            raise ValueError("Learned safety must remain enabled")
        return LearnedMotionChecker(head, pusht_constraints(shape), margin, resources)

    def forbidden(*args, **kwargs):
        raise RuntimeError("Reference motion prediction is forbidden in this study")

    try:
        with ExitStack() as guard:
            for target in (
                "scripts.run_pusht_smoke.pd_response",
                "helpers.safecem10_planner.pd_response",
                "helpers.pusht_smoke.ShapeChecker.check",
                "helpers.pusht_smoke.ShapeChecker.paths",
            ):
                guard.enter_context(patch(target, side_effect=forbidden))
            result, _ = run_episode(
                "safety-model",
                case["case"],
                model,
                process,
                pair,
                protocol,
                directory,
                time.time(),
                checker_factory=factory,
                planner_type=PlanningCEM,
                device=design.get("device", "cpu"),
                resource_counters=counters,
            )
    except Exception as exc:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(
            directory / "engineering-failure.json",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "resources": dict(counters),
            },
        )
        raise
    result.update(
        {
            "study_case_id": case["id"],
            "study_split": case["split"],
            "configuration_id": config["id"],
            "reference_predictors_forbidden": True,
        }
    )
    write_json(directory / "result.json", result)
    if result["error"]:
        raise RuntimeError("Engineering error preserved; stop before further trials")
    return result


def development(design, panel, output):
    marker = output / "development-freeze.json"
    if marker.exists():
        raise FileExistsError(marker)
    write_json(
        marker,
        {
            "panel_sha256": file_digest(output / "prepared-panel.json"),
            "sources": hashes(),
            "ordered_configurations": panel["configurations"],
        },
    )
    runtime = load_runtime(design, panel, output)
    cases = [c for c in panel["cases"] if c["split"] == "development"]
    results = {}
    for config in panel["configurations"]:
        results[config["id"]] = [
            trial(design, panel, output, runtime, config, c, "development") for c in cases
        ]
        write_json(
            output / "development-progress.json",
            {key: summarize(rows) for key, rows in results.items()},
        )
    selection = select_configuration(panel["configurations"], results, [c["id"] for c in cases])
    selection.update(
        {
            "development_panel_sha256": file_digest(output / "prepared-panel.json"),
            "development_trial_result_sha256": {
                str(p.relative_to(output)): file_digest(p)
                for p in (output / "development").glob("*/*/result.json")
            },
        }
    )
    write_json(output / "selection.json", selection)
    print(
        json.dumps(
            {
                "selection": selection["selected"],
                "development_summaries": selection["development_summaries"],
            }
        )
    )


def evaluate(design, panel, output):
    selection = json.loads((output / "selection.json").read_text())
    marker = output / "evaluation-freeze.json"
    if marker.exists():
        raise FileExistsError(marker)
    write_json(
        marker,
        {
            "selection_sha256": file_digest(output / "selection.json"),
            "panel_sha256": file_digest(output / "prepared-panel.json"),
            "sources": hashes(),
        },
    )
    runtime = load_runtime(design, panel, output)
    cases = [c for c in panel["cases"] if c["split"] == "evaluation"]
    current, selected = panel["configurations"][0], selection["selected"]
    arms = [current] if selected["id"] == current["id"] else [current, selected]
    results = {}
    for config in arms:
        results[config["id"]] = [
            trial(design, panel, output, runtime, config, c, "evaluation") for c in cases
        ]
    write_json(
        output / "evaluation-results.json",
        {
            "selected_id": selected["id"],
            "shared_observations": len(arms) == 1,
            "current": summarize(results[current["id"]]),
            "selected": summarize(results[selected["id"]]),
            "paired": [
                {
                    "case": c["id"],
                    "current_safe": results[current["id"]][i]["safe_completion"],
                    "selected_safe": results[selected["id"]][i]["safe_completion"],
                }
                for i, c in enumerate(cases)
            ],
        },
    )
    print((output / "evaluation-results.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "development", "evaluate"))
    parser.add_argument(
        "--design", type=Path, default=ROOT / "configs/pusht_tuning_20261029/design.json"
    )
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    if not os.environ.get("RUNPOD_POD_ID") and not os.environ.get("VAST_CONTAINERLABEL"):
        raise RuntimeError("Experiment execution requires a RunPod or Vast host")
    if design.get("execution_pod_id") != os.environ.get("RUNPOD_POD_ID"):
        raise RuntimeError("RunPod host differs from the frozen study host")
    if design.get("device") != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The remote study requires CUDA; CPU fallback is forbidden")
    output = ROOT / design["output"]
    torch.set_num_threads(4)
    if (output / "invalidated.json").exists():
        raise ValueError(
            "This stage was invalidated; its results cannot select or evaluate settings"
        )
    if args.stage == "prepare":
        prepare(args.design, design, output)
    else:
        panel = json.loads((output / "prepared-panel.json").read_text())
        if file_digest(args.design) != panel["design_sha256"]:
            raise ValueError("Study design changed after panel preparation")
        {"development": development, "evaluate": evaluate}[args.stage](design, panel, output)


if __name__ == "__main__":
    main()
