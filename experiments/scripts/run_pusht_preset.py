"""Frozen PushT author-preset comparison, bounded tuning and paired hazard controls."""

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
from helpers.pusht_preset import KINDS, configurations, geometry, select, summary
from helpers.pusht_smoke import Scaler
from helpers.pusht_tuning import render_goal_image
from helpers.safety_data import write_json
from helpers.safety_model import file_digest, pusht_constraints
from scripts.run_pusht_smoke import run_episode

SOURCES = [
    "experiments/scripts/run_pusht_preset.py",
    "experiments/helpers/pusht_preset.py",
    "experiments/scripts/run_pusht_smoke.py",
    "experiments/helpers/pusht_smoke.py",
    "experiments/helpers/learned_motion.py",
    "experiments/helpers/lewm18_planner.py",
    "experiments/helpers/cem_refinements.py",
    "experiments/helpers/lewm16_planner.py",
    "experiments/helpers/safecem10_planner.py",
    "experiments/helpers/safety_model.py",
    "experiments/helpers/pusht_tuning.py",
]


def hashes():
    return {name: file_digest(ROOT / name) for name in SOURCES}


def prepare(design, design_path, out):
    import zarr
    from stable_worldmodel.envs.pusht.env import PushT

    out.mkdir(parents=True, exist_ok=False)
    for key, fingerprint in (
        ("world_model", "world_model_sha256"),
        ("motion_model", "motion_model_sha256"),
    ):
        if file_digest(ROOT / design[key]) != design[fingerprint]:
            raise ValueError("Frozen model changed")
    sources = hashes()
    write_json(
        out / "preparation-freeze.json",
        {
            "design": design,
            "design_sha256": file_digest(design_path),
            "sources": sources,
            "configurations": configurations(design),
        },
    )
    for name in SOURCES:
        p = out / "source" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, p)
    data = zarr.open_group(str(ROOT / design["pose_data"]), mode="r")
    states = np.asarray(data["data/state"])
    ends = np.asarray(data["meta/episode_ends"])
    counters = collections.Counter()
    cases = []
    rejected = []

    class Counted(PushT):
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

    inputs = out / "inputs"
    inputs.mkdir()
    starts = np.r_[0, ends[:-1]]
    try:
        for episode, (begin, end) in enumerate(zip(starts, ends)):
            if len(cases) == 12:
                break
            for row in range(int(begin), int(end) - 25, 25):
                a, b = states[row].copy(), states[row + 25].copy()
                if np.linalg.norm(a[:2] - b[:2]) < 100 or np.linalg.norm(a[2:4] - b[2:4]) < 30:
                    continue
                if not (((np.r_[a[:2], b[:2]] >= 70) & (np.r_[a[:2], b[:2]] <= 442)).all()):
                    continue
                if counters["environment_resets"] >= design["preparation_max_resets"]:
                    raise RuntimeError("Preparation reset budget exhausted")
                a[4] %= 2 * np.pi
                b[4] %= 2 * np.pi
                initial = np.r_[a, np.zeros(2)]
                goal = np.r_[b, np.zeros(2)]
                index = len(cases)
                kind, strength = KINDS[index % 6]
                seed = 29110500 + index
                env = Counted()
                try:
                    counters["environment_resets"] += 1
                    obs, _ = env.reset(seed=seed, options={"state": initial, "goal_state": goal})
                    settled = np.asarray(obs["state"]).copy()
                    shape = geometry(settled, goal, states[row : row + 26, :2], kind, strength)
                    if shape is None:
                        rejected.append(
                            {"episode": episode, "row": row, "reason": "geometry admission"}
                        )
                        continue
                    if env.eval_state(goal, settled)[0]:
                        raise ValueError("Already solved start")
                    start_image = env.render().copy()
                    try:
                        goal_image = render_goal_image(env, goal)
                    except ValueError as exc:
                        rejected.append({"episode": episode, "row": row, "reason": str(exc)})
                        continue
                    if np.abs(start_image.astype(float) - goal_image.astype(float)).mean() < 0.5:
                        rejected.append(
                            {"episode": episode, "row": row, "reason": "indistinct target"}
                        )
                        continue
                    case = {
                        "id": f"case-{index:02d}-{kind}-{strength}",
                        "index": index,
                        "split": "development" if index < 6 else "evaluation",
                        "source_episode": episode,
                        "source_row": row,
                        "goal_row": row + 25,
                        "seed": seed,
                        "shape": shape,
                        "settled_initial": settled.tolist(),
                        "initial_state": initial.tolist(),
                        "goal_state": goal.tolist(),
                    }
                    path = inputs / (case["id"] + ".npz")
                    np.savez_compressed(
                        path,
                        initial_state=initial,
                        goal_state=goal,
                        start_image=start_image,
                        goal_image=goal_image,
                    )
                    case["pair_sha256"] = file_digest(path)
                    cases.append(case)
                    break
                finally:
                    env.close()
        if len(cases) != 12:
            raise ValueError("Not enough admissible distinct episodes")
    finally:
        write_json(out / "preparation-resources.json", dict(counters))
    shutil.copy2(ROOT / design["scalers"], inputs / "scalers.json")
    write_json(
        out / "panel.json",
        {
            "design": design,
            "design_sha256": file_digest(design_path),
            "sources": sources,
            "cases": cases,
            "configurations": configurations(design),
            "preparation_resources": dict(counters),
            "rejected": rejected,
            "scaler_sha256": file_digest(inputs / "scalers.json"),
        },
    )
    print(json.dumps({"cases": len(cases), "resources": dict(counters), "rejected": len(rejected)}))


def runtime(design):
    from stable_worldmodel.policy import AutoCostModel

    p = ROOT / design["world_model"]
    if (
        file_digest(p) != design["world_model_sha256"]
        or file_digest(ROOT / design["motion_model"]) != design["motion_model_sha256"]
    ):
        raise ValueError("A frozen checkpoint changed")
    if file_digest(ROOT / design["scalers"]) != design["scalers_sha256"]:
        raise ValueError("Frozen action normalization changed")
    head, _ = load_motion(ROOT / design["motion_model"], device="cuda")
    if head.spec.max_horizon != 5 or not head.calibration["ready"]:
        raise ValueError("The shared safety model must support all 25 future actions")
    model = AutoCostModel(str(p)[: -len("_object.ckpt")], cache_dir=p.parent).to("cuda")
    model.eval().requires_grad_(False)
    columns = json.loads((ROOT / design["scalers"]).read_text())["columns"]
    process = {k: Scaler(v["mean"], v["scale"]) for k, v in columns.items()}
    return head, model, process


def trial(design, out, rt, config, case, condition):
    head, model, process = rt
    if file_digest(out / "inputs" / (case["id"] + ".npz")) != case["pair_sha256"]:
        raise ValueError("Frozen case inputs changed")
    with np.load(out / "inputs" / (case["id"] + ".npz"), allow_pickle=False) as data:
        pair = dict(data)
    shape = (
        case["shape"]
        if condition == "hazard"
        else {"kind": "none", "origin": [0, 0], "basis": [[1, 0], [0, 1]]}
    )
    folder = out / case["split"] / config["id"] / case["id"] / condition
    protocol = {
        "planner": config["params"],
        "replan_blocks": config["params"]["replan_blocks"],
        "seed": case["seed"],
        "margin_px": 6,
        "max_episode_actions": 50,
        "max_runtime_minutes": 5,
        "verify_analytic_dynamics": False,
    }
    write_json(
        out / "active.json",
        {
            "split": case["split"],
            "config": config["id"],
            "case": case["id"],
            "condition": condition,
            "epoch": time.time(),
        },
    )
    counters = collections.Counter()

    def factory(shape, margin, scaler, resources, enabled):
        if not enabled:
            raise ValueError("Safety must remain active")
        return LearnedMotionChecker(head, pusht_constraints(shape), margin, resources)

    def forbidden(*args, **kwargs):
        raise RuntimeError("Reference prediction is forbidden")

    try:
        with ExitStack() as guard:
            for name in (
                "scripts.run_pusht_smoke.pd_response",
                "helpers.safecem10_planner.pd_response",
                "helpers.pusht_smoke.ShapeChecker.check",
                "helpers.pusht_smoke.ShapeChecker.paths",
            ):
                guard.enter_context(patch(name, side_effect=forbidden))
            row, _ = run_episode(
                "safety-model",
                shape,
                model,
                process,
                pair,
                protocol,
                folder,
                time.time(),
                checker_factory=factory,
                device="cuda",
                resource_counters=counters,
            )
    except Exception as exc:
        folder.mkdir(parents=True, exist_ok=True)
        write_json(
            folder / "engineering-failure.json", {"error": str(exc), "resources": dict(counters)}
        )
        raise
    if not np.allclose(row["initial_state"], case["settled_initial"], atol=1e-9, rtol=0):
        raise ValueError("Actual reset changed after admission")
    row.update(
        config_id=config["id"],
        case_id=case["id"],
        case_index=case["index"],
        condition=condition,
        split=case["split"],
        reference_predictors_forbidden=True,
        planner=config["params"],
    )
    write_json(folder / "result.json", row)
    if row["error"]:
        raise RuntimeError("Engineering error; stop without hiding its costs")
    return row


def load_results(out, config, cases):
    return [
        json.loads(
            (out / c["split"] / config["id"] / c["id"] / condition / "result.json").read_text()
        )
        for c in cases
        for condition in ("control", "hazard")
    ]


def execute(stage, design, out, panel):
    configs = panel["configurations"]
    cases = [c for c in panel["cases"] if c["split"] == "development"]
    if stage == "evaluate":
        selection = json.loads((out / "selection.json").read_text())
        configs = (
            [configs[0]]
            if selection["selected"]["id"] == "authors"
            else [configs[0], selection["selected"]]
        )
        cases = [c for c in panel["cases"] if c["split"] == "evaluation"]
    else:
        configs = configs[:2] if stage == "probe" else configs[2:]
    marker = out / (stage + "-freeze.json")
    if marker.exists():
        raise FileExistsError("Inspect existing stage before retry")
    write_json(
        marker,
        {
            "sources": hashes(),
            "panel_sha256": file_digest(out / "panel.json"),
            "selection_sha256": file_digest(out / "selection.json")
            if stage == "evaluate"
            else None,
            "configurations": configs,
        },
    )
    rt = runtime(design)
    for config in configs:
        rows = [
            trial(design, out, rt, config, c, condition)
            for c in cases
            for condition in ("control", "hazard")
        ]
        write_json(
            out / (stage + "-progress.json"), {"last": config["id"], "summary": summary(rows)}
        )
    if stage == "probe":
        write_json(
            out / "probe-results.json",
            {
                c["id"]: {
                    condition: summary(
                        [r for r in load_results(out, c, cases) if r["condition"] == condition]
                    )
                    for condition in ("control", "hazard")
                }
                for c in configs
            },
        )
    if stage == "tune":
        results = {c["id"]: load_results(out, c, cases) for c in panel["configurations"]}
        selected = select(panel["configurations"], results)
        selected["development_result_hashes"] = {
            str(p.relative_to(out)): file_digest(p)
            for p in (out / "development").glob("*/*/*/result.json")
        }
        write_json(out / "selection.json", selected)
        print(json.dumps(selected), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "probe", "tune", "evaluate"))
    parser.add_argument("--design", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    out = ROOT / design["output"]
    if os.environ.get("RUNPOD_POD_ID") != design["pod_id"] or not torch.cuda.is_available():
        raise RuntimeError("This study requires its frozen remote CUDA host")
    torch.set_num_threads(4)
    if args.stage == "prepare":
        prepare(design, args.design, out)
    else:
        panel = json.loads((out / "panel.json").read_text())
        if hashes() != panel["sources"] or file_digest(args.design) != panel["design_sha256"]:
            raise ValueError("Protocol or source changed after freeze")
        execute(args.stage, design, out, panel)


if __name__ == "__main__":
    main()
