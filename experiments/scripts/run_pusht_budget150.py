"""Posthoc fixed-setting PushT budget sensitivity, preserving the original study."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import collections
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from scripts.run_pusht_preset import runtime, SOURCES
from scripts.run_pusht_smoke import run_episode
from helpers.learned_motion import LearnedMotionChecker
from helpers.safety_model import pusht_constraints, file_digest
from helpers.safety_data import write_json


def main():
    if os.environ.get("RUNPOD_POD_ID") != "r8w4b6lwkbjyrz":
        raise RuntimeError("Remote pod only")
    prior = ROOT / "outputs/pusht-preset-20261029-v1"
    out = ROOT / "outputs/pusht-budget150-20261030-v1"
    out.mkdir(parents=True, exist_ok=False)
    panel = json.loads((prior / "panel.json").read_text())
    report = json.loads((prior / "report.json").read_text())
    design = json.loads((ROOT / "configs/pusht_preset_20261029/design.json").read_text())
    configs = [panel["configurations"][0], report["selected"]]
    protocol = {
        "purpose": "User-approved longer action budget; posthoc diagnostic on the existing cases, not a new independent confirmation.",
        "settings": "Freeze authors and already-selected random-02; no new tuning or case selection.",
        "episode_actions": 150,
        "completion_checkpoints": [50, 100, 150],
        "maximum_trials": 24,
        "max_environment_actions": 3600,
        "max_search_predictor_steps_per_episode": 270000,
        "native_margin_px": 6,
        "seed": "Same per-case seed as original; audit first50-action prefix.",
        "models": {k: design[k] for k in ("world_model_sha256", "motion_model_sha256")},
        "source_sha256": {name: file_digest(ROOT / name) for name in SOURCES},
        "driver_sha256": file_digest(__file__),
        "epoch": time.time(),
    }
    write_json(out / "freeze.json", protocol)
    for name in [*SOURCES, str(Path(__file__).relative_to(ROOT))]:
        p = out / "source" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, p)
    head, model, process = runtime(design)
    results = []
    for config in configs:
        for case in panel["cases"][6:]:
            with np.load(prior / "inputs" / (case["id"] + ".npz")) as z:
                pair = dict(z)
            for condition in ("control", "hazard"):
                shape = (
                    case["shape"]
                    if condition == "hazard"
                    else {"kind": "none", "origin": [0, 0], "basis": [[1, 0], [0, 1]]}
                )
                folder = out / "evaluation" / config["id"] / case["id"] / condition
                run_protocol = {
                    "planner": config["params"],
                    "replan_blocks": config["params"]["replan_blocks"],
                    "seed": case["seed"],
                    "margin_px": 6,
                    "max_episode_actions": 150,
                    "max_runtime_minutes": 15,
                    "verify_analytic_dynamics": False,
                }
                counters = collections.Counter()

                def factory(shape, margin, scaler, resources, enabled):
                    return LearnedMotionChecker(head, pusht_constraints(shape), margin, resources)

                def forbidden(*args, **kwargs):
                    raise RuntimeError("Reference predictions forbidden")

                with ExitStack() as guard:
                    for name in (
                        "scripts.run_pusht_smoke.pd_response",
                        "helpers.safecem10_planner.pd_response",
                        "helpers.pusht_smoke.ShapeChecker.check",
                        "helpers.pusht_smoke.ShapeChecker.paths",
                    ):
                        guard.enter_context(patch(name, side_effect=forbidden))
                    result, _ = run_episode(
                        "island",
                        shape,
                        model,
                        process,
                        pair,
                        run_protocol,
                        folder,
                        time.time(),
                        checker_factory=factory,
                        resource_counters=counters,
                    )
                old = prior / "evaluation" / config["id"] / case["id"] / condition / "trace.npz"
                with np.load(old) as z:
                    old_states = z["states"]
                    old_actions = z["actions"]
                with np.load(folder / "trace.npz") as z:
                    new_states = z["states"]
                    new_actions = z["actions"]
                n = min(len(old_states), len(new_states))
                state_error = float(np.max(np.abs(old_states[:n] - new_states[:n])))
                a = min(len(old_actions), len(new_actions))
                action_error = (
                    float(np.max(np.abs(old_actions[:a] - new_actions[:a]))) if a else 0.0
                )
                result.update(
                    configuration=config["id"],
                    case_id=case["id"],
                    condition=condition,
                    original_prefix_state_max_error=state_error,
                    original_prefix_action_max_error=action_error,
                    original_prefix_matches=state_error < 1e-7 and action_error < 1e-7,
                )
                write_json(folder / "result.json", result)
                results.append(result)
                write_json(
                    out / "progress.json",
                    {
                        "trials": len(results),
                        "resources": dict(
                            sum(
                                (collections.Counter(r["resources"]) for r in results),
                                collections.Counter(),
                            )
                        ),
                    },
                )
    summaries = {}
    for config in configs:
        summaries[config["id"]] = {}
        for condition in ("control", "hazard"):
            rows = [
                r
                for r in results
                if r["configuration"] == config["id"] and r["condition"] == condition
            ]
            summaries[config["id"]][condition] = {
                "episodes": len(rows),
                "completion_at": {
                    str(k): sum(r["safe_completion"] and r["actions"] <= k for r in rows)
                    for k in (50, 100, 150)
                },
                "violations": sum(r["margin_breach"] or r["arena_margin_breach"] for r in rows),
                "unfinished_at150": sum(r["unfinished"] for r in rows),
            }
    write_json(
        out / "report.json",
        {
            "protocol": protocol,
            "summaries": summaries,
            "all_original_prefixes_match": all(r["original_prefix_matches"] for r in results),
            "resources": dict(
                sum((collections.Counter(r["resources"]) for r in results), collections.Counter())
            ),
            "results": results,
        },
    )
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
