"""Collect bounded native PushT motion labels, fit once and evaluate independent sources."""

import argparse
import collections
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from helpers.learned_motion import (
    MotionSpec,
    calibrate_motion,
    fit_motion,
    motion_errors,
    save_motion,
)
from helpers.safety_data import write_json
from helpers.safety_model import file_digest, scaler_digest


def collect(config, scaler, output):
    from stable_worldmodel.envs.pusht.env import PushT

    counters = collections.Counter()

    class CountedPushT(PushT):
        def _set_state(self, state):
            original = self.space.step

            def reset_step(dt):
                counters["reset_physics_steps"] += 1
                return original(dt)

            self.space.step = reset_step
            try:
                return super()._set_state(state)
            finally:
                self.space.step = original

    env = CountedPushT()
    observations, actions, positions, groups, splits = [], [], [], [], []
    partitions = {
        split: config[f"{split}_scene_seeds"] for split in ("train", "calibration", "test")
    }
    all_seeds = sum(list(partitions.values()), [])
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Source scenes overlap across partitions")
    normal_mean, normal_scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
    try:
        for split, seeds in partitions.items():
            for seed in seeds:
                rng = np.random.default_rng(seed)
                initial = np.r_[
                    rng.uniform(*config["pusher_position_range_px"], 2),
                    rng.uniform(*config["block_position_range_px"], 2),
                    rng.uniform(0, 2 * np.pi),
                    rng.uniform(*config["pusher_velocity_range_px_s"], 2),
                ]
                for trajectory in range(config["trajectories_per_scene"]):
                    counters["environment_resets"] += 1
                    obs, _ = env.reset(seed=seed, options={"state": initial})
                    dense = []
                    original = env.space.step

                    def record(dt):
                        original(dt)
                        counters["executed_physics_steps"] += 1
                        dense.append(np.array(env.agent.position))

                    env.space.step = record
                    sequence = rng.normal(size=(config["actions_per_trajectory"], 2))
                    rho = config["temporal_correlation"]
                    if trajectory % 2:
                        for t in range(1, len(sequence)):
                            sequence[t] = rho * sequence[t - 1] + np.sqrt(1 - rho**2) * sequence[t]
                    sequence *= config["normalized_action_spreads"][trajectory % 3]
                    sequence += rng.uniform(-1, 1, 2)
                    observations.append(obs["proprio"].copy())
                    actions.append(sequence)
                    for action in sequence:
                        counters["environment_actions"] += 1
                        env.step(action * normal_scale + normal_mean)
                    positions.append(np.asarray(dense))
                    groups.append(f"scene-{seed}")
                    splits.append(split)
                    write_json(output / "live.json", {"resources": dict(counters), "split": split})
            print(json.dumps({"finished": split, "resources": dict(counters)}), flush=True)
    finally:
        env.close()
        write_json(output / "resources.json", dict(counters))
    arrays = {
        "observations": np.asarray(observations),
        "actions": np.asarray(actions),
        "positions": np.asarray(positions),
        "groups": np.asarray(groups),
        "splits": np.asarray(splits),
    }
    np.savez_compressed(output / "motions.npz", **arrays)
    return arrays, counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not os.environ.get("RUNPOD_POD_ID") and not os.environ.get("VAST_CONTAINERLABEL"):
        raise RuntimeError("Motion training requires a remote experiment host")
    if config.get("execution_pod_id") and config["execution_pod_id"] != os.environ.get(
        "RUNPOD_POD_ID"
    ):
        raise RuntimeError("Motion training host differs from the frozen host")
    horizon = config.get("max_horizon", 3)
    if config["actions_per_trajectory"] != 5 * horizon:
        raise ValueError("Motion trajectory length must cover the full planning horizon")
    from stable_worldmodel.envs.pusht.env import PushT

    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    scaler = json.loads((ROOT / config["normalizer_source"]).read_text())["columns"]["action"]
    checkpoint = ROOT / config["world_model_checkpoint"]
    freeze = {
        "config": config,
        "config_sha256": file_digest(args.config),
        "source_sha256": {
            name: file_digest(ROOT / name)
            for name in (
                "experiments/scripts/train_pusht_motion.py",
                "experiments/helpers/learned_motion.py",
            )
        },
        "native_environment_source": {
            "path": inspect.getfile(PushT),
            "sha256": file_digest(inspect.getfile(PushT)),
        },
        "world_model_sha256": file_digest(checkpoint),
        "action_scaler": {key: scaler[key] for key in ("mean", "scale")},
        "threads": args.threads,
        "purpose": "One ordinary supervised motion fit; no optimizer claim",
    }
    write_json(output / "freeze.json", freeze)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    arrays, counters = collect(config, scaler, output)
    spec = MotionSpec(
        max_horizon=horizon,
        world_model_sha256=freeze["world_model_sha256"],
        action_scaler_sha256=scaler_digest(scaler["mean"], scaler["scale"]),
    )
    training = arrays["splits"] == "train"
    model, ranks = fit_motion(
        spec,
        arrays["observations"][training],
        arrays["actions"][training],
        arrays["positions"][training],
    )
    metrics = {}
    calibration_errors = None
    for split in ("train", "calibration"):
        selected = arrays["splits"] == split
        metrics[split], errors = motion_errors(
            model,
            arrays["observations"][selected],
            arrays["actions"][selected],
            arrays["positions"][selected],
        )
        if split == "calibration":
            calibration_errors = errors
    selected = arrays["splits"] == "calibration"
    calibrate_motion(
        model, calibration_errors, arrays["groups"][selected], config["calibration_alpha"]
    )
    passed = model.calibration["ready"] and (
        model.calibration["allowance"] <= config["max_calibration_allowance_px"]
    )
    if passed:
        selected = arrays["splits"] == "test"
        metrics["test"], _ = motion_errors(
            model,
            arrays["observations"][selected],
            arrays["actions"][selected],
            arrays["positions"][selected],
        )
        passed = metrics["test"]["max_position_error_px"] <= config["max_position_error_px"]
    report = {
        "metrics": metrics,
        "calibration": model.calibration,
        "passed": bool(passed),
        "resources": dict(counters),
        "design_ranks": ranks,
        "elapsed_seconds": time.perf_counter() - started,
        "new_external_compute_usd": None,
        "compute_billed_with_existing_remote_session": True,
        "dataset_sha256": file_digest(output / "motions.npz"),
    }
    save_motion(output / "model.pt", model, {"freeze": freeze, "report": report})
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)
    if not passed:
        raise RuntimeError("Motion validation criteria failed; no release")


if __name__ == "__main__":
    main()
