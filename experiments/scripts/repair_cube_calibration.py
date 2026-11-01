"""Fresh, predeclared calibration after the failed twelve-group Cube release check."""

import os
import importlib

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
importlib.import_module("triton")
import collections
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from scripts import run_portable_lewm as study
from helpers.portable_env import NativeEnv
from helpers.portable_motion import load_model
from helpers.safety_data import write_json
from helpers.safety_model import file_digest


def main():
    if os.environ.get("RUNPOD_POD_ID") != study.DESIGN["pod_id"]:
        raise RuntimeError("Remote execution only")
    previous = study.OUTPUT / "cube/motion"
    out = study.OUTPUT / "cube-v2"
    dest = out / "motion"
    dest.mkdir(parents=True, exist_ok=False)
    protocol = {
        "reason": "Original independent check covered10/12 groups, below11/12 gate; retain as failed developmental evidence.",
        "model": "Same frozen neural weights, no refitting.",
        "calibration_groups": list(range(104, 152)),
        "test_groups": list(range(152, 200)),
        "trajectories_per_group": 4,
        "actions_per_trajectory": 25,
        "alpha": 0.05,
        "test_required_groups": 44,
        "test_groups_total": 48,
        "maximum_allowance_m": 0.08,
        "max_new_actions": 9600,
        "max_new_resets": 384,
        "unchanged_comparison": "150 actions, checkpoints50/100/150, same fixed random configurations and source-disjoint Cube validation episodes.",
        "previous_model_sha256": file_digest(previous / "model.pt"),
        "sources": {
            str(p.relative_to(ROOT)): file_digest(p) for p in [Path(__file__), *study.SOURCES]
        },
        "epoch": time.time(),
    }
    write_json(dest / "repair-freeze.json", protocol)
    model = load_model(previous / "model.pt")
    model.requires_grad_(False)
    normal = study.scaler("cube")
    dataset = study.cube_data("train")
    c = collections.Counter()
    env = NativeEnv("cube", c, dest / "interactions.jsonl")
    stats = {}
    started = time.perf_counter()
    try:
        for split, groups in (("calibration", range(104, 152)), ("test", range(152, 200))):
            initial, actions, positions, labels = [], [], [], []
            for group in groups:
                for repeat in range(4):
                    seed = 301030 + group * 100 + repeat
                    rng = np.random.default_rng(seed)
                    observed = env.reset(seed, study.training_state("cube", group, rng, dataset))
                    initial.append(observed.copy())
                    acts = []
                    points = []
                    prev = np.zeros(5)
                    for _ in range(25):
                        a = (
                            0.5 * prev
                            + np.sqrt(0.75) * rng.normal(size=5) * (0.5, 1.0, 2.0, 1.0)[repeat]
                        )
                        raw = np.clip(
                            normal.inverse_transform(a),
                            env.env.action_space.low,
                            env.env.action_space.high,
                        )
                        normalized = normal.transform(raw)
                        observed, dense = env.step(raw)
                        acts.append(normalized)
                        points.extend(dense)
                        prev = normalized
                    actions.append(acts)
                    positions.append(points)
                    labels.append(group)
                write_json(
                    dest / "live.json", {"split": split, "group": group, "resources": dict(c)}
                )
            initial = np.asarray(initial)
            actions = np.asarray(actions)
            positions = np.asarray(positions)
            labels = np.asarray(labels)
            with torch.inference_mode():
                prediction = (
                    model(
                        torch.tensor(initial, dtype=torch.float32, device="cuda"),
                        torch.tensor(actions, dtype=torch.float32, device="cuda"),
                    )
                    .cpu()
                    .numpy()
                )
            errors = np.linalg.norm(prediction - positions, axis=-1).max(-1)
            grouped = [float(errors[labels == g].max()) for g in groups]
            stats[split] = {
                "group_errors": grouped,
                "maximum_error": max(grouped),
                "mean_trajectory_max_error": float(errors.mean()),
            }
            np.savez_compressed(
                dest / (split + ".npz"),
                initial=initial,
                actions=actions,
                positions=positions,
                groups=labels,
                predicted=prediction,
            )
        rank = int(np.ceil(49 * 0.95))
        allowance = max(sorted(stats["calibration"]["group_errors"])[rank - 1], 1e-4)
        covered = sum(e <= allowance + 1e-6 for e in stats["test"]["group_errors"])
        calibration = {
            "allowance": allowance,
            "calibration_groups": 48,
            "alpha": 0.05,
            "order_statistic": rank,
            "test_groups": 48,
            "test_groups_covered": covered,
            "ready": covered >= 44 and allowance <= 0.08,
            "interpretation": "Fresh group calibration with unchanged weights; empirical coverage, not a guarantee.",
        }
        data = torch.load(previous / "model.pt", map_location="cpu", weights_only=False)
        data["calibration"] = calibration
        torch.save(data, dest / "model.pt")
        old = json.loads((previous / "report.json").read_text())
        resources = collections.Counter(old["resources"])
        resources.update(c)
        report = {
            **old,
            "calibration": calibration,
            "errors": stats,
            "resources": dict(resources),
            "repair_resources": dict(c),
            "elapsed_seconds": old["elapsed_seconds"] + time.perf_counter() - started,
            "repair_protocol": protocol,
            "old_failed_check": str(previous / "report.json"),
            "model_sha256": file_digest(dest / "model.pt"),
            "unchanged_weights": True,
        }
        write_json(dest / "report.json", report)
        write_json(dest / "resources.json", dict(resources))
        print(json.dumps({"calibration": calibration, "resources": dict(resources)}), flush=True)
    finally:
        env.close()
        write_json(dest / "repair-resources.json", dict(c))
    if not calibration["ready"]:
        raise RuntimeError("Fresh Cube release check failed; no candidate planning allowed")
    for stage in ("prepare", "tune", "evaluate", "report"):
        record = out / (stage + "-timing.json")
        start = time.time()
        write_json(record, {"stage": stage, "status": "running", "epoch": start})
        if stage == "prepare":
            study.prepare("cube", out)
        elif stage in ("tune", "evaluate"):
            study.compare("cube", out, stage)
        else:
            study.report("cube", out)
        write_json(
            record, {"stage": stage, "status": "complete", "elapsed_seconds": time.time() - start}
        )


if __name__ == "__main__":
    main()
