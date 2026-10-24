"""Supplement the frozen raw-error analysis with a post-hoc joint-limit audit.

The first completed native trace showed an exact joint-limit plateau. This audit
was added after that observation. It changes no rollout, outcome, raw trace or
planned lag metric. Clipped targets are inferred from pinned controller/asset
source, not presented as an additional runtime measurement.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.tracking import aligned_arm, events, load_recording


def summarize(target, effective, actual, mask):
    raw_error = np.sqrt(np.mean((target[mask] - actual[mask]) ** 2, axis=1))
    effective_error = np.sqrt(np.mean((effective[mask] - actual[mask]) ** 2, axis=1))
    outside = np.any(np.abs(target[mask] - effective[mask]) > 1e-5, axis=1)
    if not mask.any():
        return {"steps": 0}
    return {
        "steps": int(mask.sum()),
        "outside_asset_limits_steps": int(outside.sum()),
        "outside_asset_limits_fraction": float(np.mean(outside)),
        "raw_post_action_rms_error_mean_rad": float(np.mean(raw_error)),
        "inferred_effective_post_action_rms_error_mean_rad": float(np.mean(effective_error)),
        "inferred_effective_post_action_rms_error_p99_rad": float(
            np.quantile(effective_error, .99)),
        "inferred_effective_post_action_rms_error_max_rad": float(np.max(effective_error)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/tracking04/remote")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/tracking04/analysis")
    args = parser.parse_args()
    source = args.root / "source/control_audit"
    evidence = json.loads((source / "joint-limits.json").read_text())
    asset = source / "franka_panda_longfinger.usda"
    if hashlib.sha256(asset.read_bytes()).hexdigest() != evidence["sha256"]:
        raise ValueError("Robot asset differs from its recorded source hash")
    limits = {x["joint"]: x for x in evidence["joint_limits"]}
    episodes = []
    for path in sorted(args.root.glob("episodes/*/validation.json")):
        data, meta = load_recording(next(path.parent.glob("recordings/*.npz")))
        names = [meta["joint_names"][i] for i in meta["arm_control_idx"]]
        low = np.array([limits[name]["lower_rad"] for name in names])
        high = np.array([limits[name]["upper_rad"] for name in names])
        target, actual, steps = aligned_arm(data, meta, 1)
        effective = np.clip(target, low, high)
        event = events(data, meta)
        masks = {"all": np.ones(len(steps), dtype=bool),
                 "last_200_steps": steps >= max(0, len(steps) - 200)}
        if event["contact"] is not None:
            masks["pre_contact"] = steps + 1 < event["contact"]
        if event["violation"] is not None:
            masks["before_first_violation"] = steps + 1 < event["violation"]
            masks["two_seconds_before_violation"] = (
                (steps >= max(0, event["violation"] - 40)) & (steps < event["violation"]))
        episodes.append({
            "episode": path.parent.name,
            "job": json.loads((path.parent / "job.json").read_text()),
            "joint_names": names,
            "windows": {name: summarize(target, effective, actual, mask)
                        for name, mask in masks.items()},
            "outside_limits_steps_per_joint": np.sum(
                np.abs(target - effective) > 1e-5, axis=0).tolist(),
        })
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "Post-hoc explanatory audit, added after the first episode",
        "target_definition": "Environment command clipped to pinned USD joint limits; inferred",
        "outside_limit_tolerance_rad": 1e-5,
        "source": evidence,
        "episodes": episodes,
    }
    (args.out / "joint-limit-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"audited_episodes": len(episodes), "source_sha256": evidence["sha256"]}))


if __name__ == "__main__":
    main()
