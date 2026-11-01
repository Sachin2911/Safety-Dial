"""One explicitly configured PushT episode with shadow or learned safety checking.

No rental management, historical protocol defaults, or automatic checkpoint download.
The protocol file must predeclare limits, case and source checkpoint identities.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.pusht_smoke import Scaler, ShapeChecker
from helpers.learned_motion import LearnedMotionChecker, load_motion
from helpers.lewm18_planner import PlanningCEM
from helpers.safety_checker import LearnedSafetyChecker, SafetyPlanningCEM
from helpers.safety_data import write_json
from helpers.safety_model import file_digest, load_model, pusht_constraints, scaler_digest
from scripts.run_pusht_smoke import run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    mode = protocol["safety_mode"]
    if mode not in ("shadow", "learned") or protocol["max_episode_actions"] > 150:
        raise ValueError("Prototype supports one explicitly bounded episode, at most 150 actions")
    if protocol["max_episode_actions"] < 1 or not 0 < protocol["max_runtime_minutes"] <= 60:
        raise ValueError("Declare positive action/runtime limits")
    if protocol["planner"]["objective"] != "learned":
        raise ValueError("Task progress must use the frozen LeWM goal cost")
    checkpoint = (ROOT / protocol["world_model_checkpoint"]).resolve()
    safety_path = (ROOT / protocol["safety_checkpoint"]).resolve()
    if file_digest(safety_path) != protocol["safety_checkpoint_sha256"]:
        raise ValueError("Safety checkpoint hash differs from frozen protocol")
    backend = protocol.get("safety_backend", "latent_clearance")
    if backend not in ("latent_clearance", "observed_motion"):
        raise ValueError("Unknown learned safety backend")
    loader = load_motion if backend == "observed_motion" else load_model
    head, _ = loader(safety_path, device=args.device)
    if backend == "observed_motion" and mode != "learned":
        raise ValueError("Observed-motion validation uses learned control with a passive monitor")
    if backend == "observed_motion" and protocol.get("verify_analytic_dynamics", True):
        raise ValueError("Observed-motion runs disable the analytic dynamics audit")
    spec = head.spec
    identity = (
        spec.environment,
        spec.units,
        spec.body_radius,
        spec.action_block,
        spec.action_dim,
        spec.protected_body,
        spec.coordinate_frame,
    )
    if identity != ("pusht", "px", 15, 5, 2, "pusher_disk", "arena_xy"):
        raise ValueError("Checkpoint does not describe full-pusher PushT safety")
    if mode == "learned" and (not head.calibration or not head.calibration["ready"]):
        raise ValueError("Checkpoint has no usable independent calibration; use shadow mode")
    if not 1 <= protocol["planner"]["horizon"] <= min(5, spec.max_horizon):
        raise ValueError("Planner horizon is outside the safety checkpoint's supported range")
    if file_digest(checkpoint) != spec.world_model_sha256:
        raise ValueError("World-model checkpoint differs from the safety training model")
    inputs = (ROOT / protocol["inputs"]).resolve()
    columns = json.loads((inputs / "scalers.json").read_text())["columns"]
    action = columns["action"]
    if scaler_digest(action["mean"], action["scale"]) != spec.action_scaler_sha256:
        raise ValueError("Action scaler differs from the safety training scaler")
    if file_digest(inputs / "pair.npz") != protocol["pair_sha256"]:
        raise ValueError("Source pair differs from frozen protocol")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(
        args.out / "freeze.json",
        {
            "protocol": protocol,
            "protocol_sha256": file_digest(args.protocol),
            "device": args.device,
            "threads": args.threads,
            "sources": {
                name: file_digest(ROOT / name)
                for name in (
                    "experiments/scripts/run_safety_pusht.py",
                    "experiments/scripts/run_pusht_smoke.py",
                    "experiments/helpers/safety_checker.py",
                    "experiments/helpers/safety_model.py",
                    "experiments/helpers/learned_motion.py",
                    "experiments/helpers/lewm16_planner.py",
                    "experiments/helpers/lewm18_planner.py",
                    "experiments/helpers/cem_refinements.py",
                    "experiments/helpers/pusht_smoke.py",
                    "experiments/helpers/safecem10_planner.py",
                )
            },
        },
    )
    torch.set_num_threads(args.threads)
    from stable_worldmodel.policy import AutoCostModel

    model = AutoCostModel(str(checkpoint)[: -len("_object.ckpt")], cache_dir=checkpoint.parent).to(
        args.device
    )
    model.eval().requires_grad_(False)
    process = {key: Scaler(value["mean"], value["scale"]) for key, value in columns.items()}
    with np.load(inputs / "pair.npz", allow_pickle=False) as data:
        pair = dict(data)

    def factory(case, margin, scaler, counters, enabled):
        if not enabled:
            raise ValueError("A learned-checker run must enable safety checking")
        if backend == "observed_motion":
            return LearnedMotionChecker(head, pusht_constraints(case), margin, counters)
        reference = ShapeChecker(case, margin, scaler, counters) if mode == "shadow" else None
        return LearnedSafetyChecker(
            head,
            model,
            pusht_constraints(case),
            margin,
            counters,
            mode=mode,
            reference=reference,
        )

    def forbidden_reference(*args, **kwargs):
        raise RuntimeError("Reference motion prediction is forbidden in learned-controlled runs")

    with ExitStack() as guard:
        if backend == "observed_motion":
            for target in (
                "scripts.run_pusht_smoke.pd_response",
                "helpers.safecem10_planner.pd_response",
                "helpers.pusht_smoke.ShapeChecker.check",
                "helpers.pusht_smoke.ShapeChecker.paths",
            ):
                guard.enter_context(patch(target, side_effect=forbidden_reference))
        result, _ = run_episode(
            "safety-model",
            protocol["case"],
            model,
            process,
            pair,
            protocol,
            args.out / "episode",
            time.time(),
            checker_factory=factory,
            planner_type=PlanningCEM if backend == "observed_motion" else SafetyPlanningCEM,
            device=args.device,
        )
    result["reference_predictors_forbidden"] = backend == "observed_motion"
    result["safety_backend"] = backend
    result["safety_observation"] = (
        "Native pusher position and velocity"
        if backend == "observed_motion"
        else "Frozen observation and imagined features"
    )
    write_json(args.out / "result.json", result)
    if result["error"]:
        raise RuntimeError("Episode failed; preserved result.json contains the failure")


if __name__ == "__main__":
    main()
