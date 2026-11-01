"""One declared rented-host integration episode, or saved-evidence audit.

The adapter is seeded and untrained. No real fitness evaluations or ES updates are
authorized by this smoke runner. Software tests exercise ES with synthetic fitness.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.lewm_proposal_adapter import ProposalAdapter, ProposalSpec


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".part")
    staging.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    staging.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def controller(protocol):
    spec = ProposalSpec(**protocol["adapter"]["spec"])
    adapter = ProposalAdapter(
        spec, spec.initial_parameters(protocol["adapter"]["seed"]), protocol["case"]
    )
    path = ROOT / protocol["adapter"]["checkpoint"]
    artifact = {
        **adapter.artifact(),
        "initialization_seed": protocol["adapter"]["seed"],
        "training_status": "Untrained seeded engineering controller; zero optimizer updates",
    }
    if path.exists():
        if json.loads(path.read_text()) != artifact:
            raise ValueError("Existing controller differs from the declared initialization")
    else:
        write_json(path, artifact)
    return adapter, path


def run(args, protocol):
    from helpers.lewm18_planner import PlanningCEM
    from scripts import run_safety_pusht as runtime

    if args.out.exists():
        raise FileExistsError("Retain prior attempts and use a new output path")
    if protocol["purpose"] != "Single-episode proposal-adapter engineering integration":
        raise ValueError("Only the explicitly frozen smoke protocol is supported")
    adapter, path = controller(protocol)
    sources = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "experiments").rglob("*.py")}
    freeze = {
        "protocol_sha256": digest(args.protocol),
        "sources": sources,
        "controller_sha256": digest(path),
        "spec": asdict(adapter.spec),
        "parameter_count": adapter.spec.parameter_count,
        "pod_id": os.environ["RUNPOD_POD_ID"],
        "new_optimizer_updates": 0,
        "no_local_model_execution": True,
    }
    freeze_path = args.out.parent / (args.out.name + "-source-freeze.json")
    if freeze_path.exists():
        raise FileExistsError("Never overwrite the freeze of an earlier attempt")
    write_json(freeze_path, freeze)

    def make_solver(model, env, params, seed, device):
        return PlanningCEM(model, env, params, seed, device, proposal_adapter=adapter)

    argv = [
        str(Path(runtime.__file__)),
        "--protocol",
        str(args.protocol),
        "--out",
        str(args.out),
        "--device",
        args.device,
        "--threads",
        str(args.threads),
    ]
    try:
        with patch.object(runtime, "PlanningCEM", make_solver), patch.object(sys, "argv", argv):
            runtime.main()
    finally:
        if args.out.exists():
            write_json(args.out / "adapter-proposals.json", adapter.history)
            write_json(
                args.out / "adapter-identity.json",
                {
                    "checkpoint": str(path.relative_to(ROOT)),
                    "sha256": digest(path),
                    "parameter_count": adapter.spec.parameter_count,
                    "feature_names": adapter.artifact()["feature_names"],
                    "untrained": True,
                    "optimizer_updates": 0,
                },
            )


def audit(args, protocol):
    import numpy as np

    result = json.loads((args.out / "result.json").read_text())
    history = json.loads((args.out / "adapter-proposals.json").read_text())
    plans = json.loads((args.out / "episode/replans.json").read_text())
    freeze = json.loads((args.out.parent / (args.out.name + "-source-freeze.json")).read_text())
    if freeze["protocol_sha256"] != digest(args.protocol):
        raise ValueError("Protocol changed after launch")
    for name, expected in freeze["sources"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Executed source changed: " + name)
    adapter = ProposalAdapter.load(ROOT / protocol["adapter"]["checkpoint"], protocol["case"])
    if digest(ROOT / protocol["adapter"]["checkpoint"]) != freeze["controller_sha256"]:
        raise ValueError("Controller changed")
    if not history:
        raise ValueError("Adapter never participated in planning")
    for row in history:
        wanted = adapter.forward(row["features"], len(row["proposal"]))
        if not np.array_equal(wanted, np.asarray(row["proposal"])):
            raise ValueError("Saved proposal cannot be replayed from frozen weights and features")
    executed = [p for p in plans if p.get("executed") != "none"]
    for plan in executed:
        if len(plan["adapter_checks"]) != protocol["planner"]["iterations"]:
            raise ValueError("Adapter candidate was not evaluated in every declared iteration")
        if plan["executed_violation"] > 0:
            raise ValueError("Unchecked plan reached execution")
        saved = next(r for r in history if r["replan"] == plan["features"]["replan"])
        if not np.allclose(saved["proposal"], plan["adapter_proposal"], atol=1e-6, rtol=0):
            raise ValueError("Planner proposal differs from adapter output")
    with np.load(args.out / "episode/trace.npz", allow_pickle=False) as data:
        states, actions, dense = data["states"], data["actions"], data["dense"]
    shape, goal = protocol["case"], np.asarray(result["goal_state"])
    if shape["kind"] != "island":
        raise ValueError("This independent smoke audit is frozen to the known island")
    hazard = np.linalg.norm(dense[:, :2] - shape["origin"], axis=-1) - shape["radius_px"] - 15
    arena = np.minimum(dense[:, :2], 512 - dense[:, :2]).min(-1) - 15
    violation = bool((hazard < 6 - 1e-6).any() or (arena < 6 - 1e-6).any())
    angular = abs(goal[4] - states[-1, 4]) % (2 * np.pi)
    angular = min(angular, 2 * np.pi - angular)
    success = bool(np.linalg.norm(goal[:4] - states[-1, :4]) < 20 and angular < np.pi / 9)
    safe = success and not violation and not result["error"] and not result["controller_failure"]
    counts = result["resources"]
    if success != result["success"] or bool(safe) != result["safe_completion"]:
        raise ValueError("Independent native-goal outcome mismatch")
    if violation != bool(result["margin_breach"] or result["arena_margin_breach"]):
        raise ValueError("Independent dense safety outcome mismatch")
    if len(actions) > protocol["max_episode_actions"] or len(dense) != 10 * len(actions) + 1:
        raise ValueError("Action or dense-physics accounting mismatch")
    if counts["executed_physics_steps"] != 10 * len(actions):
        raise ValueError("Physics counter mismatch")
    if counts["adapter_forward_calls"] != len(history):
        raise ValueError("Adapter call accounting mismatch")
    if counts["predictor_steps"] > protocol["maximum_predictor_steps"]:
        raise ValueError("Frozen world-model query ceiling exceeded")
    value = {
        "integration_verified": True,
        "episodes": 1,
        "safe_completion": int(bool(safe)),
        "violation": int(violation),
        "unfinished": int(not success),
        "engaged": int(result["pusher_path_length_px"] > 1 and len(actions) > 0),
        "actions": len(actions),
        "replans": len(history),
        "minimum_hazard_clearance_px": float(hazard.min()),
        "minimum_arena_clearance_px": float(arena.min()),
        "adapter_direct_selections": sum(p["adapter_selected_directly"] for p in executed),
        "proposal_replay_verified": True,
        "resources": counts,
        "optimizer_updates": 0,
        "controller_untrained": True,
        "limits": "One known engineering case and seeded untrained adapter; no efficacy or evolutionary advantage claim",
    }
    write_json(args.out / "audit.json", value)
    print(json.dumps(value, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "audit"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if os.environ.get("RUNPOD_POD_ID") != protocol["execution_pod_id"]:
        raise RuntimeError("Only the declared fresh RunPod may execute this entry point")
    if args.device != "cuda":
        raise ValueError("The frozen smoke uses GPU world-model inference")
    (run if args.mode == "run" else audit)(args, protocol)


if __name__ == "__main__":
    main()
