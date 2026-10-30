"""Train, resume, inspect or replay an evolved ManiGuard residual controller."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/helpers"))
from maniguard_adapter import AdapterSpec
from maniguard_evolution import Search
from maniguard_evolution_episode import ManiGuardEvaluator, atomic_json, totals, utc
from maniguard_evolution_scoring import analyze_episode, summarize
from prepare_maniguard_evolution import prepare


SOURCES = (
    "experiments/helpers/maniguard_adapter.py",
    "experiments/helpers/maniguard_adapter_runtime.py",
    "experiments/helpers/maniguard_evolution.py",
    "experiments/helpers/maniguard_evolution_episode.py",
    "experiments/helpers/maniguard_evolution_scoring.py",
    "experiments/helpers/maniguard_evolution_v3.py",
    "experiments/helpers/maniguard_evolution_selection.py",
    "experiments/helpers/maniguard_evolution_diagnostics.py",
    "experiments/helpers/accounted_recorder.py",
    "experiments/scripts/run_maniguard_evolution.py",
    "experiments/scripts/prepare_maniguard_evolution.py",
    "experiments/scripts/build_maniguard_v3_config.py",
    "experiments/scripts/prime_maniguard_v3_policy.py",
    "scripts/todo2/record_rollout.py",
    "progress/panels.json",
)


def validate_config(config, panels):
    spec = AdapterSpec(**config["adapter"])
    search, validation = config["search"], config["validation"]
    if search["generations"] < 1 or search["population"] < 4:
        raise ValueError("Training requires positive generations and population >= 4")
    if not 0 < search["sigma"] <= 1:
        raise ValueError("Invalid initial search sigma")
    if len(search["policy_seeds"]) != search["generations"]:
        raise ValueError("Declare one training policy seed per generation")
    if search["objective"] not in (
            "completion_only", "completion_progress", "completion_progress_contact_v3"):
        raise ValueError("Unknown training objective")
    if search.get("mutation_profile", "isotropic") not in ("isotropic", "conservative_arm_v3"):
        raise ValueError("Unknown mutation profile")
    blocks = [search, validation]
    if "selection" in config:
        blocks.append(config["selection"])
    for block in blocks:
        if not block["tasks"] or not block["policy_seeds"]:
            raise ValueError("Empty development panel")
        if len(set(block["tasks"])) != len(block["tasks"]):
            raise ValueError("Duplicate task in a development panel")
        if not set(block["tasks"]) <= set(panels["development"]):
            raise ValueError("Only the frozen development tasks are eligible")
        if any(not isinstance(s, int) or s <= 0 for s in block["policy_seeds"]):
            raise ValueError("Declare positive integer policy seeds")
    for i, block in enumerate(blocks):
        if len(set(block["policy_seeds"])) != len(block["policy_seeds"]):
            raise ValueError("Duplicate policy seed in a development panel")
        for other in blocks[i + 1:]:
            if set(block["policy_seeds"]) & set(other["policy_seeds"]):
                raise ValueError("Training, selection and validation seeds must be disjoint")
            if "selection" in config and set(block["tasks"]) & set(other["tasks"]):
                raise ValueError("V3 stage task panels must be disjoint")
    if len(validation["policy_seeds"]) != 3:
        raise ValueError("Validation uses three policy seeds")
    evaluator = config["evaluator"]
    required = {"action_frequency": 20, "execute_horizon": 8, "max_steps": 2000,
                "success_hold_steps": 10, "camera_resolution": 256,
                "rendering_frequency": 20, "physics_frequency": 120,
                "source_profile": "legacy", "controller_preset": "joint_position_raw",
                "ik_eef_to_joint": False, "gripper_binarize": True,
                "grasping_mode": "assisted", "longfinger": True}
    if any(evaluator.get(k) != v for k, v in required.items()):
        raise ValueError("Evaluator differs from the pinned paper execution contract")
    limits = config["limits"]
    if config.get("environment", {}).get("OMNI_KIT_ACCEPT_EULA") != "YES":
        raise ValueError("Carry over the existing simulator EULA launch setting")
    if any(limits[k] <= 0 for k in ("max_starts", "max_environment_step_attempts",
                                   "max_model_query_attempts", "episode_timeout_seconds",
                                   "minimum_start_window_seconds", "shutdown_reserve_seconds")):
        raise ValueError("Explicit positive run budgets are required")
    if config["information"]["privileged_simulator_inputs"] is not True:
        raise ValueError("This controller requires declared privileged simulator information")
    if "selection" in config:
        if (search["objective"] != "completion_progress_contact_v3"
                or not search.get("native_anchor") or search.get("initial_controller")):
            raise ValueError("V3 requires native initialization, native anchors and v3 fitness")
        binding = config["protocol_binding"]
        path = ROOT / binding["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]:
            raise ValueError("Runtime configuration is not bound to the frozen v3 protocol")
        protocol = json.loads(path.read_text())
        checks = [
            (search["method"], "cmaes"),
            (search["generations"], protocol["training"]["generations"]),
            (search["population"], protocol["training"]["population"]),
            (search["tasks"], protocol["training"]["tasks"]),
            (search["policy_seeds"], protocol["training"]["policy_seeds"]),
            (search["seed"], protocol["training"]["optimizer_seed"]),
            (search["initialization_seed"], protocol["training"]["initialization_seed"]),
            (search["sigma"], protocol["training"]["sigma"]),
            (search.get("mutation_profile"), protocol["training"]["mutation_profile"]),
            (config["selection"]["tasks"], protocol["selection"]["tasks"]),
            (config["selection"]["policy_seeds"], protocol["selection"]["policy_seeds"]),
            (validation["tasks"], protocol["validation"]["tasks"]),
            (validation["policy_seeds"], protocol["validation"]["policy_seeds"]),
            (limits["deadline_utc"], protocol["rental_binding"]["deadline_utc"]),
            (limits["shutdown_reserve_seconds"], protocol["limits"]["preservation_reserve_seconds"]),
            (limits["max_starts"], protocol["limits"]["max_episode_starts_including_failures"]),
            (limits["max_environment_step_attempts"],
             protocol["limits"]["max_instrumented_environment_step_attempts"]),
            (limits["max_model_query_attempts"], protocol["limits"]["max_total_model_query_attempts"]),
            (limits.get("minimum_remaining_seconds_before_first_episode"),
             protocol["limits"]["minimum_remaining_seconds_before_first_scored_episode"]),
        ]
        if any(left != right for left, right in checks):
            raise ValueError("Executable configuration changes the prospectively written protocol")
        if config["rental"]["instance_id"] != protocol["rental_binding"]["instance_id"]:
            raise ValueError("Runtime rental identity differs from the protocol")
        if config["adapter"] != {k: protocol["controller"][k] for k in config["adapter"]}:
            raise ValueError("Runtime controller specification differs from the protocol")
    return spec.parameter_count


def source_manifest(config):
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    warmstart = config["search"].get("initial_controller")
    if warmstart:
        hashes["initial_controller"] = hashlib.sha256(Path(warmstart).read_bytes()).hexdigest()
    if config.get("protocol_binding"):
        name = config["protocol_binding"]["path"]
        hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return {"sources": hashes, "configuration": config}


def on_rented_host():
    if os.environ.get("CONTAINER_ID") or os.environ.get("RUNPOD_POD_ID"):
        return True
    # Vast SSH shells may omit the container identity that the independent guard inherited.
    for ready in Path("/workspace").glob("setup-maniguard*/self-stop-ready.json"):
        try:
            identity = json.loads(ready.read_text())
            pid = int((ready.parent / "self-stop.pid").read_text())
            proc = Path("/proc") / str(pid)
            if ((proc / "stat").read_text().split()[2] != "Z"
                    and str(ready.parent / "self-stop.py").encode()
                    in (proc / "cmdline").read_bytes()
                    and int(identity["instance_id"]) > 0):
                return True
        except (OSError, ValueError, KeyError):
            continue
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--controller", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--resume", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("out", type=Path)
    replay = sub.add_parser("replay")
    replay.add_argument("job", type=Path)
    args = parser.parse_args()
    if args.command == "status":
        if not (args.out / "status.json").exists():
            print((args.out / "evaluation.json").read_text())
            return
        state = json.loads((args.out / "status.json").read_text())
        print(json.dumps({"status": state["status"], "generation": state["generation"],
                          "active": state.get("active"), "error": state.get("error"),
                          "validation": state["validation"], "accounting": totals(args.out)},
                         indent=2))
        return
    if not on_rented_host():
        raise SystemExit("Run experiment computation on the rented Vast or RunPod host")
    if args.command == "replay":
        print(json.dumps(analyze_episode(args.job), indent=2))
        return
    config = json.loads(args.config.read_text())
    artifact = None
    if args.command == "evaluate":
        artifact = json.loads(args.controller.read_text())
        if artifact["spec"] != config["adapter"]:
            raise ValueError("Evaluation controller specification differs from the config")
        AdapterSpec(**artifact["spec"]).unpack(artifact["parameters"])
        config["evaluation_controller_sha256"] = hashlib.sha256(
            args.controller.read_bytes()).hexdigest()
    panels = json.loads((ROOT / "progress/panels.json").read_text())
    validate_config(config, panels)
    out = args.out.resolve()
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
    lock = (out / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another trainer owns this run directory") from None
    manifest = source_manifest(config)
    if args.resume:
        if json.loads((out / "source-manifest.json").read_text()) != manifest:
            raise ValueError("Run inputs changed; use a fresh output path for revised code")
    else:
        atomic_json(out / "source-manifest.json", manifest)
        atomic_json(out / "freeze-receipt.json", {
            "frozen_utc": utc(),
            "source_manifest_sha256": hashlib.sha256(
                (out / "source-manifest.json").read_bytes()).hexdigest(),
            "protocol_binding": config.get("protocol_binding"),
            "runtime_configuration_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        })
        snapshot = out / "source"
        for name in manifest["sources"]:
            if name == "initial_controller":
                continue
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
        prepare(config["maniguard_checkout"], out / "generated")
        import importlib.metadata

        gpu = subprocess.run([
            "nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader",
        ], capture_output=True, text=True, check=True)
        atomic_json(out / "host.json", {
            "recorded_utc": utc(), "hostname": platform.node(), "gpu": gpu.stdout.strip(),
            "python": sys.version,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("numpy", "scipy", "cma", "torch", "omnigibson")},
        })
    backend = ManiGuardEvaluator(out, config, out / "generated", ROOT)

    def terminate(_signum, _frame):
        raise KeyboardInterrupt("Trainer received termination signal")

    signal.signal(signal.SIGTERM, terminate)
    if args.command == "evaluate":
        atomic_json(out / "controller.json", artifact)
        records = {"native": [], "selected": []}
        validation = config["validation"]
        cases = [(task, seed) for seed in validation["policy_seeds"]
                 for task in validation["tasks"]]
        for i, (task, seed) in enumerate(cases):
            arms = ("native", "selected") if i % 2 == 0 else ("selected", "native")
            for arm in arms:
                controller = {**artifact, "enabled": arm != "native"}
                records[arm].append(backend("evaluation_" + arm, task, seed, controller))
                atomic_json(out / "evaluation.json", {
                    "updated_utc": utc(), "outcomes": records,
                    "summary": {name: summarize(rows) for name, rows in records.items()}})
        print(json.dumps({name: summarize(rows) for name, rows in records.items()}))
        return
    search = Search(config, out, backend, resume=args.resume)
    result = search.run()
    atomic_json(out / "accounting.json", totals(out))
    print(json.dumps({"status": result["status"], "generation": result["generation"],
                      "validation": result["validation"]}), flush=True)


if __name__ == "__main__":
    main()
