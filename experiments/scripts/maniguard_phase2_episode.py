"""Execute one pinned phase-two episode; labels never enter actor observations."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "third_party/ManiGuard"))

from helpers.maniguard_phase2_families import (
    family_spec,
    load_registry,
    load_sources,
    validate_registry,
    validate_scene_access,
)
from helpers.maniguard_phase2_model import genome_hash, identity_genome, load_model
from helpers.maniguard_phase2_protocol import IMPLEMENTATION_FILES
from helpers.maniguard_phase2_source import prepare_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(load_registry()), required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--purpose", choices=("engineering", "train", "validation", "test",
                                              "transfer"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("learned", "fixed", "contextual"), default="learned")
    parser.add_argument("--fixed-choices", type=int, nargs=4)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--frozen-protocol", type=Path)
    parser.add_argument("--encoder-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--baseline-executor", action="store_true",
                        help="Engineering equivalence check with the original eight-action executor")
    args = parser.parse_args()
    validate_registry()
    genome, _, _ = load_model(args.model)
    actor_sha = genome_hash(genome)
    if args.baseline_executor and (
        args.purpose != "engineering" or args.stochastic or args.mode != "learned"
        or actor_sha != genome_hash(identity_genome())
    ):
        raise ValueError("Original executor checks require engineering identity episodes")
    frozen = False
    if args.frozen_protocol:
        protocol = json.loads(args.frozen_protocol.read_text())
        frozen = protocol.get("evaluation_frozen") is True
        if not frozen or actor_sha not in protocol.get("actor_sha256", []):
            raise ValueError("Reserved evaluation requires a frozen protocol matching the actor")
    validate_scene_access(args.family, args.scene, args.purpose, frozen_evaluation=frozen)
    if args.stochastic and args.purpose not in ("train", "engineering"):
        raise ValueError("Evaluation must use deterministic controller decoding")
    if args.max_steps is not None and (
        args.purpose != "engineering" or not 1 <= args.max_steps <= 32
    ):
        raise ValueError("Only preparation checks may shorten the horizon, to at most32 actions")
    if min(args.seed, args.sampling_seed) < 0:
        raise ValueError("Seeds must be nonnegative")
    spec = family_spec(args.family)
    scene_root = ROOT / "data/maniguard/benchmark" / spec["benchmark_dir"]
    scene = (scene_root / args.scene).resolve()
    scene.relative_to(scene_root.resolve())
    if not (scene / "scene_ep1.json").is_file():
        raise ValueError("The exact pinned scene has not been prepared")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = output / "actor.npz"
    shutil.copyfile(args.model, model)
    sources = load_sources()
    job = {
        "phase": "phase2", "family": args.family, "scene": args.scene, "purpose": args.purpose,
        "seed": args.seed, "sampling_seed": args.sampling_seed, "stochastic": args.stochastic,
        "mode": args.mode, "fixed_choices": args.fixed_choices, "model": str(model),
        "actor_sha256": actor_sha, "adapter": {"kind": "phase2", "mode": args.mode},
        "output": str(output), "episode_id": str(output), "random_policy": False,
        "max_steps_override": args.max_steps,
        "action_horizon": args.max_steps or spec["max_steps"],
        "policy": sources["policies"][args.family], "encoder_device": args.encoder_device,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "code_hashes": {},
    }
    if args.baseline_executor:
        from dataclasses import asdict

        from helpers.maniguard_adaptation import AdapterConfig
        job["phase"] = "phase2_baseline_check"
        job["adapter"] = asdict(AdapterConfig(kind="identity", horizon=8))
    paths = list((ROOT / "experiments/helpers").glob("maniguard_phase2_*.py"))
    paths += [Path(__file__), ROOT / "experiments/helpers/maniguard_runtime.py",
              ROOT / "experiments/helpers/maniguard_source.py",
              ROOT / "experiments/helpers/maniguard_adaptation.py",
              ROOT / "experiments/helpers/maniguard_client.py"]
    paths += [ROOT / relative for relative in IMPLEMENTATION_FILES]
    paths += list((ROOT / "configs/maniguard/phase2").glob("*.json"))
    for path in sorted(set(paths)):
        relative = path.relative_to(ROOT)
        raw = path.read_bytes()
        destination = output / "study_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        job["code_hashes"][str(relative)] = hashlib.sha256(raw).hexdigest()
    job_path = output / "job.json"
    job_path.write_text(json.dumps(job, indent=2) + "\n")
    os.environ["SAFETY_DIAL_MANIGUARD_JOB"] = str(job_path)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ["OMNIGIBSON_DATA_PATH"] = str(ROOT / "data/maniguard/datasets")
    if args.baseline_executor:
        from helpers.maniguard_source import prepare_runtime as prepare_original_runtime
        runtime = prepare_original_runtime(output)
    else:
        runtime = prepare_runtime(output)
    sys.argv = [
        str(runtime), "--config", str(ROOT / "third_party/ManiGuard" / spec["eval_config"]),
        "--benchmark-root", str(scene_root), "--scenes", args.scene, "--seed", str(args.seed),
        "--host", "127.0.0.1", "--port", str(args.port), "--source-profile", "legacy",
        "--output-dir", str(output), "--run-name", "episode", "--execute-horizon",
        "8" if args.baseline_executor else "16",
    ]
    if args.max_steps is not None:
        sys.argv += ["--max-steps", str(args.max_steps)]
    runpy.run_path(str(runtime), run_name="__main__")


if __name__ == "__main__":
    main()
