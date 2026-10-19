"""Execute one new physical episode with native horizons and a frozen controller."""

import argparse
import calendar
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "third_party/ManiGuard"))

import numpy as np

from helpers.maniguard_adaptation import AdapterConfig
from helpers.maniguard_latent import LatentController
from helpers.maniguard_latent_source import prepare_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["jar", "clutter"], required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--engineering-max-steps", type=int)
    parser.add_argument("--camera-smoke", action="store_true")
    args = parser.parse_args()
    family = {"jar": "jar_transport", "clutter": "clutter_pickup"}[args.family]
    scene_root = ROOT / "data/maniguard/benchmark" / family
    scene_path = (scene_root / args.scene).resolve()
    scene_path.relative_to(scene_root.resolve())
    if not (scene_path / "scene_ep1.json").is_file():
        raise ValueError("Select one exact pinned benchmark scene")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with np.load(args.controller, allow_pickle=False) as data:
        controller = LatentController(data["weights"])
    if args.camera_smoke and (args.engineering_max_steps is None or not controller.identity):
        raise ValueError("Camera-only random smoke needs an identity controller and short horizon")
    np.savez(output / "controller.npz", weights=controller.weights)
    sources = json.loads((ROOT / "configs/maniguard/phase2/sources.json").read_text())
    protocol = json.loads(args.protocol.read_text())
    if time.time() >= calendar.timegm(time.strptime(
            protocol["physical_cutoff_utc"], "%Y-%m-%dT%H:%M:%SZ")):
        raise RuntimeError("Physical acquisition cutoff reached; preserve and hand off")
    if protocol["instance_id"] != int(os.environ.get("CONTAINER_ID", "0")):
        raise ValueError("Protocol is not authorized on this instance")
    horizon = {"jar": 2000, "clutter": 1100}[args.family]
    if args.engineering_max_steps is not None:
        if not 1 <= args.engineering_max_steps <= 32:
            raise ValueError("Short engineering attempts have a 32-action ceiling")
        horizon = args.engineering_max_steps
    job = {"study": "latent-evolution-v1", "family": args.family, "scene": args.scene,
           "seed": args.seed, "output": str(output), "episode_id": str(output),
           "controller": str(output / "controller.npz"),
           "controller_sha256": controller.sha256(), "encoder_device": "cuda:0",
           "action_horizon": horizon, "max_steps_override": args.engineering_max_steps,
           "adapter": asdict(AdapterConfig(kind="identity", horizon=8)),
           "random_policy": args.camera_smoke, "camera_smoke": args.camera_smoke,
           "policy": sources["policies"][args.family],
           "source_profile": "high_fidelity",
           "renderer_vulkan_index": 0,
           "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
           "repository_commit": subprocess.check_output(
               ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}
    (output / "protocol.json").write_bytes(args.protocol.read_bytes())
    snapshot = output / "study_source"
    files = list((ROOT / "experiments/helpers").glob("maniguard_latent*.py"))
    files += [Path(__file__), ROOT / "experiments/helpers/maniguard_runtime.py",
              ROOT / "experiments/helpers/maniguard_source.py",
              ROOT / "experiments/helpers/maniguard_adaptation.py",
              ROOT / "experiments/helpers/maniguard_phase2_features.py",
              ROOT / "experiments/helpers/maniguard_phase2_outcomes.py",
              ROOT / "configs/maniguard/phase2/encoder.json",
              ROOT / "configs/maniguard/phase2/completion.json",
              ROOT / "configs/maniguard/phase2/sources.json"]
    files += [ROOT / "configs/maniguard/latent-evolution/platform-v1.json",
              ROOT / "configs/maniguard/latent-evolution/platform-v2.json",
              ROOT / "configs/maniguard/latent-evolution/nvidia-egl-vendor.json"]
    job["code_hashes"] = {}
    for source in files:
        relative = source.relative_to(ROOT)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        job["code_hashes"][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    path = output / "job.json"
    path.write_text(json.dumps(job, indent=2) + "\n")
    os.environ["SAFETY_DIAL_MANIGUARD_JOB"] = str(path)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ["OMNIGIBSON_DATA_PATH"] = str(ROOT / "data/maniguard/datasets")
    runtime = prepare_runtime(output)
    sys.argv = [str(runtime), "--config", str(ROOT / "third_party/ManiGuard/configs/eval"
                                           / (family + "_joint.yaml")),
                "--benchmark-root", str(scene_root), "--scenes", args.scene,
                "--seed", str(args.seed), "--host", "127.0.0.1", "--port", str(args.port),
                "--source-profile", "high_fidelity", "--camera-resolution", "640",
                "--execute-horizon", "8", "--output-dir", str(output), "--run-name", "episode"]
    if args.engineering_max_steps is not None:
        sys.argv += ["--max-steps", str(horizon)]
    if args.camera_smoke:
        sys.argv += ["--random-policy"]
    runpy.run_path(str(runtime), run_name="__main__")


if __name__ == "__main__":
    main()
