"""Export frozen-imagination clearance examples from explicit, hashed recorded traces."""

import argparse
import collections
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.safety_data import prepare_examples, save_examples, source_partitions, write_json
from helpers.safety_model import SafetySpec, file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    spec = SafetySpec(**manifest["spec"])
    checkpoint = Path(manifest["world_model_checkpoint"])
    if not checkpoint.name.endswith("_object.ckpt"):
        raise ValueError("Supply the exact pinned local LeWM _object.ckpt file")
    if file_digest(checkpoint) != spec.world_model_sha256:
        raise ValueError("Frozen world-model checkpoint hash differs from the manifest")
    source_partitions(manifest["traces"])
    args.out.mkdir(parents=True, exist_ok=False)
    freeze = {
        "manifest": manifest,
        "manifest_sha256": file_digest(args.manifest),
        "source_sha256": {
            name: file_digest(ROOT / name)
            for name in (
                "experiments/scripts/prepare_safety_data.py",
                "experiments/helpers/safety_data.py",
                "experiments/helpers/safety_model.py",
                "experiments/helpers/safecem10_planner.py",
            )
        },
        "device": args.device,
        "threads": args.threads,
        "simulator_interactions": 0,
        "purpose": "Offline engineering export, no efficacy claim",
    }
    write_json(args.out / "freeze.json", freeze)
    torch.set_num_threads(args.threads)
    from stable_worldmodel.policy import AutoCostModel

    started = time.perf_counter()
    model = AutoCostModel(str(checkpoint)[: -len("_object.ckpt")], cache_dir=checkpoint.parent).to(
        args.device
    )
    counters = collections.Counter()
    scaler = manifest["action_scaler"]
    arrays = prepare_examples(
        model,
        spec,
        manifest["traces"],
        scaler["mean"],
        scaler["scale"],
        manifest.get("trace_root", str(ROOT)),
        counters,
    )
    provenance = {
        **freeze,
        "resources": dict(counters),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_examples(args.out / "examples.npz", arrays, spec, provenance)
    print(
        json.dumps(
            {
                "examples": len(arrays["targets"]),
                "source_groups": len(set(arrays["groups"])),
                "resources": dict(counters),
                "simulator_interactions": 0,
                "elapsed_seconds": provenance["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
