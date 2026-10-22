"""Serve a verified frozen family checkpoint with bounded native-noise steering."""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "third_party/ManiGuard"))
os.environ.setdefault("MANIGUARD_SKIP_OMNIGIBSON_PATCH", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from helpers.maniguard_latent_policy import NativeNoisePolicy
from helpers.maniguard_policy_provenance import load_policy_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["jar", "clutter"], required=True)
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    sources = json.loads((ROOT / "configs/maniguard/phase2/sources.json").read_text())
    for name, folder in [("maniguard", "ManiGuard"), ("openpi", "openpi")]:
        actual = subprocess.check_output(
            ["git", "-C", str(ROOT / "third_party" / folder), "rev-parse", "HEAD"], text=True,
        ).strip()
        if actual != sources[name]["revision"]:
            raise ValueError(f"Unpinned {name} source")
    spec = sources["policies"][args.family]
    checkpoint, provenance = load_policy_provenance(
        ROOT / "data/maniguard/policies" / args.family, spec, family=args.family)
    from maniguard.openpi_sft import train_configs
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config

    train_configs.register()
    policy = policy_config.create_trained_policy(config.get_config(spec["config"]), checkpoint)
    wrapper = NativeNoisePolicy(policy, args.ledger, audit=args.audit)
    device = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device not in ["0", "1"]:
        raise ValueError("Policy serving requires one explicit authorized physical GPU")
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", device, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    files = [Path(__file__), ROOT / "experiments/helpers/maniguard_latent.py",
             ROOT / "experiments/helpers/maniguard_latent_policy.py",
             ROOT / "third_party/openpi/src/openpi/policies/policy.py",
             ROOT / "third_party/openpi/src/openpi/models/pi0.py"]
    metadata = {"serve_config": spec["config"], "checkpoint_revision": spec["revision"],
                "episode_seed_reset": "study_episode_id", "native_noise_steering": True,
                "audit_service": args.audit, "model_shape": [wrapper.horizon, wrapper.action_dim],
                "physical_gpu_uuid": gpu_uuid, "cuda_visible_devices": device,
                "checkpoint_provenance": provenance,
                "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in files}}
    logging.getLogger(__name__).info("Serving %s at 127.0.0.1:%s", args.family, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        wrapper, host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
