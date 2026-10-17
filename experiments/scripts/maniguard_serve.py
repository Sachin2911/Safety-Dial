"""Serve a pinned frozen policy locally with repeatable per-episode sampling."""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "third_party/ManiGuard"))
os.environ.setdefault("MANIGUARD_SKIP_OMNIGIBSON_PATCH", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from helpers.maniguard_parallel_policy import IsolatedEpisodePolicy
from helpers.maniguard_policy import EpisodeSeededPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("jar", "lid", "stack"), required=True)
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--parallel-clients", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    import jax
    from maniguard.openpi_sft import train_configs
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as train_config

    spec = json.loads((ROOT / "configs/maniguard/sources.json").read_text())["policies"][
        args.family
    ]
    folder = ROOT / "data/maniguard/policies" / args.family
    provenance = json.loads((folder / "download-provenance.json").read_text())
    if provenance["revision"] != spec["revision"] or provenance["repo"] != spec["repo"]:
        raise ValueError("Policy download provenance does not match the study pin")
    checkpoint = folder / spec["step"]
    train_configs.register()
    policy = policy_config.create_trained_policy(
        train_config.get_config(spec["config"]), checkpoint,
    )
    wrapper = IsolatedEpisodePolicy if args.parallel_clients else EpisodeSeededPolicy
    sources = [Path(__file__), ROOT / 'experiments/helpers/maniguard_policy.py',
               ROOT / 'experiments/helpers/maniguard_parallel_policy.py']
    metadata = {"serve_config": spec["config"], "checkpoint": str(checkpoint),
                "checkpoint_revision": spec["revision"], "episode_seed_reset": "study_episode_id",
                "rng_isolation": "per_episode" if args.parallel_clients else "active_episode",
                "server_source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sources}}
    logging.getLogger(__name__).info("Serving %s on localhost:%s", args.family, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        wrapper(policy, jax.random.key), host="127.0.0.1",
        port=args.port, metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
