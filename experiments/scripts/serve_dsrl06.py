"""Serve the pinned frozen Jar policy with accounted DSRL06 routing."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_server import PolicyBridge, QueryLedger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05-base_datagen_v1_jar_joint_2cam_lora")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--query-ledger", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=21000)
    parser.add_argument("--actor-url", default="http://127.0.0.1:9009")
    args = parser.parse_args()

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as train_config
    import maniguard.openpi_sft.train_configs as maniguard_configs

    maniguard_configs.register()
    cfg = train_config.get_config(args.config)
    if (cfg.model.action_horizon, cfg.model.action_dim) != (16, 32):
        raise ValueError("Registered Jar model differs from the frozen latent interface")
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    bridge = PolicyBridge(
        policy,
        args.record_root,
        QueryLedger(args.query_ledger, args.max_queries),
        actor_url=args.actor_url,
    )
    metadata = {**bridge.metadata, "serve_config": args.config, "checkpoint": args.checkpoint}
    websocket_policy_server.WebsocketPolicyServer(
        bridge, host=args.host, port=args.port, metadata=metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
