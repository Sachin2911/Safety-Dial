"""CPU-only published DSRL actor for the four-GPU frozen-policy evaluation."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.dsrl06_learner import FrozenActorBridge
from dsrl06_learner_service import serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "third_party/dsrl_pi0")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/dsrl06/actor-eval")
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9009)
    args = parser.parse_args()
    # Set these before the first lazy JAX import. This worker must not claim a GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("MPLCONFIGDIR", str(args.work_dir / "matplotlib-cache"))
    bridge = FrozenActorBridge(args.work_dir, args.source, args.checkpoint)
    serve(bridge, args.host, args.port, actor_only=True)


if __name__ == "__main__":
    main()
