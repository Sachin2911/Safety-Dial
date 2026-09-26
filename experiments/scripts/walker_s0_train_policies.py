#!/usr/bin/env python3
"""S0: train Walker2d behaviour policies with OmniSafe in the isolated Python 3.10 venv.

Run with the OmniSafe interpreter, NOT the project venv:
    /workspace/omnisafe/.venv/bin/python experiments/scripts/walker_s0_train_policies.py --algo PPO
    /workspace/omnisafe/.venv/bin/python experiments/scripts/walker_s0_train_policies.py --algo PPOLag

Only policy weights leave this environment (AGENTS.md: never datasets or reported
numbers). Checkpoints are saved every `--save-every` epochs to give a ladder of
competence and speed; `walker_s0_export_policies.py` turns them into plain MLP state
dicts + observation-normaliser stats for `helpers.locoPolicies.TorchActorPolicy`.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["PPO", "PPOLag"], required=True)
    ap.add_argument("--env", default="SafetyWalker2dVelocity-v1")
    ap.add_argument("--total-steps", type=int, default=4_000_000)
    ap.add_argument("--steps-per-epoch", type=int, default=20_000)
    ap.add_argument("--save-every", type=int, default=10, help="epochs between checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cost-limit", type=float, default=25.0)
    ap.add_argument("--log-dir", default=str(REPO_ROOT / "runs" / "omnisafe"))
    args = ap.parse_args()

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, str(args.threads))
    import omnisafe

    custom = {
        "seed": args.seed,
        "train_cfgs": {"device": "cpu", "torch_threads": args.threads, "total_steps": args.total_steps,
                       "vector_env_nums": 1, "parallel": 1},
        "algo_cfgs": {"steps_per_epoch": args.steps_per_epoch},
        "logger_cfgs": {"use_wandb": False, "use_tensorboard": False, "save_model_freq": args.save_every,
                        "log_dir": args.log_dir},
    }
    if args.algo == "PPOLag":
        custom["lagrange_cfgs"] = {"cost_limit": args.cost_limit}
    t0 = time.time()
    agent = omnisafe.Agent(args.algo, args.env, custom_cfgs=custom)
    agent.learn()
    print(f"[s0] {args.algo} done in {(time.time() - t0) / 60:.1f} min; logs under {args.log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
