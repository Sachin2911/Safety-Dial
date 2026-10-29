"""Run an immutable task-17 episode manifest on one registered GPU lane."""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm17_execution import Executor, configure, ready, register, verify_freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=2)
    args = parser.parse_args()
    register()
    ready()
    configure(args.gpu)
    import torch

    torch.set_num_threads(1)
    manifest = verify_freeze(args.manifest)
    jobs = []
    for job in manifest["jobs"]:
        h = int(hashlib.sha256(job["block"].encode()).hexdigest()[:8], 16)
        gpu = h % 2 + (2 if job["env"] == "reacher" else 0)
        lane = h // 2 % args.lanes
        if (gpu, lane) == (args.gpu, args.lane):
            jobs.append(job)
    executor = Executor(
        args.gpu, args.manifest, f"{manifest['name']}-g{args.gpu}-l{args.lane}", len(jobs)
    )
    for job in jobs:
        executor.run(job)


if __name__ == "__main__":
    main()
