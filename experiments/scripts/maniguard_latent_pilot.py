"""Resume matched search, seal finalists and execute the protected comparison."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_latent_pilot import Pilot
from helpers.maniguard_latent_services import ensure_policy, ensure_worker_policies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--search-only", action="store_true")
    args = parser.parse_args()
    def checkpoint(stage):
        environment = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
        subprocess.run([sys.executable, str(ROOT / "experiments/scripts"
                                          / "maniguard_latent_checkpoint.py"),
                        "--run", str(args.run), "--stage", stage], env=environment, check=True)

    pilot = Pilot(args.run, args.protocol, checkpoint=checkpoint)
    if pilot.protocol.get("workers") == 2:
        ensure_worker_policies(args.run, pilot.protocol["family"])
    else:
        ensure_policy(args.run, pilot.protocol["family"])
    pilot.run_search()
    if not args.search_only:
        pilot.run_audit()


if __name__ == "__main__":
    main()
