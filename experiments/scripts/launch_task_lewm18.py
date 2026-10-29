"""Run a task after setup, with counted physics/model checks and frozen stage admission."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/lewm18"


def wait_for(path):
    while not path.exists() or path.stat().st_size == 0:
        if (OUT / "STOP").exists():
            raise SystemExit("Stopped while waiting for preparation")
        time.sleep(10)


def run(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / "experiments/scripts" / script), *arguments], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", choices=("cube", "pusht"))
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"
    wait_for(OUT / "environment/installed-freeze.txt")
    run("validate_lewm18.py")
    wait_for(ROOT / f"data/lewm18/stablewm/checkpoints/{args.env}/lewm_object.json")
    run("check_model_lewm18.py", "--env", args.env)
    run("engineering_lewm18.py", args.env)
    run("campaign_lewm18.py", "run", "--env", args.env)


if __name__ == "__main__":
    main()
