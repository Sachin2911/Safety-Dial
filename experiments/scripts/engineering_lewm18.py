"""Start a task's frozen engineering panel once its pinned public inputs are ready."""

import argparse
import json
import subprocess
import sys
import time

from campaign_lewm18 import OUT, ROOT, run_stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", choices=("cube", "pusht"))
    args = parser.parse_args()
    while not (OUT / f"{args.env}-prepared.json").exists():
        if (OUT / "STOP").exists():
            raise SystemExit("Stopped before preparation completed")
        time.sleep(15)
    checks = sorted(OUT.glob("validation-*-result.json"))
    if not any(json.loads(p.read_text()).get("pass") for p in checks):
        raise RuntimeError("Pre-data physics validation has not passed")
    subprocess.run(
        [sys.executable, str(ROOT / "experiments/scripts/panels_lewm18.py"), "--env", args.env],
        check=True,
    )
    run_stage(OUT / f"engineering-{args.env}-freeze.json")


if __name__ == "__main__":
    main()
