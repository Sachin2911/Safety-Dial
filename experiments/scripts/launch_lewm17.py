"""Launch a frozen batch on registered task-17 workers, with persistent process receipts."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm17_execution import ready
from helpers.lewm17_protocol import OUT, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--optimizer", action="store_true")
    args = parser.parse_args()
    ready()
    data = json.loads(Path(args.manifest).read_text())
    receipt = OUT / "control" / f"launch-{data['name']}.json"
    if receipt.exists():
        raise FileExistsError("Do not duplicate a batch launch")
    script = (
        "experiments/scripts/optimize_lewm17.py"
        if args.optimizer
        else "experiments/scripts/worker_lewm17.py"
    )
    workers = []
    for gpu in range(4):
        for lane in range(2):
            log_path = OUT / "logs" / f"{data['name']}-g{gpu}-l{lane}.log"
            with log_path.open("w") as log:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        script,
                        "--manifest",
                        args.manifest,
                        "--gpu",
                        str(gpu),
                        "--lane",
                        str(lane),
                        "--lanes",
                        "2",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            subprocess.run(
                [
                    sys.executable,
                    "experiments/scripts/control_lewm17.py",
                    "register",
                    "--pid",
                    str(proc.pid),
                    "--script",
                    script,
                ],
                check=True,
            )
            workers.append({"pid": proc.pid, "gpu": gpu, "lane": lane, "log": str(log_path)})
    write_json(
        receipt, {"started_epoch": time.time(), "workers": workers, "manifest": args.manifest}
    )
    print(receipt)


if __name__ == "__main__":
    main()
