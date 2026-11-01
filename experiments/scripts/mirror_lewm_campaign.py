"""Continuously copy campaign evidence, never models, data downloads or credentials."""

import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/lewm-campaign-20261031/mirror"
CONTROL = ROOT / "outputs/runpod-lewm-campaign-20261031"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true")
    p.add_argument("--lane", action="append", choices=("pusht", "navigation", "recovery"))
    p.add_argument("--state-prefix", choices=("mirror", "recovery-mirror"), default="mirror")
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location(
        "control", ROOT / "experiments/scripts/lewm_campaign_pods.py"
    )
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    OUT.mkdir(parents=True, exist_ok=True)
    while not (CONTROL / "mirror-cancelled").exists():
        records = []
        for lane in args.lane or ("pusht", "navigation"):
            folder = OUT / lane
            folder.mkdir(exist_ok=True)
            ssh, host = c.ssh_args(lane)
            command = [
                "rsync",
                "-az",
                "--partial",
                "--delay-updates",
                "--exclude=*.part",
                "--exclude=__pycache__",
                "--exclude=.pytest_cache",
                "-e",
                shlex.join(["ssh", *ssh]),
            ]
            started = time.monotonic()
            completed = []
            error = None
            try:
                for group in ("environment", "results", "configs"):
                    target = folder / group
                    target.mkdir(exist_ok=True)
                    subprocess.run(
                        [
                            *command,
                            host + ":/workspace/Safety-Dial/" + group + "/",
                            str(target) + "/",
                        ],
                        check=True,
                        capture_output=True,
                        timeout=180,
                    )
                    completed.append(group)
            except Exception as exc:
                error = type(exc).__name__
            records.append(
                {
                    "lane": lane,
                    "groups": completed,
                    "seconds": time.monotonic() - started,
                    "error": error,
                }
            )
        record = {"checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "lanes": records}
        with (CONTROL / (args.state_prefix + "-events.jsonl")).open("a", buffering=1) as stream:
            stream.write(json.dumps(record) + "\n")
        c.write(CONTROL / (args.state_prefix + "-state.json"), record)
        if args.once:
            print(json.dumps(record))
            return
        for _ in range(12):
            if (CONTROL / "mirror-cancelled").exists():
                return
            time.sleep(10)


if __name__ == "__main__":
    main()
