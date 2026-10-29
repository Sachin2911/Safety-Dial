"""Advance already-frozen batches when verified live workers finish; never restart them."""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm17_execution import ready, register
from helpers.lewm17_protocol import CONFIG, OUT, write_json


def live_workers(launch_path):
    launch = json.loads(launch_path.read_text())
    result = []
    for worker in launch["workers"]:
        pid = worker["pid"]
        proc = Path(f"/proc/{pid}")
        registration = json.loads((OUT / f"control/workers/{pid}.json").read_text())
        try:
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            command = (proc / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        if (
            stat[0] != "Z"
            and stat[19] == registration["start_ticks"]
            and registration["script"].encode() in command
        ):
            result.append(pid)
    return result


def run(script):
    ready()
    print(json.dumps({"starting": script, "epoch": time.time()}), flush=True)
    subprocess.run([sys.executable, "-u", f"experiments/scripts/{script}"], check=True)


def main():
    register()
    ready()
    while True:
        ready()
        live = live_workers(OUT / "control/launch-optimizer-comparison.json")
        complete = len(list((OUT / "optimizers").glob("*/selected.json")))
        write_json(
            OUT / "control/continuation-status.json",
            {
                "stage": "optimizer",
                "live_pids": live,
                "completed_runs": complete,
                "updated_epoch": time.time(),
            },
        )
        if not live:
            if complete != 30:
                raise RuntimeError(
                    "Optimizer workers are terminal before all runs completed; inspect failures"
                )
            break
        time.sleep(30)
    run("preserve_lewm17.py")
    config = json.loads((CONFIG / "comparison.json").read_text())
    guard = json.loads((OUT / "control/guard-status.json").read_text())
    timing = max(config["timing_seconds_per_episode_mean_times_1_5"].values())
    final_per_env = config["final_cases"] * 2 * 2 * 17
    final_forecast = final_per_env / 4 * timing * 1.4 + 1200
    if time.time() + final_forecast >= guard["budget"]["deadline_epoch"] - 1800:
        raise RuntimeError(
            "Full frozen final batch no longer fits before reporting and closeout reserve"
        )
    run("finalize_lewm17.py")
    run("preserve_lewm17.py")
    subprocess.run(
        [
            sys.executable,
            "experiments/scripts/launch_lewm17.py",
            "progress/receipts/17-final-evaluation.json",
        ],
        check=True,
    )
    while True:
        ready()
        live = live_workers(OUT / "control/launch-final-evaluation.json")
        complete = len(list((OUT / "attempts").glob("H-*/result.json")))
        write_json(
            OUT / "control/continuation-status.json",
            {
                "stage": "final",
                "live_pids": live,
                "completed_final_attempts": complete,
                "updated_epoch": time.time(),
            },
        )
        if not live:
            if complete != final_per_env * 2:
                raise RuntimeError("Final workers terminal with incomplete frozen workload")
            break
        time.sleep(30)
    run("preserve_lewm17.py")
    run("analyze_lewm17.py")
    run("report_lewm17.py")
    write_json(
        OUT / "control/continuation-status.json",
        {
            "stage": "analysis_ready_for_review",
            "updated_epoch": time.time(),
            "next": "Review report and figures; write Odyssey and completion; verify backups; hand off to guard",
        },
    )


if __name__ == "__main__":
    main()
