"""Finish only saved-evidence audits and media after a known live comparison exits."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker-pid", type=int, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    out = args.root
    status = out / "artifact-postprocessing.json"
    while not (out / "report.json").exists():
        process = Path("/proc") / str(args.worker_pid) / "cmdline"
        live = process.exists() and b"run_cube_bounded.py" in process.read_bytes()
        state = {
            "status": "waiting" if live else "failed",
            "worker_pid": args.worker_pid,
            "worker_live": bool(live),
            "epoch": time.time(),
        }
        status.write_text(json.dumps(state, indent=2) + "\n")
        if not live:
            raise RuntimeError(
                "Comparison worker exited without a final report; do not restart it automatically"
            )
        time.sleep(15)
    commands = [["audit_portable_lewm.py", "--root", str(out)]]
    if not (out / "media").exists():
        commands.append(["media_portable_lewm.py", "--root", str(out)])
    commands.append(["plot_safety_adaptation.py", "--root", str(out)])
    records = []
    for command in commands:
        started = time.time()
        code = subprocess.call(
            [sys.executable, str(repo / "experiments/scripts" / command[0]), *command[1:]], cwd=repo
        )
        records.append(
            {"script": command[0], "exit_code": code, "elapsed_seconds": time.time() - started}
        )
        status.write_text(
            json.dumps({"status": "running" if code == 0 else "failed", "steps": records}, indent=2)
            + "\n"
        )
        if code:
            raise RuntimeError("Saved-evidence postprocessing failed")
    status.write_text(
        json.dumps({"status": "complete", "steps": records, "epoch": time.time()}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
