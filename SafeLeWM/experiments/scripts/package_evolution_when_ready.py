"""Package a completed audited job without starting or retrying any experiment."""

import json
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/evolution-budget-20261102"


def main():
    assert socket.gethostname() == "8cf3a706cdd1"
    if len(sys.argv) == 1:
        assert not (BASE / "packaging-launch.json").exists()
        with (BASE / "packaging.log").open("xb") as log:
            child = subprocess.Popen(
                [sys.executable, __file__, "watch"],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        (BASE / "packaging-launch.json").write_text(
            json.dumps(dict(pid=child.pid, started=time.time())) + "\n"
        )
        print(json.dumps(dict(pid=child.pid)), flush=True)
        return
    assert sys.argv[1:] == ["watch"]
    run = json.loads((BASE / "launch.json").read_text())
    while time.time() < run["started"] + 9 * 3600:
        if (BASE / "pipeline-complete.json").exists():
            assert json.loads((BASE / "run-v1/audit.json").read_text())["passed"]
            subprocess.run(
                [sys.executable, "experiments/scripts/preserve_evolution_budget.py"],
                cwd=ROOT,
                check=True,
            )
            (BASE / "packaging-complete.json").write_text(
                json.dumps(dict(completed=time.time())) + "\n"
            )
            return
        process = Path("/proc") / str(run["pid"]) / "cmdline"
        if not process.exists() or b"run_evolution_budget.py" not in process.read_bytes():
            raise RuntimeError("Experiment process ended before audited completion; no retry")
        time.sleep(30)
    raise TimeoutError("Packaging watcher deadline; rental remains unchanged")


if __name__ == "__main__":
    main()
