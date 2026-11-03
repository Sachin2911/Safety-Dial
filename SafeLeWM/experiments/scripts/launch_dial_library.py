"""Detached, fail-closed library evaluation; never changes rental state."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/safety-dial-library-20261102"
LAUNCH = ROOT / "outputs/safety-dial-library-launch-20261102"


def main():
    assert socket.gethostname() == "075cb160c9b3"
    if len(sys.argv) == 1:
        assert not BASE.exists()
        LAUNCH.mkdir(parents=True, exist_ok=False)
        with (LAUNCH / "pipeline.log").open("xb") as log:
            child = subprocess.Popen(
                [sys.executable, __file__, "execute"],
                cwd=ROOT,
                env=dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1"),
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        (LAUNCH / "launch.json").write_text(
            json.dumps(dict(pid=child.pid, started=time.time(), rental_shutdown="manual")) + "\n"
        )
        print(json.dumps(dict(pid=child.pid)), flush=True)
        return
    assert sys.argv[1:] == ["execute"]
    for script in [
        "prepare_dial_library",
        "run_dial_library",
        "audit_dial_library",
        "preserve_dial_library",
    ]:
        with (LAUNCH / (script + ".log")).open("xb") as log:
            result = subprocess.run(
                [sys.executable, "experiments/scripts/" + script + ".py"],
                cwd=ROOT,
                stdout=log,
                stderr=log,
                timeout=14400,
            )
        print(
            json.dumps(dict(stage=script, returncode=result.returncode, time=time.time())),
            flush=True,
        )
        if result.returncode:
            raise SystemExit(result.returncode)
    (LAUNCH / "pipeline-complete.json").write_text(json.dumps(dict(completed=time.time())) + "\n")


if __name__ == "__main__":
    main()
