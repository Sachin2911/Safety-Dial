"""Detached job orchestration. Never changes the rental lifecycle or retries science."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/ppo-20261101"


def main():
    assert socket.gethostname() in ["075cb160c9b3", "8cf3a706cdd1"]
    assert (OUT / "software-checks-passed").exists()
    if len(sys.argv) == 1:
        assert not (OUT / "launch.json").exists()
        environment = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
        with (OUT / "orchestration.log").open("xb") as log:
            process = subprocess.Popen(
                [sys.executable, __file__, "execute"],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=log,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
            )
        record = dict(
            pid=process.pid,
            launched_at=time.time(),
            host=socket.gethostname(),
            rental_stop="manual only",
        )
        (OUT / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        return
    assert sys.argv[1:] == ["execute"]
    for script, log_name in [
        ("run_matched_ppo.py", "run.log"),
        ("audit_matched_ppo.py", "audit.log"),
        ("preserve_matched_ppo.py", "preservation.log"),
    ]:
        with (OUT / log_name).open("xb") as log:
            result = subprocess.run(
                [sys.executable, str(ROOT / "experiments/scripts" / script)],
                cwd=ROOT,
                stdout=log,
                stderr=log,
                check=False,
            )
        print(
            json.dumps(dict(script=script, returncode=result.returncode, time=time.time())),
            flush=True,
        )
        if result.returncode:
            raise SystemExit(result.returncode)
    (OUT / "pipeline-complete.json").write_text(json.dumps(dict(completed_at=time.time())) + "\n")


if __name__ == "__main__":
    main()
