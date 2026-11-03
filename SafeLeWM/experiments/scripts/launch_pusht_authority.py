"""Detached preparation, bounded qualification, audit and packaging. Never stops rentals."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/pusht-authority-20261102"


def run_command(args, log_name):
    with (OUT / log_name).open("xb") as log:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=log, stderr=log)
    print(
        json.dumps(dict(command=args, returncode=result.returncode, time=time.time())), flush=True
    )
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    assert socket.gethostname() == "c45db82630e6"
    assert (OUT / "software-checks-passed").exists()
    if len(sys.argv) == 1:
        assert not (OUT / "launch.json").exists()
        env = dict(
            os.environ,
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="1",
            PYTHONPATH=str(ROOT / "ap-deps-pusht") + ":" + str(ROOT / "ap-deps"),
        )
        with (OUT / "orchestration.log").open("xb") as log:
            process = subprocess.Popen(
                [sys.executable, __file__, "execute"],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
            )
        record = dict(
            pid=process.pid,
            launched_at=time.time(),
            host=socket.gethostname(),
            rental_shutdown="manual only",
        )
        (OUT / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        return
    assert sys.argv[1:] == ["execute"]
    assert json.loads((OUT / "pusht/panel.json").read_text())["status"] == "ready"
    deadline = json.loads((OUT / "data-receipt.json").read_text())["started"] + 14400
    processes = []
    logs = []
    for lane in range(8):
        log = (OUT / f"cache-gpu{lane}.log").open("xb")
        logs.append(log)
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "experiments/scripts/prepare_pusht_authority.py",
                    "cache",
                    "--shard",
                    str(lane),
                ],
                cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(lane)),
                stdout=log,
                stderr=log,
            )
        )
    while any(p.poll() is None for p in processes):
        if time.time() > deadline or any(p.poll() not in (None, 0) for p in processes):
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            for p in processes:
                p.wait()
            raise RuntimeError("Cache failure/deadline; no experiment launched")
        time.sleep(5)
    for log in logs:
        log.close()
    assert all(p.returncode == 0 for p in processes)
    run_command(
        ["experiments/scripts/prepare_pusht_authority.py", "cache-finish"], "cache-finish.log"
    )
    for script, log in [
        ("run_pusht_authority.py", "run.log"),
        ("audit_pusht_authority.py", "audit.log"),
        ("preserve_pusht_authority.py", "preservation.log"),
    ]:
        run_command(["experiments/scripts/" + script], log)
    (OUT / "pipeline-complete.json").write_text(json.dumps(dict(completed=time.time())) + "\n")


if __name__ == "__main__":
    main()
