"""Bounded orchestration for the fresh AP comparison. Never shuts down rentals."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/pusht-ap-reproduction-20261102"


def command(stage, *arguments):
    return [sys.executable, str(ROOT / "experiments/scripts/pusht_ap_reproduction.py"),
            stage, *arguments]


def group(commands, name):
    from pusht_ap_reproduction import guard

    processes, logs = [], []
    try:
        for index, args in enumerate(commands):
            log = (OUT / f"{name}-{index}.log").open("xb")
            logs.append(log)
            processes.append(subprocess.Popen(
                args, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(index)),
                stdout=log, stderr=log,
            ))
        while any(p.poll() is None for p in processes):
            guard()
            if any(p.poll() not in [None, 0] for p in processes):
                raise RuntimeError(f"{name} failed; no retry or following phase")
            time.sleep(3)
        assert all(p.returncode == 0 for p in processes), name
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            proc.wait()
        for log in logs:
            log.close()
    print(json.dumps(dict(phase=name, status="complete", utc=time.time())), flush=True)


def main():
    assert socket.gethostname() == "c45db82630e6"
    assert (OUT / "setup-complete.json").exists()
    if len(sys.argv) == 1:
        assert not (OUT / "launch.json").exists()
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa",
                   GALLIUM_DRIVER="llvmpipe", LP_NUM_THREADS="1", SDL_VIDEODRIVER="dummy",
                   PYGAME_HIDE_SUPPORT_PROMPT="1")
        env.pop("PYTHONPATH", None)
        with (OUT / "orchestration.log").open("xb") as log:
            process = subprocess.Popen(
                [sys.executable, __file__, "execute"], cwd=ROOT, env=env,
                stdout=log, stderr=log, start_new_session=True, stdin=subprocess.DEVNULL,
            )
        save = dict(pid=process.pid, started=time.time(), rental_shutdown="manual")
        (OUT / "launch.json").write_text(json.dumps(save) + "\n")
        print(json.dumps(save), flush=True)
        return
    assert sys.argv[1:] == ["execute"]
    group([command("prepare")], "prepare")
    group([command("engineering")], "engineering")
    group([command("cache", "--shard", str(i)) for i in range(8)], "cache")
    group([command("cache-finish")], "cache-finish")
    for arm in ["authors", "wrapper"]:
        group([command("eval", "--arm", arm, "--shard", str(i)) for i in range(8)], arm)
    group([[sys.executable, str(ROOT / "experiments/scripts/audit_pusht_ap_reproduction.py")]],
          "audit")
    (OUT / "pipeline-complete.json").write_text(json.dumps(dict(completed=time.time())) + "\n")


if __name__ == "__main__":
    main()
