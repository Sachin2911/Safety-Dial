"""One-off guarded pause of PushT's original batch at a fully finished pair.

Run only on the current rented host through compute.py ssh. Never holds the
ledger lock across an unfinished case. SIGINT lets the evaluator save its
actual global counters in its existing finally block.
"""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path("/workspace/LeWM_Experiment1")
RUN = "ap-20261105-eval-pusht-s0"
PID = 33890
START = "321898777"
CODE = ROOT / "runs" / RUN / ".control/code"
ASSETS = ROOT / "assets/ap_reproduction"
TOOLS = ROOT / ".control/tools/64233744bbeacda9896ce4298f455b243066c58b8004bd174b713cfe2398df4e"
sys.path.insert(0, str(TOOLS))
from run_io import publish_artifact  # noqa: E402


def identity():
    proc = Path(f"/proc/{PID}")
    stat = (proc / "stat").read_text().split(") ", 1)[1].split()
    expected = ["/venv/main/bin/python", "evaluate.py", "pusht", "--shards", "4",
                "--shard-index", "0"]
    command = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    if stat[19] != START or command != expected or (proc / "cwd").resolve() != CODE:
        raise RuntimeError("Exact evaluator process identity changed")


def main():
    identity()
    until = min(time.time() + 240, 1793877520)
    with (ASSETS / "attempt-ledger.lock").open("a") as lock:
        while time.time() < until:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(0.005)
                continue
            raw = (ASSETS / "attempt-ledger.json").read_bytes()
            ledger = json.loads(raw)
            attempts = {k: v for k, v in ledger["attempts"].items() if v.get("run") == RUN}
            complete = bool(attempts) and all(
                v.get("completed") is True and v.get("finished_epoch") and v.get("counts")
                for v in attempts.values())
            latest = max(attempts, key=lambda k: attempts[k].get("finished_epoch", 0))
            if not complete or not latest.endswith("_gaussian_observed"):
                fcntl.flock(lock, fcntl.LOCK_UN)
                time.sleep(0.005)
                continue
            identity()
            for key in attempts:
                case = key.split(":", 2)[2]
                for relative in (f"metrics/cases/{case}.json",
                                 f"metrics/trajectories/{case}.json.gz",
                                 f"metrics/journals/{case}.jsonl",
                                 f"artifacts/videos/{case}.mp4"):
                    if not (ROOT / "runs" / RUN / relative).is_file():
                        raise RuntimeError("Finished case lacks a published artifact: " + relative)
            receipt = {"run_id": RUN, "epoch": time.time(), "pid": PID, "start": START,
                       "signal": "SIGINT", "all_reserved_cases_completed": True,
                       "completed_identities": sorted(attempts),
                       "ledger_sha256": hashlib.sha256(raw).hexdigest(),
                       "reason": "Pause between completed pairs to correct passive image serialization"}
            publish_artifact("artifacts/recording-boundary-pause.json",
                             data=json.dumps(receipt) + "\n", run_dir=ROOT / "runs" / RUN)
            os.kill(PID, signal.SIGINT)
            print(json.dumps(receipt), flush=True)
            # Retain the quiescence guard until exit. The existing supervisor
            # independently enforces this job's 1,800-second runtime bound.
            while Path(f"/proc/{PID}").exists():
                time.sleep(0.1)
            print(json.dumps({"exited": True, "epoch": time.time()}), flush=True)
            return
    raise TimeoutError("No fully completed pair boundary caught; no signal sent")


if __name__ == "__main__":
    main()
