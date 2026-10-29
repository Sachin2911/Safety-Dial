"""Close out task 19 hosts once their last episode is recorded: preserve, then destroy.

After the no-new-episode time, each live host is watched until it has no queued or
claimed job and the orchestrator's state has no job still assigned to it (or until the
hard limit), then `preserve_evo19.py <host>` verifies the local mirror and
`manage_evo19.py destroy <host>` removes the instance. Hosts are handled in parallel.

    .venv/bin/python experiments/scripts/closeout_evo19.py
"""

import concurrent.futures
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments/scripts"))
import manage_evo19 as infra  # noqa: E402
from helpers.dsrl06_infra import ssh_endpoint, utc  # noqa: E402

STATE = ROOT / "outputs/evo19/search/state.json"
LOG = ROOT / "outputs/evo19/control/closeout.log"
START = dt.datetime(2026, 10, 27, 2, 50, tzinfo=dt.timezone.utc).timestamp()
HARD = dt.datetime(2026, 10, 27, 3, 8, tzinfo=dt.timezone.utc).timestamp()
REMOTE = "/workspace/Safety-Dial/outputs/evo19"
PY = str(ROOT / ".venv/bin/python")


def log(**fields):
    line = json.dumps({"utc": utc(), **fields})
    print(line, flush=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def host_busy(name):
    _, row = infra.host_row(name)
    address, port = ssh_endpoint(row)
    command = f"ls {REMOTE}/queue {REMOTE}/claimed 2>/dev/null | grep -c json || true"
    result = subprocess.run(["ssh", *infra.ssh_options(port), f"root@{address}", command],
                            capture_output=True, text=True, timeout=60)
    remote = int((result.stdout.strip().splitlines() or ["1"])[-1])
    state = json.loads(STATE.read_text())
    pending = sum(j["status"] == "assigned" and j["host"] == name
                  for j in state["jobs"].values())
    return remote, pending


def close(name):
    while time.time() < HARD:
        try:
            remote, pending = host_busy(name)
        except (subprocess.TimeoutExpired, ValueError, RuntimeError) as error:
            log(host=name, check_error=str(error)[:200])
            time.sleep(20)
            continue
        if remote == 0 and pending == 0:
            break
        time.sleep(20)
    log(host=name, event="preserve_start")
    preserve = subprocess.run([PY, "experiments/scripts/preserve_evo19.py", name], cwd=ROOT,
                              capture_output=True, text=True, timeout=1800)
    log(host=name, event="preserve_end", rc=preserve.returncode,
        out=preserve.stdout[-400:], err=preserve.stderr[-400:])
    if preserve.returncode:
        return name, "preserve_failed"
    destroy = subprocess.run([PY, "experiments/scripts/manage_evo19.py", "destroy", name],
                             cwd=ROOT, capture_output=True, text=True, timeout=300)
    log(host=name, event="destroy_end", rc=destroy.returncode, out=destroy.stdout[-300:],
        err=destroy.stderr[-300:])
    return name, "destroyed" if destroy.returncode == 0 else "destroy_failed"


def main():
    while time.time() < START:
        time.sleep(15)
    record = infra.load()
    live = [h["name"] for h in record["hosts"] if h.get("status") == "created"]
    log(event="closeout_start", hosts=live)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(live) or 1) as pool:
        results = dict(pool.map(close, live))
    log(event="closeout_end", results=results)


if __name__ == "__main__":
    main()
