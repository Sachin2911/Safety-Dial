"""Verify and prepare the additional host; do not start simulator episodes."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/workspace/Safety-Dial")
freeze = json.loads((ROOT / "progress/receipts/09-unisafe-parallel-preparation.json").read_text())
rental = json.loads((ROOT / "progress/rental-unisafe09p-20261025.json").read_text())
assert os.environ.get("CONTAINER_ID") == str(rental["instance_id"])
for name, expected in freeze["source_sha256"].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True, check=True,
)
gpus = result.stdout.strip().splitlines()
assert len(gpus) == 4 and all("RTX 4090" in row and "580." in row for row in gpus)
output = ROOT / "outputs/unisafe09p"
for folder in ("source", "environment", "logs", "control"):
    (output / folder).mkdir(parents=True, exist_ok=True)
(output / "environment/gpus-before-setup.txt").write_text(result.stdout)
command = (
    "cd /workspace/Safety-Dial && /venv/main/bin/python -u "
    "experiments/scripts/unisafe09p_watchdog.py --remote "
    "> outputs/unisafe09p/logs/watchdog.log 2>&1"
)
assert subprocess.run(["tmux", "has-session", "-t", "watchdog09p"], capture_output=True).returncode
subprocess.run(["tmux", "new-session", "-d", "-s", "watchdog09p", command], check=True)
ready = output / "watchdog-ready.json"
for _ in range(100):
    if ready.exists():
        row = json.loads(ready.read_text())
        cmdline = Path(f"/proc/{row['pid']}/cmdline").read_bytes()
        if b"unisafe09p_watchdog.py" in cmdline:
            assert row["instance_id"] == rental["instance_id"]
            break
    time.sleep(0.1)
else:
    raise RuntimeError("Additional-host remote guard did not become ready")
setup = "cd /workspace/Safety-Dial && bash experiments/scripts/setup_unisafe09.sh"
setup += " > outputs/unisafe09p/logs/preparation.log 2>&1"
assert subprocess.run(["tmux", "has-session", "-t", "prepare09p"], capture_output=True).returncode
subprocess.run(["tmux", "new-session", "-d", "-s", "prepare09p", setup], check=True)
print(json.dumps({
    "instance_id": rental["instance_id"], "remote_guard_pid": row["pid"],
    "setup_session": "prepare09p", "simulator_episodes_started": 0,
}))
