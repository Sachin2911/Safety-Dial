"""Copy added-host evidence over four independent resumable SSH streams."""

import concurrent.futures
import fcntl
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.unisafe09_infra import atomic_json, instances, ssh_endpoint, ssh_options, utc


def main():
    output = ROOT / "outputs/unisafe09p"
    control = output / "control"
    control.mkdir(parents=True, exist_ok=True)
    local = output / "remote"
    (local / "attempts").mkdir(parents=True, exist_ok=True)
    with (control / "sync.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rental = json.loads((ROOT / "progress/rental-unisafe09p-20261025.json").read_text())
        provider = next((r for r in instances() if r["id"] == rental["instance_id"]), None)
        if provider is None or provider["actual_status"] != "running":
            raise ValueError("The recorded added host must still be running for this copy")
        host, port = ssh_endpoint(provider)
        transport = shlex.join(["ssh", *ssh_options(port)])
        remote = f"root@{host}:/workspace/Safety-Dial/outputs/unisafe09p/"
        base = ["rsync", "-az", "--partial", "--timeout=120", "-e", transport]

        def copy(lane):
            start = time.monotonic()
            with (control / f"sync-lane-{lane}.log").open("w") as log:
                result = subprocess.run(base + [
                    f"--include=/p{lane}-*/", f"--include=/p{lane}-*/***", "--exclude=*",
                    remote+"attempts/", str(local / "attempts")+"/",
                ], stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            return {"lane": lane, "exit_code": result.returncode,
                    "seconds": time.monotonic()-start}

        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(copy, range(4)))
        with (control / "sync-metadata.log").open("w") as log:
            metadata = subprocess.run(
                base+["--exclude=/attempts/", remote, str(local)+"/"],
                stdout=log, stderr=subprocess.STDOUT, timeout=600,
            )
        record = {"utc": utc(), "instance_id": rental["instance_id"], "lanes": results,
                  "metadata_exit": metadata.returncode, "seconds": time.monotonic()-start,
                  "all_copies_complete": metadata.returncode == 0
                  and all(r["exit_code"] == 0 for r in results),
                  "copied_files_are_not_proof_of_panel_completion": True}
        atomic_json(control / "sync-result.json", record)
        print(json.dumps(record), flush=True)
        if not record["all_copies_complete"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
