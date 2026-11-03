"""Wait for one existing Block run, then audit and package it without new rollouts."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, choices=[0, 2], required=True)
    parser.add_argument("--driver-pid", type=int, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--package-sha256", required=True)
    args = parser.parse_args()
    repo = Path("/workspace/Safety-Dial")
    assert repo.is_dir()
    root = repo / "SafeLeWM/outputs/block-paper-audit-20261102"
    run = repo / f"outputs/search2-20261102-r{args.repeat}"
    audit = root / "source-v1.py"
    package = root / "package-v1.py"
    out = root / f"r{args.repeat}-v1"
    receipt = root / f"r{args.repeat}-watcher-v1.json"
    assert not receipt.exists() and not out.exists(), "Do not restart earlier closeout work"
    result = dict(repeat=args.repeat, driver_pid=args.driver_pid, started_at=time.time(),
                  status="waiting", no_new_rollouts=True, manual_rental_shutdown=True)

    def save():
        result["updated_at"] = time.time()
        temp = receipt.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(receipt)

    save()
    try:
        assert sha(audit) == args.audit_sha256 and sha(package) == args.package_sha256
        deadline = time.monotonic() + 12 * 3600
        while True:
            status_path = run / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            proc = Path(f"/proc/{args.driver_pid}/cmdline")
            try:
                command = proc.read_bytes().split(bytes([0]))
            except FileNotFoundError:
                command = []
            live = any(c.endswith(b"/search2.py") for c in command)
            if live:
                assert any(f"search2-20261102-r{args.repeat}".encode() in c for c in command)
            result.update(driver_live=live, observed_run_status=status)
            save()
            if status.get("phase") == "complete" and not live:
                break
            if not live:
                raise RuntimeError("Original driver ended without a complete run; no restart")
            if time.monotonic() >= deadline:
                raise TimeoutError("Closeout watcher deadline reached; rental and job unchanged")
            time.sleep(30)
        assert sha(audit) == args.audit_sha256 and sha(package) == args.package_sha256
        result["status"] = "auditing"
        save()
        with (root / f"r{args.repeat}-v1.log").open("x") as log:
            subprocess.run([sys.executable, str(audit), "--repo", str(repo), "--repeat",
                            str(args.repeat), "--output", str(out)], check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        assert json.loads((out / "audit.json").read_text())["passed"]
        result["status"] = "packaging"
        save()
        with (root / f"r{args.repeat}-package-v1.log").open("x") as log:
            subprocess.run([sys.executable, str(package), "--repeat", str(args.repeat),
                            "--audit", str(out)], check=True, stdout=log,
                           stderr=subprocess.STDOUT)
        result.update(status="complete", packaging=json.loads((out / "packaging.json").read_text()))
        save()
    except BaseException:
        result.update(status="failed", error=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
