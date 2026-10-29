"""Manage only the task 15 recovery rental, in its own namespace and financial book.

The rental workflow is manage_regrasp15.py's, rebound to the recovery book
(outputs/regrasp15r1/control/finance.json, which imports every earlier task 15
allocation), the recovery rental record and the recovery output tree. Offer limits,
the $10 cap, the reservation-before-create rule, the local watchdog and the stop and
destroy guards are unchanged. Only sync, deploy and destroy are restated, because the
originals name the old remote paths and preservation helper.
"""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments/scripts"))
import manage_regrasp15 as base  # noqa: E402
from helpers.regrasp15_infra import (  # noqa: E402
    atomic_json,
    instances,
    provider_stopped,
    ssh_endpoint,
    ssh_options,
    utc,
    vast,
)

OUT = ROOT / "outputs/regrasp15r1"
BOOK = OUT / "control/finance.json"
RENTAL = ROOT / "progress/rental-regrasp15r1-20261026.json"
REMOTE_OUT = "/workspace/Safety-Dial/outputs/regrasp15r1/"
PREP_FREEZE = ROOT / "progress/receipts/15-regrasp-recovery-prep-v1.json"
base.OUT, base.BOOK, base.RENTAL = OUT, BOOK, RENTAL


def provider_row():
    rental = json.loads(RENTAL.read_text())
    row = next((r for r in instances() if r["id"] == rental["instance_id"]), None)
    if row is None:
        raise ValueError("Recovery rental is absent from provider state")
    return rental, row


def sync():
    _, provider = provider_row()
    host, port = ssh_endpoint(provider)
    destination = OUT / "remote"
    destination.mkdir(parents=True, exist_ok=True)
    transport = shlex.join(["ssh", *ssh_options(port)])
    command = ["rsync", "-az", "--partial", "--timeout=90", "-e", transport,
               f"root@{host}:{REMOTE_OUT}", str(destination) + "/"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    print(json.dumps({"sync_exit": result.returncode, "utc": utc()}))
    if result.returncode:
        raise SystemExit(result.returncode)


def deploy(args):
    base.verify_freeze(args.plan)
    _, provider = provider_row()
    host, port = ssh_endpoint(provider)
    bundle = args.bundle.resolve(strict=True)
    transport = shlex.join(["ssh", *ssh_options(port)])
    subprocess.run(["rsync", "-az", "--timeout=90", "-e", transport, str(bundle),
                    f"root@{host}:/tmp/regrasp15r1-source.tar.gz"],
                   capture_output=True, text=True, timeout=300, check=True)
    check = subprocess.run(
        ["ssh", *ssh_options(port), f"root@{host}", "sha256sum /tmp/regrasp15r1-source.tar.gz"],
        capture_output=True, text=True, timeout=30, check=True)
    if check.stdout.split()[0] != hashlib.sha256(bundle.read_bytes()).hexdigest():
        raise ValueError("Transferred source bundle hash differs")
    if args.extract_only:
        remote = (
            "tar -xzf /tmp/regrasp15r1-source.tar.gz -C /workspace/Safety-Dial && "
            "cp /tmp/regrasp15r1-source.tar.gz /workspace/Safety-Dial/outputs/regrasp15r1/"
            f"source/{bundle.name} && echo extracted {bundle.name}"
        )
    else:
        remote = ("tar -xOzf /tmp/regrasp15r1-source.tar.gz "
                  "experiments/scripts/bootstrap_regrasp15r1.sh | bash")
    result = subprocess.run(["ssh", *ssh_options(port), f"root@{host}", remote],
                            capture_output=True, text=True, timeout=120, check=True)
    print(result.stdout, end="")


def destroy(args):
    rental = json.loads(RENTAL.read_text())
    marker = json.loads(args.marker.read_text())
    if not (
        marker.get("instance_id") == rental["instance_id"]
        and marker.get("local_verified") is True
        and marker.get("private_hf_verified") is True
        and isinstance(marker.get("pinned_revisions"), dict)
        and marker["pinned_revisions"]
        and len(marker.get("manifest_sha256", "")) == 64
    ):
        raise ValueError("Complete local and private pinned preservation is required")
    manifest = Path(marker["manifest_path"])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != marker["manifest_sha256"]:
        raise ValueError("Preservation manifest changed after private verification")
    from preserve_regrasp15r1 import verify_local

    verify_local(manifest.parent.parent.parent)
    rows = instances()
    row = next((r for r in rows if r["id"] == rental["instance_id"]), None)
    if not provider_stopped(row):
        raise ValueError("Stop and verify compute before destroying its disk")
    if row is not None:
        try:
            vast(["destroy", "instance", str(rental["instance_id"]), "--yes"])
        except RuntimeError:
            pass  # Never repeat this mutation on an uncertain acknowledgement.
        for _ in range(12):
            if not any(r["id"] == rental["instance_id"] for r in instances()):
                break
            time.sleep(5)
        else:
            raise RuntimeError("Destruction is not yet verified by provider absence")
    rental["status"] = "destroyed"
    rental["destroy_verified_utc"] = utc()
    rental["preservation_marker_sha256"] = hashlib.sha256(args.marker.read_bytes()).hexdigest()
    atomic_json(RENTAL, rental)
    print(json.dumps({"destroyed": rental["instance_id"], "absence_verified": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("offers")
    sub.add_parser("status")
    create_parser = sub.add_parser("create")
    create_parser.add_argument("--offer", type=int, required=True)
    create_parser.add_argument("--plan", type=Path, default=PREP_FREEZE)
    remote = sub.add_parser("ssh")
    remote.add_argument("command")
    remote.add_argument("--timeout", type=float, default=60)
    sub.add_parser("sync")
    sub.add_parser("stop")
    deployment = sub.add_parser("deploy")
    deployment.add_argument("--bundle", type=Path, required=True)
    deployment.add_argument("--extract-only", action="store_true")
    deployment.add_argument("--plan", type=Path, default=PREP_FREEZE)
    destruction = sub.add_parser("destroy")
    destruction.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "offers":
        print(json.dumps({"checked_utc": utc(), "offers": base.offers()}))
    elif args.action == "create":
        if RENTAL.exists():
            raise SystemExit("A recovery rental record already exists; reconcile it first")
        base.create(args)
    elif args.action == "status":
        base.status()
    elif args.action == "ssh":
        base.ssh(args)
    elif args.action == "sync":
        sync()
    elif args.action == "deploy":
        deploy(args)
    elif args.action == "destroy":
        destroy(args)
    else:
        base.stop()


if __name__ == "__main__":
    main()
