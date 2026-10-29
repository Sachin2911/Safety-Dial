"""Explicit task-16 inventory, financial snapshots and verified rental teardown."""

import argparse
import json
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import instances
from helpers.lewm16_infra import RENTAL, VAST, current, remote, safe_mutation, snapshot, utc
from helpers.lewm16_protocol import OUT, digest, write_json

SEAL_CODE = r"""
import datetime, hashlib, json, os, signal, time
from pathlib import Path
root=Path('/workspace/Safety-Dial'); out=root/'outputs/lewm16'
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try: args=(p/'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if any(a.endswith(b'/worker_lewm16.py') for a in args):
        raise RuntimeError('A task-16 worker is still alive')
manifest=json.loads((root/'progress/receipts/16-final-ab-v1.json').read_text())
for job in manifest['jobs']:
    result=json.loads((out/'attempts'/job['id']/'result.json').read_text())
    if result['status']!='complete' or result['job']!=job:
        raise RuntimeError('Final comparison is incomplete or invalid')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
prepared=json.loads((out/'environment/prepared.json').read_text())
for env,record in prepared.items():
    if sha(root/f'data/lewm16/stablewm/checkpoints/{env}/lewm_object.ckpt')!=record['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed')
guard=json.loads((out/'remote-guard-ready.json').read_text())
pid=guard['pid']; cmd=Path(f'/proc/{pid}/cmdline')
if cmd.exists():
    if b'experiments/scripts/remote_guard_lewm16.py' not in cmd.read_bytes().split(b'\0'):
        raise RuntimeError('Guard process identity differs')
    os.kill(pid,signal.SIGTERM)
    for _ in range(20):
        if not cmd.exists(): break
        time.sleep(.1)
    if cmd.exists(): raise RuntimeError('Remote guard has not exited')
sealed={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'final_episodes':len(manifest['jobs']),'checkpoint_files_unchanged':True,
        'workers_absent':True,'remote_guard_stopped':True,
        'local_watchdog_remains_responsible_until_provider_destruction':True}
(out/'sealed.json').write_text(json.dumps(sealed,indent=2)+'\n')
files={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}
       for p in sorted(out.rglob('*')) if p.is_file() and p.name!='remote-inventory.json'}
(out/'remote-inventory.json').write_text(json.dumps({'seal':sealed,'files':files},indent=2)+'\n')
print(json.dumps({'sealed':True,'files':len(files),'bytes':sum(v['bytes'] for v in files.values())}))
"""


def verified_audit():
    audit = json.loads((ROOT / "progress/receipts/16-final-audit.json").read_text())
    if not audit["valid"] or audit["required_comparison_episodes"] < 1824:
        raise ValueError("Final comparison audit has not passed")
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["seal", "verify", "stop", "destroy", "billing"])
    parser.add_argument("--tag", default="closeout")
    args = parser.parse_args()
    rental = json.loads(RENTAL.read_text())
    instance = rental["instance_id"]
    if not rental["label"].startswith("lewm16-"):
        raise ValueError("Unexpected rental label")
    if args.action == "seal":
        result = remote("/venv/main/bin/python -c " + shlex.quote(SEAL_CODE), timeout=90)
        print(result.stdout)
    elif args.action == "verify":
        inventory = OUT / "remote/remote-inventory.json"
        data = json.loads(inventory.read_text())
        for name, item in data["files"].items():
            path = (OUT / "remote" / name).resolve()
            if not path.is_relative_to((OUT / "remote").resolve()):
                raise ValueError("Inventory path is outside the local mirror")
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError(f"Remote/local payload differs: {name}")
        receipt = dict(
            utc=utc(),
            instance_id=instance,
            verified=True,
            files=len(data["files"]),
            inventory_sha256=digest(inventory),
            seal=data["seal"],
        )
        write_json(ROOT / "progress/receipts/16-local-preservation.json", receipt)
        print(json.dumps(receipt))
    elif args.action in ("stop", "destroy"):
        audit = verified_audit()
        local = json.loads((ROOT / "progress/receipts/16-local-preservation.json").read_text())
        if local["instance_id"] != instance or not local["verified"]:
            raise ValueError("Local preservation is not verified for this rental")
        if args.action == "destroy":
            marker = json.loads(
                (ROOT / "progress/receipts/16-preservation-final-raw.json").read_text()
            )
            stage = OUT / "preservation/final-raw"
            if not marker["private"] or marker["attempts"] != audit["attempts"]:
                raise ValueError("Private final evidence is not verified")
            if digest(stage / "index.json") != marker["index_sha256"]:
                raise ValueError("Preservation index changed")
            with tarfile.open(stage / "metadata-and-sources.tar.gz") as archive:
                saved = json.load(archive.extractfile("progress/rental-lewm16-20261026.json"))
            if saved["instance_id"] != instance:
                raise ValueError("Private preservation belongs to another rental")
        _, provider = current()
        if provider is None:
            raise ValueError(
                "Rental already absent; reconcile complete instance list before mutation"
            )
        command = [args.action, "instance", str(instance)]
        if args.action == "destroy":
            # User-authorized teardown, after both preservation checks above.
            command.append("--yes")
        acknowledgement = safe_mutation(command)
        write_json(
            ROOT / f"progress/receipts/16-rental-{args.action}-request.json",
            dict(utc=utc(), instance_id=instance, acknowledgement=acknowledgement),
        )
        for _ in range(20):
            listing = instances()
            matching = [r for r in listing if r["id"] == instance or r["label"] == rental["label"]]
            done = (
                not matching
                if args.action == "destroy"
                else (len(matching) == 1 and matching[0]["actual_status"] in ("stopped", "exited"))
            )
            if done:
                key = "destroyed_utc" if args.action == "destroy" else "compute_stopped_utc"
                rental[key] = utc()
                rental["status"] = "destroyed" if args.action == "destroy" else "stopped"
                write_json(RENTAL, rental)
                receipt = dict(
                    utc=utc(),
                    action=args.action,
                    instance_id=instance,
                    verified=True,
                    acknowledgement=acknowledgement,
                    matching_provider_instances=matching,
                    financial_upper_estimate=snapshot(rental),
                )
                write_json(ROOT / f"progress/receipts/16-rental-{args.action}.json", receipt)
                print(json.dumps(receipt))
                return
            time.sleep(2)
        raise RuntimeError("Provider mutation not yet confirmed; re-poll the same rental")
    else:
        if not args.tag.replace("-", "").isalnum():
            raise ValueError("Invalid billing snapshot tag")
        result = subprocess.run(
            [VAST, "show", "invoices", "--start_date", "2026-10-26", "--only_charges", "--raw"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode:
            raise RuntimeError("Provider billing read failed; raw response suppressed")
        data = json.loads(result.stdout)
        rows = data if isinstance(data, list) else data["invoices"]
        rows = [
            {
                k: r.get(k)
                for k in (
                    "instance_id",
                    "amount",
                    "quantity",
                    "rate",
                    "timestamp",
                    "type",
                    "description",
                )
            }
            for r in rows
            if str(r.get("instance_id")) == str(instance)
        ]
        receipt = dict(
            utc=utc(),
            instance_id=instance,
            rows=rows,
            posted_snapshot_usd=sum(float(r["amount"]) for r in rows),
            financial_upper_estimate=snapshot(rental),
            note="Provider-posted snapshot, subject to billing rounding or later adjustments.",
        )
        write_json(ROOT / f"progress/receipts/16-billing-{args.tag}.json", receipt)
        print(json.dumps(receipt))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
