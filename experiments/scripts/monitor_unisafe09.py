"""Read live task-9 processes and progress without starting or stopping any work."""

import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.unisafe09_infra import (
    FinancialBook,
    atomic_json,
    instances,
    ssh_endpoint,
    ssh_options,
    utc,
)

REMOTE = r'''
import json, os
from pathlib import Path
root=Path('/workspace/Safety-Dial/outputs/unisafe09')
stages={}
for p in sorted((root/'attempts').glob('*')):
    intent_path=p/'intent.json'
    if not intent_path.exists():
        continue
    intent=json.loads(intent_path.read_text())
    stage=stages.setdefault(intent['job']['stage'],
                            dict(attempted=0,finished=0,complete=0,invalid=[],live=[]))
    stage['attempted']+=1
    result=p/'process-result.json'
    if result.exists():
        stage['finished']+=1
        result=json.loads(result.read_text())
        if result['exit_code']==0 and (p/'summary.json').exists() and not (
            p/'invalidation.json'
        ).exists():
            stage['complete']+=1
        else:
            stage['invalid'].append(p.name)
        continue
    process_path=p/'process.json'
    process=json.loads(process_path.read_text()) if process_path.exists() else {}
    pid=process.get('pid')
    command=Path(f'/proc/{pid}/cmdline')
    try:
        command=command.read_bytes().split(b'\0')
    except FileNotFoundError:
        command=[]
    live=any(x.endswith(b'/worker_unisafe09.py') for x in command) and (
        str(p).encode() in command
    )
    account=p/'accounting.json'
    account=json.loads(account.read_text()) if account.exists() else {}
    stage['live'].append(dict(attempt=p.name,pid=pid,process_live=live,
                             phase=account.get('phase','startup'),
                             completed_controls=account.get('completed_controls',0),
                             account_utc=account.get('utc')))
markers={}
for p in root.glob('stage-*-complete.json'):
    markers[p.name]=json.loads(p.read_text())
w=json.loads((root/'watchdog-ready.json').read_text())
try:
    cmd=Path(f"/proc/{w['pid']}/cmdline").read_bytes()
except FileNotFoundError:
    cmd=b''
print(json.dumps(dict(stages=stages,completion_markers=markers,
                     remote_watchdog_live=b'unisafe09_watchdog.py' in cmd,
                     remote_watchdog_pid=w['pid'],stop_file=(root/'STOP').exists())))
'''


def main():
    rental = json.loads((ROOT / "progress/rental-unisafe09-20261025.json").read_text())
    provider = next((x for x in instances() if x["id"] == rental["instance_id"]), None)
    control = ROOT / "outputs/unisafe09/control"
    pid = int((control / "watchdog.pid").read_text())
    try:
        os.kill(pid, 0)
        local_live = b"unisafe09_watchdog.py" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except (ProcessLookupError, FileNotFoundError):
        local_live = False
    record = {
        "utc": utc(), "instance_id": rental["instance_id"], "provider": provider,
        "local_watchdog_live": local_live, "local_watchdog_pid": pid,
        "finance": FinancialBook(control / "finance.json").snapshot(),
        "allocation_deadline_utc": rental["allocation_deadline_utc"],
        "seconds_before_preservation": (
            dt.datetime.fromisoformat(rental["allocation_deadline_utc"])
            - dt.datetime.now(dt.timezone.utc)
        ).total_seconds() - 1800,
    }
    if provider and provider["actual_status"] == "running":
        host, port = ssh_endpoint(provider)
        result = subprocess.run(
            ["ssh", *ssh_options(port), f"root@{host}", "python3 -c " + shlex.quote(REMOTE)],
            capture_output=True, text=True, timeout=45,
        )
        if result.returncode:
            raise RuntimeError("Remote observation failed; no process state is inferred")
        record["remote"] = json.loads(result.stdout)
    atomic_json(control / "live-status.json", record)
    compact = {k: record[k] for k in (
        "utc", "instance_id", "local_watchdog_live", "seconds_before_preservation"
    )}
    compact["provider_status"] = provider["actual_status"] if provider else "absent"
    compact["active_compute_disk_estimate_usd"] = record["finance"]["active_estimate_usd"]
    if "remote" in record:
        compact.update(record["remote"])
    print(json.dumps(compact))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
