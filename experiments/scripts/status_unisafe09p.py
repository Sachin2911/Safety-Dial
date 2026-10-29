"""Read-only handover status for the two authorized UNISafe rentals."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.unisafe09_infra import instances, ssh_endpoint, ssh_options
from helpers.unisafe09p_infra import current_finances

REMOTE = r'''
import json,subprocess
from pathlib import Path
r=Path('/workspace/Safety-Dial');out=r/'outputs/unisafe09p'
result={'setup_ready':(r/'outputs/unisafe09/environment/ready').exists(),
        'models_verified':(r/'data/unisafe09/pretrained/transfer-ready.json').exists(),
        'stop':(out/'STOP').exists(),'pause':(out/'PAUSE').exists(),
        'attempts_by_stage':{},'live_workers':[]}
for p in (out/'attempts').glob('*/intent.json'):
    intent=json.loads(p.read_text());stage=intent['job']['stage']
    count=result['attempts_by_stage'].setdefault(stage,{'attempted':0,'finished':0,'invalid':0})
    count['attempted']+=1
    finished=p.parent/'process-result.json'
    if finished.exists():
        row=json.loads(finished.read_text());count['finished']+=1
        count['invalid']+=int(row['exit_code']!=0 or not row['summary_present'])
    else:
        account=p.parent/'accounting.json';row=json.loads(account.read_text()) if account.exists() else {}
        process=p.parent/'process.json';pid=json.loads(process.read_text())['pid'] if process.exists() else None
        result['live_workers'].append({'attempt':p.parent.name,'lane':intent['gpu_lane'],
            'pid':pid,'process_exists':pid is not None and Path(f'/proc/{pid}/cmdline').exists(),
            'phase':row.get('phase'),'controls':row.get('completed_controls',0)})
p=out/'watchdog-ready.json'
if p.exists():
    row=json.loads(p.read_text());pid=row['pid'];process=Path(f'/proc/{pid}/cmdline')
    result['remote_guard']={'pid':pid,'live':process.exists() and b'unisafe09p_watchdog.py' in process.read_bytes()}
p=out/'control/engineering-launch.json'
result['engineering_launch']=json.loads(p.read_text()) if p.exists() else None
for name in ['preparation-cached.log','public-wheel-preparation.log']:
    p=out/'logs'/name;result[name]=p.read_text()[-1700:] if p.exists() else None
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
result['gpu_processes']=gpu.stdout.strip()
print(json.dumps(result))
'''


def main():
    provider = {r["id"]: r for r in instances()}
    _, _, finance = current_finances()
    fields = ("id", "machine_id", "actual_status", "intended_status", "dph_total",
              "public_ipaddr", "ssh_host", "ssh_port", "ports")
    result = {"utc": utc(), "combined_finance": finance,
              "rentals": {str(i): {k: provider[i].get(k) for k in fields} if i in provider
                          else None for i in (57647787, 57690181)}}
    processes = []
    names = (b'unisafe09p_start_engineering.py', b'unisafe09_preserve_original_partition.py',
             b'unisafe09p_watchdog.py', b'unisafe09_watchdog.py')
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = p.read_bytes().split(b'\0')
        except OSError:
            continue
        for name in names:
            if any(a.endswith(b'/'+name) or a == name for a in args):
                processes.append({'pid': int(p.parent.name), 'script': name.decode()})
    result['local_processes'] = processes
    p = provider.get(57690181)
    if p and p['actual_status'] == 'running':
        host, port = ssh_endpoint(p)
        completed = subprocess.run(
            ['ssh', *ssh_options(port), f'root@{host}', '/venv/main/bin/python -'],
            input=REMOTE, text=True, capture_output=True, timeout=75,
        )
        result['remote_read_exit_code'] = completed.returncode
        if completed.returncode == 0:
            result['additional_host'] = json.loads(completed.stdout)
        else:
            result['remote_read_failed'] = True
    for name in ('09-unisafe-original-partition-complete.json',
                 '09-unisafe-original-partition-preservation.json',
                 '09-unisafe-parallel-model-transfer.json', '09-unisafe-parallel-A-launch.json'):
        p = ROOT/'progress/receipts'/name
        result[name] = json.loads(p.read_text()) if p.exists() else None
    atomic_json(ROOT/'outputs/unisafe09p/control/handover-status.json', result)
    display = dict(result)
    for key, value in display.items():
        if isinstance(value, dict) and 'job_ids' in value:
            display[key] = {k: v for k, v in value.items() if k != 'job_ids'}
    print(json.dumps(display, indent=2))


if __name__ == '__main__':
    main()
