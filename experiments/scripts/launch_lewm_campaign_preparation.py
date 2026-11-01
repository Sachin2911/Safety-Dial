"""Register the frozen six-lock final preparer using its bound primary deadline."""

import argparse
import hashlib
import importlib.util
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lane", choices=("navigation",))
    p.add_argument("--name", required=True)
    p.add_argument("--gate", required=True)
    p.add_argument("--tests", type=int, required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--entrypoint", required=True)
    p.add_argument("--entry-args", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location(
        "control", ROOT / "experiments/scripts/lewm_campaign_pods.py"
    )
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    metadata = vars(args).copy()
    metadata["orchestration_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata["expected_protocol_sha256"] = hashlib.sha256(
        (ROOT / args.protocol).read_bytes()
    ).hexdigest()
    remote = r"""import datetime as dt,hashlib,json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
r=Path('/workspace/Safety-Dial')
gate=r/'environment'/M['gate'];suite=ET.parse(gate/'tests.xml').getroot()
if len(list(suite.iter('testcase')))!=M['tests'] or any(list(suite.iter(k)) for k in ('failure','error','skipped')):raise RuntimeError('Existing software gate did not pass')
if 'All checks passed!' not in (gate/'ruff.log').read_text():raise RuntimeError('Existing lint gate missing')
p=r/M['protocol'];d=json.loads(p.read_text());evaluation=d.get('evaluation',d)
if hashlib.sha256(p.read_bytes()).hexdigest()!=M['expected_protocol_sha256']:raise RuntimeError('Frozen protocol differs')
if d.get('schema')!='lewm-navigation-joint-final-preparation-v2':raise RuntimeError('Only the frozen six-lock preparation wrapper is supported')
if M['entrypoint']!='experiments/scripts/prepare_lewm_navigation_joint_final_cases_v2.py':raise RuntimeError('Wrong preparation entrypoint')
ref=d['primary_preparation_protocol'];primary_path=(r/ref['path']).resolve()
if not primary_path.is_relative_to(r) or hashlib.sha256(primary_path.read_bytes()).hexdigest()!=ref['sha256']:raise RuntimeError('Bound primary preparation differs')
primary=json.loads(primary_path.read_text());deadline=primary['deadline_utc']
if deadline!='2026-10-31T07:23:14+00:00' or primary['allowed_pod_ids']!=['vukovc3t4o1dui']:raise RuntimeError('Preparation authority or cutoff differs')
if primary['maximum_preparation_resets']!=16 or primary['maximum_preparation_actions']!=0 or primary['maximum_native_render_frame_calls']!=32:raise RuntimeError('Preparation budget differs')
if M['out']!='results/navigation-final-cases-v1':raise RuntimeError('Fresh canonical final-case namespace required')
if dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=20)>=dt.datetime.fromisoformat(deadline):raise RuntimeError('Preservation reserve')
for declaration in (d,evaluation):
 for fields in ('source_sha256','asset_sha256'):
  for name,sha in declaration.get(fields,{}).items():
   with (r/name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
   if actual!=sha:raise RuntimeError('Frozen dependency changed: '+name)
ready=json.loads((r/'environment/ready.json').read_text());pod=ready['pod_id']
if pod!='vukovc3t4o1dui':raise RuntimeError('Only the original Nav preparation host is allowed')
if evaluation.get('execution_pod_id',pod)!=pod or pod not in d.get('allowed_pod_ids',[pod]):raise RuntimeError('Wrong pod identity')
workers=r/'environment/workers';workers.mkdir(exist_ok=True)
for path in workers.glob('*.json'):
 x=json.loads(path.read_text());proc=Path('/proc')/str(x['pid'])
 if proc.exists():
  stat=(proc/'stat').read_text().split()
  if stat[2]!='Z' and stat[21]==str(x['start_ticks']):raise RuntimeError('A previous stage worker remains live: '+path.name)
record=workers/(M['name']+'.json')
if record.exists() or (r/M['out']).exists():raise RuntimeError('Never duplicate a stage or replay outcomes')
remaining=int((dt.datetime.fromisoformat(deadline)-dt.datetime.now(dt.timezone.utc)).total_seconds())-600
command=chr(10).join(['set -eu','source /workspace/activate-lewm-campaign.sh','export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace/Safety-Dial/experiments','cd /workspace/Safety-Dial','timeout --signal=TERM --kill-after=15s '+str(remaining)+'s python -u '+M['entrypoint']+' '+M['entry_args']+' --protocol '+M['protocol']+' --out '+M['out'],''])
script=r/'environment'/(M['name']+'.sh');script.write_text(command)
with (r/'environment'/(M['name']+'.log')).open('ab',buffering=0) as log:
 worker=subprocess.Popen(['bash',str(script)],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={**M,'pid':worker.pid,'start_ticks':Path('/proc/'+str(worker.pid)+'/stat').read_text().split()[21],'pod_id':pod,'launched_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
record.write_text(json.dumps(value,indent=2)+chr(10));print(json.dumps(value),flush=True)
"""
    remote = "M=" + repr(metadata) + "\n" + remote
    ssh, host = control.ssh_args(args.lane)
    subprocess.run(["ssh", *ssh, host, "python3 -c " + shlex.quote(remote)], check=True)


if __name__ == "__main__":
    main()
