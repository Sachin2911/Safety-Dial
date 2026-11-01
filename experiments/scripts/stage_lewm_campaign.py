"""Transfer a versioned source snapshot, run rented-GPU gates, launch one frozen stage."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
TESTS = [
    "test_lewm_proposal_adapter.py",
    "test_cem_refinements.py",
    "test_lewm_proposal_pilot.py",
    "test_lewm_residual.py",
    "test_lewm_screen.py",
    "test_lewm_navigation_evolution.py",
    "test_lewm_push_evolution.py",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lane", choices=("pusht", "navigation", "recovery"))
    p.add_argument("--version", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--entrypoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tests", nargs="+", default=TESTS)
    p.add_argument("--expected-tests", type=int, default=37)
    p.add_argument("--test-basetemp")
    p.add_argument("--gate-only", action="store_true")
    p.add_argument("--entry-args", default="")
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location(
        "control", ROOT / "experiments/scripts/lewm_campaign_pods.py"
    )
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    target = control.lane_dir(args.lane)
    version = target / args.version
    version.mkdir(exist_ok=False)
    sources = []
    for group in ("helpers", "scripts"):
        sources.extend(
            path
            for path in (ROOT / "experiments" / group).glob("*.py")
            if "observed" not in path.name
        )
    sources += [ROOT / "experiments/tests" / name for name in args.tests]
    sources += list((ROOT / "configs/lewm_campaign_20261031").rglob("*.json"))
    sources += [ROOT / "configs/lewm_screen_20261031/protocol.json"]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    reference = ROOT / "outputs/lewm-screen-20261031/remote/reference/lewm18_planner.py"
    hashes["reference/lewm18_planner.py"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    archive = version / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sources:
            tar.add(path, arcname=str(path.relative_to(ROOT)))
        tar.add(reference, arcname="reference/lewm18_planner.py")
    (version / "manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    scp, host = control.ssh_args(args.lane, scp=True)
    remote_archive = "/workspace/lewm-campaign-" + args.version + ".tar.gz"
    subprocess.run(["scp", *scp, str(archive), host + ":" + remote_archive], check=True)
    subprocess.run(
        [
            "scp",
            *scp,
            str(ROOT / "outputs/runpod-lewm-adapter-20261030/ruff.toml"),
            host + ":/workspace/Safety-Dial/ruff.toml",
        ],
        check=True,
    )
    metadata = {
        "version": args.version,
        "protocol": args.protocol,
        "entrypoint": args.entrypoint,
        "out": args.out,
        "expected_protocol_sha256": hashes[args.protocol],
        "archive": remote_archive,
        "tests": args.tests,
        "expected_tests": args.expected_tests,
        "test_basetemp": args.test_basetemp,
        "gate_only": args.gate_only,
        "entry_args": args.entry_args,
    }
    remote = r"""import datetime as dt,json,subprocess,time,xml.etree.ElementTree as ET
from pathlib import Path
r=Path('/workspace/Safety-Dial')
end=time.time()+900
while not (r/'environment/ready.json').exists():
 if time.time()>end:raise RuntimeError('Environment setup readiness timed out; inspect exact setup PID')
 process=json.loads((r/'environment/setup-process.json').read_text());proc=Path('/proc')/str(process['pid'])
 if not proc.exists():raise RuntimeError('Setup worker ended without ready.json')
 time.sleep(3)
for path in (r/'environment/workers').glob('*.json'):
 x=json.loads(path.read_text());proc=Path('/proc')/str(x['pid'])
 if proc.exists():
  stat=(proc/'stat').read_text().split()
  if stat[2]!='Z' and stat[21]==str(x['start_ticks']):raise RuntimeError('Existing stage worker remains live; do not overwrite source')
folder=r/'environment'/M['version'];folder.mkdir(exist_ok=False)
subprocess.run(['cp',M['archive'],str(folder/'source.tar.gz')],check=True)
subprocess.run(['tar','--no-same-owner','-xzf',M['archive'],'-C',str(r)],check=True)
activation=chr(10).join(['source /workspace/activate-lewm-campaign.sh','export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace/Safety-Dial/experiments','cd /workspace/Safety-Dial',''])
command=activation+'python -m pytest -q '+' '.join('experiments/tests/'+name for name in M['tests'])+' --junitxml='+str(folder/'tests.xml')
if M.get('test_basetemp'):
 import shlex
 command+=' --basetemp='+shlex.quote(M['test_basetemp'])
command+=' > '+str(folder/'tests.log')+' 2>&1'+chr(10)
subprocess.run(['bash','-lc',command],check=True)
suite=ET.parse(folder/'tests.xml').getroot()
if len(list(suite.iter('testcase')))!=M['expected_tests'] or any(list(suite.iter(k)) for k in ('failure','error','skipped')):raise RuntimeError('GPU software gate incomplete')
files=['experiments/helpers/lewm_navigation_evolution.py','experiments/helpers/lewm_push_evolution.py','experiments/scripts/run_lewm_navigation_diagnostic.py','experiments/scripts/run_lewm_push_diagnostic.py','experiments/tests/test_lewm_navigation_evolution.py','experiments/tests/test_lewm_push_evolution.py']
command=activation+'ruff check '+' '.join(files)+' > '+str(folder/'ruff.log')+' 2>&1'+chr(10)
subprocess.run(['bash','-lc',command],check=True)
import hashlib
if hashlib.sha256((r/M['protocol']).read_bytes()).hexdigest()!=M['expected_protocol_sha256']:raise RuntimeError('Prospective protocol changed')
if M['gate_only']:
 value={'software_gate_passed':True,'tests':M['expected_tests'],'pod_id':json.loads((r/'environment/ready.json').read_text())['pod_id'],'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
 (folder/'gate.json').write_text(json.dumps(value,indent=2)+chr(10));print(json.dumps(value),flush=True);raise SystemExit(0)
d=json.loads((r/M['protocol']).read_text())
if dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=25)>=dt.datetime.fromisoformat(d.get('deadline_utc',d.get('evaluation',{}).get('deadline_utc'))):raise RuntimeError('Insufficient stage and preservation time')
workers=r/'environment/workers';workers.mkdir(exist_ok=True)
record_path=workers/(M['version']+'.json')
if record_path.exists() or (r/M['out']).exists():raise RuntimeError('Never duplicate an executed stage')
worker=folder/'worker.sh'
command=activation+'timeout --signal=TERM --kill-after=15s 1800s python -u '+M['entrypoint']+' '+M['entry_args']+' --protocol '+M['protocol']+' --out '+M['out']+chr(10)
worker.write_text(command)
(r/'results').mkdir(exist_ok=True)
with (folder/'worker.log').open('ab',buffering=0) as log:
 process=subprocess.Popen(['bash',str(worker)],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={**M,'pid':process.pid,'start_ticks':Path('/proc/'+str(process.pid)+'/stat').read_text().split()[21],'launched_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'pod_id':d.get('execution_pod_id',json.loads((r/'environment/ready.json').read_text())['pod_id']),'gpu_tests_passed':M['expected_tests']}
record_path.write_text(json.dumps(value,indent=2)+chr(10));print(json.dumps(value),flush=True)
"""
    remote = "M=" + repr(metadata) + "\n" + remote
    ssh, host = control.ssh_args(args.lane)
    subprocess.run(["ssh", *ssh, host, "python3 -c " + shlex.quote(remote)], check=True)


if __name__ == "__main__":
    main()
