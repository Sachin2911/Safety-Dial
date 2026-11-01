"""Prepare a fresh authorized campaign host; local work only copies/transfers assets."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "outputs/lewm-campaign-20261031"
CONTROL = ROOT / "outputs/runpod-lewm-campaign-20261031"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_common():
    folder = TASK / "staging"
    folder.mkdir(exist_ok=True)
    sources = {}
    screen = ROOT / "outputs/lewm-screen-20261031/staging"
    for path in (screen / "assets").rglob("*"):
        if path.is_file():
            sources[str(path.relative_to(screen))] = path
    old = ROOT / "outputs/lewm-preset-expansion-20261030"
    sources.update(
        {
            "assets/world-model/tworoom/lewm_object.ckpt": old
            / "checkpoints/tworoom-lewm_object.ckpt",
            "assets/tworoom-safety.pt": old / "tworoom/motion/model.pt",
            "assets/tworoom-scalers.json": ROOT
            / "configs/lewm_portable_20261030/tworoom-scalers.json",
        }
    )
    for path in (old / "tworoom/inputs").glob("case-*.npz"):
        sources["assets/inputs/tworoom/" + path.name] = path
    provenance = []
    for name, source in sources.items():
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        sha = digest(source)
        if digest(target) != sha:
            raise RuntimeError("Common staged asset differs: " + name)
        provenance.append(
            {
                "asset": name,
                "source": str(source.relative_to(ROOT)),
                "sha256": sha,
                "bytes": source.stat().st_size,
            }
        )
    manifest = {
        "asset_sha256": {p["asset"]: p["sha256"] for p in provenance},
        "provenance": provenance,
    }
    (folder / "assets-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(screen / "requirements.txt", folder / "requirements.txt")
    return folder, manifest


def stage_common():
    with (TASK / "staging.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        folder = TASK / "staging"
        cached = folder / "assets-manifest.json"
        if cached.exists() and (folder / "requirements.txt").exists():
            return folder, json.loads(cached.read_text())
        return build_common()


REMOTE_GUARD = r"""import datetime as dt,json,os,signal,time
from pathlib import Path
cfg=json.loads(Path('/root/.config/lewm-campaign/lifecycle.json').read_text())
project=Path('/workspace/Safety-Dial')
if os.environ.get('RUNPOD_POD_ID')!=cfg['pod_id']:raise RuntimeError('Wrong pod')
deadline=dt.datetime.fromisoformat(cfg['deadline_utc']).timestamp()
def record(value):
 p=project/'environment/deadline-guard.json';p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix('.part');tmp.write_text(json.dumps(value,indent=2)+chr(10));tmp.replace(p)
while True:
 record({'pod_id':cfg['pod_id'],'heartbeat_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'deadline_utc':cfg['deadline_utc'],'status':'armed' if time.time()<deadline else 'stopping'})
 if time.time()>=deadline:
  for proc in Path('/proc').iterdir():
   if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
   try:
    command=(proc/'cmdline').read_bytes()
    if b'/opt/lewm' in command or b'/workspace/Safety-Dial' in command:os.kill(int(proc.name),signal.SIGTERM)
   except (FileNotFoundError,ProcessLookupError,PermissionError):pass
  record({'pod_id':cfg['pod_id'],'status':'experiment_process_stop_requested','provider_stop':'Independent local credential holder','deadline_utc':cfg['deadline_utc']})
  break
 time.sleep(min(15,deadline-time.time()))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", choices=("pusht", "navigation", "recovery"))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "campaign_control", ROOT / "experiments/scripts/lewm_campaign_pods.py"
    )
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    session = json.loads((CONTROL / args.lane / "session.json").read_text())
    common, manifest = stage_common()
    out = CONTROL / args.lane
    setup = (ROOT / "outputs/pusht-improvements-20261030-v1/staging/setup.sh").read_text()
    setup = setup.replace("pusht-repro-uv-cache", "lewm-campaign-uv-cache").replace(
        "activate-lewm-repro.sh", "activate-lewm-campaign.sh"
    )
    identity = 'export RUNPOD_POD_ID=$(python3 -c \'from pathlib import Path; print(next(v.decode().split("=",1)[1] for v in Path("/proc/1/environ").read_bytes().split(bytes([0])) if v.startswith(b"RUNPOD_POD_ID=")))\')'
    setup = setup.replace(
        ': "${RUNPOD_POD_ID:?Rented RunPod only}"',
        identity + '\n: "${RUNPOD_POD_ID:?Rented RunPod only}"',
    )
    setup = setup.replace(
        "source /opt/lewm/bin/activate\n", "source /opt/lewm/bin/activate\n" + identity + "\n"
    )
    setup_path = out / "setup.sh"
    setup_path.write_text(setup)
    design_path = out / "design.json"
    design_path.write_text(
        json.dumps(
            {
                "pod_id": session["id"],
                "deadline_utc": session["deadline_utc"],
                "lane": args.lane,
                "asset_sha256": manifest["asset_sha256"],
                "campaign_shared_cap_usd": 10,
            },
            indent=2,
        )
        + "\n"
    )
    archive = out / "environment-inputs.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in common.rglob("*"):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(common)))
        tar.add(setup_path, arcname="setup.sh")
        tar.add(design_path, arcname="design.json")
    bootstrap = r"""import json,os,subprocess,sys
from pathlib import Path
x=json.load(sys.stdin)
identity=next((v.decode().split('=',1)[1] for v in Path('/proc/1/environ').read_bytes().split(bytes([0])) if v.startswith(b'RUNPOD_POD_ID=')),None)
if identity!=x['pod_id']:raise RuntimeError('Wrong declared pod')
os.environ['RUNPOD_POD_ID']=identity
home=Path('/root/.config/lewm-campaign');home.mkdir(parents=True,exist_ok=True)
code=x.pop('guard_source');script=home/'deadline.py';script.write_text(code)
(home/'lifecycle.json').write_text(json.dumps(x)+chr(10))
root=Path('/workspace/Safety-Dial');(root/'environment').mkdir(parents=True,exist_ok=True)
if (root/'environment/deadline-process.json').exists():raise RuntimeError('Guard already armed; inspect before any duplicate setup')
with (root/'environment/deadline-guard.log').open('ab',buffering=0) as log:
 p=subprocess.Popen(['python3',str(script)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':p.pid,'start_ticks':Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],'pod_id':x['pod_id'],'deadline_utc':x['deadline_utc']}
(root/'environment/deadline-process.json').write_text(json.dumps(record)+chr(10));print(json.dumps(record))
"""
    ssh, host = control.ssh_args(args.lane)
    subprocess.run(
        ["ssh", *ssh, host, "python3 -c " + shlex.quote(bootstrap)],
        input=json.dumps(
            {
                "pod_id": session["id"],
                "deadline_utc": session["deadline_utc"],
                "lane": args.lane,
                "guard_source": REMOTE_GUARD,
            }
        ),
        text=True,
        check=True,
    )
    scp, host = control.ssh_args(args.lane, scp=True)
    subprocess.run(
        ["scp", *scp, str(archive), host + ":/workspace/lewm-campaign-environment.tar.gz"],
        check=True,
    )
    launch = r"""import json,subprocess,time
from pathlib import Path
r=Path('/workspace/Safety-Dial')
subprocess.run(['tar','--no-same-owner','-xzf','/workspace/lewm-campaign-environment.tar.gz','-C',str(r)],check=True)
if (r/'environment/setup-process.json').exists():raise RuntimeError('Never duplicate an existing setup')
with (r/'environment/setup.log').open('ab',buffering=0) as log:
 p=subprocess.Popen(['bash',str(r/'setup.sh')],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':p.pid,'start_ticks':Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],'epoch':time.time()}
(r/'environment/setup-process.json').write_text(json.dumps(record)+chr(10));print(json.dumps(record))
"""
    ssh, host = control.ssh_args(args.lane)
    subprocess.run(["ssh", *ssh, host, "python3 -c " + shlex.quote(launch)], check=True)
    print(
        json.dumps(
            {
                "lane": args.lane,
                "asset_files": len(manifest["asset_sha256"]),
                "archive_bytes": archive.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
