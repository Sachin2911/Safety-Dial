#!/usr/bin/env python3
"""Plan or upload exactly one completed follow-up run to its existing private archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from helpers.hfStore import HFStore
from helpers.runManifest import file_sha256
from huggingface_hub.hf_api import RepoFile


def inventory(run):
    files={}
    for p in sorted(run.rglob('*')):
        if not p.is_file() or p.name in ('hf_upload.json','hf_upload.json.tmp'):
            continue
        if p.is_symlink() or p.name=='.env' or p.name.endswith(('.pending.json','.tmp')):
            raise ValueError(f'Unfinished or unexpected archive member: {p.name}')
        body=p.read_bytes()
        files[str(p.relative_to(run))]=dict(bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),
            git_blob_sha1=hashlib.sha1(f'blob {len(body)}\0'.encode()+body).hexdigest())
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    run=args.run.resolve()
    if run.parent!=ROOT/'runs' or not run.name.startswith('walker2d-evo-s6-'):
        raise ValueError('Only Stage 6 run directories are accepted')
    result=json.loads((run/'study/completion.json').read_text())
    assert result['complete']
    stage='stage6-'+result['phase']
    docs=ROOT/'docs/evoPlan/results'/stage/run.name
    audit=json.loads((docs/'integrity_audit.json').read_text())
    assert audit['passed'] and audit['completion_sha256']==file_sha256(run/'study/completion.json')
    report=run/'report'
    report.mkdir(exist_ok=True)
    for name in ['REPORT.md','INTERPRETATION.md','summary_table.json','development_choice.json','integrity_audit.json',
                 'development_failures.png','development_failures.pdf']:
        if (docs/name).exists():
            shutil.copy2(docs/name,report/name)
    shutil.copy2(ROOT/'experiments/scripts/evo_followup_report.py',report/'report_generator.py')
    assert file_sha256(report/'report_generator.py')==audit['report_script_sha256']
    files=inventory(run)
    plan=dict(repo_id='Sachioster/safetydial-walker2d',path='evo/'+run.name,
              files=len(files),bytes=sum(x['bytes'] for x in files.values()),
              inventory_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
              contains='Simulator trajectories, policy checkpoints, query receipts, code snapshots, and reports')
    (docs/'archive_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2),flush=True)
    if not args.execute:
        return
    store=HFStore(namespace='Sachioster')
    assert store.repo_id('walker2d')==plan['repo_id']
    assert store.api.repo_info(plan['repo_id'],repo_type='model').private
    revision=store.upload_run('walker2d','evo',run,run_id=run.name)
    assert files==inventory(run), 'Local payload changed during upload'
    prefix=plan['path']+'/'
    remote={f.path[len(prefix):]:f for f in store.api.list_repo_tree(plan['repo_id'],
        path_in_repo=plan['path'],repo_type='model',revision=revision,recursive=True) if isinstance(f,RepoFile)}
    assert set(remote)==set(files), 'Remote archive membership differs'
    for name,record in files.items():
        f=remote[name]
        assert f.size==record['bytes']
        assert (f.lfs.sha256==record['sha256'] if f.lfs else f.blob_id==record['git_blob_sha1']), name
    receipt=dict(**plan,revision=revision,private=True,all_remote_file_hashes_verified=True)
    (docs/'archive_verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
    shutil.copy2(run/'hf_upload.json',docs/'hf_upload.json')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    main()
