"""Archive live pilot evidence and upload it to the private handoff dataset repository.

Kit cache seeds and per-slot worker folders are excluded; attempt records keep the
parsed renderer GPU provenance. The archive is refused if any file contains the value
of a configured credential. Uploads go to ``pilot-progress/`` and never replace the
pinned handoff files.
"""

import argparse
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / 'configs/maniguard/phase2/handoff-20261018.json'
RUN = ROOT / 'runs/maniguard/phase2/pilot-v1'
EXTRA = ('configs/maniguard/pilot-migration-20261018.json',)
SECRET_KEYS = ('HF_TOKEN', 'GITHUB_TOKEN', 'WANDB_API_KEY', 'CONTAINER_API_KEY')


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def included(path):
    relative = path.relative_to(RUN)
    return relative.parts[:1] != ('workers',) and 'appdata' not in relative.parts


def scan_for_secrets(paths):
    secrets = [os.environ[key].encode() for key in SECRET_KEYS if len(os.environ.get(key, '')) >= 8]
    for path in paths:
        data = path.read_bytes()
        if any(secret in data for secret in secrets):
            raise RuntimeError(f'Credential value found in {path}; refusing to archive')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('/workspace/maniguard-backups'))
    parser.add_argument('--no-upload', action='store_true')
    args = parser.parse_args()
    receipt = json.loads(RECEIPT.read_text())
    files = sorted(path for path in RUN.rglob('*') if path.is_file() and included(path))
    files += [ROOT / name for name in EXTRA]
    scan_for_secrets(files)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    name = f'pilot-v1-{stamp}-{args.label}.tar.gz'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / name
    with tarfile.open(archive, 'w:gz') as bundle:
        for path in files:
            bundle.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    state = RUN / ('lanes.json' if (RUN / 'lanes.json').exists() else 'coordinator.json')
    manifest = {'schema': 'maniguard-pilot-progress-backup-v1', 'created_at': time.time(),
                'archive': name, 'bytes': archive.stat().st_size, 'sha256': digest(archive),
                'files': len(files), 'state_file': str(state.relative_to(ROOT)),
                'state_content_sha256': json.loads(state.read_text()).get('sha256'),
                'excluded': ['runs/maniguard/phase2/pilot-v1/workers (Kit caches and logs)',
                             'any appdata directory'],
                'repository_commit': os.popen(f'git -C {ROOT} rev-parse HEAD').read().strip()}
    (args.output_dir / (name + '.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    if not args.no_upload:
        api = HfApi()
        for local, remote in ((archive, name), (args.output_dir / (name + '.json'), name + '.json')):
            api.upload_file(path_or_fileobj=str(local), path_in_repo='pilot-progress/' + remote,
                            repo_id=receipt['repo_id'], repo_type=receipt['repo_type'],
                            commit_message=f'Pilot progress backup {stamp} ({args.label})')
        manifest['revision'] = api.dataset_info(receipt['repo_id']).sha
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
