"""Download public, revision-pinned sources for this fresh ManiGuard installation."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


SETUP = Path('/workspace/setup-maniguard-v3-20261030')
TP = Path('/workspace/Safety-Dial/third_party')
SOURCES = {
    'benchmark': {
        'repo': 'IDEAS-Lab-Northwestern/ManiGuard-Bench', 'type': 'dataset',
        'revision': 'af09438cebcd8f532e82cb5302701173ebb59f6e',
        'patterns': ['README.md', 'LICENSE', 'jar_transport/*'],
        'destination': TP / 'ManiGuard/outputs/lerobot_datasets/maniguard-bench',
    },
    'checkpoint': {
        'repo': 'IDEAS-Lab-Northwestern/pi05-base-datagen-v1-jar-joint-2cam-lora',
        'type': 'model', 'revision': '1d84eda070313a202a595e449fcf41a1a1e8a546',
        'patterns': ['7400/*', 'README.md', 'LICENSE*', '.gitattributes'],
        'destination': TP / 'checkpoints/jar',
    },
    'longfinger': {
        'repo': 'IDEAS-Lab-Northwestern/franka-panda-longfinger', 'type': 'dataset',
        'revision': '5f65ec0a715bc88ea865c2c69133b08b395c66bc',
        'patterns': None,
        'destination': SETUP / 'longfinger-staging',
    },
}


def digest(path, algorithm):
    hasher = hashlib.new(algorithm)
    if algorithm == 'sha1':
        hasher.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_source(name, spec):
    destination = spec['destination']
    destination.mkdir(parents=True, exist_ok=True)
    print('starting', name, flush=True)
    snapshot_download(spec['repo'], repo_type=spec['type'], revision=spec['revision'],
                      allow_patterns=spec['patterns'], local_dir=destination, max_workers=8)
    tree = {entry.path: entry for entry in HfApi().list_repo_tree(
        spec['repo'], repo_type=spec['type'], revision=spec['revision'], recursive=True,
        expand=True) if hasattr(entry, 'size')}
    files = {}
    for path in destination.rglob('*'):
        if not path.is_file() or '.cache' in path.parts or path.name == 'download-provenance.json':
            continue
        relative = path.relative_to(destination).as_posix()
        entry = tree[relative]
        lfs = getattr(entry, 'lfs', None)
        algorithm = 'sha256' if lfs else 'sha1'
        expected = (lfs.sha256 if hasattr(lfs, 'sha256') else lfs['sha256']) if lfs else entry.blob_id
        actual = digest(path, algorithm)
        if actual != expected:
            raise RuntimeError(f'{name}: hash mismatch for {relative}')
        files[relative] = {'bytes': path.stat().st_size, algorithm: actual}
    record = {'name': name, 'repo': spec['repo'], 'revision': spec['revision'],
              'file_count': len(files), 'bytes': sum(row['bytes'] for row in files.values()),
              'files': files, 'verified_utc': datetime.now(timezone.utc).isoformat()}
    (destination / 'download-provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    (SETUP / (name + '.downloaded')).write_text(json.dumps({k: v for k, v in record.items() if k != 'files'}) + '\n')
    print('verified', name, record['file_count'], record['bytes'], flush=True)


def download_assets():
    revision = '9f0d57d465726976ed98138d3f8b8ca3e2186775'
    archives = {
        'behavior-1k-assets-3.7.2rc1.zip': '3bd7cdc1a8c5621768c2aaaf57c12a024560ab4f49d0162e540e769d8b165e8b',
        'omnigibson-robot-assets.zip': '5e9fe726172c8b4beb4ba4e12dbf5cf6c5791553a05c52be661eb2d6a3d241b8',
    }
    paths = {}
    for filename, expected in archives.items():
        print('starting archive', filename, flush=True)
        path = Path(hf_hub_download('behavior-1k/zipped-datasets', filename,
                                   repo_type='dataset', revision=revision))
        actual = digest(path, 'sha256')
        if actual != expected:
            raise RuntimeError('Asset archive hash mismatch: ' + filename)
        paths[filename] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': actual}
        print('verified archive', filename, path.stat().st_size, flush=True)
    (SETUP / 'assets.downloaded').write_text(json.dumps({'revision': revision, 'archives': paths}, indent=2) + '\n')


with ThreadPoolExecutor(max_workers=4) as pool:
    jobs = [pool.submit(download_source, name, spec) for name, spec in SOURCES.items()]
    jobs.append(pool.submit(download_assets))
    errors = []
    for job in jobs:
        try:
            job.result()
        except Exception as exc:
            errors.append(type(exc).__name__ + ': ' + str(exc).split('\n')[0][:240])
    if errors:
        (SETUP / 'downloads.failed').write_text('\n'.join(errors) + '\n')
        raise SystemExit('Some public downloads failed; inspect downloads.failed.')
(SETUP / 'downloads.complete').write_text(datetime.now(timezone.utc).isoformat() + '\n')
