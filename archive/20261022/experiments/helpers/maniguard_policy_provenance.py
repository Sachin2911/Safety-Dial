"""Validate pinned legacy and phase-two policy receipts before loading a model.

Receipts are local download evidence, not signatures from the model author. Structured
receipts additionally retain the upstream file inventory and digests. All declared files
are hashed at startup so an interrupted or subsequently changed checkpoint is refused.
"""

import hashlib
import json
import re
from pathlib import Path, PurePosixPath


def _relative(name):
    if (not isinstance(name, str) or not name or '\\' in name
            or PurePosixPath(name).is_absolute()
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError(f'Unsafe provenance path: {name!r}')
    return name


def _digest(value, length):
    return isinstance(value, str) and re.fullmatch(rf'[0-9a-f]{{{length}}}', value) is not None


def validate_provenance_record(record, spec, *, family):
    """Return the validated file inventory without opening checkpoint files."""
    if not isinstance(record, dict) or not isinstance(record.get('files'), dict):
        raise TypeError('Policy provenance requires a file inventory')
    step = spec.get('step')
    if not isinstance(step, str) or not step.isdecimal():
        raise ValueError('Policy checkpoint step must be a decimal directory name')
    structured = 'plan' in record
    source = record.get('plan') if structured else record
    if not isinstance(source, dict):
        raise TypeError('Policy provenance plan must be an object')
    for key in ('repo', 'revision'):
        if not isinstance(spec.get(key), str) or source.get(key) != spec[key]:
            raise ValueError(f'Policy provenance {key} does not match the study pin')
        if structured and key in record and record[key] != source[key]:
            raise ValueError(f'Conflicting policy provenance {key}')
    if source.get('repo_type') != 'model':
        raise ValueError('Policy provenance must identify a model repository')
    files = record['files']
    for name, entry in files.items():
        _relative(name)
        if (not isinstance(entry, dict) or type(entry.get('bytes')) is not int
                or entry['bytes'] < 0 or not _digest(entry.get('sha256'), 64)):
            raise ValueError(f'Invalid policy file metadata: {name}')
    checkpoint_files = {name: entry for name, entry in files.items()
                        if name.startswith(step + '/')}
    required = {f'{step}/_CHECKPOINT_METADATA', f'{step}/params/_METADATA',
                f'{step}/params/manifest.ocdbt', f'{step}/params/_sharding'}
    if (not required.issubset(checkpoint_files)
            or not any(name.startswith(f'{step}/assets/') and name.endswith('/norm_stats.json')
                       for name in checkpoint_files)
            or not any(name.startswith(f'{step}/params/d/') for name in checkpoint_files)):
        raise ValueError('Policy provenance lacks a complete checkpoint inventory')
    for key, actual in (('checkpoint_files', len(checkpoint_files)),
                        ('checkpoint_bytes', sum(e['bytes'] for e in checkpoint_files.values()))):
        if key in spec and spec[key] != actual:
            raise ValueError(f'Policy inventory {key} does not match the study pin')
    if structured:
        if (source.get('schema_version') != 1 or source.get('family') != family
                or source.get('kind') != 'policy' or record.get('complete') is not True):
            raise ValueError('Policy download plan is unsupported, incomplete or for another family')
        planned = source.get('files')
        if not isinstance(planned, list) or not planned:
            raise ValueError('Policy download plan requires its complete file inventory')
        seen = set()
        for entry in planned:
            if not isinstance(entry, dict):
                raise TypeError('Invalid planned policy file')
            name = _relative(entry.get('relative'))
            if name in seen or entry.get('remote') != name:
                raise ValueError('Duplicate or remapped policy file in download plan')
            seen.add(name)
            receipt = files.get(name)
            if receipt is None or entry.get('bytes') != receipt['bytes']:
                raise ValueError('Policy receipt does not match the planned file inventory')
            sha256 = entry.get('sha256')
            if sha256 is not None:
                if not _digest(sha256, 64) or sha256 != receipt['sha256']:
                    raise ValueError('Policy file SHA256 differs from its download plan')
            elif not _digest(entry.get('git_blob_sha1'), 40):
                raise ValueError('Planned policy file lacks an upstream digest')
        if seen != set(files) or source.get('bytes') != sum(e['bytes'] for e in files.values()):
            raise ValueError('Policy receipt does not match the complete download plan')
    return files


def load_policy_provenance(folder, spec, *, family):
    """Check the receipt and every file, returning the checkpoint path and audit metadata.

    Full SHA256 verification intentionally adds disk reads at each service startup. It
    makes no network calls, changes no receipts and does not load a policy or simulator.
    """
    folder = Path(folder)
    receipt_path = folder / 'download-provenance.json'
    if folder.is_symlink() or not folder.is_dir() or receipt_path.is_symlink():
        raise ValueError('Policy folder and receipt must be ordinary local paths')
    try:
        receipt_bytes = receipt_path.read_bytes()
        record = json.loads(receipt_bytes)
    except (OSError, ValueError) as exc:
        raise ValueError('Policy download provenance is missing or unreadable') from exc
    files = validate_provenance_record(record, spec, family=family)
    plan = record.get('plan')
    planned = {entry['relative']: entry for entry in plan['files']} if plan else {}
    for name, entry in files.items():
        path = folder / name
        if any(parent.is_symlink() for parent in (path, *path.parents)
               if parent != folder.parent and folder.parent in parent.parents):
            raise ValueError(f'Policy file uses a symbolic link: {name}')
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            raise ValueError(f'Policy checkpoint file is missing or has the wrong size: {name}')
        sha256 = hashlib.sha256()
        blob = hashlib.sha1()
        blob.update(f"blob {entry['bytes']}\0".encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                sha256.update(block)
                if name in planned and planned[name].get('sha256') is None:
                    blob.update(block)
        if sha256.hexdigest() != entry['sha256']:
            raise ValueError(f'Policy checkpoint SHA256 mismatch: {name}')
        if (name in planned and planned[name].get('sha256') is None
                and blob.hexdigest() != planned[name]['git_blob_sha1']):
            raise ValueError(f'Policy checkpoint upstream Git digest mismatch: {name}')
    checkpoint = folder / spec['step']
    return checkpoint, {
        'schema': 'phase2-plan-v1' if plan else 'legacy-flat',
        'receipt_sha256': hashlib.sha256(receipt_bytes).hexdigest(),
        'verified_files': len(files),
        'verified_bytes': sum(entry['bytes'] for entry in files.values()),
        'verification': 'all_declared_files_sha256_and_available_upstream_digests',
    }
