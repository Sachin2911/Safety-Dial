"""Exercise the two real download contracts without loading JAX or checkpoint tensors."""

import builtins
import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
from helpers.maniguard_policy_provenance import (
    load_policy_provenance,
    validate_provenance_record,
)

SOURCES = json.loads((ROOT / 'configs/maniguard/phase2/sources.json').read_text())['policies']


def _fixture(folder, family):
    """Tiny byte payloads with the paths and receipt layouts of the actual downloads."""
    spec = deepcopy(SOURCES[family])
    spec.pop('checkpoint_bytes', None)
    spec.pop('checkpoint_files', None)
    step = spec['step']
    names = [f'{step}/_CHECKPOINT_METADATA', f'{step}/params/_METADATA',
             f'{step}/params/_sharding', f'{step}/params/manifest.ocdbt',
             f'{step}/params/d/abc',
             f'{step}/params/ocdbt.process_0/d/def',
             f'{step}/assets/IDEAS-Lab-Northwestern/datagen-{family}-v1-joint-5cam/norm_stats.json',
             'README.md']
    files, planned = {}, []
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (name + '\n').encode()
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        files[name] = {'bytes': len(payload), 'sha256': digest}
        planned.append({'relative': name, 'remote': name, 'bytes': len(payload),
                        'sha256': digest if '/d/' in name else None,
                        'git_blob_sha1': hashlib.sha1(
                            f'blob {len(payload)}\0'.encode() + payload).hexdigest()})
    source = {'repo': spec['repo'], 'revision': spec['revision'], 'repo_type': 'model'}
    if family == 'clutter':
        record = {'plan': source | {'schema_version': 1, 'family': family, 'kind': 'policy',
                                   'destination': str(folder), 'files': planned,
                                   'bytes': sum(e['bytes'] for e in files.values()),
                                   'public_unauthenticated': True},
                  'files': files, 'complete': True, 'behavioral_evaluation_performed': False}
    else:
        record = source | {'files': files}
    _write(folder, record)
    return spec, record


def _write(folder, record):
    (folder / 'download-provenance.json').write_text(json.dumps(record))


@pytest.mark.parametrize('family', ['jar', 'stack', 'clutter'])
def test_accepts_both_actual_receipt_layouts_and_verifies_bytes(tmp_path, family):
    spec, record = _fixture(tmp_path, family)
    checkpoint, audit = load_policy_provenance(tmp_path, spec, family=family)
    assert checkpoint == tmp_path / spec['step']
    assert audit['verified_files'] == len(record['files'])
    assert audit['verified_bytes'] == sum(e['bytes'] for e in record['files'].values())
    assert audit['schema'] == ('phase2-plan-v1' if family == 'clutter' else 'legacy-flat')
    assert audit['receipt_sha256'] == hashlib.sha256(
        (tmp_path / 'download-provenance.json').read_bytes()).hexdigest()


@pytest.mark.parametrize('family,expected_count', [('jar', 25), ('stack', 27), ('clutter', 18)])
def test_actual_download_metadata_matches_pinned_contract(family, expected_count):
    path = ROOT / 'data/maniguard/policies' / family / 'download-provenance.json'
    if not path.is_file():
        pytest.skip('Local checkpoint download is optional for unit tests')
    record = json.loads(path.read_text())
    files = validate_provenance_record(record, SOURCES[family], family=family)
    assert len(files) == expected_count
    # Schema regression: the Clutter receipt has no top-level revision.
    assert ('revision' not in record) == (family == 'clutter')


@pytest.mark.parametrize('family', ['jar', 'stack', 'clutter'])
@pytest.mark.parametrize('key,replacement', [('repo', 'other/model'), ('revision', '0' * 40)])
def test_wrong_repository_or_revision_fails_before_reading_weights(tmp_path, family, key, replacement):
    spec, record = _fixture(tmp_path, family)
    record.get('plan', record)[key] = replacement
    _write(tmp_path, record)
    with pytest.raises(ValueError, match=key):
        load_policy_provenance(tmp_path, spec, family=family)


@pytest.mark.parametrize('family', ['jar', 'stack', 'clutter'])
@pytest.mark.parametrize('damage', ['missing', 'truncated', 'same_size_corruption'])
def test_partial_or_modified_checkpoint_rejected(tmp_path, family, damage):
    spec, _ = _fixture(tmp_path, family)
    shard = tmp_path / spec['step'] / 'params/ocdbt.process_0/d/def'
    payload = shard.read_bytes()
    if damage == 'missing':
        shard.unlink()
    elif damage == 'truncated':
        shard.write_bytes(payload[:-1])
    else:
        shard.write_bytes(b'X' * len(payload))
    with pytest.raises(ValueError, match='missing|wrong size|SHA256 mismatch'):
        load_policy_provenance(tmp_path, spec, family=family)


@pytest.mark.parametrize('damage', ['incomplete', 'missing_receipt_file', 'missing_plan_file',
                                    'plan_size', 'plan_digest', 'wrong_family', 'wrong_kind',
                                    'wrong_schema', 'conflicting_revision', 'duplicate_file'])
def test_structured_download_plan_is_not_a_bypass(tmp_path, damage):
    spec, record = _fixture(tmp_path, 'clutter')
    if damage == 'incomplete':
        record['complete'] = False
    elif damage == 'missing_receipt_file':
        del record['files'][record['plan']['files'][5]['relative']]
    elif damage == 'missing_plan_file':
        record['plan']['files'].pop()
    elif damage == 'plan_size':
        record['plan']['files'][0]['bytes'] += 1
    elif damage == 'plan_digest':
        record['plan']['files'][4]['sha256'] = '0' * 64
    elif damage == 'wrong_family':
        record['plan']['family'] = 'jar'
    elif damage == 'wrong_kind':
        record['plan']['kind'] = 'benchmark'
    elif damage == 'wrong_schema':
        record['plan']['schema_version'] = 2
    elif damage == 'conflicting_revision':
        record['revision'] = '0' * 40
    else:
        record['plan']['files'].append(record['plan']['files'][0])
    _write(tmp_path, record)
    with pytest.raises(ValueError):
        load_policy_provenance(tmp_path, spec, family='clutter')


def test_structured_git_blob_digest_checked_even_with_matching_receipt_sha256(tmp_path):
    spec, record = _fixture(tmp_path, 'clutter')
    record['plan']['files'][0]['git_blob_sha1'] = '0' * 40
    _write(tmp_path, record)
    with pytest.raises(ValueError, match='upstream Git digest'):
        load_policy_provenance(tmp_path, spec, family='clutter')


@pytest.mark.parametrize('name', ['../outside', '/tmp/outside', 'a/../outside', 'a//b', 'a\\b'])
def test_manifest_cannot_escape_or_alias_paths(tmp_path, name):
    spec, record = _fixture(tmp_path, 'jar')
    record['files'][name] = {'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()}
    _write(tmp_path, record)
    with pytest.raises(ValueError, match='Unsafe provenance path'):
        load_policy_provenance(tmp_path, spec, family='jar')


@pytest.mark.parametrize('directory_link', [False, True])
def test_file_and_parent_symlinks_rejected(tmp_path, directory_link):
    folder = tmp_path / 'policy'
    spec, _ = _fixture(folder, 'jar')
    target = folder / spec['step'] / 'params/d'
    if not directory_link:
        target /= 'abc'
    moved = tmp_path / 'elsewhere'
    target.rename(moved)
    target.symlink_to(moved, target_is_directory=directory_link)
    with pytest.raises(ValueError, match='symbolic link'):
        load_policy_provenance(folder, spec, family='jar')


def test_manifest_without_checkpoint_or_pinned_inventory_count_rejected(tmp_path):
    spec, record = _fixture(tmp_path, 'clutter')
    with pytest.raises(ValueError, match='checkpoint_files'):
        validate_provenance_record(record, spec | {'checkpoint_files': 99}, family='clutter')
    with pytest.raises(ValueError, match='checkpoint_bytes'):
        validate_provenance_record(record, spec | {'checkpoint_bytes': 99}, family='clutter')
    record['files'] = {'README.md': record['files']['README.md']}
    with pytest.raises(ValueError, match='complete checkpoint inventory'):
        validate_provenance_record(record, spec, family='clutter')


def test_server_validates_before_importing_model_stack(tmp_path, monkeypatch):
    config = tmp_path / 'configs/maniguard/phase2/sources.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'policies': {'clutter': SOURCES['clutter']}}))
    module_spec = importlib.util.spec_from_file_location(
        'provenance_server_test', ROOT / 'experiments/scripts/maniguard_serve.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['serve', '--phase2', '--family', 'clutter'])
    original_import = builtins.__import__

    def forbid_model_stack(name, *args, **kwargs):
        if name.split('.')[0] in ('jax', 'maniguard', 'openpi'):
            raise AssertionError('Model stack imported before provenance was validated')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', forbid_model_stack)
    with pytest.raises(ValueError, match='ordinary local paths'):
        module.main()
