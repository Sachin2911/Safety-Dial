"""Synthetic main-reader evidence only: no reserved data or simulator/model services."""

import hashlib
import json
import shutil
import sys
import zlib
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from helpers import maniguard_main_results as audit
from helpers.maniguard_adaptation import Frame
from helpers.maniguard_main_protocol import controller_resource_contract, episode_resource_envelope
from helpers.maniguard_phase2_adapter import Phase2Adapter
from helpers.maniguard_phase2_model import genome_hash, identity_genome, save_model
from helpers.maniguard_phase2_optim import record_hash


def signed(value):
    return {**value, 'sha256': record_hash({key: item for key, item in value.items() if key != 'sha256'})}


def write(path, value, *, sign=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(signed(value) if sign else value) + '\n')


def lines(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_episode(tmp_path, monkeypatch, mode='learned'):
    root, output = tmp_path / 'root', tmp_path / 'root' / 'episode'
    output.mkdir(parents=True)
    profile = {'learned': 'phase2_learned', 'fixed': 'phase2_fixed',
               'contextual': 'phase2_contextual'}[mode]
    genome = identity_genome()
    save_model(output / 'actor.npz', genome)
    shutil.copyfile(output / 'actor.npz', root / 'selected.npz')
    choices = [0, 0, 1, 0] if mode == 'fixed' else None
    resources = controller_resource_contract(profile, choices=choices)
    pin = {'config': 'synthetic-policy', 'revision': 'b' * 40, 'repo': 'synthetic/repo', 'step': '1'}
    job = {'episode_id': 'synthetic/run/jar/task_0000/base/p1/s0', 'job_id': 'synthetic/job',
           'family': 'jar', 'scene': 'task_0000/base', 'base_scene': 'task_0000', 'condition': 'base',
           'purpose': 'test', 'partition': 'test', 'exposure_role': 'development', 'seed': 1,
           'sampling_seed': 0, 'action_horizon': 12, 'stochastic': False, 'checkpoint': pin,
           'method': mode, 'run_id': mode + '-run', 'run_kind': 'trained' if mode == 'learned' else 'fixed',
           'actor_sha256': genome_hash(genome) if mode == 'learned' else None,
           'resource_contract': resources, 'resource_contract_sha256': record_hash(resources),
           'resource_envelope': episode_resource_envelope(12, resources)}
    if mode == 'learned':
        job.update(actor={'path': 'selected.npz', 'sha256': sha(root / 'selected.npz')},
                   controller_sha256=genome_hash(genome))
    else:
        controller = {'kind': mode, 'deterministic': True, 'resource_contract': resources}
        if mode == 'fixed':
            controller['choices'] = choices
        write(root / 'controller.json', controller, sign=True)
        job.update(controller={'path': 'controller.json', 'sha256': sha(root / 'controller.json')},
                   controller_sha256=json.loads((root / 'controller.json').read_text())['sha256'])
    encoder = json.loads((ROOT / 'configs/maniguard/phase2/encoder.json').read_text())
    configs = {'encoder': encoder, 'sources': {'maniguard': {'revision': 'c' * 40}},
               'completion': {'synthetic_contract': True}}
    references = []
    for name, value in configs.items():
        path = root / f'configs/maniguard/phase2/{name}.json'
        write(path, value)
        references.append({'path': str(path.relative_to(root)), 'sha256': sha(path)})
    source_names = ['maniguard_main_results.py', 'maniguard_phase2_results.py',
                    'maniguard_phase2_adapter.py', 'maniguard_phase2_runtime.py',
                    'maniguard_runtime.py', 'maniguard_phase2_features.py',
                    'maniguard_phase2_model.py', 'maniguard_client.py', 'maniguard_main_episode_meter.py',
                    'maniguard_main_physics.py', 'maniguard_aac_runtime.py',
                    'maniguard_aac_sampling.py', 'maniguard_phase2_aac.py']
    implementation = []
    for name in source_names:
        relative = 'experiments/helpers/' + name
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, path)
        implementation.append({'path': relative, 'sha256': sha(path)})
    for relative in ('experiments/helpers/maniguard_main_episode_runtime.py',
                     'experiments/helpers/maniguard_main_episode_source.py',
                     'experiments/helpers/synthetic_physics_driver.py',
                     'experiments/scripts/maniguard_serve.py',
                     'experiments/helpers/maniguard_policy.py',
                     'experiments/helpers/maniguard_parallel_policy.py',
                     'experiments/helpers/maniguard_policy_provenance.py'):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# declared synthetic implementation fixture\n')
        implementation.append({'path': relative, 'sha256': sha(path)})
    planner = root / 'planner.py'
    planner.write_text('# synthetic planner fixture\n')
    protocol = signed({'status': 'frozen', 'synthetic_fixture': True, 'input_files': references,
        'planner_source': {'path': 'planner.py', 'sha256': sha(planner)},
        'spec': {'preprocessing_files': [], 'implementation_files': implementation}})
    bindings = {'protocol_sha256': protocol['sha256'], 'job_sha256': record_hash(job),
                'output': str(output)}
    write(output / 'main_job.json', {'schema': 'maniguard-main-job-v1', **bindings, 'job': job}, sign=True)
    resolved = {key: job[key] for key in ('family', 'scene', 'purpose', 'seed', 'sampling_seed',
                                         'episode_id', 'action_horizon', 'stochastic')}
    resolved.update(mode=mode, fixed_choices=choices, policy=pin)
    carrier = {**resolved, 'phase': 'phase2', 'actor_sha256': genome_hash(genome),
               'model': str(output / 'actor.npz'), 'random_policy': False,
               'max_steps_override': None}
    write(output / 'job.json', carrier)
    write(output / 'main_protocol.json', protocol)
    request = {'schema': 'maniguard-main-episode-request-v1', 'job': job, 'output': str(output),
               'command': ['/synthetic/python', '/synthetic/main.py', '--request', '/synthetic/request.json']}
    write(output / 'request.json', request, sign=True)
    generated = output / 'benchmark.py'
    generated.write_text('# synthetic generated evaluator, never executed\n')
    write(output / 'provenance.json', {'runtime_sha256': sha(generated), 'revision': 'c' * 40})
    archived = []
    extra = [job['actor']] if mode == 'learned' else [job['controller']]
    for reference in references + implementation + [protocol['planner_source']] + extra:
        target = output / 'main_source' / reference['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / reference['path'], target)
        archived.append({'original': reference['path'], 'path': str(target.relative_to(output)),
                         'sha256': reference['sha256']})
    execution = {'schema': 'maniguard-main-execution-v1', **bindings, 'backend': 'phase2',
        'resolved': resolved, 'execution_model': {'path': 'actor.npz', 'sha256': sha(output / 'actor.npz'),
            'genome_sha256': genome_hash(genome), 'role': 'selected_actor' if mode == 'learned' else 'identity_carrier'},
        'sources': archived, 'generated': {'path': 'benchmark.py', 'sha256': sha(generated),
            'provenance_path': 'provenance.json', 'provenance_sha256': sha(output / 'provenance.json')},
        'request': {'path': 'request.json', 'sha256': sha(output / 'request.json')},
        'physics_enforcement': {'pre_advance': True, 'complete': True,
            'source': {'path': 'experiments/helpers/synthetic_physics_driver.py',
                       'sha256': sha(root / 'experiments/helpers/synthetic_physics_driver.py')}}}
    write(output / 'main_execution.json', execution, sign=True)
    inventory = {name: {'bytes': 1, 'sha256': 'd' * 64} for name in (
        '1/_CHECKPOINT_METADATA', '1/params/_METADATA', '1/params/manifest.ocdbt',
        '1/params/_sharding', '1/params/d/data', '1/assets/synthetic/norm_stats.json')}
    write(output / 'policy-provenance.json', {'repo': pin['repo'], 'revision': pin['revision'],
                                             'repo_type': 'model', 'files': inventory})
    metadata = {'serve_config': pin['config'], 'checkpoint_revision': pin['revision'],
                'episode_seed_reset': 'study_episode_id', 'rng_isolation': 'per_episode',
                'checkpoint': str(root / 'data/maniguard/policies/jar/1'),
                'checkpoint_provenance': {'receipt_sha256': sha(output / 'policy-provenance.json'),
                    'verified_files': 6, 'verified_bytes': 6,
                    'verification': 'all_declared_files_sha256_and_available_upstream_digests'},
                'server_source_hashes': {ref['path']: ref['sha256'] for ref in implementation
                    if Path(ref['path']).name in ('maniguard_serve.py', 'maniguard_policy.py',
                        'maniguard_parallel_policy.py', 'maniguard_policy_provenance.py')}}
    write(output / 'main_runtime_evidence.json', {'schema': 'maniguard-main-runtime-evidence-v1', **bindings, 'synthetic': False, 'handshake': metadata,
        'loaded_modules': {ref['path']: ref['sha256'] for ref in implementation},
        'policy_provenance': {'path': 'policy-provenance.json', 'sha256': sha(output / 'policy-provenance.json')}}, sign=True)
    process = {'command': request['command'], 'returncode': 0, 'timed_out': False,
               'interrupted': False, 'process_group_remaining': False, 'process_elapsed_seconds': 20.0}
    write(output / 'main_process.json', {'schema': 'maniguard-main-process-v1', **bindings,
                                        'process': process}, sign=True)
    result = {'status': 'completed', 'scene_name': job['scene'], 'steps': 12, 'success': True,
              'episode_seed': zlib.crc32(b'1:task_0000/base') & 0x7FFFFFFF,
              'ltl_monitored': True, 'ltl_violated': False, 'counted_violation': False}
    observations = [{'step': step, 'ap': {'jar_closed': step >= 1, 'jar_on_support': step < 2},
                     'state': 0, 'accepting': True, 'doomed': False} for step in range(13)]
    write(output / 'episode/results.jsonl', result)
    write(output / 'episode/task_0000/base_ltl.json', {'error': None, 'violated': False,
          'total_steps_monitored': 13, 'log': observations})
    trace = [{'type': 'job', 'job': carrier}, {'type': 'policy_metadata', 'metadata': metadata},
        {'type': 'phase2_features', 'feature_dim': 64, 'actor_sha256': genome_hash(genome),
         'information_access': 'observation_only', 'encoder': encoder,
         'robot_features': 40, 'vision_features': 16, 'text_features': 8},
        {'type': 'observed', 'step': 0, 'monitor': {'valid': True, 'error': None,
                                                 'observation': observations[0]}}]
    adapter = Phase2Adapter(genome, mode=mode, fixed_choices=choices)
    frame = Frame(np.zeros(7), np.zeros(2), np.zeros(3))
    frame_record = {key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in asdict(frame).items()}
    chunk = np.zeros((16, 8), dtype=np.float32)
    chunk[:, 0] = .05
    actions, operations = [], []
    def operation(identifier, attempts, returned, category='slot'):
        operations.append({'index': len(operations), 'type': 'intent', 'id': identifier,
                           'category': category, 'attempt_counts': attempts, 'return_counts': returned})
        operations.append({'index': len(operations), 'type': 'returned', 'id': identifier,
                           'return_counts': returned})
    operation('startup-physics', {}, {'physics_steps': 12}, 'startup')
    operation('encoder-load', {'encoder_load_attempts': 1}, {})
    operation('text', {'encoder_text_attempts': 1}, {'encoder_text_calls': 1})
    query_step = None
    for step in range(12):
        if step % 4 == 0 and step != 0:
            decision = adapter.decide(np.zeros(64), step)
        if step % 8 == 0:
            adapter.on_query()
            query_step = step
            trace += [{'type': 'query', 'step': step}, {'type': 'chunk', 'step': step, 'actions': chunk.tolist()}]
            operation('query-' + str(step), {'policy_queries': 1}, {})
        if step % 4 == 0:
            if step == 0:
                decision = adapter.decide(np.zeros(64), step)
            record = {**decision, 'type': 'decision', 'observation_step': step,
                      'actor_sha256': genome_hash(genome), 'context': {'image_hashes': ['a' * 64, 'b' * 64],
                        'text_sha256': 'c' * 64, 'checkpoint_sha256': encoder['checkpoint_sha256']}}
            if step and step % 8 == 0:
                trace.insert(len(trace) - 2, record)
            else:
                trace.append(record)
            operation('image-' + str(step), {'encoder_image_attempts': 1},
                      {'encoder_image_calls': 1, 'encoder_images': 2, 'adapter_decisions': 1})
        original = chunk[step - query_step].copy()
        original[-1] = -1
        adapted, event = adapter.act(original, frame)
        environment = np.clip(adapted, -1, 1)
        trace.append({'type': 'proposal', 'step': step, 'chunk_index': step - query_step,
                      'original': original.tolist(), 'adapted': adapted.tolist(),
                      'frame': frame_record, 'event': event})
        trace.append({'type': 'executed', 'step': step, 'action': environment.tolist()})
        adapter.applied(environment)
        trace.append({'type': 'observed', 'step': step + 1,
                      'monitor': {'valid': True, 'error': None, 'observation': observations[step + 1]}})
        trace.append({'type': 'goal', 'step': step + 1, 'success': step >= 2,
                      'detail': {'mode': 'held_intersection', 'held': step >= 2, 'intersects': step >= 2}})
        identifier = 'action-' + str(step)
        actions.extend([{'index': len(actions), 'type': 'intent', 'step': step,
                         'operation_id': identifier, 'raw': chunk[step - query_step].tolist(),
                         'adapted': adapted.tolist(), 'environment_input': environment.tolist(),
                         'action_space_low': [-1] * 8, 'action_space_high': [1] * 8},
                        {'index': len(actions) + 1, 'type': 'returned', 'step': step,
                         'operation_id': identifier}])
        operation(identifier, {}, {'policy_control_steps': 1, 'physics_steps': 6})
    trace.extend([{'type': 'phase2_actor', 'actor_sha256': genome_hash(genome),
                   'decisions': 3, 'stochastic': False}, {'type': 'result', 'result': result}])
    lines(output / 'trace.jsonl', trace)
    lines(output / 'main_actions.jsonl', actions)
    lines(output / 'main_operations.jsonl', operations)
    totals = audit.audit_operation_journal(operations)
    ledger = {'schema': 'maniguard-main-costs-v1', 'job_sha256': record_hash(job),
              **{key: totals[key] for key in ('counts', 'by_category', 'reserved_inflight')},
              'pending': [], 'issues': [], 'phase': 'completed', 'elapsed_seconds': 19.0,
              'journal_sha256': sha(output / 'main_operations.jsonl')}
    write(output / 'main_costs.json', ledger, sign=True)
    def authorize(actual_protocol, actual_job, **_):
        if actual_protocol != protocol or actual_job != job:
            raise ValueError('Synthetic exact authorization rejected mutation')
        return {'authorized': True, 'protocol_sha256': protocol['sha256'],
                'episode_id': job['episode_id'], 'job_sha256': record_hash(job)}
    monkeypatch.setattr(audit.protocol_api, 'authorize_job', authorize)
    monkeypatch.setattr(audit, '_reconstruct', lambda *_: generated.read_text())
    def physics(execution, *_):
        if execution['physics_enforcement'].get('complete') is not True:
            raise ValueError('Synthetic declared physics fixture incomplete')
    monkeypatch.setattr(audit, '_physics', physics)
    return {'root': root, 'output': output, 'job': job, 'protocol': protocol,
            'trace': trace, 'actions': actions, 'operations': operations}


def read(case):
    return audit.read_main_episode(case['output'], case['protocol'], case['job'], root=case['root'])


def change(case, filename, function, *, sign=True):
    path = case['output'] / filename
    value = json.loads(path.read_text())
    function(value)
    write(path, value, sign=sign)


@pytest.mark.parametrize('mode', ['learned', 'fixed', 'contextual'])
def test_verified_execution_preserves_scientific_identity_and_all_costs(tmp_path, monkeypatch, mode):
    case = fixture_episode(tmp_path, monkeypatch, mode)
    row = read(case)
    assert row['valid'], row['invalid_reasons']
    assert all(row[key] == value for key, value in case['job'].items())
    assert row['outcome']['S_req'] is True and row['outcome']['U'] == 1
    assert row['costs']['physics_steps'] == 84 and row['costs']['elapsed_seconds'] == 20
    assert row['cost_breakdown']['startup']['physics_steps'] == 12
    assert row['cost_breakdown']['slot']['physics_steps'] == 72
    assert row['costs']['actor_rpc_attempts'] == 0
    assert row['actor_sha256'] is None if mode != 'learned' else row['actor_sha256'] is not None


@pytest.mark.parametrize('mutation', ['job', 'resolved', 'carrier', 'source', 'model', 'handshake',
                                     'process', 'command', 'synthetic', 'driver', 'provenance'])
def test_identity_or_process_corruption_cannot_promote_plan_fields(tmp_path, monkeypatch, mutation):
    case = fixture_episode(tmp_path, monkeypatch)
    output = case['output']
    if mutation == 'job':
        change(case, 'main_job.json', lambda value: value['job'].update(seed=2))
    elif mutation == 'resolved':
        change(case, 'main_execution.json', lambda value: value['resolved'].update(seed=2))
    elif mutation == 'carrier':
        change(case, 'job.json', lambda value: value.update(actor_sha256='a' * 64), sign=False)
    elif mutation == 'source':
        (output / 'main_source/experiments/helpers/maniguard_phase2_adapter.py').write_text('changed')
    elif mutation == 'model':
        (output / 'actor.npz').write_bytes(b'not the selected actor')
    elif mutation == 'handshake':
        change(case, 'main_runtime_evidence.json', lambda value: value['handshake'].update(checkpoint_revision='a' * 40))
    elif mutation == 'process':
        change(case, 'main_process.json', lambda value: value['process'].update(returncode=1))
    elif mutation == 'command':
        change(case, 'main_process.json', lambda value: value['process'].update(command=['different']))
    elif mutation == 'synthetic':
        change(case, 'main_runtime_evidence.json', lambda value: value.update(synthetic=True))
    elif mutation == 'driver':
        change(case, 'main_execution.json', lambda value: value['physics_enforcement'].update(complete=False))
    else:
        (output / 'policy-provenance.json').write_text('{}')
    row = read(case)
    assert not row['valid'] and row['outcome'] is None
    assert not row['execution_binding_verified'] and 'method' not in row
    assert row['episode_id'] == case['job']['episode_id']
    assert row['costs']['physics_steps'] == 84


@pytest.mark.parametrize('mutation', ['adapted', 'executed', 'chunk', 'duplicate_chunk', 'hidden', 'goal',
                                     'native_seed', 'monitor', 'action_operation'])
def test_replay_rejects_changed_behavior_or_native_evidence(tmp_path, monkeypatch, mutation):
    case = fixture_episode(tmp_path, monkeypatch)
    trace = case['trace']
    if mutation in ('adapted', 'executed', 'chunk', 'duplicate_chunk', 'hidden'):
        kind = {'adapted': 'proposal', 'executed': 'executed', 'chunk': 'chunk',
                'duplicate_chunk': 'chunk', 'hidden': 'decision'}[mutation]
        event = next(event for event in trace if event['type'] == kind)
        if mutation == 'duplicate_chunk':
            trace.insert(trace.index(event), deepcopy(event))
        elif mutation == 'chunk':
            event['actions'][0][0] += .1
        elif mutation == 'hidden':
            event['hidden_before'][0] = 1
        else:
            event['adapted' if mutation == 'adapted' else 'action'][0] += .1
        lines(case['output'] / 'trace.jsonl', trace)
    elif mutation == 'goal':
        next(event for event in trace if event['type'] == 'goal')['step'] = 7
        lines(case['output'] / 'trace.jsonl', trace)
    elif mutation == 'native_seed':
        change(case, 'episode/results.jsonl', lambda value: value.update(episode_seed=99), sign=False)
    elif mutation == 'monitor':
        change(case, 'episode/task_0000/base_ltl.json', lambda value: value.update(violated=True), sign=False)
    else:
        case['actions'][0]['operation_id'] = 'unmetered'
        lines(case['output'] / 'main_actions.jsonl', case['actions'])
    row = read(case)
    assert not row['valid'] and row['outcome'] is None and row['invalid_reasons']
    assert row['costs']['policy_control_steps'] == 12


@pytest.mark.parametrize('mutation', ['missing', 'partial', 'pending', 'category', 'understated', 'hash'])
def test_bad_accounting_retains_known_work_without_zero_filling(tmp_path, monkeypatch, mutation):
    case = fixture_episode(tmp_path, monkeypatch)
    if mutation == 'missing':
        (case['output'] / 'main_costs.json').unlink()
    elif mutation == 'partial':
        change(case, 'main_costs.json', lambda value: value['counts'].pop('encoder_load_attempts'))
    elif mutation == 'pending':
        change(case, 'main_costs.json', lambda value: value.update(pending=[{'id': 'unknown'}]))
    elif mutation == 'category':
        change(case, 'main_costs.json', lambda value: value['by_category']['slot'].update(physics_steps=0))
    elif mutation == 'understated':
        change(case, 'main_costs.json', lambda value: value['counts'].update(policy_queries=0))
    else:
        change(case, 'main_costs.json', lambda value: value.update(sha256='a' * 64), sign=False)
    row = read(case)
    assert not row['valid'] and row['cost_issues'] and row['outcome'] is None
    if mutation == 'missing':
        assert row['costs']['physics_steps'] == 84
        assert 'actor_rpc_attempts' not in row['costs']
        assert 'physics_steps' in row['unknown_cost_fields']
    else:
        assert row['costs']['physics_steps'] == 84
        if mutation == 'understated':
            assert row['costs']['policy_queries'] == 2


def test_pending_and_failed_operations_retain_attempts_and_reserved_returns():
    events = [{'index': 0, 'type': 'intent', 'id': 'image', 'category': 'slot',
               'attempt_counts': {'encoder_image_attempts': 1},
               'return_counts': {'encoder_image_calls': 1, 'encoder_images': 2}},
              {'index': 1, 'type': 'failed', 'id': 'image', 'return_counts': {}, 'error_type': 'Interrupted'}]
    result = audit.audit_operation_journal(events)
    assert result['counts']['encoder_image_attempts'] == 1
    assert result['counts']['encoder_image_calls'] == 0
    assert result['reserved_inflight']['encoder_images'] == 2
    assert result['pending_ids'] == ['image']
    with pytest.raises(ValueError, match='unmatched'):
        audit.audit_operation_journal(events + [{'index': 2, 'type': 'returned', 'id': 'image',
                                                'return_counts': {'encoder_image_calls': 1, 'encoder_images': 2}}])


def original_fixture(tmp_path, monkeypatch):
    case = fixture_episode(tmp_path, monkeypatch)
    output, job = case['output'], case['job']
    job.pop('actor')
    resources = controller_resource_contract('benchmark_original')
    controller = signed({'kind': 'benchmark_original', 'deterministic': True,
                         'resource_contract': resources})
    write(case['root'] / 'original.json', controller)
    job.update(actor_sha256=None, run_kind='fixed', method='original', run_id='original-fixed',
               controller={'path': 'original.json', 'sha256': sha(case['root'] / 'original.json')},
               controller_sha256=controller['sha256'], resource_contract=resources,
               resource_contract_sha256=record_hash(resources),
               resource_envelope=episode_resource_envelope(12, resources))
    write(output / 'main_source/original.json', controller)
    request = json.loads((output / 'request.json').read_text())
    request['job'] = job
    write(output / 'request.json', request, sign=True)
    from helpers.maniguard_adaptation import AdapterConfig
    carrier = json.loads((output / 'job.json').read_text())
    carrier.update(phase='main_original', mode='benchmark_original', model=None,
                   actor_sha256=None, adapter=asdict(AdapterConfig(kind='identity', horizon=8)))
    write(output / 'job.json', carrier)
    def execution(value):
        value.update(backend='benchmark_original', execution_model=None,
                     job_sha256=record_hash(job))
        value['resolved'].update(mode='benchmark_original')
        value['request']['sha256'] = sha(output / 'request.json')
        value['sources'].append({'original': 'original.json', 'path': 'main_source/original.json',
                                 'sha256': sha(case['root'] / 'original.json')})
    change(case, 'main_execution.json', execution)
    for filename in ('main_job.json', 'main_runtime_evidence.json', 'main_process.json'):
        def binding(value, filename=filename):
            value['job_sha256'] = record_hash(job)
            if filename == 'main_job.json':
                value['job'] = job
        change(case, filename, binding)
    case['trace'] = [event for event in case['trace'] if event['type'] not in
                     ('decision', 'phase2_actor', 'phase2_features')]
    case['trace'][0]['job'] = carrier
    lines(output / 'trace.jsonl', case['trace'])
    case['operations'] = [event for event in case['operations'] if not
                          event['id'].startswith(('encoder-', 'image-', 'text'))]
    for index, event in enumerate(case['operations']):
        event['index'] = index
    lines(output / 'main_operations.jsonl', case['operations'])
    totals = audit.audit_operation_journal(case['operations'])
    def costs(value):
        value.update(job_sha256=record_hash(job), journal_sha256=sha(output / 'main_operations.jsonl'))
        value.update({key: totals[key] for key in ('counts', 'by_category', 'reserved_inflight')})
    change(case, 'main_costs.json', costs)
    return case


def test_original_benchmark_uses_native_eight_prefix_and_explicit_zero_supplementary_costs(tmp_path, monkeypatch):
    case = original_fixture(tmp_path, monkeypatch)
    row = read(case)
    assert row['valid'], row['invalid_reasons']
    assert row['actor_sha256'] is None and row['resource_contract']['profile'] == 'benchmark_original'
    assert row['short_check'] is False
    assert row['costs']['policy_queries'] == 2
    assert row['costs']['encoder_load_attempts'] == 0
    assert row['costs']['adapter_decisions'] == 0
    event = next(event for event in case['trace'] if event['type'] == 'proposal')
    event['adapted'][0] += .1
    lines(case['output'] / 'trace.jsonl', case['trace'])
    assert not read(case)['valid']


@pytest.mark.parametrize('mutation', ['physics', 'module', 'server', 'request', 'scope'])
def test_missing_actual_runtime_bindings_remain_invalid(tmp_path, monkeypatch, mutation):
    case = fixture_episode(tmp_path, monkeypatch)
    if mutation == 'module':
        change(case, 'main_runtime_evidence.json', lambda value: value['loaded_modules'].pop(
            'experiments/helpers/maniguard_phase2_features.py'))
    elif mutation == 'server':
        change(case, 'main_runtime_evidence.json', lambda value: value['handshake']['server_source_hashes'].clear())
    elif mutation == 'request':
        change(case, 'request.json', lambda value: value['job'].update(seed=44))
        change(case, 'main_execution.json', lambda value: value['request'].update(sha256=sha(case['output'] / 'request.json')))
    elif mutation == 'scope':
        change(case, 'main_protocol.json', lambda value: value.update(status='draft'))
    else:
        change(case, 'main_costs.json', lambda value: value['counts'].update(physics_steps=0))
    assert not read(case)['valid']


@pytest.mark.parametrize('schedule', ['aac', 'fixed8'])
def test_aac_replay_uses_real_saved_proposal_panel_and_exact_main_controller(tmp_path, schedule):
    from types import SimpleNamespace

    from test_maniguard_aac_runtime import Client, Robot, observation, remap

    from helpers.maniguard_aac_runtime import AACRuntime
    root = tmp_path
    config_path = root / 'aac.json'
    config_path.write_bytes((ROOT / 'configs/maniguard/phase2/aac.json').read_bytes())
    profile = 'aac_joint20' if schedule == 'aac' else 'aac_fixed8_joint20'
    resources = controller_resource_contract(profile)
    controller = signed({'kind': profile, 'deterministic': True, 'resource_contract': resources,
                         'aac_config': {'path': 'aac.json', 'sha256': sha(config_path)}})
    write(root / 'controller.json', controller)
    job = {'episode_id': 'synthetic/aac/one', 'job_id': 'synthetic/aac/one', 'family': 'jar',
           'scene': 'task_0000/base', 'seed': 1, 'sampling_seed': 0, 'action_horizon': 2,
           'actor_sha256': None, 'run_kind': 'fixed', 'stochastic': False,
           'checkpoint': {'config': 'synthetic', 'revision': 'b' * 40},
           'controller': {'path': 'controller.json', 'sha256': sha(root / 'controller.json')},
           'controller_sha256': controller['sha256'], 'resource_contract': resources,
           'resource_contract_sha256': record_hash(resources),
           'resource_envelope': episode_resource_envelope(2, resources)}
    protocol = {'sha256': 'f' * 64}
    runtime = AACRuntime(job, root / 'episode', protocol_sha256=protocol['sha256'], root=root)
    client, robot = Client(job, length=2), Robot()
    obs = observation()
    obs['episode_seed'] = zlib.crc32(b'1:task_0000/base') & 0x7FFFFFFF
    runtime.validate_policy(client, 'openpi')
    runtime.start({'name': job['scene']}, obs, None, robot, None, obs['episode_seed'])
    prefix = runtime.query_policy(client, obs, 'openpi', SimpleNamespace(), 0, robot, remap)
    intents = []
    for index, raw in enumerate(prefix):
        converted = raw.copy()
        converted[-1] = np.sign(converted[-1]) if abs(converted[-1]) > .01 else -1
        low, high = np.full(8, -1.), np.full(8, 1.)
        executed = np.clip(converted, low, high)
        runtime.before_action(index, index, raw, converted, executed, low, high)
        runtime.applied(index, executed)
        intents.append({'raw': raw.tolist(), 'adapted': converted.tolist(),
                        'environment_input': executed.tolist(), 'action_space_low': low.tolist(),
                        'action_space_high': high.tolist()})
    runtime.finish({'status': 'completed'})
    runtime.close()
    counts = {name: runtime.counts[name] for name in audit.WORK_COUNTS}
    archived = {'controller.json': root / 'controller.json', 'aac.json': config_path}
    audit._aac_replay(runtime.output, protocol, job, archived, counts, intents)
    proposal = runtime.output / 'aac-group-0000/proposal-001.npy'
    value = np.load(proposal, allow_pickle=False)
    value[0, 0] += .1
    np.save(proposal, value, allow_pickle=False)
    with pytest.raises(ValueError, match='AAC replay'):
        audit._aac_replay(runtime.output, protocol, job, archived, counts, intents)


@pytest.mark.parametrize('mutation', [None, 'identity', 'unattached', 'pending', 'failed', 'pre', 'post'])
def test_final_native_physics_descriptor_requires_qualified_identity_and_exact_counters(
        tmp_path, monkeypatch, mutation):
    from types import SimpleNamespace

    from helpers import maniguard_main_physics
    path = tmp_path / 'runtime-contract.json'
    write(path, {'schema': 'maniguard-main-runtime-contract-v1', 'backend': {'declared': True}}, sign=True)
    reference = {'path': 'runtime-contract.json', 'sha256': sha(path)}
    protocol = {'spec': {'implementation_files': [reference]}}
    descriptor = {'kind': 'physx-prepost-v1', 'accounting_scope': 'post_app_launch_physx_callbacks',
                  'qualified': True, 'installed': False, 'attached': False,
                  'pre_events': 0, 'post_events': 0, 'pending_tick': False, 'failed': False,
                  'native_sources': {'pinned': 'source'}, 'qualification': {'pinned': 'review'}}
    final = {**descriptor, 'installed': True, 'attached': True, 'pre_events': 84, 'post_events': 84}
    if mutation == 'identity':
        final['native_sources'] = {'different': 'source'}
    elif mutation == 'unattached':
        final['attached'] = False
    elif mutation == 'pending':
        final['pending_tick'] = True
    elif mutation == 'failed':
        final['failed'] = True
    elif mutation == 'pre':
        final['pre_events'] = 85
    elif mutation == 'post':
        final['pre_events'] = final['post_events'] = 83
    monkeypatch.setattr(maniguard_main_physics, 'preflight_driver',
                        lambda *_args, **_kwargs: SimpleNamespace(describe=lambda: descriptor))
    call = lambda: audit._physics({'physics_enforcement': descriptor},
        {'physics_enforcement': final}, {'runtime_contract': reference}, protocol,
        {'runtime-contract.json': path}, tmp_path, {'physics_steps': 84})
    if mutation is None:
        call()
    else:
        with pytest.raises(ValueError, match='physics'):
            call()


def test_failed_child_before_manifest_still_retains_parent_elapsed_and_journal_lower_bounds(tmp_path, monkeypatch):
    case = fixture_episode(tmp_path, monkeypatch)
    (case['output'] / 'main_job.json').unlink()
    (case['output'] / 'main_costs.json').unlink()
    row = read(case)
    assert not row['valid'] and row['outcome'] is None
    assert row['costs']['elapsed_seconds'] == 20
    assert row['costs']['physics_steps'] == 84
    assert 'physics_steps' in row['unknown_cost_fields']
    assert 'actor_rpc_attempts' not in row['costs']


@pytest.mark.parametrize('filename,field,value', [
    ('main_costs.json', 'counts', None), ('main_costs.json', 'by_category', {'slot': None}),
    ('main_runtime_evidence.json', 'handshake', None), ('main_process.json', 'process', []),
    ('main_execution.json', 'sources', [None]),
])
def test_malformed_nested_evidence_returns_invalid_row_and_retains_available_work(
        tmp_path, monkeypatch, filename, field, value):
    case = fixture_episode(tmp_path, monkeypatch)
    change(case, filename, lambda record: record.update({field: value}))
    row = read(case)
    assert not row['valid'] and row['outcome'] is None and row['invalid_reasons']
    assert row['costs']['physics_steps'] == 84


@pytest.mark.parametrize('tail', ['valid', 'duplicate_failure', 'truncated_json'])
def test_newer_durable_journal_preserves_paid_prefix_after_stale_ledger_or_malformed_tail(
        tmp_path, monkeypatch, tail):
    case = fixture_episode(tmp_path, monkeypatch)
    events = case['operations']
    events.extend([{'index': len(events), 'type': 'intent', 'id': 'cleanup-tick',
                   'category': 'startup', 'attempt_counts': {}, 'return_counts': {'physics_steps': 1}},
                  {'index': len(events) + 1, 'type': 'returned', 'id': 'cleanup-tick',
                   'return_counts': {'physics_steps': 1}}])
    if tail == 'duplicate_failure':
        events.append({'index': len(events), 'type': 'failed', 'id': 'cleanup-tick',
                       'return_counts': {}, 'error_type': 'OSError'})
    path = case['output'] / 'main_operations.jsonl'
    lines(path, events)
    if tail == 'truncated_json':
        with path.open('a') as stream:
            stream.write('{"index":')
    row = read(case)
    assert not row['valid'] and row['outcome'] is None
    assert row['costs']['physics_steps'] == 85
    assert row['cost_breakdown']['startup']['physics_steps'] == 13
    assert row['cost_breakdown']['slot']['physics_steps'] == 72
    assert 'physics_steps' in row['unknown_cost_fields']
    assert any('journal' in issue.lower() for issue in row['cost_issues'])
