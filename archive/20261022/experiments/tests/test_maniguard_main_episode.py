# ruff: noqa: F811
"""Synthetic exact-child and actual generated-loop checks; no services or physics."""
import ast
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from test_maniguard_aac_runtime import Client, Robot, observation
from test_maniguard_main_protocol import build, compile_jobs, freeze_fixture, planned  # noqa: F401 - shared fixture

from helpers import maniguard_main_episode as episode
from helpers.maniguard_adaptation import Frame
from helpers.maniguard_main_episode_meter import OperationMeter, signed
from helpers.maniguard_main_episode_runtime import GuardedClient, GuardedEncoder, MainRuntime
from helpers.maniguard_main_episode_source import instrument_source, reconstruct_runtime
from helpers.maniguard_main_protocol import (
    WORK_COUNTS,
    controller_resource_contract,
    episode_resource_envelope,
)
from helpers.maniguard_main_results import audit_operation_journal
from helpers.maniguard_phase2_model import genome_hash, identity_genome, save_model
from helpers.maniguard_phase2_optim import record_hash


def reservation(limit=10):
    return {'counts': dict.fromkeys(WORK_COUNTS, limit), 'startup_counts': dict.fromkeys(WORK_COUNTS, limit),
            'failed_work_counts': dict.fromkeys(WORK_COUNTS, 0), 'deadline': 100}


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize('name', WORK_COUNTS)
def test_every_resource_guard_precedes_callback_and_respects_pending(tmp_path, name):
    budget = reservation(0)
    budget['counts'][name] = 1
    meter = OperationMeter(tmp_path, 'a' * 64, budget, clock=lambda: 1)
    first = meter.begin('first', returns={name: 1})
    calls = []
    with pytest.raises(RuntimeError, match='exceeds reserved'):
        meter.call('overflow', lambda: calls.append(1), returns={name: 1})
    assert not calls
    meter.end(first)
    assert meter.snapshot()['counts'][name] == 1
    meter.close()


def test_durable_failed_rpc_has_paid_attempt_unknown_return_and_no_retry(tmp_path):
    meter = OperationMeter(tmp_path, 'a' * 64, reservation(), clock=lambda: 1)

    def fail():
        assert events(tmp_path / 'main_operations.jsonl')[-1]['type'] == 'intent'
        raise RuntimeError('synthetic failure')

    with pytest.raises(RuntimeError, match='synthetic failure'):
        meter.call('rpc', fail, attempts={'policy_queries': 1})
    with pytest.raises(RuntimeError, match='poisoned'):
        meter.call('retry', lambda: None, attempts={'policy_queries': 1})
    ledger = meter.snapshot()
    replay = audit_operation_journal(events(tmp_path / 'main_operations.jsonl'))
    assert ledger['counts']['policy_queries'] == replay['counts']['policy_queries'] == 1
    assert replay['pending_ids'] == ['0'] and replay['failed_ids'] == ['0']
    meter.close()


def test_deadline_and_fsync_failure_stop_before_work(tmp_path, monkeypatch):
    now = [1]
    meter = OperationMeter(tmp_path, 'a' * 64, reservation(), clock=lambda: now[0])
    now[0] = 100
    calls = []
    with pytest.raises(RuntimeError, match='deadline'):
        meter.call('late', lambda: calls.append(1))
    now[0] = 101
    meter.deadline = 200
    monkeypatch.setattr(os, 'fsync', lambda *_: (_ for _ in ()).throw(OSError('disk failure')))
    with pytest.raises(OSError, match='disk failure'):
        meter.call('unrecordable', lambda: calls.append(1))
    assert not calls and meter.broken
    meter.stream.close()


def test_incremental_journal_hash_replays_without_rereads(tmp_path, monkeypatch):
    meter = OperationMeter(tmp_path, 'a' * 64, reservation(500), clock=lambda: 1)
    original = Path.read_bytes

    def protected(path):
        if path.name == 'main_operations.jsonl':
            raise AssertionError('Quadratic journal reread')
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_bytes', protected)
        for _ in range(50):
            meter.call('tick', lambda: None, returns={'physics_steps': 1})
        ledger = meter.snapshot()
    assert ledger['journal_sha256'] == hashlib.sha256(original(tmp_path / 'main_operations.jsonl')).hexdigest()
    assert audit_operation_journal(events(tmp_path / 'main_operations.jsonl'))['counts'] == ledger['counts']
    meter.close()


def test_categories_and_encoder_attempt_return_semantics(tmp_path):
    meter = OperationMeter(tmp_path, 'a' * 64, reservation(), clock=lambda: 1)
    meter.call('startup_tick', lambda: None, category='startup', returns={'physics_steps': 1})
    encoder = GuardedEncoder(SimpleNamespace(begin_episode=lambda _: np.zeros(8),
                                              encode_images=lambda *_: ('v1', 'v2')), meter)
    encoder.begin_episode('instruction')
    assert encoder.encode_images(None, None) == ('v1', 'v2')
    counts = meter.snapshot()['counts']
    assert counts['encoder_text_attempts'] == counts['encoder_text_calls'] == 1
    assert counts['encoder_image_attempts'] == counts['encoder_image_calls'] == 1
    assert counts['encoder_images'] == 2 and counts['physics_steps'] == 1
    assert meter.by_category['startup']['physics_steps'] == 1
    meter.close()


def test_draft_and_old_boolean_gate_rejected_before_ownership_or_output(planned, monkeypatch):
    protocol = build(planned)
    protocol['evaluation_frozen'] = True
    protocol = signed(protocol)
    path = planned['root'] / 'draft.json'
    path.write_text(json.dumps(protocol))
    output = planned['root'] / 'uncreated'
    request = signed({'schema': episode.REQUEST_SCHEMA, 'protocol_path': str(path),
                      'job': compile_jobs(protocol)[0], 'output': str(output), 'parent_binding': {},
                      'reservation': {}, 'policy': {}, 'runtime_contract': {}, 'command': []})
    with pytest.raises(ValueError):
        episode.preflight_request(request, root=planned['root'], config_dir=planned['config'])
    assert not output.exists()


def test_selected_actor_and_execution_only_carrier_are_distinct(planned):
    protocol = build(planned)
    trained = next(job for job in compile_jobs(protocol) if job['run_kind'] == 'trained')
    fixed = next(job for job in compile_jobs(protocol) if job['run_kind'] == 'fixed')
    assert episode.resolve_execution(trained, root=planned['root'])['model']['role'] == 'selected_actor'
    before = deepcopy(fixed)
    route = episode.resolve_execution(fixed, root=planned['root'])
    assert route['mode'] == 'fixed' and route['fixed_choices'] == [0, 0, 0, 0]
    assert route['model']['role'] == 'identity_carrier'
    assert fixed == before and fixed['actor_sha256'] is None


class FakeDriver:
    def describe(self):
        return {'synthetic': True, 'pre_advance': True, 'complete': True,
                'accounting_scope': 'synthetic_explicit_advance_only'}

    def install(self, og, *, meter, category):
        raise AssertionError('Synthetic loop must not import or install a simulator')


class FakeEncoder:
    def __init__(self, **_):
        self.spec, self.runtime_provenance = {}, {}
        self.text_features = None
        self.image_calls = self.text_calls = 0

    def begin_episode(self, _):
        self.text_features = np.zeros(8)
        self.text_calls += 1

    def encode_images(self, *_):
        self.image_calls += 1
        return np.zeros(8), np.zeros(8), {}

    def ledger(self):
        return {'encoder_image_calls': self.image_calls, 'encoder_images': self.image_calls * 2,
                'encoder_image_attempts': self.image_calls, 'encoder_text_calls': self.text_calls,
                'encoder_text_attempts': self.text_calls, 'encoder_seconds': 0}


def make_main_runtime(tmp_path, monkeypatch, profile, *, horizon=9, length=16):
    backend = 'aac' if profile.startswith('aac') else 'benchmark_original' if profile == 'benchmark_original' else 'phase2'
    choices = [1, 1, 1, 1] if profile == 'phase2_fixed' else None
    resources = controller_resource_contract(profile, choices=choices)
    checkpoint = {'repo': 'synthetic/repo', 'revision': 'a' * 40, 'config': 'synthetic', 'step': '1'}
    job = {'family': 'jar', 'scene': 'synthetic/base', 'purpose': 'test', 'seed': 3, 'sampling_seed': 7,
           'episode_id': 'logical/main/controller/jar/synthetic/base/p3/s7', 'action_horizon': horizon,
           'stochastic': False, 'checkpoint': checkpoint, 'actor_sha256': None,
           'run_kind': 'fixed', 'resource_contract': resources, 'resource_contract_sha256': record_hash(resources),
           'resource_envelope': episode_resource_envelope(horizon, resources)}
    if backend == 'aac':
        config = tmp_path / 'aac.json'
        config.write_bytes((ROOT / 'configs/maniguard/phase2/aac.json').read_bytes())
        controller = signed({'kind': profile, 'deterministic': True, 'resource_contract': resources,
                             'aac_config': {'path': 'aac.json', 'sha256': hashlib.sha256(config.read_bytes()).hexdigest()}})
        path = tmp_path / 'controller.json'
        path.write_text(json.dumps(controller))
        job.update(controller={'path': 'controller.json', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
                   controller_sha256=controller['sha256'])
    output = tmp_path / 'episode'
    output.mkdir()
    model = output / 'execution.npz'
    save_model(model, identity_genome())
    carrier = {**job, 'phase': 'phase2', 'output': str(output), 'model': str(model),
               'actor_sha256': genome_hash(identity_genome()), 'mode': 'learned', 'fixed_choices': None,
               'policy': checkpoint, 'encoder_device': 'cpu', 'random_policy': False,
               'adapter': {'kind': 'phase2', 'mode': 'learned'}}
    if profile in ('phase2_identity', 'phase2_fixed'):
        carrier.update(mode='fixed', fixed_choices=choices or [0, 0, 0, 0])
    elif profile == 'phase2_contextual':
        carrier['mode'] = 'contextual'
    if backend == 'benchmark_original':
        carrier['adapter'] = {'kind': 'identity', 'horizon': 8}
    (output / 'job.json').write_text(json.dumps(carrier))
    monkeypatch.setenv('SAFETY_DIAL_MANIGUARD_JOB', str(output / 'job.json'))
    cfg = SimpleNamespace(random_policy=False, use_openpi_client=True, action_dim=8,
                          ik_eef_to_joint=False, gripper_binarize=True, max_steps=horizon,
                          seed=3, scenes=[job['scene']], execute_horizon=8 if backend == 'benchmark_original' else 16,
                          host='127.0.0.1', port=18000)
    client = Client(job, length=length)
    prepared = {'root': str(tmp_path), 'output': str(output), 'job': job, 'job_sha256': record_hash(job),
                'protocol_sha256': 'b' * 64, 'execution': {'backend': backend},
                'handshake': client.get_server_metadata(), 'policy': {'host': cfg.host, 'port': cfg.port},
                'policy_provenance': {'path': 'fake.json', 'sha256': 'c' * 64}}
    budget = reservation(1000)
    budget['counts'].update(job['resource_envelope']['successful_work'])
    meter = OperationMeter(output, record_hash(job), budget, clock=lambda: 1)
    runtime = MainRuntime(prepared, meter, FakeDriver(), client_factory=lambda **_: client,
                          encoder_factory=FakeEncoder)
    policy, client_type = runtime.study.connect_policy(cfg)
    if backend != 'aac':
        runtime.study.validate_policy(policy, client_type)
    robot = Robot()
    frame = Frame(np.zeros(7), np.zeros(2), np.zeros(3))
    if backend == 'aac':
        runtime.study.start({'name': job['scene']}, observation(), None, robot, None, 918)
    else:
        runtime.study.frame = lambda *_: frame
        runtime.study.start({'name': job['scene']}, observation(), None, robot, None)
    return runtime, client, policy, robot, cfg


def actual_loop(source):
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    loop = next(node for node in ast.walk(main) if isinstance(node, ast.While) and 'step_idx' in ast.unparse(node.test))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in (
        '_remap_obs_for_openpi', '_record_action_finiteness', 'query_policy')]
    return compile(ast.fix_missing_locations(ast.Module(body=functions + [loop], type_ignores=[])),
                   'actual-main-generated-loop', 'exec')


@pytest.mark.parametrize('profile,length,horizon,queries,decisions', [
    ('phase2_learned', 16, 9, 2, 3), ('phase2_identity', 1, 5, 5, 2),
    ('phase2_fixed', 16, 9, 3, 3), ('phase2_contextual', 16, 9, 3, 3),
    ('benchmark_original', 16, 9, 2, 0), ('aac_joint20', 2, 5, 60, 0),
    ('aac_fixed8_joint20', 16, 9, 40, 0),
])
def test_actual_generated_loops_meter_exact_rpc_actions_and_encoder_work(
        tmp_path, monkeypatch, profile, length, horizon, queries, decisions):
    runtime, client, policy, robot, cfg = make_main_runtime(tmp_path, monkeypatch, profile,
                                                         horizon=horizon, length=length)
    source = reconstruct_runtime(ROOT, runtime.backend)
    actual = []

    class Tensor:
        def __init__(self, value):
            self.value = value

        def unsqueeze(self, axis):
            return self.value[None, :]

    class Environment:
        def step(self, command):
            assert runtime.action_pending is not None
            assert events(runtime.output / 'main_actions.jsonl')[-1]['type'] == 'intent'
            for _ in range(6):
                runtime.meter.call('synthetic_tick', lambda: None, category=runtime.physics_category(),
                                   returns={'physics_steps': 1})
            actual.append(command.copy())
            return None, 0, None, None, None

    client.action_count = lambda: len(actual)
    for key in ('EVAL_DEBUG_IMG', 'EVAL_DEBUG_IO', 'EVAL_DEBUG_STEP'):
        monkeypatch.delenv(key, raising=False)
    namespace = {'np': np, 'os': os, 'json': json, 'MAIN': runtime, 'AAC': runtime.study,
                 'STUDY': runtime.study, 'policy': policy, 'obs': observation(), 'EvalConfig': SimpleNamespace,
                 'client_type': 'openpi', 'cfg': cfg, 'step_idx': 0, 'done': False,
                 'rollout_diagnostics': {'action_attempts': 0, 'env_steps_returned': 0, 'observations_returned': 0},
                 'action_space': SimpleNamespace(shape=(8,), low=np.full(8, -.25), high=np.full(8, .25)),
                 'recorder': None, 'robot': robot, 'env': Environment(), '_eef_jacobian_arm': None,
                 'torch': SimpleNamespace(from_numpy=Tensor), 'extract_obs': lambda *args: observation(),
                 'scene_info': {'prompt': 'Synthetic instruction.'}, 'episode_seed': 918, 'videos': {},
                 'total_reward': 0, 'ever_contacted': True, '_target_obj': None, 'monitor': None,
                 'goal_checker': None, 'success': False, 'goal_detail': {}}
    exec(actual_loop(source), namespace)  # noqa: S102 - real generated loop, fake dependencies only
    runtime.study.finish({'status': 'completed'})
    ledger = runtime.meter.snapshot()
    replay = audit_operation_journal(events(runtime.output / 'main_operations.jsonl'))
    assert len(actual) == ledger['counts']['policy_control_steps'] == horizon
    assert ledger['counts']['physics_steps'] == horizon * 6
    assert len(client.requests) == ledger['counts']['policy_queries'] == queries
    assert ledger['counts']['adapter_decisions'] == ledger['counts']['encoder_image_calls'] == decisions
    assert ledger['counts']['encoder_images'] == decisions * 2
    assert ledger['counts']['encoder_load_attempts'] == int(decisions > 0)
    assert replay['counts'] == ledger['counts'] and not replay['pending_ids']
    if runtime.backend != 'aac':
        assert {request['study_episode_id'] for request in client.requests} == {runtime.job['episode_id']}
    assert json.loads((runtime.output / 'main_runtime_evidence.json').read_text())['synthetic'] is True
    runtime.close()


def test_guarded_rpc_exhaustion_does_not_invoke_policy(tmp_path):
    meter = OperationMeter(tmp_path, 'a' * 64, reservation(0), clock=lambda: 1)
    calls = []
    client = GuardedClient(SimpleNamespace(infer=lambda _: calls.append(1)), meter)
    with pytest.raises(RuntimeError, match='policy_queries'):
        client.infer({'observation': 'synthetic'})
    assert not calls
    meter.close()


def test_main_source_refuses_changed_anchors():
    with pytest.raises(ValueError, match='anchor'):
        instrument_source('# missing pinned source', 'phase2')


@pytest.mark.parametrize('field,value', [
    ('scene', 'other/base'), ('seed', 999), ('sampling_seed', 999), ('action_horizon', 32),
    ('actor_sha256', 'f' * 64), ('controller_sha256', 'e' * 64), ('purpose', 'engineering'),
    ('episode_id', 'replacement-physical-identity'), ('stochastic', True),
])
def test_exact_grid_rejects_overrides_before_claim_or_side_effects(planned, monkeypatch, field, value):
    _, _, protocol = freeze_fixture(planned)
    path = planned['root'] / 'frozen.json'
    path.write_text(json.dumps(protocol))
    job = deepcopy(compile_jobs(protocol)[0])
    job[field] = value
    output = planned['root'] / 'not-created'
    request = signed({'schema': episode.REQUEST_SCHEMA, 'protocol_path': str(path), 'job': job,
                      'output': str(output), 'parent_binding': {}, 'reservation': {}, 'policy': {},
                      'runtime_contract': {}, 'command': []})
    reached = []
    import helpers.maniguard_main_launcher as launcher
    monkeypatch.setattr(launcher, 'verify_parent_binding', lambda *_: reached.append('claim'))
    with pytest.raises(ValueError, match='exact member'):
        episode.preflight_request(request, root=planned['root'], config_dir=planned['config'])
    assert not reached and not output.exists()


def test_changed_source_fails_before_output_policy_or_ownership(planned, monkeypatch):
    _, _, protocol = freeze_fixture(planned)
    path = planned['root'] / 'frozen.json'
    path.write_text(json.dumps(protocol))
    (planned['root'] / 'synthetic_runtime.py').write_text('# changed\n')
    request = signed({'schema': episode.REQUEST_SCHEMA, 'protocol_path': str(path),
                      'job': compile_jobs(protocol)[0], 'output': str(planned['root'] / 'not-created'),
                      'parent_binding': {}, 'reservation': {}, 'policy': {}, 'runtime_contract': {}, 'command': []})
    reached = []
    import helpers.maniguard_main_launcher as launcher
    monkeypatch.setattr(launcher, 'verify_parent_binding', lambda *_: reached.append('claim'))
    with pytest.raises(ValueError, match='changed'):
        episode.preflight_request(request, root=planned['root'], config_dir=planned['config'])
    assert not reached and not Path(request['output']).exists()


def test_native_os_exit_has_durable_finalization_for_both_branches():
    for backend in ('phase2', 'benchmark_original', 'aac'):
        source = reconstruct_runtime(ROOT, backend)
        assert source.count('MAIN.finalize_exit()') == 2
        assert 'MAIN.finalize_exit()\n        os._exit(1)' in source
        assert 'MAIN.finalize_exit()\n    os._exit(1 if _failed else 0)' in source


def test_action_guard_exhausted_before_environment_call(tmp_path, monkeypatch):
    runtime, _, _, _, _ = make_main_runtime(tmp_path, monkeypatch, 'phase2_learned', horizon=1)
    runtime.meter.limits['slot']['policy_control_steps'] = 0
    vector = np.zeros(8)
    with pytest.raises(RuntimeError, match='policy_control_steps'):
        runtime.before_environment_action(0, vector, vector, vector, np.full(8, -1), np.ones(8))
    assert runtime.action_pending is None
    assert not events(runtime.output / 'main_actions.jsonl')
    runtime.close()


def test_pending_action_and_encoder_failure_never_become_zero_work(tmp_path, monkeypatch):
    runtime, _, _, _, _ = make_main_runtime(tmp_path, monkeypatch, 'phase2_learned', horizon=1)
    vector = np.zeros(8)
    runtime.before_environment_action(0, vector, vector, vector, np.full(8, -1), np.ones(8))
    runtime.study.finish({'status': 'rollout_failed'})
    ledger = runtime.meter.snapshot()
    assert ledger['counts']['policy_control_steps'] == 0
    assert ledger['reserved_inflight']['policy_control_steps'] == 1
    assert ledger['pending'] and ledger['issues']
    runtime.close()


@pytest.mark.parametrize('kind', ['trained', 'fixed'])
def test_preparation_archives_exact_scientific_identity_and_separate_carrier(planned, monkeypatch, kind):
    protocol = build(planned)
    job = next(row for row in compile_jobs(protocol) if row['run_kind'] == kind)
    output = planned['root'] / 'synthetic-output'
    provenance = planned['root'] / 'fake-policy-provenance.json'
    provenance.write_text('{"synthetic":true}\n')
    route = episode.resolve_execution(job, root=planned['root'])
    prepared = {'root': str(planned['root']), 'output': str(output), 'protocol': protocol,
                'job': job, 'job_sha256': record_hash(job), 'protocol_sha256': protocol['sha256'],
                'driver': FakeDriver(), 'route': route,
                'sources': episode._source_references(protocol, job, planned['root']),
                'policy_provenance_source': provenance, 'encoder_device': 'cpu'}
    monkeypatch.setattr(episode, 'preflight_request', lambda *_, **__: prepared)
    from helpers import maniguard_main_episode_source as generator

    def fake_source(destination, backend, *, root):
        path = Path(destination) / 'benchmark_main.py'
        path.write_text('# Explicit synthetic archive fixture; never executed.\n')
        (Path(destination) / 'main-source-provenance.json').write_text('{"synthetic":true}\n')
        return path

    monkeypatch.setattr(generator, 'prepare_runtime', fake_source)
    result = episode.prepare_episode({'synthetic': True}, root=planned['root'])
    recorded = json.loads((output / 'main_job.json').read_text())
    execution = json.loads((output / 'main_execution.json').read_text())
    carrier = json.loads((output / 'job.json').read_text())
    assert recorded['job'] == job
    assert carrier['episode_id'] == job['episode_id'] != str(output)
    assert carrier['scientific_actor_sha256'] == job['actor_sha256']
    assert carrier['scientific_controller_sha256'] == job['controller_sha256']
    assert execution['execution_model']['path'] == 'actor.npz'
    assert execution['execution_model']['role'] == ('selected_actor' if kind == 'trained' else 'identity_carrier')
    assert execution['resolved']['mode'] == ('learned' if kind == 'trained' else 'fixed')
    assert execution['physics_enforcement']['synthetic'] is True
    assert result['evaluator'] == str(output / 'benchmark_main.py')
    for item in execution['sources']:
        assert hashlib.sha256((output / item['path']).read_bytes()).hexdigest() == item['sha256']


def test_capability_gate_checks_sources_then_rejects_unqualified_native_driver(monkeypatch, tmp_path):
    checked = []
    monkeypatch.setattr(episode, '_verify_sources', lambda *_: checked.append('sources'))
    monkeypatch.setattr(episode, 'verify_upstream_trees', lambda *_, **__: checked.append('upstream'))
    contract = signed({'schema': episode.RUNTIME_SCHEMA, 'supported_profiles': ['phase2_learned'],
                       'episode_timeout_seconds': 60, 'policy_startup_timeout_seconds': 60,
                       'termination_grace_seconds': 2, 'cleanup_reserve_seconds': 6,
                       'audit_timeout_seconds': 5, 'audit_cleanup_reserve_seconds': 6,
                       'per_episode_startup_counts': dict.fromkeys(WORK_COUNTS, 0),
                       'per_episode_failed_work_counts': dict.fromkeys(WORK_COUNTS, 0),
                       'backend': {'kind': 'maniguard-main-guarded-v1', 'physics_driver': 'unavailable'}})
    job = {'resource_contract': {'profile': 'phase2_learned'}}
    with pytest.raises(ValueError, match='physics driver'):
        episode.preflight_backend({}, job, contract, root=tmp_path)
    assert checked == ['sources', 'upstream'] and not list(tmp_path.iterdir())


@pytest.fixture
def upstream_copy(tmp_path):
    # Read the existing Git objects through a disposable shared clone. No commits,
    # index updates or tracked-source writes occur in the actual upstream checkout.
    path = tmp_path / 'copied-upstream'
    subprocess.run(['git', 'clone', '--shared', '--quiet', str(ROOT / 'third_party/ManiGuard'), str(path)],
                   check=True, capture_output=True)
    revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    return path, revision


def test_upstream_changed_monitor_rejected_before_backend_or_output(upstream_copy, monkeypatch, tmp_path):
    path, revision = upstream_copy
    episode._verify_repository_tree(path, revision, ('maniguard',))
    dependency = next((path / 'maniguard').rglob('*monitor*.py'))
    dependency.write_text(dependency.read_text() + '\n# synthetic unpinned monitor change\n')
    reached = []
    monkeypatch.setattr(episode, '_verify_sources', lambda *_: None)
    monkeypatch.setattr(episode, 'verify_upstream_trees',
                        lambda *_, **__: episode._verify_repository_tree(path, revision, ('maniguard',)))
    import helpers.maniguard_main_physics as physics
    monkeypatch.setattr(physics, 'preflight_driver', lambda *_, **__: reached.append('driver'))
    with pytest.raises(ValueError, match='tracked tree has local edits'):
        episode.preflight_backend({}, {}, {}, root=tmp_path)
    assert not reached and not (tmp_path / 'episode').exists()


@pytest.mark.parametrize('change', ['head', 'staged', 'untracked', 'ignored_source'])
def test_upstream_head_staged_and_import_shadowing_sources_are_refused(upstream_copy, change):
    path, revision = upstream_copy
    if change == 'head':
        revision = '0' * 40
        expected = 'HEAD differs'
    elif change == 'staged':
        dependency = next((path / 'maniguard').rglob('*monitor*.py'))
        dependency.write_text(dependency.read_text() + '\n# staged fixture change\n')
        subprocess.run(['git', '-C', str(path), 'add', str(dependency)], check=True, capture_output=True)
        expected = 'tracked tree'
    else:
        source = path / 'maniguard' / 'untracked_dependency.py'
        source.write_text('# Synthetic import-shadowing dependency.\n')
        if change == 'ignored_source':
            with (path / '.git/info/exclude').open('a') as stream:
                stream.write('\nmaniguard/untracked_dependency.py\n')
        expected = 'source is untracked'
    with pytest.raises(ValueError, match=expected):
        episode._verify_repository_tree(path, revision, ('maniguard',))


def test_generated_importable_source_requires_explicit_unchanged_pin(upstream_copy):
    path, revision = upstream_copy
    generated = path / 'maniguard/_synthetic_version.py'
    generated.write_text('version = "synthetic"\n')
    pins = {str(generated.resolve()): hashlib.sha256(generated.read_bytes()).hexdigest()}
    episode._verify_repository_tree(path, revision, ('maniguard',), pins)
    generated.write_text('version = "changed"\n')
    with pytest.raises(ValueError, match='untracked and unpinned'):
        episode._verify_repository_tree(path, revision, ('maniguard',), pins)
