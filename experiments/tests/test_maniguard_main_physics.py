"""Native guard contracts, callback ordering and subprocess termination evidence."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import maniguard_main_physics as physics
from helpers.maniguard_phase2_optim import record_hash


def reference(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def signed(value):
    return value | {'sha256': record_hash(value)}


class Meter:
    def __init__(self, output, cap=2):
        self.output, self.cap = output, cap
        self.events, self.pending = [], {}
        self.returned = 0
        self.fail_at = None

    def begin(self, kind, *, category, returns):
        if self.returned + len(self.pending) >= self.cap:
            raise RuntimeError('cap exhausted')
        assert kind == 'physics_tick' and returns == {'physics_steps': 1}
        token = str(len(self.events))
        self.events.append(('intent', token, category))
        self.pending[token] = returns
        if self.fail_at == 'begin':
            raise OSError('persisted intent followed by failed snapshot')
        return token

    def end(self, token):
        if self.fail_at == 'end':
            raise OSError('returned evidence could not be persisted')
        self.events.append(('returned', token))
        self.pending.pop(token)
        self.returned += 1

    def fail(self, token, error):
        self.events.append(('failed', token, type(error).__name__))


def stop(code):
    raise SystemExit(code)


def driver(tmp_path, cap=2):
    result = physics.NativePhysXDriver({}, root=tmp_path)
    result.meter = Meter(tmp_path, cap)
    result.category = lambda: 'startup'
    result._terminate = stop
    return result


@pytest.fixture
def frozen(tmp_path):
    native = {}
    for role in physics.ROLES:
        path = tmp_path / role
        path.write_text('native fixture ' + role)
        native[role] = reference(path)
    (tmp_path / 'api_stub').write_text(
        'def subscribe_physics_on_step_events(self, fn, pre_step: bool, order: int): pass\n')
    (tmp_path / 'extension_manifest').write_text('[package]\nversion = "106.5.7"\n')
    for role in ('api_stub', 'extension_manifest'):
        native[role] = reference(tmp_path / role)
    receipt = tmp_path / 'native-receipt.json'
    receipt.write_text('{"synthetic_test_fixture":true}\n')
    own = reference(Path(physics.__file__).resolve())
    qualification = tmp_path / 'qualification.json'
    body = {'schema': 'maniguard-physx-qualification-v1',
            'driver_source_sha256': own['sha256'],
            'native_sources_sha256': record_hash(native),
            'accounting_scope': physics.SCOPE, 'passed': True, 'task_episodes': 0,
            'checks': {name: {'pre_post_order_verified': True,
                              'hard_exit_before_overbudget_advance': True,
                              'completed_ticks': 1, 'overbudget_process_exit_code': 86}
                       for name in physics.QUALIFICATION_PATHS},
            'sources': [reference(receipt)]}
    qualification.write_text(json.dumps(signed(body)))
    backend = {'kind': 'maniguard-main-guarded-v1', 'physics_driver': {
        'kind': physics.KIND, 'accounting_scope': physics.SCOPE,
        'native_sources': native, 'qualification': reference(qualification)}}
    protocol = {'spec': {'implementation_files': [own, *native.values(), reference(qualification)]},
                'input_files': []}
    return backend, protocol, body, qualification


def test_preflight_verifies_qualified_source_graph_without_simulator_import(tmp_path, frozen):
    backend, protocol, _, _ = frozen
    before = set(sys.modules)
    result = physics.preflight_driver(backend, root=tmp_path, protocol=protocol)
    assert result.describe()['qualified']
    assert not result.attached
    assert not any(key.startswith(('omni.', 'omnigibson', 'isaacsim'))
                   for key in set(sys.modules) - before)
    assert 'outside this counter' in result.describe()['unobserved_startup']


@pytest.mark.parametrize('mutation', ['no_qualification', 'unbound_driver', 'changed_native',
                                     'scope', 'version', 'missing_native_role'])
def test_preflight_refuses_unqualified_or_changed_native_sources(tmp_path, frozen, mutation):
    backend, protocol, _, _qualification = frozen
    declared = backend['physics_driver']
    if mutation == 'no_qualification':
        declared['qualification'] = None
    elif mutation == 'unbound_driver':
        protocol['spec']['implementation_files'] = protocol['spec']['implementation_files'][1:]
    elif mutation == 'changed_native':
        (tmp_path / 'physx_plugin').write_text('different native binary')
    elif mutation == 'scope':
        declared['accounting_scope'] = 'all physics since process start'
    elif mutation == 'version':
        (tmp_path / 'extension_manifest').write_text('version = "999"\n')
        old = declared['native_sources']['extension_manifest']
        fresh = reference(tmp_path / 'extension_manifest')
        protocol['spec']['implementation_files'].remove(old)
        protocol['spec']['implementation_files'].append(fresh)
        declared['native_sources']['extension_manifest'] = fresh
    else:
        del declared['native_sources']['api_stub']
    with pytest.raises((ValueError, TypeError)):
        physics.preflight_driver(backend, root=tmp_path, protocol=protocol)


@pytest.mark.parametrize('mutation', ['unsigned', 'not_passed', 'wrong_driver', 'wrong_native',
                                     'missing_path', 'failed_path', 'receipt_tamper'])
def test_native_qualification_is_bound_and_all_paths_required(tmp_path, frozen, mutation):
    backend, protocol, body, qualification = frozen
    if mutation == 'not_passed':
        body['passed'] = False
    elif mutation == 'wrong_driver':
        body['driver_source_sha256'] = '0' * 64
    elif mutation == 'wrong_native':
        body['native_sources_sha256'] = '1' * 64
    elif mutation == 'missing_path':
        body['checks'].pop('app_update')
    elif mutation == 'failed_path':
        body['checks']['app_update']['hard_exit_before_overbudget_advance'] = False
    elif mutation == 'receipt_tamper':
        (tmp_path / 'native-receipt.json').write_text('tampered')
    qualification.write_text(json.dumps(body if mutation == 'unsigned' else signed(body)))
    protocol['spec']['implementation_files'][-1] = reference(qualification)
    backend['physics_driver']['qualification'] = reference(qualification)
    with pytest.raises(ValueError):
        physics.preflight_driver(backend, root=tmp_path, protocol=protocol)


def test_tick_reserved_before_advance_and_settled_after_under_current_category(tmp_path):
    guard = driver(tmp_path)
    physical = []
    guard._before(1 / 60)
    assert guard.meter.pending and guard.meter.returned == 0
    physical.append('advance')
    guard._after(1 / 60)
    guard.category = lambda: 'slot'
    guard._before(1 / 60)
    physical.append('advance')
    guard._after(1 / 60)
    assert guard.pre_events == guard.post_events == len(physical) == 2
    assert [event[2] for event in guard.meter.events if event[0] == 'intent'] == ['startup', 'slot']
    with pytest.raises(SystemExit) as error:
        guard._before(1 / 60)
        physical.append('forbidden')
    assert error.value.code == 86 and physical == ['advance', 'advance']
    assert json.loads((tmp_path / 'main_physics_failure.json').read_text())['failed']


@pytest.mark.parametrize('mode', ['nested', 'orphan', 'begin_io', 'end_io', 'bad_dt'])
def test_callback_anomalies_terminate_and_keep_pending_evidence(tmp_path, mode):
    guard = driver(tmp_path)
    with pytest.raises(SystemExit) as error:
        if mode == 'orphan':
            guard._after(1 / 60)
        elif mode == 'bad_dt':
            guard._before(float('nan'))
        else:
            guard.meter.fail_at = {'begin_io': 'begin', 'end_io': 'end'}.get(mode)
            guard._before(1 / 60)
            if mode == 'nested':
                guard._before(1 / 60)
            else:
                guard._after(1 / 60)
    assert error.value.code == 86 and guard.failed
    if mode in ('nested', 'begin_io', 'end_io'):
        assert guard.meter.pending  # Pending physical work is never a known zero.


def test_native_subscriptions_have_pre_post_order_and_loaded_binary_binding(tmp_path):
    binding_path = tmp_path / '_physx.so'
    binding_path.write_bytes(b'native fixture')
    guard = physics.NativePhysXDriver({'python_binding': reference(binding_path)}, root=tmp_path)
    calls = []

    def subscribe(callback, pre_step, order):
        calls.append((callback, pre_step, order))
        return object()

    interface = SimpleNamespace(subscribe_physics_on_step_events=subscribe)
    guard._attach(interface, SimpleNamespace(__file__=str(binding_path)))
    assert guard.attached and len(guard.subscriptions) == 2
    assert [(row[1], row[2]) for row in calls] == [(True, 0), (False, 0)]
    with pytest.raises(RuntimeError, match='twice'):
        guard._attach(interface, SimpleNamespace(__file__=str(binding_path)))
    other = tmp_path / 'other.so'
    other.write_bytes(b'native fixture')
    second = physics.NativePhysXDriver({'python_binding': reference(binding_path)}, root=tmp_path)
    with pytest.raises(ValueError, match='Loaded PhysX'):
        second._attach(interface, SimpleNamespace(__file__=str(other)))


def test_install_wraps_telemetry_app_launch_before_scene_initialization(tmp_path, monkeypatch):
    binding_path = tmp_path / '_physx.so'
    binding_path.write_bytes(b'native fixture')
    events = []
    simulator = SimpleNamespace(_launch_app=lambda: events.append('existing_app_and_telemetry') or 'app')
    interface = SimpleNamespace(subscribe_physics_on_step_events=lambda fn, pre, order:
                                events.append('pre' if pre else 'post') or object())
    modules = {'omnigibson.simulator': simulator,
               'omni.physx': SimpleNamespace(get_physx_interface=lambda: interface),
               'omni.physx.bindings._physx': SimpleNamespace(__file__=str(binding_path))}
    monkeypatch.setattr(physics.importlib, 'import_module', modules.__getitem__)
    guard = physics.NativePhysXDriver({'python_binding': reference(binding_path)}, root=tmp_path)
    guard.install(SimpleNamespace(sim=None), meter=Meter(tmp_path), category=lambda: 'startup')
    assert events == []
    assert simulator._launch_app() == 'app'
    events.append('scene_initialization')
    assert events == ['existing_app_and_telemetry', 'pre', 'post', 'scene_initialization']
    with pytest.raises(ValueError):
        guard.install(SimpleNamespace(sim=None), meter=Meter(tmp_path), category=lambda: 'startup')


def test_real_process_exit_cannot_be_swallowed_by_native_style_callback_handler(tmp_path):
    marker = tmp_path / 'forbidden-advance'
    code = '''
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from helpers.maniguard_main_physics import NativePhysXDriver
p=Path(sys.argv[2])
def deny(*args, **kwargs):
    raise RuntimeError('no reserved tick remains')
g=NativePhysXDriver({})
g.meter=SimpleNamespace(output=p.parent,begin=deny)
g.category=lambda:'slot'
try:
    g._before(1/60)
except BaseException:
    pass
p.write_text('advanced despite callback failure')
'''
    result = subprocess.run([sys.executable, '-c', code,
                             str(Path(__file__).resolve().parents[1]), str(marker)],
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 86, result.stderr
    assert not marker.exists()
    assert (tmp_path / 'main_physics_failure.json').is_file()
