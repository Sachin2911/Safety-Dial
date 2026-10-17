"""Actual benchmark loop with fake actions/clients; no simulator, models or services."""

import ast
import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from helpers.maniguard_aac_runtime import AACRuntime, replay_episode, replay_group
from helpers.maniguard_aac_sampling import SamplingFailure
from helpers.maniguard_aac_source import instrument_source, prepare_runtime
from helpers.maniguard_main_protocol import (
    WORK_COUNTS,
    _signed,
    controller_resource_contract,
    episode_resource_envelope,
    source_reference,
)
from helpers.maniguard_phase2_optim import record_hash


class Client:
    def __init__(self, job, *, length=16, fail_at=None, malformed=None):
        self.job, self.length, self.fail_at, self.malformed = job, length, fail_at, malformed
        self.requests = []
        self.responses = []
        self.action_count = lambda: 0
        self.actions_seen = []

    def get_server_metadata(self):
        return {'serve_config': self.job['checkpoint']['config'],
                'checkpoint_revision': self.job['checkpoint']['revision'],
                'episode_seed_reset': 'study_episode_id', 'rng_isolation': 'per_episode'}

    def infer(self, request):
        self.actions_seen.append(self.action_count())
        self.requests.append(deepcopy(request))
        if len(self.requests) == self.fail_at:
            raise RuntimeError('synthetic RPC failure')
        length = self.length + int(self.malformed == 'unequal' and len(self.requests) == 2)
        # Distinct accessible samples with a moving sample zero. Values also test
        # float32 conversion, clipping, and the benchmark gripper dead zone.
        actions = np.zeros((length, 8), dtype=np.float64)
        actions[:, :7] = .3 + .001 * ((len(self.requests) - 1) % 20)
        actions[:, 7] = np.resize([0, .005, .01, .02, -1], length)
        if self.malformed == 'nonfinite':
            actions[0, 0] = np.nan
        if length > 16:
            actions[16:] = np.nan  # inaccessible tail never enters selection
        self.responses.append(actions.copy())
        return {'actions': actions}


class Robot:
    default_arm = 'arm'
    def __init__(self):
        self.arm_control_idx = {'arm': np.arange(7)}

    def get_joint_positions(self):
        return np.zeros(9)


def observation():
    return {'overview_image': np.zeros((2, 3, 3), dtype=np.uint8),
            'wrist_images': np.ones((2, 3, 3), dtype=np.uint8), 'states': np.zeros(8),
            'task_descriptions': 'Synthetic instruction.', 'episode_seed': 918}


def remap(obs, cfg):
    return {'observation/image_left': obs['overview_image'],
            'observation/wrist_image': obs['wrist_images'], 'observation/state': obs['states'],
            'prompt': obs['task_descriptions'], 'episode_seed': obs['episode_seed']}


@pytest.fixture
def make_runtime(tmp_path):
    runtimes = []

    def make(*, schedule='aac', length=16, horizon=20, query_cap=None, fail_at=None, malformed=None):
        root = tmp_path / str(len(runtimes))
        root.mkdir()
        config_path = root / 'aac.json'
        config_path.write_bytes((ROOT / 'configs/maniguard/phase2/aac.json').read_bytes())
        profile = 'aac_joint20' if schedule == 'aac' else 'aac_fixed8_joint20'
        resources = controller_resource_contract(profile)
        controller = _signed({'kind': profile, 'deterministic': True,
                              'resource_contract': resources,
                              'aac_config': source_reference(config_path, root=root)})
        controller_path = root / 'controller.json'
        controller_path.write_text(json.dumps(controller))
        job = {'episode_id': 'synthetic/aac/physical-0', 'job_id': 'synthetic/aac/physical-0',
               'family': 'jar', 'scene': 'synthetic-scene/base', 'seed': 7, 'sampling_seed': 9,
               'action_horizon': horizon, 'actor_sha256': None, 'run_kind': 'fixed',
               'stochastic': False, 'checkpoint': {'config': 'synthetic', 'revision': 'a' * 40},
               'controller': source_reference(controller_path, root=root),
               'controller_sha256': controller['sha256'], 'resource_contract': resources,
               'resource_contract_sha256': record_hash(resources),
               'resource_envelope': episode_resource_envelope(horizon, resources)}
        runtime = AACRuntime(job, root / 'episode', protocol_sha256='b' * 64,
                             root=root, query_cap=query_cap)
        runtimes.append(runtime)
        client = Client(job, length=length, fail_at=fail_at, malformed=malformed)
        runtime.validate_policy(client, 'openpi')
        robot = Robot()
        runtime.start({'name': job['scene']}, observation(), None, robot, None, 918)
        cfg = SimpleNamespace(max_steps=horizon, execute_horizon=4, gripper_binarize=True,
                              ik_eef_to_joint=False, action_dim=8, random_policy=False,
                              use_openpi_client=True, seed=7, scenes=[job['scene']])
        return runtime, client, robot, cfg

    yield make
    for runtime in runtimes:
        runtime.close()


def query(runtime, client, robot, cfg):
    return runtime.query_policy(client, observation(), 'openpi', cfg,
                                runtime.counts['policy_control_steps'], robot, remap)


def consume(runtime, prefix, *, count=None):
    low, high = np.full(8, -.25), np.full(8, .25)
    for index, raw in enumerate(prefix[:count]):
        converted = raw.copy()
        converted[-1] = np.sign(converted[-1]) if abs(converted[-1]) > .01 else -1.0
        clipped = np.clip(converted, low, high)
        step = runtime.counts['policy_control_steps']
        runtime.before_action(step, index, raw, converted, clipped, low, high)
        runtime.applied(step, clipped)


@pytest.mark.parametrize('schedule,length', [(s, n) for s in ('aac', 'fixed8') for n in (1, 2, 7, 8, 16, 32)])
def test_real_sampler_twenty_calls_exact_prefix_replay_and_zero_clip(make_runtime, schedule, length):
    runtime, client, robot, cfg = make_runtime(schedule=schedule, length=length, horizon=40)
    prefix = query(runtime, client, robot, cfg)
    assert len(client.requests) == 20
    assert runtime.counts['policy_queries'] == 20
    assert prefix.dtype == np.float32
    assert prefix.tobytes() == client.responses[0][:len(prefix)].astype(np.float32).tobytes()
    assert len(prefix) == min(8, length) if schedule == 'fixed8' else 1 <= len(prefix) <= min(16, length)
    assert len({item['study_episode_id'] for item in client.requests}) == 20
    assert client.requests[0]['study_episode_id'] == runtime.job['episode_id']
    assert all(not item['observation/image_left'].any() for item in client.requests)
    consume(runtime, prefix)
    runtime.finish({'status': 'completed'})
    report = replay_episode(runtime.output)
    assert report['valid'], report
    assert report['counts']['policy_queries'] == 20
    assert all(report['counts'][key] == 0 for key in WORK_COUNTS
               if key.startswith('encoder_') or key in ('actor_rpc_attempts', 'adapter_decisions'))
    assert len(list(runtime.output.glob('aac-group-0000/proposal-*.npy'))) == 20
    assert len(list(runtime.output.glob('aac-group-0000/receipt-*.json'))) == 41


def test_one_sampler_persists_streams_and_disallows_prefix_double_use(make_runtime):
    runtime, client, robot, cfg = make_runtime(length=2)
    sampler = runtime.sampler
    first = query(runtime, client, robot, cfg)
    with pytest.raises(ValueError, match='consumption'):
        query(runtime, client, robot, cfg)
    consume(runtime, first)
    second = query(runtime, client, robot, cfg)
    assert runtime.sampler is sampler and len(client.requests) == 40
    for index in range(20):
        assert client.requests[index]['study_episode_id'] == client.requests[index + 20]['study_episode_id']
        assert client.requests[index]['episode_seed'] == client.requests[index + 20]['episode_seed']
    consume(runtime, second)
    with pytest.raises(ValueError, match='once'):
        consume(runtime, second)
    runtime.finish({'status': 'completed'})
    assert replay_episode(runtime.output)['valid']


@pytest.mark.parametrize('failure', ['rpc', 'unequal', 'nonfinite', 'budget'])
def test_failed_rpc_validation_and_cap_keep_costs_and_never_retry(make_runtime, failure):
    kwargs = {'fail_at': 4} if failure == 'rpc' else {'query_cap': 3} if failure == 'budget' else {'malformed': failure}
    runtime, client, robot, cfg = make_runtime(**kwargs)
    with pytest.raises(SamplingFailure):
        query(runtime, client, robot, cfg)
    expected = {'rpc': 4, 'budget': 3, 'unequal': 2, 'nonfinite': 1}[failure]
    assert len(client.requests) == expected == runtime.counts['policy_queries']
    with pytest.raises(ValueError, match='consumption'):
        query(runtime, client, robot, cfg)
    runtime.finish({'status': 'rollout_failed'})
    report = replay_episode(runtime.output)
    assert not report['valid'] and not report['costs_complete']
    assert report['counts']['policy_queries'] == expected
    assert report['counts']['policy_control_steps'] == 0


def test_sink_failure_before_rpc_does_not_dispatch(make_runtime, monkeypatch):
    import helpers.maniguard_aac_runtime as implementation
    runtime, client, robot, cfg = make_runtime()
    original = implementation._write_json

    def fail(path, value):
        if Path(path).name.startswith('receipt-'):
            raise OSError('synthetic durable sink failure')
        return original(path, value)

    monkeypatch.setattr(implementation, '_write_json', fail)
    with pytest.raises(SamplingFailure):
        query(runtime, client, robot, cfg)
    assert not client.requests and runtime.counts['policy_queries'] == 0
    assert runtime.broken


@pytest.mark.parametrize('tamper', ['proposal', 'prefix', 'receipt', 'count', 'executed', 'source'])
def test_independent_replay_rejects_tampered_arrays_receipts_counts_and_actions(make_runtime, tamper):
    runtime, client, robot, cfg = make_runtime(length=2)
    prefix = query(runtime, client, robot, cfg)
    consume(runtime, prefix)
    runtime.finish({'status': 'completed'})
    assert replay_episode(runtime.output)['valid']
    group = runtime.output / 'aac-group-0000'
    if tamper in ('proposal', 'prefix'):
        path = group / ('proposal-000.npy' if tamper == 'proposal' else 'selected.npy')
        value = np.load(path)
        value[0, 0] += .01
        np.save(path, value)
    elif tamper == 'receipt':
        path = group / 'receipt-0040.json'
        value = json.loads(path.read_text())
        value['receipt']['rpc_attempts'] = 19
        path.write_text(json.dumps(value))
    elif tamper == 'count':
        path = runtime.output / 'aac-ledger.json'
        value = json.loads(path.read_text())
        value['policy_queries'] = 1
        path.write_text(json.dumps(value))
    elif tamper == 'source':
        (runtime.output / 'aac_source/maniguard_phase2_aac.py').write_text('# changed')
    else:
        path = runtime.output / 'aac-trace.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        next(row for row in rows if row['type'] == 'executed')['action'][0] = 99
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert not replay_episode(runtime.output)['valid']


def test_pending_dispatch_without_terminal_update_is_unknown_not_free(make_runtime):
    runtime, client, robot, cfg = make_runtime(length=2)
    query(runtime, client, robot, cfg)
    group = runtime.output / 'aac-group-0000'
    # Simulate the durable filesystem left by a process lost immediately after
    # dispatch intent: later return/selection records are unavailable.
    for path in group.glob('receipt-*.json'):
        if path.name != 'receipt-0000.json':
            path.unlink()
    with pytest.raises(ValueError, match='intent/return'):
        replay_group(group, config=runtime.config, schedule=runtime.schedule)
    report = replay_episode(runtime.output)
    assert not report['valid'] and not report['costs_complete']


def _actual_loop(source):
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    loop = next(node for node in ast.walk(main) if isinstance(node, ast.While)
                and 'step_idx' in ast.unparse(node.test))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('_remap_obs_for_openpi', '_record_action_finiteness')]
    return compile(ast.fix_missing_locations(ast.Module(body=functions + [loop], type_ignores=[])),
                   'actual-benchmark-aac-loop', 'exec')


@pytest.mark.parametrize('schedule,length,horizon,starts', [
    ('aac', 1, 3, [0, 1, 2]), ('aac', 2, 5, [0, 2, 4]), ('fixed8', 16, 17, [0, 8, 16]),
])
def test_actual_generated_benchmark_loop_queries_once_per_group_without_four_step_quantization(
        make_runtime, schedule, length, horizon, starts, monkeypatch):
    runtime, client, robot, cfg = make_runtime(schedule=schedule, length=length, horizon=horizon)
    source = instrument_source((ROOT / 'third_party/ManiGuard/maniguard/eval/benchmark.py').read_text(),
                               ROOT / 'third_party/ManiGuard')
    actual = []

    class Tensor:
        def __init__(self, array):
            self.array = array

        def unsqueeze(self, axis):
            return self.array[None, :]

    class Environment:
        def step(self, command):
            actual.append(command.copy())
            return None, 0, None, None, None

    client.action_count = lambda: len(actual)
    for key in ('EVAL_DEBUG_IMG', 'EVAL_DEBUG_IO', 'EVAL_DEBUG_STEP'):
        monkeypatch.delenv(key, raising=False)
    namespace = {'np': np, 'os': os, 'json': json, 'AAC': runtime, 'policy': client,
                 'obs': observation(), 'EvalConfig': SimpleNamespace, 'client_type': 'openpi',
                 'cfg': cfg, 'step_idx': 0, 'done': False,
                 'rollout_diagnostics': {'action_attempts': 0, 'env_steps_returned': 0,
                                         'observations_returned': 0},
                 'action_space': SimpleNamespace(shape=(8,), low=np.full(8, -.25), high=np.full(8, .25)),
                 'recorder': None, 'robot': robot, 'env': Environment(),
                 'torch': SimpleNamespace(from_numpy=Tensor), 'extract_obs': lambda *args: observation(),
                 'scene_info': {'prompt': 'Synthetic instruction.'}, 'episode_seed': 918, 'videos': {},
                 'total_reward': 0, 'ever_contacted': True, '_target_obj': None, 'monitor': None,
                 'goal_checker': None, 'success': False, 'goal_detail': {}}
    exec(_actual_loop(source), namespace)  # noqa: S102 -- pinned loop, fake dependencies only
    runtime.finish({'status': 'completed'})
    groups = [json.loads((runtime.output / f'aac-group-{index:04d}/group.json').read_text())
              for index in range(len(starts))]
    assert [group['step'] for group in groups] == starts
    assert len(client.requests) == 20 * len(starts)
    assert len(actual) == horizon
    assert client.actions_seen == [step for step in starts for _ in range(20)]
    assert replay_episode(runtime.output)['valid']


def test_source_hooks_fail_closed_and_keep_upstream_monitor_and_conversion(tmp_path):
    generated = prepare_runtime(tmp_path)
    source = generated.read_text()
    assert 'chunk = query_policy(policy, obs, client_type, cfg)' not in source
    assert source.count('chunk = AAC.query_policy(') == 1
    assert 'min(len(chunk), cfg.max_steps - step_idx)' in source
    assert 'np.sign(action[-1]) if abs(action[-1]) > 0.01 else -1.0' in source
    assert 'action_clipped = np.clip(ctrl_action, action_space.low, action_space.high)' in source
    assert 'monitor.step(step_idx)' in source and 'goal_checker.check(env)' in source
    assert 'Phase2Study' not in source and 'FrozenContextEncoder' not in source
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
             and node.func.value.id == 'AAC']
    assert sum(node.func.attr == 'start' for node in calls) == 1
    assert sum(node.func.attr == 'before_action' for node in calls) == 1
    assert sum(node.func.attr == 'query_policy' for node in calls) == 1
    assert hashlib.sha256(source.encode()).hexdigest() == json.loads(
        (tmp_path / 'aac-source-provenance.json').read_text())['runtime_sha256']
    with pytest.raises(ValueError, match='not unique'):
        instrument_source(source, ROOT / 'third_party/ManiGuard')


def test_archive_failure_latches_before_sampling_and_after_paid_return(make_runtime, monkeypatch):
    import helpers.maniguard_aac_runtime as implementation
    original = implementation._write_array
    for failing_name, paid in [('overview.npy', 0), ('proposal-000.npy', 1)]:
        runtime, client, robot, cfg = make_runtime()

        def fail(path, value, failing_name=failing_name):
            if Path(path).name == failing_name:
                raise OSError('synthetic archive failure')
            return original(path, value)

        monkeypatch.setattr(implementation, '_write_array', fail)
        with pytest.raises((OSError, SamplingFailure)):
            query(runtime, client, robot, cfg)
        assert runtime.broken and runtime.counts['policy_queries'] == paid
        assert len(client.requests) == paid
        with pytest.raises(ValueError, match='consumption'):
            query(runtime, client, robot, cfg)
        monkeypatch.setattr(implementation, '_write_array', original)


def test_process_exit_after_dispatch_intent_is_paid_unknown_work(make_runtime):
    import multiprocessing

    runtime, client, robot, cfg = make_runtime()
    client.infer = lambda request: os._exit(73)
    child = multiprocessing.get_context('fork').Process(target=query, args=(runtime, client, robot, cfg))
    child.start()
    child.join(timeout=5)
    if child.is_alive():
        child.kill()
        child.join(timeout=5)
        pytest.fail('Synthetic crash child did not exit')
    assert child.exitcode == 73
    # The parent must not invent a terminal ledger for the crashed child's files.
    runtime.trace.close()
    report = replay_episode(runtime.output)
    assert not report['valid'] and not report['costs_complete']
    assert report['unresolved_dispatches'] == 1
    assert 'policy_queries' in report['unknown_cost_fields']
    assert report['counts']['policy_queries'] == 1
    assert report['counts']['policy_control_steps'] == 0


def test_no_clip_or_simulator_imports_on_aac_sampling_path(make_runtime, monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(('omnigibson', 'open_clip', 'torch')) or name == 'helpers.maniguard_phase2_features':
            raise AssertionError('Unexpected model/simulator/CLIP import: ' + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    runtime, client, robot, cfg = make_runtime(length=2)
    consume(runtime, query(runtime, client, robot, cfg))
    runtime.finish({'status': 'completed'})
    assert replay_episode(runtime.output)['valid']


def test_wrong_cfg_or_native_seed_fails_before_policy_calls(make_runtime):
    runtime, client, robot, cfg = make_runtime()
    runtime.validate_config(cfg)
    cfg.max_steps += 1
    with pytest.raises(ValueError, match='configuration'):
        runtime.validate_config(cfg)
    cfg.max_steps -= 1
    bad = observation()
    bad['episode_seed'] += 1
    with pytest.raises(ValueError, match='RNG route'):
        runtime.query_policy(client, bad, 'openpi', cfg, 0, robot, remap)
    assert not client.requests



def test_pending_action_is_unresolved_execution_not_zero_physics(make_runtime):
    runtime, client, robot, cfg = make_runtime(length=2)
    raw = query(runtime, client, robot, cfg)[0]
    converted = raw.copy()
    converted[-1] = -1
    low, high = np.full(8, -1.0), np.full(8, 1.0)
    runtime.before_action(0, 0, raw, converted, np.clip(converted, low, high), low, high)
    runtime.finish({'status': 'rollout_failed'})
    report = replay_episode(runtime.output)
    assert not report['valid'] and report['unresolved_action_intents'] == 1
    assert {'policy_control_steps', 'physics_steps'} <= set(report['unknown_cost_fields'])
    assert report['counts']['policy_queries'] == 20



def test_resource_contract_types_cannot_change_behind_the_declared_hash(make_runtime):
    runtime, _, _, _ = make_runtime()
    job = deepcopy(runtime.job)
    job['resource_contract']['policy_proposals_per_query_group'] = 20.0
    target = runtime.output.parent / 'invalid-episode'
    with pytest.raises(ValueError, match='execution contract'):
        AACRuntime(job, target, protocol_sha256='b' * 64, root=runtime.output.parent)
    assert not target.exists()
