"""Dedicated AAC execution and replay; no CLIP, simulator or service imports at import time.

An authorized main child must construct/install one runtime before executing the
instrumented benchmark. This module does not authorize jobs or acquire a lease. The
source hook replaces the ordinary query, so sample zero is one of twenty paid RPCs.
Per-RPC intent receipts and accessible float32 arrays are fsynced before continuation.
A process lost after dispatch intent has unresolved work and is never resumable here.
"""

import atexit
import hashlib
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from helpers.maniguard_aac_sampling import AACSampler, _observation_hash, stream_routes
from helpers.maniguard_main_protocol import (
    WORK_COUNTS,
    _comparator_resources,
    _file,
    _record,
    episode_resource_envelope,
)
from helpers.maniguard_phase2_aac import load_config, select_joint_prefix
from helpers.maniguard_phase2_optim import record_hash

SCHEMA = 'maniguard-aac-runtime-v1'
ROOT = Path(__file__).resolve().parents[2]
_ACTIVE = None


def install_runtime(runtime):
    """Explicit child-bootstrap handoff, after its own authorization/lease checks."""
    global _ACTIVE
    if not isinstance(runtime, AACRuntime) or _ACTIVE is not None:
        raise ValueError('Install exactly one dedicated AAC runtime per child process')
    _ACTIVE = runtime


def get_runtime():
    if _ACTIVE is None:
        raise RuntimeError('The authorized child must install its AAC runtime before evaluator import')
    return _ACTIVE


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _sync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False, default=_json_value)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(Path(path).parent)


def _write_array(path, value):
    with Path(path).open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(Path(path).parent)


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif hasattr(value, 'cpu'):
        value = value.cpu().numpy()
    return np.asarray(value)


def measured_joints(robot):
    indices = _numpy(robot.arm_control_idx[robot.default_arm]).astype(int)
    joints = _numpy(robot.get_joint_positions())[indices].copy()
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError('AAC requires seven finite measured arm joints')
    return joints


class _ArchivingClient:
    def __init__(self, runtime, client):
        self.runtime, self.client = runtime, client

    def get_server_metadata(self):
        return self.client.get_server_metadata()

    def infer(self, request):
        runtime = self.runtime
        if runtime.counts['policy_queries'] >= runtime.query_cap:
            raise RuntimeError('AAC policy RPC reservation exhausted before dispatch')
        if _observation_hash(request) != runtime.group_observation_hash:
            raise ValueError('A proposal request changed the common group observation')
        index = len(runtime.rpc_entries)
        route = stream_routes(runtime.job['episode_id'], runtime.native_seed, 20)[index]
        if any(request[key] != value for key, value in route.items()):
            raise ValueError('Proposal request differs from its persistent episode RNG route')
        runtime.counts['policy_queries'] += 1
        runtime.rpc_entries.append(index)
        runtime.write('rpc_entering', group=runtime.group_index, sample_index=index,
                      step=runtime.group_step, route=route)
        runtime.ledger()
        started = runtime.clock()
        try:
            response = self.client.infer(request)
        except BaseException as error:
            runtime.write('rpc_failed', group=runtime.group_index, sample_index=index,
                          error_type=type(error).__name__)
            raise
        finally:
            runtime.counts['policy_query_seconds'] += runtime.clock() - started
            runtime.ledger()
        runtime.write('rpc_returned', group=runtime.group_index, sample_index=index)
        raw = np.asarray(response['actions'])
        if raw.ndim == 1:
            raw = raw[None, :]
        # Malformed responses remain failed work; never manufacture a proposal file.
        if raw.ndim == 2 and raw.shape[1] == 8 and len(raw) and raw.dtype.kind in 'fiu':
            with np.errstate(over='ignore', invalid='ignore'):
                accessible = raw[:16].astype(np.float32)
            path = runtime.group_dir / f'proposal-{index:03d}.npy'
            _write_array(path, accessible)
            runtime.write('proposal_archived', group=runtime.group_index, sample_index=index,
                          path=str(path.relative_to(runtime.output)), sha256=_hash(path),
                          native_shape=list(raw.shape), native_dtype=raw.dtype.str,
                          accessible_sha256=hashlib.sha256(accessible.tobytes()).hexdigest())
        return response


class AACRuntime:
    """One physical episode, immutable evidence, and no automatic restart/retry.

    ``job`` is the exact compiled main job, already authorized by the caller.
    ``query_cap`` can tighten its reserved successful-query envelope for a test or
    external budget meter; it cannot expand it. Physics callbacks count completed
    ticks; they do not constitute a hard pre-advance physics limit.
    """

    def __init__(self, job, output, *, protocol_sha256, root=ROOT, query_cap=None,
                 clock=time.monotonic):
        self.job, self.output, self.clock = deepcopy(job), Path(output), clock
        controller = _record(job['controller'], root)
        resources = _comparator_resources(controller, root)
        profile = resources['profile']
        if (profile not in ('aac_joint20', 'aac_fixed8_joint20')
                or job.get('actor_sha256') is not None or job.get('run_kind') != 'fixed'
                or job.get('stochastic') is not False or controller.get('deterministic') is not True
                or job.get('controller_sha256') != controller['sha256']
                or record_hash(job.get('resource_contract')) != record_hash(resources)
                or job.get('resource_contract_sha256') != record_hash(resources)
                or record_hash(job.get('resource_envelope')) != record_hash(
                    episode_resource_envelope(job['action_horizon'], resources))):
            raise ValueError('AAC job differs from its supported fixed-controller execution contract')
        if (not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64
                or any(char not in '0123456789abcdef' for char in protocol_sha256)):
            raise ValueError('An explicit frozen-protocol content hash is required')
        self.config = load_config(_file(controller['aac_config'], root))
        self.schedule = 'aac' if profile == 'aac_joint20' else 'fixed8'
        maximum = job['resource_envelope']['successful_work']['policy_queries']
        self.query_cap = maximum if query_cap is None else query_cap
        if type(self.query_cap) is not int or not 0 <= self.query_cap <= maximum:
            raise ValueError('Query cap cannot enlarge the exact job reservation')
        self.output.mkdir(parents=True, exist_ok=True)
        self.source_dir = self.output / 'aac_source'
        self.source_dir.mkdir()
        sources = {}
        for name in ('maniguard_aac_runtime.py', 'maniguard_aac_sampling.py', 'maniguard_phase2_aac.py'):
            source = Path(__file__).parent / name
            destination = self.source_dir / name
            with destination.open('xb') as stream:
                stream.write(source.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            sources[name] = _hash(destination)
        self.source_hashes = sources
        self.started = clock()
        self.counts = dict.fromkeys(WORK_COUNTS, 0)
        self.counts.update(policy_query_seconds=0.0, physics_seconds=0.0,
                           env_step_attempts=0, env_steps_returned=0, policy_action_attempts=0)
        self.phase, self.finished, self.broken = 'initialization', False, False
        self.physics_accounting_attached = False
        self.physics_subscription = None
        self.client = self.sampler = None
        self.scene = self.env = self.robot = None
        self.group_index = -1
        self.pending_action = None
        self.selected = None
        self.executed_in_group = 0
        self.receipt_index = 0
        self.last_receipt = None
        self.trace = (self.output / 'aac-trace.jsonl').open('x')
        self.event_index = 0
        manifest = {'schema': SCHEMA, 'job': self.job, 'protocol_sha256': protocol_sha256,
                    'settings': self.config.as_dict(), 'schedule': self.schedule,
                    'source_hashes': sources, 'query_cap': self.query_cap,
                    'authorization_scope': 'Caller must verify main claim, lease and reserved budget.',
                    'physics_limit_scope': 'Completed callbacks; no hard pre-advance ceiling.'}
        manifest['sha256'] = record_hash(manifest)
        _write_json(self.output / 'aac-runtime.json', manifest)
        self.write('job', job=self.job, protocol_sha256=protocol_sha256)
        self.ledger()
        atexit.register(self.close)

    def write(self, kind, **values):
        record = {'type': kind, 'index': self.event_index, **values}
        self.trace.write(json.dumps(record, allow_nan=False, default=_json_value) + '\n')
        self.trace.flush()
        os.fsync(self.trace.fileno())
        self.event_index += 1

    def ledger(self):
        record = {**self.counts, 'phase': self.phase, 'elapsed_seconds': self.clock() - self.started,
                  'physics_accounting_attached': self.physics_accounting_attached,
                  'last_receipt_status': None if self.last_receipt is None else self.last_receipt['status']}
        temporary = self.output / 'aac-ledger.tmp'
        with temporary.open('w') as stream:
            json.dump(record, stream, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.output / 'aac-ledger.json')
        _sync_dir(self.output)

    def validate_config(self, cfg):
        if (cfg.random_policy or not cfg.use_openpi_client or cfg.action_dim != 8
                or cfg.ik_eef_to_joint or not cfg.gripper_binarize
                or cfg.max_steps != self.job['action_horizon'] or cfg.seed != self.job['seed']
                or list(cfg.scenes) != [self.job['scene']]):
            raise ValueError('Benchmark configuration differs from the compiled AAC episode')

    def connect_policy(self, cfg):
        from helpers.maniguard_client import StudyClient
        client = StudyClient(host=cfg.host, port=cfg.port)
        self.validate_policy(client, 'openpi')
        return client, 'openpi'

    def validate_policy(self, policy, client_type):
        if self.client is not None or client_type != 'openpi':
            raise ValueError('Bind exactly one isolated OpenPI client per physical episode')
        metadata = policy.get_server_metadata()
        checkpoint = self.job['checkpoint']
        expected = {'serve_config': checkpoint['config'], 'checkpoint_revision': checkpoint['revision'],
                    'episode_seed_reset': 'study_episode_id', 'rng_isolation': 'per_episode'}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError('AAC policy handshake differs from the pinned checkpoint/RNG contract')
        self.client = policy
        self.write('policy_metadata', metadata=metadata)

    def start(self, scene, obs, env, robot, monitor, episode_seed):
        if (self.sampler is not None or self.finished or self.broken
                or scene['name'] != self.job['scene'] or self.client is None):
            raise ValueError('Exactly one matching physical episode is required')
        if type(episode_seed) is not int or not 0 <= episode_seed < 2**32:
            raise ValueError('The native derived episode seed must be uint32')
        if obs.get('episode_seed') != episode_seed:
            raise ValueError('Initial observation lost the native episode seed')
        self.scene, self.env, self.robot, self.native_seed = scene, env, robot, episode_seed
        self.sampler = AACSampler(_ArchivingClient(self, self.client), config=self.config,
                                  schedule=self.schedule, sink=self._receipt, clock=self.clock)
        self.phase = 'policy'
        self.write('initial', scene=scene, native_episode_seed=episode_seed, state=obs['states'])
        self.observed(0, obs, monitor)
        self.ledger()

    def _receipt(self, receipt):
        path = self.group_dir / f'receipt-{self.receipt_index:04d}.json'
        record = {'step': self.group_step, 'source_hashes': self.source_hashes, 'receipt': receipt}
        _write_json(path, record)
        self.write('sampler_receipt', group=self.group_index, step=self.group_step,
                   path=str(path.relative_to(self.output)), sha256=_hash(path))
        self.receipt_index += 1
        self.last_receipt = deepcopy(receipt)
        self.ledger()

    def query_policy(self, policy, obs, client_type, cfg, step, robot, remap):
        if (self.broken or self.finished or self.sampler is None or policy is not self.client
                or client_type != 'openpi' or step != self.counts['policy_control_steps']
                or self.pending_action is not None
                or (self.selected is not None and self.executed_in_group != len(self.selected))):
            raise ValueError('AAC query must follow complete consumption of its preceding prefix')
        if step >= self.job['action_horizon'] or obs.get('episode_seed') != self.native_seed:
            raise ValueError('AAC query lies outside the episode horizon or native RNG route')
        try:
            request = remap(obs, cfg)
            request['study_episode_id'] = self.job['episode_id']
            self.group_index += 1
            self.group_step, self.rpc_entries, self.receipt_index = step, [], 0
            self.group_dir = self.output / f'aac-group-{self.group_index:04d}'
            self.group_dir.mkdir()
            self.group_observation_hash = _observation_hash(request)
            joints = measured_joints(robot)
            for field, name in (('observation/image_left', 'overview'), ('observation/wrist_image', 'wrist'),
                                ('observation/state', 'state')):
                _write_array(self.group_dir / (name + '.npy'), request[field])
            group = {'step': step, 'group_index': self.group_index, 'episode_id': self.job['episode_id'],
                     'native_episode_seed': self.native_seed, 'prompt': request['prompt'],
                     'observation_sha256': self.group_observation_hash, 'current_joints': joints.tolist(),
                     'source_hashes': self.source_hashes}
            _write_json(self.group_dir / 'group.json', group)
            self.write('query_group', **group)
            sampled = self.sampler.sample(request, joints)
            self.selected = sampled.actions.copy()
            self.executed_in_group = 0
            _write_array(self.group_dir / 'selected.npy', self.selected)
            self.write('chunk', step=step, group=self.group_index, actions=self.selected,
                       prefix_file=str((self.group_dir / 'selected.npy').relative_to(self.output)),
                       prefix_file_sha256=_hash(self.group_dir / 'selected.npy'))
            return self.selected.copy()
        except BaseException:
            self.broken = True
            self.phase = 'sampling_failed'
            self.ledger()
            raise

    def before_action(self, step, ci, raw, converted, clipped, low, high):
        if (self.broken or self.finished or self.selected is None or self.pending_action is not None
                or step != self.counts['policy_control_steps'] or ci != self.executed_in_group
                or ci >= len(self.selected) or step >= self.job['action_horizon']
                or np.asarray(raw).dtype != np.float32
                or np.asarray(raw).tobytes() != self.selected[ci].tobytes()):
            raise ValueError('AAC action does not consume sample zero exactly once and in order')
        canonical = np.asarray(raw).copy()
        canonical[-1] = np.sign(canonical[-1]) if abs(canonical[-1]) > .01 else -1.0
        expected = np.clip(canonical, low, high)
        if not np.array_equal(converted, canonical) or not np.array_equal(clipped, expected):
            raise ValueError('AAC action conversion differs from benchmark gripper/clipping semantics')
        self.pending_action = {'step': step, 'action': np.asarray(clipped).copy()}
        self.counts['policy_action_attempts'] += 1
        self.write('action_intent', step=step, group=self.group_index, chunk_index=ci,
                   raw=raw, converted=converted, clipped=clipped, low=low, high=high)
        self.ledger()

    def applied(self, step, executed):
        if (self.pending_action is None or step != self.pending_action['step']
                or not np.array_equal(executed, self.pending_action['action'])):
            raise ValueError('Applied action lacks its unique preceding AAC action intent')
        self.counts['policy_control_steps'] += 1
        self.executed_in_group += 1
        self.write('executed', step=step, action=executed, group=self.group_index)
        self.pending_action = None
        self.ledger()

    def observed(self, step, obs, monitor):
        state = None if monitor is None else monitor.recording_state(step)
        self.write('observed', step=step, state=obs['states'], monitor=state)
        if self.env is not None:
            from helpers.maniguard_phase2_outcomes import evaluation_state
            extra = evaluation_state(self.env, self.scene, self.job['family'], step)
            if extra:
                self.write('evaluation_only', **extra)

    def install_accounting(self, og):
        # Lazy simulator imports: importing/testing this module never initializes OG.
        from omnigibson import simulator
        original_launch = simulator._launch_app

        def launch(*args, **kwargs):
            self.phase = 'app_startup'
            self.ledger()
            app = original_launch(*args, **kwargs)
            from omni.physx import get_physx_interface

            def physics_step(dt):
                self.counts['physics_steps'] += 1
                self.counts['physics_seconds'] += float(dt)
                if self.counts['physics_steps'] % 120 == 0:
                    self.ledger()

            self.physics_subscription = get_physx_interface().subscribe_physics_step_events(physics_step)
            self.physics_accounting_attached = True
            self.phase = 'scene_initialization'
            self.ledger()
            return app

        simulator._launch_app = launch
        original_step = og.Environment.step

        def step(env, *args, **kwargs):
            self.counts['env_step_attempts'] += 1
            result = original_step(env, *args, **kwargs)
            self.counts['env_steps_returned'] += 1
            return result

        og.Environment.step = step

    def finish(self, result):
        if self.finished:
            raise ValueError('AAC physical episode cannot finish twice')
        self.finished = True
        self.phase = result['status']
        self.write('result', result=result)
        self.ledger()

    def close(self):
        if self.trace.closed:
            return
        if not self.finished:
            self.phase = 'process_exited_without_result'
            self.write('incomplete')
        self.ledger()
        self.trace.close()


def replay_group(path, *, config, schedule):
    """Recompute a completed group from saved accessible arrays, without policy calls."""
    path = Path(path)
    group = json.loads((path / 'group.json').read_text())
    snapshots = [json.loads(item.read_text()) for item in sorted(path.glob('receipt-*.json'))]
    if not snapshots:
        raise ValueError('Group has no durable dispatch accounting')
    for snapshot in snapshots:
        if snapshot['step'] != group['step'] or snapshot['source_hashes'] != group['source_hashes']:
            raise ValueError('Receipt differs from its group step/source binding')
    if len(snapshots) != 41:
        raise ValueError('Expected durable intent/return snapshots for twenty calls plus completion')
    for index in range(20):
        before, after = (snapshots[2 * index + offset]['receipt'] for offset in (0, 1))
        if (before['dispatch_intents'] != index + 1 or before['rpc_attempts'] != index
                or before['rpc_returns'] != index or before['valid_proposals'] != index
                or len(before['calls']) != index + 1 or before['calls'][-1]['status'] != 'dispatch_pending'
                or after['rpc_attempts'] != index + 1 or after['rpc_returns'] != index + 1
                or after['valid_proposals'] != index + 1 or after['calls'][-1]['status'] != 'valid'):
            raise ValueError('Per-RPC durable intent/return ordering or accounting differs')
    receipt = snapshots[-1]['receipt']
    if (receipt['status'] != 'completed' or receipt['rpc_attempts'] != 20
            or receipt['rpc_returns'] != 20 or receipt['valid_proposals'] != 20
            or receipt['dispatch_intents'] != 20 or receipt['unknown_generation_calls'] != 0
            or receipt['episode_id'] != group['episode_id'] or receipt['group_index'] != group['group_index']
            or receipt['schedule'] != schedule or receipt['settings'] != config.as_dict()
            or receipt['current_joints'] != group['current_joints']):
        raise ValueError('Group is incomplete or differs from its declared sampling contract')
    request = {'observation/image_left': np.load(path / 'overview.npy', allow_pickle=False),
               'observation/wrist_image': np.load(path / 'wrist.npy', allow_pickle=False),
               'observation/state': np.load(path / 'state.npy', allow_pickle=False),
               'prompt': group['prompt']}
    digest = _observation_hash(request)
    if digest != group['observation_sha256'] or digest != receipt['observation_sha256']:
        raise ValueError('Archived observation differs from the common policy request')
    samples = []
    routes = stream_routes(group['episode_id'], group['native_episode_seed'], 20)
    if len(receipt['calls']) != 20:
        raise ValueError('Expected exactly twenty individually accounted calls')
    for index, call in enumerate(receipt['calls']):
        sample = np.load(path / f'proposal-{index:03d}.npy', allow_pickle=False)
        if (sample.dtype != np.float32 or sample.ndim != 2 or sample.shape[1] != 8
                or not 1 <= len(sample) <= 16 or not np.isfinite(sample).all()
                or call['sample_index'] != index or call['status'] != 'valid'
                or call['rpc_attempted'] is not True or call['observation_sha256'] != digest
                or any(call[key] != value for key, value in routes[index].items())
                or call['accessible_actions'] != len(sample)
                or call['prefix_sha256'] != hashlib.sha256(sample.tobytes()).hexdigest()):
            raise ValueError('Saved proposal differs from its per-RPC receipt')
        samples.append(sample)
    native_lengths = {call['returned_actions'] for call in receipt['calls']}
    if (len(native_lengths) != 1 or receipt['returned_action_count'] != 20 * next(iter(native_lengths))
            or any(len(sample) != min(16, next(iter(native_lengths))) for sample in samples)):
        raise ValueError('Proposal native lengths or returned action accounting differ')
    panel = np.stack(samples)
    if schedule == 'aac':
        selection = select_joint_prefix(panel, np.asarray(group['current_joints']), config)
        prefix, diagnostics = selection.prefix, selection.record()
    else:
        prefix = panel[0, :min(8, panel.shape[1])].copy()
        diagnostics = {'method': 'fixed8_with_aac_sampling_cost', 'sample_index': 0,
                       'horizon': len(prefix), 'sample_count': 20}
    saved = np.load(path / 'selected.npy', allow_pickle=False)
    if (saved.dtype != prefix.dtype or saved.shape != prefix.shape or saved.tobytes() != prefix.tobytes()
            or record_hash(receipt['selection']) != record_hash(diagnostics)):
        raise ValueError('Selected prefix or diagnostics do not replay from the proposal panel')
    return {'group': group, 'receipt': receipt, 'actions': prefix}


def replay_episode(output):
    """Audit AAC-specific evidence; this is not a monitor/outcome or physics audit.

    Invalid/partial evidence remains invalid. Persisted counters are retained as lower
    bounds when dispatch/execution lacks terminal evidence; missing fields stay unknown.
    A future main result reader must combine this with raw benchmark/physical checks.
    """
    output = Path(output)
    manifest = json.loads((output / 'aac-runtime.json').read_text())
    ledger = json.loads((output / 'aac-ledger.json').read_text())
    reasons, groups, events = [], [], []
    try:
        if manifest['sha256'] != record_hash({k: v for k, v in manifest.items() if k != 'sha256'}):
            raise ValueError('Runtime manifest content hash differs')
        for name, expected in manifest['source_hashes'].items():
            if _hash(output / 'aac_source' / name) != expected:
                raise ValueError('Archived AAC implementation changed')
        events = [json.loads(line) for line in (output / 'aac-trace.jsonl').read_text().splitlines()]
        if [event['index'] for event in events] != list(range(len(events))):
            raise ValueError('Trace events are missing, duplicated or reordered')
        for event in events:
            if event['type'] in ('sampler_receipt', 'proposal_archived'):
                path = output / event['path']
                if not path.resolve().is_relative_to(output.resolve()) or _hash(path) != event['sha256']:
                    raise ValueError('Receipt/proposal archive differs from its durable trace hash')
        from helpers.maniguard_phase2_aac import AACConfig
        config = AACConfig(**manifest['settings'])
        for path in sorted(output.glob('aac-group-*')):
            groups.append(replay_group(path, config=config, schedule=manifest['schedule']))
        if [item['group']['group_index'] for item in groups] != list(range(len(groups))):
            raise ValueError('Query groups are missing or duplicated')
        for item in groups:
            if (item['group']['episode_id'] != manifest['job']['episode_id']
                    or item['group']['source_hashes'] != manifest['source_hashes']):
                raise ValueError('Query group differs from the episode/source contract')
        for kind in ('rpc_entering', 'rpc_returned', 'proposal_archived'):
            calls = [event for event in events if event['type'] == kind]
            if [(event['group'], event['sample_index']) for event in calls] != [
                    (group, index) for group in range(len(groups)) for index in range(20)]:
                raise ValueError('Trace RPC/archive inventory differs from sampled groups')
        if len([event for event in events if event['type'] == 'query_group']) != len(groups):
            raise ValueError('Trace query inventory differs from archived groups')
        groups_by_index = {item['group']['group_index']: item for item in groups}
        actions, cursor, active, pending = 0, 0, None, None
        for event in events:
            kind = event['type']
            if kind == 'query_group':
                if (event['step'] != actions or pending is not None
                        or (active is not None and cursor != len(active['actions']))):
                    raise ValueError('Query did not follow exact prefix consumption')
                active, cursor = groups_by_index[event['group_index']], 0
                for key, value in active['group'].items():
                    if event.get(key) != value:
                        raise ValueError('Query event differs from archived group')
            elif kind == 'action_intent':
                raw = np.asarray(event['raw'], dtype=np.float32)
                if (active is None or pending is not None or event['step'] != actions
                        or event['chunk_index'] != cursor or cursor >= len(active['actions'])
                        or raw.tobytes() != active['actions'][cursor].tobytes()):
                    raise ValueError('Action did not consume sample zero once in order')
                converted = raw.copy()
                converted[-1] = np.sign(converted[-1]) if abs(converted[-1]) > .01 else -1.0
                clipped = np.clip(converted, event['low'], event['high'])
                if (not np.array_equal(event['converted'], converted)
                        or not np.array_equal(event['clipped'], clipped)):
                    raise ValueError('Recorded gripper conversion/clipping changed')
                pending = event
            elif kind == 'executed':
                if (pending is None or event['step'] != actions
                        or event['action'] != pending['clipped']):
                    raise ValueError('Executed action lacks one matching intent')
                actions += 1
                cursor += 1
                pending = None
        terminals = [event for event in events if event['type'] == 'result']
        if (pending is not None or len(terminals) != 1
                or terminals[0]['result']['status'] != 'completed' or ledger['phase'] != 'completed'):
            raise ValueError('Episode has pending execution or no successful terminal result')
        if actions > manifest['job']['action_horizon'] or ledger['policy_action_attempts'] != actions:
            raise ValueError('Action horizon or attempted-action count differs')
        if (ledger['policy_control_steps'] != actions or ledger['policy_queries'] != 20 * len(groups)
                or ledger['policy_queries'] > manifest['query_cap']):
            raise ValueError('Action/query totals differ from individually replayed work')
        if any(ledger[name] != 0 for name in WORK_COUNTS if name.startswith('encoder_')
               or name in ('adapter_decisions', 'actor_rpc_attempts')):
            raise ValueError('AAC unexpectedly used supplementary encoder/actor resources')
    except (ValueError, KeyError, OSError, TypeError) as error:
        reasons.append(str(error))
    counts = {key: ledger[key] for key in WORK_COUNTS
              if type(ledger.get(key)) is int and ledger[key] >= 0}
    missing = [key for key in WORK_COUNTS if key not in counts]
    if missing:
        reasons.append('Missing or malformed counters: ' + ', '.join(missing))
    unresolved = 0
    for group_path in output.glob('aac-group-*'):
        receipts = sorted(group_path.glob('receipt-*.json'))
        if receipts:
            try:
                final = json.loads(receipts[-1].read_text())['receipt']
                unresolved += sum(call.get('status') == 'dispatch_pending'
                                  for call in final.get('calls', []))
            except (ValueError, KeyError, OSError):
                unresolved += 1
    pending_actions = max(0, sum(event.get('type') == 'action_intent' for event in events)
                          - sum(event.get('type') == 'executed' for event in events))
    unknown = sorted(set(missing + (['policy_queries'] if unresolved else [])
                         + (['policy_control_steps', 'physics_steps'] if pending_actions else [])))
    return {'valid': not reasons, 'invalid_reasons': reasons, 'groups_replayed': len(groups),
            'counts': counts, 'missing_cost_fields': missing, 'costs_complete': not reasons,
            'unknown_cost_fields': unknown, 'unresolved_dispatches': unresolved,
            'unresolved_action_intents': pending_actions,
            'count_scope': 'Exact AAC calls/actions only when valid; otherwise persisted unverified counts.',
            'accounting_scope': 'AAC calls/actions and no supplementary encoder; physics/process audit is separate.',
            'monitor_outcomes_audited': False, 'physics_counts_independently_audited': False}
