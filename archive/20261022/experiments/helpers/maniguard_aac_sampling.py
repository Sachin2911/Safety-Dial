"""Sequential proposal sampling for a future, explicitly costed AAC-joint arm.

This harness is not installed into the live evaluator. A caller supplies an isolated
OpenPI-compatible client and a durable receipt sink. Each request uses the same copied
observation; sample zero retains the original episode ID/seed and other samples have
independent persistent RNG streams. No simulator calls or model imports occur here.

Twenty sequential calls are not a batched implementation or a faithful reproduction
of AAC's deployment latency. The caller must include all receipts in study accounting,
including a failed group. Identical sampled actions can occur despite independent keys.
"""

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from helpers.maniguard_phase2_aac import AACConfig, select_joint_prefix

SCHEMA = 'maniguard-aac-sampling-v1'
OBSERVATION_KEYS = frozenset(('observation/image_left', 'observation/wrist_image',
                              'observation/state', 'prompt'))
ROUTING_KEYS = frozenset(('study_episode_id', 'episode_seed'))


def _observation_hash(observation):
    """Bind actual common inputs, excluding the independently varied routing fields."""
    fields = {}
    for key in sorted(OBSERVATION_KEYS):
        value = observation[key]
        if key == 'prompt':
            if not isinstance(value, str):
                raise ValueError('The policy prompt must be a string')
            fields[key] = value
        else:
            array = np.asarray(value)
            if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
                raise ValueError('Policy observations must be finite numeric arrays')
            fields[key] = {'shape': list(array.shape), 'dtype': array.dtype.str,
                           'sha256': hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def stream_routes(episode_id, seed, sample_count):
    """Preserve sample zero and create reproducible, collision-checked extra streams."""
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError('A nonempty original episode ID is required')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('The original policy seed must be a uint32 integer')
    if type(sample_count) is not int or sample_count < 2:
        raise ValueError('At least two independently sampled proposals are required')
    routes = [{'study_episode_id': episode_id, 'episode_seed': seed}]
    used = {seed}
    for index in range(1, sample_count):
        nonce = 0
        while True:
            material = json.dumps([SCHEMA, seed, index, nonce], separators=(',', ':')).encode()
            extra_seed = int.from_bytes(hashlib.sha256(material).digest()[:4], 'big')
            if extra_seed not in used:
                break
            nonce += 1
        used.add(extra_seed)
        routes.append({'study_episode_id': episode_id + f'/aac-sample-{index:03d}',
                       'episode_seed': extra_seed})
    return routes


class SamplingFailure(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = deepcopy(receipt)


@dataclass(frozen=True)
class SampledPrefix:
    actions: np.ndarray
    receipt: dict


class AACSampler:
    """Sample/choose a prefix without acting; errors block further use of this instance.

    ``sink`` receives deep-copied JSON-compatible snapshots before each RPC and after
    each return/failure. A production caller must persist them durably; in-memory tests
    do not establish durable accounting. An RPC error can have consumed server work even
    without a returned proposal, recorded explicitly as unknown generation for that call.
    The fixed8 control generates the identical proposal panel and executes sample zero.
    Dispatch intents are persisted before RPC entry; a surviving dispatch_pending
    receipt without a terminal update has unknown dispatch/generation, not zero cost.
    """

    def __init__(self, client, *, config=None, schedule='aac', sink, clock=time.perf_counter):
        self.config = AACConfig() if config is None else config
        if not isinstance(self.config, AACConfig) or schedule not in ('aac', 'fixed8'):
            raise ValueError('Declare AACConfig and either aac or fixed8 scheduling')
        if not callable(sink):
            raise TypeError('Provide an explicit receipt sink')
        metadata = client.get_server_metadata()
        if (metadata.get('episode_seed_reset') != 'study_episode_id'
                or metadata.get('rng_isolation') != 'per_episode'):
            raise ValueError('Independent samples require a per-episode isolated policy server')
        self.client, self.schedule, self.sink, self.clock = client, schedule, sink, clock
        self.episode = None
        self.group_index = 0
        self.failed = False
        self.last_receipt = None

    def _emit(self, receipt):
        self.last_receipt = deepcopy(receipt)
        self.sink(deepcopy(receipt))

    def sample(self, observation, current_joints):
        if self.failed:
            raise RuntimeError('A prior sampling failure must be reconciled; no automatic retry')
        if not isinstance(observation, dict) or set(observation) != OBSERVATION_KEYS | ROUTING_KEYS:
            raise ValueError('Use only the declared policy observation and episode routing fields')
        observation = deepcopy(observation)
        identity = (observation['study_episode_id'], observation['episode_seed'])
        routes = stream_routes(*identity, self.config.sample_count)
        if self.episode is not None and identity != self.episode:
            raise ValueError('Use a separate sampler per physical episode')
        joints = np.asarray(current_joints).copy()
        if joints.shape != (7,) or joints.dtype.kind not in 'fiu' or not np.isfinite(joints).all():
            raise ValueError('Seven finite measured arm-joint positions are required')
        observation_sha = _observation_hash(observation)
        self.episode = identity
        receipt = {'schema': SCHEMA, 'episode_id': identity[0], 'group_index': self.group_index,
                   'schedule': self.schedule, 'settings': self.config.as_dict(),
                   'observation_sha256': observation_sha, 'current_joints': joints.tolist(),
                   'status': 'started', 'dispatch_intents': 0, 'rpc_attempts': 0, 'rpc_returns': 0, 'valid_proposals': 0,
                   'returned_action_count': 0, 'policy_rpc_seconds': 0.0,
                   'unknown_generation_calls': 0, 'calls': []}
        group_started = self.clock()
        current = None
        stage = 'preparation'
        try:
            proposals = []
            native_length = None
            for index, route in enumerate(routes):
                request = deepcopy(observation)
                request.update(route)
                current = {'sample_index': index, **route, 'status': 'dispatch_pending',
                           'rpc_attempted': False, 'observation_sha256': observation_sha}
                receipt['calls'].append(current)
                receipt['dispatch_intents'] += 1
                stage = 'receipt_before_rpc'
                self._emit(receipt)
                started = self.clock()
                returned = False
                current.update(status='started', rpc_attempted=True)
                receipt['rpc_attempts'] += 1
                stage = 'rpc'
                try:
                    response = self.client.infer(request)
                    returned = True
                    receipt['rpc_returns'] += 1
                finally:
                    elapsed = self.clock() - started
                    current['elapsed_seconds'] = elapsed
                    receipt['policy_rpc_seconds'] += elapsed
                    if not returned:
                        receipt['unknown_generation_calls'] += 1
                current['status'] = 'returned_unvalidated'
                stage = 'validation'
                raw = np.asarray(response['actions'])
                if raw.ndim == 1:
                    raw = raw[None, :]
                if raw.ndim != 2 or raw.shape[1] != 8 or not len(raw) or raw.dtype.kind not in 'fiu':
                    raise ValueError('Policy must return a nonempty numeric [T,8] action chunk')
                current['returned_actions'] = len(raw)
                receipt['returned_action_count'] += len(raw)
                if native_length is not None and len(raw) != native_length:
                    raise ValueError('Independent proposals have unequal native chunk lengths')
                native_length = len(raw)
                # Match the benchmark's float32 action boundary; never inspect a suffix
                # outside the shared sixteen-action prefix for scheduling decisions.
                with np.errstate(over='ignore', invalid='ignore'):
                    prefix = raw[:self.config.max_prefix].astype(np.float32)
                if not np.isfinite(prefix).all():
                    raise ValueError('Accessible action prefix is nonfinite after float32 conversion')
                proposals.append(prefix.copy())
                receipt['valid_proposals'] += 1
                current.update(status='valid', accessible_actions=len(prefix),
                               prefix_sha256=hashlib.sha256(prefix.tobytes()).hexdigest())
                stage = 'receipt_after_rpc'
                self._emit(receipt)
            stage = 'selection'
            samples = np.stack(proposals)
            if self.schedule == 'aac':
                selection = select_joint_prefix(samples, joints, self.config)
                actions, diagnostics = selection.prefix, selection.record()
            else:
                actions = samples[0, :min(8, samples.shape[1])].copy()
                diagnostics = {'method': 'fixed8_with_aac_sampling_cost', 'sample_index': 0,
                               'horizon': len(actions), 'sample_count': len(samples)}
            receipt.update(status='completed', selection=diagnostics,
                           elapsed_seconds=self.clock() - group_started)
            stage = 'receipt_completed'
            self._emit(receipt)
            self.group_index += 1
            return SampledPrefix(actions.copy(), deepcopy(receipt))
        except BaseException as error:
            self.failed = True
            if current is not None and stage in ('receipt_before_rpc', 'rpc', 'validation'):
                current['status'] = {'receipt_before_rpc': 'not_dispatched',
                                     'rpc': 'failed', 'validation': 'invalid'}[stage]
            receipt.update(status='failed', failure_stage=stage, error={'type': type(error).__name__, 'message': str(error)},
                           elapsed_seconds=self.clock() - group_started)
            # Preserve the original error if the receipt sink itself is unavailable.
            self.last_receipt = deepcopy(receipt)
            try:
                self.sink(deepcopy(receipt))
            except Exception as sink_error:
                receipt['sink_error'] = {'type': type(sink_error).__name__,
                                         'message': str(sink_error)}
                self.last_receipt = deepcopy(receipt)
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise error from sink_error
                raise SamplingFailure(str(error), receipt) from sink_error
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise SamplingFailure(str(error), receipt) from error
