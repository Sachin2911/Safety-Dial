"""Proposal-cost and RNG-isolation checks; no model, network or simulator calls."""
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_aac_sampling import AACSampler, SamplingFailure, stream_routes
from helpers.maniguard_parallel_policy import IsolatedEpisodePolicy
from helpers.maniguard_phase2_aac import AACConfig


def observation():
    return {'observation/image_left': np.zeros((4, 4, 3), dtype=np.uint8),
            'observation/wrist_image': np.ones((4, 4, 3), dtype=np.uint8),
            'observation/state': np.zeros(8, dtype=np.float32), 'prompt': 'Move the object.',
            'study_episode_id': 'episode-one', 'episode_seed': 17}


class FakeModel:
    def infer(self, observation):
        return {'actions': self._rng.normal(size=(32, 8)).astype(np.float32)}


class Client:
    def __init__(self, fail_at=None, mutation=False):
        self.policy = IsolatedEpisodePolicy(FakeModel(), np.random.default_rng)
        self.requests = []
        self.responses = []
        self.fail_at = fail_at
        self.mutation = mutation

    def get_server_metadata(self):
        return {'episode_seed_reset': 'study_episode_id', 'rng_isolation': 'per_episode'}

    def infer(self, request):
        self.requests.append(deepcopy(request))
        if len(self.requests) == self.fail_at:
            raise RuntimeError('simulated RPC failure')
        result = self.policy.infer(request)
        self.responses.append(deepcopy(result))
        if self.mutation:
            request['observation/image_left'][:] = 99
        return result


def test_extra_samples_preserve_original_rng_stream_over_multiple_queries():
    baseline, sampled = Client(), Client()
    sink = []
    sampler = AACSampler(sampled, sink=sink.append)
    obs = observation()
    for _ in range(3):
        expected = baseline.infer(obs)['actions']
        result = sampler.sample(obs, np.zeros(7))
        np.testing.assert_array_equal(result.actions, expected[:len(result.actions)])
        assert result.receipt['rpc_attempts'] == 20
        assert result.receipt['rpc_returns'] == 20
        assert result.receipt['valid_proposals'] == 20
        assert result.receipt['returned_action_count'] == 640
        assert result.receipt['unknown_generation_calls'] == 0
    assert [x['group_index'] for x in sink if x['status'] == 'completed'] == [0, 1, 2]
    assert len(sampled.requests) == 60
    assert len({x['episode_seed'] for x in sampled.requests}) == 20
    for index in range(20):
        assert sampled.requests[index]['study_episode_id'] == sampled.requests[index+20]['study_episode_id']


def test_same_copied_observation_and_original_inputs_survive_mutating_client():
    client = Client(mutation=True)
    obs = observation()
    result = AACSampler(client, sink=lambda _: None).sample(obs, np.zeros(7))
    assert not obs['observation/image_left'].any()
    assert all(not x['observation/image_left'].any() for x in client.requests)
    assert len({x['observation_sha256'] for x in result.receipt['calls']}) == 1
    assert client.requests[0]['study_episode_id'] == obs['study_episode_id']
    assert client.requests[0]['episode_seed'] == obs['episode_seed']


def test_failed_query_preserves_paid_partial_work_and_blocks_retry():
    client = Client(fail_at=4)
    receipts = []
    sampler = AACSampler(client, sink=receipts.append)
    with pytest.raises(SamplingFailure) as caught:
        sampler.sample(observation(), np.zeros(7))
    r = caught.value.receipt
    assert (r['rpc_attempts'], r['rpc_returns'], r['valid_proposals']) == (4, 3, 3)
    assert r['returned_action_count'] == 96
    assert r['unknown_generation_calls'] == 1
    assert r['calls'][-1]['status'] == 'failed'
    assert r['policy_rpc_seconds'] >= 0
    assert receipts[-1]['status'] == 'failed'
    with pytest.raises(RuntimeError, match='no automatic retry'):
        sampler.sample(observation(), np.zeros(7))
    assert len(client.requests) == 4


def test_fixed_control_pays_same_queries_and_executes_original_eight_actions():
    client = Client()
    result = AACSampler(client, schedule='fixed8', sink=lambda _: None).sample(observation(), np.zeros(7))
    assert result.receipt['rpc_attempts'] == 20
    np.testing.assert_array_equal(result.actions, client.responses[0]['actions'][:8])
    assert result.receipt['selection']['method'] == 'fixed8_with_aac_sampling_cost'


def test_short_chunks_are_never_padded_or_quantized():
    class ShortClient(Client):
        def infer(self, request):
            result = super().infer(request)
            result['actions'] = result['actions'][:3]
            return result
    client = ShortClient()
    config = AACConfig(movement_threshold=1e6)
    result = AACSampler(client, config=config, sink=lambda _: None).sample(observation(), np.zeros(7))
    assert result.actions.shape == (3, 8)
    assert result.receipt['returned_action_count'] == 60
    assert result.receipt['selection']['horizon'] == 3


def test_equal_samples_allowed_with_independent_streams():
    class ConstantClient(Client):
        def infer(self, request):
            self.requests.append(deepcopy(request))
            return {'actions': np.zeros((16, 8), dtype=np.float32)}
    client = ConstantClient()
    result = AACSampler(client, sink=lambda _: None).sample(observation(), np.zeros(7))
    assert result.receipt['status'] == 'completed'
    assert len({x['episode_seed'] for x in client.requests}) == 20


@pytest.mark.parametrize('bad_response', [np.zeros((4, 7)), np.zeros((0, 8)),
                                           np.full((16, 8), np.nan), np.full((16, 8), 1e300)])
def test_bad_return_is_not_a_valid_proposal(bad_response):
    class BadClient(Client):
        def infer(self, request):
            return {'actions': bad_response}
    with pytest.raises(SamplingFailure) as caught:
        AACSampler(BadClient(), sink=lambda _: None).sample(observation(), np.zeros(7))
    r = caught.value.receipt
    assert r['rpc_attempts'] == r['rpc_returns'] == 1
    assert r['valid_proposals'] == 0
    assert r['unknown_generation_calls'] == 0


def test_unequal_chunk_lengths_refused_and_both_returns_counted():
    class UnequalClient(Client):
        def infer(self, request):
            result = super().infer(request)
            if len(self.requests) == 2:
                result['actions'] = result['actions'][:8]
            return result
    with pytest.raises(SamplingFailure) as caught:
        AACSampler(UnequalClient(), sink=lambda _: None).sample(observation(), np.zeros(7))
    assert caught.value.receipt['rpc_returns'] == 2
    assert caught.value.receipt['returned_action_count'] == 40


def test_unisolated_server_refused_before_requests():
    client = Client()
    client.get_server_metadata = lambda: {'episode_seed_reset': 'study_episode_id', 'rng_isolation': 'active_episode'}
    with pytest.raises(ValueError, match='isolated'):
        AACSampler(client, sink=lambda _: None)
    assert not client.requests


def test_privileged_extra_observations_rejected_before_queries():
    client = Client()
    obs = observation() | {'goal_success': True}
    with pytest.raises(ValueError, match='only the declared'):
        AACSampler(client, sink=lambda _: None).sample(obs, np.zeros(7))
    assert not client.requests


def test_episode_identity_cannot_change_midstream():
    sampler = AACSampler(Client(), sink=lambda _: None)
    sampler.sample(observation(), np.zeros(7))
    with pytest.raises(ValueError, match='separate sampler'):
        sampler.sample(observation() | {'episode_seed': 18}, np.zeros(7))


def test_receipts_cannot_mutate_internal_accounting():
    def sink(receipt):
        receipt['rpc_attempts'] = -1
    sampler = AACSampler(Client(), sink=sink)
    result = sampler.sample(observation(), np.zeros(7))
    assert result.receipt['rpc_attempts'] == sampler.last_receipt['rpc_attempts'] == 20


def test_stream_routes_stable_and_sample_zero_exact():
    routes = stream_routes('physical-episode', 17, 20)
    assert routes == stream_routes('physical-episode', 17, 20)
    assert routes[0] == {'study_episode_id': 'physical-episode', 'episode_seed': 17}
    assert len({x['study_episode_id'] for x in routes}) == 20
    assert len({x['episode_seed'] for x in routes}) == 20


def test_failed_receipt_sink_never_dispatches_unrecorded_rpc():
    def sink(receipt):
        raise OSError('disk unavailable')
    client = Client()
    sampler = AACSampler(client, sink=sink)
    with pytest.raises(SamplingFailure) as caught:
        sampler.sample(observation(), np.zeros(7))
    assert not client.requests
    assert caught.value.receipt['status'] == 'failed'
    assert caught.value.receipt['dispatch_intents'] == 1
    assert caught.value.receipt['rpc_attempts'] == 0
    assert caught.value.receipt['unknown_generation_calls'] == 0
    assert caught.value.receipt['calls'][0]['status'] == 'not_dispatched'
    assert caught.value.receipt['calls'][0]['rpc_attempted'] is False
    assert caught.value.receipt['failure_stage'] == 'receipt_before_rpc'
    assert caught.value.receipt['sink_error']['type'] == 'OSError'
    assert sampler.failed



def test_current_joints_are_copied_before_sampling_and_bound_in_receipt():
    joints = np.zeros(7, dtype=np.float32)

    class MutatesCallerJoints(Client):
        def infer(self, request):
            joints[:] = 100
            return super().infer(request)

    expected = AACSampler(Client(), sink=lambda _: None).sample(observation(), np.zeros(7))
    actual = AACSampler(MutatesCallerJoints(), sink=lambda _: None).sample(observation(), joints)
    assert np.all(joints == 100)
    assert actual.receipt['current_joints'] == [0] * 7
    assert actual.receipt['selection'] == expected.receipt['selection']
    np.testing.assert_array_equal(actual.actions, expected.actions)


def test_failure_after_paid_valid_return_preserves_rpc_status_and_costs():
    def sink(receipt):
        if receipt['valid_proposals']:
            raise OSError('disk filled after valid return')

    client = Client()
    sampler = AACSampler(client, sink=sink)
    with pytest.raises(SamplingFailure) as caught:
        sampler.sample(observation(), np.zeros(7))
    receipt = caught.value.receipt
    assert len(client.requests) == 1
    assert receipt['dispatch_intents'] == receipt['rpc_attempts'] == receipt['rpc_returns'] == 1
    assert receipt['valid_proposals'] == 1
    assert receipt['returned_action_count'] == 32
    assert receipt['calls'][0]['status'] == 'valid'
    assert receipt['failure_stage'] == 'receipt_after_rpc'
    assert receipt['unknown_generation_calls'] == 0
    assert sampler.failed


def test_selector_failure_retains_all_paid_successful_rpc_records(monkeypatch):
    def fail(*args):
        raise ValueError('synthetic selector failure')

    monkeypatch.setattr('helpers.maniguard_aac_sampling.select_joint_prefix', fail)
    client = Client()
    with pytest.raises(SamplingFailure) as caught:
        AACSampler(client, sink=lambda _: None).sample(observation(), np.zeros(7))
    receipt = caught.value.receipt
    assert receipt['rpc_attempts'] == receipt['rpc_returns'] == receipt['valid_proposals'] == 20
    assert receipt['returned_action_count'] == 640
    assert receipt['unknown_generation_calls'] == 0
    assert all(call['status'] == 'valid' for call in receipt['calls'])
    assert receipt['failure_stage'] == 'selection'


def test_pre_rpc_snapshot_distinguishes_intent_from_confirmed_dispatch():
    snapshots = []
    result = AACSampler(Client(), sink=snapshots.append).sample(observation(), np.zeros(7))
    first = snapshots[0]
    assert first['dispatch_intents'] == 1 and first['rpc_attempts'] == 0
    assert first['calls'][0]['status'] == 'dispatch_pending'
    assert first['calls'][0]['rpc_attempted'] is False
    assert result.receipt['dispatch_intents'] == result.receipt['rpc_attempts'] == 20
    assert all(call['rpc_attempted'] for call in result.receipt['calls'])
