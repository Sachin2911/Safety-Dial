"""Synthetic dispatcher tests: no model/data reads, protocol freezes or process launches."""

import hashlib
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import maniguard_main_dispatch as dispatch_api
from helpers import maniguard_main_protocol as protocol_api
from helpers.maniguard_main_dispatch import (
    PRIOR_SCHEMA,
    WORK_COUNTS,
    DispatchPersistenceError,
    MainDispatch,
    _signed,
)
from helpers.maniguard_phase2_optim import record_hash


def counts(**values):
    return {**dict.fromkeys(WORK_COUNTS, 0), **values}


class Clock:
    def __init__(self, value=100.0):
        self.value = value

    def __call__(self):
        return self.value


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'synthetic-root'
    root.mkdir()
    # The strict injected validator substitutes for model/source evidence validation.
    # The real compile_jobs and authorize_job are exercised without creating a freeze.
    controller = {'method': 'synthetic', 'run_id': 'synthetic-s1', 'run_kind': 'trained',
                  'actor_sha256': 'a' * 64, 'controller_sha256': 'a' * 64,
                  'resource_contract': protocol_api.controller_resource_contract('phase2_learned')}
    controller['resource_contract_sha256'] = record_hash(controller['resource_contract'])
    case = {'family': 'jar', 'base_scene': 'task_0000', 'scene': 'task_0000/base', 'condition': 'base',
            'seed': 1, 'sampling_seed': 0, 'purpose': 'test', 'partition': 'test',
            'exposure_role': 'development', 'action_horizon': 9,
            'checkpoint': {'config': 'synthetic', 'revision': 'b' * 40, 'repo': 'synthetic', 'step': '1'},
            'eval_config_sha256': 'c' * 64}
    protocol = {'schema': protocol_api.SCHEMA, 'status': 'frozen', 'study_id': 'synthetic-dispatch',
                'controllers': [controller], 'cases': [case, {**case, 'seed': 2}],
                'preprocessing_sha256': 'd' * 64, 'exposure': {'jar': {'role': 'development'}},
                'spec': {'deployment': {'synthetic': True}, 'analysis': {}, 'implementation_files': []}}
    jobs = protocol_api.compile_jobs(protocol)
    protocol.update(grid_sha256=record_hash(jobs), envelope=protocol_api.action_envelope(jobs))
    caps = {name: counts(**{counter: 100 for counter in WORK_COUNTS})
            for name in ('prior', 'startup', 'failed_work')}
    caps['startup']['policy_control_steps'] = caps['failed_work']['policy_control_steps'] = 0
    protocol['spec']['budget'] = {'episode_attempts_max': len(jobs), 'separate_caps': caps,
        'allocated_gpu_hours_max': 1,
        **{key: protocol['envelope']['successful_work'][name] + sum(cap[name] for cap in caps.values())
           for name, key in protocol_api.BUDGET_KEYS.items()}}
    protocol = _signed(protocol)
    source = root / 'synthetic-prior.json'
    source.write_text('{"synthetic": true}\n')
    prior = _signed({'schema': PRIOR_SCHEMA, 'study_root': str(root),
                     'protocol_sha256': protocol['sha256'], 'counts': counts(), 'unknown_costs': [],
                     'timestamp': 100.0, 'reservation_started_at': 90.0,
                     'reserved_gpus': 1, 'allocated_gpu_seconds': 10.0,
                     'sources': [{'path': source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}]})
    value = {'root': root, 'protocol': protocol, 'prior': prior, 'clock': Clock(), 'gate_calls': []}

    def gate(actual, **kwargs):
        value['gate_calls'].append(kwargs)
        if record_hash(actual) != record_hash(value['protocol']) or actual['status'] != 'frozen':
            raise ValueError('Synthetic exact protocol validation failed')
        assert kwargs['require_frozen'] is True
        assert Path(kwargs['root']).resolve() == root
        return {'ready_for_exact_job_authorization': True}

    monkeypatch.setattr(protocol_api, 'validate_main_protocol', gate)
    return value


def open_dispatch(fixture, **kwargs):
    return MainDispatch.open(fixture['protocol'], fixture['prior'], root=fixture['root'],
                             clock=fixture['clock'], **kwargs)


def rebind(fixture):
    fixture['protocol'] = _signed(fixture['protocol'])
    fixture['prior'] = _signed({**fixture['prior'], 'protocol_sha256': fixture['protocol']['sha256']})


def row_for(dispatch, claim, **updates):
    return {'evidence_kind': 'synthetic', 'valid': True, 'job': deepcopy(claim['job']),
            'protocol_sha256': dispatch.protocol['sha256'], 'output': claim['output'],
            'costs': {**claim['reservation'], 'elapsed_seconds': 0.5},
            'cost_issues': [], 'missing_cost_fields': [], **updates}


def test_exact_claim_is_durable_before_backend_and_never_performance_evidence(fixture):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]

        def runner(claim):
            saved = json.loads(dispatch.path.read_text())
            assert saved['claims'][job['episode_id']]['status'] == 'claimed'
            assert claim['job'] == job and claim['job_sha256'] == record_hash(job)
            assert not Path(claim['output']).exists()  # No output/process backend.
            return {'evidence_kind': 'synthetic', 'process_succeeded': True,
                    'costs': {**claim['reservation'], 'elapsed_seconds': 0.5}}

        report = dispatch.run_synthetic(job, runner=runner,
                                       reader=lambda receipt, claim: row_for(dispatch, claim))
        assert report['terminal_jobs'] == 1 and report['blocked_reasons'] == []
        assert not report['live_execution_authorized'] and not report['performance_results']
        assert report['accounting']['known_counts'] == job['resource_envelope']['successful_work']
        assert report['accounting']['reserved_counts'] == counts()
        assert len(dispatch.pending_jobs()) == 1
    assert len(fixture['gate_calls']) >= 4  # Creation, readiness and the real authorizer.


def test_duplicate_result_and_claim_cannot_reuse_a_slot(fixture):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]
        claim = dispatch.claim(job)
        dispatch.record_result(job['episode_id'], row_for(dispatch, claim))
        with pytest.raises(ValueError, match='only once'):
            dispatch.claim(job)
        with pytest.raises(ValueError, match='recorded twice'):
            dispatch.record_result(job['episode_id'], row_for(dispatch, claim))


def test_resume_preserves_terminal_counts_and_still_consumes_original_claim(fixture):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]
        claim = dispatch.claim(job)
        dispatch.record_result(job['episode_id'], row_for(dispatch, claim))
    fixture['clock'].value += 25
    with open_dispatch(fixture, resume=True) as resumed:
        assert resumed.report()['accounting']['allocated_gpu_seconds'] == 35
        assert resumed.report()['claimed_jobs'] == 1 and len(resumed.pending_jobs()) == 1
        assert resumed.report()['accounting']['known_counts'] == claim['reservation']
        with pytest.raises(ValueError, match='only once'):
            resumed.claim(job)


def test_unresolved_claim_survives_restart_and_is_never_dispatched_again(fixture):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]
        dispatch.claim(job)
    with open_dispatch(fixture, resume=True) as resumed:
        assert resumed.report()['unresolved_claims'] == [job['episode_id']]
        assert resumed.report()['accounting']['reserved_counts'] == job['resource_envelope']['successful_work']
        with pytest.raises(RuntimeError, match='Unresolved durable claim'):
            resumed.run_synthetic(job, runner=lambda _: pytest.fail('Must not execute'), reader=lambda *_: {})


def test_canonical_anchor_prevents_reset_when_state_is_missing(fixture):
    with open_dispatch(fixture) as dispatch:
        path = dispatch.path
    path.unlink()
    with pytest.raises(ValueError, match='resume instead'):
        open_dispatch(fixture)
    with pytest.raises(FileNotFoundError):
        open_dispatch(fixture, resume=True)


def test_root_bound_prior_and_state_cannot_move_to_a_fresh_execution_root(fixture, tmp_path):
    with open_dispatch(fixture):
        pass
    new_root = tmp_path / 'other-root'
    shutil.copytree(fixture['root'], new_root)
    with pytest.raises(ValueError, match='exact study root and protocol'):
        MainDispatch.open(fixture['protocol'], fixture['prior'], root=new_root,
                          clock=fixture['clock'], resume=True, validator=lambda *_args, **_kwargs: None)


def test_second_owner_is_rejected_while_state_lock_is_held(fixture):
    with open_dispatch(fixture), pytest.raises(BlockingIOError):
        open_dispatch(fixture, resume=True)


@pytest.mark.parametrize('field,value', [('scene', 'task_9999/base'), ('seed', 7),
                                        ('actor_sha256', 'e' * 64), ('action_horizon', 32),
                                        ('exposure_role', 'reserved_transfer')])
def test_changed_exact_job_fails_before_claim_or_backend(fixture, field, value):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]
        job[field] = value
        with pytest.raises(ValueError, match='exact member'):
            dispatch.claim(job)
        assert dispatch.report()['claimed_jobs'] == 0


def test_changed_source_prior_or_protocol_is_revalidated_before_claim(fixture):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]
        source = fixture['root'] / fixture['prior']['sources'][0]['path']
        source.write_text('tampered')
        with pytest.raises(ValueError, match='source changed'):
            dispatch.claim(job)
        assert dispatch.report()['claimed_jobs'] == 0


def test_unfrozen_protocol_fails_before_any_output_creation(fixture):
    fixture['protocol']['status'] = 'draft'
    with pytest.raises(ValueError, match='explicitly frozen'):
        open_dispatch(fixture)
    assert not (fixture['root'] / 'runs').exists()


def test_aac_is_rejected_even_with_an_exact_authorized_synthetic_job(fixture):
    controller = fixture['protocol']['controllers'][0]
    controller['resource_contract'] = protocol_api.controller_resource_contract('aac_joint20')
    controller['resource_contract_sha256'] = record_hash(controller['resource_contract'])
    jobs = protocol_api.compile_jobs(fixture['protocol'])
    fixture['protocol'].update(grid_sha256=record_hash(jobs), envelope=protocol_api.action_envelope(jobs))
    fixture['protocol']['spec']['budget']['policy_queries_max'] += 1000
    rebind(fixture)
    with open_dispatch(fixture) as dispatch:
        with pytest.raises(ValueError, match='AAC'):
            dispatch.claim(dispatch.pending_jobs()[0])
        assert dispatch.report()['claimed_jobs'] == 0


@pytest.mark.parametrize('name', WORK_COUNTS)
def test_every_total_resource_ceiling_is_checked_before_extra_work(fixture, name):
    fixture['protocol']['spec']['budget'][protocol_api.BUDGET_KEYS[name]] = 0
    rebind(fixture)
    with open_dispatch(fixture) as dispatch:
        with pytest.raises(RuntimeError, match='Total resource ceiling exceeded: ' + name):
            dispatch.reserve_extra('startup', 'startup', counts(**{name: 1}))
        assert dispatch.state['extra_events'] == {}


@pytest.mark.parametrize('category', ['prior', 'startup', 'failed_work'])
def test_separate_caps_cannot_borrow_from_other_categories(fixture, category):
    fixture['protocol']['spec']['budget']['separate_caps'][category]['policy_queries'] = 0
    if category == 'prior':
        fixture['prior']['counts']['policy_queries'] = 1
    rebind(fixture)
    with open_dispatch(fixture) as dispatch:
        if category == 'prior':
            assert 'prior/policy_queries' in ' '.join(dispatch.report()['blocked_reasons'])
        else:
            with pytest.raises(RuntimeError, match=category + '/policy_queries'):
                dispatch.reserve_extra('extra', category, counts(policy_queries=1))


def test_startup_failure_and_prior_work_are_counted_once_in_separate_buckets(fixture):
    fixture['prior']['counts'] = counts(policy_queries=2, physics_steps=3)
    rebind(fixture)
    with open_dispatch(fixture) as dispatch:
        dispatch.reserve_extra('startup', 'startup', counts(physics_steps=10, encoder_load_attempts=2))
        dispatch.settle_extra('startup', counts(physics_steps=7, encoder_load_attempts=1))
        dispatch.reserve_extra('failure', 'failed_work', counts(policy_queries=2))
        dispatch.settle_extra('failure', counts(policy_queries=1))
        report = dispatch.report()['accounting']
        assert report['known_counts'] == counts(policy_queries=3, physics_steps=10, encoder_load_attempts=1)
        assert report['category_known_counts']['prior']['policy_queries'] == 2
        assert report['category_known_counts']['startup']['physics_steps'] == 7
        assert report['category_known_counts']['failed_work']['policy_queries'] == 1
        assert report['reserved_counts'] == counts()
        with pytest.raises(ValueError, match='counted twice'):
            dispatch.settle_extra('failure', counts(policy_queries=1))


def test_unknown_extra_costs_preserve_lower_bounds_and_block_remaining_jobs(fixture):
    with open_dispatch(fixture) as dispatch:
        dispatch.reserve_extra('failure', 'failed_work', counts(policy_queries=5, encoder_image_attempts=2))
        report = dispatch.settle_extra('failure', {'policy_queries': 2}, issues=['Server dispatch is unknown'])
        assert report['accounting']['known_counts']['policy_queries'] == 2
        assert report['accounting']['reserved_counts']['policy_queries'] == 3
        assert report['accounting']['reserved_counts']['encoder_image_attempts'] == 2
        assert 'Server dispatch is unknown' in report['blocked_reasons']
        with pytest.raises(RuntimeError, match='blocked'):
            dispatch.pending_jobs()


def test_unknown_prior_is_retained_and_blocks_without_inventing_zeros(fixture):
    fixture['prior']['counts']['physics_steps'] = 86
    fixture['prior']['unknown_costs'] = ['Historical encoder attempts unresolved']
    fixture['prior'] = _signed(fixture['prior'])
    with open_dispatch(fixture) as dispatch:
        assert dispatch.report()['accounting']['known_counts']['physics_steps'] == 86
        assert dispatch.report()['blocked_reasons'] == ['Prior accounting has unresolved costs']


def test_extra_reservation_and_known_overrun_are_retained_across_resume(fixture):
    with open_dispatch(fixture) as dispatch:
        dispatch.reserve_extra('event', 'startup', counts(physics_steps=1))
    with open_dispatch(fixture, resume=True) as resumed:
        with pytest.raises(RuntimeError, match='Unresolved extra-work'):
            resumed.pending_jobs()
        report = resumed.settle_extra('event', counts(physics_steps=2))
        assert report['accounting']['known_counts']['physics_steps'] == 2
        assert 'reservation exceeded' in ' '.join(report['blocked_reasons'])


def test_failed_reader_retains_backend_costs_and_never_retries(fixture):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]

        def reader(*_):
            raise ValueError('Synthetic audit crash')

        report = dispatch.run_synthetic(job, runner=lambda _: {
            'evidence_kind': 'synthetic', 'process_succeeded': False,
            'costs': {'policy_queries': 2, 'elapsed_seconds': 1.5}}, reader=reader)
        assert report['terminal_jobs'] == 1 and report['blocked_reasons']
        assert report['accounting']['known_counts']['policy_queries'] == 2
        saved = dispatch.state['claims'][job['episode_id']]
        assert saved['backend_receipt']['process_succeeded'] is False
        assert 'physics_steps' in saved['unknown_counts']
        with pytest.raises(RuntimeError, match='blocked'):
            dispatch.claim(job)


def test_reader_cannot_erase_known_backend_costs_or_process_failure(fixture):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]
        report = dispatch.run_synthetic(job, runner=lambda _: {
            'evidence_kind': 'synthetic', 'process_succeeded': False,
            'costs': {**counts(policy_queries=3), 'elapsed_seconds': 1}},
            reader=lambda receipt, claim: row_for(dispatch, claim,
                                                  costs={**counts(), 'elapsed_seconds': 1}))
        assert report['blocked_reasons'] and report['accounting']['known_counts']['policy_queries'] == 3
        retained = dispatch.state['claims'][job['episode_id']]['row']
        assert retained['valid'] is False and retained['raw_reader_row']['valid'] is True
        assert retained['raw_reader_row']['costs']['policy_queries'] == 0


def test_runner_exception_retains_claim_and_unknown_costs(fixture):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]

        def runner(_):
            raise RuntimeError('Fake runner interruption')

        report = dispatch.run_synthetic(job, runner=runner, reader=lambda *_: pytest.fail('Do not read'))
        assert report['terminal_jobs'] == 1 and report['blocked_reasons']
        assert dispatch.state['claims'][job['episode_id']]['unknown_counts'] == list(WORK_COUNTS)
        assert report['accounting']['reserved_counts'] == job['resource_envelope']['successful_work']


def test_interrupt_is_persisted_then_propagated(fixture):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]

        def runner(_):
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            dispatch.run_synthetic(job, runner=runner, reader=lambda *_: {})
        assert json.loads(dispatch.path.read_text())['claims'][job['episode_id']]['status'] == 'terminal'
        assert dispatch.report()['blocked_reasons']


def test_gpu_clock_covers_idle_and_restarts_and_refuses_backwards_time(fixture):
    with open_dispatch(fixture) as dispatch:
        fixture['clock'].value = 3590
        assert dispatch.report()['accounting']['allocated_gpu_seconds'] == 3500
    fixture['clock'].value = 3690
    with open_dispatch(fixture, resume=True) as resumed:
        assert resumed.report()['accounting']['allocated_gpu_seconds'] == 3600
        with pytest.raises(RuntimeError, match='GPU reservation'):
            resumed.pending_jobs()
        fixture['clock'].value = 3689
        with pytest.raises(ValueError, match='backwards'):
            resumed.report()


def test_missing_prior_counter_rejected_before_state_creation(fixture):
    del fixture['prior']['counts']['encoder_load_attempts']
    fixture['prior'] = _signed(fixture['prior'])
    with pytest.raises(ValueError, match='eleven'):
        open_dispatch(fixture)
    assert not (fixture['root'] / 'runs').exists()


def test_changed_durable_state_is_not_silently_overwritten(fixture):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]
        value = json.loads(dispatch.path.read_text())
        value['revision'] += 1
        dispatch.path.write_text(json.dumps(_signed(value)))
        with pytest.raises(DispatchPersistenceError, match='changed while'):
            dispatch.claim(job)


def test_rehashed_saved_accounting_cannot_disagree_with_retained_result(fixture):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]
        claim = dispatch.claim(job)
        dispatch.record_result(job['episode_id'], row_for(dispatch, claim))
        path = dispatch.path
    value = json.loads(path.read_text())
    value['claims'][job['episode_id']]['known_counts']['policy_queries'] -= 1
    path.write_text(json.dumps(_signed(value)))
    with pytest.raises(ValueError, match='retained row'):
        open_dispatch(fixture, resume=True)


def test_clearing_block_status_cannot_hide_unresolved_terminal_costs(fixture):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]
        claim = dispatch.claim(job)
        dispatch.record_result(job['episode_id'], row_for(dispatch, claim,
                              valid=False, costs={'policy_queries': 2, 'elapsed_seconds': 1}))
        path = dispatch.path
    value = json.loads(path.read_text())
    value['blocked_reasons'] = []
    path.write_text(json.dumps(_signed(value)))
    with open_dispatch(fixture, resume=True) as resumed:
        assert resumed.report()['accounting']['known_counts']['policy_queries'] == 2
        with pytest.raises(RuntimeError, match='Unresolved terminal'):
            resumed.pending_jobs()


def test_fixed_comparator_retains_none_actor_and_its_controller_identity(fixture):
    controller = fixture['protocol']['controllers'][0]
    controller.update(run_kind='fixed', actor_sha256=None, controller_sha256='f' * 64,
                      resource_contract=protocol_api.controller_resource_contract('phase2_identity'))
    controller['resource_contract_sha256'] = record_hash(controller['resource_contract'])
    jobs = protocol_api.compile_jobs(fixture['protocol'])
    fixture['protocol'].update(grid_sha256=record_hash(jobs), envelope=protocol_api.action_envelope(jobs))
    rebind(fixture)
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]
        claim = dispatch.claim(job)
        assert claim['job']['actor_sha256'] is None
        assert claim['job']['controller_sha256'] == 'f' * 64
        assert not dispatch.record_result(job['episode_id'], row_for(dispatch, claim))['blocked_reasons']


def test_full_slot_reservation_refuses_insufficient_remaining_queries(fixture):
    fixture['protocol']['spec']['budget']['policy_queries_max'] = 8
    rebind(fixture)
    with open_dispatch(fixture) as dispatch:
        with pytest.raises(RuntimeError, match='Total resource ceiling exceeded: policy_queries'):
            dispatch.claim(dispatch.pending_jobs()[0])
        assert dispatch.report()['claimed_jobs'] == 0


def test_nonfinite_failure_counter_does_not_erase_other_measured_work(fixture):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]
        claim = dispatch.claim(job)
        report = dispatch.record_result(job['episode_id'], row_for(dispatch, claim,
            costs={**counts(policy_queries=2), 'physics_steps': float('nan'), 'elapsed_seconds': 1}))
        assert report['blocked_reasons']
        assert report['accounting']['known_counts']['policy_queries'] == 2
        stored = json.loads(dispatch.path.read_text())['claims'][job['episode_id']]
        assert stored['row']['costs']['physics_steps'] == {'nonfinite_number': 'nan'}
        assert 'physics_steps' in stored['unknown_counts']


def test_rebinding_prior_to_copied_root_does_not_reset_the_existing_study_anchor(fixture, tmp_path):
    with open_dispatch(fixture):
        pass
    copied = tmp_path / 'copied-root'
    shutil.copytree(fixture['root'], copied)
    prior = _signed({**fixture['prior'], 'study_root': str(copied)})
    with pytest.raises(ValueError, match='binding changed'):
        MainDispatch.open(fixture['protocol'], prior, root=copied, clock=fixture['clock'],
                          resume=True, validator=lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match='resume instead'):
        MainDispatch.open(fixture['protocol'], prior, root=copied, clock=fixture['clock'],
                          validator=lambda *_args, **_kwargs: None)


def fail_state_write_once(monkeypatch, predicate):
    original = dispatch_api._atomic
    failures = []

    def injected(path, value):
        if Path(path).name == 'state.json' and not failures and predicate(value):
            failures.append(deepcopy(value))
            raise OSError('Injected state write failure')
        return original(path, value)

    monkeypatch.setattr(dispatch_api, '_atomic', injected)
    return failures


def assert_poisoned(dispatch, job):
    callbacks = []
    operations = [dispatch.pending_jobs, dispatch.report, dispatch.snapshot,
                  lambda: dispatch.claim(job),
                  lambda: dispatch.reserve_extra('later', 'startup', counts()),
                  lambda: dispatch.settle_extra('missing', counts()),
                  lambda: dispatch.record_result(job['episode_id'], {}),
                  lambda: dispatch.run_synthetic(job, runner=lambda *_: callbacks.append('runner'),
                                                reader=lambda *_: callbacks.append('reader'))]
    for operation in operations:
        with pytest.raises(DispatchPersistenceError, match='poisoned'):
            operation()
    assert callbacks == []


@pytest.mark.parametrize('stage', ['claim', 'backend', 'terminal', 'blocked'])
def test_write_failure_propagates_at_every_dispatch_boundary_and_poison_stops_callbacks(
        fixture, monkeypatch, stage):
    callbacks = []
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]
        job_id = job['episode_id']

        def at_stage(value):
            claim = value['claims'].get(job_id)
            if claim is None:
                return False
            return {'claim': claim['status'] == 'claimed' and claim['backend_receipt'] is None,
                    'backend': claim['status'] == 'claimed' and claim['backend_receipt'] is not None,
                    'terminal': claim['status'] == 'terminal',
                    'blocked': bool(value['blocked_reasons'])}[stage]

        failures = fail_state_write_once(monkeypatch, at_stage)

        def runner(claim):
            callbacks.append('runner')
            return {'evidence_kind': 'synthetic', 'process_succeeded': True,
                    'costs': {**claim['reservation'], 'elapsed_seconds': 0.5}}

        def reader(_receipt, claim):
            callbacks.append('reader')
            return row_for(dispatch, claim, valid=stage != 'blocked')

        with pytest.raises(DispatchPersistenceError, match='Injected state write failure') as error:
            dispatch.run_synthetic(job, runner=runner, reader=reader)
        assert isinstance(error.value.__cause__, OSError)
        assert len(failures) == 1
        assert callbacks == {'claim': [], 'backend': ['runner'],
                             'terminal': ['runner', 'reader'], 'blocked': ['runner', 'reader']}[stage]
        saved = json.loads(dispatch.path.read_text())
        if stage == 'claim':
            assert saved['claims'] == {}  # Runner was never called.
        else:
            prior = saved['claims'][job_id]
            assert prior['status'] == ('terminal' if stage == 'blocked' else 'claimed')
            assert (prior['backend_receipt'] is None) == (stage == 'backend')
            if stage == 'blocked':
                assert prior['row']['valid'] is False and prior['issues']
                assert saved['blocked_reasons'] == []
        diagnostic = json.loads(Path(dispatch.persistence_failure['diagnostic_path']).read_text())
        assert diagnostic['last_committed_sha256'] == saved['sha256']
        assert diagnostic['durability_uncertain'] and not diagnostic['automatic_recovery_authorized']
        candidate = diagnostic['uncommitted_state']['claims'][job_id]
        if stage != 'claim':
            assert candidate['backend_receipt']['costs']['policy_queries'] == job[
                'resource_envelope']['successful_work']['policy_queries']
        if stage in ('terminal', 'blocked'):
            assert candidate['row']['costs']['policy_queries'] > 0
        assert_poisoned(dispatch, job)
        assert json.loads(dispatch.path.read_text()) == saved
    # Reopen never silently reconciles the uncommitted evidence or starts more work.
    with open_dispatch(fixture, resume=True) as resumed:
        assert any('persistence failure' in reason for reason in resumed.report()['blocked_reasons'])
        with pytest.raises(RuntimeError, match='blocked'):
            resumed.pending_jobs()


@pytest.mark.parametrize('stage', ['reserve', 'settle', 'blocked'])
def test_extra_work_write_failures_preserve_reservation_and_paid_evidence(fixture, monkeypatch, stage):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]
        reserved = counts(physics_steps=3)
        if stage != 'reserve':
            dispatch.reserve_extra('startup', 'startup', reserved)

        def at_stage(value):
            event = value['extra_events'].get('startup')
            return event is not None and (
                stage == 'reserve' or stage == 'settle' and event['status'] == 'terminal' or
                stage == 'blocked' and bool(value['blocked_reasons']))

        fail_state_write_once(monkeypatch, at_stage)
        with pytest.raises(DispatchPersistenceError, match='Injected state write failure'):
            if stage == 'reserve':
                dispatch.reserve_extra('startup', 'startup', reserved)
            else:
                dispatch.settle_extra('startup', counts(physics_steps=2),
                                      issues=['Synthetic unresolved work'] if stage == 'blocked' else [])
        saved = json.loads(dispatch.path.read_text())
        if stage == 'reserve':
            assert saved['extra_events'] == {}
        else:
            assert saved['extra_events']['startup']['status'] == (
                'terminal' if stage == 'blocked' else 'reserved')
            assert saved['extra_events']['startup']['reservation'] == reserved
        diagnostic = json.loads(Path(dispatch.persistence_failure['diagnostic_path']).read_text())
        event = diagnostic['uncommitted_state']['extra_events']['startup']
        if stage != 'reserve':
            assert event['known_counts']['physics_steps'] == 2
            assert event['raw_counts'] == counts(physics_steps=2)
        assert_poisoned(dispatch, job)
    with open_dispatch(fixture, resume=True) as resumed, pytest.raises(RuntimeError, match='blocked'):
        resumed.pending_jobs()


def test_diagnostic_write_failure_cannot_mask_original_failure_or_allow_callbacks(fixture, monkeypatch):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]
        claim = dispatch.claim(job)
        saved = dispatch.path.read_bytes()

        def fail_all(path, _value):
            raise OSError('Primary state failure' if Path(path).name == 'state.json' else 'Diagnostic failure')

        monkeypatch.setattr(dispatch_api, '_atomic', fail_all)
        with pytest.raises(DispatchPersistenceError, match='Primary state failure') as error:
            dispatch.record_result(job['episode_id'], row_for(dispatch, claim))
        assert str(error.value.__cause__) == 'Primary state failure'
        assert dispatch.path.read_bytes() == saved
        assert 'diagnostic_path' not in dispatch.persistence_failure
        assert_poisoned(dispatch, job)


def test_post_replace_error_poison_does_not_claim_rollback_or_terminal_success(fixture, monkeypatch):
    with open_dispatch(fixture) as dispatch:
        job = dispatch.pending_jobs()[0]
        original = dispatch_api._atomic
        fired = []

        def fail_after_replace(path, value):
            original(path, value)
            if (Path(path).name == 'state.json' and not fired and
                    any(claim['status'] == 'terminal' for claim in value['claims'].values())):
                fired.append(True)
                raise OSError('Injected directory sync error after replace')

        monkeypatch.setattr(dispatch_api, '_atomic', fail_after_replace)
        with pytest.raises(DispatchPersistenceError, match='after replace'):
            dispatch.run_synthetic(job, runner=lambda claim: {
                'evidence_kind': 'synthetic', 'process_succeeded': True, 'costs': claim['reservation']},
                reader=lambda receipt, claim: row_for(dispatch, claim))
        assert json.loads(dispatch.path.read_text())['claims'][job['episode_id']]['status'] == 'terminal'
        assert_poisoned(dispatch, job)
    with open_dispatch(fixture, resume=True) as resumed:
        assert resumed.report()['terminal_jobs'] == 1
        with pytest.raises(RuntimeError, match='persistence failure'):
            resumed.pending_jobs()
