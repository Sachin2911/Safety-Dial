"""Fake services/processes/readers only; no simulator, checkpoint or outcome access."""

import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_maniguard_main_dispatch import counts, rebind
from test_maniguard_main_dispatch import fixture as base_fixture

from helpers import maniguard_main_launcher as api
from helpers import maniguard_main_protocol as protocol_api
from helpers.maniguard_main_dispatch import DispatchPersistenceError, _signed
from helpers.maniguard_phase2_optim import record_hash


@pytest.fixture
def main_fixture(tmp_path, monkeypatch):
    value = base_fixture.__wrapped__(tmp_path, monkeypatch)
    root, protocol = value['root'], value['protocol']
    for name in api.PARENT_SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Synthetic frozen source: ' + name)
        protocol['spec']['implementation_files'].append(protocol_api.source_reference(path, root=root))
    contract = _signed({'schema': api.RUNTIME_SCHEMA, 'supported_profiles': ['phase2_learned'],
        'episode_timeout_seconds': 10.0, 'policy_startup_timeout_seconds': 5.0,
        'termination_grace_seconds': 1.0, 'cleanup_reserve_seconds': 3.0,
        'audit_timeout_seconds': 5.0, 'audit_cleanup_reserve_seconds': 3.0,
        'per_episode_startup_counts': counts(physics_steps=2),
        'per_episode_failed_work_counts': counts(physics_steps=1),
        'backend': {'kind': 'test-only-not-a-live-driver'}})
    contract_path = root / 'synthetic-runtime.json'
    contract_path.write_text(json.dumps(contract))
    reference = protocol_api.source_reference(contract_path, root=root)
    protocol['spec']['implementation_files'].append(reference)
    protocol['grid_sha256'] = record_hash(protocol_api.compile_jobs(protocol))
    rebind(value)
    value.update(contract=contract, runtime_reference=reference,
                 protocol_path=root / 'synthetic-protocol.json', prior_path=root / 'synthetic-prior-input.json',
                 events=[], requests=[], process_receipts=[])
    save_inputs(value)
    monkeypatch.setattr(api.services, 'EPISODE_SCRIPTS', {'maniguard_main_episode.py'})
    monkeypatch.setattr(api.services, 'active_episode_processes', list)

    class Service:
        def expected_metadata(self, family):
            job = next(job for job in protocol_api.compile_jobs(value['protocol']) if job['family'] == family)
            return api.expected_handshake(job, value['protocol'], root=root)

        def ensure_family(self, family, audit_dir, *, startup_timeout):
            value['events'].append('service')
            assert startup_timeout == contract['policy_startup_timeout_seconds']
            # A separate descriptor cannot obtain the machine lease while services run.
            with (root / 'runs/maniguard/setup/execution.lock').open('r') as stream, \
                    pytest.raises(BlockingIOError):
                api.fcntl.flock(stream, api.fcntl.LOCK_EX | api.fcntl.LOCK_NB)
            return {'family': family, 'status': 'reused', 'metadata': {
                **self.expected_metadata(family), 'checkpoint_provenance': {
                    'verification': 'all_declared_files_sha256_and_available_upstream_digests',
                    'receipt_sha256': 'f' * 64, 'verified_files': 2, 'verified_bytes': 100}}}

    def runner(command, log_path, timeout, *, popen, termination_grace, on_started):
        value['events'].append('runner')
        request = json.loads(Path(command[-1]).read_text())
        assert command == request['command']
        assert popen.keywords['pass_fds'] == (request['parent_binding']['lease_fd'],)
        api.verify_parent_binding(request, parent_pid=os.getpid())
        value['requests'].append(request)
        saved = json.loads(Path(request['parent_binding']['dispatch_root'], 'state.json').read_text())
        assert saved['claims'][request['job']['episode_id']]['status'] == 'claimed'
        assert saved['execution_kind'] == 'managed_main_child'
        assert all(event['status'] == 'reserved' for event in saved['extra_events'].values()
                   if event['event_id'].startswith(request['job']['episode_id']))
        Path(request['output']).mkdir(parents=True)
        process = {'command': command, 'pid': 12345, 'started_at': value['clock'](),
                   'finished_at': value['clock']() + 1, 'returncode': 0, 'timed_out': False,
                   'interrupted': False, 'process_group_remaining': False, 'process_elapsed_seconds': 1.0}
        on_started(process)
        value['clock'].value += 1
        return process

    def reader(output, protocol, job, *, process_receipt, root, config_dir):
        value['events'].append('reader')
        value['process_receipts'].append(process_receipt)
        assert json.loads((Path(output) / 'main_process.json').read_text()) == process_receipt
        assert process_receipt['job_sha256'] == record_hash(job)
        split = {'slot': deepcopy(job['resource_envelope']['successful_work']),
                 'startup': counts(physics_steps=1), 'failed_work': counts()}
        total = {key: sum(part[key] for part in split.values()) for key in api.WORK_COUNTS}
        reader_pin = next(ref['sha256'] for ref in protocol['spec']['implementation_files']
                          if ref['path'] == 'experiments/helpers/maniguard_main_results.py')
        return {**job, 'output': output, 'protocol_sha256': protocol['sha256'],
                'audit_reader_sha256': reader_pin, 'full_episode': True, 'short_check': False,
                'valid': True, 'costs': {**total, 'elapsed_seconds': 1.0}, 'cost_breakdown': split,
                'cost_issues': [], 'missing_cost_fields': []}

    value.update(service=Service(), runner=runner, reader=reader)
    return value


def save_inputs(value):
    value['protocol_path'].write_text(json.dumps(value['protocol']))
    value['prior_path'].write_text(json.dumps(value['prior']))


def run(value, **kwargs):
    options = {'root': value['root'], 'wall': value['clock'], 'service': value['service'],
               'runner': value['runner'], 'reader': value['reader'],
               'preflight': lambda *_args, **_kwargs: None, 'managed_check': lambda: None}
    return api.run_main(value['protocol_path'], value['prior_path'], value['runtime_reference'],
                        **{**options, **kwargs})


def test_managed_claim_lease_request_process_reader_and_category_accounting(main_fixture):
    value = main_fixture
    report = run(value, max_jobs=1)
    assert value['events'] == ['service', 'runner', 'reader']
    assert report['execution_kind'] == 'managed_main_child'
    assert report['terminal_jobs'] == 1 and report['blocked_reasons'] == []
    assert not report['performance_results'] and not report['physical_evidence_audited']
    first = protocol_api.compile_jobs(value['protocol'])[0]
    assert report['accounting']['known_counts']['physics_steps'] == first[
        'resource_envelope']['successful_work']['physics_steps'] + 1
    assert report['accounting']['category_known_counts']['startup']['physics_steps'] == 1
    assert report['accounting']['category_known_counts']['failed_work'] == counts()
    saved = json.loads(Path(report['state_path']).read_text())
    assert saved['claims'][first['episode_id']]['row']['audited_main_row']['valid'] is True
    report = run(value, resume=True)
    assert report['terminal_jobs'] == 2 and report['blocked_reasons'] == []
    assert len(value['requests']) == 2
    assert value['requests'][0]['job']['episode_id'] != value['requests'][1]['job']['episode_id']


def test_unsupported_physics_preflight_has_no_claim_service_or_output(main_fixture):
    def blocked(*_args, **_kwargs):
        raise RuntimeError('Pre-advance physics driver is unsupported')

    with pytest.raises(RuntimeError, match='physics driver'):
        run(main_fixture, preflight=blocked)
    assert main_fixture['events'] == []
    assert not (main_fixture['root'] / 'runs').exists()


def test_missing_main_process_scan_patch_blocks_before_claim(main_fixture, monkeypatch):
    monkeypatch.setattr(api.services, 'EPISODE_SCRIPTS', {'maniguard_phase2_episode.py'})
    with pytest.raises(RuntimeError, match='scanner'):
        run(main_fixture)
    assert main_fixture['events'] == []
    assert not (main_fixture['root'] / 'runs').exists()


def test_unmanaged_cli_path_fails_without_service_calls(main_fixture, monkeypatch):
    monkeypatch.delenv('SUPERVISOR_ENABLED', raising=False)
    with pytest.raises(RuntimeError, match='supervisor-managed'):
        run(main_fixture, managed_check=None)
    assert main_fixture['events'] == []


@pytest.mark.parametrize('counter', api.WORK_COUNTS)
def test_every_counter_is_reserved_before_policy_service(main_fixture, counter):
    value = main_fixture
    value['protocol']['spec']['budget'][protocol_api.BUDGET_KEYS[counter]] = 0
    # Resource profiles legitimately require zero actor RPCs, so use an explicit
    # frozen failed-work allowance to test that counter without fabricating work.
    if counter == 'actor_rpc_attempts':
        value['contract']['per_episode_failed_work_counts'][counter] = 1
        path = value['root'] / value['runtime_reference']['path']
        path.write_text(json.dumps(_signed(value['contract'])))
        updated = protocol_api.source_reference(path, root=value['root'])
        refs = value['protocol']['spec']['implementation_files']
        refs[refs.index(value['runtime_reference'])] = updated
        value['runtime_reference'] = updated
        value['protocol']['grid_sha256'] = record_hash(protocol_api.compile_jobs(value['protocol']))
    rebind(value)
    save_inputs(value)
    with pytest.raises(RuntimeError, match='resource ceiling'):
        run(value)
    assert value['events'] == []


def test_gpu_deadline_funds_startup_episode_and_cleanup_before_service(main_fixture):
    main_fixture['clock'].value = 3685  # 3595 GPU-seconds already reserved; only five remain.
    with pytest.raises(RuntimeError, match='continuous GPU reservation'):
        run(main_fixture)
    assert main_fixture['events'] == []


def test_exact_handshake_mismatch_never_starts_child_and_consumes_claim(main_fixture):
    service = main_fixture['service']
    original = service.ensure_family

    def mismatched(*args, **kwargs):
        record = original(*args, **kwargs)
        record['metadata']['checkpoint_revision'] = 'bad'
        return record

    service.ensure_family = mismatched
    report = run(main_fixture)
    assert main_fixture['events'] == ['service']
    assert report['claimed_jobs'] == 1 and report['terminal_jobs'] == 1 and report['blocked_reasons']
    assert report['accounting']['known_counts'] == counts()
    with pytest.raises(RuntimeError, match='blocked'):
        run(main_fixture, resume=True)
    assert main_fixture['events'] == ['service']


def test_unknown_spawn_failure_is_paid_unknown_never_zero_or_retried(main_fixture):
    def runner(*_args, **_kwargs):
        main_fixture['events'].append('unknown_spawn')
        raise RuntimeError('Spawn status unknown')

    report = run(main_fixture, runner=runner)
    assert report['terminal_jobs'] == 1 and report['blocked_reasons']
    assert report['accounting']['reserved_counts']['policy_queries'] > 0
    saved = json.loads(Path(report['state_path']).read_text())
    claim = next(iter(saved['claims'].values()))
    assert claim['unknown_counts'] == list(api.WORK_COUNTS)
    assert all(event['unknown_counts'] for event in saved['extra_events'].values())
    with pytest.raises(RuntimeError, match='blocked'):
        run(main_fixture, resume=True)


def test_failed_process_cannot_be_overridden_by_a_valid_reader(main_fixture):
    original = main_fixture['runner']

    def failed(*args, **kwargs):
        process = original(*args, **kwargs)
        process.update(returncode=-9, timed_out=True)
        return process

    report = run(main_fixture, runner=failed)
    assert report['blocked_reasons'] and report['terminal_jobs'] == 1
    assert report['accounting']['known_counts']['policy_queries'] > 0
    assert len(main_fixture['requests']) == 1


@pytest.mark.parametrize('change', ['identity', 'reader', 'split', 'unknown'])
def test_independent_audit_mismatch_retains_paid_costs_and_stops(main_fixture, change):
    original = main_fixture['reader']

    def reader(*args, **kwargs):
        row = original(*args, **kwargs)
        if change == 'identity':
            row['scene'] = 'wrong-scene'
        elif change == 'reader':
            row['audit_reader_sha256'] = 'wrong-reader'
        elif change == 'split':
            row['cost_breakdown']['startup']['physics_steps'] += 1
        else:
            del row['costs']['encoder_load_attempts']
            row['valid'] = False
        return row

    report = run(main_fixture, reader=reader)
    assert report['blocked_reasons'] and report['terminal_jobs'] == 1
    assert report['accounting']['known_counts']['policy_queries'] > 0
    assert len(main_fixture['requests']) == 1


def test_inherited_claim_rejects_wrong_parent_nonce_and_noninherited_fd(main_fixture):
    original = main_fixture['runner']

    def runner(command, *args, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        with pytest.raises(ValueError, match='direct child'):
            api.verify_parent_binding(request, parent_pid=os.getpid() + 1)
        changed = deepcopy(request)
        changed['parent_binding']['lease_nonce'] = 'wrong'
        with pytest.raises(ValueError, match='claim differs'):
            api.verify_parent_binding(changed, parent_pid=os.getpid())
        with Path(command[-1]).open() as wrong:
            changed = deepcopy(request)
            changed['parent_binding']['lease_fd'] = wrong.fileno()
            with pytest.raises(ValueError, match='canonical execution lease'):
                api.verify_parent_binding(changed, parent_pid=os.getpid())
        return original(command, *args, **kwargs)

    report = run(main_fixture, runner=runner, max_jobs=1)
    assert report['blocked_reasons'] == []


def test_process_evidence_write_failure_propagates_and_prevents_reader(main_fixture, monkeypatch):
    original = api._atomic

    def failure(path, record):
        if Path(path).name == 'main_process.json':
            raise OSError('Injected process receipt durability failure')
        return original(path, record)

    monkeypatch.setattr(api, '_atomic', failure)
    with pytest.raises(DispatchPersistenceError, match='process receipt'):
        run(main_fixture)
    assert main_fixture['events'] == ['service', 'runner']
    with pytest.raises(RuntimeError, match='persistence failure'):
        run(main_fixture, resume=True)


def test_source_change_after_service_blocks_before_child(main_fixture):
    service = main_fixture['service']
    original = service.ensure_family

    def changed(*args, **kwargs):
        record = original(*args, **kwargs)
        (main_fixture['root'] / api.PARENT_SOURCES[0]).write_text('changed after service')
        return record

    service.ensure_family = changed
    report = run(main_fixture)
    assert main_fixture['events'] == ['service']
    assert report['blocked_reasons'] and report['terminal_jobs'] == 1
    assert report['accounting']['known_counts'] == counts()


def test_policy_registry_drift_blocks_before_service_switch(main_fixture):
    main_fixture['service'].expected_metadata = lambda _family: {'serve_config': 'wrong'}
    report = run(main_fixture)
    assert main_fixture['events'] == []
    assert report['terminal_jobs'] == 1 and report['blocked_reasons']


def test_active_episode_blocks_before_service_switch(main_fixture, monkeypatch):
    monkeypatch.setattr(api.services, 'active_episode_processes', lambda: [{'pid': 999}])
    report = run(main_fixture)
    assert main_fixture['events'] == []
    assert report['blocked_reasons'] and report['terminal_jobs'] == 1


def test_pending_operation_with_numeric_lower_bounds_keeps_unused_reservations(main_fixture):
    original = main_fixture['reader']

    def pending(*args, **kwargs):
        row = original(*args, **kwargs)
        row['cost_breakdown']['slot']['policy_queries'] = 2
        row['costs']['policy_queries'] = 2
        row['unknown_cost_fields'] = ['policy_queries']
        row['valid'] = False
        return row

    report = run(main_fixture, reader=pending)
    assert report['accounting']['known_counts']['policy_queries'] == 2
    assert report['accounting']['reserved_counts']['policy_queries'] == 7
    assert report['terminal_jobs'] == 1 and report['blocked_reasons']
    with pytest.raises(RuntimeError, match='blocked'):
        run(main_fixture, resume=True)


def test_runtime_contract_cannot_borrow_separate_startup_cap(main_fixture):
    main_fixture['protocol']['spec']['budget']['separate_caps']['startup']['physics_steps'] = 1
    rebind(main_fixture)
    save_inputs(main_fixture)
    with pytest.raises(RuntimeError, match='startup/physics_steps'):
        run(main_fixture)
    assert main_fixture['events'] == []


def test_extra_request_allowance_must_match_durable_claim(main_fixture):
    original = main_fixture['runner']

    def runner(command, *args, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        request['reservation']['startup_counts']['physics_steps'] += 1
        with pytest.raises(ValueError, match='category allowance'):
            api.verify_parent_binding(request, parent_pid=os.getpid())
        return original(command, *args, **kwargs)

    assert run(main_fixture, runner=runner, max_jobs=1)['blocked_reasons'] == []


def test_live_dispatch_cannot_enter_the_synthetic_callback_loop(main_fixture):
    value = main_fixture
    with api.MainLiveDispatch.open(value['protocol'], value['prior'], root=value['root'],
                                   clock=value['clock']) as dispatch:
        with pytest.raises(RuntimeError, match='unavailable on a live dispatcher'):
            dispatch.run_synthetic(dispatch.pending_jobs()[0],
                                   runner=lambda *_: pytest.fail('No callback'), reader=lambda *_: {})
        assert dispatch.report()['claimed_jobs'] == 0


def test_startup_overhead_cannot_extend_the_frozen_child_deadline(main_fixture):
    original_service = main_fixture['service'].ensure_family
    original_runner = main_fixture['runner']
    observed = []

    def delayed(*args, **kwargs):
        result = original_service(*args, **kwargs)
        main_fixture['clock'].value += 6
        return result

    def runner(command, log_path, timeout, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        claim = json.loads(Path(request['parent_binding']['claim_path']).read_text())
        assert request['reservation']['deadline'] == claim['claimed_at'] + 15
        observed.append(timeout)
        return original_runner(command, log_path, timeout, **kwargs)

    main_fixture['service'].ensure_family = delayed
    assert run(main_fixture, runner=runner, max_jobs=1)['blocked_reasons'] == []
    assert observed == [9]


def test_unsupported_reader_capability_is_rejected_before_work(main_fixture, monkeypatch):
    from helpers import maniguard_main_results
    monkeypatch.setattr(maniguard_main_results, 'SUPPORTED_PROFILES', {'phase2_learned'})
    value = main_fixture
    controller = value['protocol']['controllers'][0]
    controller['resource_contract'] = protocol_api.controller_resource_contract('benchmark_original')
    controller['resource_contract_sha256'] = record_hash(controller['resource_contract'])
    jobs = protocol_api.compile_jobs(value['protocol'])
    value['protocol'].update(grid_sha256=record_hash(jobs), envelope=protocol_api.action_envelope(jobs))
    rebind(value)
    save_inputs(value)
    with pytest.raises(RuntimeError, match='Independent execution replay'):
        run(value)
    assert value['events'] == []
    assert not (value['root'] / 'runs').exists()


@pytest.mark.parametrize('mode', ['learned', 'fixed', 'contextual', 'original'])
def test_independently_replayed_row_keeps_identity_and_category_counts(tmp_path, monkeypatch, mode):
    from test_maniguard_main_results import fixture_episode, original_fixture, read
    case = original_fixture(tmp_path, monkeypatch) if mode == 'original' else fixture_episode(
        tmp_path, monkeypatch, mode=mode)
    row = read(case)
    assert row['valid'], row['invalid_reasons']
    settled = {}

    def settle(event_id, measured, **kwargs):
        settled[event_id] = {'counts': measured, **kwargs}

    adapter = SimpleNamespace(_usable=lambda: None, protocol=case['protocol'], root=case['root'],
                              evidence_kind='main_audited', settle_extra=settle,
                              record_result=lambda _job_id, retained: retained)
    claim = {'job': case['job'], 'output': str(case['output']),
             'allowance_events': {'startup': 'startup', 'failed_work': 'failed_work'}}
    retained = api.MainLiveDispatch.record_audited(adapter, claim, row)
    assert retained['valid'] is True
    assert retained['audited_main_row'] == row
    assert retained['actor_sha256'] == case['job']['actor_sha256']
    assert retained['controller_sha256'] == case['job']['controller_sha256']
    assert retained['costs']['physics_steps'] == 72
    assert settled['startup']['counts']['physics_steps'] == 12
    assert settled['failed_work']['counts']['physics_steps'] == 0
    assert retained['unknown_cost_fields'] == []


def test_child_rejects_copied_resigned_claim_registry_even_with_real_inherited_lease(main_fixture):
    original = main_fixture['runner']

    def runner(command, *args, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        relocated = main_fixture['root'] / 'relocated-registry'
        old_root = Path(request['parent_binding']['dispatch_root'])
        shutil.copytree(old_root, relocated)
        new_claim_path = relocated / Path(request['parent_binding']['claim_path']).relative_to(old_root)
        claim = json.loads(new_claim_path.read_text())
        claim['dispatch_state_path'] = str(relocated / 'state.json')
        claim = _signed(claim)
        new_claim_path.write_text(json.dumps(claim))
        state = json.loads((relocated / 'state.json').read_text())
        state['claims'][request['job']['episode_id']]['live_claim_sha256'] = claim['sha256']
        (relocated / 'state.json').write_text(json.dumps(_signed(state)))
        request['parent_binding'].update(dispatch_root=str(relocated), claim_path=str(new_claim_path),
                                         claim_sha256=claim['sha256'])
        with pytest.raises(ValueError, match='canonical live study directory'):
            api.verify_parent_binding(request, parent_pid=os.getpid())
        return original(command, *args, **kwargs)

    assert run(main_fixture, runner=runner, max_jobs=1)['blocked_reasons'] == []


def test_child_rejects_claim_state_disagreeing_with_signed_anchor(main_fixture):
    original = main_fixture['runner']

    def runner(command, *args, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        anchor_path = Path(request['parent_binding']['dispatch_root']) / 'binding.json'
        saved = anchor_path.read_bytes()
        anchor = json.loads(saved)
        anchor['prior_sha256'] = 'changed'
        anchor_path.write_text(json.dumps(_signed(anchor)))
        try:
            with pytest.raises(ValueError, match='immutable study anchor'):
                api.verify_parent_binding(request, parent_pid=os.getpid())
        finally:
            anchor_path.write_bytes(saved)
        return original(command, *args, **kwargs)

    assert run(main_fixture, runner=runner, max_jobs=1)['blocked_reasons'] == []


@pytest.mark.parametrize('failure', [None, 'timeout', 'wrong_command', 'wrong_binding'])
def test_parent_routes_production_audit_and_rejects_worker_failures(main_fixture, failure):
    value = main_fixture

    def audit_runner(command, log_path, timeout, *, env, termination_grace, on_started):
        assert env['CUDA_VISIBLE_DEVICES'] == ''
        assert timeout == value['contract']['audit_timeout_seconds']
        request = json.loads(Path(command[-1]).read_text())
        process = {'command': command, 'pid': 45678, 'returncode': 0,
                   'timed_out': False, 'interrupted': False, 'process_group_remaining': False}
        on_started(process)
        receipt = json.loads(Path(request['process_receipt_path']).read_text())
        row = value['reader'](request['output'], value['protocol'], request['job'],
                              process_receipt=receipt, root=value['root'], config_dir=None)
        result = _signed({'schema': api.audit_worker.RESULT_SCHEMA,
            'protocol_sha256': value['protocol']['sha256'],
            'job_sha256': record_hash(request['job']), 'output': request['output'],
            'request_sha256': request['sha256'],
            'audit_worker_sha256': api._hash_file(value['root'] / api.audit_worker.SOURCE_PATH),
            'audit_reader_sha256': api._hash_file(value['root'] / api.audit_worker.READER_PATH),
            'row': row})
        if failure == 'wrong_binding':
            result = _signed({**result, 'request_sha256': '0' * 64})
        Path(request['result_path']).write_text(json.dumps(result))
        if failure == 'timeout':
            process['timed_out'] = True
        if failure == 'wrong_command':
            process['command'] = ['different-worker']
        return process

    report = run(value, max_jobs=1, reader=None, audit_runner=audit_runner)
    saved = json.loads(Path(report['state_path']).read_text())
    claim = next(iter(saved['claims'].values()))
    row = claim['row']['audited_main_row']
    assert row['valid'] is (failure is None)
    assert bool(report['blocked_reasons']) is (failure is not None)
    if failure is not None:
        assert row['unknown_cost_fields'] == list(api.WORK_COUNTS)
        assert row['outcome'] is None
