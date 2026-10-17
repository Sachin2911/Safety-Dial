"""Launcher wiring tests with fake episodes/services; no model server or simulator starts."""

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.maniguard_phase2_train as train
from helpers.maniguard_phase2_model import genome_hash, identity_genome, load_model
from helpers.maniguard_phase2_optim import record_hash
from helpers.maniguard_phase2_services import (
    PRIOR_COUNTS,
    PRIOR_SCHEMA,
    atomic_json,
    execution_lease,
    file_hash,
)


class FakeCoordinator:
    """Persist claims/outcomes as the real coordinator does; optimization is tested separately."""

    def __init__(self, protocol):
        self.protocol = protocol
        self.status = 'training'
        genome = identity_genome().tolist()
        self.jobs = [{'job_id': f'job-{index}', 'family': family, 'scene': 'task0/base', 'seed': 0,
                      'sampling_seed': 0, 'actor_sha256': genome_hash(genome), 'mode': 'learned',
                      'fixed_choices': None, 'purpose': 'train', 'stochastic': False, 'max_steps': 100,
                      'genome': genome}
                     for index, family in enumerate(('jar', 'stack', 'jar', 'clutter'))]
        self.claims = {}
        self.rows = {}

    def next_jobs(self):
        return [deepcopy(job) for job in self.jobs
                if job['job_id'] not in self.claims and job['job_id'] not in self.rows]

    def mark_started(self, job_id, output):
        assert job_id not in self.claims and job_id not in self.rows
        self.claims[job_id] = str(output)

    def record_result(self, job_id, row):
        assert self.claims.pop(job_id) == row['output']
        self.rows[job_id] = row
        if not row['valid']:
            self.status = 'blocked'
        elif len(self.rows) == len(self.jobs):
            self.status = 'complete'
        return self.report()

    def report(self):
        totals = {}
        for row in self.rows.values():
            for key, value in row['costs'].items():
                totals[key] = totals.get(key, 0) + value
        return {'status': self.status, 'in_flight_claims': deepcopy(self.claims),
                'cost_totals': totals, 'physical_attempts_recorded': len(self.rows)}

    def save(self, path):
        atomic_json(path, self.__dict__)

    @classmethod
    def load(cls, path):
        value = cls.__new__(cls)
        value.__dict__.update(json.loads(Path(path).read_text()))
        return value


@pytest.fixture
def harness(tmp_path, monkeypatch):
    now = [200.0]
    protocol = {'status': 'frozen', 'input_files': {}, 'implementation_files': {},
                'resource_envelope': {'gpu_hours_ceiling': 1, 'policy_actions_max': 10000,
                                     'exhaustion_safe_successful_queries_max': 10000}}
    for field, relative in (
            ('input_files', 'configs/maniguard/phase2/completion.json'),
            ('implementation_files', 'experiments/helpers/maniguard_phase2_model.py')):
        protocol[field][relative] = file_hash(train.ROOT / relative)
    protocol['sha256'] = record_hash(protocol)
    protocol_path = tmp_path / 'protocol.json'
    atomic_json(protocol_path, protocol)
    evidence = tmp_path / 'preparation.json'
    evidence.write_text('{"counts_audited": true}\n')
    prior = {'schema': PRIOR_SCHEMA, 'timestamp': 200.0, 'reservation_started_at': 100.0,
             'reserved_gpus': 1, 'wall_seconds': 100.0, 'allocated_gpu_seconds': 100.0,
             'counts': dict.fromkeys(PRIOR_COUNTS, 0),
             'sources': [{'path': str(evidence), 'sha256': file_hash(evidence)}]}
    prior['sha256'] = record_hash(prior)
    prior_path = tmp_path / 'prior.json'
    atomic_json(prior_path, prior)
    validation_calls = []

    def validate(value, **kwargs):
        assert value['status'] == 'frozen' and kwargs == {'require_frozen': True}
        validation_calls.append(len(validation_calls))

    # Protocol freeze/source validation and the optimizer state machine have separate
    # tests. Keep this fixture focused on durable launcher/process/service sequencing.
    monkeypatch.setattr(train, 'validate_protocol', validate)
    monkeypatch.setattr(train, '_source_files', dict)
    services, runs, rows = [], [], {}
    output = tmp_path / 'pilot'

    def ensure_family(family, audit_dir, *, startup_timeout):
        assert startup_timeout > 0
        services.append(family)
        now[0] += 5
        record = {'family': family, 'status': 'reused', 'metadata': {'checkpoint': family},
                  'elapsed_seconds': 5}
        atomic_json(audit_dir / 'service.json', record)
        return record

    def process(command, log, timeout, *, env, on_started):
        assert 0 < timeout <= 3600 and env['OMNI_KIT_ALLOW_ROOT'] == '1'
        episode = Path(command[command.index('--output') + 1])
        claim = json.loads((output / 'coordinator.json').read_text())['claims']
        assert claim == {episode.name: str(episode)}  # Durable before subprocess creation.
        attempt = json.loads(Path(log).with_suffix('.json').read_text())
        assert attempt['status'] == 'claimed'
        on_started({'pid': 1234, 'returncode': None})
        assert json.loads(Path(log).with_suffix('.json').read_text())['process']['pid'] == 1234
        actor = Path(command[command.index('--model') + 1])
        genome, _, metadata = load_model(actor)
        assert metadata['actor_sha256'] == genome_hash(genome)
        assert metadata['protocol_sha256'] == protocol['sha256']
        episode.mkdir(parents=True)
        pins = dict(protocol['input_files'], **protocol['implementation_files'])
        atomic_json(episode / 'job.json', {'code_hashes': pins})
        for name in pins:
            archived = episode / 'study_source' / name
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes((train.ROOT / name).read_bytes())
        rows[str(episode)] = {'output': str(episode), 'valid': True, 'invalid_reasons': [],
                              'ppo_eligible': False, 'costs': {'policy_control_steps': 8,
                              'policy_queries': 2, 'physics_steps': 48}, 'cost_issues': [],
                              'outcome': {'U': 1}}
        runs.append(episode.name)
        now[0] += 10
        return {'returncode': 0, 'pid': 1234, 'timed_out': False,
                'process_elapsed_seconds': 10}

    arguments = {'service': SimpleNamespace(ensure_family=ensure_family),
                 'process_runner': process, 'result_reader': lambda path: deepcopy(rows[str(path)]),
                 'scan': list, 'wall': lambda: now[0],
                 'coordinator_factory': FakeCoordinator, 'coordinator_loader': FakeCoordinator.load,
                 'execution_lock': lambda: execution_lease(tmp_path / 'execution.lock')}
    return SimpleNamespace(protocol=protocol, protocol_path=protocol_path, prior_path=prior_path,
                           output=output, now=now, services=services, runs=runs, rows=rows,
                           validation_calls=validation_calls, kwargs=arguments)


def invoke(harness, **overrides):
    return train.run_pilot(harness.protocol_path, harness.output, harness.prior_path,
                           **dict(harness.kwargs, **overrides))


def test_full_fake_run_persists_claims_groups_family_and_counts_startup(harness):
    report = invoke(harness)
    assert report['status'] == 'complete'
    assert harness.services == ['jar', 'jar', 'stack', 'clutter']
    assert harness.runs == ['job-0', 'job-2', 'job-1', 'job-3']
    assert report['coordinator_report']['physical_attempts_recorded'] == 4
    assert report['reservation']['counts']['policy_control_steps'] == 32
    assert report['reservation']['allocated_gpu_seconds'] == 160
    assert len(harness.validation_calls) == 9  # At start and twice before each episode.
    assert len(report['episode_attempts']) == 4
    assert all(json.loads(Path(path).read_text())['status'] == 'audited'
               for path in report['episode_attempts'])


def test_failure_retains_costs_and_stops_without_scoring_or_next_launch(harness):
    original = harness.kwargs['process_runner']

    def failed(*args, **kwargs):
        record = original(*args, **kwargs)
        record.update(returncode=-9, timed_out=True)
        return record

    report = invoke(harness, process_runner=failed)
    assert report['status'] == 'blocked' and harness.runs == ['job-0']
    assert report['coordinator_report']['cost_totals']['physics_steps'] == 48
    attempt = json.loads(Path(report['episode_attempts'][0]).read_text())
    assert not attempt['valid'] and attempt['process']['timed_out']
    assert 'Episode wrapper did not exit cleanly' in attempt['invalid_reasons']


def test_resume_never_reruns_a_persisted_claim(harness):
    def died_before_result(*args, **kwargs):
        raise RuntimeError('Injected launcher failure after durable claim')

    report = invoke(harness, process_runner=died_before_result)
    assert report['status'] == 'blocked' and harness.runs == []
    assert report['coordinator_report']['in_flight_claims']
    before = list(harness.services)
    resumed = invoke(harness, resume=True)
    assert resumed['status'] == 'blocked_inflight'
    assert harness.services == before and harness.runs == []
    assert 'never rerun automatically' in resumed['error']


def test_pause_then_resume_preserves_family_and_avoids_duplicate_attempts(harness):
    first = invoke(harness, max_new_episodes=1)
    assert first['status'] == 'paused' and harness.runs == ['job-0']
    second = invoke(harness, resume=True)
    assert second['status'] == 'complete'
    assert harness.runs == ['job-0', 'job-2', 'job-1', 'job-3']
    assert second['reservation']['counts']['policy_queries'] == 8


def test_service_startup_consumes_same_reservation_and_can_prevent_episode(harness):
    original = harness.kwargs['service'].ensure_family

    def expensive(*args, **kwargs):
        record = original(*args, **kwargs)
        harness.now[0] += 3490
        return record

    report = invoke(harness, service=SimpleNamespace(ensure_family=expensive))
    assert report['status'] == 'blocked' and harness.runs == []
    assert report['reservation']['allocated_gpu_seconds'] == 3595
    assert 'shutdown margin' in report['error']
    assert len(report['service_events']) == 1 and report['episode_attempts'] == []


@pytest.mark.parametrize('tamper', ['declared', 'archived'])
def test_episode_must_archive_frozen_source_hashes_before_outcome_acceptance(harness, tamper):
    original = harness.kwargs['process_runner']

    def changed(*args, **kwargs):
        record = original(*args, **kwargs)
        episode = harness.output / 'episodes/job-0'
        name = next(iter(harness.protocol['input_files']))
        if tamper == 'declared':
            atomic_json(episode / 'job.json', {'code_hashes': {}})
        else:
            (episode / 'study_source' / name).write_text('changed completion semantics')
        return record

    report = invoke(harness, process_runner=changed)
    assert report['status'] == 'blocked' and harness.runs == ['job-0']
    attempt = json.loads(Path(report['episode_attempts'][0]).read_text())
    assert attempt['costs']['policy_queries'] == 2
    assert not attempt['valid'] and any('source' in text for text in attempt['invalid_reasons'])


def test_active_legacy_process_blocks_before_service_or_claim(harness):
    report = invoke(harness, scan=lambda: [{'pid': 42, 'episode_script': 'maniguard_episode.py'}])
    assert report['status'] == 'blocked' and harness.services == [] and harness.runs == []
    assert report['episode_attempts'] == []


def test_changed_implementation_on_resume_blocks_before_launch(harness, monkeypatch):
    invoke(harness, max_new_episodes=1)
    monkeypatch.setattr(train, '_source_files', lambda: {'changed.py': 'different'})
    report = invoke(harness, resume=True)
    assert report['status'] == 'blocked' and harness.runs == ['job-0']
    assert 'implementation changed on resume' in report['error']


def test_protocol_is_revalidated_before_each_new_episode(harness, monkeypatch):
    validations = []

    def changed_protocol(protocol, **kwargs):
        validations.append(True)
        if len(validations) == 4:
            raise ValueError('Pinned protocol input/source changed')

    monkeypatch.setattr(train, 'validate_protocol', changed_protocol)
    report = invoke(harness)
    assert report['status'] == 'blocked' and harness.runs == ['job-0']
    assert 'Pinned protocol input/source changed' in report['error']


def test_unfrozen_protocol_is_refused_before_run_directory_creation(tmp_path):
    # No validator stub here: exercise the production freeze guard.
    from helpers.maniguard_phase2_protocol import build_pilot

    path = tmp_path / 'unfrozen.json'
    atomic_json(path, build_pilot())
    with pytest.raises(ValueError, match='freeze incomplete'):
        train.run_pilot(path, tmp_path / 'never-created', tmp_path / 'no-prior.json')
    assert not (tmp_path / 'never-created').exists()
