"""Run the frozen phase-two pilot through supervised policy service and fresh episodes.

Start this entry point as a managed supervisor job. It never launches itself as a daemon.
A saved in-flight claim requires explicit output reconciliation; --resume never reruns it.
"""

import argparse
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from helpers.maniguard_phase2_coordinator import PilotCoordinator
from helpers.maniguard_phase2_model import genome_hash, load_model, save_model
from helpers.maniguard_phase2_protocol import load_protocol, validate_protocol
from helpers.maniguard_phase2_results import read_episode
from helpers.maniguard_phase2_services import (
    PolicyService,
    ReservationBudget,
    active_episode_processes,
    atomic_json,
    execution_lease,
    file_hash,
    run_fresh_episode,
)

SCHEMA = 'maniguard-phase2-launcher-v1'


def choose_job(jobs, previous_family=None):
    """Keep the current checkpoint when possible within one frozen dependency block."""
    matching = [job for job in jobs if job['family'] == previous_family]
    return (matching or jobs)[0]


def episode_command(job, model, output, *, python=sys.executable, port=18000, encoder_device='cuda'):
    command = [str(python), '-u', str(ROOT / 'experiments/scripts/maniguard_phase2_episode.py'),
               '--family', job['family'], '--scene', job['scene'], '--purpose', job['purpose'],
               '--seed', str(job['seed']), '--sampling-seed', str(job['sampling_seed']),
               '--model', str(model), '--mode', job['mode'], '--output', str(output),
               '--port', str(port), '--encoder-device', encoder_device]
    if job['fixed_choices'] is not None:
        command += ['--fixed-choices', *[str(value) for value in job['fixed_choices']]]
    if job['stochastic']:
        command += ['--stochastic']
    return command


def _source_files():
    files = list((ROOT / 'experiments/helpers').glob('maniguard_phase2_*.py'))
    files += [Path(__file__), ROOT / 'experiments/scripts/maniguard_phase2_episode.py',
              ROOT / 'experiments/scripts/maniguard_serve.py', ROOT / 'scripts/maniguard/serve.sh']
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(set(files))}


def _save_actor(output, job, protocol_sha256):
    directory = output / 'actors'
    directory.mkdir(exist_ok=True)
    path = directory / (job['actor_sha256'] + '.npz')
    if path.exists():
        genome, _, metadata = load_model(path)
        if genome_hash(genome) != job['actor_sha256'] or metadata['protocol_sha256'] != protocol_sha256:
            raise ValueError('Saved actor differs from the dispatched controller/protocol')
    else:
        save_model(path, job['genome'], metadata={'protocol_sha256': protocol_sha256,
                                                 'actor_sha256': job['actor_sha256']})
    return path


def _unreadable_result(job, output, error):
    return {'output': str(output), 'episode_id': str(output), 'family': job['family'],
            'scene': job['scene'], 'seed': job['seed'], 'sampling_seed': job['sampling_seed'],
            'actor_sha256': job['actor_sha256'], 'mode': job['mode'],
            'fixed_choices': job['fixed_choices'], 'purpose': job['purpose'],
            'stochastic': job['stochastic'], 'action_horizon': job['max_steps'],
            'valid': False, 'full_episode': True, 'short_check': False,
            'invalid_reasons': [f'Result auditing failed: {error}'], 'outcome': None,
            'trajectory': None, 'ppo_eligible': False, 'costs': {}, 'observed_work': {},
            'cost_issues': ['Attempt costs require reconciliation; absent counters are not zeros'],
            'missing_cost_fields': ['all']}


def audit_episode_sources(output, protocol):
    """Bind archived episode code/config to frozen pins, not just today's checkout."""
    output = Path(output)
    job = json.loads((output / 'job.json').read_text())
    pinned = dict(protocol['input_files'])
    pinned.update(protocol['implementation_files'])
    hashes = job.get('code_hashes', {})
    for name, digest in pinned.items():
        if hashes.get(name) != digest:
            raise ValueError(f'Episode source pin mismatch: {name}')
        if file_hash(output / 'study_source' / name) != digest:
            raise ValueError(f'Episode archived source differs from its hash: {name}')
    return pinned


def run_pilot(protocol_path, output, prior_accounting_path, *, resume=False, reserved_gpus=1,
              episode_timeout=3600, startup_timeout=600, port=18000, encoder_device='cuda',
              max_new_episodes=None, service=None, process_runner=run_fresh_episode,
              result_reader=read_episode, scan=active_episode_processes, wall=time.time,
              coordinator_factory=PilotCoordinator, coordinator_loader=PilotCoordinator.load,
              execution_lock=execution_lease):
    """Execute serial fresh episodes, switching family only at a safe process boundary.

    Injected dependencies support tests without service control or simulation. Production
    always validates the protocol and prior cost evidence before creating the run directory.
    """
    protocol = load_protocol(protocol_path)
    validate_protocol(protocol, require_frozen=True)
    prior = json.loads(Path(prior_accounting_path).read_text())
    budget = ReservationBudget(prior, protocol['resource_envelope'], reserved_gpus=reserved_gpus,
                               wall=wall)
    if episode_timeout <= 0 or startup_timeout <= 0 or (max_new_episodes is not None and
                                                       max_new_episodes < 1):
        raise ValueError('Timeouts and invocation episode limit must be positive')
    if encoder_device != 'cuda':
        raise ValueError('The frozen primary protocol uses the CUDA encoder')
    output = Path(output).resolve()
    if resume:
        if not output.is_dir():
            raise ValueError('Cannot resume a missing run directory')
    else:
        output.mkdir(parents=True, exist_ok=False)
    lock = (output / 'launcher.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError('Another launcher owns this pilot output') from error
    manifest_path, state_path = output / 'launch.json', output / 'coordinator.json'
    source_hashes = _source_files()
    manifest = None
    lease = None
    lease_acquired = False
    try:
        lease = execution_lock()
        lease.__enter__()
        lease_acquired = True
        if resume:
            manifest = json.loads(manifest_path.read_text())
            expected = {'protocol_sha256': protocol['sha256'], 'prior_sha256': prior['sha256'],
                        'reserved_gpus': reserved_gpus, 'source_hashes': source_hashes}
            if manifest['schema'] != SCHEMA or any(manifest.get(key) != value
                                                  for key, value in expected.items()):
                raise ValueError('Frozen launcher inputs or implementation changed on resume')
            coordinator = coordinator_loader(state_path)
        else:
            manifest = {'schema': SCHEMA, 'protocol_sha256': protocol['sha256'],
                        'prior_sha256': prior['sha256'], 'reserved_gpus': reserved_gpus,
                        'source_hashes': source_hashes, 'started_at': wall(), 'status': 'initializing',
                        'last_family': None, 'service_events': [], 'episode_attempts': []}
            atomic_json(manifest_path, manifest)
            atomic_json(output / 'protocol.json', protocol)
            atomic_json(output / 'prior-accounting.json', prior)
            for relative, digest in source_hashes.items():
                raw = (ROOT / relative).read_bytes()
                destination = output / 'source' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
                if file_hash(destination) != digest:
                    raise ValueError('Launcher source snapshot changed during acquisition')
            coordinator = coordinator_factory(protocol)
            coordinator.save(state_path)
        if coordinator.protocol['sha256'] != protocol['sha256']:
            raise ValueError('Saved coordinator uses a different protocol')
        report = coordinator.report()
        if report['in_flight_claims']:
            manifest.update(status='blocked_inflight', unresolved_claims=report['in_flight_claims'],
                            active_episodes=scan(), error='Reconcile persisted attempts before resume; '
                                                         'claimed episodes are never rerun automatically')
            return manifest
        service = service or PolicyService(port=port, wall=wall)
        manifest['status'] = 'running'
        completed_now = 0
        while coordinator.status not in ('blocked', 'complete'):
            if max_new_episodes is not None and completed_now >= max_new_episodes:
                manifest['status'] = 'paused'
                break
            jobs = coordinator.next_jobs()
            if not jobs:
                raise RuntimeError('No pending work but coordinator is not terminal; reconcile state')
            validate_protocol(protocol, require_frozen=True)
            job = choose_job(jobs, manifest['last_family'])
            costs = coordinator.report()['cost_totals']
            manifest['reservation'] = budget.check(costs, next_actions=job['max_steps'],
                                                   next_queries=job['max_steps'])
            atomic_json(manifest_path, manifest)
            if scan():
                raise RuntimeError('A manipulation episode is active; cannot start another pilot episode')
            remaining = budget.remaining_wall_seconds(costs)
            if remaining <= 15:
                raise RuntimeError('GPU reservation has no safe startup/shutdown margin')
            event_dir = output / 'services' / f'event-{len(manifest["service_events"]):04d}'
            # Persist the event reference before service acquisition, even if it fails.
            manifest['service_events'].append(str(event_dir / 'service.json'))
            atomic_json(manifest_path, manifest)
            handshake = service.ensure_family(job['family'], event_dir,
                                               startup_timeout=min(startup_timeout, remaining - 10))
            manifest['last_family'] = job['family']
            costs = coordinator.report()['cost_totals']
            budget.check(costs, next_actions=job['max_steps'], next_queries=job['max_steps'])
            remaining = budget.remaining_wall_seconds(costs)
            if remaining <= 15:
                raise RuntimeError('GPU reservation has no full-process shutdown margin')
            model = _save_actor(output, job, protocol['sha256'])
            episode_output = output / 'episodes' / job['job_id']
            if episode_output.exists():
                raise RuntimeError('Unclaimed output already exists; reconcile it instead of rerunning')
            attempt_path = output / 'attempts' / (job['job_id'] + '.json')
            attempt_path.parent.mkdir(parents=True, exist_ok=True)
            command = episode_command(job, model, episode_output, port=port,
                                       encoder_device=encoder_device)
            attempt = {'job': {key: value for key, value in job.items() if key != 'genome'},
                       'output': str(episode_output), 'model': str(model), 'command': command,
                       'policy_handshake': handshake, 'status': 'planned',
                       'reservation_before': budget.snapshot(costs)}
            if scan():
                raise RuntimeError('An episode appeared during setup; refusing concurrent launch')
            atomic_json(attempt_path, attempt)
            coordinator.mark_started(job['job_id'], episode_output)
            coordinator.save(state_path)
            manifest['episode_attempts'].append(str(attempt_path))
            atomic_json(manifest_path, manifest)
            attempt['status'] = 'claimed'
            atomic_json(attempt_path, attempt)

            def started(process_record, attempt_record=attempt, destination=attempt_path):
                attempt_record.update(status='running', process=process_record)
                atomic_json(destination, attempt_record)

            environment = dict(os.environ, OMNI_KIT_ALLOW_ROOT='1', OMP_NUM_THREADS='4',
                               OPENBLAS_NUM_THREADS='4')
            validate_protocol(protocol, require_frozen=True)
            remaining = budget.remaining_wall_seconds(costs)
            if remaining <= 15:
                raise RuntimeError('GPU reservation expired during durable launch preparation')
            process_record = process_runner(command, attempt_path.with_suffix('.log'),
                                             min(episode_timeout, remaining - 10), env=environment,
                                             on_started=started)
            attempt.update(status='process_finished', process=process_record)
            atomic_json(attempt_path, attempt)
            try:
                row = result_reader(episode_output)
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
                row = _unreadable_result(job, episode_output, error)
            try:
                attempt['verified_episode_sources'] = audit_episode_sources(episode_output, protocol)
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
                row['valid'] = False
                row['ppo_eligible'] = False
                row.setdefault('invalid_reasons', []).append(str(error))
            if (process_record.get('returncode') != 0 or process_record.get('timed_out') or
                    process_record.get('interrupted') or process_record.get('error')):
                row['valid'] = False
                row['ppo_eligible'] = False
                row.setdefault('invalid_reasons', []).append('Episode wrapper did not exit cleanly')
            report = coordinator.record_result(job['job_id'], row)
            coordinator.save(state_path)
            attempt.update(status='audited', valid=row.get('valid'), costs=row.get('costs'),
                           cost_issues=row.get('cost_issues'), invalid_reasons=row.get('invalid_reasons'),
                           outcome=row.get('outcome'), coordinator_status=report['status'])
            atomic_json(attempt_path, attempt)
            completed_now += 1
            manifest['coordinator_report'] = report
            manifest['reservation'] = budget.snapshot(report['cost_totals'])
            atomic_json(manifest_path, manifest)
            if report['status'] == 'blocked':
                break
            budget.check(report['cost_totals'])
        if manifest['status'] != 'paused':
            manifest['status'] = coordinator.status
        manifest['coordinator_report'] = coordinator.report()
        manifest['reservation'] = budget.snapshot(coordinator.report()['cost_totals'])
        return manifest
    except (OSError, RuntimeError, ValueError, KeyError, KeyboardInterrupt, SystemExit) as error:
        if manifest is None:
            raise
        manifest.update(status='blocked', error=f'{type(error).__name__}: {error}')
        if 'coordinator' in locals():
            manifest['coordinator_report'] = coordinator.report()
            manifest['reservation'] = budget.snapshot(coordinator.report()['cost_totals'])
        return manifest
    finally:
        if manifest is not None:
            manifest['updated_at'] = wall()
            atomic_json(manifest_path, manifest)
        if lease_acquired:
            lease.__exit__(None, None, None)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-accounting', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reserved-gpus', type=int, default=1)
    parser.add_argument('--episode-timeout', type=float, default=3600)
    parser.add_argument('--startup-timeout', type=float, default=600)
    parser.add_argument('--port', type=int, default=18000)
    parser.add_argument('--max-new-episodes', type=int)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    report = run_pilot(args.protocol, args.output, args.prior_accounting, resume=args.resume,
                       reserved_gpus=args.reserved_gpus, episode_timeout=args.episode_timeout,
                       startup_timeout=args.startup_timeout, port=args.port,
                       max_new_episodes=args.max_new_episodes)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] in ('complete', 'paused') else 1


if __name__ == '__main__':
    raise SystemExit(main())
