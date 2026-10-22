"""Run the frozen replicate diagnostic on fixed per-GPU Clutter policy routes.

Reuses the pilot launcher's one-GPU worker machinery, the frozen episode runner and
the frozen result audits. The diagnostic has its own prospective protocol, budget and
state; it never touches pilot state. Touch ``<deployment>/pause`` or send SIGTERM to
stop launching and drain in-flight work.
"""

import argparse
import fcntl
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'experiments/scripts'))

from maniguard_phase2_train import _unreadable_result, audit_episode_sources
from maniguard_pilot_parallel import (
    ENVIRONMENT_KEYS,
    gpu_inventory,
    gpu_memory,
    gpu_uuids,
    kit_render_gpu,
    prepare_worker_folder,
    verify_routes,
    worker_command,
    worker_environment,
)
from maniguard_pilot_void import interaction_evidence

from helpers.maniguard_phase2_model import genome_hash, load_model, save_model
from helpers.maniguard_phase2_protocol import load_protocol, validate_protocol
from helpers.maniguard_phase2_results import read_episode
from helpers.maniguard_phase2_services import (
    active_episode_processes,
    atomic_json,
    execution_lease,
    file_hash,
    run_fresh_episode,
)
from helpers.maniguard_pilot_parallel import run_parallel, validate_routes
from helpers.maniguard_replicate import (
    DiagnosticBudget,
    ReplicateCoordinator,
    analyze,
    load_replicate_protocol,
)

SCHEMA = 'maniguard-replicate-launch-v1'
SOURCES = ('experiments/scripts/maniguard_replicate_run.py', 'experiments/helpers/maniguard_replicate.py',
           'experiments/scripts/maniguard_pilot_parallel.py', 'experiments/scripts/maniguard_pilot_worker.py',
           'experiments/helpers/maniguard_pilot_parallel.py', 'experiments/scripts/maniguard_pilot_void.py')
AUTO_VOID_PER_JOB = 2
AUTO_VOID_PER_DEPLOYMENT = 8


def actor_path(output, controller, pilot_sha):
    """Copy the pilot actor into the diagnostic run after checking its genome hash."""
    source = ROOT / controller['actor_file']
    if file_hash(source) != controller['actor_file_sha256']:
        raise ValueError('Pilot actor file differs from the protocol pin')
    genome, _, metadata = load_model(source)
    if genome_hash(genome) != controller['actor_sha256'] or metadata['protocol_sha256'] != pilot_sha:
        raise ValueError('Pilot actor content differs from the protocol pin')
    destination = output / 'actors' / (controller['actor_sha256'] + '.npz')
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        save_model(destination, genome, metadata={'protocol_sha256': pilot_sha,
                                                  'actor_sha256': controller['actor_sha256']})
    return destination


def run(protocol_path, config_path, output, deployment, *, max_new_jobs=None):
    protocol = load_replicate_protocol(protocol_path)
    pilot = load_protocol()
    validate_protocol(pilot, require_frozen=True)
    if pilot['sha256'] != protocol['pilot_protocol_sha256']:
        raise ValueError('Frozen pilot protocol differs from the diagnostic pin')
    config = json.loads(Path(config_path).read_text())
    validate_routes(config['routes'], ['clutter'])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    folder = output / 'deployments' / deployment
    folder.mkdir(parents=True, exist_ok=False)
    clock_path = output / 'diagnostic.json'
    if clock_path.exists():
        clock = json.loads(clock_path.read_text())
        if clock['protocol_sha256'] != protocol['sha256']:
            raise ValueError('Existing diagnostic run uses another protocol')
    else:
        clock = {'protocol_sha256': protocol['sha256'], 'started_at': time.time(),
                 'instance_id': config['instance_id']}
        atomic_json(clock_path, clock)
    budget = DiagnosticBudget(protocol['resource_envelope'], clock['started_at'])
    budget.check()
    state_path = output / 'coordinator.json'
    uuids = gpu_uuids()
    stop = {'requested': False}

    def request_stop(signum, frame):
        stop['requested'] = True

    previous = {number: signal.signal(number, request_stop) for number in (signal.SIGTERM, signal.SIGINT)}
    manifest = {'schema': SCHEMA, 'status': 'checking', 'started_at': time.time(),
                'protocol_sha256': protocol['sha256'], 'pilot_protocol_sha256': pilot['sha256'],
                'config': config, 'config_sha256': file_hash(config_path),
                'source_hashes': {name: file_hash(ROOT / name) for name in SOURCES},
                'repository_commit': subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                                    capture_output=True, text=True, check=True).stdout.strip(),
                'host': platform.uname()._asdict(), 'gpu_inventory': gpu_inventory(),
                'diagnostic_started_at': clock['started_at'], 'episode_attempts': [], 'error': None}
    atomic_json(folder / 'launch.json', manifest)
    for name in SOURCES:
        destination = folder / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    (folder / 'source' / 'replicate-v1.json').write_bytes(Path(protocol_path).read_bytes())
    coordinator = None
    try:
        with (output / 'launcher.lock').open('a') as lock, execution_lease():
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if active_episode_processes():
                raise RuntimeError('An existing episode must be reconciled before takeover')
            if state_path.exists():
                coordinator = ReplicateCoordinator.load(protocol, state_path)
            else:
                coordinator = ReplicateCoordinator(protocol)
                coordinator.save(state_path)
            if coordinator.report()['in_flight_claims']:
                raise RuntimeError('Existing claims require reconciliation; never replay them')
            manifest['policy_routes'] = verify_routes(config['routes'])
            manifest['worker_folders'] = {
                slot['slot_id']: prepare_worker_folder(output / 'workers' / slot['slot_id'])
                for slot in validate_routes(config['routes'], ['clutter'])}
            actors = {c['actor_sha256']: str(actor_path(output, c, pilot['sha256']))
                      for c in protocol['controllers']}
            manifest.update(gpu_memory_before=gpu_memory(), initial_report=coordinator.report(),
                            status='running')
            atomic_json(folder / 'launch.json', manifest)
            attempt_paths = {}
            auto_voids = {'count': 0, 'by_job': {}}
            void_folder = output / 'void-attempts' / deployment

            def release(job_id):
                if job_id not in coordinator.claims or job_id in coordinator.results:
                    raise RuntimeError('Cannot release ' + job_id)
                del coordinator.claims[job_id]
                coordinator.revision += 1

            def auto_void(job, destination, process, error, path):
                job_id = job['job_id']
                if (error is not None or process is None or process.get('returncode') in (0, None)
                        or process.get('timed_out') or process.get('process_group_remaining') is not False):
                    return False
                try:
                    evidence = interaction_evidence(destination)
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    return False
                index = auto_voids['by_job'].get(job_id, 0) + 1
                if index > AUTO_VOID_PER_JOB or auto_voids['count'] >= AUTO_VOID_PER_DEPLOYMENT:
                    return False
                target = void_folder / 'episodes' / f'{job_id}.void{index}'
                record_target = void_folder / 'attempts' / f'{job_id}.void{index}.json'
                for folder_path in (target.parent, record_target.parent):
                    folder_path.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(target))
                attempt = json.loads(path.read_text())
                attempt.update(status='voided_zero_interaction', process=process, finished_at=time.time(),
                               interaction_evidence=evidence, void_output=str(target))
                atomic_json(path, attempt)
                shutil.move(str(path), str(record_target))
                if path.with_suffix('.log').exists():
                    shutil.move(str(path.with_suffix('.log')), str(record_target.with_suffix('.log')))
                entry = {'job_id': job_id, 'void_index': index, 'at': time.time(),
                         'rule': 'standing zero-interaction rule approved by Sachin on 2026-10-18',
                         'returncode': process.get('returncode'), 'evidence': evidence,
                         'attempt_record': str(record_target), 'output': str(target),
                         'attempt_record_sha256': file_hash(record_target)}
                with (void_folder / 'auto-void.jsonl').open('a') as stream:
                    stream.write(json.dumps(entry, sort_keys=True) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                auto_voids['count'] += 1
                auto_voids['by_job'][job_id] = index
                manifest.setdefault('auto_voids', []).append(entry)
                attempt_paths.pop(job_id, None)
                return True

            def output_for(job):
                destination = output / 'episodes' / job['job_id']
                if destination.exists():
                    raise RuntimeError('Unclaimed output exists; refusing to replay ' + job['job_id'])
                path = folder / 'attempts' / (job['job_id'] + '.json')
                if path.exists():
                    raise RuntimeError('Attempt record already exists for ' + job['job_id'])
                atomic_json(path, {'status': 'planned', 'actor': actors[job['actor_sha256']], 'job': job,
                                  'output': str(destination), 'deployment': deployment,
                                  'source_hashes': manifest['source_hashes'],
                                  'budget_before': budget.snapshot(coordinator.report()['cost_totals'])})
                attempt_paths[job['job_id']] = path
                manifest['episode_attempts'].append(str(path))
                atomic_json(folder / 'launch.json', manifest)
                return str(destination)

            def execute(job, slot, destination):
                path = attempt_paths[job['job_id']]
                attempt = json.loads(path.read_text())
                worker_folder = output / 'workers' / slot['slot_id']
                environment = worker_environment(slot, worker_folder, config['threads_per_process'])
                command = worker_command(job, attempt['actor'], destination, slot, sys.executable,
                                         uuids[slot['gpu']])
                attempt.update(status='claimed', slot=slot, gpu_uuid=uuids[slot['gpu']], command=command,
                               environment={key: environment[key] for key in ENVIRONMENT_KEYS})
                atomic_json(path, attempt)
                timeout = min(config['episode_timeout'],
                              budget.remaining_wall_seconds() - config['cleanup_reserve'])
                if timeout <= 0:
                    raise RuntimeError('No budget remains for bounded episode cleanup')
                attempt['timeout_seconds'] = timeout

                def started(record):
                    attempt.update(status='running', process=record)
                    atomic_json(path, attempt)

                process = run_fresh_episode(command, path.with_suffix('.log'), timeout, env=environment,
                                            on_started=started)
                attempt.update(status='process_finished', process=process)
                atomic_json(path, attempt)
                return process

            def accept(job, slot, destination, process, error):
                path = attempt_paths[job['job_id']]
                if auto_void(job, destination, process, error, path):
                    return None
                attempt = json.loads(path.read_text())
                try:
                    row = read_episode(destination)
                except Exception as audit_error:  # noqa: BLE001 - retain the paid attempt
                    row = _unreadable_result(job, destination, audit_error)
                try:
                    attempt['verified_episode_sources'] = audit_episode_sources(destination, pilot)
                except Exception as source_error:  # noqa: BLE001 - invalidate, never rerun
                    row['valid'] = False
                    row.setdefault('invalid_reasons', []).append(str(source_error))
                clean = (error is None and process is not None and process.get('returncode') == 0
                         and process.get('process_group_remaining') is False
                         and not any(process.get(key) for key in
                                     ('timed_out', 'interrupted', 'error', 'cleanup_error')))
                if not clean:
                    row['valid'] = False
                    row.setdefault('invalid_reasons', []).append('Worker failed: ' + str(error or process))
                render = kit_render_gpu(attempt['environment']['OMNIGIBSON_APPDATA_PATH'],
                                        (process or {}).get('started_at') or 0)
                render['matches_route'] = (render['active_uuid_prefix'] is not None
                                           and attempt['gpu_uuid'][4:12] == render['active_uuid_prefix'])
                attempt.update(status='audited', valid=row.get('valid'), costs=row.get('costs'),
                               invalid_reasons=row.get('invalid_reasons'), outcome=row.get('outcome'),
                               kit_render_gpu=render, finished_at=time.time())
                atomic_json(path, attempt)
                return row

            last = [0.0]

            def progress(report, inflight):
                if time.time() - last[0] < 15:
                    return
                last[0] = time.time()
                manifest.update(coordinator_report=report, budget=budget.snapshot(report['cost_totals']),
                                in_flight_jobs=[entry['job']['job_id'] for entry in inflight],
                                gpu_memory=gpu_memory(), updated_at=time.time())
                atomic_json(folder / 'launch.json', manifest)

            def should_stop():
                if stop['requested'] or (folder / 'pause').exists():
                    manifest['stop_reason'] = 'pause requested'
                    return True
                try:
                    left = budget.remaining_wall_seconds(coordinator.report()['cost_totals'])
                except RuntimeError as error:
                    manifest['stop_reason'] = f'budget: {error}'
                    return True
                if left <= config['cleanup_reserve']:
                    manifest['stop_reason'] = 'diagnostic budget reached'
                    return True
                return False

            report = run_parallel(coordinator, config['routes'], state_path, output_for, execute, accept,
                                  budget, max_new_jobs=max_new_jobs, on_progress=progress,
                                  stop_requested=should_stop, timeout_reserve=config['cleanup_reserve'],
                                  release=release)
            manifest['status'] = report['status'] if report['status'] in ('complete', 'blocked') else 'paused'
    except Exception as error:  # noqa: BLE001 - persist every launcher failure in the receipt
        manifest.update(status='blocked', error=f'{type(error).__name__}: {error}')
    finally:
        if coordinator is not None:
            manifest['coordinator_report'] = coordinator.report()
            manifest['budget'] = budget.snapshot(coordinator.report()['cost_totals'])
            manifest['analysis'] = analyze(coordinator)
        manifest['finished_at'] = time.time()
        atomic_json(folder / 'launch.json', manifest)
        for number, handler in previous.items():
            signal.signal(number, handler)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/maniguard/phase2/replicate-v1.json')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/maniguard/replicate-deployment-20261018.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/maniguard/phase2/replicate-v1')
    parser.add_argument('--deployment', required=True)
    parser.add_argument('--max-new-jobs', type=int)
    args = parser.parse_args()
    if Path(args.deployment).name != args.deployment:
        raise ValueError('Deployment must be a single directory name')
    result = run(args.protocol, args.config, args.output, args.deployment, max_new_jobs=args.max_new_jobs)
    print(json.dumps({'status': result['status'], 'error': result['error'],
                      'stop_reason': result.get('stop_reason'), 'report': result.get('coordinator_report')},
                     indent=2), flush=True)
    return 0 if result['status'] in ('paused', 'complete') else 1


if __name__ == '__main__':
    raise SystemExit(main())
