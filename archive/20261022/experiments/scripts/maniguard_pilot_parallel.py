"""Resume a frozen pilot using fixed GPU policy routes and one durable parent writer.

Run as a managed supervisor job from the simulator interpreter. The coordinator,
episode runner and result audits are the frozen phase-two sources; this entry point
only routes already-claimed jobs to preloaded per-GPU family policy servers. Touch
``<deployment>/pause`` (or send SIGTERM) to stop launching and drain in-flight work.

With ``"dispatch": "lanes"`` the frozen coordinator state is taken over once by
LaneCoordinator (``lanes.json``), which runs independent optimizer lanes together.
The serial file is renamed so nothing can resume it, and a frozen-format
``coordinator.json`` is written again only when every block is complete.
"""

import argparse
import fcntl
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'experiments/scripts'))

from maniguard_phase2_train import (
    _save_actor,
    _unreadable_result,
    audit_episode_sources,
    episode_command,
)
from maniguard_pilot_void import interaction_evidence

from helpers.maniguard_phase2_coordinator import PilotCoordinator
from helpers.maniguard_phase2_protocol import load_protocol, validate_protocol
from helpers.maniguard_phase2_results import read_episode
from helpers.maniguard_phase2_services import (
    PolicyService,
    ReservationBudget,
    active_episode_processes,
    atomic_json,
    execution_lease,
    file_hash,
    probe_policy_metadata,
    run_fresh_episode,
)
from helpers.maniguard_pilot_lanes import LaneCoordinator, lane_key
from helpers.maniguard_pilot_parallel import AllocationClock, run_parallel, validate_routes

SCHEMA = 'maniguard-parallel-launch-v1'
SOURCES = (
    'experiments/scripts/maniguard_pilot_parallel.py',
    'experiments/scripts/maniguard_pilot_worker.py',
    'experiments/helpers/maniguard_pilot_parallel.py',
    'experiments/helpers/maniguard_pilot_lanes.py',
    'experiments/scripts/maniguard_pilot_void.py',
)
# Standing rule approved by Sachin on 2026-10-18: an attempt failing before simulator
# initialization and before any policy query may be voided and rerun with disclosure.
AUTO_VOID_PER_JOB = 2
AUTO_VOID_PER_DEPLOYMENT = 8
APPDATA_SEED = ROOT / 'third_party/ManiGuard/behavior-1k/OmniGibson/appdata'
ENVIRONMENT_KEYS = ('CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES', 'OMNIGIBSON_APPDATA_PATH',
                    'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
KIT_GPU_ROW = re.compile(r'^\|\s*(\d+)\s*\|[^|]*\|\s*(Yes: \d+)?\s*\|')
KIT_UUID_ROW = re.compile(r'^\|\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*[0-9a-f]+\s*\|\s*([0-9a-f]{8})\.\.')


def worker_environment(slot, folder, threads, base=None):
    """One visible GPU per simulator, like the frozen single-GPU host."""
    environment = dict(os.environ if base is None else base)
    environment.pop('OMNIGIBSON_GPU_ID', None)
    environment.update(OMNI_KIT_ALLOW_ROOT='1', OMNI_KIT_ACCEPT_EULA='YES',
                       CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(slot['gpu']),
                       OMNIGIBSON_APPDATA_PATH=str(folder / 'appdata'),
                       XDG_CACHE_HOME=str(folder / 'cache'),
                       XDG_CONFIG_HOME=str(folder / 'config'),
                       OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                       MKL_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads))
    return environment


def worker_command(job, actor, output, slot, python, uuid):
    original = episode_command(job, actor, output, python=python, port=slot['port'])
    if original[2] != str(ROOT / 'experiments/scripts/maniguard_phase2_episode.py'):
        raise ValueError('Unexpected frozen episode command')
    return [str(python), '-u', str(ROOT / SOURCES[1]), '--gpu', str(slot['gpu']),
            '--gpu-uuid', uuid, *original[2:]]


def kit_render_gpu(appdata, started_at):
    """Return the UUID prefix Kit marked active in the newest log written by this attempt."""
    logs = [path for path in Path(appdata).glob('local/logs/Kit/*/*/kit_*.log')
            if path.stat().st_mtime >= started_at]
    if not logs:
        return {'log': None, 'active_uuid_prefix': None}
    log = max(logs, key=lambda path: path.stat().st_mtime)
    active, waiting = None, False
    for line in log.read_text(errors='replace').splitlines():
        row = KIT_GPU_ROW.match(line)
        if row:
            waiting = bool(row.group(2))
            continue
        uuid = KIT_UUID_ROW.match(line)
        if uuid and waiting:
            active, waiting = uuid.group(1), False
    physx = sorted(set(re.findall(r'Using CUDA device ordinal (\d+)', log.read_text(errors='replace'))))
    return {'log': str(log), 'active_uuid_prefix': active, 'physx_cuda_ordinals': physx}


class AllowlistedLanes(LaneCoordinator):
    """Dispatch-only lane restriction; frozen state, hashes and updates are untouched."""

    allowlist = None

    @classmethod
    def load(cls, path, *, allowlist=None, **kwargs):
        lanes = super().load(path, **kwargs)
        lanes.set_allowlist(allowlist)
        return lanes

    def __init__(self, base, *, allowlist=None, **kwargs):
        super().__init__(base, **kwargs)
        self.set_allowlist(allowlist)

    def set_allowlist(self, allowlist):
        if allowlist is not None and not set(allowlist) <= set(self.lanes):
            raise ValueError('Lane allowlist must name known lanes')
        self.allowlist = None if allowlist is None else set(allowlist)

    def next_jobs(self):
        jobs = super().next_jobs()
        if self.allowlist is None:
            return jobs
        return [job for job in jobs
                if lane_key(self.base.stage_plan[self._active_stage_of(job['job_id'])]) in self.allowlist]

    def report(self):
        report = super().report()
        report['lane_allowlist'] = None if self.allowlist is None else sorted(self.allowlist)
        return report


def take_over_lanes(output, priority, allowlist=None):
    """Move the frozen serial state into lanes exactly once; never leave both resumable."""
    serial, lanes_path = output / 'coordinator.json', output / 'lanes.json'
    moved = output / 'coordinator-serial-takeover.json'
    if lanes_path.exists():
        coordinator = AllowlistedLanes.load(lanes_path, priority=priority, allowlist=allowlist)
        if serial.exists():
            if file_hash(serial) != coordinator.takeover['file_sha256'] or moved.exists():
                raise RuntimeError('Serial and lane coordinator states disagree; reconcile first')
            os.replace(serial, moved)
        return coordinator
    if moved.exists():
        raise RuntimeError('A previous serial takeover exists without lane state')
    base = PilotCoordinator.load(serial)
    takeover = {'serial_file': moved.name, 'file_sha256': file_hash(serial),
                'content_sha256': base._last_saved_sha256, 'taken_over_at': time.time(),
                'report': base.report()}
    coordinator = AllowlistedLanes(base, takeover=takeover, priority=priority, allowlist=allowlist)
    coordinator.save(lanes_path)
    os.replace(serial, moved)
    return coordinator


def load_dispatch(config, output):
    if config.get('dispatch', 'serial') == 'lanes':
        return (take_over_lanes(output, config.get('lane_priority'), config.get('lane_allowlist')),
                output / 'lanes.json')
    if config.get('dispatch', 'serial') != 'serial':
        raise ValueError('Dispatch must be serial or lanes')
    return PilotCoordinator.load(output / 'coordinator.json'), output / 'coordinator.json'


def prepare_worker_folder(folder, seed=APPDATA_SEED):
    """Give each worker private Kit appdata, seeded once from the cache-only default tree."""
    if folder.exists():
        return {'seeded': False}
    folder.mkdir(parents=True)
    if seed.is_dir():
        shutil.copytree(seed, folder / 'appdata', symlinks=True)
    return {'seeded': seed.is_dir(), 'seed': str(seed)}


def gpu_inventory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total',
         '--format=csv,noheader'], check=True, capture_output=True, text=True, timeout=15)
    return result.stdout.strip().splitlines()


def gpu_memory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'],
        check=True, capture_output=True, text=True, timeout=15)
    return result.stdout.strip().splitlines()


def gpu_uuids():
    return {int(line.split(',')[0]): line.split(',')[1].strip() for line in gpu_inventory()}


def verify_routes(routes, probe=probe_policy_metadata):
    records = []
    for route in routes:
        expected = PolicyService(port=route['port']).expected_metadata(route['family'])
        actual = probe(route['port'], 30)
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError('Policy handshake differs from frozen family route')
        if actual.get('rng_isolation') != 'per_episode':
            raise ValueError('Concurrent clients require per-episode policy RNG isolation')
        records.append({'route': route, 'expected': expected, 'metadata': actual})
    return records


def load_prior(protocol):
    pin = protocol['freeze_provenance']['prior_accounting']
    path = ROOT / pin['evidence_path']
    if file_hash(path) != pin['sha256']:
        raise ValueError('Prior accounting differs from its frozen pin')
    prior = json.loads(path.read_text())
    ReservationBudget(prior, protocol['resource_envelope'])
    return prior


def check(config_path, output):
    """Validate every launch input without claiming work or touching the run directory."""
    config = json.loads(Path(config_path).read_text())
    protocol = load_protocol()
    validate_protocol(protocol, require_frozen=True)
    slots = validate_routes(config['routes'], protocol['families'])
    budget = AllocationClock(load_prior(protocol), config['allocations'],
                             protocol['resource_envelope'])
    output = Path(output)
    if config.get('dispatch') == 'lanes' and (output / 'lanes.json').exists():
        coordinator = AllowlistedLanes.load(output / 'lanes.json', priority=config.get('lane_priority'),
                                            allowlist=config.get('lane_allowlist'))
    else:
        coordinator = PilotCoordinator.load(output / 'coordinator.json')
    report = coordinator.report()
    return {'protocol_sha256': protocol['sha256'], 'coordinator_sha256': coordinator._last_saved_sha256,
            'slots': [slot['slot_id'] for slot in slots], 'report': report,
            'pending': len(coordinator.next_jobs()) if report['status'] == 'training' else None,
            'reservation': budget.check(report['cost_totals']),
            'remaining_wall_seconds': budget.remaining_wall_seconds(report['cost_totals']),
            'routes': verify_routes(config['routes']), 'active_episodes': active_episode_processes()}


def run(config_path, output, deployment, *, max_new_jobs=None):
    config = json.loads(Path(config_path).read_text())
    protocol = load_protocol()
    validate_protocol(protocol, require_frozen=True)
    validate_routes(config['routes'], protocol['families'])
    if any(route['slots'] != 1 for route in config['routes']) and not config.get(
            'two_slot_qualification'):
        raise ValueError('Two simulators per GPU require recorded memory/throughput qualification')
    prior = load_prior(protocol)
    budget = AllocationClock(prior, config['allocations'], protocol['resource_envelope'])
    budget.check()
    output = Path(output).resolve()
    resumable = output / ('lanes.json' if (output / 'lanes.json').exists() else 'coordinator.json')
    if not resumable.is_file():
        raise ValueError('Parallel launcher only resumes an existing coordinator')
    folder = output / 'deployments' / deployment
    folder.mkdir(parents=True, exist_ok=False)
    uuids = gpu_uuids()
    stop = {'requested': False}

    def request_stop(signum, frame):
        stop['requested'] = True

    previous = {number: signal.signal(number, request_stop)
                for number in (signal.SIGTERM, signal.SIGINT)}
    manifest = {'schema': SCHEMA, 'status': 'checking', 'started_at': time.time(),
                'config': config, 'config_path': str(Path(config_path).resolve()),
                'config_sha256': file_hash(config_path), 'protocol_sha256': protocol['sha256'],
                'prior_sha256': prior['sha256'], 'resume_file': str(resumable),
                'resume_file_sha256': file_hash(resumable),
                'source_hashes': {name: file_hash(ROOT / name) for name in SOURCES},
                'repository_commit': subprocess.run(
                    ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True,
                    text=True, check=True).stdout.strip(),
                'host': platform.uname()._asdict(), 'instance_id': config['instance_id'],
                'gpu_inventory': gpu_inventory(), 'python': sys.version,
                'episode_attempts': [], 'error': None}
    atomic_json(folder / 'launch.json', manifest)
    for name in SOURCES:
        destination = folder / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    (folder / 'source' / 'deployment.json').write_bytes(Path(config_path).read_bytes())
    coordinator = None
    try:
        with (output / 'launcher.lock').open('a') as lock, execution_lease():
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if active_episode_processes():
                raise RuntimeError('An existing episode must be reconciled before takeover')
            coordinator, state_path = load_dispatch(config, output)
            manifest.update(resume_content_sha256=coordinator._last_saved_sha256,
                            state_path=str(state_path), dispatch=config.get('dispatch', 'serial'),
                            lane_priority=config.get('lane_priority'),
                            lane_allowlist=config.get('lane_allowlist'),
                            takeover=getattr(coordinator, 'takeover', None))
            if coordinator.report()['in_flight_claims']:
                raise RuntimeError('Existing claims require reconciliation; never replay them')
            manifest['policy_routes'] = verify_routes(config['routes'])
            # One persistent private folder per GPU slot, shared by later deployments,
            # so the 1.3 GB Kit cache seed is copied once rather than per launch.
            manifest['worker_folders'] = {
                slot['slot_id']: prepare_worker_folder(output / 'workers' / slot['slot_id'])
                for slot in validate_routes(config['routes'], protocol['families'])}
            manifest['gpu_memory_before'] = gpu_memory()
            manifest['initial_report'] = coordinator.report()
            manifest['status'] = 'running'
            atomic_json(folder / 'launch.json', manifest)
            attempt_paths = {}
            auto_voids = {'count': 0, 'by_job': {}}
            void_folder = output / 'void-attempts' / deployment

            def release(job_id):
                """Free a voided claim with no recorded result, so the identical job reruns."""
                base = getattr(coordinator, 'base', coordinator)
                if job_id not in base.claims:
                    raise RuntimeError('Cannot release an unclaimed job: ' + job_id)
                results = ([record['active_results'] for record in coordinator.stages.values()
                            if record['status'] == 'active'] if isinstance(coordinator, LaneCoordinator)
                           else [coordinator.active_results or {}])
                if any(job_id in result for result in results):
                    raise RuntimeError('Cannot release a job with a recorded result: ' + job_id)
                del base.claims[job_id]
                base.revision += 1

            def auto_void(job, destination, process, error, path):
                """Void a proven zero-interaction failure under the standing rule, within caps."""
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
                actor = _save_actor(output, job, protocol['sha256'])
                path = folder / 'attempts' / (job['job_id'] + '.json')
                if path.exists():
                    raise RuntimeError('Attempt record already exists for ' + job['job_id'])
                atomic_json(path, {'status': 'planned', 'actor': str(actor),
                                  'job': {key: value for key, value in job.items() if key != 'genome'},
                                  'output': str(destination), 'deployment': deployment,
                                  'source_hashes': manifest['source_hashes'],
                                  'reservation_before': budget.snapshot(
                                      coordinator.report()['cost_totals'])})
                attempt_paths[job['job_id']] = path
                manifest['episode_attempts'].append(str(path))
                atomic_json(folder / 'launch.json', manifest)
                return str(destination)

            def execute(job, slot, destination):
                path = attempt_paths[job['job_id']]
                attempt = json.loads(path.read_text())
                worker_folder = output / 'workers' / slot['slot_id']
                seeded = prepare_worker_folder(worker_folder)
                environment = worker_environment(slot, worker_folder, config['threads_per_process'])
                command = worker_command(job, attempt['actor'], destination, slot, sys.executable,
                                         uuids[slot['gpu']])
                attempt.update(status='claimed', slot=slot, gpu_uuid=uuids[slot['gpu']],
                               command=command, worker_folder=str(worker_folder),
                               worker_folder_seed=seeded,
                               environment={key: environment[key] for key in ENVIRONMENT_KEYS})
                atomic_json(path, attempt)
                validate_protocol(protocol, require_frozen=True)
                remaining = min(budget.remaining_wall_seconds(), config['deadline'] - time.time())
                timeout = min(config['episode_timeout'], remaining - config['cleanup_reserve'])
                if timeout <= 0:
                    raise RuntimeError('No budget remains for bounded episode cleanup')
                attempt['timeout_seconds'] = timeout

                def started(record):
                    attempt.update(status='running', process=record)
                    atomic_json(path, attempt)

                process = run_fresh_episode(command, path.with_suffix('.log'), timeout,
                                            env=environment, on_started=started)
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
                    attempt['verified_episode_sources'] = audit_episode_sources(destination, protocol)
                except Exception as source_error:  # noqa: BLE001 - invalidate, never rerun
                    row['valid'], row['ppo_eligible'] = False, False
                    row.setdefault('invalid_reasons', []).append(str(source_error))
                clean = (error is None and process is not None and process.get('returncode') == 0
                         and process.get('process_group_remaining') is False
                         and not any(process.get(key) for key in
                                     ('timed_out', 'interrupted', 'error', 'cleanup_error')))
                if not clean:
                    row['valid'], row['ppo_eligible'] = False, False
                    row.setdefault('invalid_reasons', []).append(
                        'Worker failed: ' + str(error or process))
                render = kit_render_gpu(attempt['environment']['OMNIGIBSON_APPDATA_PATH'],
                                        (process or {}).get('started_at') or 0)
                render['matches_route'] = (render['active_uuid_prefix'] is not None and
                                           attempt['gpu_uuid'][4:12] == render['active_uuid_prefix'])
                attempt['kit_render_gpu'] = render
                if render['active_uuid_prefix'] is not None and not render['matches_route']:
                    manifest.setdefault('render_route_mismatches', []).append(job['job_id'])
                attempt.update(status='audited', valid=row.get('valid'), costs=row.get('costs'),
                               cost_issues=row.get('cost_issues'),
                               invalid_reasons=row.get('invalid_reasons'), outcome=row.get('outcome'),
                               finished_at=time.time())
                atomic_json(path, attempt)
                return row

            last_progress = [0.0]

            def progress(report, inflight):
                if time.time() - last_progress[0] < 15:
                    return
                last_progress[0] = time.time()
                manifest.update(coordinator_report=report,
                                reservation=budget.snapshot(report['cost_totals']),
                                in_flight_jobs=[entry['job']['job_id'] for entry in inflight],
                                gpu_memory=gpu_memory(), updated_at=time.time())
                atomic_json(folder / 'launch.json', manifest)

            def should_stop():
                if stop['requested'] or (folder / 'pause').exists():
                    manifest['stop_reason'] = 'pause requested'
                    return True
                left = config['deadline'] - time.time()
                if left <= config['cleanup_reserve']:
                    manifest['stop_reason'] = 'operational deadline reached'
                    return True
                try:
                    left = min(left, budget.remaining_wall_seconds(coordinator.report()['cost_totals']))
                except RuntimeError as error:
                    manifest['stop_reason'] = f'resource ceiling: {error}'
                    return True
                if left <= config['cleanup_reserve']:
                    manifest['stop_reason'] = 'aggregate GPU-hour ceiling reached'
                    return True
                return False

            report = run_parallel(coordinator, config['routes'], state_path, output_for, execute,
                                  accept, budget, max_new_jobs=max_new_jobs, on_progress=progress,
                                  stop_requested=should_stop, timeout_reserve=config['cleanup_reserve'],
                                  release=release)
            manifest['status'] = report['status'] if report['status'] in ('complete', 'blocked') else 'paused'
            if isinstance(coordinator, LaneCoordinator) and report['status'] == 'complete':
                canonical = output / 'coordinator.json'
                if canonical.exists():
                    raise RuntimeError('Refusing to overwrite an existing coordinator.json')
                atomic_json(canonical, coordinator.canonical_state())
                manifest['canonical_coordinator'] = {
                    'path': str(canonical), 'file_sha256': file_hash(canonical),
                    'content_sha256': PilotCoordinator.load(canonical)._last_saved_sha256}
    except Exception as error:  # noqa: BLE001 - persist every launcher failure in the receipt
        manifest.update(status='blocked', error=f'{type(error).__name__}: {error}')
    finally:
        if coordinator is not None:
            manifest['coordinator_report'] = coordinator.report()
            manifest['reservation'] = budget.snapshot(coordinator.report()['cost_totals'])
        manifest['finished_at'] = time.time()
        atomic_json(folder / 'launch.json', manifest)
        for number, handler in previous.items():
            signal.signal(number, handler)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/maniguard/phase2/pilot-v1')
    parser.add_argument('--deployment')
    parser.add_argument('--max-new-jobs', type=int)
    parser.add_argument('--check', action='store_true', help='Validate inputs without launching')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check(args.config, args.output), indent=2, default=str), flush=True)
        return 0
    if not args.deployment or Path(args.deployment).name != args.deployment:
        raise ValueError('Deployment must be a single directory name')
    result = run(args.config, args.output, args.deployment, max_new_jobs=args.max_new_jobs)
    print(json.dumps({'status': result['status'], 'error': result['error'],
                      'stop_reason': result.get('stop_reason'),
                      'coordinator': result.get('coordinator_report')}, indent=2), flush=True)
    return 0 if result['status'] in ('paused', 'complete') else 1


if __name__ == '__main__':
    raise SystemExit(main())
