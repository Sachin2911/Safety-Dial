"""Bounded service switching, reservation accounting and fresh episode processes.

Nothing executes on import. Service mutations occur only through an explicitly invoked
PolicyService.ensure_family, after checking for running manipulation episodes.
"""

import fcntl
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from helpers.maniguard_phase2_families import ROOT, load_sources
from helpers.maniguard_phase2_optim import record_hash

PRIOR_SCHEMA = 'maniguard-phase2-prior-accounting-v1'
PRIOR_COUNTS = ('policy_control_steps', 'policy_queries', 'physics_steps', 'encoder_image_calls',
                'encoder_images', 'encoder_text_calls', 'adapter_decisions', 'actor_rpc_attempts')
EPISODE_SCRIPTS = {'maniguard_episode.py', 'maniguard_phase2_episode.py'}
SERVER_SOURCES = ('experiments/scripts/maniguard_serve.py',
                  'experiments/helpers/maniguard_policy_provenance.py',
                  'experiments/helpers/maniguard_policy.py',
                  'experiments/helpers/maniguard_parallel_policy.py')


_EXECUTION_LEASES = {}
_EXECUTION_LEASE_GUARD = threading.RLock()
EXECUTION_LOCK = ROOT / 'runs/maniguard/setup/execution.lock'


@contextmanager
def execution_lease(path=EXECUTION_LOCK):
    """Serialize cooperative services/launchers, reentrantly within one process.

    Hold this across checkpoint acquisition and the entire episode. Direct legacy/manual
    processes do not honor this lease; idle scans remain required, and operators must not
    start those uncooperative entry points concurrently with a managed phase-two run.
    """
    path = Path(path).resolve()
    key, pid = str(path), os.getpid()
    with _EXECUTION_LEASE_GUARD:
        entry = _EXECUTION_LEASES.get(key)
        if entry is not None and entry['pid'] == pid:
            entry['depth'] += 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open('a+')
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                stream.close()
                raise RuntimeError('Another process owns the manipulation execution lease') from error
            try:
                stream.seek(0)
                stream.truncate()
                json.dump({'pid': pid, 'acquired_at': time.time()}, stream)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                stream.close()
                raise
            entry = {'pid': pid, 'depth': 1, 'stream': stream}
            _EXECUTION_LEASES[key] = entry
    try:
        yield path
    finally:
        with _EXECUTION_LEASE_GUARD:
            entry['depth'] -= 1
            if entry['depth'] == 0:
                fcntl.flock(entry['stream'], fcntl.LOCK_UN)
                entry['stream'].close()
                del _EXECUTION_LEASES[key]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def active_episode_processes(proc_root=Path('/proc')):
    """Read command-line identities only; never inspect process environment secrets."""
    found = []
    for directory in Path(proc_root).iterdir():
        if not directory.name.isdigit():
            continue
        try:
            raw = (directory / 'cmdline').read_bytes()
        except FileNotFoundError:
            continue  # Process ended between enumeration and read.
        except PermissionError as error:
            raise RuntimeError('Cannot establish that manipulation episodes are idle') from error
        arguments = [value.decode(errors='replace') for value in raw.split(b'\0') if value]
        matches = [value for value in arguments if Path(value).name in EPISODE_SCRIPTS]
        if matches:
            found.append({'pid': int(directory.name), 'episode_script': matches[0]})
    return found


def probe_policy_metadata(port, timeout, *, python=sys.executable, root=ROOT,
                          runner=subprocess.run):
    """Probe in a bounded subprocess because a missing WebSocket hello can block recv."""
    code = (
        'import json,sys; sys.path.insert(0,sys.argv[1]); '
        'from helpers.maniguard_client import StudyClient; '
        'client=StudyClient(host="127.0.0.1",port=int(sys.argv[2])); '
        'print(json.dumps(client.get_server_metadata(),sort_keys=True)); client._ws.close()'
    )
    completed = runner([str(python), '-c', code, str(Path(root) / 'experiments'), str(port)],
                       capture_output=True, text=True, timeout=timeout, check=False)
    if completed.returncode:
        raise RuntimeError('Policy handshake failed: ' + completed.stderr[-1000:])
    return json.loads(completed.stdout.strip().splitlines()[-1])


def render_policy_config(family, port=18000, root=ROOT, *, log_path=None):
    if family not in load_sources()['policies'] or type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('Unknown policy family or invalid loopback port')
    root = Path(root).resolve()
    if any(c in str(root) for c in '\n\r'):
        raise ValueError('Invalid service root')
    command = shlex.join([str(root / 'scripts/maniguard/serve.sh'), '--family', family,
                          '--phase2', '--parallel-clients', '--port', str(port)])
    if log_path is not None:
        log_path = Path(log_path).resolve()
        if any(c in str(log_path) for c in '\n\r'):
            raise ValueError('Invalid policy log path')
        # Keep platform stdout and an attempt-specific transcript. pipefail preserves
        # a failed policy exit when tee itself exits successfully.
        pipeline = command + ' 2>&1 | tee -- ' + shlex.quote(str(log_path))
        command = shlex.join(['/bin/bash', '-o', 'pipefail', '-c', pipeline])
    command = command.replace('%', '%%')  # Supervisor percent interpolation.
    return ('[program:maniguard_policy]\n' + f'command={command}\ndirectory={root}\n'
            'autostart=false\nautorestart=false\nstartsecs=5\nstopwaitsecs=30\n'
            'stopasgroup=true\nkillasgroup=true\nredirect_stderr=true\n'
            'stdout_logfile=/dev/stdout\nstdout_logfile_maxbytes=0\n')


class ServiceFailure(RuntimeError):
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


class PolicyService:
    """Reuse only an exact handshake match; otherwise switch the one managed policy."""

    def __init__(self, *, root=ROOT, port=18000,
                 config_path='/etc/supervisor/conf.d/maniguard_policy.conf',
                 runner=subprocess.run, probe=None, scan=active_episode_processes,
                 monotonic=time.monotonic, sleep=time.sleep, wall=time.time,
                 lease_path=EXECUTION_LOCK):
        self.root = Path(root)
        self.port = port
        self.config_path = Path(config_path)
        self.runner = runner
        self.probe = probe or (lambda timeout: probe_policy_metadata(
            port, timeout, root=self.root, runner=self.runner))
        self.scan = scan
        self.monotonic = monotonic
        self.sleep = sleep
        self.wall = wall
        self.lease_path = lease_path

    def expected_metadata(self, family):
        spec = load_sources(self.root / 'configs/maniguard/phase2/sources.json')['policies'][family]
        return {'serve_config': spec['config'], 'checkpoint_revision': spec['revision'],
                'checkpoint': str((self.root / 'data/maniguard/policies' / family /
                                   spec['step']).resolve()),
                'episode_seed_reset': 'study_episode_id', 'rng_isolation': 'per_episode',
                'server_source_hashes': {name: file_hash(self.root / name) for name in SERVER_SOURCES}}

    def ensure_family(self, family, audit_dir, *, startup_timeout=600):
        with execution_lease(self.lease_path):
            return self._ensure_family(family, audit_dir, startup_timeout=startup_timeout)

    def _ensure_family(self, family, audit_dir, *, startup_timeout=600):
        if not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError('Policy startup needs a finite positive timeout')
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=False)
        started = self.monotonic()
        deadline = started + startup_timeout
        record = {'family': family, 'port': self.port, 'started_at': self.wall(),
                  'status': 'checking', 'service_changed': False, 'commands': [],
                  'expected': self.expected_metadata(family), 'metadata': None,
                  'probe_errors': [], 'source_sha256': file_hash(__file__)}
        path = audit_dir / 'service.json'

        def persist():
            record['elapsed_seconds'] = self.monotonic() - started
            record['finished_at'] = self.wall()
            atomic_json(path, record)

        def remaining():
            left = deadline - self.monotonic()
            if left <= 0:
                raise TimeoutError('Managed policy startup deadline expired')
            return left

        def probe():
            try:
                metadata = self.probe(min(30.0, remaining()))
                record['metadata'] = metadata
                return metadata
            except (OSError, RuntimeError, ValueError, TimeoutError, subprocess.TimeoutExpired) as error:
                record['probe_errors'].append(str(error))
                return None

        def matches(metadata):
            return isinstance(metadata, dict) and all(
                metadata.get(key) == value for key, value in record['expected'].items())

        def supervisor(*args, allow_stopped=False):
            command = ['supervisorctl', *args]
            completed = self.runner(command, capture_output=True, text=True,
                                    timeout=min(30.0, remaining()), check=False)
            response = {'command': command, 'returncode': completed.returncode,
                        'stdout': completed.stdout, 'stderr': completed.stderr}
            record['commands'].append(response)
            persist()
            benign = allow_stopped and any(message in (completed.stdout + completed.stderr).lower()
                                          for message in ('not running', 'no such process'))
            if completed.returncode and not benign:
                raise RuntimeError('Supervisor operation failed: ' + ' '.join(args))

        persist()
        try:
            if matches(probe()):
                record['status'] = 'reused'
                persist()
                return record
            active = self.scan()
            if active:
                record['active_episodes'] = active
                raise RuntimeError('Refusing policy checkpoint change while manipulation episodes run')
            if self.config_path.exists():
                old = self.config_path.read_bytes()
                (audit_dir / 'previous-policy.conf').write_bytes(old)
                record['previous_config_sha256'] = hashlib.sha256(old).hexdigest()
            supervisor('stop', 'maniguard_policy', allow_stopped=True)
            # Check again immediately before replacing the configuration.
            if self.scan():
                raise RuntimeError('An episode appeared during the managed policy switch')
            policy_log = audit_dir.resolve() / 'policy.log'
            if policy_log.exists():
                raise ValueError('New policy startup log already exists')
            config = render_policy_config(family, self.port, self.root, log_path=policy_log)
            record['policy_log'] = str(policy_log)
            self.config_path.write_text(config)
            (audit_dir / 'requested-policy.conf').write_text(config)
            record['requested_config_sha256'] = hashlib.sha256(config.encode()).hexdigest()
            record['service_changed'] = True
            persist()
            supervisor('reread')
            supervisor('update', 'maniguard_policy')
            supervisor('start', 'maniguard_policy')
            while remaining() > 0:
                if matches(probe()):
                    record['status'] = 'started'
                    persist()
                    return record
                self.sleep(min(2.0, remaining()))
        except (OSError, RuntimeError, ValueError, TimeoutError, subprocess.TimeoutExpired,
                KeyboardInterrupt, SystemExit) as error:
            record['status'] = 'failed'
            record['error'] = f'{type(error).__name__}: {error}'
            persist()
            raise ServiceFailure(record['error'], record) from error
        raise AssertionError('Unreachable managed-service state')


class ReservationBudget:
    """Count a continuous GPU reservation once, including coding, idle and restart gaps."""

    def __init__(self, prior, envelope, *, reserved_gpus=1, wall=time.time, verify_sources=True):
        self.wall = wall
        self.prior = dict(prior)
        self.envelope = envelope
        self.reserved_gpus = reserved_gpus
        body = {key: value for key, value in prior.items() if key != 'sha256'}
        if prior.get('schema') != PRIOR_SCHEMA or prior.get('sha256') != record_hash(body):
            raise ValueError('Prior accounting schema/hash mismatch')
        if type(reserved_gpus) is not int or reserved_gpus < 1 or prior['reserved_gpus'] != reserved_gpus:
            raise ValueError('Reservation GPU count must be positive and match prior accounting')
        fields = ('timestamp', 'reservation_started_at', 'wall_seconds', 'allocated_gpu_seconds')
        if any(type(prior.get(key)) not in (float, int) or not math.isfinite(prior[key])
               for key in fields):
            raise ValueError('Prior reservation clocks must be finite')
        duration = prior['timestamp'] - prior['reservation_started_at']
        if (duration < 0 or prior['timestamp'] > self.wall() + 1 or
                abs(prior['wall_seconds'] - duration) > 1 or
                abs(prior['allocated_gpu_seconds'] - reserved_gpus * duration) > reserved_gpus):
            raise ValueError('Prior reservation interval is inconsistent or in the future')
        if set(prior['counts']) != set(PRIOR_COUNTS) or any(
                type(value) is not int or value < 0 for value in prior['counts'].values()):
            raise ValueError('Prior accounting needs every nonnegative attempted-work counter')
        if not prior.get('sources'):
            raise ValueError('Prior accounting requires hashed evidence sources')
        if verify_sources:
            for source in prior['sources']:
                if file_hash(source['path']) != source['sha256']:
                    raise ValueError('Prior accounting evidence source changed')

    def snapshot(self, measured_costs=None):
        now = self.wall()
        if now < self.prior['timestamp']:
            raise ValueError('Wall clock moved before the prior accounting timestamp')
        extra_wall = now - self.prior['timestamp']
        counts = dict(self.prior['counts'])
        for key, value in (measured_costs or {}).items():
            if key in counts:
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError('Measured work counter is invalid')
                counts[key] += value
        return {'timestamp': now, 'wall_seconds': self.prior['wall_seconds'] + extra_wall,
                'allocated_gpu_seconds': self.prior['allocated_gpu_seconds'] +
                                         self.reserved_gpus * extra_wall,
                'counts': counts, 'reserved_gpus': self.reserved_gpus}

    def check(self, measured_costs=None, *, next_actions=0, next_queries=0):
        snapshot = self.snapshot(measured_costs)
        envelope = self.envelope
        if snapshot['allocated_gpu_seconds'] >= envelope['gpu_hours_ceiling'] * 3600:
            raise RuntimeError('Allocated GPU-hour ceiling reached')
        if snapshot['counts']['policy_control_steps'] + next_actions > envelope['policy_actions_max']:
            raise RuntimeError('Policy-action ceiling cannot reserve another full episode')
        if snapshot['counts']['policy_queries'] + next_queries > envelope[
                'exhaustion_safe_successful_queries_max']:
            raise RuntimeError('Policy-query ceiling cannot reserve another full episode')
        return snapshot

    def remaining_wall_seconds(self, measured_costs=None):
        snapshot = self.check(measured_costs)
        return (self.envelope['gpu_hours_ceiling'] * 3600 -
                snapshot['allocated_gpu_seconds']) / self.reserved_gpus


def process_group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def run_fresh_episode(command, log_path, timeout, *, env=None, popen=subprocess.Popen,
                      killpg=os.killpg, group_alive=process_group_alive, wall=time.time,
                      monotonic=time.monotonic, sleep=time.sleep, termination_grace=5,
                      on_started=None):
    """Run one process group, reaping descendants even when their leader exits first."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Episode timeout must be finite and positive')
    if not math.isfinite(termination_grace) or termination_grace <= 0:
        raise ValueError('Termination grace must be finite and positive')
    started = monotonic()
    record = {'command': list(command), 'started_at': wall(), 'pid': None,
              'returncode': None, 'timed_out': False, 'interrupted': False}
    process = None

    def wait_group_exit():
        deadline = monotonic() + termination_grace
        while group_alive(process.pid):
            process.poll()  # Reap the leader; its exit alone does not prove group exit.
            left = deadline - monotonic()
            if left <= 0:
                return False
            sleep(min(0.1, left))
        return True

    try:
        with Path(log_path).open('x') as log:
            process = popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env=env, start_new_session=True)
            record['pid'] = process.pid
            if on_started is not None:
                on_started(dict(record))
            try:
                record['returncode'] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                record['timed_out'] = True
    except (OSError, RuntimeError, ValueError, KeyboardInterrupt, SystemExit) as error:
        record['error'] = f'{type(error).__name__}: {error}'
        record['interrupted'] = isinstance(error, (KeyboardInterrupt, SystemExit))
    finally:
        try:
            if process is not None:
                leader_finished = process.poll() is not None
                if group_alive(process.pid):
                    if leader_finished:
                        record['orphaned_descendants'] = True
                        record.setdefault('error', 'Episode leader exited while descendants remained')
                    for termination_signal in (signal.SIGTERM, signal.SIGKILL):
                        try:
                            killpg(process.pid, termination_signal)
                        except ProcessLookupError:
                            pass
                        if wait_group_exit():
                            break
                    record['process_group_remaining'] = group_alive(process.pid)
                    if record['process_group_remaining']:
                        record['error'] = 'Episode process group did not exit after SIGKILL'
                else:
                    record['process_group_remaining'] = False
                final_code = process.poll()
                if final_code is not None:
                    record['returncode'] = final_code
        except (OSError, RuntimeError, KeyboardInterrupt, SystemExit) as error:
            record['cleanup_error'] = f'{type(error).__name__}: {error}'
            record['error'] = 'Episode process-group exit could not be established'
            record['process_group_remaining'] = None
        finally:
            record['finished_at'] = wall()
            record['process_elapsed_seconds'] = monotonic() - started
    return record
