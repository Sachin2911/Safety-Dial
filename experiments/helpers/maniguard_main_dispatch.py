"""Synthetic-only main dispatch/claim/accounting slice; no process or service backend.

``MainDispatch.open(...)`` holds a canonical study-root lock until ``close``. Every
claim is persisted before an explicitly injected synthetic runner/reader is called.
``reserve_extra`` / ``settle_extra`` account for startup or failed work outside a
job's slot; callers must never include that same work in the job's ``costs``. Missing
counts remain unknown and block further work. There is deliberately no reconciliation,
retry, physical audit, per-operation runtime meter, or live execution authorization.
A persistence failure poisons the current handle. Best-effort failure snapshots are
evidence only; their presence blocks work on an explicit reopen pending review.
"""

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
import time
from copy import deepcopy
from pathlib import Path

from helpers import maniguard_main_protocol as main_protocol
from helpers.maniguard_phase2_optim import record_hash

SCHEMA = 'maniguard-main-dispatch-synthetic-v1'
PRIOR_SCHEMA = 'maniguard-main-dispatch-prior-v1'
WORK_COUNTS = main_protocol.WORK_COUNTS
BUDGET_KEYS = main_protocol.BUDGET_KEYS
SUPPORTED_PROFILES = {'phase2_learned', 'phase2_identity', 'phase2_fixed', 'phase2_contextual'}


class DispatchPersistenceError(RuntimeError):
    """State durability is uncertain; this handle must never dispatch more work."""


def _signed(value):
    return {**value, 'sha256': record_hash({key: item for key, item in value.items() if key != 'sha256'})}


def _check_signed(value):
    if value.get('sha256') != _signed(value)['sha256']:
        raise ValueError('Content hash mismatch')


def _counts(value, *, complete=True):
    if not isinstance(value, dict) or set(value) - set(WORK_COUNTS):
        raise ValueError('Work counts must use only the eleven declared counters')
    if complete and set(value) != set(WORK_COUNTS):
        raise ValueError('All eleven work counters must be explicit, including zeros')
    if any(type(number) is not int or number < 0 for number in value.values()):
        raise ValueError('Work counters must be nonnegative integers')
    return dict(value)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _atomic(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _evidence(value):
    """Keep malformed evidence serializable without inventing numerical values."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {'nonfinite_number': repr(value)}
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _evidence(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_evidence(item) for item in value]
    return {'unsupported_evidence': repr(value)}


def _known(value):
    """Preserve raw input elsewhere; extract only trustworthy nonnegative counters."""
    source = value if isinstance(value, dict) else {}
    known = {name: source[name] for name in WORK_COUNTS
             if type(source.get(name)) is int and source[name] >= 0}
    unknown = [name for name in WORK_COUNTS if name not in known]
    return known, unknown


class MainDispatch:
    """Root-bound, single-owner synthetic dispatcher with durable single-use claims.

    The default gate is the real main-protocol validator/authorizer. Tests may inject
    a validator for wholly synthetic metadata; they do not create a real study freeze.
    All returned reports explicitly deny live execution and performance evidence.
    """

    @classmethod
    def open(cls, protocol, prior, *, root, config_dir=None, resume=False, clock=time.time,
             validator=None, authorizer=None):
        self = cls()
        self.root = Path(root).resolve(strict=True)
        self.config_dir = Path(config_dir) if config_dir is not None else self.root / 'configs/maniguard/phase2'
        self.clock = clock
        self.validator = validator or main_protocol.validate_main_protocol
        self.authorizer = authorizer or main_protocol.authorize_job
        self.protocol, self.prior = deepcopy(protocol), deepcopy(prior)
        self.lock = None
        self.state = None
        self._saved_sha = None
        self.persistence_failure = None
        self._validate_protocol()
        study = protocol['study_id']
        if not isinstance(study, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', study) or study in ('.', '..'):
            raise ValueError('Invalid canonical study ID')
        self.directory = self.root / 'runs/maniguard/main-dispatch' / study
        if not self.directory.resolve().is_relative_to(self.root):
            raise ValueError('Dispatch directory escapes the bound study root')
        self.path = self.directory / 'state.json'
        self.anchor = self.directory / 'binding.json'
        self._validate_prior()
        self.jobs = {job['episode_id']: job for job in main_protocol.compile_jobs(protocol)}
        if any(any(part in ('', '.', '..') for part in key.split('/')) for key in self.jobs):
            raise ValueError('Frozen job IDs must use canonical relative path components')
        if len(self.jobs) != protocol['envelope']['episode_attempts']:
            raise ValueError('Frozen grid has duplicate or inconsistent job IDs')
        if record_hash(list(self.jobs.values())) != protocol['grid_sha256']:
            raise ValueError('Frozen grid digest differs from compiled jobs')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / 'dispatch.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            binding = {'schema': SCHEMA, 'study_root': str(self.root), 'study_id': study,
                       'protocol_sha256': protocol['sha256'], 'grid_sha256': protocol['grid_sha256'],
                       'prior_sha256': prior['sha256'],
                       'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       'execution_kind': 'synthetic_only'}
            if resume:
                anchor = json.loads(self.anchor.read_text())
                _check_signed(anchor)
                if any(anchor.get(key) != value for key, value in binding.items()):
                    raise ValueError('Canonical study binding changed on resume')
                self.state = json.loads(self.path.read_text())
                _check_signed(self.state)
                if any(self.state.get(key) != value for key, value in binding.items()):
                    raise ValueError('Dispatch root, protocol, prior or implementation changed on resume')
                if self.state['protocol'] != protocol or self.state['prior'] != prior:
                    raise ValueError('Saved protocol/prior differs from its declared binding')
                self._validate_history()
                self._saved_sha = self.state['sha256']
            else:
                if self.anchor.exists() or self.path.exists():
                    raise ValueError('This study already has durable state; resume instead of resetting claims')
                _atomic(self.anchor, _signed({**binding, 'created_at': self._now()}))
                self.state = {**binding, 'protocol': deepcopy(protocol), 'prior': deepcopy(prior),
                              'claims': {}, 'extra_events': {}, 'blocked_reasons': [], 'revision': 0,
                              'last_timestamp': self._now()}
                self._persist()
            reasons = self._budget_issues() + self._unresolved_issues()
            reasons.extend('Prior persistence failure requires explicit review: ' + path.name
                           for path in sorted(self.directory.glob('persistence-failure-*.json')))
            if self.prior.get('unknown_costs'):
                reasons.append('Prior accounting has unresolved costs')
            if reasons:
                self._block(reasons)
            return self
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.lock is not None:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None

    def _now(self):
        now = self.clock()
        if not _number(now) or now < self.prior['timestamp']:
            raise ValueError('Reservation clock precedes its prior or is nonfinite')
        if self.state is not None and now < self.state['last_timestamp']:
            raise ValueError('Reservation clock moved backwards')
        return now

    def _validate_protocol(self):
        if self.protocol.get('status') != 'frozen':
            raise ValueError('Dispatch requires an explicitly frozen main protocol')
        self.validator(self.protocol, config_dir=self.config_dir, root=self.root, require_frozen=True)

    def _validate_prior(self):
        prior = self.prior
        _check_signed(prior)
        if (prior.get('schema') != PRIOR_SCHEMA or prior.get('study_root') != str(self.root)
                or prior.get('protocol_sha256') != self.protocol['sha256']):
            raise ValueError('Prior accounting must bind this exact study root and protocol')
        _counts(prior.get('counts'))
        if (not isinstance(prior.get('unknown_costs'), list)
                or any(not isinstance(item, str) for item in prior['unknown_costs'])):
            raise ValueError('Prior accounting must explicitly list unresolved costs')
        fields = ('timestamp', 'reservation_started_at', 'allocated_gpu_seconds')
        if any(not _number(prior.get(name)) for name in fields):
            raise ValueError('Prior reservation clocks must be finite and nonnegative')
        gpus = prior.get('reserved_gpus')
        duration = prior['timestamp'] - prior['reservation_started_at']
        if (type(gpus) is not int or gpus < 1 or duration < 0 or
                not math.isclose(prior['allocated_gpu_seconds'], gpus * duration, abs_tol=1e-6)):
            raise ValueError('Prior GPU reservation interval is inconsistent')
        self._now()
        if not isinstance(prior.get('sources'), list) or not prior['sources']:
            raise ValueError('Prior accounting needs immutable hashed sources')
        for source in prior['sources']:
            if set(source) != {'path', 'sha256'}:
                raise ValueError('Invalid prior source binding')
            path = (self.root / source['path']).resolve()
            if (not path.is_relative_to(self.root) or
                    hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']):
                raise ValueError('Prior accounting source changed or escapes the study root')

    def _validate_history(self):
        for job_id, claim in self.state['claims'].items():
            if (job_id not in self.jobs or record_hash(claim['job']) != record_hash(self.jobs[job_id])
                    or claim['job_sha256'] != record_hash(claim['job'])
                    or claim['reservation'] != claim['job']['resource_envelope']['successful_work']
                    or claim['output'] != str((self.directory / 'episodes' / job_id).resolve())):
                raise ValueError('Saved claim differs from the frozen job or canonical output')
            if claim['status'] not in ('claimed', 'terminal'):
                raise ValueError('Unknown claim status')
            _counts(claim['known_counts'], complete=claim['status'] == 'claimed')
            if claim['status'] == 'terminal':
                known, unknown = _known(claim['row'].get('costs'))
                if claim['known_counts'] != known or claim['unknown_counts'] != unknown:
                    raise ValueError('Saved result accounting differs from its retained row')
        for key, event in self.state['extra_events'].items():
            if key != event['event_id'] or event['category'] not in ('startup', 'failed_work'):
                raise ValueError('Extra event identity/category mismatch')
            _counts(event['reservation'])
            if event['status'] not in ('reserved', 'terminal'):
                raise ValueError('Unknown extra-event status')
            if event['status'] == 'terminal':
                known, unknown = _known(event['raw_counts'])
                if known != event['known_counts'] or unknown != event['unknown_counts']:
                    raise ValueError('Saved extra accounting differs from retained counters')

    def _usable(self):
        if self.persistence_failure is not None:
            raise DispatchPersistenceError('Dispatch handle is poisoned by a persistence failure; close it')
        if self.lock is None:
            raise RuntimeError('Dispatch state is closed')

    def _persist(self):
        self._usable()
        try:
            if self._saved_sha is not None:
                current = json.loads(self.path.read_text())
                _check_signed(current)
                if current['sha256'] != self._saved_sha:
                    raise ValueError('Persisted state changed while the dispatcher held its lock')
            self.state['last_timestamp'] = self._now()
            self.state['revision'] += 1
            self.state = _signed(self.state)
            _atomic(self.path, self.state)
            self._saved_sha = self.state['sha256']
        except BaseException as error:
            # Do not roll back or retry: an error after replace may already have
            # changed the file, and the candidate may contain paid-work evidence.
            self.persistence_failure = {
                'error_type': type(error).__name__, 'error': str(error),
                'last_committed_sha256': self._saved_sha,
            }
            try:
                diagnostic = self.directory / ('persistence-failure-' + str(time.time_ns()) + '.json')
                _atomic(diagnostic, _signed({
                    'schema': 'maniguard-main-dispatch-persistence-failure-v1',
                    'study_root': str(self.root), 'state_path': str(self.path),
                    'protocol_sha256': self.protocol['sha256'],
                    **self.persistence_failure, 'uncommitted_state': _evidence(self.state),
                    'durability_uncertain': True, 'automatic_recovery_authorized': False,
                }))
                self.persistence_failure['diagnostic_path'] = str(diagnostic)
            except BaseException as diagnostic_error:  # noqa: BLE001 - never mask the primary failure
                # A failing filesystem may prevent even the diagnostic write.
                self.persistence_failure['diagnostic_error'] = (
                    type(diagnostic_error).__name__ + ': ' + str(diagnostic_error))
            if not isinstance(error, Exception):
                raise
            raise DispatchPersistenceError('Dispatch persistence failed: ' + str(error)) from error

    def _block(self, reasons):
        self._usable()
        self.state['blocked_reasons'] = list(dict.fromkeys(self.state['blocked_reasons'] + reasons))
        self._persist()

    def snapshot(self):
        self._usable()
        totals = dict(self.prior['counts'])
        reserved = dict.fromkeys(WORK_COUNTS, 0)
        categories = {'prior': dict(self.prior['counts']), 'startup': dict.fromkeys(WORK_COUNTS, 0),
                      'failed_work': dict.fromkeys(WORK_COUNTS, 0)}
        category_reserved = {key: dict.fromkeys(WORK_COUNTS, 0) for key in categories}
        for item in self.state['claims'].values():
            for name in WORK_COUNTS:
                known = item['known_counts'].get(name, 0)
                totals[name] += known
                if item['status'] == 'claimed' or item['unknown_counts']:
                    reserved[name] += max(0, item['reservation'][name] - known)
        for item in self.state['extra_events'].values():
            category = item['category']
            for name in WORK_COUNTS:
                known = item['known_counts'].get(name, 0)
                totals[name] += known
                categories[category][name] += known
                if item['status'] == 'reserved' or item['unknown_counts']:
                    remaining = max(0, item['reservation'][name] - known)
                    reserved[name] += remaining
                    category_reserved[category][name] += remaining
        now = self._now()
        return {'known_counts': totals, 'reserved_counts': reserved, 'category_known_counts': categories,
                'category_reserved_counts': category_reserved, 'timestamp': now,
                'allocated_gpu_seconds': self.prior['allocated_gpu_seconds'] +
                    (now - self.prior['timestamp']) * self.prior['reserved_gpus']}

    def _budget_issues(self, additional=None, category=None):
        snap = self.snapshot()
        budget = self.protocol['spec']['budget']
        additional = additional or dict.fromkeys(WORK_COUNTS, 0)
        issues = []
        for name, key in BUDGET_KEYS.items():
            if snap['known_counts'][name] + snap['reserved_counts'][name] + additional[name] > budget[key]:
                issues.append('Total resource ceiling exceeded: ' + name)
            for part in ('prior', 'startup', 'failed_work'):
                amount = snap['category_known_counts'][part][name] + snap['category_reserved_counts'][part][name]
                if part == category:
                    amount += additional[name]
                if amount > budget['separate_caps'][part][name]:
                    issues.append('Separate resource ceiling exceeded: ' + part + '/' + name)
        if snap['allocated_gpu_seconds'] >= budget['allocated_gpu_hours_max'] * 3600:
            issues.append('Allocated GPU reservation ceiling reached')
        if len(self.state['claims']) > budget['episode_attempts_max']:
            issues.append('Episode-attempt ceiling exceeded')
        return issues

    def _unresolved_issues(self):
        issues = []
        for key, claim in self.state['claims'].items():
            if claim['status'] == 'terminal' and (
                    claim['unknown_counts'] or claim.get('issues') or claim['row'].get('valid') is not True):
                issues.append('Unresolved terminal episode accounting/status: ' + key)
        for key, event in self.state['extra_events'].items():
            if event['status'] == 'terminal' and (event['unknown_counts'] or event.get('issues')):
                issues.append('Unresolved terminal extra-work accounting/status: ' + key)
        return issues

    def _ready(self):
        self._usable()
        self._validate_protocol()
        self._validate_prior()
        unresolved = self._unresolved_issues()
        if unresolved:
            self._block(unresolved)
        if self.state['blocked_reasons']:
            raise RuntimeError('Dispatch blocked: ' + '; '.join(self.state['blocked_reasons']))
        if any(item['status'] == 'claimed' for item in self.state['claims'].values()):
            raise RuntimeError('Unresolved durable claim; never rerun automatically')
        if any(item['status'] == 'reserved' for item in self.state['extra_events'].values()):
            raise RuntimeError('Unresolved extra-work reservation')
        issues = self._budget_issues()
        if issues:
            self._block(issues)
            raise RuntimeError('; '.join(issues))

    def pending_jobs(self):
        self._ready()
        return [deepcopy(job) for key, job in self.jobs.items() if key not in self.state['claims']]

    def claim(self, job):
        self._ready()
        authorization = self.authorizer(self.protocol, job, config_dir=self.config_dir, root=self.root)
        job_id = job.get('episode_id')
        if (job_id not in self.jobs or record_hash(job) != record_hash(self.jobs[job_id])
                or authorization != {'authorized': True, 'protocol_sha256': self.protocol['sha256'],
                    'episode_id': job_id, 'job_sha256': record_hash(job)}):
            raise ValueError('Exact frozen-job authorization differs from requested dispatch')
        if job['resource_contract']['profile'] not in SUPPORTED_PROFILES:
            raise ValueError('AAC and unsupported live execution profiles are not implemented in this slice')
        if job_id in self.state['claims']:
            raise ValueError('A durable episode claim can be consumed only once')
        if len(self.state['claims']) >= self.protocol['spec']['budget']['episode_attempts_max']:
            raise RuntimeError('Episode-attempt ceiling reached')
        reservation = _counts(job['resource_envelope']['successful_work'])
        issues = self._budget_issues(reservation)
        if issues:
            raise RuntimeError('; '.join(issues))
        output = (self.directory / 'episodes' / job_id).resolve()
        if not output.is_relative_to((self.directory / 'episodes').resolve()):
            raise ValueError('Claim output escapes the canonical episode directory')
        claim = {'job': deepcopy(job), 'job_sha256': record_hash(job), 'status': 'claimed',
                 'claimed_at': self._now(), 'output': str(output),
                 'reservation': reservation, 'known_counts': dict.fromkeys(WORK_COUNTS, 0),
                 'unknown_counts': [], 'backend_receipt': None, 'row': None}
        self.state['claims'][job_id] = claim
        self._persist()
        return deepcopy(claim)

    def reserve_extra(self, event_id, category, counts):
        self._ready()
        if (not isinstance(event_id, str) or not event_id or event_id in self.state['extra_events']
                or category not in ('startup', 'failed_work')):
            raise ValueError('Extra work needs a new event ID and startup/failed_work category')
        counts = _counts(counts)
        issues = self._budget_issues(counts, category)
        if issues:
            raise RuntimeError('; '.join(issues))
        self.state['extra_events'][event_id] = {'event_id': event_id, 'category': category,
            'status': 'reserved', 'reservation': counts, 'known_counts': dict.fromkeys(WORK_COUNTS, 0),
            'unknown_counts': [], 'raw_counts': None}
        self._persist()
        return deepcopy(self.state['extra_events'][event_id])

    def settle_extra(self, event_id, counts, *, issues=()):
        self._usable()
        event = self.state['extra_events'][event_id]
        if event['status'] != 'reserved':
            raise ValueError('Extra work is already terminal and cannot be counted twice')
        counts = _evidence(counts)
        known, unknown = _known(counts)
        reasons = list(issues)
        if unknown:
            reasons.append('Unknown extra-work counts: ' + ', '.join(unknown))
        if not isinstance(counts, dict) or set(counts) - set(WORK_COUNTS):
            reasons.append('Malformed extra-work counter map')
        reasons.extend('Extra-work reservation exceeded: ' + name for name, value in known.items()
                       if value > event['reservation'][name])
        event.update(status='terminal', known_counts=known, unknown_counts=unknown, raw_counts=deepcopy(counts),
                     issues=reasons)
        self._persist()
        self._block(reasons + self._budget_issues())
        return self.report()

    def record_result(self, job_id, row):
        self._usable()
        claim = self.state['claims'][job_id]
        if claim['status'] != 'claimed':
            raise ValueError('A terminal episode cannot be recorded twice')
        row = _evidence(row)
        if not isinstance(row, dict):
            row = {'raw_row': row, 'valid': False, 'costs': {}}
        known, unknown = _known(row.get('costs'))
        reasons = []
        if (row.get('evidence_kind') != 'synthetic' or row.get('valid') is not True or
                row.get('protocol_sha256') != self.protocol['sha256'] or
                record_hash(row.get('job')) != claim['job_sha256'] or
                row.get('output') != claim['output']):
            reasons.append('Result is invalid or differs from its synthetic exact-job binding')
        for name in ('cost_issues', 'missing_cost_fields'):
            if row.get(name) != []:
                reasons.append('Unresolved result accounting: ' + name)
        if unknown:
            reasons.append('Unknown result counts: ' + ', '.join(unknown))
        costs = row.get('costs')
        if not isinstance(costs, dict) or set(costs) - (set(WORK_COUNTS) | {'elapsed_seconds'}):
            reasons.append('Malformed slot-work counter map')
        if not isinstance(costs, dict) or not _number(costs.get('elapsed_seconds')):
            reasons.append('Explicit finite elapsed_seconds is required')
        reasons.extend('Slot resource reservation exceeded: ' + name for name, value in known.items()
                       if value > claim['reservation'][name])
        claim.update(status='terminal', row=row, known_counts=known, unknown_counts=unknown, issues=reasons)
        self._persist()
        self._block(reasons + self._budget_issues())
        return self.report()

    def run_synthetic(self, job, *, runner, reader):
        """Call only explicitly supplied test callbacks; no production backend exists.

        Runner returns a synthetic receipt. Reader receives (receipt, durable_claim).
        A failure preserves the receipt and any known counts, blocks the dispatcher,
        and never retries. Callbacks are trusted test code, not a security sandbox.
        """
        if not callable(runner) or not callable(reader):
            raise TypeError('Supply explicit synthetic runner and reader callbacks')
        claim = self.claim(job)
        receipt = None
        try:
            receipt = _evidence(runner(deepcopy(claim)))
            self.state['claims'][job['episode_id']]['backend_receipt'] = deepcopy(receipt)
            self._persist()
            if not isinstance(receipt, dict) or receipt.get('evidence_kind') != 'synthetic':
                raise ValueError('Only synthetic backend receipts are accepted')
            row = reader(deepcopy(receipt), deepcopy(claim))
            if not isinstance(row, dict):
                raise TypeError('Synthetic reader must return a JSON-compatible mapping')
            lower_bounds, _ = _known(receipt.get('costs'))
            measured, _ = _known(row.get('costs'))
            missing = [name for name, count in lower_bounds.items() if count > measured.get(name, -1)]
            if missing:
                raw_row = deepcopy(row)
                retained = dict(row.get('costs') or {})
                for name in missing:
                    retained[name] = lower_bounds[name]
                row.update(raw_reader_row=raw_row, costs=retained, valid=False,
                           cost_issues=list(row.get('cost_issues') or []) +
                               ['Reader omits or understates backend counts: ' + ', '.join(missing)])
            if receipt.get('process_succeeded') is not True:
                row = {**row, 'valid': False}
            return self.record_result(job['episode_id'], row)
        except BaseException as error:
            if self.persistence_failure is not None:
                raise
            current = self.state['claims'][job['episode_id']]
            if current['status'] == 'claimed':
                self.record_result(job['episode_id'], {
                    'evidence_kind': 'synthetic', 'job': claim['job'], 'output': claim['output'],
                    'protocol_sha256': self.protocol['sha256'], 'valid': False,
                    'costs': receipt.get('costs', {}) if isinstance(receipt, dict) else {},
                    'cost_issues': [f'Backend/reader failed: {type(error).__name__}: {error}'],
                    'missing_cost_fields': ['audit_incomplete'],
                })
            if not isinstance(error, Exception):
                raise
            return self.report()

    def report(self):
        self._usable()
        unresolved = [key for key, value in self.state['claims'].items() if value['status'] == 'claimed']
        return {'schema': SCHEMA, 'state_path': str(self.path), 'protocol_sha256': self.protocol['sha256'],
                'claimed_jobs': len(self.state['claims']), 'terminal_jobs': sum(
                    value['status'] == 'terminal' for value in self.state['claims'].values()),
                'unresolved_claims': unresolved, 'unresolved_extra_events': [key for key, value in
                    self.state['extra_events'].items() if value['status'] == 'reserved'],
                'blocked_reasons': list(self.state['blocked_reasons']),
                'accounting': self.snapshot(), 'execution_kind': 'synthetic_only',
                'live_execution_authorized': False, 'physical_evidence_audited': False,
                'performance_results': False}
