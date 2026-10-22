"""Durable pre-operation reservations for one main child; no simulator imports.

Counters named calls/images/steps are returned work; *_attempts and policy_queries
are dispatch attempts. An interrupted operation retains its reserved return work as
unknown. Parent cumulative accounting must include all three category allocations.
"""
import hashlib
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path

from helpers.maniguard_main_protocol import WORK_COUNTS
from helpers.maniguard_phase2_optim import record_hash

CATEGORIES = ('slot', 'startup', 'failed_work')


def counts(value, *, partial=False):
    if (not isinstance(value, dict) or set(value) - set(WORK_COUNTS)
            or (not partial and set(value) != set(WORK_COUNTS))
            or any(type(number) is not int or number < 0 for number in value.values())):
        raise ValueError('Explicit nonnegative integer work counters are required')
    return dict(value)


def signed(value):
    return {**value, 'sha256': record_hash({key: item for key, item in value.items() if key != 'sha256'})}


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class OperationMeter:
    """Single-process, fail-closed reservations written before calling expensive work."""

    def __init__(self, output, job_sha256, reservation, *, clock=time.time):
        self.output, self.clock = Path(output), clock
        self.job_sha256 = job_sha256
        self.limits = {name: counts(reservation[key]) for name, key in (
            ('slot', 'counts'), ('startup', 'startup_counts'), ('failed_work', 'failed_work_counts'))}
        self.deadline = reservation['deadline']
        if type(self.deadline) not in (int, float) or not math.isfinite(self.deadline):
            raise ValueError('An absolute finite parent deadline is required')
        self.started = self.last_time = clock()
        if self.started >= self.deadline:
            raise RuntimeError('Parent deadline already exhausted')
        self.by_category = {key: dict.fromkeys(WORK_COUNTS, 0) for key in CATEGORIES}
        self.pending, self.issues = {}, []
        self.phase = 'initialization'
        self.index = self.next_id = 0
        self.broken = False
        self.journal_digest = hashlib.sha256()
        self.stream = (self.output / 'main_operations.jsonl').open('x')
        self.snapshot()

    def _time(self):
        now = self.clock()
        if not math.isfinite(now) or now < self.last_time:
            raise RuntimeError('Main child clock moved backwards or is nonfinite')
        self.last_time = now
        return now

    def _event(self, event):
        try:
            raw = json.dumps({'index': self.index, **event}, sort_keys=True, allow_nan=False) + '\n'
            self.stream.write(raw)
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.journal_digest.update(raw.encode())
            self.index += 1
        except BaseException:
            self.broken = True
            raise

    def begin(self, kind, *, category='slot', attempts=None, returns=None):
        if self.broken:
            raise RuntimeError('Operation meter is poisoned; no further work is authorized')
        if self._time() >= self.deadline:
            raise RuntimeError('Parent wall deadline reached before operation')
        attempts, returns = counts(attempts or {}, partial=True), counts(returns or {}, partial=True)
        if category not in CATEGORIES:
            raise ValueError('Unknown work category')
        for name in WORK_COUNTS:
            held = sum(item['return_counts'].get(name, 0) for item in self.pending.values()
                       if item['category'] == category)
            if self.by_category[category][name] + held + attempts.get(name, 0) + returns.get(name, 0) > self.limits[category][name]:
                raise RuntimeError('Operation exceeds reserved ' + category + '/' + name)
        identifier = str(self.next_id)
        event = {'type': 'intent', 'id': identifier, 'kind': kind, 'category': category,
                 'attempt_counts': attempts, 'return_counts': returns}
        self._event(event)
        self.next_id += 1
        self.pending[identifier] = deepcopy(event)
        for name, value in attempts.items():
            self.by_category[category][name] += value
        self.snapshot()
        return identifier

    def end(self, identifier):
        if self.broken:
            raise RuntimeError('Cannot complete a poisoned operation meter')
        event = self.pending[identifier]
        self._event({'type': 'returned', 'id': identifier, 'return_counts': event['return_counts']})
        for name, value in event['return_counts'].items():
            self.by_category[event['category']][name] += value
        del self.pending[identifier]
        self.snapshot()

    def fail(self, identifier, error):
        self._event({'type': 'failed', 'id': identifier, 'return_counts': {},
                     'error_type': type(error).__name__})
        # Retain the intent, including any unresolved returned work. No retry.
        self.issues.append('Operation failed; retained dispatch/return uncertainty: ' + identifier)
        self.broken = True
        self.phase = 'failed'
        self.snapshot()

    def call(self, kind, function, *, category='slot', attempts=None, returns=None):
        identifier = self.begin(kind, category=category, attempts=attempts, returns=returns)
        try:
            value = function()
        except BaseException as error:
            self.fail(identifier, error)
            raise
        self.end(identifier)
        return value

    def snapshot(self):
        total = {name: sum(part[name] for part in self.by_category.values()) for name in WORK_COUNTS}
        inflight = {name: sum(item['return_counts'].get(name, 0) for item in self.pending.values())
                    for name in WORK_COUNTS}
        value = signed({'schema': 'maniguard-main-costs-v1', 'job_sha256': self.job_sha256,
                        'counts': total, 'reserved_inflight': inflight,
                        'by_category': deepcopy(self.by_category), 'cost_breakdown': deepcopy(self.by_category),
                        'pending': list(self.pending.values()), 'issues': list(self.issues),
                        'phase': self.phase, 'elapsed_seconds': self._time() - self.started,
                        'journal_sha256': self.journal_digest.copy().hexdigest()})
        try:
            atomic(self.output / 'main_costs.json', value)
        except BaseException:
            self.broken = True
            raise
        return value

    def close(self):
        if not self.stream.closed:
            self.snapshot()
            self.stream.close()
