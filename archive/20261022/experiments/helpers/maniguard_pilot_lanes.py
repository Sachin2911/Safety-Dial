"""Concurrent independent optimizer lanes over the frozen pilot coordinator.

The frozen PilotCoordinator runs one dependency block at a time in a fixed global
order. Every block belongs to one lane: a (method, optimizer seed) run, whose training
panels, PPO batches and finalist validation must stay ordered, or the shared reference
block. Lanes share no optimizer state, so blocks of different lanes may run together.

LaneCoordinator keeps at most one active block per lane. Every preparation, claim,
binding check, optimizer update and selection executes through the frozen methods by
placing that block in the frozen coordinator's single active slot, so a lane sees
exactly its frozen blocks, candidates, dispatch permutation and updates. Only the
interleaving across lanes changes. Any invalid attempt still blocks every lane, and
claimed work is never replayed. ``canonical_state`` returns a frozen-format coordinator
state once every block is complete.
"""

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from helpers.maniguard_phase2_coordinator import PilotCoordinator
from helpers.maniguard_phase2_model import genome_hash
from helpers.maniguard_phase2_optim import record_hash

VERSION = 'maniguard-pilot-lanes-v1'
STAGE_FIELDS = ('active_jobs', 'active_job_order', 'active_results')


def lane_key(stage):
    if stage['kind'] == 'references':
        return 'references'
    return f"{stage['method']}/s{stage['seed']}"


def _source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _empty_stage():
    return {'status': 'pending', 'active_jobs': None, 'active_job_order': [], 'active_results': {},
            'prepared_at': None, 'completed_at': None}


class LaneCoordinator:
    """Run one active frozen block per independent lane; the frozen code does all scoring."""

    def __init__(self, base, *, takeover=None, clock=time.time, priority=None):
        if base.claims or base.blocked_reasons:
            raise ValueError('Lane takeover requires an unblocked state with no in-flight claims')
        self.base, self.protocol, self.clock = base, base.protocol, clock
        self.lanes = self._lanes(base)
        self.stages = {index: _empty_stage() for index in range(len(base.stage_plan))}
        for index in range(base.stage_index):
            self.stages[index]['status'] = 'complete'
        if base.active_jobs is not None:
            self.stages[base.stage_index].update(
                status='active', **{field: getattr(base, field) for field in STAGE_FIELDS})
        self.completion_order = [entry['stage_index'] for entry in base.history]
        self.takeover = deepcopy(takeover)
        self._last_saved_sha256 = None
        self.set_priority(priority)
        self._park()
        self._validate()

    def set_priority(self, priority=None):
        """Order lanes for dispatch only; jobs and updates never depend on this order."""
        if priority is not None and (len(set(priority)) != len(priority)
                                     or not set(priority) <= set(self.lanes)):
            raise ValueError('Lane priority must name distinct known lanes')
        self.priority = None if priority is None else list(priority)

    @staticmethod
    def _lanes(base):
        lanes = {}
        for index, stage in enumerate(base.stage_plan):
            lanes.setdefault(lane_key(stage), []).append(index)
        return lanes

    def _park(self):
        base = self.base
        base.stage_index = len(base.history)
        base.active_jobs, base.active_job_order, base.active_results = None, [], {}

    @contextmanager
    def _stage(self, index):
        """Place one lane block in the frozen coordinator's active slot for a frozen call."""
        record = self.stages[index]
        if record['status'] == 'complete':
            raise ValueError('Stage is already complete')
        base = self.base
        base.stage_index = index
        for field in STAGE_FIELDS:
            setattr(base, field, record[field])
        completed = len(base.history)
        try:
            yield base
        finally:
            if len(base.history) > completed:
                if len(base.history) != completed + 1 or base.history[-1]['stage_index'] != index:
                    raise RuntimeError('Frozen block completion is inconsistent with its lane')
                record.update(status='complete', active_jobs=None, active_job_order=[],
                              active_results={}, completed_at=self.clock())
                self.completion_order.append(index)
            else:
                for field in STAGE_FIELDS:
                    record[field] = getattr(base, field)
                if record['active_jobs'] is not None and record['status'] == 'pending':
                    record.update(status='active', prepared_at=self.clock())
            self._park()

    def _current(self, key):
        return next((index for index in self.lanes[key]
                     if self.stages[index]['status'] != 'complete'), None)

    def _active_stage_of(self, job_id):
        hits = [index for index, record in self.stages.items() if record['status'] == 'active'
                and job_id in (record['active_jobs'] or {})]
        return hits[0] if len(hits) == 1 else None

    @property
    def revision(self):
        return self.base.revision

    @property
    def status(self):
        if self.base.blocked_reasons:
            return 'blocked'
        open_kinds = {self.base.stage_plan[index]['kind'] for index, record in self.stages.items()
                      if record['status'] != 'complete'}
        if not open_kinds:
            return 'complete'
        return 'training' if open_kinds & {'search', 'ppo'} else 'validation'

    def next_jobs(self):
        """Unclaimed jobs of every lane's current block, in priority or critical-path order."""
        if self.status in ('blocked', 'complete'):
            return []
        ranked = []
        for key, indices in self.lanes.items():
            index = self._current(key)
            if index is None:
                continue
            with self._stage(index) as base:
                jobs = base.next_jobs()
            if self.priority is None:
                rank = -sum(self.stages[value]['status'] != 'complete' for value in indices)
            else:
                rank = self.priority.index(key) if key in self.priority else len(self.priority)
            ranked.extend(((rank, index, position), job) for position, job in enumerate(jobs))
        ranked.sort(key=lambda item: item[0])
        return [job for _, job in ranked]

    def mark_started(self, job_id, output):
        index = self._active_stage_of(job_id)
        if index is None:
            raise ValueError('Job is not in exactly one active lane block')
        with self._stage(index) as base:
            return base.mark_started(job_id, output)

    def record_result(self, job_id, row):
        claim = self.base.claims.get(job_id)
        index = claim['stage_index'] if claim else self._active_stage_of(job_id)
        if index is None or self.stages[index]['status'] != 'active':
            # Parked frozen code records the attempt, keeps its costs and blocks every lane.
            self.base.record_result(job_id, row)
        else:
            with self._stage(index) as base:
                base.record_result(job_id, row)
        return self.report()

    def report(self):
        report = self.base.report()
        lanes = {}
        for key, indices in self.lanes.items():
            current = self._current(key)
            lanes[key] = {'blocks': len(indices),
                          'completed': sum(self.stages[i]['status'] == 'complete' for i in indices),
                          'current_stage': current,
                          'current_status': None if current is None else self.stages[current]['status']}
        report.update(status=self.status, dispatch='concurrent-lanes', lanes=lanes,
                      lane_priority=self.priority,
                      completed_stages=sum(r['status'] == 'complete' for r in self.stages.values()))
        return report

    def _validate(self):
        base = self.base
        if set(self.stages) != set(range(len(base.stage_plan))):
            raise ValueError('Lane state does not cover the frozen block plan')
        complete = [index for index, record in self.stages.items() if record['status'] == 'complete']
        history = [entry['stage_index'] for entry in base.history]
        if history != self.completion_order or sorted(history) != sorted(complete):
            raise ValueError('Completed blocks differ from the frozen history')
        if base.stage_index != len(base.history) or base.active_jobs is not None:
            raise ValueError('Frozen coordinator must be parked between lane calls')
        for key, indices in self.lanes.items():
            statuses = [self.stages[index]['status'] for index in indices]
            first_open = next((i for i, value in enumerate(statuses) if value != 'complete'),
                              len(statuses))
            if any(value != 'pending' for value in statuses[first_open + 1:]):
                raise ValueError('A lane block became active before its predecessor completed')
            if any(value not in ('complete', 'active', 'pending') for value in statuses):
                raise ValueError('Unknown lane block status')
        for index, record in self.stages.items():
            if record['status'] != 'active':
                if record['active_jobs'] is not None or record['active_results']:
                    raise ValueError('Only active blocks may hold jobs or results')
                continue
            jobs = record['active_jobs']
            if (not isinstance(jobs, dict) or len(record['active_job_order']) != len(set(record[
                    'active_job_order'])) or set(record['active_job_order']) != set(jobs)
                    or not set(record['active_results']) <= set(jobs)):
                raise ValueError('Active block dispatch state is inconsistent')
            for job in jobs.values():
                if genome_hash(job['genome']) != job['actor_sha256']:
                    raise ValueError('Pending actor hash mismatch')
        for job_id, claim in base.claims.items():
            record = self.stages.get(claim.get('stage_index'))
            if (record is None or record['status'] != 'active' or job_id not in record['active_jobs']
                    or job_id in record['active_results']):
                raise ValueError('In-flight claim does not belong to an active lane block')

    def state_dict(self):
        claims = self.base.claims
        self.base.claims = {}
        try:
            base_state = self.base.state_dict()
        finally:
            self.base.claims = claims
        state = deepcopy({'version': VERSION, 'source_sha256': _source_hash(), 'base': base_state,
                          'claims': claims, 'completion_order': self.completion_order,
                          'stages': {str(index): record for index, record in self.stages.items()},
                          'takeover': self.takeover})
        return state | {'sha256': record_hash(state)}

    @classmethod
    def from_state_dict(cls, state, *, clock=time.time, priority=None):
        data = deepcopy(state)
        digest = data.pop('sha256')
        if (digest != record_hash(data) or data['version'] != VERSION or
                data['source_sha256'] != _source_hash()):
            raise ValueError('Lane state/source integrity mismatch')
        base = PilotCoordinator.from_state_dict(data['base'])
        lanes = cls.__new__(cls)
        lanes.base, lanes.protocol, lanes.clock = base, base.protocol, clock
        lanes.lanes = cls._lanes(base)
        lanes.stages = {int(index): record for index, record in data['stages'].items()}
        lanes.completion_order = data['completion_order']
        lanes.takeover = data['takeover']
        lanes._last_saved_sha256 = None
        lanes.set_priority(priority)
        base.claims = data['claims']
        lanes._park()
        lanes._validate()
        return lanes

    def save(self, path):
        """Atomically persist state; reject stale writers and unrelated existing files."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            previous = json.loads(path.read_text())
            if self._last_saved_sha256 is None or previous.get('sha256') != self._last_saved_sha256:
                raise ValueError('Lane state file changed or belongs to another writer; reload first')
        elif self._last_saved_sha256 is not None:
            raise ValueError('Previously saved lane state disappeared; do not silently recreate it')
        state = self.state_dict()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                             suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(state, stream, sort_keys=True, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self._last_saved_sha256 = state['sha256']
        return state['sha256']

    @classmethod
    def load(cls, path, *, clock=time.time, priority=None):
        state = json.loads(Path(path).read_text())
        lanes = cls.from_state_dict(state, clock=clock, priority=priority)
        lanes._last_saved_sha256 = state['sha256']
        return lanes

    def canonical_state(self):
        """Frozen-format state of the completed pilot, with history in block order."""
        if self.status != 'complete' or self.base.claims:
            raise ValueError('Only a complete, unblocked lane run has a canonical state')
        history = self.base.history
        self.base.history = sorted(history, key=lambda entry: entry['stage_index'])
        try:
            state = self.base.state_dict()
        finally:
            self.base.history = history
        PilotCoordinator.from_state_dict(state)
        return state
