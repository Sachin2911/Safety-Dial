"""Resumable replicate-diagnostic scheduling and prespecified descriptive analysis.

The diagnostic reruns frozen pilot controllers on frozen pilot cases several times. It
never selects, updates or ranks for use. ``ReplicateCoordinator`` offers the subset of
the pilot coordinator interface that ``run_parallel`` needs: durable claims before
launch, one recorded attempt per cell replicate, retained costs for every attempt, and a
blocked status after any non-voided failure.
"""

import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from helpers.maniguard_phase2_optim import record_hash

VERSION = 'maniguard-replicate-coordinator-v1'


def protocol_hash(protocol):
    body = {key: value for key, value in protocol.items() if key != 'sha256'}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def load_replicate_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get('status') != 'frozen' or protocol.get('sha256') != protocol_hash(protocol):
        raise ValueError('Replicate protocol is not frozen or its hash changed')
    return protocol


def compile_jobs(protocol):
    cells = [(controller, case) for controller in protocol['controllers'] for case in protocol['cases']]
    rng = np.random.default_rng(protocol['dispatch']['seed'])
    jobs = []
    for replicate in range(protocol['replicates_per_cell']):
        for index in rng.permutation(len(cells)):
            controller, case = cells[int(index)]
            jobs.append({'job_id': f"replicate/{controller['slot']}/{case['scene'].split('/')[0]}/r{replicate}",
                         'slot': controller['slot'], 'actor_sha256': controller['actor_sha256'],
                         'actor_file': controller['actor_file'], 'replicate': replicate,
                         'family': case['family'], 'scene': case['scene'], 'seed': case['policy_seed'],
                         'sampling_seed': protocol['sampling_seed'], 'max_steps': case['max_steps'],
                         'purpose': case['purpose'], 'mode': protocol['mode'],
                         'fixed_choices': None, 'stochastic': protocol['stochastic'], 'stage': 'replicate'})
    return jobs


class DiagnosticBudget:
    """Allocated GPU time from the diagnostic start plus episode/action ceilings."""

    def __init__(self, envelope, started_at, *, clock=time.time):
        self.envelope, self.started_at, self.clock = deepcopy(envelope), started_at, clock
        self.absolute_stop = time.mktime(time.strptime(envelope['absolute_stop_utc'], '%Y-%m-%dT%H:%M:%SZ')) \
            - time.timezone

    def snapshot(self, costs=None):
        now = self.clock()
        return {'timestamp': now, 'allocated_gpu_seconds': self.envelope['gpus'] * (now - self.started_at),
                'policy_control_steps': float((costs or {}).get('policy_control_steps', 0))}

    def check(self, costs=None, *, next_actions=0, next_queries=0):
        value = self.snapshot(costs)
        if value['allocated_gpu_seconds'] >= self.envelope['gpu_hours_ceiling_from_start'] * 3600:
            raise RuntimeError('Diagnostic GPU-hour ceiling reached')
        if value['timestamp'] >= self.absolute_stop:
            raise RuntimeError('Diagnostic absolute stop time reached')
        if value['policy_control_steps'] + next_actions > self.envelope['policy_actions_max']:
            raise RuntimeError('Diagnostic policy-action ceiling cannot reserve the attempt')
        return value

    def remaining_wall_seconds(self, costs=None):
        value = self.check(costs)
        by_gpu = (self.envelope['gpu_hours_ceiling_from_start'] * 3600 - value['allocated_gpu_seconds']) \
            / self.envelope['gpus']
        return min(by_gpu, self.absolute_stop - value['timestamp'])


class ReplicateCoordinator:
    def __init__(self, protocol):
        self.protocol = protocol
        self.jobs = {job['job_id']: job for job in compile_jobs(protocol)}
        self.order = [job['job_id'] for job in compile_jobs(protocol)]
        self.claims, self.results, self.ledger, self.blocked_reasons = {}, {}, [], []
        self.revision = 0
        self._last_saved_sha256 = None

    @property
    def status(self):
        if self.blocked_reasons:
            return 'blocked'
        return 'complete' if len(self.results) == len(self.jobs) else 'running'

    def next_jobs(self):
        if self.status != 'running':
            return []
        return [deepcopy(self.jobs[job_id]) for job_id in self.order
                if job_id not in self.results and job_id not in self.claims]

    def mark_started(self, job_id, output):
        if job_id not in self.jobs or job_id in self.claims or job_id in self.results:
            raise ValueError('Job is unknown, already claimed or already recorded')
        output = str(Path(output).resolve())
        if output in {claim['output'] for claim in self.claims.values()} | {
                entry['result']['output'] for entry in self.ledger}:
            raise ValueError('Every physical attempt requires a unique output directory')
        self.claims[job_id] = {'output': output}
        self.revision += 1
        return deepcopy(self.claims[job_id])

    def record_result(self, job_id, row):
        output, episode = row.get('output'), row.get('episode_id')
        if any(entry['result']['output'] == output or entry['result']['episode_id'] == episode
               for entry in self.ledger):
            raise ValueError('Physical episode already recorded')
        job, claim = self.jobs.get(job_id), self.claims.pop(job_id, None)
        summary = {key: deepcopy(row.get(key)) for key in (
            'output', 'episode_id', 'family', 'scene', 'seed', 'actor_sha256', 'valid', 'invalid_reasons',
            'outcome', 'costs', 'cost_issues', 'missing_cost_fields', 'steps')}
        summary['measured_row_sha256'] = record_hash(row)
        entry = {'job_id': job_id, 'job': job, 'result': summary, 'accepted': False}
        self.ledger.append(entry)
        self.revision += 1
        reasons = []
        if job is None or claim is None:
            reasons.append('Result has no matching persisted claim')
        else:
            if row.get('valid') is not True:
                reasons.append('Attempt is not an audited valid full episode')
            if row.get('output') != claim['output']:
                reasons.append('Result output differs from the persisted claim')
            for key in ('family', 'scene', 'seed', 'actor_sha256'):
                if row.get(key) != job[key]:
                    reasons.append('Result differs from the dispatched ' + key)
            outcome = row.get('outcome') or {}
            if outcome.get('valid') is not True or outcome.get('S_req') not in (0, 1):
                reasons.append('Outcome does not satisfy the frozen contract')
        if reasons:
            entry['binding_errors'] = reasons
            self.blocked_reasons.extend(reasons)
        else:
            entry['accepted'] = True
            self.results[job_id] = summary
        return self.report()

    def report(self):
        totals = defaultdict(float)
        for entry in self.ledger:
            for key, value in (entry['result'].get('costs') or {}).items():
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    totals[key] += value
        return {'status': self.status, 'expected_episode_slots': len(self.jobs),
                'physical_attempts_recorded': len(self.ledger), 'accepted_episode_slots': len(self.results),
                'in_flight_claims': deepcopy(self.claims), 'cost_totals': dict(totals),
                'blocked_reasons': list(self.blocked_reasons)}

    def state_dict(self):
        state = deepcopy({'version': VERSION, 'protocol_sha256': self.protocol['sha256'],
                          'claims': self.claims, 'results': self.results, 'ledger': self.ledger,
                          'blocked_reasons': self.blocked_reasons, 'revision': self.revision})
        return state | {'sha256': record_hash(state)}

    @classmethod
    def from_state_dict(cls, protocol, state):
        data = deepcopy(state)
        if data.pop('sha256') != record_hash(data) or data['version'] != VERSION:
            raise ValueError('Replicate state integrity mismatch')
        if data['protocol_sha256'] != protocol['sha256']:
            raise ValueError('Replicate state belongs to another protocol')
        coordinator = cls(protocol)
        for field in ('claims', 'results', 'ledger', 'blocked_reasons', 'revision'):
            setattr(coordinator, field, data[field])
        return coordinator

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            previous = json.loads(path.read_text())
            if self._last_saved_sha256 is None or previous.get('sha256') != self._last_saved_sha256:
                raise ValueError('State file changed or belongs to another coordinator; reload first')
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
    def load(cls, protocol, path):
        state = json.loads(Path(path).read_text())
        coordinator = cls.from_state_dict(protocol, state)
        coordinator._last_saved_sha256 = state['sha256']
        return coordinator


def wilson(successes, n, z=1.96):
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def required_per_arm(p, delta, *, power=0.8, alpha=0.05):
    """Two-sample binomial sample size per arm for a one-sided test of delta at base rate p."""
    from scipy.stats import norm

    q = min(p + delta, 1.0)
    z_alpha, z_beta = norm.ppf(1 - alpha), norm.ppf(power)
    pooled = (p + q) / 2
    return math.ceil(((z_alpha * math.sqrt(2 * pooled * (1 - pooled))
                       + z_beta * math.sqrt(p * (1 - p) + q * (1 - q))) / delta) ** 2)


def analyze(coordinator, pilot_vectors=None):
    """Prespecified descriptive analysis; accepts partial results and says so."""
    from scipy.stats import chi2_contingency, spearmanr

    protocol = coordinator.protocol
    rows = list(coordinator.results.values())
    by_controller = defaultdict(list)
    by_case = defaultdict(list)
    for job_id, row in coordinator.results.items():
        job = coordinator.jobs[job_id]
        by_controller[job['slot']].append(row['outcome'])
        by_case[job['scene']].append(row['outcome'])

    def rates(outcomes):
        n = len(outcomes)
        s = sum(int(o['S_req']) for o in outcomes)
        return {'episodes': n, 'safe_completion': s / n if n else None, 'safe_completion_ci': wilson(s, n),
                'completion': sum(int(o['C_req']) for o in outcomes) / n if n else None,
                'violation': sum(int(o['V']) for o in outcomes) / n if n else None,
                'mean_utility': sum(o['U'] for o in outcomes) / n if n else None}

    controllers = {str(slot): rates(by_controller[slot]) for slot in sorted(by_controller)}
    cases = {scene: rates(by_case[scene]) for scene in sorted(by_case)}
    table = [[sum(int(o['S_req']) for o in by_controller[s]), sum(1 - int(o['S_req']) for o in by_controller[s])]
             for s in sorted(by_controller)]
    homogeneity = None
    if len(table) >= 2 and all(sum(row) > 0 for row in table) and sum(r[0] for r in table) and sum(r[1] for r in table):
        stat, p_value, dof, _ = chi2_contingency(table)
        homogeneity = {'chi2': stat, 'p_value': p_value, 'dof': dof}
    correlations = {}
    means = [controllers[str(s)]['safe_completion'] for s in sorted(by_controller)]
    for name, vector in (pilot_vectors or {}).items():
        if len(vector) == len(means) and len(means) >= 3 and None not in means:
            rho = spearmanr(means, vector)
            correlations[name] = {'spearman_rho': float(rho.statistic), 'p_value': float(rho.pvalue)}
    pooled = sum(int(o['S_req']) for outcomes in by_controller.values() for o in outcomes)
    total = sum(len(outcomes) for outcomes in by_controller.values())
    base = pooled / total if total else None
    power = None if base is None else {str(delta): required_per_arm(base, delta) for delta in (0.1, 0.2, 0.3)}
    return {'schema': 'maniguard-replicate-analysis-v1', 'protocol_sha256': protocol['sha256'],
            'complete': coordinator.status == 'complete', 'status': coordinator.status,
            'episodes_analyzed': len(rows), 'expected_episodes': len(coordinator.jobs),
            'controllers': controllers, 'cases': cases, 'pooled_safe_completion': base,
            'controller_homogeneity': homogeneity, 'pilot_vector_correlations': correlations,
            'required_episodes_per_arm_for_80pct_power': power,
            'claims': ['Descriptive only. No controller is selected, ranked for use or carried forward.',
                       'Replicates share the frozen per-scene policy seed; they differ through renderer and physics nondeterminism, as observed in pilot replays.']}
