"""Resumable pilot coordination over audited results, without process launching.

Typical use::

    coordinator = PilotCoordinator(frozen_protocol)
    job = coordinator.next_jobs()[0]
    coordinator.mark_started(job['job_id'], unique_output_path)
    coordinator.save(state_path)  # persist the claim before launching
    # External launcher executes job, then read_episode audits the output.
    coordinator.record_result(job['job_id'], audited_row)
    coordinator.save(state_path)

A restart retains in-flight claims. Reconcile their output directories before launching
anything else; there is no automatic replay, retry, or reuse of a physical episode.
"""

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from helpers.maniguard_phase2_families import CONFIG, ROOT
from helpers.maniguard_phase2_model import genome_hash, identity_genome
from helpers.maniguard_phase2_optim import (
    PPOConfig,
    RecurrentPPO,
    SearchConfig,
    WeightSearch,
    record_hash,
)
from helpers.maniguard_phase2_protocol import compile_schedule, validate_protocol

VERSION = 'maniguard-phase2-coordinator-v1'
RESULT_KEYS = (
    'output', 'episode_id', 'family', 'scene', 'seed', 'sampling_seed', 'actor_sha256', 'mode',
    'fixed_choices', 'stochastic', 'purpose', 'action_horizon', 'short_check', 'full_episode',
    'valid', 'invalid_reasons', 'benchmark_result', 'outcome', 'ppo_eligible', 'costs',
    'cost_issues', 'missing_cost_fields', 'cost_provenance', 'observed_work', 'steps',
)


def _run_key(method, seed):
    return f'{method}/s{seed}'


def _source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _summary(row):
    result = {key: deepcopy(row.get(key)) for key in RESULT_KEYS}
    if row.get('trajectory') is not None:
        result['trajectory_sha256'] = record_hash(row['trajectory'])
    result['measured_row_sha256'] = record_hash(row)
    return result


def _numeric_cost(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _metrics(rows, families):
    """Use equal-family means; missing observations are never zero-valued outcomes."""
    groups = {family: [row for row in rows if row['family'] == family] for family in families}
    if any(not group for group in groups.values()):
        raise ValueError('Cannot select a controller with a missing validation family')
    metrics = {}
    for name in ('U', 'V', 'S_req'):
        metrics[name] = float(np.mean([
            np.mean([row['outcome'][name] for row in groups[family]]) for family in families]))
    metrics['policy_queries'] = float(np.mean([
        np.mean([row['costs']['policy_queries'] for row in groups[family]])
        for family in families]))
    metrics['episodes'] = len(rows)
    metrics['family_episodes'] = {family: len(group) for family, group in groups.items()}
    return metrics


class PilotCoordinator:
    """Pure episode scheduling and measured-result transitions for one frozen pilot.

    ``next_jobs`` returns only unclaimed jobs in the active dependency block. Each job
    contains a 692-weight genome and canonical hash. ``mark_started`` reserves an output
    directory before any external process starts. ``record_result`` retains cost evidence
    even when the attempt is invalid, mismatched or unsuccessful; invalid attempts block
    the study instead of being assigned fitness. Duplicate physical records are rejected.
    """

    def __init__(self, protocol, *, config_dir=CONFIG, root=ROOT):
        audit = validate_protocol(protocol, config_dir, root, require_frozen=True)
        if audit['ready_to_run'] is not True:
            raise ValueError('The full pilot protocol must be frozen before coordination')
        self.protocol = json.loads(json.dumps(protocol, allow_nan=False))
        self.config_dir = str(Path(config_dir).resolve())
        self.root = str(Path(root).resolve())
        self.templates = compile_schedule(protocol)
        self.stage_plan = self._make_stage_plan()
        panels = tuple(panel['panel_id'] for panel in protocol['training_panels'])
        self.searches = {
            _run_key(method, seed): WeightSearch(method, seed, panels,
                                                SearchConfig(**protocol['search_config']))
            for method in ('evolution', 'random') for seed in protocol['optimizer_seeds']}
        self.learners = {
            _run_key('ppo', seed): RecurrentPPO(seed=seed, config=PPOConfig(**protocol['ppo_config']))
            for seed in protocol['optimizer_seeds']}
        self.stage_index = 0
        self.active_jobs = None
        self.active_job_order = []
        self.active_results = {}
        self.claims = {}
        self.ledger = []
        self.history = []
        self.nominees = {}
        self.selected = {}
        self.reference_metrics = {}
        self.blocked_reasons = []
        self.revision = 0
        self._last_saved_sha256 = None

    def _make_stage_plan(self):
        rng = np.random.default_rng(self.protocol['execution_order']['seed'])
        stages = []
        for seed in self.protocol['optimizer_seeds']:
            for panel in range(3):
                for method in rng.permutation(self.protocol['methods']).tolist():
                    if method == 'ppo':
                        for repetition in range(8):
                            stages.append({'kind': 'ppo', 'method': 'ppo', 'seed': seed,
                                           'panel': panel, 'batch': panel * 8 + repetition})
                    else:
                        stages.append({'kind': 'search', 'method': method, 'seed': seed,
                                       'panel': panel})
        runs = [(method, seed) for method in self.protocol['methods']
                for seed in self.protocol['optimizer_seeds']]
        for index in rng.permutation(len(runs)):
            method, seed = runs[int(index)]
            stages.append({'kind': 'validation', 'method': method, 'seed': seed})
        stages.append({'kind': 'references'})
        return stages

    @property
    def status(self):
        if self.blocked_reasons:
            return 'blocked'
        if self.stage_index == len(self.stage_plan):
            return 'complete'
        return 'training' if self.stage_plan[self.stage_index]['kind'] in ('search', 'ppo') else (
            'validation')

    def _jobs_for_stage(self, stage):
        kind = stage['kind']
        if kind in ('search', 'ppo'):
            jobs = [deepcopy(job) for job in self.templates if job['stage'] == 'training' and
                    job['method'] == stage['method'] and job['optimizer_seed'] == stage['seed'] and
                    job['round'] == stage['panel']]
            if kind == 'ppo':
                return [job for job in jobs if job['ppo_batch'] == stage['batch']]
            return jobs
        if kind == 'validation':
            return [deepcopy(job) for job in self.templates if job['stage'] == 'finalist_validation'
                    and job['method'] == stage['method'] and job['optimizer_seed'] == stage['seed']]
        return [deepcopy(job) for job in self.templates if job['stage'] == 'reference_validation']

    def _prepare_stage(self):
        stage = self.stage_plan[self.stage_index]
        jobs = self._jobs_for_stage(stage)
        kind = stage['kind']
        if kind == 'search':
            search = self.searches[_run_key(stage['method'], stage['seed'])]
            if search.round_index != stage['panel']:
                raise ValueError('Search optimizer and scheduled panel disagree')
            candidates = search.ask()
        elif kind == 'ppo':
            learner = self.learners[_run_key('ppo', stage['seed'])]
            if learner.episodes_seen != stage['batch'] * 3:
                raise ValueError('PPO episode ledger and scheduled batch disagree')
        for job in jobs:
            if kind == 'search':
                genome = candidates[job['candidate_slot']]['genome']
            elif kind == 'ppo':
                genome = learner.actor.export_genome().tolist()
            elif kind == 'validation':
                genome = self.nominees[_run_key(stage['method'], stage['seed'])][
                    job['finalist']]['genome']
            else:
                genome = identity_genome().tolist()
            job.update(genome=deepcopy(genome), actor_sha256=genome_hash(genome), mode='learned',
                       fixed_choices=None, stochastic=kind == 'ppo', purpose=job['split'],
                       sampling_seed=(job['optimizer_seed'] * 1000 + job['ppo_episode']
                                      if kind == 'ppo' else 0))
            if kind == 'references':
                reference = next(row for row in self.protocol['references']
                                 if row['name'] == job['method'])
                if job['method'] == 'contextual':
                    job['mode'] = 'contextual'
                else:
                    job['mode'] = 'fixed'
                    job['fixed_choices'] = reference.get('selected_choices', reference.get(
                        'choices', reference.get('proposed_choices')))
                    if job['fixed_choices'] is None:
                        raise ValueError('Fixed reference choices must be frozen')
        # A recurrent learner consumes exact Jar/Stack/Clutter triplets. Other blocks
        # randomize their recorded dispatch order without looking at outcomes.
        if kind != 'ppo':
            rng = np.random.default_rng(self.protocol['execution_order']['seed'] + self.stage_index)
            jobs = [jobs[int(index)] for index in rng.permutation(len(jobs))]
        self.active_jobs = {job['job_id']: job for job in jobs}
        self.active_job_order = [job['job_id'] for job in jobs]
        self.active_results = {}
        self.revision += 1

    def next_jobs(self):
        if self.status in ('blocked', 'complete'):
            return []
        if self.active_jobs is None:
            self._prepare_stage()
        return [deepcopy(self.active_jobs[job_id]) for job_id in self.active_job_order
                if job_id not in self.active_results and job_id not in self.claims]

    def mark_started(self, job_id, output):
        if self.status in ('blocked', 'complete') or self.active_jobs is None:
            raise ValueError('There is no runnable active job')
        if job_id not in self.active_jobs or job_id in self.claims or job_id in self.active_results:
            raise ValueError('Job is unknown, already claimed or already recorded')
        output = str(Path(output).resolve())
        used = {claim['output'] for claim in self.claims.values()} | {
            row['result']['output'] for row in self.ledger}
        if output in used:
            raise ValueError('Every physical attempt requires a unique output directory')
        self.claims[job_id] = {'output': output, 'stage_index': self.stage_index}
        self.revision += 1
        return deepcopy(self.claims[job_id])

    def _bind_result(self, job, row, claim):
        reasons = []
        if row.get('valid') is not True or row.get('full_episode') is not True or row.get('short_check'):
            reasons.append('Fitness requires an audited valid full episode')
        fields = {'family': 'family', 'scene': 'scene', 'seed': 'seed',
                  'sampling_seed': 'sampling_seed', 'actor_sha256': 'actor_sha256', 'mode': 'mode',
                  'stochastic': 'stochastic', 'purpose': 'purpose', 'action_horizon': 'max_steps'}
        for key, expected_key in fields.items():
            if row.get(key) != job[expected_key]:
                reasons.append(f'Result differs from dispatched {key}')
        if row.get('output') != claim['output']:
            reasons.append('Result output differs from the persisted launch claim')
        if job['mode'] == 'fixed' and row.get('fixed_choices') != job['fixed_choices']:
            reasons.append('Fixed reference choices differ from the frozen job')
        if row.get('cost_issues') or row.get('missing_cost_fields'):
            reasons.append('Recorded costs are incomplete or inconsistent')
        costs = row.get('costs') or {}
        if not costs or any(not _numeric_cost(value) for value in costs.values()) or not (
                _numeric_cost(costs.get('policy_queries'))):
            reasons.append('Finite nonnegative measured costs, including policy_queries, are required')
        outcome = row.get('outcome') or {}
        if (outcome.get('valid') is not True or outcome.get('C_req') not in (0, 1) or
                outcome.get('S_req') not in (0, 1) or outcome.get('V') not in (0, 1) or
                outcome.get('S_req') != outcome.get('C_req', 0) * (1 - outcome.get('V', 0)) or
                outcome.get('U') != outcome.get('S_req', 0) - 2 * outcome.get('V', 0)):
            reasons.append('Outcome does not establish the frozen measured terminal utility')
        if job['method'] == 'ppo' and job['stage'] == 'training':
            trajectory = row.get('trajectory')
            if row.get('ppo_eligible') is not True or not isinstance(trajectory, dict):
                reasons.append('PPO requires an audited stochastic on-policy trajectory')
            elif (trajectory.get('actor_sha256') != job['actor_sha256'] or
                  trajectory.get('family') != job['family'] or
                  trajectory.get('terminal_utility') != outcome.get('U') or
                  trajectory.get('episode_id') != row.get('episode_id')):
                reasons.append('PPO trajectory metadata differs from the audited physical result')
        return reasons

    def record_result(self, job_id, row):
        """Retain a measured attempt, then score it only after all binding checks pass.

        Returns a status report. Invalid/misrouted attempts preserve costs and block
        advancement. Repeated output/episode identifiers raise before double accounting.
        """
        if self.status == 'complete':
            raise ValueError('The pilot already completed')
        output = row.get('output')
        episode_id = row.get('episode_id')
        if not isinstance(output, str) or not isinstance(episode_id, str):
            raise TypeError('Every received attempt needs a stable output and episode identifier')
        if any(entry['result']['output'] == output or entry['result']['episode_id'] == episode_id
               for entry in self.ledger):
            raise ValueError('Physical episode already recorded; duplicate costs and fitness rejected')
        summary = _summary(row)
        job = None if self.active_jobs is None else self.active_jobs.get(job_id)
        claim = self.claims.get(job_id)
        entry = {'job_id': job_id, 'stage_index': self.stage_index,
                 'job': ({key: deepcopy(value) for key, value in job.items() if key != 'genome'}
                         if job is not None else None), 'result': summary, 'accepted': False}
        self.ledger.append(entry)
        self.revision += 1
        if job is None or claim is None:
            self.blocked_reasons.append('Received result has no matching active persisted launch claim')
            return self.report()
        if job_id in self.active_results:
            self.blocked_reasons.append('A second physical attempt was received for an already filled slot')
            return self.report()
        reasons = self._bind_result(job, row, claim)
        if reasons:
            self.blocked_reasons.extend(reasons)
            entry['binding_errors'] = reasons
            del self.claims[job_id]
            return self.report()
        # Drain valid in-flight results after another attempt failed, retaining their costs
        # without applying a partial optimizer update or starting another stage.
        entry['accepted'] = True
        self.active_results[job_id] = (deepcopy(row) if job['method'] == 'ppo' and
                                      job['stage'] == 'training' else summary)
        del self.claims[job_id]
        if not self.blocked_reasons and len(self.active_results) == len(self.active_jobs):
            try:
                self._complete_stage()
            except (ValueError, RuntimeError, KeyError, TypeError) as error:
                self.blocked_reasons.append(f'Optimizer/selection update failed: {error}')
        return self.report()

    def _complete_stage(self):
        stage = self.stage_plan[self.stage_index]
        kind = stage['kind']
        rows = [self.active_results[job_id] for job_id in self.active_job_order]
        history = {'stage_index': self.stage_index, 'stage': deepcopy(stage), 'episodes': len(rows)}
        if kind == 'search':
            fitness = []
            for slot in range(8):
                values = [self.active_results[job_id] for job_id, job in self.active_jobs.items()
                          if job['candidate_slot'] == slot]
                if Counter(row['family'] for row in values) != dict.fromkeys(self.protocol['families'], 1):
                    raise ValueError('Search slot is missing a matched family result')
                fitness.append(_metrics(values, self.protocol['families'])['U'])
            key = _run_key(stage['method'], stage['seed'])
            search = self.searches[key]
            search.tell(fitness, evidence=[row['measured_row_sha256'] for row in rows])
            history['fitness'] = fitness
            if stage['panel'] == 2:
                self.nominees[key] = search.finalists()
        elif kind == 'ppo':
            key = _run_key('ppo', stage['seed'])
            history['update'] = self.learners[key].update([row['trajectory'] for row in rows])
            if self.learners[key].episodes_seen == 72:
                self.nominees[key] = [deepcopy(self.learners[key].checkpoints[str(n)])
                                      for n in (48, 72)]
        elif kind == 'validation':
            key = _run_key(stage['method'], stage['seed'])
            candidates = []
            for finalist in range(2):
                values = [self.active_results[job_id] for job_id, job in self.active_jobs.items()
                          if job['finalist'] == finalist]
                if Counter(row['family'] for row in values) != dict.fromkeys(self.protocol['families'], 2):
                    raise ValueError('Finalist validation must use exactly two cases per family')
                candidates.append({'finalist': finalist,
                                   'metrics': _metrics(values, self.protocol['families'])})
            order = sorted(candidates, key=lambda value: (
                -value['metrics']['U'], value['metrics']['V'], value['metrics']['policy_queries'],
                value['finalist']))
            chosen = order[0]
            genome = self.nominees[key][chosen['finalist']]['genome']
            self.selected[key] = {'finalist': chosen['finalist'], 'genome': deepcopy(genome),
                                  'actor_sha256': genome_hash(genome), 'metrics': chosen['metrics'],
                                  'validation_candidates': candidates}
            history['selection'] = deepcopy(self.selected[key])
        else:
            for name in ('identity', 'fixed', 'contextual'):
                values = [self.active_results[job_id] for job_id, job in self.active_jobs.items()
                          if job['method'] == name]
                self.reference_metrics[name] = _metrics(values, self.protocol['families'])
            history['reference_metrics'] = deepcopy(self.reference_metrics)
        self.history.append(history)
        self.stage_index += 1
        self.active_jobs = None
        self.active_job_order = []
        self.active_results = {}
        self.revision += 1

    def report(self):
        totals = defaultdict(float)
        observed = defaultdict(float)
        cost_incomplete = []
        for entry in self.ledger:
            row = entry['result']
            for name, value in (row.get('costs') or {}).items():
                if _numeric_cost(value):
                    totals[name] += value
            for name, value in (row.get('observed_work') or {}).items():
                if _numeric_cost(value):
                    observed[name] += value
            if row.get('cost_issues') or row.get('missing_cost_fields'):
                cost_incomplete.append(row['output'])
        return {'status': self.status, 'completed_stages': self.stage_index,
                'total_stages': len(self.stage_plan), 'expected_episode_slots': len(self.templates),
                'physical_attempts_recorded': len(self.ledger),
                'accepted_episode_slots': sum(row['accepted'] for row in self.ledger),
                'in_flight_claims': deepcopy(self.claims), 'cost_totals': dict(totals),
                'observed_work_lower_bounds': dict(observed),
                'incomplete_cost_records': cost_incomplete,
                'blocked_reasons': list(self.blocked_reasons),
                'nominee_runs': sorted(self.nominees), 'selected_runs': sorted(self.selected),
                'reference_metrics': deepcopy(self.reference_metrics),
                'simulator_launched_by_coordinator': False}

    def state_dict(self):
        state = {'version': VERSION, 'source_sha256': _source_hash(), 'protocol': self.protocol,
                 'config_dir': self.config_dir, 'root': self.root, 'stage_plan': self.stage_plan,
                 'stage_index': self.stage_index, 'active_jobs': self.active_jobs,
                 'active_job_order': self.active_job_order,
                 'active_results': self.active_results, 'claims': self.claims,
                 'ledger': self.ledger, 'history': self.history, 'nominees': self.nominees,
                 'selected': self.selected, 'reference_metrics': self.reference_metrics,
                 'blocked_reasons': self.blocked_reasons, 'revision': self.revision,
                 'searches': {key: search.state_dict() for key, search in self.searches.items()},
                 'learners': {key: learner.state_dict() for key, learner in self.learners.items()}}
        state = deepcopy(state)
        return state | {'sha256': record_hash(state)}

    @classmethod
    def from_state_dict(cls, state):
        data = deepcopy(state)
        digest = data.pop('sha256')
        if (digest != record_hash(data) or data['version'] != VERSION or
                data['source_sha256'] != _source_hash()):
            raise ValueError('Coordinator state/source integrity mismatch')
        coordinator = cls(data['protocol'], config_dir=data['config_dir'], root=data['root'])
        if coordinator.stage_plan != data['stage_plan']:
            raise ValueError('Coordinator dependency schedule changed')
        for field in ('stage_index', 'active_jobs', 'active_job_order', 'active_results', 'claims',
                      'ledger', 'history',
                      'nominees', 'selected', 'reference_metrics', 'blocked_reasons', 'revision'):
            setattr(coordinator, field, data[field])
        coordinator.searches = {key: WeightSearch.from_state_dict(value)
                                for key, value in data['searches'].items()}
        coordinator.learners = {key: RecurrentPPO.from_state_dict(value)
                                for key, value in data['learners'].items()}
        if (coordinator.stage_index != len(coordinator.history) or
                not 0 <= coordinator.stage_index <= len(coordinator.stage_plan)):
            raise ValueError('Coordinator stage ledger is inconsistent')
        outputs = [entry['result']['output'] for entry in coordinator.ledger]
        episodes = [entry['result']['episode_id'] for entry in coordinator.ledger]
        if len(set(outputs)) != len(outputs) or len(set(episodes)) != len(episodes):
            raise ValueError('Coordinator contains duplicated physical episodes')
        if coordinator.claims and coordinator.active_jobs is None:
            raise ValueError('In-flight claims require an active stage')
        if (len(coordinator.active_job_order) != len(set(coordinator.active_job_order)) or
                set(coordinator.active_job_order) != set(coordinator.active_jobs or {})):
            raise ValueError('Saved dispatch order differs from the active jobs')
        for job in (coordinator.active_jobs or {}).values():
            if genome_hash(job['genome']) != job['actor_sha256']:
                raise ValueError('Pending actor hash mismatch')
        return coordinator

    def save(self, path):
        """Atomically persist state; reject stale writers and unrelated existing files."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            previous = json.loads(path.read_text())
            if self._last_saved_sha256 is None or previous.get('sha256') != self._last_saved_sha256:
                raise ValueError('State file changed or belongs to another coordinator; reload first')
        elif self._last_saved_sha256 is not None:
            raise ValueError('Previously saved state file disappeared; do not silently recreate it')
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
    def load(cls, path):
        state = json.loads(Path(path).read_text())
        coordinator = cls.from_state_dict(state)
        coordinator._last_saved_sha256 = state['sha256']
        return coordinator
