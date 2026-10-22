"""Audit planned episode coverage and recorded costs without dropping failed work."""

import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from helpers.maniguard_adaptation import AdapterConfig
from helpers.maniguard_results import read_episode

OUTCOMES = ('safe_completion', 'unsafe_completion', 'safe_unfinished', 'unsafe_unfinished')
COSTS = ('physics_steps', 'physics_seconds', 'env_step_attempts', 'env_steps_returned',
         'policy_queries', 'policy_query_seconds', 'policy_control_steps', 'elapsed_seconds')


def summarize_batch(directory):
    directory = Path(directory)
    plan = json.loads((directory / 'plan.json').read_text())
    process_path = directory / 'processes.jsonl'
    processes = {}
    if process_path.exists():
        for line in process_path.read_text().splitlines():
            record = json.loads(line)
            index = record['index']
            if index in processes or not 0 <= index < len(plan['jobs']):
                raise ValueError('Duplicate or out-of-plan process index')
            processes[index] = record
    groups = {}
    episodes = []
    recorded_costs = Counter()
    missing_cost_fields = Counter()
    physics_attachment_unverified = 0
    for index, job in enumerate(plan['jobs']):
        label = job.get('label', job['adapter']['kind'])
        key = (job['family'], label)
        config = asdict(AdapterConfig(**job['adapter']))
        if key not in groups:
            groups[key] = {'family': key[0], 'label': key[1], 'adapter': config,
                           'planned': 0, 'statuses': Counter(), 'outcomes': Counter()}
        group = groups[key]
        if config != group['adapter']:
            raise ValueError('A batch label mixes different adapter configurations')
        group['planned'] += 1
        output = directory / f'episode-{index:04d}'
        record = {'index': index, 'family': job['family'], 'scene': job['scene'],
                  'seed': job['seed'], 'label': label, 'output': str(output)}
        if output.exists():
            ledger_path = output / 'ledger.json'
            ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
            physics_attachment_unverified += ledger.get('physics_accounting_attached') is not True
            for field in COSTS:
                if field in ledger:
                    value = ledger[field]
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError(f'Invalid recorded cost {field} in {output}')
                    recorded_costs[field] += value
                else:
                    missing_cost_fields[field] += 1
            record['recorded_ledger'] = ledger
        process = processes.get(index)
        if process is None:
            record['status'] = 'awaiting_process_record' if output.exists() else 'not_started'
        elif process.get('returncode') != 0 or process.get('validation_error'):
            record['status'] = 'process_or_validation_failed'
            record['process'] = process
        else:
            try:
                row = read_episode(output)
                expected = (job['family'], job['scene'], job['seed'], job.get('max_steps'))
                actual = (row['family'], row['scene'], row['seed'], row['engineering_override'])
                if actual != expected or asdict(AdapterConfig(**row['adapter'])) != config:
                    raise ValueError('Physical result does not match the declared job')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record['status'] = 'invalid_result'
                record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            else:
                record['status'] = 'completed'
                record['result'] = row
                group['outcomes'].update({name: int(row[name]) for name in OUTCOMES})
        group['statuses'][record['status']] += 1
        episodes.append(record)
    statuses = Counter(record['status'] for record in episodes)
    marker = directory / 'complete.json'
    if marker.exists():
        declared = json.loads(marker.read_text())['episodes']
        if declared != len(plan['jobs']) or statuses['completed'] != declared:
            raise ValueError('Batch completion marker disagrees with validated planned work')
    return {'batch': str(directory.resolve()), 'split': plan['split'],
            'planned': len(plan['jobs']), 'statuses': dict(statuses),
            'groups': list(groups.values()), 'episodes': episodes,
            'recorded_cost_totals': dict(recorded_costs),
            'started_episodes_missing_cost_fields': dict(missing_cost_fields),
            'started_episodes_with_unverified_physics_attachment': physics_attachment_unverified,
            'finished_processes_missing_elapsed_seconds': sum('elapsed_seconds' not in p
                                                               for p in processes.values()),
            'finished_process_elapsed_seconds_sum': sum(p.get('elapsed_seconds', 0)
                                                        for p in processes.values()),
            'accounting_scope': ('Recorded episode costs include failed and incomplete work. '
                                 'Missing fields and unattached physics are explicit; totals may be '
                                 'partial. Elapsed seconds are sums across episodes, not batch wall time.'),
            'interpretation': ('Outcome counts cover only validated completed episodes; all other '
                               'planned jobs remain in statuses. A missing process record does not '
                               'prove a process is live. This report makes no causal or safety guarantee.')}
