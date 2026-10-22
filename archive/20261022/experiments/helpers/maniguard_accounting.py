"""Inventory raw cost records, including failed and partial work, exactly once."""

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

from helpers.maniguard_batch_report import COSTS

COUNTS = {'physics_steps', 'env_step_attempts', 'env_steps_returned',
          'policy_queries', 'policy_control_steps'}


def checked_cost(value, field, path):
    if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
            or (field in COUNTS and type(value) is not int)):
        raise ValueError(f'Invalid recorded cost {field} in {path}')
    return value


def inventory_costs(directory):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError('The run directory must exist')
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    records, warnings, seen = [], [], set()
    totals = Counter()
    # Read raw records only. Derived evidence, reports and queue-process durations
    # often repeat the same costs and must not be added to these totals.
    paths = sorted([*directory.rglob('ledger.json'), *directory.rglob('check.json')])
    for path in paths:
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        relative = str(path.relative_to(directory))
        if identity in seen:
            warnings.append({'path': relative, 'issue': 'Duplicate file alias excluded'})
            continue
        seen.add(identity)
        raw = path.read_bytes()
        value = json.loads(raw)
        costs, issues = {}, []
        if path.name == 'ledger.json':
            kind, status = 'physical_episode', value.get('phase', 'unknown')
            for field in COSTS:
                if field in value:
                    costs[field] = checked_cost(value[field], field, path)
                else:
                    issues.append('Missing cost field: ' + field)
            if value.get('physics_accounting_attached') is not True:
                issues.append('Physics accounting attachment unverified; zero is not proof of no work')
            if not (path.parent / 'job.json').exists():
                issues.append('No associated episode job record')
        else:
            kind, status = 'inference_diagnostic', value['status']
            if type(value['physics_steps']) is not int or value['physics_steps'] != 0:
                raise ValueError(f'Expected inference-only diagnostic in {path}')
            queries = checked_cost(value['policy_queries_attempted'], 'policy_queries', path)
            timings = value['query_seconds']
            if not isinstance(timings, list) or len(timings) > queries:
                raise ValueError(f'Invalid diagnostic query timings in {path}')
            costs = {field: 0 for field in COSTS}
            costs['policy_queries'] = queries
            costs['policy_query_seconds'] = sum(
                checked_cost(t, 'policy_query_seconds', path) for t in timings)
            costs['elapsed_seconds'] = checked_cost(value['elapsed_seconds'], 'elapsed_seconds', path)
            if len(timings) < queries:
                issues.append('Not every attempted query has a recorded latency')
        totals.update(costs)
        records.append({'path': relative, 'sha256': hashlib.sha256(raw).hexdigest(),
                        'kind': kind, 'recorded_status': status, 'costs': costs, 'issues': issues})
    # A process can fail before instrumentation initializes. Keep these gaps visible.
    for path in sorted(directory.rglob('job.json')):
        if not (path.parent / 'ledger.json').exists():
            warnings.append({'path': str(path.relative_to(directory)),
                             'issue': 'Episode job has no cost ledger'})
    return {'run_directory': str(directory), 'snapshot_started_at': started,
            'snapshot_finished_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'records': records, 'record_types': dict(Counter(r['kind'] for r in records)),
            'recorded_cost_totals': dict(totals), 'warnings': warnings,
            'scope': ('Inventory of raw ledger.json and inference check.json files under this '
                      'directory. Failed and partial records count. Historical omissions must '
                      'be reported separately; recorded totals are not a completeness guarantee.'),
            'interpretation': ('A snapshot of running jobs is not atomic. Status fields are '
                               'recorded phases, not verified live handles or validated outcomes. '
                               'Elapsed seconds sum episode/diagnostic durations, not study wall '
                               'time; policy-query latency can include server queueing. Derived '
                               'reports and queue durations are excluded to avoid double counting.')}
