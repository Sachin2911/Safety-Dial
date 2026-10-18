"""Write a read-only comparison of a concurrent-lane pilot, complete or not.

The frozen ``build_report`` reconstructs every finished selection and comparison from
the lane run's inner frozen-format state. This script adds what it cannot see: in-flight
lane claims, per-lane block progress, search fitness by round, voided attempts whose
measured interactions must be added to every total, and allocated GPU-hours.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'experiments/scripts'))

from maniguard_pilot_report import markdown

from helpers.maniguard_phase2_protocol import load_protocol
from helpers.maniguard_pilot_lanes import LaneCoordinator
from helpers.maniguard_pilot_parallel import AllocationClock
from helpers.maniguard_pilot_report import build_report

RUN = ROOT / 'runs/maniguard/phase2/pilot-v1'
CONFIG = ROOT / 'configs/maniguard/pilot-migration-20261018.json'


def void_ledger(run):
    receipts, totals = [], defaultdict(float)
    for path in sorted(run.glob('void-attempts/*/void.json')):
        receipt = json.loads(path.read_text())
        costs = receipt.get('voided_costs', {})
        for key, value in costs.items():
            totals[key] += value
        receipts.append({'receipt': str(path.relative_to(ROOT)), 'jobs': receipt['job_ids'],
                         'mode': receipt.get('mode', 'restore'), 'costs': costs,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return receipts, dict(totals)


def lane_progress(lanes, state):
    progress = {}
    for key, indices in lanes.lanes.items():
        blocks = []
        for index in indices:
            stage = lanes.base.stage_plan[index]
            record = lanes.stages[index]
            history = next((entry for entry in state['history'] if entry['stage_index'] == index), None)
            blocks.append({'stage_index': index, 'kind': stage['kind'], 'panel': stage.get('panel'),
                           'batch': stage.get('batch'), 'status': record['status'],
                           'results': len(record['active_results'] or {}) if record['status'] == 'active'
                           else (history or {}).get('episodes'),
                           'fitness': (history or {}).get('fitness'),
                           'completed_at': record.get('completed_at')})
        progress[key] = {'complete_blocks': sum(block['status'] == 'complete' for block in blocks),
                         'blocks': blocks}
    return progress


def throughput(run):
    """Measured process times of audited lane-deployment attempts, per family."""
    rows = []
    for path in sorted(run.glob('deployments/lanes-*/attempts/**/*.json')):
        attempt = json.loads(path.read_text())
        process = attempt.get('process') or {}
        if attempt.get('status') == 'audited' and process.get('process_elapsed_seconds'):
            rows.append((attempt['job']['family'], process['started_at'], process['finished_at'],
                         process['process_elapsed_seconds'], attempt.get('valid'),
                         (attempt.get('costs') or {}).get('policy_control_steps')))
    if not rows:
        return {}
    span = (max(row[2] for row in rows) - min(row[1] for row in rows)) / 3600
    families = {}
    for family in sorted({row[0] for row in rows}):
        times = sorted(row[3] for row in rows if row[0] == family)
        families[family] = {'episodes': len(times), 'median_seconds': times[len(times) // 2],
                            'mean_seconds': sum(times) / len(times),
                            'valid': sum(row[4] is True for row in rows if row[0] == family)}
    return {'audited_attempts': len(rows), 'span_hours': span,
            'episodes_per_hour': len(rows) / span if span > 0 else None, 'families': families}


def table(report, progress, voids, void_totals, budget, rates=None):
    allocated = (f"Allocated GPU-hours: **{budget['allocated_gpu_seconds'] / 3600:.2f}** of the frozen 72 "
                 f"(snapshot {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(budget['timestamp']))}).")
    lines = [markdown(report), '', '## Concurrent-lane run status', '', allocated, '',
             '| Lane | Complete blocks | Current block |', '| --- | ---: | --- |']
    for key, lane in progress.items():
        current = next((block for block in lane['blocks'] if block['status'] != 'complete'), None)
        where = '' if current is None else (
            f"batch {current['batch']}" if current['kind'] == 'ppo' else f"panel {current['panel']}")
        label = 'done' if current is None else (
            f"{current['kind']} {where} ({current['status']}, {current['results'] or 0} results)")
        lines.append(f"| {key} | {lane['complete_blocks']}/{len(lane['blocks'])} | {label} |")
    lines += ['', 'Search fitness by panel (family-macro utility of the eight candidates):', '']
    for key, lane in progress.items():
        for block in lane['blocks']:
            if block['fitness'] is not None:
                values = ', '.join(f'{value:.2f}' for value in block['fitness'])
                lines.append(f"- {key} panel {block['panel']}: {values}")
    note = ('These attempts are excluded from scoring by recorded decisions; their measured '
            'interactions are part of every cost total below.')
    lines += ['', '## Voided attempts', '', note, '']
    for receipt in voids:
        lines.append(f"- `{receipt['receipt']}` ({receipt['mode']}): {len(receipt['jobs'])} attempts, "
                     f"{int(receipt['costs'].get('policy_control_steps', 0))} policy actions, "
                     f"{int(receipt['costs'].get('policy_queries', 0))} queries")
    if rates:
        lines += ['', '## Measured throughput (lane deployments)', '',
                  f"{rates['audited_attempts']} audited attempts over {rates['span_hours']:.2f} h: "
                  f"{rates['episodes_per_hour']:.1f} episodes per hour on eight RTX 3090s.", '',
                  '| Family | Episodes | Valid | Median process s | Mean process s |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for family, row in rates['families'].items():
            lines.append(f"| {family} | {row['episodes']} | {row['valid']} | "
                         f"{row['median_seconds']:.0f} | {row['mean_seconds']:.0f} |")
    recorded = report['costs']['recorded_totals']
    lines += ['', '| Counter | Scored ledger | Voided | Total |', '| --- | ---: | ---: | ---: |']
    for key in ('policy_control_steps', 'policy_queries', 'physics_steps', 'elapsed_seconds'):
        a, b = recorded.get(key, 0), void_totals.get(key, 0)
        lines.append(f'| {key} | {a:.0f} | {b:.0f} | {a + b:.0f} |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Report output already exists')
    config = json.loads(CONFIG.read_text())
    lanes = LaneCoordinator.load(RUN / 'lanes.json', priority=config.get('lane_priority'))
    state = lanes.state_dict()['base']
    report = build_report(state)
    protocol = load_protocol()
    pin = protocol['freeze_provenance']['prior_accounting']
    prior = json.loads((ROOT / pin['evidence_path']).read_text())
    budget = AllocationClock(prior, config['allocations'], protocol['resource_envelope']).snapshot(
        report['costs']['recorded_totals'])
    voids, void_totals = void_ledger(RUN)
    progress = lane_progress(lanes, state)
    combined = {key: report['costs']['recorded_totals'].get(key, 0) + void_totals.get(key, 0)
                for key in set(report['costs']['recorded_totals']) | set(void_totals)}
    summary = {'schema': 'maniguard-pilot-partial-report-v1', 'created_at': time.time(),
               'lanes_state_sha256': json.loads((RUN / 'lanes.json').read_text())['sha256'],
               'frozen_report': report, 'in_flight_claims': lanes.base.claims,
               'lane_progress': progress, 'void_receipts': voids, 'void_costs': void_totals,
               'costs_including_voids': combined, 'allocation': budget,
               'lane_priority': lanes.priority, 'takeover': lanes.takeover,
               'throughput': throughput(RUN)}
    args.output.mkdir(parents=True)
    (args.output / 'report.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    (args.output / 'report.md').write_text(table(report, progress, voids, void_totals, budget,
                                                 summary['throughput']))
    print(json.dumps({'decision': report['screening']['decision'], 'complete': report['complete'],
                      'valid': report['valid'], 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
