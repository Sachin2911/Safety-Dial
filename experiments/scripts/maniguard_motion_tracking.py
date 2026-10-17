"""Export command/state audits for all validated completed motion-comparison trials."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from helpers.maniguard_batch_report import summarize_batch
from helpers.maniguard_motion_tracking import audit_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output path to preserve previous tracking audits')
    batch = summarize_batch(args.batch)
    rows = [audit_episode(e['output']) for e in batch['episodes'] if e['status'] == 'completed']
    if not rows:
        raise ValueError('No validated completed episodes are available for audit')
    sources = ('experiments/scripts/maniguard_motion_tracking.py',
               'experiments/helpers/maniguard_motion_tracking.py',
               'experiments/helpers/maniguard_batch_report.py',
               'experiments/helpers/maniguard_results.py')
    report = {
        'recorded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'scope': (f'{len(rows)} of {batch["planned"]} planned episodes completed and audited. '
                  'Descriptive command and measured-state checks; no new simulator actions '
                  'or policy queries. Incomplete episodes are excluded from these statistics.'),
        'batch': str(args.batch), 'batch_statuses': batch['statuses'],
        'plan_sha256': hashlib.sha256((args.batch / 'plan.json').read_bytes()).hexdigest(),
        'source_hashes': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'method': {
            'arm_joints': 7,
            'units': 'radians per logged policy action, maximum absolute arm-joint component',
            'command_step': 'max(abs(executed_target - proposal.frame.joints))',
            'measured_step': 'max(abs(observed.state[:7] - proposal.frame.joints))',
            'alignment': ('Proposal and execution index t precede observation index t+1. '
                          'Post-action joints equal the next proposal frame exactly; adapted '
                          'and executed arm targets match exactly.'),
            'command_cap_tolerance_rad': 1e-6,
            'active_cap': 'Grasp cap for grasp_evidence with a closing proposal; free cap otherwise.',
            'quantile': 'numpy.quantile(measured_steps, 0.95), default linear interpolation',
        },
        'episodes': rows,
        'interpretation': ('Command caps bound target displacement, not measured joint motion. '
                           'A measured cap exceedance is descriptive and does not identify a '
                           'failure cause. Different trajectories do not establish causal speed '
                           'or safety effects; unfinished cases are not completions.'),
    }
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'audited': len(rows), 'batch_statuses': batch['statuses'],
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
