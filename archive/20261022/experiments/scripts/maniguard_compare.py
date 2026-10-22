"""Export audited scene/seed comparisons against a declared control condition."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from helpers.maniguard_comparison import compare_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--control', default='identity')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output path to preserve previous comparison snapshots')
    report = compare_batch(args.batch, args.control)
    report['recorded_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    paths = [Path(__file__), ROOT / 'experiments/helpers/maniguard_comparison.py',
             ROOT / 'experiments/helpers/maniguard_batch_report.py',
             ROOT / 'experiments/helpers/maniguard_results.py']
    report['source_hashes'] = {str(path.relative_to(ROOT)): hashlib.sha256(
        path.read_bytes()).hexdigest() for path in paths}
    report['plan_sha256'] = hashlib.sha256((args.batch / 'plan.json').read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'batch_statuses': report['batch_report']['statuses'],
                      'comparisons': [{k: row[k] for k in (
                          'candidate_label', 'planned_pairs', 'completed_pairs',
                          'safe_completion_gains', 'safe_completion_losses',
                      )} for row in report['comparisons']]}))


if __name__ == '__main__':
    main()
