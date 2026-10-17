"""Report planned coverage, four outcomes and recorded costs for explicit study batches."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from helpers.maniguard_batch_report import summarize_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'batches': [summarize_batch(path) for path in args.batch]}
    # Exclusive creation preserves prior snapshots of a running batch.
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    for batch in report['batches']:
        print(json.dumps({k: batch[k] for k in ('batch', 'planned', 'statuses', 'groups')}))


if __name__ == '__main__':
    main()
