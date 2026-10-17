"""Save an immutable inventory of all raw ManiGuard episode and diagnostic costs."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from helpers.maniguard_accounting import inventory_costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = inventory_costs(args.runs)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('record_types', 'recorded_cost_totals',
                                                  'warnings')}, indent=2))


if __name__ == '__main__':
    main()
