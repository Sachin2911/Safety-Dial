#!/usr/bin/env python3
"""Supervisor-managed main evaluation parent; never creates a freeze or extra slots."""

import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_main_launcher import run_main
from helpers.maniguard_phase2_families import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--prior-accounting', type=Path, required=True)
    parser.add_argument('--runtime-contract', type=Path, required=True)
    parser.add_argument('--runtime-contract-sha256', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-jobs', type=int)
    parser.add_argument('--port', type=int, default=18000)
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise KeyboardInterrupt('Main parent received signal ' + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    result = run_main(args.protocol, args.prior_accounting,
                      {'path': str(args.runtime_contract), 'sha256': args.runtime_contract_sha256},
                      root=ROOT, resume=args.resume, max_jobs=args.max_jobs, port=args.port)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 1 if result['blocked_reasons'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
