#!/usr/bin/env python3
"""Fresh CPU-only main audit child; the managed parent owns its hard wall timeout."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_main_audit_worker import run_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    run_request(args.request)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
