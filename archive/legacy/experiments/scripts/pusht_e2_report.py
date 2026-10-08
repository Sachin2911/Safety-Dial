#!/usr/bin/env python3
"""Write E2 paired tables and deterministic examples using saved rows/bank images only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.pushtRepairReport import write_repair_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-report", type=Path, required=True)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--banks-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.repair_report.read_text())
    path = write_repair_report(args.repair_report,
        args.rows or args.repair_report.with_name("repair_rows.npz"),
        args.banks_dir or Path(report["banks_dir"]), args.output_dir)
    print(f"[e2-report] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
