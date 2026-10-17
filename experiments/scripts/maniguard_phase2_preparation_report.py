"""Write an immutable preparation report; never freeze a protocol or launch jobs."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_phase2_preparation_report import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path,
                        default=ROOT / "runs/maniguard/phase2/preparation-v1")
    parser.add_argument("--setup", type=Path, default=ROOT / "runs/maniguard/phase2/setup")
    parser.add_argument("--pilot", type=Path, default=ROOT / "configs/maniguard/phase2/pilot.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="New report directory; existing output is never overwritten")
    args = parser.parse_args()
    report = build_report(args.preparation, args.setup, args.output, pilot_path=args.pilot)
    print(json.dumps({"report": str(args.output.resolve() / "report.json"),
                      "preparation_complete": report["preparation_complete"],
                      "prior_accounting_ready": report["prior_accounting_ready"],
                      "ready_for_freeze_review": report["ready_for_freeze_review"],
                      "training_ready": report["training_ready"]}))


if __name__ == "__main__":
    main()
