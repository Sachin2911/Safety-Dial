"""Export independently verified latent-pilot outcomes and complete cost accounting."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_latent_report import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = export(args.run, args.output)
    print(json.dumps({"physical_comparison_complete": report["physical_comparison_complete"],
                      "outcome_counts": report["outcome_counts"],
                      "accounting": report["accounting"]["recorded_totals"]}))


if __name__ == "__main__":
    main()
