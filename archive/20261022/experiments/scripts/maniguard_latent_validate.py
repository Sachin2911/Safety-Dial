"""Verify the frozen physical comparison without launching physics or model queries."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_latent_completion import verify
from helpers.maniguard_latent_queue import atomic
from helpers.maniguard_latent_report import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report = export(args.run, args.output)
    result = verify(args.run, args.protocol, report)
    atomic(args.output / "design-verification.json", result)
    print(json.dumps({key: result[key] for key in
                      ("physical_design_complete", "verified_full_jobs", "full_attempts", "errors")}))
    print(json.dumps({"pending_items": len(result["pending"])}))
    if result["errors"] or (result["pending"] and not args.allow_incomplete):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
