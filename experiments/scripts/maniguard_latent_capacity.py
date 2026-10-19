"""Write the immutable timing-only decision after all development and decoder audits."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_latent_capacity import select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--development-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decision = select(args.run, args.development_protocol)
    with args.output.open("x") as stream:
        stream.write(json.dumps(decision, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": decision["status"], "family": decision["family"],
                      "capacity": decision["capacity"]}))


if __name__ == "__main__":
    main()
