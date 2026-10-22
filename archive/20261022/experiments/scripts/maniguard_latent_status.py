"""Refresh durable accounting without querying or advancing either research worker."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_latent_status import refresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        result = refresh(args.run)
        if not args.watch:
            print(json.dumps(result))
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
