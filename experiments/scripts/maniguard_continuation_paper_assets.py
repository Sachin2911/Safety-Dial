"""Render a verified completed screen's factual gripper traces without inference."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_continuation_plot import plot_screen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plot_screen(args.run, args.audit, args.output)
    print(json.dumps({"files": result["files"], "new_physical_controls": 0}))


if __name__ == "__main__":
    main()
