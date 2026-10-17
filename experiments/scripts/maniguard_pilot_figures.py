"""Export descriptive pilot performance-versus-cost data and standalone figures."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_pilot_history import build_history, write_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--title", default="Pilot descriptive results")
    args = parser.parse_args()
    state_bytes = args.state.read_bytes()
    history = build_history(json.loads(state_bytes))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output / "source-state.json").write_bytes(state_bytes)
    if not history["valid"]:
        print(json.dumps({"valid": False, "history": str(args.output / "history.json"),
                          "figures": [], "reason": "Resolve invalid bindings or unknown costs"}))
        return 1
    files = write_figures(history, args.output / "figures", title=args.title)
    print(json.dumps({"valid": True, "complete": history["complete"], "figures": files}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
