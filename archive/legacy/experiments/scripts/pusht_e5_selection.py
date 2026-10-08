#!/usr/bin/env python3
"""Resolve E5 without simulation, then optionally execute the unchanged verified argv."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.pushtE5Selection import resolve_selection
from helpers.runManifest import validate_run_id


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    resolve = sub.add_parser("resolve")
    for key in ("acquisition-report", "e2-report", "feasibility-report", "retention-report", "cases",
                "banks-dir", "checkpoint-root", "e5-output-dir", "out"):
        resolve.add_argument("--" + key, type=Path, required=True)
    resolve.add_argument("--e5-run-id", type=validate_run_id, help="Immutable E5 identity; defaults to E5 output basename")
    resolve.add_argument("--acquisition-seed", default="0")
    resolve.add_argument("--blocks", type=int, default=10)
    resolve.add_argument("--samples", type=int, default=300)
    resolve.add_argument("--iterations", type=int, default=30)
    resolve.add_argument("--case-seed", type=int, default=20261005)
    resolve.add_argument("--lam", type=float, default=.05)
    run = sub.add_parser("run")
    run.add_argument("--selection", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "resolve":
        if args.out.exists():
            raise FileExistsError(f"Preserving E5 selection: {args.out}")
        spec = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items() if key not in {"command", "out"}}
        selection = resolve_selection(spec)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as handle:
            json.dump(selection, handle, indent=2)
            handle.write("\n")
        print(json.dumps({"status": selection["status"], "gate": selection["gate"], "simulator_steps": 0}))
        return 0 if selection["gate"]["passes"] else 2
    selection = json.loads(args.selection.read_text())
    if selection.get("status") != "ready" or selection.get("gate", {}).get("passes") is not True:
        raise ValueError("The E5 selection gate is not ready")
    verified = resolve_selection(selection["spec"])
    if verified != selection:
        raise ValueError("E5 input evidence or resolved arguments changed since selection")
    script = ROOT / "experiments/scripts/pusht_e5_gated.py"
    os.execv(sys.executable, [sys.executable, str(script), *verified["e5_arguments"]])


if __name__ == "__main__":
    raise SystemExit(main())
