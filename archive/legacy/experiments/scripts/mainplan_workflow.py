#!/usr/bin/env python3
"""Execute a finite mainPlan workflow under supervisor, or validate it with --dry-run.

Resume requires the exact same YAML configuration and unchanged completed outputs.
A failed stage with partial outputs needs a fresh configured output path/state directory;
the runner never deletes or adopts existing experiment results. Scientific gate stops
exit successfully with state.status=gate_stopped, preserving a valid negative outcome.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.managedWorkflow import WorkflowError, run_workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    def terminate(_signum, _frame):
        raise KeyboardInterrupt("Workflow interrupted by supervisor")

    signal.signal(signal.SIGTERM, terminate)
    try:
        state = run_workflow(args.config, args.repo_root, resume=args.resume, dry_run=args.dry_run)
    except KeyboardInterrupt:
        return 130
    except (WorkflowError, ValueError, OSError) as exc:
        print(f"[workflow] failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: state[key] for key in ("workflow_id", "status", "state_dir") if key in state}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
