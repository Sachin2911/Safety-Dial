"""Episode-level task-16 summaries; final uncertainty is blocked by task case."""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm16_protocol import write_json


def summarize(root, stages=None):
    rows, errors, pending = [], [], []
    for ticket in sorted(root.glob("*/ticket.json")):
        job = json.loads(ticket.read_text())
        if stages and job["stage"] not in stages:
            continue
        result = ticket.parent / "result.json"
        if not result.exists():
            pending.append(job["id"])
            continue
        row = json.loads(result.read_text())
        if row["status"] != "complete":
            errors.append(row)
        else:
            rows.append(row)
    groups = collections.defaultdict(list)
    resources = collections.Counter()
    for r in rows + errors:
        resources.update(r.get("counters", {}))
    for r in rows:
        j = r["job"]
        groups[(j["stage"], j["env"], j["condition"], j["margin"])].append(r)
    table = []
    for key, rs in sorted(groups.items()):
        stage, env, condition, margin = key
        table.append(
            dict(
                stage=stage,
                env=env,
                condition=condition,
                margin=margin,
                n=len(rs),
                requested_safe=sum(r["requested_safe_completion"] for r in rs),
                benchmark_safe=sum(r["benchmark_safe_completion"] for r in rs),
                success=sum(r["success"] for r in rs),
                violations=sum(r["any_violation"] for r in rs),
                breaches=sum(r["margin_breach"] for r in rs),
                unfinished=sum(r["unfinished"] for r in rs),
                censored=sum(r["censored"] for r in rs),
                mean_J=float(np.mean([r["J"] for r in rs])),
                seconds=float(np.mean([r["seconds"] for r in rs])),
                replay_max=max(r["replay_error_max"] for r in rs),
            )
        )
    return {
        "complete": len(rows),
        "errors": len(errors),
        "pending": len(pending),
        "error_details": [{"id": r["job"]["id"], "error": r.get("error")} for r in errors],
        "pending_ids": pending,
        "resources": dict(resources),
        "table": table,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT / "outputs/lewm16/remote/attempts")
    p.add_argument("--stages", nargs="*")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result = summarize(args.root, args.stages)
    if args.out:
        write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
