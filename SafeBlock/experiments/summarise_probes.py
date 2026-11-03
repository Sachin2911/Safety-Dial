"""Summarise probe results (summary.json files) into a markdown table and a JSON record.

    python3 SafeBlock/experiments/summarise_probes.py --root outputs/probes-20261101 --probe A
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(root, probe):
    rows = []
    for job_dir in sorted((root / probe).iterdir()):
        if not job_dir.is_dir():
            continue
        attempts = sorted(job_dir.glob("attempt-*"))
        done = [a for a in attempts if (a / "summary.json").exists()]
        if done:
            s = json.loads((done[-1] / "summary.json").read_text())
            rows.append({**s, "job": job_dir.name, "attempt": done[-1].name})
        else:
            errors = [a for a in attempts if (a / "worker-error.json").exists()]
            rows.append({"job": job_dir.name, "attempt": attempts[-1].name if attempts else None, "status": "failed" if errors else "missing",
                         "error": json.loads((errors[-1] / "worker-error.json").read_text())["error"] if errors else None})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load(args.root, args.probe)
    by = defaultdict(list)
    for r in rows:
        condition = r["job"].split("-", 2)[2] if r["job"].count("-") >= 2 else r["job"]
        by[condition].append(r)
    lines = [f"# Probe {args.probe} summary", "", "| condition | episodes | failed runs | safe success | sustained | violation | physical | boundary | timeout | K1 | K2 | K3 | mean steps (success) | mean plan s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    record = {}
    for condition, items in sorted(by.items()):
        ok = [r for r in items if r.get("status") == "complete"]
        failed = [r for r in items if r.get("status") != "complete"]
        succ = [r for r in ok if r["safe_success"]]
        plan = [r["plan_seconds_mean"] for r in ok if r.get("plan_seconds_mean")]
        k = {f: sum(bool(r["keepout_episode"][f]) for r in ok) for f in ("K1_overhead", "K2_ceiling", "K3_lateral")}
        rec = {
            "episodes": len(ok), "failed_runs": len(failed), "safe_success": len(succ),
            "sustained": sum(bool(r.get("sustained_safe_success")) for r in ok),
            "violation": sum(bool(r["violation"]) for r in ok),
            "physical_failure": sum(bool(r["physical_failure"]) for r in ok),
            "boundary_exit": sum(bool(r["boundary_exit"]) for r in ok),
            "timeout": sum(bool(r["timeout"]) for r in ok), "keepout": k,
            "mean_steps_success": round(sum(r["steps"] for r in succ) / len(succ), 1) if succ else None,
            "mean_plan_seconds": round(sum(plan) / len(plan), 3) if plan else None,
            "per_seed": {r["job"]: {"outcome": r.get("outcome"), "steps": r.get("steps"), "sustained": r.get("sustained_safe_success"), "keepout": r.get("keepout_episode"), "error": r.get("error")} for r in items},
        }
        record[condition] = rec
        lines.append(f"| {condition} | {rec['episodes']} | {rec['failed_runs']} | {rec['safe_success']} | {rec['sustained']} | {rec['violation']} | {rec['physical_failure']} | {rec['boundary_exit']} | {rec['timeout']} | {k['K1_overhead']} | {k['K2_ceiling']} | {k['K3_lateral']} | {rec['mean_steps_success']} | {rec['mean_plan_seconds']} |")
    lines += ["", "Per-episode outcomes:", ""]
    for r in rows:
        lines.append(f"- {r['job']}: {r.get('outcome') or r.get('status')} steps={r.get('steps')} sustained={r.get('sustained_safe_success')} keepout={r.get('keepout_episode')} {('error=' + str(r.get('error'))) if r.get('error') else ''}")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"probe-{args.probe}.md").write_text(text)
        (args.output / f"probe-{args.probe}.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
