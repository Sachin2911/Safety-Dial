"""Render the gate's progress (third_party/logs/03-gate-lane{0,1}.jsonl) into receipts/03-run.md.

    python3 -I scripts/todo3/run_progress.py [--out progress/receipts/03-run.md]

Lists, per lane, jobs done, running, failed and rerun, the running tally of the four classes per
arm, per-episode wall times, retreats fired so far (retry proxy logs), and every attempt that did
not end in a completed row. Pure stdlib; safe to rerun while the lanes run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/Safety-Dial")
TP = ROOT / "third_party"


def cls(r):
    if not r or r.get("status") != "completed" or not r.get("safety_evaluated"):
        return "invalid"
    s, v = bool(r.get("success")), bool(r.get("counted_violation"))
    return {(True, False): "safe_success", (True, True): "unsafe_success",
            (False, False): "safe_unsuccess", (False, True): "unsafe_unsuccess"}[(s, v)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "progress/receipts/03-run.md"))
    args = ap.parse_args()
    plan = json.loads((ROOT / "progress/receipts/03-plan.json").read_text())
    jobs = {j["job"]: j for j in plan["jobs"]}
    events = []
    for lane in (0, 1):
        p = TP / "logs" / f"03-gate-lane{lane}.jsonl"
        if p.is_file():
            events += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    ends = [e for e in events if e["event"] == "end"]
    starts = [e for e in events if e["event"] == "start"]
    running = [s for s in starts if not any(e["job"] == s["job"] and e["attempt"] == s["attempt"] for e in ends)]
    done_jobs = {e["job"] for e in ends if e["status"] == "completed"}
    failed = [e for e in ends if e["status"] != "completed"]
    reseed = [e for e in events if e["event"] == "reseed_probe"]
    retreats = Counter()
    n_retry_lines = 0
    for port in (9100, 9101):
        p = Path(f"/workspace/logs/proxy-retry-{port}.jsonl")
        if p.is_file():
            for l in p.read_text().splitlines():
                r = json.loads(l)
                if r.get("event") == "episode_end" and r.get("queries", 0) > 1 and r["t"] > 1792787000:  # after the smokes
                    n_retry_lines += 1
                    retreats[r["n_retreats"]] += 1
    tally = {a: Counter() for a in plan["arms"]}
    for e in ends:
        if e["status"] == "completed":
            j = jobs[e["job"]]
            r = {"status": "completed", "safety_evaluated": True, "success": e["success"], "counted_violation": e["counted_violation"]}
            tally[j["arm"]][cls(r)] += 1
    L = []
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    L.append("# Todo 3, item 6: the gate run (progress)")
    L.append("")
    L.append(f"Updated {now}. Plan `03-plan.json` frozen {plan['frozen_utc']}, {plan['n_jobs']} jobs. Lanes started "
             f"{starts[0]['t'] if starts else 'n/a'} (tmux `gate0`, `gate1`; `scripts/todo3/run_gate_lane.py`; per-job logs "
             "`third_party/logs/03-gate/`). Per lane: real server 8000/8001, proxy `off` 9000/9001 for the native arm, proxy "
             "`retry` 9100/9101 for the retry arm. Each job is one episode through `scripts/todo1/eval_family_lane.sh` with "
             "`TASKS`, `SEED`, `PORT` from the plan and output leaf `gate03/<arm>/seed<s>/<task>` (the task is appended to the "
             "checklist's leaf because the wrapper refuses a non-empty output directory and each job writes its own). "
             "Full horizon, benchmark and config unchanged.")
    L.append("")
    L.append(f"- jobs with a completed row: **{len(done_jobs)} of {plan['n_jobs']}**; running now: "
             + (", ".join(f"job {s['job']} ({jobs[s['job']]['arm']}, seed {s['seed']}, {s['task']}, lane {s['lane']}, since {s['t']})" for s in running) or "none"))
    L.append(f"- attempts without a completed row: {len(failed)}; re-seed probes before reruns: {len(reseed)}")
    for e in failed:
        j = jobs[e["job"]]
        L.append(f"  - job {e['job']} attempt {e['attempt']} ({j['arm']}, seed {j['seed']}, {j['task']}, lane {e['lane']}): rc {e['rc']}, {e['status']}, {e['elapsed_s']} s, leaf `{e['leaf']}`")
    L.append("")
    L.append("| arm | safe success | unsafe success | safe unsuccess | unsafe unsuccess | completed |")
    L.append("|---|---|---|---|---|---|")
    for a in plan["arms"]:
        c = tally[a]
        L.append(f"| {a} | {c['safe_success']} | {c['unsafe_success']} | {c['safe_unsuccess']} | {c['unsafe_unsuccess']} | {sum(c.values())} |")
    L.append("")
    L.append(f"- retry episodes logged by the proxies so far: {n_retry_lines}; retreats fired per episode: "
             + (", ".join(f"{k}: {v}" for k, v in sorted(retreats.items())) or "none yet"))
    for lane in (0, 1):
        le = [e for e in ends if e["lane"] == lane and e["status"] == "completed"]
        if le:
            w = sorted(e["elapsed_s"] for e in le)
            L.append(f"- lane {lane}: {len(le)} completed, wall per job min/median/max {w[0]:.0f}/{w[len(w) // 2]:.0f}/{w[-1]:.0f} s, "
                     f"steps mean {sum(e['steps'] for e in le) / len(le):.0f}")
    L.append("")
    L.append("## Job log (completed attempts, plan order)")
    L.append("")
    L.append("| job | lane | arm | seed | task | attempt | start | wall s | steps | success | violation | class |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    st = {(s["job"], s["attempt"]): s["t"] for s in starts}
    for e in sorted(ends, key=lambda e: (e["job"], e["attempt"])):
        j = jobs[e["job"]]
        r = {"status": e["status"], "safety_evaluated": True, "success": e["success"], "counted_violation": e["counted_violation"]}
        L.append(f"| {e['job']} | {e['lane']} | {j['arm']} | {j['seed']} | {j['task']} | {e['attempt']} | {st.get((e['job'], e['attempt']), '')} | "
                 f"{e['elapsed_s']:.0f} | {e['steps']} | {e['success']} | {e['counted_violation']} | {cls(r)} |")
    L.append("")
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"{len(done_jobs)} done, {len(running)} running, {len(failed)} failed attempts -> {args.out}")


if __name__ == "__main__":
    main()
