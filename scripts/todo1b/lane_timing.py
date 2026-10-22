"""Per-episode wall time from a lane runner log (scripts/todo1b/run_jar_seed_lane.sh output).

    python -I scripts/todo1b/lane_timing.py <lane log> [--json out.json]

Parses the runner's "############ [i/N] <scene> START HH:MM:SS" and "OK|exit=rc HH:MM:SS"
lines plus the wrapper's START/EXIT lines (UTC). Pure stdlib.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

EP = re.compile(r"############ \[(\d+)/(\d+)\] (\S+) (START|OK|exit=\d+)\s+(\d\d:\d\d:\d\d)")
WRAP = re.compile(r"^(START|EXIT=\d+).*?(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)", re.M)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    text = args.log.read_text(errors="replace")
    wrap = WRAP.findall(text)
    t0 = datetime.strptime(wrap[0][1], "%Y-%m-%dT%H:%M:%SZ") if wrap else None
    episodes, open_ = [], {}
    day = t0.date() if t0 else None
    for i, n, scene, what, hms in EP.findall(text):
        t = datetime.combine(day, datetime.strptime(hms, "%H:%M:%S").time())
        if t0 and t < t0 - timedelta(minutes=1):
            t += timedelta(days=1)
        if what == "START":
            open_[scene] = (int(i), int(n), t)
        else:
            idx, n_, ts = open_.pop(scene, (int(i), int(n), None))
            episodes.append({"index": idx, "n": n_, "scene": scene, "start": ts.isoformat() if ts else None,
                             "end": t.isoformat(), "seconds": (t - ts).total_seconds() if ts else None,
                             "outcome": what})
    done = [e for e in episodes if e["seconds"] is not None]
    secs = [e["seconds"] for e in done]
    end = datetime.strptime(wrap[-1][1], "%Y-%m-%dT%H:%M:%SZ") if len(wrap) > 1 and wrap[-1][0].startswith("EXIT") else None
    out = {
        "log": str(args.log), "wrapper_start": t0.isoformat() if t0 else None,
        "wrapper_end": end.isoformat() if end else None,
        "wrapper_seconds": (end - t0).total_seconds() if end and t0 else None,
        "episodes_completed": len(done), "episodes_failed": sum(e["outcome"] != "OK" for e in done),
        "in_progress": list(open_),
        "per_episode_seconds": {"min": min(secs) if secs else None, "median": sorted(secs)[len(secs) // 2] if secs else None,
                                "max": max(secs) if secs else None, "mean": sum(secs) / len(secs) if secs else None},
        "episodes": episodes,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "episodes"}, indent=1))
    for e in episodes:
        print(f"  [{e['index']}/{e['n']}] {e['scene']:18} {e['outcome']:8} {e['seconds'] if e['seconds'] is not None else '?':>7} s")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
