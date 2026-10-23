"""Per-episode timelines and story labels for the 104 Jar episodes of record.

    python3 -I scripts/todo2/timeline.py --episodes /workspace/episodes --out progress/receipts

Reads each run's `index.json`, `row.json` and `base_ltl.json` and writes
`02-timelines.jsonl` (one line per episode) and `02-timelines.md` (histograms of stories
overall, per run and per panel, plus timing distributions).

Event steps (all from the monitor sidecar unless stated):
  contact   row `first_contact_step` (evaluator contact flag)
  closed    first step with `jar_closed`
  lift      first step with `!jar_on_support`
  tip       first step with `!jar_upright`
  drop      first step with `jar_dropped`
  violation sidecar `violation_step`; clause from the propositions at that step:
            `jar_upright` if the jar is not upright, else `jar_not_dropped` if dropped,
            else `close_before_lift`
  success   row `success_step` (goal held for 10 steps; `success_first_step` is the first)
  end       row `steps`

Story rule, applied in this order (one label per episode):
  no_contact        contact is None
  safe_success      success and no counted violation
  unsafe_success    success and a counted violation (clause recorded)
  lift_open         violation with clause close_before_lift
  tip_before_close  violation (tip or drop) with no closure observed before the violation step
  carry_fail        closed, then lifted, then tipped or dropped (lift <= violation step),
                    or closed and lifted and timed out without violation
  tip_after_close_on_table
                    violation (tip or drop) after closure while the jar was still on its support
  stall             no violation, no closure, no lift, timed out
  closed_stall      no violation, closed observed, never lifted, timed out
                    (not in the checklist's label list; added because the data needs it)
  unassigned        anything else
The requested-sequence rule (closure observed no later than the first step off support)
is reported alongside as `requested_sequence_ok`.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys

RUNS = ["seed0_run2", "seed0_run3", "seed1", "seed2"]
STORIES = ["no_contact", "stall", "closed_stall", "tip_before_close", "tip_after_close_on_table",
           "lift_open", "carry_fail", "safe_success", "unsafe_success", "unassigned"]


def first(log, key, value):
    for e in log:
        if e["ap"][key] is value:
            return e["step"]
    return None


def clause_at(log, step):
    ap = log[step]["ap"]
    if not ap["jar_upright"]:
        return "jar_upright"
    if ap["jar_dropped"]:
        return "jar_not_dropped"
    return "close_before_lift"


def story(t: dict) -> str:
    if t["contact"] is None:
        return "no_contact"
    if t["success"] and not t["counted_violation"]:
        return "safe_success"
    if t["success"]:
        return "unsafe_success"
    v, closed, lift = t["violation"], t["closed"], t["lift"]
    if v is not None:
        if t["violation_clause"] == "close_before_lift":
            return "lift_open"
        if closed is None or closed > v:
            return "tip_before_close"
        if lift is not None and lift <= v:
            return "carry_fail"
        return "tip_after_close_on_table"
    if closed is None and lift is None:
        return "stall"
    if closed is not None and lift is None:
        return "closed_stall"
    if closed is not None and lift is not None:
        return "carry_fail"
    return "unassigned"


def timeline(run: str, entry: dict, row: dict, side: dict) -> dict:
    log = side["log"]
    v = side["violation_step"]
    t = {
        "run": run, "task": entry["task"], "scene_name": entry["scene_name"], "folder": entry["folder"],
        "outcome": entry["outcome"], "success": bool(row["success"]),
        "counted_violation": bool(row["counted_violation"]), "episode_seed": row["episode_seed"],
        "contact": row["first_contact_step"], "ever_grasped": bool(row["ever_grasped"]),
        "grasp_steps": row["grasp_steps"],
        "closed": first(log, "jar_closed", True), "lift": first(log, "jar_on_support", False),
        "tip": first(log, "jar_upright", False), "drop": first(log, "jar_dropped", True),
        "violation": v, "violation_clause": clause_at(log, v) if v is not None else None,
        "success_first": row["success_first_step"], "success_step": row["success_step"],
        "end": row["steps"],
        "closed_toggles": sum(1 for a, b in zip(log, log[1:]) if a["ap"]["jar_closed"] != b["ap"]["jar_closed"]),
    }
    t["requested_sequence_ok"] = bool(t["success"] and t["closed"] is not None
                                      and (t["lift"] is None or t["closed"] <= t["lift"]))
    t["story"] = story(t)
    return t


def hist(items, key="story"):
    c = collections.Counter(i[key] for i in items)
    return {s: c.get(s, 0) for s in STORIES if c.get(s, 0)}


def hist_row(label, items):
    c = collections.Counter(i["story"] for i in items)
    return f"| {label} | " + " | ".join(str(c.get(s, 0)) for s in STORIES) + f" | {len(items)} |"


def dist(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return "n = 0"
    q = statistics.quantiles(v, n=4) if len(v) >= 2 else [v[0]] * 3
    return (f"n = {len(v)}, min {v[0]}, q1 {q[0]:.0f}, median {statistics.median(v):.0f}, "
            f"q3 {q[2]:.0f}, max {v[-1]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="/workspace/episodes")
    ap.add_argument("--out", default="progress/receipts")
    ap.add_argument("--panels", default="progress/panels.json")
    args = ap.parse_args()
    panels = json.load(open(args.panels))
    tl = []
    for run in RUNS:
        idx = json.load(open(os.path.join(args.episodes, "jar", run, "index.json")))
        for e in idx["episodes"]:
            base = os.path.join(args.episodes, "jar", run)
            row = json.load(open(os.path.join(base, e["files"]["row"])))
            side = json.load(open(os.path.join(base, e["files"]["ltl_sidecar"])))
            tl.append(timeline(run, e, row, side))
    with open(os.path.join(args.out, "02-timelines.jsonl"), "w") as f:
        for t in tl:
            f.write(json.dumps(t) + "\n")

    head = "| set | " + " | ".join(STORIES) + " | n |\n|" + "---|" * (len(STORIES) + 2)
    md = ["# Todo 2, item 2: episode timelines", "",
          "Produced by `scripts/todo2/timeline.py` from the 104 sidecars and rows under",
          "`/workspace/episodes` (dataset revision `9300cebf`). One line per episode is in",
          "`02-timelines.jsonl`. The story rule is in the script's docstring; `closed_stall`",
          "(lid closed, jar never lifted, timed out without violation) is not in the checklist's",
          "label list and was added because the rule as listed could not assign those episodes.", "",
          "## Stories overall and per run", "", head]
    md.append(hist_row("all 104", tl))
    for run in RUNS:
        md.append(hist_row(run, [t for t in tl if t["run"] == run]))
    md += ["", "## Stories per panel", "", head]
    for name in ("development", "held_out", "pool"):
        md.append(hist_row(f"{name} ({len(panels[name])} tasks)", [t for t in tl if t["task"] in panels[name]]))
    md += ["", "## Stories per task (four runs: seed0_run2, seed0_run3, seed1, seed2)", "",
           "| task | panel | stories | safe successes |", "|---|---|---|---|"]
    panel_of = {t: n for n in ("development", "held_out", "pool") for t in panels[n]}
    for task in range(26):
        ts = [t for t in tl if t["task"] == task]
        ts.sort(key=lambda t: RUNS.index(t["run"]))
        md.append(f"| {task} | {panel_of[task]} | " + ", ".join(t["story"] for t in ts)
                  + f" | {sum(t['story'] == 'safe_success' for t in ts)} |")
    md += ["", "## Violations by clause", ""]
    vc = collections.Counter(t["violation_clause"] for t in tl if t["violation"] is not None)
    md += [f"- {k}: {v}" for k, v in vc.most_common()]
    md += [f"- episodes with a violation: {sum(t['violation'] is not None for t in tl)} of 104",
           f"- closure observed at some step: {sum(t['closed'] is not None for t in tl)} of 104",
           f"- gripper grasp registered by the evaluator: {sum(t['ever_grasped'] for t in tl)} of 104",
           f"- lift observed: {sum(t['lift'] is not None for t in tl)} of 104",
           f"- successes: {sum(t['success'] for t in tl)}; satisfying the requested sequence: "
           f"{sum(t['requested_sequence_ok'] for t in tl)}",
           f"- episodes where `jar_closed` toggled more than once: "
           f"{sum(t['closed_toggles'] > 1 for t in tl)}"]
    md += ["", "## Timing distributions (steps at 20 Hz; 100 steps = 5 s)", ""]

    def d(label, f):
        md.append(f"- {label}: {dist([f(t) for t in tl])}")
    d("first contact (all engaged)", lambda t: t["contact"])
    d("contact to violation (violations)", lambda t: None if t["violation"] is None else t["violation"] - t["contact"])
    d("contact to violation, tip_before_close", lambda t: t["violation"] - t["contact"] if t["story"] == "tip_before_close" else None)
    d("contact to violation, tip_after_close_on_table", lambda t: t["violation"] - t["contact"] if t["story"] == "tip_after_close_on_table" else None)
    d("contact to closure (closure observed)", lambda t: None if t["closed"] is None or t["contact"] is None else t["closed"] - t["contact"])
    d("closure to lift (closed then lifted)", lambda t: None if t["closed"] is None or t["lift"] is None or t["lift"] < t["closed"] else t["lift"] - t["closed"])
    d("lift to success (successes with a lift)", lambda t: None if not t["success"] or t["lift"] is None else t["success_step"] - t["lift"])
    d("contact to success (successes)", lambda t: None if not t["success"] else t["success_step"] - t["contact"])
    d("success step (successes)", lambda t: t["success_step"])
    md += ["", "## Episodes the checklist's rule could not assign", ""]
    extra = [t for t in tl if t["story"] in ("closed_stall", "unassigned")]
    md.append(f"`closed_stall`: {sum(t['story'] == 'closed_stall' for t in tl)} episodes; "
              f"`unassigned`: {sum(t['story'] == 'unassigned' for t in tl)}.")
    md += ["", "| run | task | story | contact | closed | lift | tip | violation | end |", "|---|---|---|---|---|---|---|---|---|"]
    for t in extra:
        md.append(f"| {t['run']} | {t['task']} | {t['story']} | {t['contact']} | {t['closed']} | {t['lift']} | {t['tip']} | {t['violation']} | {t['end']} |")
    md += ["", "## Oddities", ""]
    for t in tl:
        notes = []
        if t["success"] and t["closed"] is None:
            notes.append("success without closure ever observed")
        if t["success"] and t["lift"] is None:
            notes.append("success without the jar ever leaving its support")
        if t["tip"] is not None and t["violation"] is not None and t["tip"] < t["violation"]:
            notes.append("tip before the recorded violation step")
        if t["closed_toggles"] > 4:
            notes.append(f"jar_closed toggled {t['closed_toggles']} times")
        if t["violation"] is not None and t["contact"] is not None and t["violation"] < t["contact"]:
            notes.append("violation before the evaluator's first contact")
        if notes:
            md.append(f"- {t['run']} task {t['task']} ({t['story']}): " + "; ".join(notes))
    open(os.path.join(args.out, "02-timelines.md"), "w").write("\n".join(md) + "\n")
    print(json.dumps(hist(tl)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
