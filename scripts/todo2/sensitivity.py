"""Sensitivity map: how repeatable each task's story is across the four runs.

    python3 -I scripts/todo2/sensitivity.py --episodes /workspace/episodes --timelines progress/receipts/02-timelines.jsonl --out progress/receipts/02-sensitivity.md

Per task: whether the two same-seed repeats (seed0_run2, seed0_run3) share a story,
whether all four runs do, the spread of the violation step, and the divergence step of
the same-seed pair measured two ways: the first step at which the two sidecars'
proposition vectors differ, and the first step at which the mean absolute pixel
difference between the two main videos exceeds three times its mean over steps 0 to 20.

Class: `knife_edge` if the same-seed repeats disagree on the story; `systematic` if all
four runs share a story and their key-event steps lie within 50 steps of each other;
otherwise `mixed`. The key event is the violation step, else the first lift, else the
success step, else the end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

RUNS = ["seed0_run2", "seed0_run3", "seed1", "seed2"]
PAIR = ("seed0_run2", "seed0_run3")


def key_event(t: dict) -> int:
    for k in ("violation", "lift", "success_step"):
        if t[k] is not None:
            return t[k]
    return t["end"]


def ap_divergence(episodes: str, a: dict, b: dict):
    la = json.load(open(os.path.join(episodes, "jar", a["run"], a["folder"], "base_ltl.json")))["log"]
    lb = json.load(open(os.path.join(episodes, "jar", b["run"], b["folder"], "base_ltl.json")))["log"]
    for x, y in zip(la, lb):
        if x["ap"] != y["ap"]:
            return x["step"]
    return None


def coarse(story: str) -> str:
    if story == "safe_success":
        return "safe_success"
    if story in ("unsafe_success", "lift_open", "tip_before_close", "tip_after_close_on_table", "carry_fail"):
        return "violation"
    return "safe_failure"


def video_divergence(episodes: str, a: dict, b: dict):
    ca = cv2.VideoCapture(os.path.join(episodes, "jar", a["run"], a["folder"], "base_main.mp4"))
    cb = cv2.VideoCapture(os.path.join(episodes, "jar", b["run"], b["folder"], "base_main.mp4"))
    diffs = []
    while True:
        oa, fa = ca.read()
        ob, fb = cb.read()
        if not (oa and ob):
            break
        diffs.append(float(np.mean(np.abs(fa.astype(np.int16) - fb.astype(np.int16)))))
    ca.release()
    cb.release()
    base = float(np.mean(diffs[:21]))
    thresh = 3.0 * base
    step = next((i for i, d in enumerate(diffs) if d > thresh), None)
    step15 = next((i for i, d in enumerate(diffs) if d > 1.5 * base), None)
    c = min(a["contact"] or 0, b["contact"] or 0)
    pre = float(np.mean(diffs[max(0, c - 40):max(1, c)])) / base if base > 0 else None
    return step, base, len(diffs), step15, pre


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="/workspace/episodes")
    ap.add_argument("--timelines", default="progress/receipts/02-timelines.jsonl")
    ap.add_argument("--panels", default="progress/panels.json")
    ap.add_argument("--out", default="progress/receipts/02-sensitivity.md")
    ap.add_argument("--json", default="progress/receipts/02-sensitivity.json")
    args = ap.parse_args()
    tl = [json.loads(l) for l in open(args.timelines)]
    panels = json.load(open(args.panels))
    panel_of = {t: n for n in ("development", "held_out", "pool") for t in panels[n]}
    rows = []
    for task in range(26):
        by = {t["run"]: t for t in tl if t["task"] == task}
        a, b = by[PAIR[0]], by[PAIR[1]]
        stories = [by[r]["story"] for r in RUNS]
        keys = [key_event(by[r]) for r in RUNS]
        viols = [by[r]["violation"] for r in RUNS if by[r]["violation"] is not None]
        same_pair = a["story"] == b["story"]
        same_all = len(set(stories)) == 1
        ap_div = ap_divergence(args.episodes, a, b)
        vid_div, base, n, vid_div15, pre = video_divergence(args.episodes, a, b)
        coarse_all = len({coarse(x) for x in stories}) == 1
        if not same_pair:
            cls = "knife_edge"
        elif same_all and max(keys) - min(keys) <= 50:
            cls = "systematic"
        else:
            cls = "mixed"
        rows.append({"task": task, "panel": panel_of[task], "stories": stories, "same_seed_pair_agree": same_pair,
                     "all_four_agree": same_all, "key_events": keys, "key_spread": max(keys) - min(keys),
                     "violation_steps": viols,
                     "violation_spread": (max(viols) - min(viols)) if len(viols) > 1 else None,
                     "n_violations": len(viols), "pair_ap_divergence": ap_div,
                     "pair_video_divergence": vid_div, "pair_video_baseline": round(base, 3),
                     "pair_video_divergence_1p5x": vid_div15, "pair_pre_contact_ratio": round(pre, 2),
                     "coarse_all_four_agree": coarse_all, "coarse": [coarse(x) for x in stories],
                     "pair_contact": [a["contact"], b["contact"]], "class": cls})
        print(task, cls, ap_div, vid_div, flush=True)
    json.dump(rows, open(args.json, "w"), indent=1)
    md = ["# Todo 2, item 4: sensitivity map", "",
          "Produced by `scripts/todo2/sensitivity.py` from `02-timelines.jsonl` and the sidecars and",
          "main videos of the same-seed pair (`seed0_run2`, `seed0_run3`). Class rule in the script",
          "docstring. `AP div` is the first step at which the pair's proposition vectors differ; `video",
          "div` is the first step at which the mean absolute pixel difference between the pair's main",
          "videos exceeds three times its mean over steps 0 to 20 (`base`, in 8-bit units).", "",
          "| task | panel | stories (run2, run3, seed1, seed2) | pair agree | all agree | coarse agree | key events | key spread | violation steps | AP div | video div | video div 1.5x | base | pre-contact ratio | contact (pair) | class |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['task']} | {r['panel']} | {', '.join(r['stories'])} | {'yes' if r['same_seed_pair_agree'] else 'no'} | "
                  f"{'yes' if r['all_four_agree'] else 'no'} | {'yes' if r['coarse_all_four_agree'] else 'no'} | {', '.join(map(str, r['key_events']))} | {r['key_spread']} | "
                  f"{', '.join(map(str, r['violation_steps'])) or '-'} | {r['pair_ap_divergence']} | {r['pair_video_divergence']} | "
                  f"{r['pair_video_divergence_1p5x']} | {r['pair_video_baseline']} | {r['pair_pre_contact_ratio']} | {r['pair_contact'][0]}, {r['pair_contact'][1]} | {r['class']} |")
    md += ["", "## Counts", ""]
    for cls in ("knife_edge", "systematic", "mixed"):
        ts = [r["task"] for r in rows if r["class"] == cls]
        md.append(f"- {cls}: {len(ts)} tasks: {', '.join(map(str, ts)) or 'none'}")
    for name in ("development", "held_out", "pool"):
        md.append(f"- {name}: " + ", ".join(f"{cls} {sum(1 for r in rows if r['panel'] == name and r['class'] == cls)}"
                                           for cls in ("knife_edge", "systematic", "mixed")))
    md.append(f"- all four runs share a coarse outcome (safe success / safe failure / violation) in "
              f"{sum(r['coarse_all_four_agree'] for r in rows)} of 26 tasks: "
              + ", ".join(f"{r['task']} ({r['coarse'][0]})" for r in rows if r['coarse_all_four_agree']))
    md.append(f"- same-seed pair agrees on the story in {sum(r['same_seed_pair_agree'] for r in rows)} of 26 tasks; "
              f"all four runs agree in {sum(r['all_four_agree'] for r in rows)}")
    vd = [r["pair_video_divergence"] for r in rows if r["pair_video_divergence"] is not None]
    ad = [r["pair_ap_divergence"] for r in rows if r["pair_ap_divergence"] is not None]
    md.append(f"- pair video divergence: {len(vd)} of 26 tasks diverge; steps min {min(vd)}, median {int(np.median(vd))}, max {max(vd)}")
    md.append(f"- pair AP divergence: {len(ad)} of 26 tasks differ; steps min {min(ad)}, median {int(np.median(ad))}, max {max(ad)}")
    md.append(f"- pair contact steps differ by: min {min(abs(r['pair_contact'][0]-r['pair_contact'][1]) for r in rows)}, "
              f"median {int(np.median([abs(r['pair_contact'][0]-r['pair_contact'][1]) for r in rows]))}, "
              f"max {max(abs(r['pair_contact'][0]-r['pair_contact'][1]) for r in rows)}")
    open(args.out, "w").write("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
