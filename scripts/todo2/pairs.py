"""Success-against-failure strips for tasks that have both outcomes in the four runs.

    python3 -I scripts/todo2/pairs.py --episodes /workspace/episodes --timelines progress/receipts/02-timelines.jsonl --out progress/receipts/02-pairs

Pair rule: a task qualifies if some run reached the goal (safe or unsafe success) and some
run did not. The success side prefers a safe success over an unsafe one. The same-seed pair
(seed0_run2, seed0_run3) is used whenever one of them is a success and the other is not a
safe success; otherwise the success and the failure whose first-contact steps are closest.
Steps are matched relative to each episode's own first contact, from contact - 50 to
contact + 250 every 25 steps (13 columns). Rows: success main, failure main, success wrist,
failure wrist. Tiles are scaled to 160 px so each strip stays under 2 MB. A second strip,
`task_<nn>_late.png`, uses absolute steps from anchor - 250 to anchor + 50, where the anchor
is the earliest post-contact event of either episode (success lift, failure violation or
failure lift), so that the step at which the two episodes part is on the sheet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frames import label, read_frames  # noqa: E402

RUNS = ["seed0_run2", "seed0_run3", "seed1", "seed2"]
OFFSETS = list(range(-50, 251, 25))
TILE = 160


def choose_pair(ts: list[dict]):
    by = {t["run"]: t for t in ts}
    succ = [t for t in ts if t["success"]]
    fail = [t for t in ts if not t["success"]]
    if not succ or not fail:
        return None
    a, b = by["seed0_run2"], by["seed0_run3"]
    for s, f in ((a, b), (b, a)):
        if s["success"] and f["story"] != "safe_success" and f is not s:
            return s, f, "same-seed pair"
    succ.sort(key=lambda t: (t["story"] != "safe_success", t["run"]))
    s = succ[0]
    f = min(fail, key=lambda t: abs((t["contact"] or 0) - (s["contact"] or 0)))
    return s, f, "closest contact"


def strip(episodes: str, t: dict, camera: str, anchor: int | None = None) -> np.ndarray:
    c = t["contact"] or 0
    base = c if anchor is None else anchor - 200
    steps = [max(0, min(base + o, t["end"])) for o in OFFSETS]
    frames = read_frames(os.path.join(episodes, "jar", t["run"], t["folder"], f"base_{camera}.mp4"), steps)
    tiles = []
    for o, s in zip(OFFSETS, steps):
        ev = []
        for k in ("closed", "lift", "tip", "violation"):
            if t[k] is not None and s - 25 < t[k] <= s:
                ev.append(k)
        rel = f"(c{o:+d})" if anchor is None else f"(a{s - anchor:+d})"
        tile = label(frames[s], [f"{t['run']} {t['story']}", f"{camera} step {s} {rel}" + (" " + ",".join(ev) if ev else "")])
        tiles.append(cv2.resize(tile, (TILE, TILE), interpolation=cv2.INTER_AREA))
    return np.hstack(tiles)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="/workspace/episodes")
    ap.add_argument("--timelines", default="progress/receipts/02-timelines.jsonl")
    ap.add_argument("--out", default="progress/receipts/02-pairs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tl = [json.loads(l) for l in open(args.timelines)]
    chosen = []
    for task in range(26):
        ts = [t for t in tl if t["task"] == task]
        pair = choose_pair(ts)
        if pair is None:
            continue
        s, f, how = pair
        img = np.vstack([strip(args.episodes, s, "main"), strip(args.episodes, f, "main"),
                         strip(args.episodes, s, "wrist"), strip(args.episodes, f, "wrist")])
        path = os.path.join(args.out, f"task_{task:02d}.png")
        cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        events = [x for x in (s["lift"], f["violation"], f["lift"]) if x is not None]
        anchor = min(events) if events else s["success_step"]
        late = np.vstack([strip(args.episodes, s, "main", anchor), strip(args.episodes, f, "main", anchor),
                          strip(args.episodes, s, "wrist", anchor), strip(args.episodes, f, "wrist", anchor)])
        cv2.imwrite(os.path.join(args.out, f"task_{task:02d}_late.png"), late, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        chosen.append({"task": task, "success": {"run": s["run"], "story": s["story"], "contact": s["contact"]},
                       "failure": {"run": f["run"], "story": f["story"], "contact": f["contact"]}, "how": how,
                       "late_anchor": anchor,
                       "bytes": os.path.getsize(path)})
        print(task, how, s["run"], s["story"], f["run"], f["story"], os.path.getsize(path), flush=True)
    json.dump(chosen, open(os.path.join(args.out, "pairs.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
