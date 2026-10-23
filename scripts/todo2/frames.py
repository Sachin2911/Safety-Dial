"""Frame sheets per task: four runs as rows, five labelled main-camera frames as columns.

    python3 -I scripts/todo2/frames.py --episodes /workspace/episodes --timelines progress/receipts/02-timelines.jsonl --out progress/receipts/02-frames

Columns: step 0, first contact, first contact + 40, the key event (violation step, else
first lift, else success step, else last step) and the last step. Each tile is labelled
with run, step and event. For the 15 panel tasks the wrist camera is tiled at the same
steps as `task_<nn>_wrist.png`. Frame i of a video is step i (frames = steps + 1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

RUNS = ["seed0_run2", "seed0_run3", "seed1", "seed2"]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def read_frames(path: str, steps: list[int]) -> dict[int, np.ndarray]:
    """Read the requested frame indices in one sequential pass (seeking is unreliable)."""
    wanted = set(steps)
    out: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(path)
    i = 0
    last = None
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        last = frame
        if i in wanted:
            out[i] = frame.copy()
            wanted.discard(i)
        i += 1
    cap.release()
    for s in wanted:  # a step past the end gets the last frame, marked
        out[s] = last.copy() if last is not None else np.zeros((256, 256, 3), np.uint8)
    return out


def label(tile: np.ndarray, lines: list[str]) -> np.ndarray:
    t = tile.copy()
    for k, text in enumerate(lines):
        y = 14 + 13 * k
        cv2.putText(t, text, (4, y), FONT, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, text, (4, y), FONT, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def columns(t: dict) -> list[tuple[int, str]]:
    c = t["contact"] if t["contact"] is not None else 0
    if t["violation"] is not None:
        key = (t["violation"], "violation " + t["violation_clause"].replace("jar_", "").replace("_", " "))
    elif t["lift"] is not None:
        key = (t["lift"], "first lift")
    elif t["success_step"] is not None:
        key = (t["success_step"], "success")
    else:
        key = (t["end"], "last")
    return [(0, "start"), (c, "contact" if t["contact"] is not None else "no contact"),
            (min(c + 40, t["end"]), "contact+40"), (min(key[0], t["end"]), key[1]), (t["end"], "last")]


def sheet(episodes: str, tasks: list[dict], camera: str) -> np.ndarray:
    rows = []
    for t in sorted(tasks, key=lambda t: RUNS.index(t["run"])):
        cols = columns(t)
        path = os.path.join(episodes, "jar", t["run"], t["folder"], f"base_{camera}.mp4")
        frames = read_frames(path, [s for s, _ in cols])
        tiles = []
        for s, name in cols:
            tiles.append(label(frames[s], [f"{t['run']} {t['story']}", f"step {s} {name}"]))
        rows.append(np.hstack(tiles))
    return np.vstack(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="/workspace/episodes")
    ap.add_argument("--timelines", default="progress/receipts/02-timelines.jsonl")
    ap.add_argument("--panels", default="progress/panels.json")
    ap.add_argument("--out", default="progress/receipts/02-frames")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tl = [json.loads(l) for l in open(args.timelines)]
    panels = json.load(open(args.panels))
    panel_tasks = set(panels["development"]) | set(panels["held_out"])
    for task in range(26):
        tasks = [t for t in tl if t["task"] == task]
        img = sheet(args.episodes, tasks, "main")
        cv2.imwrite(os.path.join(args.out, f"task_{task:02d}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if task in panel_tasks:
            img = sheet(args.episodes, tasks, "wrist")
            cv2.imwrite(os.path.join(args.out, f"task_{task:02d}_wrist.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print("task", task, img.shape, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
