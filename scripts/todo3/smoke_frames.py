"""Frame sheet of one episode video around the supervisor's retreats (todo 3 item 4).

    /opt/miniforge3/envs/behavior/bin/python -I scripts/todo3/smoke_frames.py <run dir>/task_0003/base_main.mp4 \
        --proxy-log /workspace/logs/proxy-retry-9100.jsonl --episode-seed <n> --contact-step <n> --out progress/receipts/03-smoke-frames.png

Frame i of a video is step i (frames = steps + 1). Each retreat starts at proxy query q, whose
chunk is executed over steps 8(q-1) .. 8q-1, then 3 more chunks (32 steps in all). Tiles: step 0,
first contact (from the row), then for each retreat the steps q*8-80 (window start), q*8-8
(retreat start), +8, +16, +24, +32 (retreat end), +80; then the last step. Also a strip every
100 steps for the whole episode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def read_frames(path, wanted):
    wanted = set(wanted)
    out, cap, i = {}, cv2.VideoCapture(str(path)), 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = fr.copy()
        i += 1
    cap.release()
    return out, i


def tile(frames, keys, labels, cols):
    tiles = []
    for k, lab in zip(keys, labels):
        fr = frames.get(k)
        if fr is None:
            fr = np.zeros((256, 256, 3), np.uint8)
        fr = cv2.resize(fr, (256, 256))
        cv2.rectangle(fr, (0, 0), (256, 18), (0, 0, 0), -1)
        cv2.putText(fr, lab, (3, 13), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(fr)
    while len(tiles) % cols:
        tiles.append(np.zeros((256, 256, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--proxy-log", required=True)
    ap.add_argument("--episode-seed", type=int, required=True)
    ap.add_argument("--contact-step", type=int, help="first_contact_step from the results row")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=100)
    args = ap.parse_args()
    ep = None
    for l in Path(args.proxy_log).read_text().splitlines():
        r = json.loads(l)
        if r.get("event") == "episode_end" and r.get("episode_seed") == args.episode_seed and r.get("queries", 0) > 1:
            ep = r
    if ep is None:
        raise SystemExit("no episode_end line for that episode seed")
    contact = args.contact_step
    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    last = n - 1
    keys, labels = [0], ["step 0"]
    if contact is not None:
        keys.append(contact); labels.append(f"contact {contact}")
    for i, q in enumerate(ep["retreats"], 1):
        s0 = 8 * (q - 1)
        for d, name in ((-80, "window start"), (0, f"retreat {i} start"), (8, "+8"), (16, "+16"), (24, "+24"),
                        (32, "retreat end"), (80, "+80 resumed")):
            k = min(max(s0 + d, 0), last)
            keys.append(k); labels.append(f"R{i} q{q} s{k} {name}")
    keys.append(last); labels.append(f"last {last}")
    strip_keys = list(range(0, last + 1, args.stride))
    frames, total = read_frames(args.video, set(keys) | set(strip_keys))
    sheet = tile(frames, keys, labels, 7)
    strip = tile(frames, strip_keys, [f"s{k}" for k in strip_keys], 7)
    out = Path(args.out)
    cv2.imwrite(str(out), sheet)
    cv2.imwrite(str(out.with_name(out.stem + "_strip.png")), strip)
    print(f"{total} frames; retreats at queries {ep['retreats']} (steps {[8 * (q - 1) for q in ep['retreats']]}); "
          f"wrote {out} and the strip")


if __name__ == "__main__":
    main()
