"""Download the four Jar runs of record at the pinned dataset revision and check them.

    python -I scripts/todo2/fetch_episodes.py --dest /workspace/episodes [--check-only]

Fetches `index.json`, `row.json`, `base_ltl.json`, `base_main.mp4` and `base_wrist.mp4` for
`jar/seed0_run2`, `jar/seed0_run3`, `jar/seed1`, `jar/seed2` at revision 9300cebf, then
checks 104 sidecars and rows, 208 videos, and frames == steps + 1 on every main video.
The token comes from HF_TOKEN in the environment and is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = "Sachioster/maniguard-jar-replication-20261022"
REVISION = "9300cebf"
RUNS = ["jar/seed0_run2", "jar/seed0_run3", "jar/seed1", "jar/seed2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="/workspace/episodes")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if not args.check_only:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            REPO, repo_type="dataset", revision=REVISION, token=os.environ["HF_TOKEN"],
            local_dir=args.dest,
            allow_patterns=[f"{r}/index.json" for r in RUNS]
            + [f"{r}/*/{f}" for r in RUNS
               for f in ("row.json", "base_ltl.json", "base_main.mp4", "base_wrist.mp4")],
        )
        print("downloaded to", path)
    import cv2
    n_rows = n_side = n_vid = 0
    bad = []
    total_bytes = 0
    for run in RUNS:
        index = json.load(open(os.path.join(args.dest, run, "index.json")))
        entries = index["episodes"]
        if isinstance(entries, dict):
            entries = list(entries.values())
        if len(entries) != 26:
            bad.append(f"{run}: index has {len(entries)} entries")
        for e in entries:
            folder = os.path.join(args.dest, run, e["folder"])
            row = json.load(open(os.path.join(folder, "row.json")))
            n_rows += 1
            side = json.load(open(os.path.join(folder, "base_ltl.json")))
            n_side += 1
            for cam in ("base_main.mp4", "base_wrist.mp4"):
                p = os.path.join(folder, cam)
                cap = cv2.VideoCapture(p)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                n_vid += 1
                total_bytes += os.path.getsize(p)
                if cam == "base_main.mp4" and frames != row["steps"] + 1:
                    bad.append(f"{run}/{e['folder']}: main frames {frames} != steps+1 {row['steps'] + 1}")
                if (w, h) != (256, 256):
                    bad.append(f"{run}/{e['folder']}/{cam}: size {w}x{h}")
            if e.get("steps") is not None and e["steps"] != row["steps"]:
                bad.append(f"{run}/{e['folder']}: index steps {e['steps']} != row {row['steps']}")
            _ = side
    print(f"rows {n_rows} sidecars {n_side} videos {n_vid} video bytes {total_bytes / 1e9:.2f} GB")
    for b in bad:
        print("BAD", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
