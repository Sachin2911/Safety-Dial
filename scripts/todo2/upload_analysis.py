"""Todo 2 item 9: upload the analysis artefacts to the dataset under analysis/02/.

    python3 -I scripts/todo2/upload_analysis.py --repo Sachioster/maniguard-jar-replication-20261022 \
        --npz /workspace/recordings/smoke/task_0000_base.npz

Uploads progress/receipts/02-frames/, 02-pairs/, 02-timelines.jsonl and the item 7 .npz,
each with its own path_in_repo, so nothing outside analysis/02/ is touched. Prints the
final revision. HF_TOKEN comes from the environment and is never printed.
"""
from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import HfApi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--receipts", default="progress/receipts")
    ap.add_argument("--npz", nargs="*", default=[])
    args = ap.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    commits = []
    for folder in ("02-frames", "02-pairs"):
        info = api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=os.path.join(args.receipts, folder),
                                 path_in_repo=f"analysis/02/{folder}", commit_message=f"todo2: analysis/02/{folder}")
        commits.append((folder, info.oid))
        print(folder, info.oid)
    info = api.upload_file(repo_id=args.repo, repo_type="dataset", path_or_fileobj=os.path.join(args.receipts, "02-timelines.jsonl"),
                           path_in_repo="analysis/02/02-timelines.jsonl", commit_message="todo2: analysis/02/02-timelines.jsonl")
    commits.append(("02-timelines.jsonl", info.oid))
    print("02-timelines.jsonl", info.oid)
    for npz in args.npz:
        info = api.upload_file(repo_id=args.repo, repo_type="dataset", path_or_fileobj=npz,
                               path_in_repo=f"analysis/02/recordings/{os.path.basename(npz)}",
                               commit_message=f"todo2: analysis/02/recordings/{os.path.basename(npz)}")
        commits.append((os.path.basename(npz), info.oid))
        print(os.path.basename(npz), info.oid)
    final = api.dataset_info(args.repo)
    files = [s.rfilename for s in final.siblings if s.rfilename.startswith("analysis/02/")]
    print("final revision", final.sha, "files under analysis/02/:", len(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
