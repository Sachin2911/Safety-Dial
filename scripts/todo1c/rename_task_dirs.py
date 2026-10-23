"""Rename the per-task folders of the private Jar replication dataset from the evaluator's
`task_<nnnn>` to `<task number>_<outcome>` (scripts/todo1/upload_hf.py: task_folder), and add
an `index.json` per run. One Hub commit per run prefix; videos are moved as LFS copies, so no
video is re-uploaded, and the small JSON files are re-added byte for byte.

    python -I scripts/todo1c/rename_task_dirs.py --repo <ns>/<name> [--prefix jar/seed0 ...] [--dry-run]

Idempotent: a prefix with no `task_<nnnn>` folders left is skipped. HF_TOKEN comes from the
environment; nothing here prints it.
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("upload_hf", HERE.parent / "todo1" / "upload_hf.py")
upload_hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload_hf)

TASK_RE = re.compile(r"^(?P<prefix>(?:jar|smoke)/[^/]+)/(?P<task>task_\d{4})/(?P<file>[^/]+)$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--prefix", action="append", default=[], help="limit to these run prefixes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import (CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi,
                                 hf_hub_download)
    api = HfApi()
    info = api.dataset_info(args.repo, files_metadata=True)
    print("start revision:", info.sha, "files:", len(info.siblings))
    by_prefix: dict[str, dict[str, dict]] = {}
    for sib in info.siblings:
        m = TASK_RE.match(sib.rfilename)
        if m and (not args.prefix or m["prefix"] in args.prefix):
            by_prefix.setdefault(m["prefix"], {}).setdefault(m["task"], {})[m["file"]] = sib
    if not by_prefix:
        print("nothing to rename")
        return 0
    revision = info.sha
    for prefix, tasks in sorted(by_prefix.items()):
        ops, entries = [], []
        for task, files in sorted(tasks.items()):
            row_path = hf_hub_download(args.repo, f"{prefix}/{task}/row.json", repo_type="dataset",
                                       revision=revision)
            row = json.loads(Path(row_path).read_text())
            level = row["scene_name"].split("/")[1]
            folder = upload_hf.task_folder(task, row)
            entries.append(upload_hf.index_entry(task, level, row))
            for name, sib in sorted(files.items()):
                src, dst = f"{prefix}/{task}/{name}", f"{prefix}/{folder}/{name}"
                if sib.lfs:
                    ops.append(CommitOperationCopy(src_path_in_repo=src, path_in_repo=dst, src_revision=revision))
                else:
                    local = hf_hub_download(args.repo, src, repo_type="dataset", revision=revision)
                    ops.append(CommitOperationAdd(path_in_repo=dst, path_or_fileobj=local))
                ops.append(CommitOperationDelete(path_in_repo=src))
            print(f"  {prefix}/{task} -> {prefix}/{folder} ({len(files)} files, "
                  f"{sum(1 for s_ in files.values() if s_.lfs)} LFS copies)")
        index = {"run": prefix, "folder_naming": "<task number>_<outcome>",
                 "outcomes": ["safe_success", "unsafe_success", "safe_unsuccess", "unsafe_unsuccess",
                              "<class>_not_engaged (no contact; safety vacuous)"],
                 "episodes": sorted(entries, key=lambda e: e["task"])}
        ops.append(CommitOperationAdd(path_in_repo=f"{prefix}/index.json",
                                      path_or_fileobj=(json.dumps(index, indent=2) + "\n").encode()))
        print(f"{prefix}: {len(tasks)} folders, {len(ops)} operations" + (" (dry run)" if args.dry_run else ""))
        if args.dry_run:
            continue
        info = api.create_commit(repo_id=args.repo, repo_type="dataset", operations=ops,
                                 commit_message=f"rename {prefix} task folders to <n>_<outcome>; add index.json",
                                 parent_commit=revision)
        revision = info.oid
        print(f"{prefix}: committed {revision}")
    print("final revision:", revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
