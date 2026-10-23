# Todo 2, item 9: preserve, push, stop

Dataset `Sachioster/maniguard-jar-replication-20261022` (private). Uploaded from the pod at
about 13:58 UTC, 23 October 2026, by `python3 -I scripts/todo2/upload_analysis.py --repo
Sachioster/maniguard-jar-replication-20261022`, each prefix with its own `path_in_repo`
under `analysis/02/`; nothing outside `analysis/02/` was written or deleted.

| path in repo | content | commit |
|---|---|---|
| `analysis/02/02-frames/` | 41 PNG sheets (26 main-camera task sheets, 15 wrist sheets), 36 MB | `c05f5d7a` |
| `analysis/02/02-pairs/` | 32 PNG strips (16 early, 16 late) and `pairs.json`, 41 MB | `c26bc60d` |
| `analysis/02/02-timelines.jsonl` | 104 lines, one per episode | `11b5a0f9` |
| `analysis/02/recordings/` | nothing: no `.npz` exists, the item 7 episode did not run ([02-host.md](02-host.md)) | |

**Final revision: `11b5a0f9b91fa51660e09474bfab0f8460c25f51`**, 75 files under
`analysis/02/`. The episode data of record is unchanged at `9300cebf` and is also present
at this revision.

## Repository

Every receipt of this todo (`02-fetch.md`, `02-timelines.md` and `.jsonl`, `02-frames/`,
`02-task-notes.md`, `02-sensitivity.md` and `.json`, `02-pairs/`, `02-pairs.md`,
`02-logging.md`, `02-host.md`, `02-logs/`, the two environment freezes), the scripts under
`scripts/todo2/`, `progress/panels.json` and `docs/research/research_notes/Where the
leverage is.md` are committed on `main` and pushed; `git rev-parse HEAD` equals
`origin/main` at the time of the stop (the commit after this file is the last one).

Left on the pod's volume only, all reproducible from pinned sources: the ManiGuard,
BEHAVIOR-1K and OpenPI checkouts and environments under `third_party/` and
`/workspace/miniforge3`, the benchmark, checkpoint and asset downloads, the local copy
of the four runs under `/workspace/episodes`, the Kit logs under
`third_party/ManiGuard/behavior-1k/OmniGibson/appdata/`, and the four empty smoke run
directories. The reviewer terminates the pod after verifying the upload.

## Stop

`nvidia-smi` shows no compute process (0 MiB). `python -I scripts/todo2/stop_pod.py
--receipt progress/receipts/02-preserve.md --dry-run`, then the same without `--dry-run`,
as the last action.
