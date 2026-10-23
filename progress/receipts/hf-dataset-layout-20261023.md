# Dataset layout change, 23 October 2026

Dataset: `Sachioster/maniguard-jar-replication-20261022` (private). Requested by the user:
task folders named by task number with the outcome in the name, and every path-bearing
file kept consistent.

## Convention

- A task folder is `<task number>_<outcome>`, for example `jar/seed1/8_safe_success/`
  for benchmark `task_0008`. Outcomes are the benchmark's four classes
  (`safe_success`, `unsafe_success`, `safe_unsuccess`, `unsafe_unsuccess`) with
  `_not_engaged` appended when the policy never touched the jar, so far only
  `jar/seed2/11_safe_unsuccess_not_engaged/`.
- Each run has an `index.json` mapping task number and benchmark `scene_name` to its
  folder, outcome, success and violation flags, violation step, first contact, steps,
  episode seed and the four file paths.
- `scene_name` inside `results.jsonl`, `row.json`, `summary.json` and
  `expected_rollouts.json` keeps the benchmark's own `task_<nnnn>/<level>`; it is the
  evaluator's identifier, not a dataset path, and the repository receipts key on it.
- The README index links use the new folders. Note that the Hub file browser sorts
  names as text, so `10_` to `19_` list before `2_`.

Implemented in `scripts/todo1/upload_hf.py` (`outcome_label`, `task_folder`,
`index_entry`, `write_index`, used by `stage_run`), so `scripts/todo1b/upload_lanes.py`
and instance D's todo 1c upload produce the same layout. Pushed as `b0dd8a0` before
instance D reached its upload step.

## Rename of the existing runs

`python -I scripts/todo1c/rename_task_dirs.py --repo Sachioster/maniguard-jar-replication-20261022`
from the workstation, starting at revision `65a6d525`: one Hub commit per run, videos
moved as LFS copies (no re-upload), JSON files re-added byte for byte, old paths
deleted, `index.json` added.

| run | commit |
|---|---|
| jar/seed0 | `bdf04c8d` |
| jar/seed1 | `32baf66c` |
| jar/seed2 | `10776275` |
| smoke/seed0 | `06392a38` |
| smoke/seed1 | `d427a569` |
| smoke/seed2 | `c70de3cd` |
| README rebuilt (`upload_lanes.py --readme-only --tag 1c-tidy`) | **`b7c1554ea6e4f7d5d4f0ae038e12461e0f3145cd`** |

## Verification at `b7c1554e`

383 files, private. Zero `task_<nnnn>` paths remain. All 324 moved files match the
pre-rename snapshot in size and, for the 162 videos, LFS sha256. 26 folders per seed,
one per smoke run, `index.json` present for all six runs with every listed file
present. The README's 162 video links all resolve and none use the old folder names.
