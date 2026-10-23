# Todo 1b, item 7: seeds 1 and 2 preserved in the shared private dataset

Dataset: `https://huggingface.co/datasets/Sachioster/maniguard-jar-replication-20261022`,
**private**, the same repository instance B used for seed 0 (its revision `ccafd65a`,
[01-results-seed0.md](01-results-seed0.md)).

## What was uploaded, 23 October 2026 00:14 to 00:17 UTC

`python -I scripts/todo1b/upload_lanes.py --repo Sachioster/maniguard-jar-replication-20261022
--seed 1=<seed1/ID> --seed 2=<seed2/ID> --smoke 1=<20261022_215034_SMOKE> --smoke 2=<20261022_215037_SMOKE>
--compare receipts/01b-compare.json --receipts progress/receipts --stage <scratch>/stage`
(tmux `upload`, log `third_party/logs/01b-upload.log`). The script stages the same per-task
layout as `scripts/todo1/upload_hf.py` and uploads each prefix with its own `path_in_repo`,
so nothing outside these prefixes was written or deleted:

| prefix | commit | content |
|---|---|---|
| `jar/seed1/` | `27e7833d` | 26 `task_<nnnn>/` folders, each `base_main.mp4`, `base_wrist.mp4`, `base_ltl.json`, `row.json`; plus `results.jsonl`, `summary.json`, `eval_config.json`, `expected_rollouts.json` (108 files) |
| `jar/seed2/` | `c5355bb9` | same, 108 files |
| `smoke/seed1/`, `smoke/seed2/` | `c8602b47`, `0ffd38bf` | the two lane smoke episodes, 7 files each |
| `receipts/01b/` | `b47c900a` | this instance's receipts as of item 6 (install, smoke, lane-wrapper diff, pre-check, results, compare JSON, both env freezes); this file was written afterwards and lives in the repository |
| `README.md` | `c2abc81a`, then **`65a6d525`** | index rebuilt from every run present in the dataset: B's `jar/seed0` and `smoke/seed0` plus the four runs above, one line per video with task, seed and four-class outcome |

**Final revision: `65a6d5255c4afe45b9393c535b0a69a1cb11e256`.**

## Verification from the instance after upload (`HfApi.repo_info`, revision `65a6d525`)

| | files | task dirs | videos |
|---|---|---|---|
| `jar/seed0` (B, unchanged) | 108 | 26 | 52 |
| `jar/seed1` | 108 | 26 | 52 |
| `jar/seed2` | 108 | 26 | 52 |
| `smoke/seed0` (B, unchanged), `smoke/seed1`, `smoke/seed2` | 7 each | 1 each | 2 each |
| `receipts/01` (B, unchanged) | 21 | | |
| `receipts/01b` | 9 | | |
| whole dataset | 377 | | 162 |

Size 0.67 GB; `private: True`. The README index has 26 rows for each of `jar/seed0`,
`jar/seed1`, `jar/seed2` and one for each smoke run; its totals table reads seed 0 = 2 / 1 / 8 / 15,
seed 1 = 6 / 0 / 10 / 10, seed 2 = 4 / 2 / 10 / 9 with one not-engaged episode, jar total
12 / 3 / 28 / 34 over 78 with 77 engaged, next to the published 21 / 4 / 27 / 26. Every local
video was checked before upload (256 x 256, frames = steps + 1, [01b-results.md](01b-results.md)).

## State left on the instance

Policy servers (ports 8000, 8001) and the GPU sampler were stopped at 00:15 UTC after the
runs; both GPUs read 0 MiB. Run trees remain under
`third_party/ManiGuard/outputs/eval_logs/jar_transport_joint_legacy256/{seed1,seed2,*_SMOKE}`
and all command logs under `third_party/logs/01b-*.log` until the instance is destroyed. The
three-seed comparison against the paper (item 7 of 01-replication.md) is left for review.
