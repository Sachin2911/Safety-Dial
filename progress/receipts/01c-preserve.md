# Todo 1c, items 5 and 6: seed 0 runs 2 and 3 preserved; instance D stopped from inside

Dataset: `https://huggingface.co/datasets/Sachioster/maniguard-jar-replication-20261022`, **private**, the
same repository that holds B's seed 0 (`jar/seed0`) and C's seeds 1 and 2 ([01b-preserve.md](01b-preserve.md)).
Before this upload the repository stood at `b7c1554e` (383 files; the task folders had been renamed to
`<task number>_<outcome>` with an `index.json` per run on 23 October, which this upload follows).

## What was uploaded, 23 October 2026 09:52:06 to 09:53:50 UTC

`python -I scripts/todo1b/upload_lanes.py --repo Sachioster/maniguard-jar-replication-20261022
--seed 0_run2=<seed0_run2/ID> --seed 0_run3=<seed0_run3/ID> --smoke 0_run2=<20261023_073129_SMOKE_2>
--smoke 0_run3=<20261023_073129_SMOKE> --compare receipts/01c-compare.json --receipts progress/receipts
--tag 1c --stage <scratch>/stage` (tmux `upload`, log `01-runs/logs/01c-upload.log.txt`), after a `--dry-run`
of the same command that staged 241 files (0.41 GB). The script stages the per-task layout of
`scripts/todo1/upload_hf.py` and uploads each prefix with its own `path_in_repo`, so nothing outside these
prefixes was written or deleted:

| prefix | commit | content |
|---|---|---|
| `jar/seed0_run2/` | `eacdf6c0` | 26 `<n>_<outcome>/` folders, each `base_main.mp4`, `base_wrist.mp4`, `base_ltl.json`, `row.json`; plus `index.json`, `results.jsonl`, `summary.json`, `eval_config.json`, `expected_rollouts.json` (109 files) |
| `jar/seed0_run3/` | `5d211a51` | same, 109 files |
| `smoke/seed0_run2/`, `smoke/seed0_run3/` | `651d992b`, `e659844b` | the two lane smoke episodes (`0_safe_success`, `0_safe_unsuccess`), 8 files each |
| `receipts/01c/` | `eb314ffb` | this instance's receipts as of item 4: `01c-install.md`, `01c-smoke.md`, `01c-smoke.jsonl`, `01c-results.md`, `01c-compare.json`, both env freezes (7 files); this file was written afterwards and lives in the repository |
| `README.md` | **`220cb283`** | index rebuilt from every run present: `jar/seed0`, `jar/seed0_run2`, `jar/seed0_run3`, `jar/seed1`, `jar/seed2` and the five smoke runs, one line per video; the jar total pools one run per seed, so runs 2 and 3 are listed but not pooled |

**Final revision: `220cb2833fa3e8f1d829464bf7f6a26934e3f0bc`.**

## Verification from the instance after upload (`HfApi.repo_info` at revision `220cb283`)

| | files | task dirs | videos | `index.json` |
|---|---|---|---|---|
| `jar/seed0` (B, unchanged), `jar/seed1`, `jar/seed2` (C, unchanged) | 109 each | 26 each | 52 each | yes |
| `jar/seed0_run2` | 109 | 26 | 52 | yes |
| `jar/seed0_run3` | 109 | 26 | 52 | yes |
| `smoke/seed0`, `smoke/seed1`, `smoke/seed2` (unchanged), `smoke/seed0_run2`, `smoke/seed0_run3` | 8 each | 1 each | 2 each | yes |
| `receipts/01` (21), `receipts/01b` (9) unchanged; `receipts/01c` | 7 | | | |
| whole dataset | 624 | | 270 | |

Size 1.08 GB; `private: True`. The README index has 26 rows for each of the five jar runs and one for each
smoke run; its totals table reads seed 0 = 2 / 1 / 8 / 15, seed0_run2 = 7 / 0 / 9 / 10, seed0_run3 = 7 / 5 / 7 / 7,
seed 1 = 6 / 0 / 10 / 10, seed 2 = 4 / 2 / 10 / 9 with one not-engaged episode, jar total (one run per seed)
12 / 3 / 28 / 34 over 78 with 77 engaged, next to the published 21 / 4 / 27 / 26. Every local video was checked
before upload (256 x 256, 20 fps, frames = steps + 1, [01c-results.md](01c-results.md)).

Deviation: `scripts/todo1b/upload_lanes.py` as pulled formed the receipts prefix as `01` + tag, so `--tag 1c`
gave `receipts/011c` with no files in the dry run; it was changed to `0` + tag (`receipts/01c`, commit
messages still `todo1c: ...`) before the live upload and pushed (`1e344cc`). No other edit.

## Nothing left only on this disk

Everything the checklist produced is in the repository or the dataset: the run inputs, upstream summaries,
lane timings and runner logs under `01-runs/seed0_run2/` and `01-runs/seed0_run3/`; the install, server,
probe, smoke, upload logs and the GPU sampler CSV under `01-runs/logs/01c-*`; every video and sidecar in the
dataset. The run trees under `third_party/ManiGuard/outputs/eval_logs/jar_transport_joint_legacy256/` and
the environments are reproducible from [01c-install.md](01c-install.md) and are lost with the instance.

## Item 6: servers stopped, GPUs idle, stop call

Policy servers (ports 8000, 8001), the GPU sampler and the finished lane and upload tmux sessions were
killed at 09:54:51 UTC; `nvidia-smi --query-compute-apps` lists no process and both GPUs read 0 MiB (the
sampler's last row, 09:54:47, still shows the idle servers at 8,490 and 8,496 MiB).

The stop sequence, run after this receipt was committed and pushed with items 5 and 6 ticked:
`python -I scripts/todo1c/stop_instance.py --instance 57334420 --dry-run` (guards: `CONTAINER_ID`, clean tree
with `HEAD == origin/main`, this file naming the revision above, no compute process), then the same command
without `--dry-run`, which asks the provider to stop instance 57334420 through the instance's own
`CONTAINER_API_KEY`. The result of the stop call cannot be written here afterwards (the instance is unreachable
once stopped), so the account's instance list is the record: if this file carries no "stop call failed"
section below, the guards passed and the stop was requested. The reviewer destroys the stopped instance from
the account after verifying the upload.
