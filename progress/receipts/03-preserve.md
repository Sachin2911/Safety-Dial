# Todo 3, item 8: both arms preserved; instance I stopped from inside

Dataset: `https://huggingface.co/datasets/Sachioster/maniguard-jar-replication-20261022`, **private**,
the repository of todo 1 and todo 2. Before this upload it stood at `11b5a0f9` (todo 2's analysis
had been added under `analysis/02/`; B's `jar/seed0` had been removed earlier, so the jar runs
present were `seed0_run2`, `seed0_run3`, `seed1`, `seed2`). Times UTC, 24 October 2026.

## What was uploaded, 03:40:01 to 03:45:50

`python -I scripts/todo1b/upload_lanes.py --repo Sachioster/maniguard-jar-replication-20261022
--run gate03_native=<gate03_merged/native> --run gate03_retry=<gate03_merged/retry> --receipts
progress/receipts --tag 3 --stage <scratch>/stage` (tmux `upload`, log `03-logs/upload.log.txt`),
after a `--dry-run` of the same command (933 files, 1.50 GB staged). `upload_lanes.py` gained the
`--run <prefix>=<dir>` option the checklist asked for: a directory of run directories is staged
one run per child at `jar/<prefix>/<child>`, each with its own `index.json`, in the
`<n>_<outcome>` folder layout of `scripts/todo1/upload_hf.py`; the README's pooled "jar total"
now pools only the replication runs `jar/seed<d>` (gate runs are listed, never pooled); a
receipts directory is copied recursively; a merged run's `expected_rollouts.json` is carried.
The per-job run leaves of item 6 were first merged per arm and seed by
`scripts/todo3/merge_gate_runs.py` (six rows per `results.jsonl`, each row annotated with its
`gate03_job`, `gate03_leaf` and `gate03_port`). Each prefix went up as its own commit with its
own `path_in_repo`, so nothing outside these prefixes was written or deleted:

| prefix | content |
|---|---|
| `jar/gate03_native/seed100` to `seed115` (16 commits) | per seed: 6 `<n>_<outcome>/` folders (`base_main.mp4`, `base_wrist.mp4`, `base_ltl.json`, `row.json`) plus `index.json`, `results.jsonl`, `summary.json`, `eval_config.json`, `expected_rollouts.json`: 29 files, 464 in all |
| `jar/gate03_retry/seed100` to `seed115` (16 commits) | same layout, 464 files |
| `receipts/03/` (commit `d2877d59`) | every `03-*` receipt as of item 7: `03-host.md`, `03-proxy.md`, `03-identity.md` + `.jsonl` + `-protocol.json`, `03-smoke.md` + `.jsonl` + `-proxy.jsonl` + `03-smoke-frames/`, `03-plan.json`, `03-run.md`, `03-results.md` + `.json`, both env freezes, `03-logs/` (37 files); this file was written afterwards and lives in the repository |
| `README.md` | index rebuilt from every run present: the four replication runs, the four smoke runs and the 32 gate runs (one totals row each, one index line per video); the gate paragraph names the plan and the proxy |

**Final revision: `447162388e816f6ef18e5c9744aa8c2ab9fb5c1d`.**

## Verification from the instance after upload (`HfApi.repo_info` at that revision)

| | files | seeds | task folders | videos | `index.json` | `expected_rollouts.json` |
|---|---|---|---|---|---|---|
| `jar/gate03_native` | 464 | 16 | 96 | 192 | 16 | 16 |
| `jar/gate03_retry` | 464 | 16 | 96 | 192 | 16 | 16 |
| `jar/seed0_run2`, `seed0_run3`, `seed1`, `seed2` (unchanged) | 109 each | | 26 each | | | |
| `smoke/seed0_run2`, `seed0_run3`, `seed1`, `seed2` (unchanged) | 8 each | | | | | |
| `receipts/01b` (9), `receipts/01c` (7), `analysis/02` (unchanged); `receipts/03` | 37 | | | | | |
| whole dataset | 1,526 | | | | | |

Size 2.43 GB; `private: True`. The README totals read, for the replication runs, seed0_run2
7 / 0 / 9 / 10, seed0_run3 7 / 5 / 7 / 7, seed 1 6 / 0 / 10 / 10, seed 2 4 / 2 / 10 / 9, jar total
(one run per seed present, now seeds 1 and 2 only since B's seed 0 is no longer in the repository)
10 / 2 / 20 / 19 over 52, next to the published 21 / 4 / 27 / 26; the 32 gate rows sum to native
17 / 7 / 31 / 41 and retry 16 / 9 / 33 / 38 over 96 each, as in `03-results.md`. Every gate video
is 256 x 256 at 20 fps with frames = steps + 1 (the evaluator's writer, unchanged; spot-checked
with `ffprobe` on the identity and smoke episodes).

## Nothing left only on this disk

Everything the checklist produced is in the repository or the dataset: receipts `03-host.md`,
`03-proxy.md`, `03-identity.md`, `03-smoke.md`, `03-plan.json`, `03-run.md`, `03-results.md`
with their JSON and JSONL sidecars, the frame sheets, the prepare, lane, proxy and upload logs
under `03-logs/`, both env freezes; every gate video, sidecar and row in the dataset. The two
identity recordings (`.npz`, 0.7 and 0.2 MB) and the identity and smoke run directories are
summarised in `03-identity.md` and `03-smoke.md` (rows in `03-identity.jsonl`, `03-smoke.jsonl`)
and are lost with the instance, as are the environments, reproducible from `03-host.md`.

## Servers stopped, GPUs idle, stop call

Policy servers (8000, 8001), the four proxies, the GPU sampler and every finished tmux session
were killed at 03:46:40; `nvidia-smi --query-compute-apps` lists no process and both GPUs read
1 MiB. The stop sequence, run after this receipt was committed and pushed with item 8 ticked:
`python -I scripts/todo1c/stop_instance.py --instance 57411902 --receipt progress/receipts/03-preserve.md --dry-run`
(guards: `CONTAINER_ID`, clean tree with `HEAD == origin/main`, this file naming the revision
above, no compute process), then the same command without `--dry-run`, which asks the provider
to stop instance 57411902 through the instance's own `CONTAINER_API_KEY`. The result cannot be
written here afterwards (the instance is unreachable once stopped), so the account's instance
list is the record: if this file carries no "stop call failed" section below, the guards passed
and the stop was requested. The reviewer destroys the stopped instance from the account after
verifying the upload. Approximate life: rented 18:56 UTC 23 October, stopped about 03:50 UTC
24 October, 9 hours at about $0.50/h.
