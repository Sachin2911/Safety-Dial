# Todo 1b: seeds 1 and 2 on two GPU lanes (instance C)

Purpose: finish todo 1 faster. Instance B (one RTX 3090) is running seed 0 of
[01-replication.md](01-replication.md). This instance runs seeds 1 and 2 as two
parallel lanes, one per GPU, using the same pinned sources, the same install recipe and
the same per-task evaluator command as instance B. Scoring is untouched; only the
scheduling changes, and that deviation is recorded.

Host: Vast instance 57271835, 2x RTX 3090, driver 580.173.02, 256 GB RAM, 20 cores,
200 GB disk ([rental record](rental-20261022.json)). Work under `/workspace`.
`HF_TOKEN` and `GITHUB_TOKEN` are in the environment; never print or commit them.

## Checklist

- [ ] **1. Install by replaying instance B's recipe.** Follow
      [receipts/01-sources.json](receipts/01-sources.json) for every revision and
      [receipts/01-install.md](receipts/01-install.md) for the commands, including the
      `CONDA_ENVS_PATH` workaround, the pre-built curobo, Spot and `openpi-client`, and
      the X11/GL apt packages from [receipts/01-smoke.md](receipts/01-smoke.md). Use
      `scripts/todo1/download_pinned.py` for the benchmark, checkpoint and robot asset
      and verify hashes. Write `receipts/01b-install.md` with any difference from B.
      Already done on this host before handover: the image's
      `/opt/instance-tools/bin/install-display-drivers` was run because OptiX and EGL
      libraries were missing (as on this machine's October rental); GL, OptiX, Vulkan
      and EGL now resolve. Verify with `ldconfig -p` before installing.
- [ ] **2. Smoke one episode per GPU.** Start one policy server per lane:
      `CUDA_VISIBLE_DEVICES=0` on port 8000 and `CUDA_VISIBLE_DEVICES=1` on port 8001,
      each warmed with `scripts/todo1/probe_server.py`. Run `task_0000/base` with seed 1
      on lane 0 and seed 2 on lane 1 using the config in
      [receipts/01-jar-eval.yaml](receipts/01-jar-eval.yaml), tagged `smoke`. Both must
      reach a verdict with 256 by 256 videos. Record wall time and peak GPU memory per
      lane in `receipts/01b-smoke.md`. Count these episodes.
- [ ] **3. Lane wrapper.** Copy upstream `scripts/eval_family.sh` to
      `scripts/todo1/eval_family_lane.sh` with exactly two additions: a `PORT`
      environment variable replacing the hard-coded `--port 8000`, and a `TASKS`
      environment variable restricting the plan to a listed subset of task directories.
      Nothing else changes: same `--config`, `--benchmark-root`, `--scenes`,
      `--max-scenes 1`, `--output-dir`, `--run-name`, `--metrics success safety`,
      `--headless`, `--seed`. Commit the diff against upstream as
      `receipts/01b-lane-wrapper.diff`.
- [ ] **4. Check instance B's seed 0 before launching.** `git pull` and read any
      seed 0 progress B has pushed, or ask the user. If B has six or more completed
      episodes with zero safe successes, stop here and wait for review. Otherwise go on.
- [ ] **5. Run seed 1 and seed 2 concurrently.** Lane 0 runs all 26 base tasks at
      `SEED=1` on GPU 0, port 8000, output leaf `jar_transport_joint_legacy256/seed1`.
      Lane 1 runs all 26 at `SEED=2` on GPU 1, port 8001, leaf `.../seed2`. Each lane
      in its own tmux session with its own log. Do not split one seed across lanes;
      keep each seed's `expected_rollouts.json` and `summary.json` whole.
- [ ] **6. Collect.** After both lanes finish, run upstream `tools/eval_summary.py`
      on each seed directory, copy both `results.jsonl` files, `summary.json` and
      `expected_rollouts.json` under `receipts/01-runs/seed1/` and `seed2/`, and write
      per-seed four-class counts, engagement and wall time into `receipts/01b-results.md`.
- [ ] **7. Preserve.** Upload both seeds' full `outputs/eval_logs` trees, including every
      video and LTL sidecar, to the same private Hugging Face dataset layout as item 8 of
      01-replication.md (`jar/seed<s>/task_<nnnn>/`) using
      `scripts/todo1/upload_hf.py --repo Sachioster/maniguard-jar-replication-20261022`,
      the same private dataset instance B uses for seed 0. Record repo and revision.
      Push and stop. The three-seed comparison (item 7 of 01-replication.md) is done after review.

## Rules

- Official evaluator per task, unchanged. The lane wrapper only chooses port and subset.
- Record every deviation, every crashed attempt and every policy query.
- Commit small and often with the prefix `todo1b:`. Pull with rebase before each push;
  instance B pushes to the same branch.
