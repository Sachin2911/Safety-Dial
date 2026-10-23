# Todo 1c: seed 0 again, twice, on instance C's machine (instance D)

Purpose: seed 0 on instance B (2 / 1 / 8 / 15) sits below seeds 1 and 2 on instance C
(6 / 0 / 10 / 10 and 4 / 2 / 10 / 9). The three-seed replication passed its frozen test
([receipts/01-review.md](receipts/01-review.md)), but we cannot tell a low draw from a
host effect, because B was a different machine with a different driver patch and the
renderer is not bit-reproducible. This instance is the same machine that ran seeds 1
and 2 (machine 54514). It runs seed 0 on both GPUs at once: two independent repeats of
the same seed on the same host. Repeat against repeat measures same-host noise; both
repeats against B's seed 0 measure the host effect on top of that noise. Nothing else
is run here.

Host: Vast instance 57334420, 2x RTX 3090, driver 580.173.02, 256 GB RAM, 20 cores,
200 GB disk ([rental record](rental-20261022.json)). Work under `/workspace`.
`HF_TOKEN` and `GITHUB_TOKEN` are in the environment; never print or commit them.

## Checklist

- [ ] **1. Install by replaying instance C's recipe.** This is C's machine, so follow
      [receipts/01b-install.md](receipts/01b-install.md) step by step with
      `scripts/todo1b/install_behavior_replay.sh` and
      `scripts/todo1b/install_maniguard_deps.sh`, at the revisions in
      [receipts/01-sources.json](receipts/01-sources.json), downloads through
      `scripts/todo1/download_pinned.py` with hashes verified. First check
      `ldconfig -p` for OptiX, EGL, GL and Vulkan; if OptiX or EGL are missing run
      `/opt/instance-tools/bin/install-display-drivers`, as C needed. The NVIDIA EGL
      vendor selection is already in `scripts/todo1b/run_jar_seed_lane.sh` and
      `run_smoke_lane.sh`. Freeze both environments and diff the freezes against
      `receipts/01-env-*-freeze.txt` and C's. Write `receipts/01c-install.md` with any
      difference from C.
- [ ] **2. Smoke one episode per GPU, both at seed 0.** One policy server per lane
      (`scripts/todo1b/serve_jar_lane.sh`: `CUDA_VISIBLE_DEVICES=0` port 8000,
      `CUDA_VISIBLE_DEVICES=1` port 8001), each warmed with
      `scripts/todo1/probe_server.py`. Run `task_0000/base` at `SEED=0` on each lane with
      [receipts/01-jar-eval.yaml](receipts/01-jar-eval.yaml), tagged `smoke`. Both must
      reach a verdict with 256 by 256 videos. Record wall time, peak GPU memory, and the
      outcome next to B's seed 0 task_0000 (unsafe unsuccess, violation at step 1016) in
      `receipts/01c-smoke.md`, rows in `receipts/01c-smoke.jsonl`. Count these episodes.
- [ ] **3. Run seed 0 on both lanes concurrently.** Lane 0: `GPU=0 PORT=8000 bash
      scripts/todo1b/run_jar_seed_lane.sh 0 jar_transport_joint_legacy256/seed0-rerun1`.
      Lane 1: `GPU=1 PORT=8001 bash scripts/todo1b/run_jar_seed_lane.sh 0
      jar_transport_joint_legacy256/seed0-rerun2`. All 26 base tasks in each lane, `TASKS`
      unset, each lane in its own tmux session with its own log and the GPU sampler
      running. Do not split one repeat across lanes; keep each repeat's
      `expected_rollouts.json` and `summary.json` whole. Expect about 2 hours.
- [ ] **4. Collect.** Run upstream `tools/eval_summary.py --full --json` on each repeat;
      copy `results.jsonl`, `summary.json`, `eval_config.json`, `expected_rollouts.json`,
      `eval_summary.json`, the runner log and `lane_timing.json` under
      `receipts/01-runs/seed0-rerun1/` and `seed0-rerun2/`. Run `scripts/todo1/compare.py
      --run rerun1=<dir> --run rerun2=<dir> --json receipts/01c-compare.json`. Write
      `receipts/01c-results.md` with: four-class counts and engagement for rerun 1, rerun 2,
      their pool, B's seed 0 ([receipts/01-runs/seed0/results.jsonl](receipts/01-runs/seed0/results.jsonl))
      and seeds 1 and 2; a per-task table with the class, violation step, first contact and
      `ever_grasped` of B's seed 0, rerun 1 and rerun 2 side by side; the number of tasks on
      which each pair agrees on class (B vs rerun 1, B vs rerun 2, rerun 1 vs rerun 2);
      violation triggers per repeat; wall time, per-episode timing and peak memory per
      lane; policy-server query counts; and every deviation. Report the two-sided Fisher
      values for the pooled reruns (52) against B's seed 0 (26) on safe success and on
      violations, and for each repeat against the published row, as information only.
      The judgment is made after review, not here.
- [ ] **5. Preserve.** Upload both repeats' full trees, every video and LTL sidecar, plus
      both smoke episodes and the 01c receipts, to the same private dataset
      `Sachioster/maniguard-jar-replication-20261022` under `jar/seed0-rerun1/`,
      `jar/seed0-rerun2/`, `smoke/seed0-rerun1/`, `smoke/seed0-rerun2/` and
      `receipts/01c/`, in the same per-task layout as `jar/seed0/`. Extend
      `scripts/todo1b/upload_lanes.py` only as far as those prefix names need (its
      `--seed` and `--smoke` specs already form `jar/seed<name>`); each prefix is
      uploaded with its own `path_in_repo` and nothing outside these prefixes is written
      or deleted. Rebuild the README index over every run present. Verify the upload
      from the instance (file counts, task folders, videos) and record the final
      revision in `receipts/01c-preserve.md`.
- [ ] **6. Push and stop.** Commit, pull with rebase, push to main, and stop. **This
      instance is destroyed after this item lands**, so nothing may remain only on
      local disk.

## Rules

- Official evaluator per task, unchanged. The lane wrapper only chooses port and subset.
- Record every deviation, every crashed attempt and every policy query.
- Commit small and often with the prefix `todo1c:`; push after each checklist item.
- Nothing beyond this checklist runs on this instance.

Budget: 2 smoke episodes plus 52 family episodes, about 45 minutes of install and
2 hours of lanes, roughly $2.
