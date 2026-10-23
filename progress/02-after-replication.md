# Todo 2: look at what actually happened

Purpose: before any intervention is tried, describe the 104 episodes of record so well
that the evidence, not a prior idea, says where leverage exists. Two layers, kept
apart. Layer 1 is pure description: what happened, when, how repeatably, and what
differs between a success and a failure of the same task. Layer 2 maps that onto every
family of intervention we might build (observation-side noise or augmentation, the
policy's sampling noise or latent, an adapter, an execution dial or action filter, an
outer loop that scores or selects candidate actions such as a world model or critic,
monitor-and-retreat or retry) and ranks where the evidence says to look first. Nothing
is decided here and no intervention runs here.

Data: dataset `Sachioster/maniguard-jar-replication-20261022` (private), revision
`9300cebf`: runs `jar/seed0_run2`, `jar/seed0_run3` (the same policy seed twice, same
machine), `jar/seed1`, `jar/seed2`; task folders `<n>_<outcome>` with `base_main.mp4`
(what the policy saw, 256 x 256, frame i is step i), `base_wrist.mp4`, `base_ltl.json`
(per-step atomic propositions `jar_on_support`, `jar_closed`, `jar_dropped`,
`jar_upright`, violation step and clause) and `row.json`. Results rows are also in
`progress/receipts/01-runs/<run>/results.jsonl`. Published and measured rates:
[Jar replication and noise floor](../docs/research/research_notes/Jar%20replication%20and%20noise%20floor.md).

Host: RunPod pod `f1o65q3iygv372` (secure cloud, one A40 48 GB, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 150 GB volume at
`/workspace`); its id is `$RUNPOD_POD_ID` in the environment
([rental record](rental-runpod-20261023.json)). Work under `/workspace`. `HF_TOKEN`,
`GITHUB_TOKEN` and `RUNPOD_API_KEY` are in the environment; never print or commit them.
Only item 7 needs the GPU and the full ManiGuard install; items 1 to 6 need the image's
`python3` with `huggingface_hub`, `opencv-python-headless`, `numpy`, `pandas`,
`matplotlib` (`pip install`). This is not the Vast image: there is no `/venv/main`, no
`/opt/miniforge3` and no `/opt/instance-tools`; install Miniforge under
`/workspace/miniforge3` when item 7 needs conda.

## Panels (frozen now, by rule, before any further looking)

Middle tasks, one to three safe successes in the four runs: 0, 2, 3, 5, 6, 8, 9, 14, 15,
16, 17, 23. Alternating assignment in ascending order gives:

- **Development panel (6):** 0, 3, 6, 9, 15, 17 (9 safe successes in 24 episodes).
- **Held-out panel (9):** 2, 5, 8, 14, 16, 23 (middle, 11 of 24), plus 7 and 18 (the two
  lowest-numbered tasks that violated in all four runs; a dial that helps here helped
  where there was no chance) and 12 (safe in all four runs; a dial must not break it).
- **Pool (11):** 1, 4, 10, 11, 13, 19, 20, 21, 22, 24, 25. Free for diagnostics, never
  for scoring an intervention.

Write these to `progress/panels.json` in item 1 and do not change them. Held-out tasks
may be described in this todo (their episodes already exist) but no new episode is run
on them until the final comparison.

## Checklist

- [x] **1. Fetch and index.** Download the four runs' `base_ltl.json`, `row.json`,
      `base_main.mp4`, `base_wrist.mp4` and `index.json` (about 0.9 GB) at revision
      `9300cebf` into `/workspace/episodes/`. Check 104 sidecars, 208 videos, frame
      count equals steps plus one on every main video. Write `progress/panels.json`
      from the panel section above. Receipt: `receipts/02-fetch.md`.
- [x] **2. Episode timelines.** Write `scripts/todo2/timeline.py`: for every episode,
      from the sidecar log and the row, the step of first contact, first `jar_closed`,
      first `!jar_on_support` (lift), first `!jar_upright` (tip), first `jar_dropped`,
      the violation step and clause, the success step, and the end; then one story
      label per episode from a fixed rule written in the script: `no_contact`;
      `stall` (contact, nothing else); `tip_before_close`; `tip_after_close_on_table`;
      `lift_open` (lifted before closed); `carry_fail` (closed, lifted, then tipped,
      dropped or timed out); `safe_success`; `unsafe_success` (with the clause).
      Output `receipts/02-timelines.jsonl` (one line per episode) and
      `receipts/02-timelines.md`: the histogram of stories overall and per run, the
      timing distributions (contact to violation, contact to closure, closure to lift,
      lift to success), and the same histogram restricted to each panel. Note any
      episode whose story the rule cannot assign.
- [x] **3. Frames.** Write `scripts/todo2/frames.py` (OpenCV). For every episode take
      the main-camera frame at step 0, first contact, first contact + 40, the key
      event (violation step, else first lift, else success step, else last) and the
      last step; label each with run, step and event. Tile one sheet per task, four
      runs as rows, under `receipts/02-frames/task_<nn>.png` (26 sheets, about
      1280 x 1024 each). For the 15 panel tasks also tile the wrist camera at the same
      steps under `receipts/02-frames/task_<nn>_wrist.png`. Look at every sheet and
      write one to three plain sentences per task in `receipts/02-task-notes.md`: what
      the robot does, where it goes wrong, whether the four runs look alike.
- [x] **4. Sensitivity map.** Write `scripts/todo2/sensitivity.py`. Per task, from the
      four runs: do the two same-seed repeats share a story; do all four; the spread
      of the violation step; and the divergence step of the same-seed pair, measured
      two ways, the first step at which the two sidecars' proposition vectors differ,
      and the first step at which the mean absolute difference between the two main
      videos exceeds three times its value over steps 0 to 20. Classify each task
      `knife_edge` (same-seed repeats disagree), `systematic` (all four share a story
      and their key-event steps lie within 50 of each other) or `mixed`. Output
      `receipts/02-sensitivity.md` with the table and a short reading: which panel
      tasks are knife-edge, which are systematic, and what that implies for a small
      push versus a systematic correction.
- [x] **5. Success against failure on the same task.** For every task with at least one
      success and one failure in the four runs (prefer the same-seed pair where it
      splits: tasks 0, 3, 8, 9, 16, 17, 20, 23), build a side-by-side strip, main and
      wrist, at matched steps from first contact minus 50 to first contact plus 250
      every 25 steps, under `receipts/02-pairs/task_<nn>.png`. Look at each and write
      in `receipts/02-pairs.md` what differs and when it first becomes visible:
      approach height, side, speed, gripper timing, where on the jar the fingers land.
      Say for each pair whether the difference is visible before contact (the policy
      already has a good and a bad mode; steering between them is possible) or only
      at contact (a physics coin flip; the lever is execution or recovery).
- [ ] **6. What the evaluator can record.** Clone ManiGuard at `6bf4c61` (code read
      only, no install yet) and list in `receipts/02-logging.md` exactly what the
      per-task evaluator writes per step today (`rollout_diagnostics`, `metrics`,
      the LTL sidecar, videos) and what it does not: the 16-step action chunk the
      server returns, the 8 executed actions, robot joint positions and velocities,
      gripper state, jar and lid poses, contact flags. Find the cleanest hook (an env
      var or config flag, a wrapper around the client call, or a callback) to save
      those per step to one `.npz` per episode without touching scoring, and write
      `scripts/todo2/record_rollout.py` or the patch for it. Any change to upstream
      files is kept as a diff under `receipts/02-logging.diff`.
- [ ] **7. Validate the recorder on one real episode.** First check the renderer can
      run in this container: `ldconfig -p | grep -E 'libnvoptix|libEGL_nvidia|libnvidia-rtcore|libGLX_nvidia'`
      and the Vulkan ICD under `/usr/share/vulkan/icd.d` or `/etc/vulkan/icd.d`. If
      OptiX or EGL are missing, write what is missing to `receipts/02-host.md`, push,
      and skip to item 8 (the recorder is then validated on the next Vast rental);
      do not try to install display drivers in the container. Otherwise replay the
      install from `receipts/01b-install.md` and `01-sources.json` at the pinned
      revisions (about 45 minutes) with Miniforge under `/workspace/miniforge3` and
      `CONDA_ENVS_PATH=/workspace/miniforge3/envs`; the Vast-only steps there
      (`/opt/instance-tools`, `/venv/main`) do not apply, and the NVIDIA EGL vendor fix
      in `scripts/todo1b/run_smoke_lane.sh` is needed only if GLVND dispatches to Mesa.
      Start one policy
      server, warm it with `scripts/todo1/probe_server.py`, and run `task_0000/base`
      at seed 0 once with the recorder on and the config in `receipts/01-jar-eval.yaml`,
      tagged `smoke`. Check the `.npz` has one entry per step with the fields above,
      the scoring is unchanged (compare the row to a normal smoke row's fields), and
      record file size per episode and the time overhead. Count the episode. Stop the
      server. Receipt: `receipts/02-logging.md` (same file, new section).
- [ ] **8. Where the leverage is.** Write
      `docs/research/research_notes/Where the leverage is.md`. For each intervention
      family listed in the purpose, say what items 2 to 5 show about whether it has
      leverage and on which tasks, quoting the numbers, and what it would need
      (per-step logging from item 6, a critic, a retreat primitive). End with a ranked
      shortlist of three stage 2 experiments, each with the question it answers, the
      panel, the episode count at the measured noise (about 200 per condition to see
      ten points), and the cost at 4.5 minutes per episode. Rank them; do not choose.
- [ ] **9. Preserve, push, stop.** Upload `receipts/02-frames/`, `02-pairs/`,
      `02-timelines.jsonl` and the item 7 `.npz` to the dataset under `analysis/02/`
      (own `path_in_repo`, nothing else touched) and record the revision in
      `receipts/02-preserve.md`. Commit, pull with rebase, push, confirm
      `git rev-parse HEAD` equals `origin/main`, confirm `nvidia-smi` shows no compute
      process, then run `python -I scripts/todo2/stop_pod.py --receipt
      progress/receipts/02-preserve.md --dry-run` and, when every guard passes, the
      same without `--dry-run`. That is the last action; the pod is unreachable once
      stopped and the reviewer terminates it afterwards.

## Rules

- Describe; do not intervene. No noise, no dial, no modified actions. The only
  simulator episode on this instance is the item 7 recorder check.
- Frames and sheets go in the repository as PNGs (keep each under 2 MB); videos and
  the `.npz` go to the dataset.
- Panels are frozen once written. Do not edit todo 1 receipts or `theOdessy.tex`; the
  reviewer writes the Odyssey entry.
- Commit small and often with the prefix `todo2:`; push after each item.

Budget: about five hours of work, one 45-minute install, one or two episodes, roughly $4 at $0.59/h.
