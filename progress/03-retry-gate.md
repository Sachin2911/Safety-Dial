# Todo 3: the retry gate

One question, one treatment, about $5. Does "lift and try again when stuck" raise safe
completion on Jar without raising violations? Decided 23 October 2026 after the deep
research in `docs/research/reports/Evolutionary methods for frozen VLA safety.md`:
retry is the knob with the best published track record, so it is the first knob of
the Safety Dial to test. Nothing else changes: same checkpoint, same evaluator, same
scoring, full 2,000-step horizon.

Host: a fresh Vast template instance with two GPUs and driver 580.x, prepared by
`scripts/stage2/prepare_vast_host.sh` (about 30 minutes; connect with the direct address
from `vastai ssh-url <id>`, not the `ssh<n>.vast.ai` proxy). Work under
`/workspace/Safety-Dial`. `HF_TOKEN`, `GITHUB_TOKEN` and `CONTAINER_API_KEY` are in the
environment; never print or commit them. Rental record: `progress/rental-<date>.json`.

Panels: `progress/panels.json` is frozen. Development tasks 0, 3, 6, 9, 15, 17 only.
Held-out tasks get no episode.

## The supervisor (defaults fixed now, not tuned here)

A proxy policy server, `scripts/todo3/supervisor_proxy.py`, sits between the evaluator
and the real policy server. The evaluator connects to the proxy; the proxy forwards every
observation to the real server (`openpi_client` websocket protocol, one connection per
episode, seeding forwarded untouched) and returns the chunk it gets back. Two modes:

- `off`: pure pass-through, byte-identical chunks. This is the native arm.
- `retry`: the pass-through plus three parts, all from the 8-D state the evaluator
  already sends (seven arm joints, one gripper), no simulator state.
  - Stall detector: max absolute arm-joint change over the last 10 queries (80 steps,
    4 s at 20 Hz) below 0.01 rad, and the episode is past query 10.
  - Retreat: open the gripper and interpolate the joints back to the pose seen at the
    episode's first query, as 4 chunks of absolute joint targets (32 steps); then hand
    back to the policy and reset the detector.
  - Limit: at most 3 retreats per episode.

The proxy writes one JSONL line per episode: mode, port, queries, retreats fired, query
of each retreat, and the joint-change trace at each query.

## Checklist

- [ ] 1. **Host ready.** Run the prepare script to `PREPARE DONE`; write
  `receipts/03-host.md` (driver, GPUs, pins verified, time taken).
- [ ] 2. **Check the proxy.** `scripts/todo3/supervisor_proxy.py` is written and passes
  `scripts/todo3/test_proxy_local.py` on the workstation (fake upstream, no GPU). Run that
  test here with the openpi venv's python (`third_party/openpi/.venv/bin/python -I`), then
  start the lane 0 server (`GPU=0 PORT=8000 bash scripts/todo1b/serve_jar_lane.sh`) and the
  proxy on 9000 in `off` mode with `--log /workspace/logs/proxy-off-9000.jsonl`. Do not
  change the defaults.
- [ ] 3. **Identity check and recorder validation.** Run `task_0000/base` seed 0 twice:
  direct on 8000, then through the proxy on 9000, both with the todo 2 recorder
  (`scripts/todo2/record_rollout.py`, `--recording-factory record_rollout:create_observer`).
  Pass: `scripts/todo3/compare_chunks.py <direct.npz> <proxied.npz>` prints PASS (every
  chunk byte-identical up to and including the first finger contact); the recorder's `.npz`
  loads and passes `scripts/todo2/check_recording.py`. Write `receipts/03-identity.md`. The recorder stays
  off for everything after this item.
- [ ] 4. **Retry smoke.** One episode in `retry` mode on a development task whose native
  episodes stalled (pick from `receipts/02-timelines.md`). Watch the video: does the
  retreat fire when the robot is idle, does the arm go home cleanly, does the policy
  resume? Write what you see in plain sentences in `receipts/03-smoke.md`. If the
  retreat itself tips the jar, say so and stop; do not change the defaults.
- [ ] 5. **Freeze the plan.** `receipts/03-plan.json`: 6 tasks x policy seeds 100 to 115
  x 2 arms = 192 jobs, ordered by seed then task with the arms alternating, dealt to
  the two lanes alternately so each lane runs both arms equally. Commit it before any
  job runs; never edit it.
- [ ] 6. **Run the gate.** Per lane: real server on 8000/8001, proxy `off` on 9000/9001,
  proxy `retry` on 9100/9101. Each job is one episode through
  `scripts/todo1/eval_family_lane.sh` with `TASKS`, `SEED` and `PORT` set from the plan
  and output leaf `gate03/<arm>/seed<s>`. Full horizon, benchmark unchanged. Keep
  `receipts/03-run.md` updated with progress and any crash (note it, rerun that job
  once, record both).
- [ ] 7. **Score.** `receipts/03-results.md`: per task and overall, the four classes per
  arm; safe success and violation rate with Wilson intervals; the paired table
  (native vs retry per seed and task); McNemar on safe success and on violations;
  engagement rate per arm; retreats fired per episode. One decision line: retry moves
  safe success (yes/no, p), violations rose (yes/no, p). No interpretation beyond that.
- [ ] 8. **Preserve and stop.** Upload both arms to the private dataset
  `Sachioster/maniguard-jar-replication-20261022` under `jar/gate03_native` and
  `jar/gate03_retry` with the `<n>_<outcome>` folder convention (extend
  `scripts/todo1b/upload_lanes.py` with a `--run <prefix>=<dir>` option rather than a
  new script), plus `receipts/03/`; rebuild the README. `receipts/03-preserve.md` names
  the dataset revision. Commit, pull with rebase, push, then stop the instance from
  inside with `python -I scripts/todo1c/stop_instance.py --instance <id> --receipt
  progress/receipts/03-preserve.md`.

Not here: the slow regrasp, the chunk prefix, early stopping, any evolution, the Odyssey
page (written after review on the workstation). Power: 96 pairs see a change of about
15 points; a smaller true effect can come back as "no" and that is recorded as such.
