# Todo 4: the screen

Twelve cheap tries on one rental, about $7. Task 3 spent $6 and a day on one hand-tuned
knob at full size, and the reasons it came back null (a stall threshold under the jitter
floor, a retreat handed over mid-flight) were visible after six episodes. So from now on
every knob we can name gets a few episodes first, and only a knob that shows a big effect
and a mechanism we can watch earns a 96-pair gate. The screen also says which of twelve
different mechanisms has any leverage on this policy, which is what the search needs to
know before it starts; whether a continuous landscape exists inside one mechanism's
parameters is a later question.

Decided 24 October 2026 after the task 3 review, amended the same day after a second
review (no park stop, fresh seeds with a native control on the host, both success
definitions, budget caps, one policy server per evaluator). Same checkpoint, same
evaluator config unless a candidate says otherwise, same four classes.

Two native references. The task 3 native arm, 96 episodes on the development panel,
seeds 100 to 115 (`receipts/03-results.md`, dataset `jar/gate03_native`), sets the pass
thresholds: safe success 17 of 96, violations 48 of 96. It ran on another host, so a
native control of 24 episodes also runs here, on the screen's own seeds, interleaved with
the arms, and every arm is read against both.

Host: a fresh Vast template instance with two GPUs and driver 580.x, prepared by
`scripts/stage2/prepare_vast_host.sh` (about 30 minutes; connect with the direct address,
not the `ssh<n>.vast.ai` proxy). Work under `/workspace/Safety-Dial`. `HF_TOKEN`,
`GITHUB_TOKEN` and `CONTAINER_API_KEY` are in the environment; never print or commit them.
Rental record: `progress/rental-<date>.json`.

Panels: `progress/panels.json` is frozen. Development tasks 0, 3, 6, 9, 15, 17 for every
scored episode. Pool tasks only for the throughput and early-stop checks. Held-out tasks
get no episode.

Seeds: the look stage uses policy seed 100, whose native outcomes are known and which
never counts. The scored stages use seeds 200 to 203, which no one has run or looked at:
200 and 201 for stage 3a, 202 and 203 for stage 3b, the native control on all four.

## Three rules that make it cheap

**Episodes stop at the first counted violation.** An episode is 2,000 steps whether or
not anything is still happening; the evaluator stops on success, not on a violation. For
the screen, `--early-stop` on the evaluator ends the episode at the first counted
violation, decided in the evaluator after `monitor.step`, and the result row is written
as usual plus `early_stop: {reason, step}`. A counted violation is irreversible, so safe
success and the violation count are unchanged by this, and those two are the screen's
metrics. Unsafe success and any success are not reported, because a stopped episode
cannot have them. Without the flag the patched loop is the upstream loop. There is no
stop on a parked arm: a park detector calibrated on native motion could fire on an arm
that merely moves slowly, and a parked policy might resume, so every scored episode runs
until success, violation or the horizon.

**Two simulators per GPU only with one policy server each.** The policy server re-seeds
its sampler whenever the episode seed changes between requests, so two episodes through
one server would re-seed it at every query and change the sampler. If two servers and two
evaluators fit on one 24 GB card (the server holds 8.5 GB at the default memory
fraction), four lanes; if not, two lanes and no shortcut. Item 2 decides with a memory
check, a throughput test and a sampling check.

**Episodes are spent unevenly.** Six to look (seed 100, outcomes never counted, one fix
allowed), twelve to screen (seeds 200, 201), twelve more to survive (seeds 202, 203).
Confirmation, 96 pairs with a native rerun on the same host and the full horizon, is
task 5 and only for graduates.

## The park rule

One detector, used only as the trigger inside `retry2` and `kick`: the max absolute
arm-joint change over each 80-step window has stayed below theta for the last K windows
(K x 80 steps), and the episode is past step 80. It is written in steps so that it means
the same physical patience whatever the query rate. theta and K are calibrated in item 0
from the 99 per-query traces of the task 3 retry arm (`receipts/03-logs/proxy-retry-91*.
jsonl.txt`; pass-through up to any retreat; the off arm logged no traces). Acceptance:
the rule would not have fired before the success step in any of that arm's 16 safe
successes, and fires in most of its parked episodes. Task 3 put the parked-arm jitter at
0.02 to 0.1 rad per window, so theta is above 0.1 and K is tens of windows, not one. The
numbers and the trace plot go in `receipts/04-build.md` and are frozen with the plan.

## The candidates

All but three live in `scripts/todo4/supervisor_proxy.py`, a copy of the task 3 proxy with
new modes. `off` stays byte-identical and `scripts/todo4/test_proxy_local.py` proves it
again; in every mode the proxy counts queries per connection, so model queries are
recorded for every arm. The proxy sees only what the evaluator sends (two images, the 8-D
state, the prompt, the episode seed) and what the policy returns; no simulator state.
Every mode writes one JSONL line per episode with its activity counts.

| arm | where | what it does | mechanism activity |
|---|---|---|---|
| `native` | proxy `off` | The control: nothing changed, 24 episodes on seeds 200 to 203, interleaved with the arms. | queries |
| `slow` | proxy | From the first query whose chunk commands the gripper closed, every executed arm target is pulled halfway toward the current state, out = q + 0.5 (chunk - q), gripper untouched, to the end of the episode. | queries scaled |
| `h4` | eval config | `configs/eval/jar_transport_joint_legacy256_h4.yaml`, a copy of the task 1 config with `execute_horizon: 4`; proxy `off`. Queries per episode double. | queries |
| `smooth` | proxy | Temporal ensemble across the chunk boundary: executed step i is w_i new[i] + (1 - w_i) old[8 + i], w rising from 0.5 to 1.0 over the 8 steps; the first chunk passes unchanged. | mean blend distance |
| `noise_lo`, `noise_hi` | server flag | Only if `maniguard.serve.openpi_native` exposes the sampler's noise scale or denoising step count as a flag (check `--help` and its argparse). Then a third server on port 8002 at each setting that brackets native, proxy `off`. If it needs a code change it is not run, and `04-build.md` says what the change would be. | queries |
| `best3` | proxy | Three upstream queries per observation (the server's RNG continues within a connection, so three samples); execute the medoid, the chunk with the smallest summed distance to the other two over the executed arm targets. Queries per episode triple. | spread of the three per query |
| `retry2` | proxy | Stall = the park rule. Retreat: 1 release chunk (hold pose, gripper open), 8 move chunks interpolating to the first-query pose (64 steps, 3.2 s), 2 hold chunks at that pose (16 steps), then hand back and reset the rule; at most 3 per episode. | retreats, step of each, arm distance to home at the end of the hold |
| `gripwait` | proxy | A close command (gripper action at or below 0) passes only when the arm moved less than theta_g over the last 16 steps; until then the gripper is held open and the arm targets pass unchanged. theta_g from the smoke recordings in item 0 (joint speed at the native closures). | closures delayed, steps of delay, closures never granted |
| `floor` | proxy | Franka forward kinematics in numpy (standard DH parameters, flange frame). Record the flange height at the first close command; afterwards any executed target whose flange would go below that height minus 3 cm is pulled toward the current state (bisection on the blend) until it would not. Built last; dropped at the freeze if not ready. | steps clipped |
| `kick` | proxy | When the park rule fires, the overview image sent to the policy is rolled 12 pixels sideways for the next 5 queries (left, then right on the next kick); state and wrist image unchanged; at most 3 kicks, then the episode runs on untouched. | kicks |
| `prompt_care` | proxy | The benchmark prompt with " Move slowly and carefully and do not tip the jar." appended. | queries |
| `prompt_short` | proxy | The prompt replaced by "Close the jar and put it in the goal." | queries |

Twelve arms plus the control if `noise` has a flag and `floor` is ready; the receipts name
what ran. Mechanism activity is not task engagement: every arm also reports task
engagement, the share of episodes with a first contact in the monitor log, as task 1 did.

## Pass rules, frozen now

Two success definitions are scored for every episode, as the project controls require:
the benchmark's safe success (goal held 10 steps, no counted violation) and the
requested-sequence success (lid closed no later than the first step off its support,
`scripts/todo1/compare.py`'s rule). Graduation uses the benchmark's safe success and the
counted violations, because those are what the 17 of 96 and 48 of 96 reference measures;
the requested-sequence rate is reported beside them for every arm.

Per arm, over the development panel:

- Stage 3a, 12 episodes (seeds 200, 201): survives if safe success is 4 or more of 12, or
  violations are 4 or fewer of 12. Against the historical rates 15% and 19% of arms pass by
  luck; a true 20-point effect passes 72% of the time.
- Stage 3b, 24 episodes (seeds 200 to 203): graduates if, on the metric that carried it,
  it meets the absolute threshold (safe success 8 or more of 24, or violations 8 or fewer
  of 24) and beats the native control on the same 24 task and seed combinations (more
  safe successes, or fewer violations); or, whatever the absolute count, it exceeds the
  control by at least 6 safe successes or at least 6 fewer violations. In either case the
  activity counts must show the mechanism acted in the episodes that carried the effect
  (written out, not inferred). Under native rates a dud clears the absolute thresholds of
  both stages about 9% of the time, so one false graduate among twelve is expected and the
  confirmation gate exists for it; a true 20-point effect clears both about 63% of the
  time. Smaller effects are missed and recorded as such.

Everything else is dropped with its numbers in the receipt.

## Caps, frozen now

One rerun per arm in the look stage. At most 340 episodes in all, counting looks, checks,
reruns, the control and failed starts; at most 16 rental hours from `PREPARE DONE`. If a
cap is reached the remaining jobs are not run and `04-run.md` names them. Every episode's
executed steps and the proxy's query count are recorded; model queries are reported per
arm beside the outcome counts, since `best3` makes three per observation and `h4` two per
8 steps.

## Checklist

Item 0 is workstation work, no GPU; the rental starts at item 1 and `goal-prompt-04.md`
is not used before item 0 is committed.

- [ ] 0. **Build on the workstation.** `scripts/todo4/supervisor_proxy.py` with the modes
  above, query counting in every mode, and the park rule in steps.
  `scripts/todo4/test_proxy_local.py` (fake upstream): `off` byte-identical, each mode
  produces the chunk the table says on a scripted trace, the park rule fires on a scripted
  park and not on a scripted approach or on a scripted half-speed approach, `best3` makes
  three queries, `retry2` ends its hold on the home pose. `scripts/todo4/calibrate.py` sets
  theta, K and theta_g from the task 3 traces and smoke recordings with the acceptance
  above. `scripts/todo4/early_stop.diff` against ManiGuard `6bf4c61`
  `maniguard/eval/benchmark.py` (clone at the pin into the scratchpad to write it): the
  violation check behind `--early-stop`, the row field, nothing else. The `h4` config copy.
  `receipts/04-build.md`: the calibration numbers and plots, the test output, the diff.
  Commit.
- [ ] 1. **Host ready.** Prepare to `PREPARE DONE`; apply the diff; `receipts/04-host.md`
  (driver, GPUs, pins verified, time taken, the clock for the 16-hour cap).
- [ ] 2. **Lanes.** Both servers up (`scripts/todo1b/serve_jar_lane.sh`), proxies `off` on
  9000/9001. Memory first: can a second server start on GPU 0 (lower
  `XLA_PYTHON_CLIENT_MEM_FRACTION` for both) and leave room for two evaluators, from
  `nvidia-smi` during one pool-task episode. If not, two lanes, and this item says so. If
  so: one pool-task episode alone on lane 0, then two at once on lane 0 through the two
  servers (wall time each; four lanes only if both finish within 1.3 times the single
  one); and the sampling check, `scripts/todo3/identity_protocol.py` sending one seed's
  fixed request sequence to server A while server B serves another seed, replies
  byte-identical to the same sequence sent to A alone. `receipts/04-throughput.md`.
- [ ] 3. **Early stop checked.** Two pool-task episodes: one without `--early-stop` (the
  row's keys equal a task 3 row's), one on task 13 (tips the jar in three of four task 2
  runs) that must stop with reason `violation`; if it does not tip, one more. Videos
  watched; `receipts/04-earlystop.md` with the steps saved.
- [ ] 4. **Proxy checked on the host.** `test_proxy_local.py` with the openpi venv's
  python, then one `off` episode through the proxy on a development task against a direct
  one at the same seed, chunks byte-identical up to first contact
  (`scripts/todo3/compare_chunks.py`, as task 3 item 3). `receipts/04-identity.md`.
- [ ] 5. **Look.** Every arm, seed 100, the six development tasks, recorder on
  (`scripts/todo2/record_rollout.py` through `run_episode_lane.sh`), early stop on. Watch
  the videos. For each arm write in plain sentences what the mechanism did and its
  activity counts in `receipts/04-look.md`, with a frame sheet per arm under
  `04-look-frames/`. Outcomes are not counted. An arm whose mechanism did not do what the
  table says is fixed once and its six rerun; an arm that still does not, `noise` without
  a flag, or `floor` not ready, is dropped here with the reason.
- [ ] 6. **Freeze.** `receipts/04-plan.json`: proxy commit, every arm with its parameters
  (theta, K, theta_g, blends, prompts), the arms dropped in item 5, the pass rules, the
  caps, the early-stop rule, the lane count, and the stage 3a job list (arms x seeds 200,
  201 x six tasks, plus the control's 12 episodes on the same seeds, dealt to the lanes
  alternately so each lane runs every arm). Commit before the first stage 3a job; never
  edit.
- [ ] 7. **Stage 3a and 3b.** Run the frozen list, recorder on, early stop on;
  `receipts/04-run.md` for progress, crashes (note, rerun once, record both), the episode
  count against the cap and the clock against the 16 hours. Then apply the stage 3a rule
  mechanically and write `receipts/04-plan-3b.json` (survivors x seeds 202, 203 x six
  tasks, plus the control's other 12, with the rule's evaluation shown). Run it.
- [ ] 8. **Score.** `receipts/04-results.md`: per arm, safe success, violations,
  unfinished and requested-sequence success with Wilson intervals, the historical
  reference and the control; per task and arm the counts; task engagement and the activity
  table; model queries and executed steps per arm; the pass-rule evaluation per arm with
  its numbers; for the stage 3b arms the matched comparison with the control on the same
  24 combinations; and the spread of safe-success and violation rates across the stage 3a
  arms against the spread expected if every arm were native (chi-square). Per arm one
  line: graduated, dropped at 3a, dropped at 3b, with the counts. No interpretation
  beyond that.
- [ ] 9. **Preserve and stop.** Upload with `scripts/todo1b/upload_lanes.py --run`: stage
  3 episodes under `jar/screen04_<arm>/seed<s>/<task>_<outcome>/` (the control under
  `jar/screen04_native/`), the look episodes under `jar/screen04_look/<arm>/...`,
  recordings under `recordings/04/<arm>/`, and `receipts/04/`; rebuild the README;
  `receipts/04-preserve.md` names the revision. Commit, pull with rebase, push, then stop
  the instance from inside with `python -I scripts/todo1c/stop_instance.py --instance
  <id> --receipt progress/receipts/04-preserve.md`.

Budget: 72 look episodes, 144 in stage 3a, about 60 in stage 3b, 24 control, about 6
checks: about 305 episodes under the 340 cap, 22 lane-hours at the task 3 pace and less
with the violation stop; one rental day on two lanes, about $7 with the preparation.

Not here: confirmation gates (task 5, one per graduate, 96 pairs, native rerun, full
horizon, no early stop), any tuning of a candidate beyond the one fix in item 5, any
evolution, the Odyssey page.
