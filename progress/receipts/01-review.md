# Review of the item 1 source pins

Reviewed 22 October 2026 from the local workstation against the public ManiGuard
history, after commit `5f6bfb0` landed from the instance.

## Verdict

The pin to ManiGuard `6bf4c61` is accepted. It is the newest revision that still
renders the paper's legacy 256 by 256 observation path; HEAD removes that path. The one
deviation the receipt flags, that the evaluator code postdates the paper by two upstream
fixes, was checked and does not change how a completed Jar episode is scored.

## What the two fixes change

`0dca966 fix(safety)` (29 September 2026):

- Monitor and predicate failures that used to be logged and swallowed now raise.
  Previously a failed automaton initialization silently disabled safety monitoring and
  a failed predicate read returned False; both would have let an episode be recorded as
  safe without a working monitor. Now such episodes fail with a null verdict and an
  `initialization_failed` or runtime status instead of counting as safe.
- A new initial-safety check refuses to start a rollout whose initial state already
  violates the specification.
- The object resolver evaluates each bound object once and lets exact scene names take
  precedence over aliases. For Jar's single jar object this does not alter the verdict.
- Seed derivation (CRC-32 of base seed and scene name), the goal-hold counter and the
  engagement instrumentation are unchanged in substance.
- `eval_family.sh` gains `SEED` and `REPEAT` handling, fails loudly on missing
  scenario files instead of skipping them, and writes `expected_rollouts.json`.

`fede9de fix(bench)` touches only benchmark construction tooling. The Jar base scenes
read at `af09438c` were confirmed byte-identical to the paper-time revision, so this
fix cannot affect our rollouts.

## Consequence for the replication

A valid episode is scored the same way the paper scored it. The difference is that
episodes the paper's code might have silently mis-scored as safe will now fail
visibly. If the local safe-success count comes in slightly under the paper's, that
direction of error is one possible explanation and should be listed before blaming
the stack. If the count is near zero, this is not the cause.

## Other pins

BEHAVIOR-1K, OpenPI, the robot asset, the BEHAVIOR assets and the Jar checkpoint at
step 7400 match the October pilot and the paper-time checkpoint update. No concerns.

## Items 3 and 4 (reviewed after commit `33b2866`)

Accepted. The eval YAML differs from upstream's Jar config in exactly the two lines the
replication requires (`source_profile: legacy`, `camera_resolution: 256`); every other
value is upstream's, and the 256 by 256 first frame was confirmed from the saved videos,
which are written from the same observation array the policy receives.

The smoke episode ran to a verdict (unsafe unsuccess, violation at step 561, never
grasped) with both videos and the LTL sidecar written. The crashed first attempt is
counted.

Timing note for the budget: 780 s wall per episode, of which 332 s was a cold Isaac Sim
startup. `eval_family.sh` launches a fresh simulator process per task, so startup repeats
for every episode. Expect 9 to 13 min per episode, 4 to 5.5 h per seed and 12 to 16 h
for all three seeds, rather than the 5 min per episode assumed in the todo budget.

## Todo 1b items 1 to 4 on instance C (reviewed after commit `bac278f`)

Accepted. The install replay reached the same pins; the OpenPI freeze is byte-identical to
B's and the behavior freeze differs only in the temporary directory names of the Isaac Sim
wheel paths, with identical wheel hashes. The lane wrapper diff against upstream
`eval_family.sh` adds exactly a `PORT` variable and a `TASKS` subset filter; the evaluator
command is otherwise unchanged. C needed one host-specific repair beyond B: GLVND was
dispatching EGL to Mesa, so the NVIDIA EGL vendor file had to be selected explicitly, the
same fix the October rental on this machine recorded. Both lane smoke episodes reached
verdicts at 256 by 256; the seed 2 smoke on task_0000 was a safe success at step 982, the
first safe success seen on C. Peak GPU memory per lane with server and simulator was about
12.3 GB of 24 GB. The pre-launch check could not see B's seed 0 rows because B pushes only
on item completion; the reviewer's direct read of B at that time showed 1 safe success in
9 episodes, so the stop rule was not met and the launch was correct.

## Seed 0 and instance B retirement (reviewed after commit `f7c4ed6`)

Seed 0 is 2 safe success, 1 unsafe success, 8 safe unsuccess, 15 unsafe unsuccess over
26 tasks with 100% engagement. Against the paper's per-seed expectation of about 7 safe
successes and 10 violations this single seed is low: under the published rates the
chance of 2 or fewer safe successes is about 1.6% and of 16 or more violations about
1.4%. The receipt's own two-sided Fisher tests against the pooled published row give
0.055 and 0.067. Judgment is deferred to the pooled 78, as the frozen protocol requires;
at the time of this note seeds 1 and 2 on instance C were halfway through at roughly
25% safe success and 33% violations, close to the paper, so seed-to-seed variance or a
host effect (different driver patch, non-deterministic renderer) are both live
explanations alongside the receipt's list.

The agent's mid-run input checks (long-finger asset loaded, rendering matched to the
training frames, all scenes and monitors initialized) and its handling of the aborted
seed 1 start are accepted. The dataset upload was verified from the workstation:
private, revision `ccafd65a`, 138 files, 26 complete seed 0 task folders, 54 videos
none truncated. Instance B was destroyed afterwards per the user's instruction.
