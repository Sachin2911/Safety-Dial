# LeWM readiness and fixed-adapter screen, 31 October 2026

The fixed untrained adapter did not improve PushT hazard completion on this screen.
Native planning completed 3/18 hazard episodes safely; the adapter completed 1/18.
Cube failed its original-object-goal readiness check at 0/6 no-hazard completions,
so its hazard and untrained-adapter stages stayed closed. Neither environment passed
the predeclared gate for a future training study. No evolutionary updates ran.

This is the separate LeWM track, not a change to the ManiGuard research objective.
The preceding [residual study](lewm-residual-study.md) found a small selection signal
for untrained guidance in one known scene. This screen asked whether that signal
persisted across different start/goal pairs and hazard conditions before expanding
optimizer work.

## Frozen design and mechanism

The [prospective protocol](../../configs/lewm_screen_20261031/protocol.json) fixed
three physically different public source pairs per environment, four conditions
(no added hazard, island, bar and gate), and two planner seeds per condition. These
source poses had appeared in earlier development. Newly frozen geometry and seeds
do not make them globally untouched validation scenes. Final evaluation cases stayed
untouched.

LeWM task costs, learned motion models, safety margins and per-environment search
settings stayed frozen. Hazards were virtual constraints and did not alter physics
or enter the policy-input images. The adapter supplied bounded residuals that shifted
half the newly sampled SafeCEM candidates while retaining native slots and the same
online candidate budget. Every candidate still passed through the learned safety
checker. Protection covered the 15 px pusher disk with a 6 px margin in PushT and
a 5 mm pinch-point disk in world XY with a 1 cm margin in Cube. Cube monitoring
did not protect the whole arm or the carried object. PushT used the exact previously
frozen 510-parameter untrained controller.
Cube required a separately seeded 1,365-parameter representation for its five-action,
five-dimensional blocks. This is not cross-environment weight transfer.

Each environment first received two six-action native/zero engineering probes and
six native no-hazard controls. The readiness rule required exact zero/native
equivalence, legal executed commands, clean images, at least four safe completions
and no observed violations. Only qualifying environments could finish the 24-case
native grid and receive 24 matching untrained cases. The overall cap was 96 scored
episodes, four engineering probes and six preparation resets, with 150 actions per
scored episode. The fresh rental had a $1 cap, one-hour cutoff and ten-minute
preservation reserve.

## Recorded outcomes

PushT passed native readiness at 4/6 safe completions, 0/6 violations, 2/6 unfinished
and 6/6 engaged. Its full paired grid then completed:

- Native no hazard: 4/6 safe, 0/6 violations, 2/6 unfinished, 6/6 engaged.
- Untrained no hazard: 5/6 safe, 0/6 violations, 1/6 unfinished, 6/6 engaged.
- Native hazards combined: 3/18 safe, 0/18 violations, 15/18 unfinished, 18/18 engaged.
- Untrained hazards combined: 1/18 safe, 0/18 violations, 17/18 unfinished, 18/18 engaged.

Across the six episodes per hazard family, native scored island 2, bar 0 and gate 1;
the adapter scored island 1, bar 0 and gate 0. The adapter gained one no-hazard
completion and lost two native hazard completions, with no compensating hazard
wins. It therefore failed the prospective training-readiness rule. The small panel
does not establish statistical inferiority or rule out a better evolved controller.

Cube's six native no-hazard controls had 0/6 safe completions, 0/6 violations,
6/6 unfinished and 6/6 engaged. Success remained the original object-position
criterion within 4 cm of its recorded goal. It was never relabelled as reaching.
Its adapter and hazard comparisons remained closed; TwoRoom was not substituted.
The reviewed images showed no visible corruption. This does not diagnose why the
original Cube task failed.

All 54 scored episodes and four engineering probes completed without a technical
rollout failure. Native/zero actions, states, dense traces, frames and model work
matched exactly in each environment. All saved policy frames passed shape checks,
with no roughness flags; the start/goal and control galleries were visually reviewed
before opening PushT's adapter stage. Cube retained one renderer and captured fresh
frames at replans and the final action, explicitly marking held frames between
replans. Software tests verified that cadence against policy buffering.

## Verification, costs and evidence

Twenty-six rented-GPU software checks passed without skips, and Ruff passed. An
independent saved-evidence audit checked frozen source and asset hashes, source-pair
identities, all episode starts, native goal outcomes, dense geometric safety,
executed action bounds, residual replay, mixture allocation and gate decisions.
The first audit incorrectly assumed absolute-pixel PushT command bounds. Inspection
of the pinned native source confirmed relative commands in [-1,1], with PD targets
at current pusher position plus 100 times the command. Only the auditor changed;
its original error, source versions and diagnostic are retained. No episode was
rerun. Cube raw proposals and clipped executed commands are saved separately.

Total charged execution was 6,518 environment actions, including 24 engineering
actions, 64 explicit resets including six preparation resets, and 79,516 native
physics integrations including reset work. Frozen-world-model predictor work was
14,395,721 steps. The audit retains encoder calls, candidate rollouts, learned safety
work and elapsed time separately. Cube's DenseCube and native-mj_step counters
measure overlapping work and must not be added together. Equal search caps did not
produce equal actual work because episodes ended at different times.

The [results receipt](../../progress/receipts/20261031-lewm-screen-results.json),
[local verification](../../progress/receipts/20261031-lewm-screen-local-verification.json)
and [rental closeout](../../progress/receipts/20261031-lewm-screen-pod-closeout.json)
index the final evidence. Full local artifacts are under
`outputs/lewm-screen-20261031/`: the sealed release, frozen models, raw traces,
audits, six Cube-control replays and 24 paired PushT replays. All 685 archive members
were independently hash-verified locally, all 30 videos
decoded, and the outcome galleries and representative video frames were inspected.
The pod was deleted with provider get/list absence confirmed, both guards retired,
and a conservative $0.2364 rental estimate under the fresh $1 cap. This is an
estimate, not an invoice. No other rental was modified. Nothing from this
completed protocol authorizes another rental or further training. Zero observed
violations in this small, correlated panel is not a zero-risk or general safety claim.
