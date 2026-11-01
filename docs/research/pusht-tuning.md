# PushT planner tuning with learned safety

Completed 29 October 2026 on the replacement RunPod RTX A4500. The small search
worked mechanically but did not establish a completion improvement on reserved cases.

The current preset and selected setting each completed 2/3 development cases safely.
On their two completed cases, execution fell from 50 actions in total to 31. The
selection score, which charges every unfinished episode 50 actions, fell from 100
to 81. The selected setting was `random-02`: horizon 1, proposal spread 0.5,
retention 0, temporal correlation 0.5 and elite fraction 0.1.

On the reserved comparison, both settings had **0/3 safe completions, 3/3 unfinished
and 1/3 margin-breach episodes**. Neither arm contacted a hazard. The island case
started at 5.5898 px full-pusher clearance after native reset settling, below the
required 6 px. Both controllers refused to execute any action. The layout rule
checked the supplied input position rather than the slightly shifted observed
reset position. This is a panel-admission defect, not a violation caused by a
controller action. All frozen cases remain in the denominator; none was replaced.
The bar and gate cases stayed clear and moved the object, but missed the original
joint task goal within 50 actions. The development efficiency gain did not transfer
in this small comparison.

The frozen LeWM supplies task cost. The frozen learned motion model receives native
pusher position and velocity plus candidate actions, predicts its dense path, and
applies generic geometry to the supplied hazards and arena. Only the pusher is
protected; the T-block remains part of the task goal. Runtime guards forbid analytic
motion prediction and simulator previews during planning.

The [frozen design](../../configs/pusht_tuning_20261029/design-remote-v1.json) compares
one existing preset with six reproducibly sampled settings. It varies horizon,
proposal spread, temporal correlation, retention and elite fraction, keeping the
6 px margin fixed. Each replan receives exactly 11,250 CEM predictor steps. Chosen-plan
forecasts and actual work are counted separately. Three development and three reserved
cases use distinct source episodes, an island, bar and gate per split, and one fixed
seed per case shared across configurations. Selection uses development only: zero
margin breaches or engineering errors, most safe completions, then lowest total
completion-action score with 50 charged for each unfinished episode. The choice was
frozen before reserved evaluation.

The 27 remote trials used 890 actions. Including six preparation resets and the one
native test reset, there were 34 resets and 8,968 physics steps. Planning used
2,059,140 predictor steps, 1,173,930 candidate rollouts, 366 encoder frames and
1,173,935 learned safety checks. There were no reference-predictor calls. Development
and reserved evaluation took 137.84 and 47.39 seconds respectively. Across the mixed
configuration trials, 25/27 engaged and 24/27 moved the object at least 5 px. The
28 relevant tests passed on the pod. Setup time is separate: the initial installation
on its network volume was slow; moving the same pinned environment to the pod's SSD
resolved it.

Earlier local work is preserved separately and fully charged. The first 21 trials
used stale target renders and are invalid engineering evidence. A corrected local
pass completed 27 trials before the remote-only instruction was verified. Its outcomes
did not select or change the remote search. The complete remote panel was rerun and
selected using remote development only. Across all three passes and native tests,
the ledger contains 75 episode trials, 2,724 actions, 96 resets, 27,432 physics steps
and 6,278,722 predictor steps. Summed measured episode time across all passes was
914.59 seconds, including 823.22 seconds of planning; these sums exclude preparation
and model loading. No performance results are pooled across hosts.

The [final receipt](../../progress/receipts/20261029-pusht-planner-tuning.json) records
hashes, budgets, model identities, the admission defect and all costs. Raw traces,
freezes, source snapshots, audit, trial CSV and all development/evaluation comparison
videos are retained under `outputs/pusht-tuning-20261029-remote-v1`. The
[rental receipt](../../progress/receipts/20261029-runpod-pusht-tuning.json) records
$0.25/hour, a five-hour deadline at 03:27 SAST on 30 October, and independent local
and pod-side termination guards within the existing $5 session cap.

This is one small search and one seed per case. Sources are historical development
inputs, so reserved does not mean globally unseen. No evolutionary, optimizer
superiority or cross-environment claim follows. Before further experiments, the
panel needs admission checks against the actual reset state; task-progress failures
on the clear bar and gate cases also need diagnosis. No additional experiments were
started after these outcomes.

## Fresh author-preset comparison, 29 October

A separately authorized comparison corrected the question being tested: the prior
"current preset" was already tuned. The new arm keeps the authors' actual shared
CEM configuration at upstream revision `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`:
300 proposals, 30 iterations, 30 elites, spread 1, five action blocks planned and
five executed between replans, with five primitive actions per block. Both arms
use the same frozen LeWM and a newly trained shared 25-action safety head. The
margin remains 6 px and only the pusher is protected.

Twelve cases come from distinct public PushT demonstration episodes, with six
for development and six reserved. Each split has an island, bar and offset gate
at two strengths, paired with no-added-hazard controls. Admission uses the actual
settled native start and requires at least 10 px full-pusher endpoint clearance.
The source is the official Diffusion Policy pose archive; overlap with LeWM
training is possible, so these are not globally unseen tasks.

The development-only search selected `random-02`: horizon 2, replan every 2 blocks,
elite fraction 0.2, retention 0.5, spread 1 and no temporal correlation. On the six
reserved hazard cases, authors completed 2/6 safely and selected completed 3/6;
4/6 and 3/6 remained unfinished respectively. Both completed all 6/6 no-hazard
controls. Neither arm had an observed violation or contact. Every case engaged.
This one-case pilot difference passes the predeclared conditional expansion gate;
it does not establish statistical superiority or an evolutionary advantage.

Both arms had at most 50 executed actions and 90,000 CEM search predictor steps
per episode. Forecast calls and encoders are additional and separately recorded.
There were 120 distinct comparison trials, 3,577 executed actions, and 6,831,966
predictor steps. Including shared training, preparation and native tests, the
ledger contains 21,177 actions, 839 resets and 213,448 physics steps. The 31 relevant
remote tests passed after a preserved integer-coordinate fixture repair. Model
fitting took 62.53 seconds; development and reserved evaluation took 250.09 and
83.66 seconds. Setup, preparation, loading and media are separate.

The frozen [design](../../configs/pusht_preset_20261029/design.json), raw audit,
source snapshots, all traces and twelve paired videos are retained under
`outputs/pusht-preset-20261029-v1`. The safety head is under
`outputs/pusht-preset-motion-h5-20261029`. A verified 17.7 MB release has been
assembled locally, then superseded by the combined four-environment release below.
The older approved private release remains unchanged.

On 30 October, Sachin approved a 150-action limit for upcoming environment
comparisons, with completion recorded at 50, 100 and 150. That amendment precedes
all new comparison outcomes. It does not change this completed 50-action PushT
pilot. A run unfinished at 50 is budget-censored, not proof that safe completion
is impossible. The separate all-environment freeze is
`configs/lewm_portable_20261030/design.json`.

## Fixed-setting action-budget sensitivity, 30 October

After Sachin raised the 50-action concern and approved 150 actions, a separately
frozen diagnostic reran the same two settings on the same reserved cases, with
both hazard and no-hazard conditions. The first 50 actions and states matched the
original traces. Safe hazard completion stayed at authors 2/6 and selected 3/6 at
50, 100 and 150 actions; all seven previously unfinished hazard cases reached 150
without completing. No-hazard completion stayed 6/6 for both arms, with no observed
violations. Extra actions alone did not resolve these particular planner failures.
This is a posthoc budget sensitivity on reused cases, not independent confirmation.
The new namespace is `outputs/pusht-budget150-20261030-v1`; the original result is
unchanged. The 24 new trials used 1,385 actions, 24 resets, 13,898 physics steps and
2,574,286 predictor steps, all additional to the first comparison.

A separate saved-control rescoring found every reserved no-hazard pusher trajectory
would contact its paired virtual hazard. The three tuned safe completions therefore
exhibit deliberate safety adaptation under that paired diagnostic. This uses existing
traces only and changes no performance outcomes or case selection.

## Learned-safety extension, completed navigation components

The new shared interface predicts protected-body XY motion from current native
proprioception and candidate actions, then applies the same circle, capsule and box
geometry. LeWM remains frozen and supplies task cost. Each environment receives its
own supervised transition model; this is a shared architecture and interface, not
transfer of one set of safety weights. Training, calibration and testing are counted
separately from parameter search. The 150-action comparison keeps all six reserved
cases and their no-added-hazard controls, and reports 50/100/150 checkpoints.

TwoRoom completed all 120 frozen trials. Authors' settings completed 0/6 reserved
hazard cases safely, with 6/6 unfinished; the selected setting completed 3/6 safely,
with 3/6 unfinished. Neither arm had an observed violation or contact. Both engaged
in all 6/6 cases. The paired controls improved from authors 1/6 to selected 6/6.
The selected hazard counts at 50/100/150 were 2/6, 2/6 and 3/6. Saved-control rescoring
supports deliberate adaptation in two of the three completions; the third control
already cleared its paired hazard. Cases are closely related within-room poses,
not a test of crossing the physical doorway or independent task replications.

Reacher also completed all 120 trials. Both settings scored 0/6 safe reserved hazard
completions and 6/6 unfinished, without an observed violation or contact. Both moved
in all 6/6 cases. Controls scored authors 5/6 and selected 4/6 at 150 actions. At
50/100/150, authors' controls scored 3/6, 4/6 and 5/6, while selected stayed 4/6.
The development gain therefore did not transfer to these reserved cases. Reacher's
small joint-pose panel establishes no reliable completion improvement.

Saved-trace diagnostics further qualify this result. On every selected Reacher
hazard case, the native maximum joint error never fell below its starting value,
despite using all 150 actions. Movement therefore did not indicate task progress.
The strong gate's 0.144 m opening leaves 0.094 m after body radius and requested
margin, but minus 0.01259 m after both sides' fixed prediction-error allowance.
Its direct opening is closed under the calibrated tube. Routes around the ends
were not ruled out, and this does not explain all six failures. Endpoint admission
checked clearance, not whole-task connectivity or 150-action feasibility. TwoRoom's
strong gate remains open under the same calculation, with only 3.189 px to spare.
These are posthoc diagnoses from unchanged traces, not reasons to replace cases or
revise the completed comparison.

Both independently selected `random-05`: two blocks planned, one executed before
replanning, proposal spread 2, elite fraction 0.2, retention 0.25 and correlation 0.
This uses 150 proposals per iteration, with the same nominal 270,000 search-predictor
step ceiling per 150-action episode as authors' 300-proposal, five-block setting.
Actual work differs with early completion or refusal; development tuning adds cost
and is not claimed to match the untuned baseline's total acquisition cost.

Both safety models covered 12/12 independent test groups under their fixed calibrated
allowances: 12.4054 px for TwoRoom and 0.0532963 m for Reacher. This empirical check
does not guarantee safety under adaptive CEM search. The independent evidence audit
reproduced model/input/source identities, development-only selection, native task
success, every retained dense monitor sample, all denominators and search-work caps.
Reported planner-time counters omit the final exception-raising planning call;
full episode and stage wall times include it and should be used for elapsed totals.

The [navigation receipt](../../progress/receipts/20261030-lewm-navigation-results.json)
links the raw reports, audits, calibrated models and all paired replays. Cube's
separate completed comparison is below. Its task uses the original cube-position
success rule, not the earlier gripper-reaching subtask; those historical results
are not directly comparable.

## Completed Cube comparison, 30 October

The valid Cube run completed 120 trials with zero runtime errors and passed the
independent source, command, native goal, dense monitor and budget audit. Both
authors' settings and the selected setting completed 0/6 reserved hazard cases,
with 6/6 unfinished, no observed violations or contacts, and engagement in 6/6.
Controls scored authors 3/6 and selected 1/6. Their 50/100/150 counts were 2/6,
3/6, 3/6 and 1/6, 1/6, 1/6 respectively. The development selection did not transfer
and reduced control completion. This is a negative result for the current pipeline.

Selection chose the manually supplied `working-anchor`, not a random candidate:
three planned blocks, one executed, 100 proposals, 30 iterations, spread 1,
elite fraction 0.1, retention 0.5 and correlation 0.5. It had 0/6 development hazard
completions and 1/6 controls, compared with authors 0/6 in both conditions.

The initial Cube motion release failed its 12-group test. A separately frozen repair
used fresh calibration and test groups with unchanged learned weights. Its fixed
allowance was 0.0608504 m and covered 46/48 independent test groups. That allowance
also closes the strong gate's direct opening by 0.007701 m; routes around the ends
were not ruled out. Endpoint clearance does not certify task feasibility.

Engineering passes exposed an inconsistent goal marker and unbounded native commands.
They were stopped before reserved evaluation and retained as excluded work. The
valid `cube-v4` pass uses consistent goal images and clips applied commands exactly
as the safety forecaster does. Latent task-cost predictions retain the same upstream
unbounded candidate imaginations used elsewhere. Renderer-stride probes failed their
declared equivalence tolerance, so the valid run renders every action.

The valid comparison alone used 15,668 actions, 120 resets, 391,700 executed physics
steps, 6,000 reset physics steps and 28,419,572 predictor steps. Initial training,
fresh calibration and all failed or interrupted engineering work are additional.

Visual QA adds a material qualification: the selected reserved island-mild hazard
episode contains noisy non-scene raw images from frames 67 through 150, including
scheduled planning inputs 70 through 145. The cause is unconfirmed. A posthoc
roughness-to-contrast screen flagged this one episode among all 120 trials and
15,800 saved frames; that screen is not a validated corruption classifier. The
raw frame and gallery display the same pattern. This is not confined to terminal
media, so Cube's performance cannot be interpreted as a clean comparison without
this observation-quality qualification. No images, actions, cases or scores were
replaced. The source, native-physics and outcome audit remains valid for its stated
scope, which did not certify image quality.

The [Cube receipt](../../progress/receipts/20261030-lewm-cube-results.json) and
`outputs/lewm-preset-expansion-20261030/cube-v4` retain all outcomes and paired media.
The combined release and pod-deletion verification are separate preservation steps.

## Preservation and rental closeout

The complete [results receipt](../../progress/receipts/20261030-lewm-results.json)
records all four comparisons. The local release is
`outputs/releases/lewm-preset-20261030-v1`, with 79 files totalling 1,057,026,889
bytes, 48 paired replays, final safety heads and three exact public world-model
objects. The unchanged PushT world model remains pinned in the earlier private
release. All 4,638 evidence-archive members were read back and hash-verified locally;
all exposed files and report summaries also passed verification. A configured
credential-value scan found no matches. Supplemental Cube visual QA is included
without altering the original archive or observations.

The combined ledger records 118,744 explicit environment actions and 2,226 native
reset control calls, totalling 120,970 recorded calls. Including the unjournaled
first startup and interrupted-call allowances gives an upper bound of 129,612.
Recorded physics integrations total 1,441,159, with an upper bound of 1,657,209;
25,846 discrete navigation steps are separate. These are bounds, not invented exact
counts for lost calls. Prior tuning pilots remain separately recorded.

RunPod `r8w4b6lwkbjyrz` was deleted after local verification. Provider get returned
not found and the pod list confirmed absence. The independent local guard then
exited. The [closeout receipt](../../progress/receipts/20261030-lewm-pod-closeout.json)
records a conservative combined rental bound of $3.31 against the $5 cap, including
the earlier rental allowance. This is an elapsed-time estimate, not an invoice.
Sachin approved the exact combined 79-file private payload after pod deletion. It
is preserved in `Sachioster/SafetyDial-PushT` at
`campaigns/lewm-preset-20261030-v1`, pinned to revision
`b374ee4c602c538e992ba0e6072e58e18abcb168`. All 79 files were downloaded from that
revision and matched local byte counts and SHA256 hashes. The earlier pinned PushT
world-model object was also downloaded and verified. The
[preservation receipt](../../progress/receipts/20261030-lewm-private-preservation.json)
records the exact private payload, and the
[release](https://huggingface.co/datasets/Sachioster/SafetyDial-PushT/tree/b374ee4c602c538e992ba0e6072e58e18abcb168/campaigns/lewm-preset-20261030-v1)
contains the results and all paired replays. No rental remains.

## Exact hand-set PushT reproduction, 30 October

Sachin then requested a reproduction of the earlier successful PushT demonstrations
with the current stack, before more tuning, chose PushT first and authorized a new
pod. The fresh protocol is `configs/pusht_repro_20261030/design-v2.json`. It retains
the exact known scene, input pair, seed, three hazards, 6 px full-pusher margin and
original 50-action limit. The hand-set planner uses horizon 3, one executed block of
five actions, 250 candidates, 15 CEM iterations, exploration setting 1, elite fraction
0.1, reuse 0.5 and temporal correlation 0.5. There is no parameter search or training.

Three arms ran the three cases. Historical CPU and current CPU each completed 3/3
safely in 26, 43 and 25 actions for island, bar and gate. Both arms' full saved state
trajectories were bitwise equal to the earlier recorded traces. Thus the current
25-action learned-motion head and current code retain the demonstrated behaviour
on CPU under the original manual recipe. Current GPU completed island and gate in
25 actions each; bar stayed safe but unfinished at 50, giving 2/3 safe completions.
All arms had zero observed violations or contacts and engagement in 3/3.

CPU and GPU can generate different CEM proposal streams under the same numerical
seed. The study did not include historical GPU, so the GPU result cannot distinguish
device/sampling effects from interactions with the changed stack. These three layouts
share one known scene and seed; this is a reproduction, not a general success-rate
estimate. No tuning, extra seeds or outcome-based case replacements followed.

The initial historical launch failed during import because its source bundle omitted
`lewm16_protocol.py`. That failure preceded environment creation and model loading,
cost zero interactions and was retained. A documented dependency repair restored the
helper with unchanged planner settings and cases before the nine valid trials.
The trials used 288 actions, 9 resets, 2,880 executed and 18 reset physics steps,
and 675,180 predictor steps. The passive dense monitor and native joint task goal
were independently rescored, and all source, input and planning-budget checks passed.

The [reproduction receipt](../../progress/receipts/20261030-pusht-hand-preset-reproduction.json)
records every outcome. Local artifacts are under
`outputs/pusht-manual-repro-20261030-v1/remote`, with three paired replay videos and
`results/media/comparison.png`. All 174 archive members and matching raw files were
hash-verified locally before pod deletion. The new rental cost bound is $0.13;
the conservative combined bound is now $3.44, still below $5. Provider get and list
confirmed deletion of pod `8i3ts21n4ut0st`. No new checkpoint was trained or uploaded.


## Matched GPU stack control, 30 October

A fresh frozen six-episode control ran the historical and current stack/head bundles
on one RTX 2000 Ada pod, both using GPU inference. It retained the same scene,
seed, island/bar/gate layouts, 6 px full-pusher margin, exact hand-set planning
recipe and 50-action reproduction limit. This was not a parameter search.
The [protocol](../../configs/pusht_gpu_control_20261030/design.json) was frozen before
any episode. Both arms completed 2/3 safely, in 25, 50 and 25 actions. Bar was safe
but unfinished at 50. Neither arm had an observed violation or contact; both
engaged in 3/3 cases, with no controller failures or runtime errors.

All paired actions, full states and dense physics traces were bitwise equal.
Consequently, the predeclared conditional score diagnostic did not trigger.
There is no observed regression from the current bundle on these three known
cases. Together with the earlier same-host CPU/GPU comparison, this supports an
inference-device/sampling-path explanation for the CPU/GPU difference. The separate
effects of random proposal streams and floating-point task/safety scoring remain
unisolated. This is one scene and seed, not evidence of general equivalence or
an estimate over independent tasks. Comparisons across rental hosts stay descriptive.

The six trials used 200 actions, 6 resets, 2,000 executed and 12 reset physics
steps, and 450,120 predictor steps. All experimental computation and replay
rendering ran on the pod. The two orchestration failures preceded simulation and
are retained in the launch receipt, each with zero interactions. Native task
outcomes, dense safety monitors, source/input identities and planning budgets
passed the independent audit. Three paired replays and the six-panel gallery are
under `outputs/pusht-gpu-control-20261030-v1/remote/results/media`.

The [result receipt](../../progress/receipts/20261030-pusht-gpu-control-results.json)
links the full evidence. All 144 archive members and their raw copies passed
local byte/hash verification; the gallery was visually inspected. The exact
executed source is retained, including the driver before its later formatting-only
repository edit. No training, additional seeds, parameter changes or private
upload followed. Pod `146xqwaev9vclh` was deleted after verification, and provider
get and list confirmed absence. Its conservative rental estimate was $0.061;
the combined estimate is $3.50 under the unchanged $5 cap, not an invoice.


## Implemented SafeCEM refinements, 30 October

The three items from the SafeCEM todo are implemented. The planner fits to up to
25 feasible elites whenever any exist, otherwise to the least-violating candidates.
A single elite retains the previous sampling spread. It scores and checks the
final mean, then compares its task cost with the best checked candidate rather
than accepting any safe mean. Distribution updates blend 0.1 old and 0.9 fitted
mean and standard deviation, with a minimum normalized standard deviation of 0.05.
These smoothing and floor values are fixed engineering defaults, not tuned optima.
The execution recheck and refusal when no safe plan exists remain in place.
`cem_refinements=False` retains the previous update/choice path for future controls;
this experiment instead used an explicit frozen pre-change planner snapshot.
Source collectors now include the new helper.

Five focused GPU software tests passed, covering scarce and singleton safe elites,
zero feasible candidates, smoothing and the spread floor, checked final selection,
actual solver integration, refusal and model-query accounting. The first test
fixture incorrectly assigned a read-only solver property. It was repaired through
the public configure API before any episode; the original source and failed report
are retained. No simulator interaction or frozen-model query occurred in that failure.
The final [protocol](../../configs/pusht_improvements_20261030/design-v3.json)
was frozen after the software gate and before all six episodes.

The short control retained the same current frozen LeWM and safety head, scene,
seed, three layouts, full-pusher 6 px margin, 250 candidates, 15 iterations and
three-block horizon. Both arms used GPU inference on the same rented host. Both
achieved 2/3 safe completions at 50, 100 and 150 actions, zero observed violations
or contacts, one unfinished episode and engagement in 3/3. In island/bar/gate order,
baseline used 25/150/25 actions; the changed planner used 26/150/40. Bar remained
unfinished in both. Thus this control establishes implementation, not a performance
gain. It has one known scene and seed, without per-change ablations or generalization
claims. No parameter search or outcome-based replacements followed.

Every one of the 660 changed-planner search iterations had at least 25 feasible
candidates. Consequently, the scarce-feasible-elite branch had no opportunity to
help on the trajectories tested, although its edge cases passed software tests.
The best checked candidate was selected in 43/44 replans. Smoothing and final choice
were bundled, so their individual effects on the slower completions are unisolated.
The baseline's saved first-50-action state prefixes exactly matched the prior GPU
records. Nominal search work remained 11,250 predictor steps per replan. Comparing
the mean and best candidate can add three predictor steps; actual query counters
charge this overhead. The six episodes used 416 actions, six resets, 4,160 executed
and 12 reset physics steps, and 945,381 predictor steps.

The [result receipt](../../progress/receipts/20261030-pusht-improvements-results.json)
links all evidence. Three paired replays and the gallery are under
`outputs/pusht-improvements-20261030-v1/remote/results/media`. All 160 archive members
and raw copies passed byte/hash verification. A media-only import-path failure was
fixed in a new renderer version, retaining the original failure; no episode or
model inference was repeated. The completed todo PDF and builder were removed from
the working documents after their original bytes were verified in the evidence
archive. No new checkpoint was trained or private upload performed.

Pod `y4oq5yi1k239im` was deleted after verification, with provider get and list
confirming absence. The local shutdown guard did not survive the interruption and
missed its intended 11:41 SAST cutoff. The pod remained running until closeout at
15:49 SAST. The [closeout receipt](../../progress/receipts/20261030-pusht-improvements-pod-closeout.json)
includes all idle time: this rental's conservative estimate is $1.26 and the combined
estimate is $4.76 under the unchanged $5 cap. These are elapsed-time estimates, not
an invoice. No other rental was modified.
