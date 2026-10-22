# Phase 3 recovery: repaired mechanisms and completed comparisons

**19 October 2026. Prospective implementation plan, with no new run started.** Sachin
requests a credible contribution, concrete repairs to Parts 1 to 4 and an autonomous
run that works through failures efficiently. This plan governs new recovery work.
Completed results, frozen protocols, original source snapshots and their stop decisions
remain evidence of the original experiments. Recovery uses new files and run identities.

The operational change is to stop treating a failed candidate, exception or scientific
premise as an automatic instruction to close the whole instance. Diagnose the failure,
use the bounded repair queue, then take the next specified route. Correctness checks,
independent evaluation and the allocation deadline remain binding.

## 1. Contribution and a useful finish line

The first contribution candidate is an empirical method for **intervention-aware
execution adaptation around a frozen VLA**: preserve the direction of joint commands,
refresh policy feedback when execution actually departs from the proposal, and evaluate
search finalists sufficiently to reduce selection on lucky rollouts. Test whether the
mechanism helps, then whether evolution adds value over random search and ordinary
learning with the same controller and information. A new optimizer name is insufficient.

The model-assisted and counterexample-acquisition ideas remain additional contribution
candidates. Their distinct questions are whether calibrated prediction improves the
allocation of physical evolutionary evaluations, and whether diverse error acquisition
improves warning prediction per collected interaction. They do not inherit a positive
result from the execution comparison.

Current evidence gives concrete repair targets:

- Part 1's full-strength controller achieved 4/8 safe completions with 2/8 violations,
  against identity's 5/8 and 0/8. Grasp coverage, cap geometry and selection noise need
  separate tests. See the [failure investigation](docs/research/reports/ManiGuard%20Part1%20failure%20investigation.md).
- Part 2's adaptive trigger never requested an early query. An unchanged rerun would
  again fail to test the proposed timing mechanism. See the [report](docs/research/reports/ManiGuard%20Part2%20correction%20triggered%20feedback.md).
- Part 3's readout already failed on actual future observations, before dynamics
  prediction. Its correction bins did not support a distinct trust rule. See the
  [report](docs/research/reports/ManiGuard%20Part3%20model%20assisted%20evolution.md).
- Part 4 met its calibration alarm target but alarmed on 91.75% of negative development
  windows; development had one event. See the [report](docs/research/reports/ManiGuard%20Part4%20counterexample%20premise.md).

A completed, audited physical comparison with independent evaluation is the first
finish line. A forecast diagnostic, working launcher or passing test suite alone does
not achieve it. A positive scientific result requires measured improvement with its
uncertainty and appropriate controls. Preserve a null or adverse comparison if that is
what happens; do not continue sampling until a desired sign or significance appears.

## 2. Parts 1 and 2: repair the execution interface

### Joint-command geometry and grasp conditioning

Begin with a uniform arm cap, smoothing one and an eight-action maximum query interval.
This removes dependence on the known grasp-heuristic gap while testing cap geometry.
Keep the original heuristic as a labelled reference. Do not simply widen its finger
band until a favourable physical result appears. A replacement detector would need
training-only coverage and false-positive audits plus its own physical comparison.

For proposed arm target p, measured joints q and cap c, test:

```text
e = p - q
scale = min(1, c / max(abs(e)))       # scale = 1 when e is zero
adapted_arm = q + scale * e
```

Scaling the whole vector preserves its joint-space direction before downstream action
clipping and bounds every coordinate's target error. Preserve the gripper proposal.
The disabled branch returns the original proposal exactly. Check dtype, finite values,
joint bounds and final environment inputs. Downstream clipping can still change the
direction, so log that effect. This does not certify Cartesian direction, object tilt,
physical velocity or collision avoidance.

Compare this with componentwise clipping at the same cap. The starting finite cap
domain is disabled, .12, .08 and .04 radians. Do not add smaller caps automatically.

### A trigger that has an opportunity to act

Preserve Part 2's causal correction definition, numerical tolerance and reset semantics.
Inspect accumulated correction at eligible query ages four through seven, rather than
using the maximum at age eight when a query is already compulsory. On designated
development traces, evaluate at most three thresholds derived from the .50/.75/.90
quantiles of positive eligible corrections. Freeze the actual numbers and derivation.
Threshold zero remains a fixed-frequency reference, not an adaptive mechanism.

Replay must show substantial corrections and eligible early queries, distinguishable
from floating-point noise. Record activation by episode and task. If no positive eligible
corrections exist, remove this gene for that representation and continue with the cap
comparison; no positive threshold can solve an absent opportunity. A trigger with no
activation must not be presented as a tested adaptive method.

Retain minimum intervals four and six, maximum eight and smoothing one. Do not search
longer stale chunks in this repair. The initial controller has three categorical
coordinates: uniform cap, correction threshold and minimum query interval. Serialize
disabled values and deduplicate equivalent effective controllers in analysis.

### A bounded physical screen

Use two preselected development tasks and four interleaved conditions per task:
identity, componentwise uniform cap at interval eight, direction-preserving uniform
cap at interval eight, and the latter with an active correction trigger. This is
**eight full episodes**, with fixed cases and seeds before outcomes. It is a mechanism
screen, not proof of benefit or a powered interaction test. Choose the common cap and
threshold using permitted trace evidence before this screen.

These observations may inform the new development recipe and fixed reference; they
cannot become final evaluation. Search need not wait for a hand-written controller to
win this small screen. Require correct, observable behaviour rather than an early EA
win. All optimizers subsequently receive the same frozen representation.

Compare active timing with fixed-four/fixed-eight references on development evidence.
If making a timing-efficiency claim, also freeze a control whose four/eight intervals
are randomly scheduled using a development-derived query allowance, independently of
task outcomes. Report actual queries, not assumed equality. A new physical timing
ablation requires declared slots; it is not free work added to the eight-episode screen.

## 3. Part 3: repair readouts before dynamics

Use cached features and exact observations first. Keep the original model and failed
audit unchanged. All old calibration/audit tasks are now exposed development evidence.

1. **Information/readout check.** Compare at most four declared recipes: visual-only
   PCA ridge, visual-only pooled-feature ridge, separately regularized visual/robot
   blocks, and a direct short-horizon outcome predictor. Keep preprocessing task-grouped
   and fit-only. A direct outcome predictor is a different model, not a repaired latent
   world model. Record which claim it can support.
2. **Prediction check.** If actual-state readouts are adequate, compare persistence,
   regularized action-conditioned residual ridge and one small residual MLP. Predict
   changes over short known action prefixes. Audit changing windows and action shuffling.
   Retain executed-action units and distinguish actual from predicted histories.
3. **Calibration check.** Replace unsupported fixed correction bins with a small
   regularized model of factual error using correction and disagreement, or omit the
   correction-dependent arm. Require support across episodes/tasks and a measurable
   difference from disagreement-only selection on development data. Identical masks or
   an affine rescaling do not establish a distinct method.
4. **Fresh check.** Freeze the recipe, readouts, thresholds and context bank before
   opening new whole-task audit records. Preserve actual-state readout error separately
   from dynamics error and report per-task ranking/acceptance as well as aggregate loss.

The initial offline development allowance is **at most four recipes and 24 trained
fits**, including ensemble members, with a **30-minute wall-time cap**. Fix a small
regularization grid and count each fit; no architecture search on the fresh audit.
Use early diagnostics on saved inputs to avoid spending simulation on a representation
that cannot read out the target. A failed recipe advances the finite queue.

Record new execution experiments passively so they can supply future training data.
Declare whole-task fitting/calibration/audit roles before using their labels. No replay
of alternate commands on recorded trajectories is physical evidence. A repaired model
can be frozen and evaluated passively during an already scheduled final comparison;
it must not alter that comparison's controllers or use those final labels for tuning.

Model assistance needs a separate new protocol matching E/M/C physical evaluations and
including unassisted evolution. Preserve the original numerical accuracy criteria when
testing the same forecast claim; any changed target/criterion needs justification and
a new study identity before fresh scoring. If assistance remains unsupported, continue
the direct physical search in section 5. Do not fill evolutionary fitness with imagined
outcomes or repeat the old gated 64-episode launcher.

## 4. Part 4: repair task support and warning calibration

### More informative data and a finite detector comparison

Use the same first native uprightness event, causal alignment, eight-action warning
horizon and four-action deadline initially. Keep native spill/drop outcomes separately.
Retain safe trajectories and censored windows; no resampling of episodes until enough
violations appear and no trajectory-failure labels substituted for event labels.

Compare at most four declared detector recipes: visual-only PCA logistic, regularized
pooled-feature logistic, visual-only small ensemble, and visual/robot fusion with separate
regularization. Reuse cached encodings and the 24-fit/30-minute offline allowance jointly
with Part 3 during an execution-focused session. Prioritize the cheapest diagnostic that
can resolve the next decision. A separate model-focused session can freeze its own cap.

Use whole-task cross-validation for development. Fit scalers/PCA within each fitting
fold, never on its evaluation task. Train with episode-balanced contributions and a
declared class-weight rule. Calibrate using natural event prevalence with episode
weights, separately from any fitting class balance. Check that implementation rather
than assuming the original loss weighting caused the failure.

Use multiple calibration tasks where available and report false alarms on each held-out
task, event recall, lead time, average precision and episode-level alarm burden. A 5%
pooled target on one task is not evidence of cross-task alarm control. Unsupported
single-class calibration is inconclusive; do not produce a purported failure probability.

If task/event support remains the bottleneck, a **separately scheduled common recording
allowance of at most twelve episodes** can comprise two batches of three new development
tasks with two repetitions each. Freeze all tasks, policies and seeds first. Allocate
their fitting/calibration/development roles in advance. All episodes remain in the
ledger, including safe outcomes. Two batches are a maximum, not a guarantee of support.
Prefer records already planned for execution experiments; any additional batch must
replace other planned work within the session, with final evaluation protected.
The twelve-episode option is not included in the default execution ceilings below.
Any combined variant needs its own budget before search; do not append data collection
or take slots from one optimizer after inspecting comparative results.

Require at least two event-bearing fitting tasks and two development event episodes
before interpreting confident-miss acquisition, and disclose the remaining small task
denominator. A fresh audit uses different, still-unexposed tasks. Never relabel the old
Part 3 audit or Part 4 development task as fresh evidence after model selection.

### Keep acquisition possible without silently changing its question

The original confident-miss study remains gated. A new acquisition protocol can choose
one of two objectives using development evidence **before** method-specific collection:

- **Confident-miss acquisition:** retain the original factual missed-warning objective
  if the new predictor has useful warning behaviour and supported low-risk misses.
- **General prediction-error acquisition:** if misses are too sparse, use a separately
  named study of diverse detector errors. Freeze a bounded, episode-normalized prediction
  loss over eligible windows, with separately declared positive/negative contributions.
  This includes false alarms and unconfident errors. Report confident hazards separately;
  improvement here does not establish the original confident-miss hypothesis.

For the latter, first specify an exact clipped loss, weighting, single-class-trajectory
rule and task aggregation; test them on synthetic and existing records. E and D share
the same score and mutation. R remains space-filling/random acquisition, U uses only
disagreement for its acquisition score, and D differs from E in archive retention.
Freeze one acquisition predictor throughout all arms, then apply matched supervised
repair updates from identical starting weights. Keep a common-data-only repair control.

Neither objective permits bypassing scene/reset validity. Implement bounded planar
scene genes, consistent object/goal/monitor references, stable initialization and
reproducible serialization first. Charge rejected proposals, settling and confirmations.
Do not use a random policy seed as a spatial genotype or treat edited scenes as native
benchmark tasks. If scene engineering fails, repair it within its allowance and continue
the execution route; do not claim a physical acquisition result from bookkeeping tests.

The original **80-episode** acquisition layout is a useful costed starting point:
8 shared initial + 48 acquisitions + 8 confirmations + 16 common fresh audit episodes.
It is an alternative primary session, not an addition to section 5. Add all prerequisite
recordings, engineering and retry allowances to its new prospective budget. Choose the
primary route before comparative outcomes; never run both and report only the winner.

## 5. First execution session: protect a complete comparison

The default next session repairs Parts 1/2 and runs **evolution, matched random search
and episodic ordinary learning**. Parts 3/4 receive bounded offline repairs and useful
new recordings. Their full physical comparisons require supported prerequisites and
their own allocation. Completing every proposed study in one eight-hour session is not
the plan.

Use a population/batch of four, two training cases per candidate and one optimizer seed
per method for this first comparison. Include the same shared initial four controllers,
with identity as a declared common anchor. Evolution re-evaluates its elite and mutates
the retained parents; random search draws from the same declared categorical domain;
ordinary learning updates an episodic categorical distribution over those same genes.
All receive measured `U = int(S_req) - 2 * int(V)` and the same permitted information.
This comparison makes no penalty-free claim.

For ordinary learning, freeze a small REINFORCE-style update, baseline, entropy term and
learning rate from CPU development. Log sampling probabilities. Forced identity anchors
and copied elites are not random policy samples for an uncorrected score-function
gradient. Train no VLA, encoder or privileged-state critic. If this baseline breaks,
repair it with the same care as the proposed method rather than silently dropping it.

Choose between these two sizes using timings before reading comparative search scores:

- **Full:** 8 shared initial episodes; 3 methods x 2 descendant populations x 4 candidates
  x 2 cases = 48; 3 methods x 2 shortlisted candidates x 2 confirmation cases = 12;
  5 frozen final conditions x 8 audit blocks = 40. **108 comparison episodes.**
- **Compact:** 8 shared initial; 24 descendant episodes for one population per method;
  the same 12 confirmation and 40 final episodes. **84 comparison episodes.**

Add the eight-episode development screen and reserve at most eight full infrastructure
replacement attempts: ceilings of **124 full attempts** or **100 full attempts**.
Their native-horizon control bounds are **136,400** and **110,000**, respectively,
plus at most six 32-action engineering attempts (**192 controls**) and measured
initialization/settling physics. All failed/partial work remains charged. Reserve slots
are not additional candidate exploration or an outcome-dependent sample extension.

The five final conditions are identity, a fixed development reference and one frozen
finalist from each optimizer. For each method, confirm its top two candidates on two
new common training cases, then select using their final-population and confirmation
observations with fixed ties. Freeze all three finalists before final evaluation.

Use four previously unexposed base tasks, two repetitions each, for the eight audit
blocks. Audit the exposure history and choose from the existing training partition,
reserving them as recovery evaluation before any new development. Keep their variants
together; preserve the main-study test partition. If four such tasks cannot be secured,
record a revised narrower design before collection rather than claiming fresh transfer.
Interleave the five conditions within blocks and balance order. Match cases and policy
seeds while retaining the known rendering/inference variability.

Freeze candidate panels, seeds, mutation/update rules, fixed-reference choice, failures
and ties in a machine-readable protocol before search. Report all methods' actual control,
query and elapsed costs as well as nominal episode allowances. Shared data are acquired
once and disclosed as a common information endowment. Equal episodes do not establish
equal realized interactions; compare cost curves at common observed cost ranges too.

One optimizer seed, even with two transitions, remains a pilot. A paper-level advantage
needs independent optimizer seeds and broader reserved evaluation, costed from the
observed effect and variability. Do not substitute many correlated frames for those
replicates or declare safety noninferiority from a nonsignificant violation difference.

### Turn a pilot lead into a contribution

Keep three claims separate: the repaired execution mechanism improves over identity
and the fixed rule; evolution improves over random search and ordinary learning; and
the effect persists on new tasks and optimizer seeds. If only the first holds, report
an execution-method result. If the fixed rule explains the gain, do not attribute it
to evolution. If only a predictor improves, report prediction rather than safe execution.

Use the pilot to freeze a practically relevant effect size, uncertainty analysis and
replication allocation before the next final evaluation. One concrete costing example
is three independent full-size searches: `3 * (8 + 48 + 12) = 204` training/confirmation
episodes. Six fresh tasks with two repetitions give twelve evaluation blocks. Evaluating
two common references and all nine frozen finalists costs `12 * (2 + 9) = 132` episodes,
for **336 full episodes**, excluding engineering, failures and reporting. At an illustrative
150 seconds each this is fourteen rollout-process hours. Three seeds are a planning
example, not an assertion of adequate statistical power or an approved allocation.

That replication study would require separately authorized sessions and new untouched
evaluation tasks. It cannot be squeezed into the current eight-hour limit or assembled
by quietly reusing exposed pilot evaluation as fresh data. Prefer one explained,
replicated mechanism over four incomplete contribution claims. Select a model-assisted
or acquisition replication instead only if its development evidence supports that
distinct claim, with the choice and complete costs recorded before final outcomes.

## 6. Allocation, repair loop and shutdown policy

### One clock and explicit ownership

At execution, discover the **currently authorized instance**, provider start, current
state, interpreters, installed simulation/policy assets and actual free disk/memory.
Do not assume the old RTX 3090 preparation base contains ManiGuard, or reuse a historical
4090 instance ID. Use only an already authorized allocation; no new rental, extra GPU,
restart of a stopped rental or extension is implied by this document.

Use at most the remaining part of the established **eight-hour, single-GPU** allowance,
with any earlier authorized deadline taking precedence. Record absolute physical-work
and hard-stop timestamps. Setup, repairs, fitting, idle allocation and preservation all
count. Starting a new goal or study never resets the provider clock.

Reserve the last hour for preservation, reporting and shutdown. Prefer CPU preparation
before rental where possible. Initial guidance is no more than 75 allocated minutes
for stack preparation/engineering and 30 for optional offline model repair. These are
caps, not extra allocation; remaining rollout work must still fit. Preserve at least
30 minutes of repair headroom before the physical-work cutoff when selecting a size.

Use end-to-end measured episode timings from the development screen. A starting
planning bound is `max(180 seconds, 1.25 * slowest completed episode)`, increased when
native-horizon or initialization timing shows it is optimistic. The 180 seconds is a
planning floor, not a hardware benchmark. Include audits, queries, capture, model work
and the replacement reserve in the estimate. Choose full/compact without search scores.
Never shorten the 1,100-action horizon to make the estimate fit.

### Recover a process without discarding the experiment

Before physical work, implement and exercise the following in new recovery launchers:

1. Persist an atomic run state with current stage/job, source/config hashes, completed
   IDs, RNG states, attempt counters, deadlines and next action. Resume from audited
   receipts rather than rerunning completed episodes after an agent/coordinator restart.
2. On an exception, capture a redacted traceback and partial cost ledger. Classify import,
   environment, source mismatch, inference/OOM, rendering, missing artifact, serialization
   and scientific-model failures separately. A task timeout is a valid task outcome,
   not an infrastructure retry.
3. Make a specific repair, run the smallest relevant CPU/saved-input check, then a
   bounded engineering check if needed. Allow at most three evidence-producing repair
   attempts per root cause, each at most twenty minutes, within the setup/remaining
   budget. Repeating an identical command with no changed evidence is not a repair.
4. Permit at most eight replacement full attempts across the session, allocated among
   stages/arms before launch with equivalent per-method rules. Preserve the original
   invalid attempts and charge replacement costs to the method. Do not replace valid
   poor outcomes. A missing required panel blocks its selection; it has no zero fitness.
   If reserve is exhausted, report the incomplete comparison rather than forcing a rank.
5. An implementation-only fix may resume under a new source snapshot after compatibility
   checks. A change to controller semantics, target, labels, model recipe or selection
   creates a new development version and freeze. Do not combine incompatible comparative
   records; any required restart consumes declared reserve and remaining time.
6. If one scientific recipe fails, run the next bounded development recipe or the
   predeclared direct-search fallback. Record abandoned routes and all costs. Once final
   audit labels are opened, changes are exploratory and require new future evaluation.

Essential checks concern native labels, causal inputs, action units, partition isolation,
scene validity and accounting. They cannot be waived to obtain a launch. Model quality
checks govern model-assisted branches; they are not prerequisites of direct evolution.

### Separate experiment supervision from provider shutdown

An independent supervisor enforces per-job/process-group limits and the final allocation
deadline even if the agent exits. Test this on synthetic subprocesses. A nonzero worker
exit, failed model gate or report exception must **not** call provider stop through an
exit trap, `finally` block or default coordinator failure path.

Do not stop the provider early merely because a branch failed or one diagnostic is
finished. Keep using remaining authorized time for the repair queue, a feasible declared
comparison and preservation. Early provider stop is allowed after the complete selected
comparison and required evidence are audited and preserved, or on explicit user direction.
At the hard allocation deadline, stop the instance retaining its disk regardless of
whether the research goal succeeded. Never disable that deadline to chase improvement.

If all executable routes are exhausted, save the exact blocker, failed repairs, remaining
time and next useful work. Continue useful local/CPU analysis where possible and surface
the blocker; do not invent a result or leave an unowned GPU worker running. Respect the
user's keep-running instruction and the hard shutdown time. A credentials/access blocker
may need user input; elapsed time does not supply that authorization.

Verify provider actual state after a stop request, preferably from an independent
controller, and save the receipt privately. A successful API acknowledgement alone does
not establish that billing stopped. Incremental backups must occur during the run so a
final upload failure cannot justify exceeding the hard limit.

## 7. Evidence and completion audit

Implement only the needed new helpers under `experiments/helpers/maniguard_recovery_*`,
entry points under `experiments/scripts/`, and configs under
`configs/maniguard/recovery/<study-id>/`. Save generated work under
`runs/maniguard/recovery/<run-id>/`. These are proposed interfaces, not existing commands.
Reuse audited policy, capture and outcome helpers deliberately; do not invoke old
absolute-time coordinators or edit frozen Part 1 to 4 implementations to bypass a gate.

Before launch, verify exact identity, cap geometry including clipping, trigger timing,
ordinary-learning updates, selection/confirmation, task isolation, crash/resume accounting
and supervisor cleanup. Use meaningful targeted tests and the intended explicit
uv-managed interpreters. Never root-sync the historical project into ManiGuard envs.

Completion requires the selected comparison's planned valid records and an audited
account of every attempt; frozen controller/model artifacts; lineage and learning logs;
independent outcome counts for every method/reference; per-task safe/unsafe completion
and unfinished categories; costs and uncertainty; representative matched videos; and
verified private evidence/model revisions. Keep generated runs, raw data, credentials
and weights out of Git. Update the research report from the new receipts, preserving
the original results and separately identifying exploratory repairs.

A stopping deadline or exhausted repair queue can leave the goal unmet. Report which
requirements are missing and the next costed action. Do not mark the goal achieved just
because the instance must stop. A completed null comparison can satisfy an objective
to obtain a rigorous result; it cannot satisfy a claim of positive improvement.

## 8. Positioning to verify before a contribution claim

The following primary sources were checked on 19 October 2026. They constrain the
claim, without establishing that the proposed repairs will succeed:

- [VLA-Pilot](https://arxiv.org/abs/2511.14178) already studies inference-time evolutionary
  steering of frozen VLAs. Distinguish selecting a reusable execution controller during
  adaptation from evolutionary action steering during deployment.
- [Extract-QD](https://arxiv.org/abs/2502.06585) addresses uncertain quality-diversity
  evaluation. Re-evaluation of noisy elites is not a new contribution by itself.
- [Fontaine and Nikolaidis](https://arxiv.org/abs/2012.04283) already use quality diversity
  for robot failure scenarios. Diversity acquisition needs a downstream repair comparison.
- [FailureSpot](https://arxiv.org/abs/2609.04277) studies fine-grained VLA failure detection
  and annotation efficiency. Keep physical specification events distinct from task failure.
- [Boetius and Leue](https://arxiv.org/abs/2405.15430) already study counterexample-guided
  agent/critic repair. This proposal keeps the VLA fixed and must measure its own benefit.

Recheck the nearest methods when the implementation and claim are frozen. The current
target is convincing comparative evidence and an explained mechanism; novelty and a
GECCO-level advantage remain claims to establish with that evidence.
