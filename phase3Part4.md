# Phase 3, Part 4: evolutionary acquisition of safety counterexamples

**Date:** 19 October 2026.

**Implementation outcome, 19 October 2026:** the bounded offline premise check is
complete and **scientifically gated before scene engineering or physical acquisition**.
The frozen primary predictor has development average precision .0121 and alarms on
2,045/2,229 negative windows, with zero confident misses among one first-event episode.
The [result and evidence](docs/research/reports/ManiGuard%20Part4%20counterexample%20premise.md)
and [frozen offline protocol](configs/maniguard/part4/offline-protocol.json) retain the
failed criteria and limited task support. No new simulator interaction was made.

**Prospective repair, 19 October 2026:** the
[shared recovery plan](phase3Recovery.md#4-part-4-repair-task-support-and-warning-calibration)
adds a bounded multi-task data and calibration route, finite visual/robot ablations and
a separately named error-acquisition alternative if confident-miss support remains
absent. These changes belong to a new protocol; the frozen Part 4 criteria and negative
result stay intact. A failed detector candidate should trigger the next permitted
development step, while a failed scene builder blocks physical scene search. Neither
event alone requests shutdown of the whole research instance. Direct controller
evolution remains available under its own frozen comparison.

**Original experiment direction:** the remaining sections preserve the prospective
comparison design. They are not claims that its gated stages were executed. The
proposed ceiling is **80 full episodes**, with a small, separately counted engineering
allowance, on one RTX 4090 or RTX 5090 within eight hours of allocated wall time. Scene
generation, predictor feasibility and measured throughput must support that allocation
before the comparative search begins.

This follows the completed [Part 2 feedback experiment](docs/research/reports/ManiGuard%20Part2%20correction%20triggered%20feedback.md)
and [Part 3 forecast audit](docs/research/reports/ManiGuard%20Part3%20model%20assisted%20evolution.md).
The broader direction remains in [phase3.md](phase3.md). Earlier plans, frozen protocols
and recorded results remain evidence of their original experiments.

## 1. Research question

> Can evolutionary search for diverse, confidently missed physical hazards improve a
> small safety predictor more efficiently than random sampling, uncertainty-guided
> acquisition or concentrated failure-seeking evolution?

Evolution would search bounded scene parameters and acquire training trajectories.
The manipulation policy and its execution controller would remain fixed. Ordinary
supervised learning would update the safety predictor after acquisition.

The immediate outcome is **better safety assessment on unseen tasks per acquisition
cost**. Safer closed-loop task completion would require a later experiment using the
improved predictor. The predictor in this pilot observes execution without changing it.

The hypothesis has three parts:

1. Useful predictors can still miss physical hazards while assigning low risk.
2. Evolution can acquire such counterexamples more efficiently than the sampling
   references, without repeatedly exploiting invalid scenes or one lucky rollout.
3. Preserving different counterexamples produces better held-out predictor repair than
   concentrating on the easiest repeatable miss.

Each part needs evidence. More failures, more occupied archive cells or lower training
loss alone would not establish the proposed contribution.

## 2. Why this follows Parts 1, 2 and 3

Part 1 did not establish a safe-completion benefit from the selected controller. Its
failure investigation identified grasp-detection gaps, direction changes from joint
clipping and weak selection evidence, without demonstrating a sufficient repair.
[Part 1 investigation](docs/research/reports/ManiGuard%20Part1%20failure%20investigation.md).

Part 2 completed sixteen episodes. Identity, restrained fixed-eight execution and
correction-triggered feedback each achieved 3/4 safe completions; fixed-four achieved
2/4. Every condition had one violating episode. The adaptive trigger never caused an
early query, and frequent feedback more than doubled queries without an observed gain.
These findings do not establish that an independently calibrated active trigger fails.

Part 3 completed its model-feasibility stage. Four-action prediction error was 45.1%
worse than persistence, and the principal readout was inaccurate even on actual future
representations. The correction-dependent trust rule became equivalent to its ensemble
reference. The proposed 64-episode evolutionary comparison was therefore not launched.

**Poor prediction accuracy does not establish confident safety mistakes.** Part 3
recorded zero false reassurances on 99 correlated negative-margin windows from one
episode under its particular definition. That neither demonstrates the new premise nor
establishes a reliable warning system. Part 4 must test its own event, timing and error
definitions before using them as an acquisition objective.

## 3. Intended contribution and nearby work

The candidate contribution is a tested evolutionary acquisition method that targets
diverse physical hazards missed by a safety predictor, with evidence that its examples
improve held-out prediction at a lower total simulator cost.

The main ingredients already have precedent:

- [Quality-diversity scenario generation](https://arxiv.org/abs/2012.04283) uses
  MAP-Elites to discover diverse failure scenarios in robotic shared autonomy.
- [FailureSpot](https://arxiv.org/abs/2609.04277) studies timestamp-level VLA failure
  detection and uncertainty-based acquisition for annotation. Its task-failure labels
  are not interchangeable with ManiGuard's physical safety specifications.
- [Counterexample-guided repair using safety critics](https://arxiv.org/abs/2405.15430)
  jointly repairs an RL agent and safety critic through constrained optimization.

Using MAP-Elites, training a detector or collecting counterexamples is not novelty by
itself. The narrower hypothesis concerns the value of diversity when collecting
confidently missed physical hazards, measured through downstream repair and complete
acquisition costs. Recheck the nearest falsification, active-learning and data-selection
methods before making a paper-level novelty claim.

The eight-hour pilot can identify a promising mechanism. A GECCO contribution would
need stronger replication and comparisons than this single acquisition run per method.

## 4. Fixed system and information contract

Retain ManiGuard Clutter, its pinned policy checkpoint and simulator sources, 20 Hz
control, 120 Hz physics, native completion rules and 1,100-action horizon. Start with
Part 2 condition B as the working fixed execution reference: .04/.02 rad caps, the
original finger heuristic, smoothing one and query interval eight. Freeze the exact
export and hashes before collection. Its selection is a common reference decision,
not a claim that it improves safety.

Every acquisition method receives the same policy, execution controller, predictor,
initial corpus, search domain, validation rules and episode allowance.

The predictor may use current and historical camera features, robot measurements,
already proposed commands and executed-action history. Its inputs must exist before
the forecasted event. If historical donors use different execution settings, either
restrict the donors to the fixed setting or condition explicitly on the known controller
configuration and query age. Freeze that choice during offline development.

Native object states and monitor predicates supply offline labels, scenario validity
checks and search evaluation. They are excluded from the predictor and live execution
controller. Scene geometry used by a search archive is privileged search information,
and must be declared as such. All methods receive equivalent access to that metadata.

The safety monitor records physical events. It is not an action-changing filter.
Predictor scores, imagined outcomes and task failure are not substitutes for measured
physical violations.

## 5. One event and a causal warning target

Begin with **the first native uprightness violation of any protected object**. Preserve
the pinned object scopes, local-axis convention and per-object tilt thresholds. Check
that numerical margins reproduce native Boolean propositions on the saved observations.
Other monitor violations remain in the physical outcome report.

At observation index t, the label asks whether the first uprightness violation occurs
within the next H control actions, while no such violation has yet occurred. The model
input ends at t. A working choice is H = 8 actions, or 0.4 seconds; the final horizon
must be justified on development data and frozen before new comparative outcomes.

Also freeze a useful warning deadline L before the event. A working choice is L = 4
actions, or 0.2 seconds, with L no greater than H. For an event first observed at index
tau, the primary event measure asks whether an alarm occurred in the eligible interval
from tau-H through tau-L. An alarm first appearing after the deadline is late.

Exclude windows without enough recorded future observations. Do not pad terminal
frames, fabricate post-success safety or label every frame of an unsuccessful episode
as hazardous. Already violated states are outside the first-event warning denominator.
Episodes with insufficient pre-event history are reported separately.

Report both eligible warning-window errors and event-level missed warnings. One
violating episode contributes one first event, however many positive windows it contains.
Task, scene and trajectory groups remain visible in every denominator.

## 6. Small direct predictor and the offline premise check

Reuse verified Part 2/3 RGB, robot measurements and frozen visual features where
available. Part 1 lacks the required later exact observations and numerical targets;
do not invent those data from its aggregate results. Verify private backup revisions
and hashes on the run machine before depending on them.

Start with a regularized direct event predictor on frozen DINOv2 features and a short
causal history. A small episode-bootstrap ensemble supplies a common disagreement
measure. Keep a simple linear/logistic reference. A compact MLP is a bounded alternative
if the linear reference is inadequate. Encoder fine-tuning and latent dynamics training
are outside the initial scope.

The Part 3 visual-only readout result motivates an information ablation; it is not a
validated replacement detector. Any scaling, PCA, representation choice, class weights
and history length are fitted or chosen using designated development tasks only.
Freeze architecture, ensemble size, seeds, optimizer, update count and stopping rule.
Avoid an open-ended architecture or hyperparameter search.

Use whole-task groups for fitting, calibration and the development premise check.
Existing Part 3 audit cases are exposed development evidence for Part 4; they cannot
serve as its fresh final audit. Historical outcomes and acquisition costs remain visible.

The bounded offline check must establish:

1. Correct observation/action/event alignment without future information in inputs.
2. Useful discrimination relative to the simple reference on separate development
   tasks, rather than success from class imbalance or an always-safe/always-alarm rule.
3. Candidate low-risk misses with useful lead time, including their episode and task
   support and their relationship to ensemble disagreement.
4. An operating rule that balances missed events and false alarms without relying on
   the yet-unseen Part 4 audit.

Fit a common probability-calibration rule where supported. Define an alarm threshold,
low-risk cutoff and low-disagreement cutoff on calibration data. A 5% negative-window
false-alarm target is a working operating point, not an empirical safety guarantee.
Report the actual calibration rate and episode-level false-alarm frequency.

Numerical discrimination and support criteria remain to be frozen before collection.
Missing event support is inconclusive. If existing evidence is promising but thin, the
predeclared eight shared initial episodes may provide the bounded additional premise
check. The predictor stays frozen through them. Do not add rescue episodes or repeatedly
change the target until a desired premise appears.

## 7. Scene genome, validity and mutation

Use meaningful spatial parameters with a local mutation structure. Random policy seeds
are repetitions, not spatial genes. Start with two to four coordinates controlling
bounded planar offsets of a declared target or protected clutter object/group.

Scene editing is a new implementation dependency. A builder must produce coherent
initial states, object references, goal metadata and monitor scopes. Do not assume that
changing a transform in one file correctly updates the benchmark. Generated scenes are
a supplemental study set, distinct from ManiGuard's locked benchmark snapshots.

Keep instructions, object identities, physical properties, lighting, cameras and safety
thresholds fixed in the first pilot. Check finite parameters, legal workspace bounds,
initial nonintersection, stable support, reachability, an initially unsatisfied goal and
absence of an initial safety violation. Recheck after any required settling physics.
Reachability does not prove that the frozen policy can solve the scene.

Freeze the selected development base tasks, coordinate frames, units, bounds, normalized
mutation scales, boundary handling, validity checks and duplicate policy. Offspring
remain attached to their parent base task. Balance base-task opportunities across arms.
Record serialized scene hashes and validate reset repeatability before scientific use.

Invalid proposals, rejection sampling, settling, reset checks and failed launches remain
in the cost ledger. Put a finite bound on validation attempts. A slot that consumes
physics and fails validity is not silently replaced with a free full episode.

## 8. Counterexample quality and diversity

Let p(t) be the frozen ensemble's mean predicted event probability and u(t) its declared
disagreement measure. A confidently missed event must be a physical uprightness onset
with adequate pre-event history, no timely alarm, low predicted risk and low disagreement
under the frozen cutoffs.

A simple working episode quality is 1-p(tau-L) for a qualifying miss, and zero otherwise.
Invalid scenes have no scientific quality score. Fix numerical ties by proposal order.
Do not reward multiple correlated windows from the same event, larger physical damage,
or a violation already present at initialization. Record ordinary misses, uncertain
misses and other physical failures separately.

The search score uses outcomes after execution. The predictor never receives those
future labels as live inputs. A low score or ensemble agreement is empirical model
behavior, not calibrated certainty that execution is safe.

For diversity, begin with a small archive over declared initial interaction geometry.
One working descriptor is protected-obstacle position along and across the nominal
robot-to-target approach, normalized to a common frame, with a 4-by-4 grid. Exact
object selection, axes, bounds, bins and out-of-range handling need freezing.

Use a descriptor that is meaningful across the selected base tasks and repeatable
without policy-rendering luck. Geometric archive coverage is not evidence of distinct
physical failure mechanisms. Report event object, contact/approach context and recurrence
to assess whether the retained cases actually differ.

## 9. Four acquisition methods

All methods start with the same eight evaluated seed scenarios. They receive twelve
new acquisition slots each and retain every valid acquired trajectory for repair.

1. **R: stratified random sampling.** Draw from the declared valid scene domain using
   a fixed space-filling schedule, without selecting parents from outcome scores.
2. **U: uncertainty-guided acquisition.** Use the common mutation operator and a small
   elitist population, selecting parents by a frozen summary of predictor disagreement
   measured along each executed trajectory. It does not use future safety labels in
   that acquisition score.
3. **E: concentrated counterexample evolution.** Use the same population size, mutation
   operator and update schedule as U, with the factual missed-hazard quality from
   Section 8. This tests the acquisition objective without a diversity archive.
4. **D: diverse counterexample evolution.** Use the same missed-hazard quality and
   variation operator as E, retaining elites by archive cell and sampling parents
   across occupied cells. This tests the additional value of diversity preservation.

A working E/U population contains four retained scenarios initialized from the shared
eight, with one evaluated offspring and update per new slot. D updates its archive
after each slot. Freeze parent selection, task balancing, ties, mutation scale and the
fallback when all qualities are zero or no archive parent is available.

U must pay for the rollouts from which it obtains uncertainty. It cannot rank a large
pool of fully simulated trajectories and charge only for selected examples. In this
pilot it targets high-uncertainty scenarios through sequential evaluated mutation, not
an uncharged pool-based annotation oracle. Initial renders or alternative candidate
probes also count if introduced.

Freeze the acquisition predictor throughout all four searches. Updating it online in
one arm would change both the search objective and information available to that arm.
Interleave method slots to reduce host/time effects. Separate method seeds, policy
sampling seeds and scene genotypes in the records.

The principal contrasts are D versus E for diversity, E versus U for targeting observed
misses rather than uncertainty, and D versus R for end-to-end acquisition value.

## 10. Proposed physical allocation

The **80-full-episode ceiling** consists of:

- **8 shared initial episodes**, available equally to all methods and charged once.
- **48 acquisition episodes**, twelve new slots for each of R, U, E and D.
- **8 confirmation episodes**, two fresh policy-seed repetitions per method.
- **16 independent audit episodes**, shared when comparing all trained predictors.

The maximum full-episode control allowance is 88,000 actions. Actual counts may be
smaller because of native success termination, invalid slots or a scientific stop.
Do not terminate at a violation to make failure-seeking methods appear cheaper.

Separately allow up to **three 32-action physical engineering checks**, at most 96
controls, for scene/reset validity, causal event logging and passive predictor execution.
These are engineering data, excluded from comparative fitting and evaluation. Setup
failures and initialization/settling physics are counted in addition; unobserved counts
remain unknown rather than zero. There are no unbounded warmup runs or automatic retries.

Before the method-specific slots, evaluate the frozen premise, scene support and timing
decision using only the designated evidence. If feasibility fails, stop with that
bounded result. A smaller balanced design can be frozen before method-specific outcomes,
with revised accounting; do not selectively enlarge a promising method's allocation.

For confirmation, repeat the highest-ranked valid acquisition scenario twice per
method, using fresh policy seeds and fixed tie/fallback rules. Record the original
discovery and both repeats, including failures to reproduce. These two repetitions are
a small diagnostic, not a precise recurrence estimate or independent search replication.

## 11. Partitions and the fresh audit

Freeze a manifest separating historical fitting, common calibration, development
premise checks, discovery base tasks, confirmation seeds and fresh audit tasks. Keep
all variants of a base task in the same development or audit group. Do not select
audit tasks according to acquisition success or known failure prevalence.

The working fresh-audit allocation is four previously unexposed base tasks with four
fresh policy-seed repetitions each. Freeze their scene distribution and any bounded
perturbations before acquisition; audit cases are sampled independently of search and
shared across every predictor. They remain separate from the existing main-study test
partition. Released VLA pretraining overlap is unknown.

Run each audit episode once under the fixed policy/controller and score every frozen
predictor on its saved observations. This is a matched predictor comparison without
four separate physical executions or hypothetical action branches. Audit labels remain
sealed until all repaired artifacts and thresholds are frozen.

If the sixteen audit episodes provide too few events, report limited or inconclusive
event-level evidence. Do not replace safe episodes with search-selected failures.
Search prevalence is not the deployment violation rate.

## 12. Matched predictor repair

After acquisition, update identical copies of the same starting predictor. Every copy
uses the same historical fitting corpus and shared initial data; each acquisition arm
adds all its own valid discovery trajectories. Include a common-data-only reference
using the same optimizer-update allowance.

Keep confirmation trajectories out of the primary repair comparison so that their
selection does not silently change training data. They remain charged diagnostic work.
Never train on fresh audit trajectories.

Match feature preprocessing, starting weights, training seeds, update counts, batch
sizes, episode/window sampling and loss weighting. Fit preprocessing using permitted
fitting data only. Cap per-trajectory contributions so long, nearly identical windows
cannot dominate. Freeze how different valid donor counts are handled; do not repair
an arm's missing data using another arm's private discoveries.

The main analysis uses all acquired data, including safe trajectories and unsuccessful
counterexample searches. Training only on archive elites would mix acquisition and
subset selection. An equal-size data-selection ablation can be done offline later,
with its selection rule and total pool-acquisition cost reported.

Use the same untouched common calibration group and predetermined rule to set each
repaired model's operating threshold. Freeze thresholds before opening the fresh
audit. Do not tune on audit labels to force equal observed false-alarm rates.

Save learning curves at common discovery prefixes where practical. Match model updates
at each prefix and charge all repeated fits, encoder work and calibration queries.

## 13. Measurements and interpretation

The primary repair evidence is **event-level missed timely warnings on the independent
audit**, reported with actual false-alarm behavior at the frozen operating points.
Also report warning lead time and a prespecified threshold sweep. If false-alarm rates
differ materially, a lower miss count alone does not establish a better trade-off.

Report acquisition quality separately: confirmed low-risk misses per charged attempt,
first-discovery cost, archive occupancy, geometric coverage, repeated-case recurrence,
validity rejection rate and failure-object/context coverage. Include all-zero quality
distributions and unstable archive entries.

For every physical stage, report safe completion, unsafe completion, safe unfinished
and violating unfinished episodes, together with native violation types. These outcomes
describe fixed-policy data collection, not gains from deploying a repaired predictor.

Preserve episode, task and scene denominators. Frames, ensemble members and nearby
windows are correlated observations. Use task/episode-grouped descriptive intervals
where supported; sixteen audit episodes and one search seed per method do not establish
an optimizer's expected performance. Zero observed violations in n independent episodes
has an approximate 95% upper rate bound of 3/n, not zero risk.

Episode allowances are matched, but realized controls and durations can differ. Show
results against cumulative simulator controls as well as episode counts, and report
physics, VLA queries, encoder/model calls and elapsed time separately. Do not claim
interaction efficiency solely from an equal number of episodes.

## 14. Eight-hour resource target

Use one existing validated single-GPU ManiGuard stack, one policy server and one serial
simulation worker. Use the explicit uv-managed policy/simulation interpreters. Do not
root-sync the repository environment into either platform environment.

Part 2 recorded 1,897.69 seconds across sixteen full episode processes, about 119 seconds
each. Part 3's eight common recordings took 1,216.82 seconds, about 152 seconds each.
At those historical rates, eighty full episodes would occupy roughly **2 hours 38
minutes to 3 hours 23 minutes** of episode-process time. At four minutes each they
would occupy 5 hours 20 minutes. These are planning extrapolations, not measurements
of new scene-generation or predictor overhead.

All preparation, downloads, engineering, validation failures, fitting, extraction,
calibration, reporting, idle allocation and preservation must fit within eight allocated
hours. The engineering/timing decision must include a conservative reserve for those
costs, not just the expected episode sum. Reserve the final hour for cleanup, export,
verification and reporting, and supervise workers independently of the coordinator.

This document does not start an allocation or authorize additional rental. Record the
actual machine, start time, physical-work cutoff and hard deadline for any execution.
An already running rental retains all time consumed; naming Part 4 does not reset its
clock. The historical Part 2/3 guard contains its own authorized twelve-hour allocation
and must not be reused unchanged as an eight-hour Part 4 budget.

If valid scene editing or useful prediction cannot be established within the bounded
preparation stage, finish with the diagnostic and its costs. Do not shorten the native
horizon, relax physical specifications or continue beyond the allocation to force a win.

## 15. Implementation sequence

1. Audit available assets, donor traces/features and source pins. Create a new Part 4
   output namespace; preserve earlier code, artifacts and reports.
2. Implement direct event labels and a small predictor, with CPU checks for causality,
   censoring, native/numerical agreement and whole-task partitioning. Complete the
   bounded offline premise check and freeze the predictor and operating definitions.
3. Implement the bounded scene builder and serialization/validity audit. Complete the
   engineering allowance, including saved-input policy and reset-repeatability checks.
4. Freeze common seeds, initial jobs, genotype domain and archive descriptor. Run the
   eight shared initial episodes. Seal the premise and capacity decision before any
   method-specific acquisition; change no score or model to fit these new outcomes.
5. Freeze the comparative schedule and run the four interleaved acquisition arms.
   Preserve every proposal, rejection, evaluated genome, score, parent and archive update.
6. Run the eight declared confirmation episodes and report recurrence without selecting
   replacement scenarios because repeats failed.
7. Train and calibrate matched repaired predictors. Export hashes, donor manifests,
   thresholds and update ledgers before opening the independent audit.
8. Run the sixteen audit episodes once, evaluate all frozen predictors on those traces,
   and export the complete matched analysis and cost accounting.

New reusable Python belongs in `experiments/helpers/`, entry points in
`experiments/scripts/`, and frozen settings/manifests in `configs/maniguard/part4/`.
New raw outputs should use `runs/maniguard/phase3-part4/<run-id>/`. These are proposed
locations, not an assertion that the implementations already exist.

Use the existing source/checkpoint pinning, passive RGB/state capture, action separation
and independent worker supervision patterns. Historical launchers may contain fixed
paths, task partitions and deadlines; reuse their interfaces deliberately rather than
restarting completed jobs.

## 16. Deliverables and prospective decisions

Deliver a frozen protocol, scene manifest and validity receipts; baseline and repaired
predictor exports; all four proposal/lineage ledgers; archive snapshots; confirmation
records; common audit predictions; grouped outcome/error figures; representative videos;
and a report distinguishing discovery from repair. Raw evidence and model weights remain
in approved private, revision-pinned storage, with verified hashes and no credentials
or licensed asset bundles in Git.

Before comparative execution, resolve and record:

- Event scope, H/L, low-risk and disagreement definitions, alarm calibration, minimum
  development event support and numerical discrimination criteria.
- Historical donor policy conditioning, representation/history, architecture, ensemble,
  training budget and common calibration tasks.
- Discovery and audit task identities, scene genes, physical bounds, validity repair,
  descriptor bins, population/archive updates, seeds and finite rejection limits.
- The exact physical engineering allowance, eighty-slot accounting, retry/failure rules,
  confirmation fallback, actual throughput and timestamped allocation.
- Primary repair metrics, sampling/weighting, treatment of incomplete or unequal donor
  sets, artifact freeze point and the complete audit-access rule.

These are implementation decisions to settle from permitted development evidence.
The plan does not require an evolutionary win before an experiment counts. A null
comparison, absent confident misses, nonreproducible discoveries or no repair benefit
is a retained result, not a reason to add unplanned trials.

## 17. What would justify the next study

A promising pilot would show reproducible missed hazards, a meaningful diversity
contrast, and a better held-out warning/false-alarm trade-off after matched repair.
The size and support of that gain matter more than an isolated successful video.

If D matches E, diversity is not supported as the explanation. If R or U matches D,
the evolutionary acquisition claim is unsupported. If failures are found but repair
does not generalize, the current method is a falsification diagnostic, not a demonstrated
data-efficiency improvement. If only search-selected cases improve, the fresh-task
contribution remains unresolved.

A larger GECCO study would need repeated independent acquisition seeds, broader tasks,
strong random/space-filling and acquisition references, an offline diversity-selection
control, and complete end-to-end cost curves. A later deployment study could ask whether
the repaired predictor improves safe completion at matched intervention burden. That
would require a specified action-changing mechanism, appropriate ordinary-learning
comparisons and fresh physical outcomes. This pilot establishes none of those later
claims automatically.
