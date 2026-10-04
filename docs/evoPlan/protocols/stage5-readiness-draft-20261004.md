# Stage 5 readiness-route declaration, draft for supervisor review

**Status: draft, not an executed or final preregistration.** The user has requested
that further attempts to repair the controller for strict Gate 0 stop, and that
Stage 5 be proposed under the readiness route. This supersedes preparation of the
direct-simulator fallback as the next action. The supervisor's scientific decision
is still required by the user's request. No message has been sent to the supervisor,
no new episode bank has been generated and no Stage 5 search has begun.

The [review configuration](../../../configs/evo/stage5_readiness_draft.yaml)
is explicitly disabled. Exact root/seed counts and scoring need approval and a
power/throughput audit before an executable protocol is committed. The original
readiness audit permitted bounded pilots only; this proposed main experiment is
an explicit extension, not an authority already granted by passing that audit.

## Decision requested from the supervisor

Approve studying selection-induced changes in an imperfect visual controller's
imagined-real safety gap, using k=1 as the main noisy-imagination setting and k=0
as the reference, despite the preserved failures of strict Gate 0 and the joint
transfer screen. Approve the matched-termination penalty comparison below and
its ROSARL-style adaptation. Agree in advance how a flat or uncertain selection
curve will be reported. The controller, model and readout remain frozen.

The scientific target is a measurable change caused by search pressure, not a
claim that the starting controller is safe. A later safety-benefit claim still
needs reduced real violations and retained real progress on new episodes.

## Evidence and qualifications

The [fresh baseline](../results/stage2-readiness-baseline/walker2d-evo-s2-readiness-baseline-20261004-1/README.md)
has 51/256 real failures, 2/256 imagined failures and 100.3% of recorded progress.
Its separately declared measurement criteria passed; its strict failure-count
criterion did not. Repeated repair diagnostics on 24 reused roots do not justify
another unrestricted controller search.

In the [transfer pilot](../results/stage4-transfer-pilot/walker2d-evo-s4-transfer-pilot-20261004-1/README.md),
all 97 baseline/residual candidates imagined zero failures despite real failure
counts from 14/64 to 49/64. The real-image readout caught 97.0% of failures in
those repeated candidate-root pairs. In the [noise diagnostic](../results/stage3-noise-diagnostic/walker2d-evo-s3-noise-diagnostic-20261004-1/README.md),
k=1 produced safety/progress correlations of 0.713/0.446 across all 101 candidates,
including four attenuation controls. The primary residual-only view was
0.675/0.374. No level passed the declared joint screen, and neither correlation
should be substituted for the other population's result.

At k=1, mean predicted failure was 3.29% versus 37.89% real failure across the
97 candidates. This supports a development hypothesis about risk ordering, not
probability calibration or established discrimination on new policies. The same
four noisy rollouts and one noise seed were used across candidates, so Monte Carlo
and search-seed robustness remain to be checked.

Two causal qualifications matter. First, the held-action teacher test identifies
a harmful action interface, but it does not prove that observation frequency is
the sole cause of failures in a controller emitting ten distinct actions.
Teacher identity, imitation and closed-loop state shift remain possible factors.
Second, an all-zero imagined safety signal at k=0 does not prevent selection from
amplifying error: selecting for overstated progress can preferentially select real
failures. The experiment must distinguish the baseline gap from its increment
under search rather than assume the increment is absent.

## Frozen assets and policy

Use the existing Walker2d health rule, frozen LeWM, physical probe, normalisers
and six-member visual/action-history controller. Keep 10-step observation blocks
and ten distinct sequential motor actions per block. No new model or controller
training is proposed.

Search over the 1,020 coefficients of a linear action residual on the existing
fixed 16-dimensional projection plus bias. Fix base gain to one; the pilot's
1,021st gain-offset coordinate is not evolved. This is a declared change from the
pilot population, not a new teacher or a repaired controller. The zero residual
must reproduce the frozen controller exactly in real execution.

Precondition residual coordinates using projected-feature RMS on the existing
64 noise-fit roots only, with a 1e-6 floor. Freeze that preconditioner before
collecting Stage 5 data. With 17 features, initial independent coefficient standard
deviation 0.05/sqrt(17) gives expected unclipped action RMS 0.05 on those calibration
inputs. Actual clipped RMS is logged. Use separable CMA-ES, explicitly fixing
diagonal covariance throughout, starting from zero residual in every run.

Controller and projection archive revisions are pinned in the review config.
The frozen noise scale comes from the completed noise diagnostic; record its
local hash and private archive revision before the executable protocol is frozen.
Existing calibrated errors must not be refitted on the new evaluation bank.

## New whole-episode banks

Proposed roles are 96 fitness episodes, 64 policy-selection episodes and **320
final evaluation episodes**, one root per source episode: 480 new episodes in
all, excluding rejected source episodes. The 320 count is provisional, pending
power review. All roles are assigned by disjoint seed ranges before collection.
They share the same frozen competent-actor/noise mixture as the readiness audit.

Preserve the disclosed healthy-recorded-future conditioning and root sampling
rule. Charge and archive every attempted episode, including rejected short
trajectories. Cap attempts at four times the requested count in each role.
If the cap is reached, report incomplete collection; do not fill missing roles
with old or sibling roots or change eligibility. Invalid snapshots trigger an
integrity review and retain their logs and charged costs.

During search, sample 32 of the 96 fitness roots without replacement each
generation. For a given search seed and generation, all arms and population
sizes share that sample and the corresponding noise stream. The fixed 64-root
selection bank chooses checkpoint policies. The final bank supplies no fitness,
checkpoint choice, stopping decision, noise fitting or rule tuning. Defer all
controller outcomes on it, including baseline and diagnostic readouts, until
all searches and checkpoint picks have been frozen and hashed. Source-teacher
collection outcomes used by the declared root eligibility rule are necessarily
observed and remain disclosed.

## Factorial search and pressure sweep

| Component | Proposed levels |
|---|---|
| Imagination noise | k=0 and k=1, frozen sigma |
| Penalty | Zero unsafe penalty and ROSARL-style unsafe penalty |
| Population size | 16, 64, 256 |
| Generations inspected | 0, 1, 4, 16, as prefixes of the same 16-generation run |
| Paired search seeds | 10, provisional pending seed-variance review |
| Fitness noise samples | 1 at k=0, 4 at k=1 |
| Selection noise samples | 1 at k=0, 16 at k=1 |
| Final audit noise samples | 1 at k=0, 32 at k=1 |

This is 120 search runs and 360 non-baseline checkpoint-policy slots. Offspring
budgets reach 4,096 at population 256 and generation 16. Generation zero is the
shared frozen baseline. Report population size, generations, offspring count,
selection-bank queries and total predictor rows separately. Noise adds queries,
so this is an equal-offspring comparison, not an equal-compute comparison.
Matched offspring budgets, such as 16x16, 64x4 and 256x1, give descriptive checks
of population/generation effects. The primary low/high contrast bundles these
choices and is not a claim about population size alone.

## What "without the penalty" means

To isolate the unsafe penalty, both arms use the same imagined termination rule.
For a rollout, let v indicate any imagined health violation and let U be progress
strictly before the first violating block, or full progress if no violation occurs.
The two fitness values, averaged over fitness roots and noise samples, are:

- Zero penalty: U.
- ROSARL-style: U + v(V_MIN - V_MAX).

V_MIN and V_MAX are the cumulative minimum/maximum U among this run's fitness
rollouts, updated once for the entire generation before ranking it. They never
use real outcomes or selection/final-root returns. Both arms log these bounds;
only the ROSARL arm applies the penalty. Zero penalty still contains the shared
termination signal. It is not pure unrestricted-progress optimisation when
violations occur. Full-horizon imagined and real progress are reported separately.

This is the repository's finite-segment, observed-return adaptation, not an
unqualified implementation of ROSARL's value-function definition. It awaits the
supervisor's check. If a pure full-horizon-progress comparator is preferred,
revise and approve this declaration before execution: that comparison changes
both termination and penalty, so its interpretation would differ.

Do not compare saved best-fitness numbers from different generations whose
penalties or fitness-root sets differ. Nominate the best current-generation
candidate on fitness roots; evaluate it once on the fixed selection bank with
its separate noise stream. Cache the baseline and every nominee's full outcomes.
At each checkpoint, rescore all cached nominees using the current fitness-only
bounds and choose the best, with chronological candidate IDs breaking ties.
Those selection outcomes never update bounds or CMA-ES. This costs one new
selection evaluation per generation plus the initial baseline, as charged below.

## Final outcomes and estimands

Replay every frozen checkpoint policy and the shared baseline on the same final
roots for exactly 100 sequential simulator steps. Continue logging after failure
to retain a common horizon, while recording first failure and pre-failure progress.
Dense health at every step is truth; block-end truth and readouts on real images
remain diagnostic decompositions. Evaluate every selected policy in imagination
at both k=0 and k=1, regardless of its training arm, with audit noise independent
of fitness/selection draws and paired across policies.

For seed s, evaluation episode e, condition c and pressure b, record real failure
R and mean imagined failure I at the specified audit noise. Define
G(c,b) = mean over s,e of [R(c,b)-I(c,b)]. Plot G, real failure, imagined failure
and real progress against pressure, as well as each change from the baseline
under the same audit noise. Also plot the common k=0 audit for every arm; that
prevents a higher noisy predicted failure rate from being presented by itself as
improved real safety.

Let A(c) = G(c, high)-G(c, low), with low=(population 16, generation 1) and
high=(population 256, generation 16). Two proposed co-primary contrasts are:

1. A(k=0, zero penalty), expected positive if search amplifies the gap.
2. A(k=1, ROSARL-style)-A(k=0, zero penalty), expected negative if the combined
   method reduces that amplification, using each arm's own audit noise.

Use two-sided 97.5% intervals for each, giving a conservative 95% family coverage
for two comparisons. Report all estimates even if their intervals include zero.
The second contrast changes both noise and penalty; secondary paired factorial
contrasts isolate their contributions. Baseline-to-checkpoint changes, common-k0
audits, the k=1 curve and matched-budget comparisons receive descriptive 95%
intervals. None replaces a failed primary contrast after inspection.

For a claimed practical fix, require an interval supporting lower **real**
failure at high pressure and a lower paired 95% bound on the ratio of mean real
progress of at least 0.90 against the comparison arm. This proposed 10% loss limit
needs supervisor approval before data collection; it is not the older readiness
floor or a passed safety gate. A nonpositive denominator makes the ratio undefined
and prevents that retention claim. Short-branch outcomes do not establish stable
full-episode walking or certified safety.

## Uncertainty and power

Resample search-seed IDs and independent evaluation-episode IDs on their two
crossed axes, maintaining identical resamples for all arms, pressures and paired
baselines. Never treat 10 seeds x 320 roots as 3,200 independent episodes.
Use 10,000 bootstrap replicates; report seed-wise estimates as well as the pooled
contrast. Uncertainty is conditional on the frozen model, fitness/selection banks
and declared source distribution. The 32-sample audit reduces but does not prove
negligible Monte Carlo error; its adequacy needs a development-only check.

The pilot's 272-root estimate was a one-selected-policy normal approximation
for an unadjusted 10-point contrast. It did not estimate between-search-seed
variance, the second primary contrast, or the effect of multiplicity. It cannot
establish power for this design.

The [budget and sensitivity calculation](stage5-readiness-draft-budget-20261004.json)
illustrates this. Assigning the old paired variance 0.3458 entirely to roots,
with 320 roots and 10 seeds, gives approximate power 0.79 at alpha 0.025 if the
paired seed standard deviation is zero. Assuming seed SD 0.10 lowers that to
about 0.48. These are scenarios with zero interaction variance, not fitted power
estimates or conservative guarantees. More roots alone cannot eliminate search
seed variance. Before execution, use development-only seed-variance evidence or
an explicitly agreed sensitivity range to lock root/seed counts and document
attainable power for both primary contrasts. Do not increase counts after looking
at final outcomes to obtain a preferred finding.

## Budget, implementation and stopping

For the draft counts, before any optional caching savings:

| Charged component | Upper/planned count |
|---|---:|
| Fitness predictor rows | 172,032,000 |
| Selection predictor rows | 11,097,600 |
| Final imagined audits, including baseline at both noise levels | 39,072,000 |
| Total predictor rows | 222,201,600 |
| Real policy, baseline and exact-reference replay steps | 11,584,000 |
| Maximum source-episode collection steps | 1,920,000 |
| Maximum new real steps before separate debug/power pilots | 13,504,000 |

The 12-hour GPU ceiling is proposed, not a verified runtime. The earlier
throughput measurements had different batch shapes; benchmark the exact proposed
fitness, selection and final-audit layouts on old development data before locking
run sizes. Count that benchmark and every failed/replayed attempt separately.
Charge historical model/data acquisition once as shared provenance, not again
as new collection in every arm. Save all per-generation choices, penalty bounds,
RNG states, costs and frozen checkpoint parameters; archive immutable bundles.

Existing CMA-ES infrastructure does not implement this exact protocol: it needs
the residual policy wrapper, matched-termination zero-penalty scorer, separate
selection bank, penalty-aware nominee rescoring and crossed seed/episode analysis.
Add new modules/configs and meaningful equivalence, pairing, resume and accounting
tests. Keep recorded experiments intact. No executable launch is authorized by
this draft. Integrity failures halt the affected run with its costs retained;
resume only from a verified checkpoint. Do not adapt search budgets, k or policy
architecture after final results are observed.

## Interpretation agreed before execution

A positive first contrast would support selection-amplified error at the tested
pressure range. A smaller gap alone does not prove a safety benefit; use real
failure and progress results. A flat curve with wide intervals is inconclusive,
not evidence of no effect. A narrow interval that excludes a 10-point increase
bounds amplification at this scale. A null or adverse noise/penalty result is
reported with the same endpoints and frozen protocol.

If selection amplification is weak, the fallback scientific framing is a measured
baseline safety-prediction failure and a test of whether calibrated noise restores
risk ordering on new episodes. The current k=1 development result motivates that
question but does not already answer it. Publication suitability cannot be
promised; no outcome justifies changing the claim or endpoint after seeing data.
The direct-simulator track remains documented and deferred, not the next automatic
experiment. Confirm this interpretation with the supervisor before Stage 5.
