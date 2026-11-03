# TwoRoom representation and mechanism study

Prospective protocol, 1 November 2026. Sachin approved the discussed two-stage study
and instructed execution. This is new work on existing Vast instance 58676387,
machine 19781, runtime hostname c45db82630e6, eight RTX 4090s, quoted $3.6637/hour.
Rental shutdown remains manual. No previous experiment or unused allowance resumes.

## Scope and fixed foundation

Stage A selects a search representation; Stage B tests its mechanism with fresh
training repeats. All work lives under SafeLeWM in new files and output paths.
The qualified LeWM/AP, checkpoint, memory, preprocessing, 396 inputs, 32-unit feature
layer and GRU, gain/residual output, passive monitor and 200-action cap are unchanged.
Training retains eight layouts and planner seed 8201. Preserve earlier evidence.
PPO, other environments, penalty tuning, risk-budget objectives and the dial remain
outside this study. No favorable-outcome extension or additional search is automatic.

## Stage A: representation development

Three representations, three paired repeats each, nine fresh CMA searches:

- full: all 19,172 standardized weight coordinates.
- head: only the last 132 coordinates (the 32-to-4 output layer); other weights stay
  at their generic neutral initialization. This freezes random internal features.
- subspace: 256 coefficients mapped to changes in all 19,172 weights. Sample a
  Gaussian matrix independently per repeat, normalize each row to Euclidean norm
  one, and save it. Decode theta = neutral + scales * (projection @ genotype).

All active standardized coordinates have the same initial marginal variance;
initial sigma 0.5 and fan-in scales are unchanged. The subspace introduces correlated
updates and restricts expressivity; it is not an equivalent parameterization.
Head-only search intentionally freezes the other layers. No action-response tuning.
Use diagonal CMA-ES 4.4.4 with explicit CSA, population eight, no mirrors and stable
ask/tell order. Stop by the exact action allowance, not suggested optimizer stopping.

A0/A1/A2 initialization seeds are 10101/10201/10301, CMA seeds 10103/10203/10303,
projection seeds 10105/10205/10305. Orders are full/head/subspace,
head/subspace/full, and subspace/full/head. Every search gets 40,000 actual actions.
Select its nominee on training outcomes: safe completion first, then all completion,
then fewer violations; exact ties choose the earliest candidate. For eight complete
cases the numeric score remains 81*safe + 9*complete - violations.

Lock all nine nominees before the development evaluation. The previous replication's
twelve final layouts are now explicitly development data, evaluated with its same
three planner seeds 9401/9402/9403: 324 episodes. Pool each representation's three
repeats (108 episodes) for the same lexicographic ordering. An exact tie prefers
full, then head, then subspace. This is configuration selection, not independent
confirmation of superiority. No selection from final Stage B outcomes.

## Stage B: mechanism comparison

Use only Stage A's selected representation, with fresh generic initializations and
no winning weights reused. B0/B1/B2 initialization seeds are 11101/11201/11301,
CMA seeds 11103/11203/11303, random-search seeds 11104/11204/11304, projection seeds
11105/11205/11305 and untrained-reference seeds 11102/11202/11302.
Within each repeat all methods share initialization, coordinate scaling and projection.

Five arms, three training repeats each, fifteen fresh 40,000-action searches:

1. main: safety-aware CMA with all inputs and carried recurrent state.
2. random: parameter random search with the same safety-aware scoring and distribution.
3. task: CMA maximizing original-goal completion count only. Neither ranking nor
   final training-nominee selection may use safety, path length or action efficiency.
   Safety remains logged and evaluated. Exact ties choose earliest candidate.
4. blind: safety-aware CMA with input coordinates 386:389 set to zero at every step,
   during training and evaluation. These are relative hazard XY and hazard radius.
   The monitor still supplies training labels; the policy receives no hazard geometry.
5. reset: safety-aware CMA resetting GRU state to zero before every action. Keep the
   explicit previous-action/displacement inputs, so this tests carried recurrent state,
   not all temporal information. Retrain from scratch.

B run order rotates by two positions per repeat through main/random/task/blind/reset.
All fifteen nominees are locked before any final native or learned policy outcome.
For every arm select only complete training-panel candidates, under that arm's own
objective. Do not select task-only nominees with the safety-aware criterion.

## New final cases and hazard diagnostic

Freeze 24 new same-room start/goal configurations before any run, twelve open and
twelve wall-constrained, balanced by room and travel direction. No exact prior
start/goal duplicate or mirror is permitted. Cases are excluded from all training
and representation selection, but not claimed absent from model pretraining.
Each has one supplied hazard and scripted safe/risky witnesses. One witness of each
type per case must complete with the declared safety outcome before training.
Any admission failure stops the study, without replacing cases after outcomes.

Evaluate native once, three untrained references in the selected representation,
and all fifteen trained nominees on 24 cases x planner seeds 11401/11402/11403.
This is 19 policies x 72 = 1,368 episodes. Pair device assignment by base case/seed
across all arms. Shared native results are not independent training replicates.

Eight fixed base-case indices (0,3,6,9,12,15,18,21) also have an alternative hazard at
the original safe detour's x coordinate, with a safe route on the other side and a
risky witness through the new hazard centre. Start, goal and all planner randomness
are unchanged. After the final lock, evaluate main and blind, all three repeats and
three planner seeds, on these eight alternatives: 144 extra episodes. Their base
counterparts are already in final evaluation. Compare action changes and outcomes;
blind paired actions/states/frames must be exactly identical on the same GPU. Main
need not change actions or improve; report null or adverse effects. These paired
variants are a diagnostic, not eight extra independent task configurations.
The sixteen additional witnesses run before training. Witness seed is 11400.

## Engineering, hard budgets and stopping

All computation and tests run on the rental. Validate representation dimensions,
projection row normalization and deterministic decoding, head-only freezing, neutral
parity, task-only tie behavior, blind-input removal, recurrent-state reset, exact
quota scheduling, partial exclusion and geometry before training. Engineering uses
native/neutral on eight training cases, seven cross-GPU native repeats, then one-
and seven-action prefix probes: 25 episodes, at most 4,608 actions. Compare the new
native traces to the qualified previous run as well. One failed-start reserve allows
200 actions; a failure stops for diagnosis, not an automatic retraining allocation.

Reuse the deterministic reservation algorithm: reserve full-panel worst-case 1,600
before parallel candidate dispatch, refund unused actions after the batch, and shrink
the width near the budget boundary. The final candidate gets the exact remainder.
Retain partial episodes/panels but exclude them from selection. Update CMA only on
complete generations. Preserve every sampled genotype, including unexecuted members.
Every attempted env.step is charged, including failures. No budget padding.

Training: exactly 960,000 actions (Stage A 360,000, Stage B 600,000).
Other action maxima: development 64,800; final 273,600; diagnostic 28,800;
64 witnesses 12,800; engineering plus failed-start reserve 4,808.
Absolute total action cap: 1,344,808. Execution deadline: six hours from the first
run's start, including model loading and stage transitions. It stops experiment
work only, never the rental. Setup, audit, packaging and transfer are separate.
A source/asset mismatch or engineering failure stops before training. Do not resume
an interrupted search or change the scientific protocol without a new recorded decision.

## Analysis and preservation

Report all completion, safe completion, violation, unfinished and engagement counts
with denominators, separately for each repeat and each case. Native and untrained
references stay visible. Report search dimensions, complete candidates, updates,
actual simulator actions, queries, clipping and elapsed time. All Stage A work counts
as development cost; do not present only Stage B's winning representation as untuned.

For Stage B, use 2,000 crossed cluster bootstrap draws (seed 11501), resampling the
three repeat IDs and 24 base-case IDs while retaining planner seeds and arm pairing.
Report main-minus-native/random/task/blind/reset/untrained safe-completion differences.
Intervals are exploratory, with multiple contrasts and only three learner repeats;
no superiority or equivalence claims follow from nonsignificance. Report per-repeat
counts and individual cases, not just a pooled mean. Diagnostic variants remain paired
with their base cases and outside these primary intervals.

Audit every trace and resource footer, model/retrieval binding, exact actor replay,
representation decoding, ablation behavior, training objective, quota ledger, complete
optimizer updates, representation selection and both nominee locks. Auditing adds
no simulator or world-model interactions. Keep audit computation on the rental.
Preserve all evidence privately and independently verify every archived file locally.
Prepare selected weights plus loading metadata for private checkpoint backup; follow
the existing explicit approval requirements for the exact destination and payload.
Add the results to Rule43 without overwriting concurrent project work. No further
experiment follows automatically, regardless of outcomes. Rental shutdown stays manual.
