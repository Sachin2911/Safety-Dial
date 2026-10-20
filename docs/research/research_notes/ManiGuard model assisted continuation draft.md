# Model-assisted safe continuation: implementation and deferred study

Prepared on 20 October 2026 following the user's request to build this follow-up
while the original comparison runs. This document does not amend that comparison.
The CPU prototype is implemented and tested. Physical integration, development
settings, a complete comparison and a new allocation remain to be frozen.

The immediate objective is a controllable sequence: close the jar's lid, grasp the
jar, then lift and transport it while preserving the prerequisite. Robot finger
closure alone is not jar lid closure. A world model or larger failure penalty
cannot supply behaviour absent from the controller's usable action alternatives.

## Implemented prototype

The implementation uses NumPy in the prepared tools environment. It imports no
simulator, neural encoder, policy client or GPU library. New artifacts remain in
the ignored reviewer run directory. Existing source, checkpoints, recorded data,
services and the active physical queue are unchanged.

The components are:

- [Recorded transition reader](../../../experiments/helpers/maniguard_model_data.py):
  freezes exact files and whole-task partitions before fitting. Independent
  checking stages are excluded before their output directories are opened.
- [Transition ensemble](../../../experiments/helpers/maniguard_model_transition.py):
  five episode-bootstrap regressors predict the next 44 compact features from
  current features, previous features, a history flag and eight known executed
  commands. A separate head predicts a new prefix violation; another learns task
  predicates from actual features. Fixed random nonlinear features and ridge
  readouts provide a small CPU baseline, not a newly validated visual world model.
- [Control components](../../../experiments/helpers/maniguard_model_control.py):
  a persistent five-coefficient noise controller, ordered training milestones,
  ROSARL's practical penalty interface, an action-changing filter and model-assisted
  candidate allocation. One assembled interface connects prediction, filtering,
  the frozen penalty and allocation at a factual observation.
- [Global work ledger](../../../experiments/helpers/maniguard_model_budget.py):
  locks admission, records it before work, flushes/fsyncs it, refuses duplicate
  request IDs and retains failed or interrupted work as charged. A new output
  directory cannot reset the canonical ledger's allowance.

Two later CPU components retain the original audited prototype unchanged.
[Filtered allocation](../../../experiments/helpers/maniguard_continuation_allocation.py)
ranks the prefix selected by the filter, using the same declared alternatives for
every candidate. A synthetic example reverses raw-proposal rankings after
filtering. Identical prefixes at one factual anchor share a model prediction,
while controller identities retain separate screening and physical-evaluation
requirements. An unqualified model makes no prediction and preserves abstention;
identifier allocation cannot authorize action execution.
[Bounded residuals](../../../experiments/helpers/maniguard_continuation_residual.py)
require runtime joint bounds and native absolute joint targets with binary gripper
commands. Zero weights preserve those commands, neutral gripper correction retains
the native command, and persistent memory is bound to the controller's checkpoint.
The 12 additional synthetic checks pass, giving 28 combined preparation checks.
These additions have no physical entry point, factual refit or deployment evidence.

The residual's supplied joint bounds must not be confused with outer
environment-input limits. The recorded native commands precede downstream
controller clipping and can exceed a position joint's physical range. The helper
rejects inputs outside its supplied bounds, so it is not yet an adapter for all
recorded native commands. Before physical use, freeze the source-correct native
clipping map and actual runtime bounds, verify zero-residual physical identity,
and retain original environment inputs, canonical targets, residual proposals,
filtered commands and execution records separately. The
[source-matched CPU wrapper](../../../experiments/helpers/maniguard_continuation_targets.py)
now implements the declared raw-position clipping map, with 15 additional passing
CPU checks against synthetic runtime snapshots and an AST-extracted native preset.
It preserves every original command coordinate whose effective target is unchanged,
including exact zero-residual identity. Its target contract and runtime-evidence
hash bind memory replay. It does not discover a runtime or validate a declared
snapshot. Actual runtime capture and physical identity remain unvalidated, and the
original study's clipping stage stays native.

The residual bound applies to the effective desired target. Undoing a saturated
native command can change the raw environment input by more than 0.05 radians;
both differences are retained and must be reported. A later comparison must freeze
this output domain and give every direct-residual method the same adapter.

The prospective noise controller receives the 44 causal actor features plus five
learned task-state beliefs. Its 5-by-49 weights produce four arm coefficients and
one explicit gripper-noise coefficient. Defaults are arm scale 0.35, gripper scale
2.0 and coefficient persistence 0.8. These are prototype settings, not approved
physical magnitudes. Native Gaussian noise remains necessary; zero weights produce
zero offset, not a zero-noise native policy. Gripper-noise changes can also affect
decoded arm commands through model coupling. Learned beliefs must pass perception
checks before use and are never replaced silently with simulator predicates.

The progress evaluator observes ordered factual milestones for training: lid closed
on support, held while closed on support, a subsequent closed/held lift, and closed
transport into the goal. It supplies secondary progress, not a completion verdict.
It does not reward grasping an open jar or closing the lid after lifting. Raw
violations and requested safe completion remain separate primary outcomes. Any
new progress signal must be shared by all adaptation methods in the next cohort.

The filter accepts a supported proposal, replaces it with a supported alternative
when appropriate, or abstains when the gate or alternatives are missing. It keeps
proposed, filtered and executed commands separate. Execution remains unobserved
until explicitly recorded from the environment-input history; a terminal native
episode can execute only part of a selected prefix. Abstention supplies no evidence
of successful safe task completion.

The ROSARL interface requires declared external value estimates and freezes one
shared penalty before candidate comparison. It implements the practical extrema
estimator; it is not a trained value function or a complete ordinary-learning
baseline. Imagined progress and risk are not passed off as factual value estimates.
The original analysis concerns stochastic shortest-path MDPs and unsafe absorbing
outcomes. It does not certify finite-budget, partially observed manipulation or
safe exploration. [ROSARL primary paper](https://arxiv.org/html/2306.00035v1)

The model's rollout scope is one known eight-command prefix at a real observation.
It does not generate future camera observations. Consequently it cannot simulate
an entire closed-loop frozen VLA episode from compact features alone. Candidate
priorities remain surrogates; measured full-episode fitness is still required.
Every future counterfactual decoder request remains charged, including discarded
proposals. [SafeDreamer](https://arxiv.org/html/2307.07176v3) already combines
world-model planning and safety objectives; the proposed combination is not an
originality claim by itself.

## Completed checks and factual audit

All 16 CPU tests pass. They check global and concurrent admission, interrupted
requests, forbidden decoder access, fitted action effects, checkpoint round trips,
episode-level calibration counts, separation of readout and dynamics errors,
persistence/reset, ordered progress, frozen ROSARL scoring, action replacement and
abstention, factual execution records and combined candidate allocation. Synthetic
tests are software evidence, not manipulation results.

The recorded-data audit was frozen under
[offline-v1.json](../../../configs/maniguard/model-assisted/offline-v1.json) before
fitting. It uses eight shared-initialization episodes on tasks 0019/0009 for
fitting, task 0017's development episode for calibration and task 0013's development
episode for diagnosis. All four tasks were already exposed to the research
programme. No independent checking outcomes were read. Final query prefixes with
no following encoded observation are excluded. Risk fitting excludes prefixes
that already follow a latched violation; their physical transitions can still
train dynamics. The fitting/calibration/diagnostic counts are 1,992/249/83.

The model **fails its prediction and coverage gate**. Diagnostic RMSE relative to
persistence is 1.97 for robot/history, 1.54 for overview and 1.16 for wrist features.
Actual-state readouts also fail for lid closure, holding, uprightness and support.
Thus part of the problem exists before imagined dynamics are introduced. The
fitting corpus has positive lid-closure and holding labels in only one episode
each and no in-goal positive episode; its closure occurs on an already violating
trajectory. No safe completion supports this fitting/diagnostic corpus.

There is only one calibration episode, below the predeclared minimum of 20. The
conservative score therefore rejects all prefixes. The resulting 100% negative-
prefix alarm rate is the fallback's behaviour, not a separately established
false-alarm rate for a calibrated classifier. Even adequate factual calibration
would not prove exchangeability or coverage for new counterfactual actions.

The exact audit is
`runs/reviewer/model-assisted-offline-20261020-01/audit-v1/report.json`, SHA-256
`c74f001d302ea148326a438ee67a7c1ea7ec90703a2bc32b3014d4d9730faffd`.
Its artifact hash is
`91f99392c34f0bffb3d46e9c65fa78b133f3feb6a57f89c05db4d19dc339c16a`.
The private corpus manifest records exact source/data hashes. Work is 27,412
training-row uses and 5,395 model-head row queries, with no unresolved admissions.
There are zero new simulator controls, physical episodes, VLA calls or encoder
images. The 18,666 original acquisition controls are identified as reused data,
not free information. Training/audit measured about 0.20 seconds of process wall
time after corpus reading; this is not a physical execution or rental estimate.
Its six exact source files are preserved under the same run's `source-v1/`.
A subsequent 100-column formatting cleanup has identical Python ASTs; it does not
change the recorded audit, model or corpus. Replaying the old manifest requires
its pinned snapshot. Current sources can freeze a new version for future work.

## Later ordered-progress diagnostic

The [completed-data progress audit](../../../configs/maniguard/latent-evolution/review-ordered-progress-summary-20261020.json)
uses 24 Jar episodes: two development and 22 comparison episodes. Among the ten
safe unfinished comparison episodes, nine have ordered stage zero and one has
supported lid closure. None reaches a safe ordered body grasp, lift or transport.
Nine comparison episodes command gripper closure, but none commands it after a
safe supported lid closure. Native initialization instead records holding before
the lid prerequisite; the closure controller never commands a grasp. These are
complementary factual partial behaviours, with no demonstrated composability.
The frozen R12 fit and failed qualification remain unchanged.

All four completed random generation-one panels have utility `[0, -2]`. The
existing selection correctly follows its frozen score and identifier tie rule.
Any new progress-based tie handling or archive needs a separate common training
signal, shared warm-start information and matched comparisons with ordinary
learning. It cannot be introduced into the current cohort after seeing this lead.

A prospective privileged engineering control could switch from the preserved
closure controller to native execution after a declared supported-closure trigger,
at a native query boundary and without resetting the sampling RNG. It needs
explicit debounce/latch/reset semantics, a new validity reader, matched native
and closure-only controls, fresh seeds and a separately frozen finite budget.
The monitor-triggered switch supplies privileged actor information and would be
a feasibility/data control. It is not a nonprivileged learned method or a safety
filter. Its failure would not establish physical irreversibility. No such
controller has been physically integrated or launched here.

Progress retention and exploration already have close precedents in
[MAP-Elites](https://arxiv.org/abs/1504.04909) and
[Go-Explore](https://arxiv.org/abs/2004.12919). A model-assisted robotics precedent
uses learned dynamics and predicted-behaviour screening with recovery on a
physical quadruped, in a GECCO 2023 Companion contribution.
[Dynamics-aware reset-free QD](https://arxiv.org/abs/2304.12080)
These primary abstract checks guide later comparisons; they do not establish a
new mechanism or equivalence to the prospective per-action filter. Full-method
review and measured component ablations remain necessary.

## Deferred physical development

The [dependency schedule](../../../configs/maniguard/model-assisted/schedule-v1.json)
records the next proposed batch. There is no booked start or automatic dispatch.
The current study retains its 19:30 UTC rollout cutoff and 20:00 UTC rental stop.

First perform a bounded decoder-authority audit, proposed ceiling 64 requests, on
development observations. Then screen four declared conditions on two development
tasks, one seed each, for eight full episodes: native, scale/persistence settings,
arm-and-gripper steering and a bounded action residual. Exact parameters and all
implementations must be fixed before comparative feedback. A residual needs true
runtime joint bounds and an explicit gripper convention; it is a separate control
representation, not an unrecorded modification of the frozen noise method.

Reserve eight more full episodes for native and the development-selected candidate
on two tasks with two fresh seeds each. The proposed ceiling is 16 valid full
episodes plus two invalid-attempt replacements. Freeze the selection rule and
ordered progress criteria before screening. Preserve unsuccessful trials and do
not treat a valid timeout as infrastructure failure. Initialization/settling,
decoder/encoder/model work and elapsed time need explicit ceilings in the new
physical protocol. Estimate paired-wave time using at least 1.25 times the latest
slowest measured full episode plus setup and preservation. The current rental and
the protected checking allowance do not fund this batch.

These first tests answer whether the required behaviour can be produced and
repeated. Inspect a credible native/reference execution or demonstration when
available; no released positive reference is assumed to be installed. If the
screen cannot produce lid closure and a valid grasp, diagnose control authority,
perception and task-policy competence before allocating a larger optimizer run.
Collect positive, negative and transition-boundary data with full provenance.
Rebuild perception and dynamics under a new prospective audit rather than tuning
the failed gate's thresholds against these diagnostic outcomes.

After capability and prediction gates pass, freeze the scientific comparison:
model-free and model-assisted EA, matched random search/allocation, competent
ordinary task-policy learning with the same fixed filter and observations, and
native, native-with-filter and strong fixed references. Give competitors matched model/data access and
count all acquisition, training, discarded candidates and queries. Model-assisted
random selection and fixed-parent local proposals distinguish allocation benefit
from evolutionary retention and its local proposal prior. Filter and ROSARL
benefits require their own ablations if claimed. Replicated optimizer runs,
independent confirmation and the same-controller family-transfer test remain
separately budgeted requirements for the GECCO contribution.

## Private preservation

The preparation bundle is privately preserved in
`Sachioster/maniguard-latent-evolution-controllers` at immutable revision
`96ef6ab0de0d44d653683c5f40ad20fdb883294d`, prefix
`20261020/model-assisted-preparation-v2/`. Its source commit is
`e2538e692ab1dfd30de9a314850b7cb3b8b9e847`. This separate preparation prefix is
outside the native-noise comparison's artifact and scientific-work ledgers.

The [preservation receipt](../../../configs/maniguard/model-assisted/model-preparation-preservation-receipt-v2.json)
pins the exact bundle bytes, manifest and private revision. The
[remote reconstruction receipt](../../../configs/maniguard/model-assisted/remote-recovery-receipt-v1.json)
verifies fresh download and extraction of all 21 members, the six frozen source
hashes and their AST agreement with the recorded Git commit. All ten referenced
physical input episodes were hash-checked against earlier private remote
reconstructions. Model loading and ledger verification added no fitting,
prediction, policy request, encoder call or physical interaction.

The first immutable upload manifest contains an incorrect manually expanded Git
hash. Its record is retained, and the
[provenance correction](../../../configs/maniguard/model-assisted/preparation-preservation-correction-v1.json)
identifies the corrected v2 manifest above. Bundle bytes are identical. Use v2
for source provenance. The upload helper now rejects abbreviated, unknown and
noncommit Git references before any remote access, with four passing CPU checks.

## Reproduction

Use the explicit prepared tools interpreter. The tests require no research
packages to be installed or the project lockfile to be synchronized.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=experiments \
  data/maniguard/envs/tools/bin/python -m pytest -q experiments/tests/test_maniguard_model_followup.py

data/maniguard/envs/tools/bin/python experiments/scripts/maniguard_model_followup.py check-schedule \
  --schedule configs/maniguard/model-assisted/schedule-v1.json
```

The completed audit's canonical ledger refuses replaying its request IDs. A fresh
audit needs a new output and explicit new attempt ID within that same allowance;
source edits require a new frozen corpus version. Preserve existing evidence.
