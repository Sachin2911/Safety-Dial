# SafetyDial: when are JEPA predictions reliable enough for safe planning?

**Status: adopted research direction, 20 September 2026.** Sachin Mohan agreed to this
direction after the Phase 0 locomotion experiments, the Push-T Safe-CEM pilot, and the
subsequent literature review. This document is the authority for the project's research
question, scope and next experiments. It supersedes the research plan in
[`revisedProp/latex/main.tex`](revisedProp/latex/main.tex) and its built PDF. Earlier
proposals and experiment reports remain records of what was proposed or measured. This
decision does not imply supervisor approval or that the proposed experiments have run.

Student: Sachin Mohan (2699183), BSc Honours Computer Science, University of the
Witwatersrand. Supervisor: Geraud Nangue Tasse. Project name: **SafetyDial**. Working
title: **When Are JEPA Predictions Reliable Enough for Safe Planning?**

**1. The central question is whether planning selects optimistic safety errors.**
Does planner optimisation amplify optimistic constraint-prediction errors in JEPA-style
world models, and can a simple correction reduce those errors while preserving useful
task performance?

The project accepts Safety-Gymnasium's supplied safety costs. It studies whether a
predictive latent model can support reliable decisions about those costs, particularly
when CEM selects actions using the model's own predictions. The contribution sought is
a controlled diagnosis of safety decision errors, followed by one measured repair.
The method sequence is **offline audit, simple repair, then optional closed-loop
intervention**. A useful empirical audit is the minimum thesis contribution; a new
controller or formal safety theorem is not a prerequisite.

**2. The pivot removes an unresolved safety-definition problem.** The previous plan
required genuinely irreversible failures, a defensible recovery evaluator, an estimator
of irreversibility in learned latents, and a deployment filter. Phase 0 showed that the
tested Walker2d falls could be recovered, while Hopper's apparent boundary depended
strongly on recovery predicates and search horizon. Those findings invalidate using
termination as an irreversibility label; they do not invalidate Safety-Gymnasium as a
benchmark for its stated constraints. See the
[`Phase 0 report`](../notes/phase0Report.md).

The new plan makes three decisions explicit:

- A benchmark cost violation is the safety event, even if recovery is easy. Falling and
  task abandonment remain additional outcomes to report, not definitions of irreversibility.
- Cost and physical-state labels may be collected automatically from simulation. This
  is **supervised constraint prediction**, not label-free safety discovery. A velocity
  probe supervises a generic physical quantity; applying a chosen velocity threshold
  supplies the safety specification. A cost head uses direct safety supervision.
- The Safety Dial controls an empirical conservatism margin on predicted constraints.
  It is no longer a reachability threshold, an irreversibility score, or a Pareto front.

OGBench-Cube failure engineering, Sokoban reachability, precedence estimation, failure-mode
label holdout, HJ reachability, and a new conformal guarantee are outside the core scope.
They may be revisited as separate future work. They are not dependencies of this thesis.

**3. The existing results motivate the new question without proving it.** In the Push-T
pilot, the probe-based arena constraint judged many candidates feasible while the pusher
left the observation domain. Computing the arena check from commanded actions repaired
the observed failure. This motivates testing the reliability of imagined constraint
decisions; it does not yet show the same effect in locomotion.

The pilot recorded zero true-box violations in nine episodes across usable dials, not a
general safety guarantee. The original penalty-fragility result was confounded by an
infeasible starting configuration. With corrected geometry, penalty-CEM performed
substantially better. Use the corrected account in [`safeCEM.md`](../notes/safeCEM.md),
including the distinction between the original probe-based sweep and the action-space
arena repair. Neither all penalty methods failing nor monotone realised safety is an
established motivation.

**4. Three hypotheses organise one project.** They are prospective empirical questions,
not conclusions to engineer into the evaluation.

- **H1, imagined constraint fidelity:** accurate readout from encoded real observations
  and low average latent prediction error do not necessarily imply accurate future
  constraint decisions. Measure the additional error introduced by imagination at
  increasing horizons and near the constraint boundary. If the gap is negligible,
  report that rather than manufacturing a harder task.
- **H2, selection amplification:** CEM-selected plans exhibit more optimistic safety
  error than ordinary candidate plans, conditional on comparable initial states,
  horizons, predicted margins and candidate budgets. Sweep search effort to determine
  whether this amplification grows, stays flat or reverses.
- **H3, useful correction:** a simple correction reduces false-safe decisions at
  comparable acceptance and task progress. Demonstrating fewer violations solely by
  rejecting everything, standing still, or ending episodes early does not support H3.

H1 and H2 form the minimum audit. H3 is the first extension. Closed-loop improvement
is a stronger extension of H3, rather than an additional mandatory thesis project.

**5. Start with one existing locomotion environment and a competent nominal policy.**
Walker2d is the initial development candidate because the vendored environment,
collection and rendering infrastructure already exist. Confirm a usable trained actor
and data coverage before fixing the robot. Hopper is an alternative within the existing
stack if it has better usable data. A second robot is replication after the first study
works, with predictor retraining where needed; it is not zero-shot cross-robot transfer.

Use the pinned implementation's cost channel. In the currently documented Walker2d-v1
implementation, the cost compares signed forward velocity with 2.3415 m/s. Do not
silently substitute absolute speed because of a documentation summary. Pin robot,
version, simulator, controller timestep, action bounds, rendering, camera, observation
history and termination settings. The official
[velocity-task documentation](https://safety-gymnasium.readthedocs.io/en/latest/environments/safe_velocity.html)
and [Walker2d source](https://github.com/PKU-Alignment/safety-gymnasium/blob/main/safety_gymnasium/tasks/safe_velocity/safety_walker2d_velocity_v0.py)
describe the task; the exact tested code determines the experiment.

Obtain or train one competent nominal policy, then collect its behaviour and controlled
action perturbations. The presence of an actor adapter is not evidence that a compatible
trained actor is available. Record policy provenance and its observations. A policy using
privileged state may be shared across filter variants, but the overall system must then
not be described as vision-only. Do not ask CEM to learn walking from scratch.

**6. Gate the observation and model before scaling experiments.** The released
Push-T/Cube checkpoints are not validated locomotion predictors. The
[official LeWM release](https://github.com/lucas-maes/le-wm) reviewed for this decision did
not supply a verified drop-in Safety-Gymnasium locomotion model. Predictor training is a
real dependency, and its cost must be measured rather than inferred from another task.

The first substrate is a frozen visual encoder with cached features, a small
history-conditioned action predictor and a velocity readout. This is a frozen-feature
predictive-embedding model. It must be named accurately; success on it alone does not
establish a result about end-to-end LeWM or SIGReg. A matched LeWM-style model can be a
later comparison if resources permit.

First test whether velocity is observable from the chosen image/action history. A camera
that tracks the robot may conceal translation. If history is insufficient, explicitly
declare a proprioceptive input or change the observation setup. Supplying the quantity
being predicted directly is a different information regime and must be identified.
Keep observations consistent across comparisons. Ground-truth future states must never
enter the model-based planning path; a simulator-informed reference is a separately
labelled baseline.

Train models and readouts on the training split, choose their settings on development
data, and freeze them for the audit. A margin sweep then changes deployment behaviour
without retraining that model. A predictor trained with an auxiliary loss is a distinct
model variant, not another point on the same frozen-model dial.

**7. Collect transition-aligned data and prevent leakage.** Reuse state-only storage and
lazy rendering. Record observations or reproducible render states, actions, episode and
step identifiers, velocity, cost, reward, health, termination, truncation and policy
provenance. Record complete simulator restoration state and verify replay; saved state
requirements are environment-specific.

The current collector stores state before each action. The returned reward and cost
describe that action's transition. Preserve both endpoints and make the indexing
explicit: an input at time t and action a_t predicts the returned transition cost and
next observation, not a label attached to the preceding frame by accident. Confirm this
with hand-checked short traces before fitting a model.

Split by complete trajectories before sampling roots or clips. All candidate branches
from one root belong to the same split. Keep training, model-development,
margin-calibration and final-test roles distinct. If data are limited, partition the
validation trajectories between model selection and calibration, and leave the test
set untouched. Report the number of independent episodes and roots as well as branches.

Include both safe and violating transitions and a range of speeds and action perturbations.
Keep a near-boundary diagnostic bank separate from a representative-policy evaluation
bank, and report their different sampling distributions. A balanced diagnostic sample
does not estimate deployment prevalence.

Termination-disabled collection is no longer a universal requirement. Use it only for an
explicitly labelled diagnostic continuation. For ordinary benchmark evaluation retain
the declared benchmark termination and time limit. Do not silently count missing
post-termination steps as safe. Report early falls and censoring, and compare horizons
consistently; diagnostic physics beyond termination must not be reported as standard
benchmark return.

**8. Separate encoding, prediction and action selection with paired simulator branches.**
For each held-out root, construct a supported population of action sequences, initially
around the nominal policy. Save the executed action sequence, initial conditions, random
seed, predicted trajectory, predicted margin and selection decision. Run each audited
sequence from the same restored simulator root.

Compare three paths on the same actions:

- The actual transition velocity and cost from simulation, used as evaluation truth.
- The velocity readout on encoded actual observation histories, isolating observation
  and readout error.
- The same readout on imagined future latents, exposing the additional predictor error.

The low-dimensional state-dynamics baseline is a further reference, not a substitute for
the paired encoded-versus-imagined comparison. Reproduce restoration of contacts, solver
state and randomisation before interpreting small decision differences.

Report errors by horizon in physical time and by action-block settings. Do not copy
Push-T's frameskip into a newly trained locomotion model without justification. Inspect
both continuous velocity residuals and the resulting threshold decisions; low aggregate
RMSE can hide a harmful one-sided error tail.

**9. Distinguish candidate adaptation from final selection.** Audit three action groups:
samples from the initial fixed proposal, samples from CEM's adapted final proposal, and
the plan actually selected for execution. Where possible, pair the selected plan with a
uniform sample from that same final candidate pool. This separates distribution changes
during search from the final ranking decision. Compare roots and horizons, report the
actual model-evaluation budget including CEM iterations, and stratify initial velocity
and predicted clearance.

Increase candidate counts or iterations while holding the root bank, predictor and
constraint fixed. A naive selected-versus-random aggregate is insufficient: selected
plans can be closer to the boundary or make more progress. Report the raw operational
difference and the conditional comparison. Keep matching rules fixed before test
evaluation and report where the groups lack common support.

Audit the actual returned sequence, including any averaging, clipping or action-block
conversion by the solver. Feasibility of an elite candidate does not establish feasibility
of an averaged returned action. Track all-infeasible searches explicitly.

**10. Define the main metric and its denominator.** For a declared horizon H, let
U = 1 when any actual transition violates the benchmark constraint. Let A = 1 when
the complete predicted plan satisfies the chosen feasibility test. The primary
false-safe acceptance rate is

`FSA = count(A = 1 and U = 1) / count(A = 1)`.

Always report accepted count, total evaluated count and acceptance rate alongside FSA.
If nothing is accepted, FSA is undefined, not zero. Report actual violation rate among
selected plans whether or not the search found a feasible candidate; a deployed fallback
must not disappear from the denominator. Distinguish FSA from the false-negative rate
among all truly unsafe plans.

Supporting offline metrics include signed velocity error, upper residual quantiles,
constraint confusion counts, unsafe-event recall, and AUROC or precision-recall curves
where meaningful. The headline is the operating decision, not AUROC alone. Use paired
comparisons and uncertainty estimates clustered by episode or root; thousands of related
branches are not thousands of independent trials. Repeat model training when making
claims across training randomness, and distinguish model seeds from rollout seeds.

Closed-loop metrics include benchmark return and cost, distance travelled, violation
fraction and severity, falls or unhealthy events, episode length, intervention frequency,
infeasible searches and planning latency. Report total cost with exposure so early falls
do not appear safer merely because the episode was shorter.

**11. Test one modest correction after the audit identifies the error.** The first
candidate is a horizon-dependent additive velocity margin m_h, fitted on held-out
prediction residuals from the intended action-selection procedure. For an upper signed
velocity limit v_max, the plan satisfies the corrected constraint when

`predicted_velocity_h + d * m_h <= v_max` for every predicted transition h.

Here d is a nonnegative, dimensionless deployment conservatism multiplier, and m_h has
velocity units. This is an initial operational definition of the Safety Dial. Record the
quantile rule, treatment of negative margins, horizon grouping and calibration sample
counts. Compare d = 0, a fixed additive margin and the horizon-dependent correction
under the same model and candidate budget. Select the tested dial grid on development
data; report the entire predeclared test sweep rather than choosing its best point.

Measure error separately at each horizon; do not assume it increases monotonically. If
a nondecreasing envelope is imposed for conservatism, state it as a design choice. A
corrected planner changes which actions are selected, so calibration on the uncorrected
planner may not cover its new distribution. Use a declared calibration procedure and
measure the corrected selection distribution on untouched test roots. Per-step residual
quantiles do not automatically cover an entire plan or a closed-loop episode.

If the diagnosis instead implicates poor propagation of velocity, the alternative repair
is an auxiliary future-velocity loss during predictor training. Hold training data,
encoder, readout supervision and evaluation budget constant across variants. With a
frozen encoder this changes the predictor, not the visual representation. Do not add both
repairs by default or present a known auxiliary loss as an invention.

**12. Baselines must be capable of disproving the need for imagination.** Include
current-speed persistence, a declared reactive or one-step guard, a small state-dynamics
predictor, the same latent predictor without correction, and a fixed-margin variant.
The reactive controller must be concrete: clipping actuator torque is not the same as
clipping velocity. Validate its action rule and task progress. If it uses privileged
velocity or state, label that information advantage and include an observation-matched
variant where feasible.

For optional control experiments, compare the nominal policy, reactive guard,
uncorrected Safe-CEM, corrected Safe-CEM and a corrected penalty-CEM arm using the same
constraint, observations, horizon and model-evaluation budget. Compare at matched
acceptance and useful progress where the methods have overlap, and present full curves.
Penalty-CEM can change lambda at inference; do not attribute a new training run to each
lambda. Published state-based PPO/CPO results can contextualise the task but are not
automatically fair direct comparisons with a pixel-based filter.

**13. Closed-loop intervention is conditional on useful offline predictions.** Warm-start
short-horizon candidates around a competent nominal policy and use feasibility-first
ranking with task progress or a declared deviation objective. Fix how the nominal action
sequence is obtained: a state-policy action tape produced by true-simulator branching is
a privileged diagnostic, not a deployable visual rollout mechanism. An implementable
filter may use a validated policy over its available predicted features or a declared
open-loop proposal around the current nominal action and previous plan. Report which
mechanism is used.

Specify and evaluate the all-infeasible fallback before running episodes. Least predicted
violation, a backup controller, and the nominal action are different policies; none is
safe by definition. Freeze model weights during the dial sweep, use paired initial
conditions, and report actual violations and progress. A tighter predicted feasible set
does not guarantee monotonically improving realised safety with approximate search and
model error. Safe-CEM is not automatically an invariant HJ safety filter.

**14. Prior work determines the scope of the claim.** SafeDreamer already studies safe
world-model planning, robust CEM already combines planning and constraints, and AnySafe
already offers adjustable latent constraints with DINO-WM. UNISafe addresses optimistic
safety estimates with uncertainty. A cost head, frozen encoder, constraint-priority
ranking, or runtime threshold alone is therefore insufficient novelty.

PSG-JEPA studies physical grounding, and the particularly close Intervention Gap work
separates current-state decodability from imagined action effects. The prospective
contribution here is a safety-decision audit of planner selection, with a reproducible
measure of false-safe amplification and a correction evaluated at useful performance.
This is a candidate contribution; check the closest papers again before writing a
novelty claim. A rigorous negative result can establish when the predictive model offers
no advantage over simpler controls.

Primary sources reviewed for the direction:

- [Robust CEM](https://arxiv.org/abs/2010.07968): existing constraint-aware planning.
- [SafeDreamer](https://arxiv.org/abs/2307.07176): safe planning with learned world models.
- [UNISafe](https://arxiv.org/abs/2505.00779): uncertainty-aware latent safety filtering.
- [AnySafe](https://arxiv.org/abs/2509.19555): adjustable latent constraints and conservatism.
- [PSG-JEPA](https://arxiv.org/abs/2608.06799): physical grounding of predictive latents.
- [The Intervention Gap in Latent World Models](https://arxiv.org/abs/2608.29998):
  decodability versus imagined action effects.

**15. The first pilot should end with a decision, not a larger infrastructure project.**
These are proposed work allocations, not promises about hardware runtime.

- **Days 1-2:** fix one robot, verify a competent actor and restoration, collect a small
  dataset spanning the cost boundary, confirm label alignment and speed observability,
  and create trajectory-disjoint splits.
- **Days 3-4:** cache features, fit a small predictor and readout, establish persistence
  and state-model references, and evaluate actual versus imagined constraint decisions.
- **Days 5-6:** audit initial, adapted and selected candidates on paired roots. Test one
  correction only if the observed error supports it. Closed-loop runs are optional at
  this point, not necessary to call the pilot complete.
- **Day 7:** record the measured bottleneck and a go/no-go decision. Continue if future
  prediction adds useful information or a robust error is diagnosed that merits a
  bounded repair. Stop expanding if observations conceal the constraint, model support
  is inadequate, or a simple guard matches predictive control. Do not invent an
  artificial constraint merely to force a world-model advantage.

The minimum thesis is a reproducible audit with strong controls and honest positive or
negative conclusions. A successful repair is the next deliverable; closed-loop benefit
and a second robot are stronger extensions. Aim to complete the core evidence in
October, write the thesis in November, and use December and early January for bounded
replication and a possible workshop paper. The existing target remains an ICLR 2027
workshop, subject to a suitable actual call; no particular workshop or acceptance is
assumed. Freeze results by mid-January if that target is pursued.

**16. Reuse the engineering work and keep the new implementation small.** Existing
modules provide the following starting points, not a claim that the pivot is implemented:

- [`locoEnv.py`](../experiments/helpers/locoEnv.py): environment registration and manifests.
- [`locoCollect.py`](../experiments/helpers/locoCollect.py) and
  [`locoData.py`](../experiments/helpers/locoData.py): state storage and lazy rendering.
- [`locoPolicies.py`](../experiments/helpers/locoPolicies.py): actor interfaces.
- [`equivCheck.py`](../experiments/helpers/equivCheck.py): environment verification support.
- [`probes.py`](../experiments/helpers/probes.py): existing probe patterns; the current
  pusher-specific probe is not a locomotion velocity probe.
- [`safeCEM.py`](../experiments/helpers/safeCEM.py): constraint-priority planning patterns;
  Push-T geometry, action semantics and margin units must be adapted explicitly.

New experiment logic belongs in `experiments/helpers/` and small entry points in
`experiments/scripts/`, with configs in `configs/<group>/`. Continue using `uv run` and
the existing project environment. Record simulator/model/data versions, policy and model
seeds, split identifiers, training time, candidate budgets and observation access. Do not
commit datasets, weights, credentials or raw run directories. The next implementation
task is the competent-policy, data and observability gate, not a new safety algorithm.

**17. Open choices have explicit decision points.** The exact actor, encoder, history
length, predictor architecture, useful physical horizon, calibration quantile and dial
grid remain to be chosen on development evidence. No new experiment in this document
has been run. The adopted commitment is the question and staged scope, not a promised
effect size, monotone curve, calibrated safety guarantee or publication outcome.
