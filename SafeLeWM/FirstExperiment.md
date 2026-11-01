# First experiment: evolutionary adapters around frozen LeWM

Working design, revised 1 November 2026 following the cross-environment review.
This is the detailed companion to Rule42, not a frozen execution protocol. It
does not authorise a rental, experiment or historical-job restart. Earlier LeWM
work informs engineering choices; its correctness and ablations are not assumed
to establish the claims of this new study.

All first-experiment work now lives under SafeLeWM, as explicitly requested on
1 November. New code, configurations, tests, documents and ignored local outputs
belong here. Existing historical implementations remain in the parent repository;
their presence does not make them the new experiment's implementation.

## Shared question and paper role

> Can evolutionary search learn compact adapters that recover safe task completion
> around frozen world-model systems, reducing reliance on manually tuned behavioural rules?

LeWM is the opening study across TwoRoom, PushT and OGBench Cube. Block Plucking is
the intended main study, and ManiGuard is deferred. The common method learns a
small behavioural adapter around a frozen pretrained foundation. The two studies
need not have identical outputs or share learned weights.

The contribution sought is useful learned behaviour with less task-specific
behavioural and safety-penalty tuning. Matching a specialised expert reference
could be valuable; outperforming it is not a prerequisite. Comparable scores alone
do not establish equivalence or reduced human effort.

Keep the scope precise: the method avoids a manually selected safety-penalty
multiplier in evolutionary selection. It still requires a task reward, a violation
definition, a safety target and algorithm settings. The binary violation signal is
still a cost signal. This is not elimination of all hand-designed costs or all
human design, and constraint handling itself is not claimed as novel.

The study has three questions:

1. Can an adapter learn useful safe behaviour around the frozen model?
2. Does the same learning procedure remain useful across navigation and manipulation?
3. Can a requested risk limit act as an empirically useful safety dial?

## Environment suite and reuse claims

Use the original task in each environment:

- **TwoRoom:** navigate the controlled body to the original goal. Use it for
  development, mechanism analysis and the initial dial experiment.
- **PushT:** move and orient the T-block to the original task goal. Pusher motion
  alone is not completion; bind the pinned environment's actual goal predicate.
- **OGBench Cube:** move the cube to the original object goal. Do not substitute
  gripper reaching, a reduced subtask or a convenient image-distance threshold.

The existing Cube wrapper supports both object-moving and gripper-reaching modes.
The new protocol must explicitly bind and test the object-moving mode. Audit task
reward and termination against the selected goal, including dataset goal overrides.

Distinguish three claims:

- **Method reuse:** the same adapter family and learning procedure can be applied
  in each environment, training new weights.
- **Within-environment generalisation:** the resulting controller works on fresh
  source cases and hazard configurations.
- **Weight transfer:** weights trained in one environment work in another without
  retraining.

The initial study targets the first two. LeWM uses separate environment checkpoints;
equal latent dimensions do not establish aligned representations. Weight transfer
is a separate experiment, not an implied result.

Normalisation, legal action dimensions and task predicates may differ by environment.
Declare those mechanical interfaces. Freeze the general learning recipe after
development and apply it without repeated behavioural redesign or per-case tuning.
Any additional environment-specific tuning must be declared and counted, narrowing
the automation claim accordingly.

## Hazards, protection and case construction

Introduce supplied geometric forbidden regions with a fixed violation definition.
They are virtual constraints, not necessarily physical obstacles or visible features
in the policy images. Give learned methods the same declared hazard information.
Start with a simple shared hazard family, such as a circle in 2D and a specified
volume in Cube, before introducing richer shapes.

Define the protected geometry separately from task state:

- TwoRoom: the navigation body.
- PushT: explicitly choose the pusher alone or the pusher and T-block.
- Cube: explicitly choose a gripper volume, carried object, arm geometry or a named
  subset. A pinch-point or XY projection check does not establish whole-robot safety.

For Cube, specify the hazard's vertical extent and whether passing above it is
allowed. A physical obstacle that can be overflown and a forbidden vertical column
are different requirements. Freeze body geometry, units, required clearance and
monitoring cadence. Do not shrink protection to improve a result.

Include no-added-hazard controls and hazard cases with feasible task completion.
Direct progress should sometimes create risk, while inactivity remains unfinished.
Endpoint clearance is not route feasibility; a free pusher or gripper path is not
proof of a feasible pushing or grasping sequence. Use a predeclared admission
procedure independent of evaluated-method outcomes. Keep unknown feasibility
explicit and do not replace final cases after observing failures.

Choose source splits and hazard-generation rules prospectively. Keep source episodes
grouped across training, development and final splits. If using memory or a readout,
keep their acquisition sources separate from final query sources. Fresh cases in
this study do not prove absence from pretrained LeWM's original data.

## Common foundation and interface qualification

Freeze LeWM, preprocessing and the shared task planner. Freeze and account for any
additional assets actually used. A learned predictive safety checker is not required
in the monitor-only main condition; a geometric monitor can label actual execution.
Do not add motion-model fitting and calibration unless its predictions have a
declared role in the method.

Choose one adapter family before the main comparison. Within each environment,
evolution, parameter random search and RL use the same actor, inputs, authority and
action interpretation. Start from a generic initialisation or matched initialisation
distribution, without a successful expert coefficient vector or expert fallback.
The fixed expert controller remains a separately labelled reference.

Two interfaces merit a bounded development qualification:

1. **Intermediate-target selection:** choose an observed visual target for the
   frozen task planner.
2. **Bounded action residual:** correct the planner's proposed command using a small
   learned output, with legal action bounds retained.

Target selection is the TwoRoom prototype, not the established cross-environment
method. Residual control is a potentially simpler common interface because it removes
target memory and goal-position readout dependencies, and also fits an adapter
around Block Plucking's actor. Neither interface is assumed successful.

Give interface qualification a declared development allowance. Check baseline
competence, useful controller authority, task feedback and real safety trade-offs.
Choose using development evidence and freeze the choice before the main comparison.
Do not multiply every architecture by every algorithm, environment and dial setting.

### Candidate A: intermediate-target selection

At each decision, observe the current and goal images, native state and hazards;
retrieve a small fixed-rule set of candidate visual observations; score candidates;
select a target; and let frozen LeWM plan towards it. Execute a fixed prefix, then
observe and repeat. The original goal remains a candidate. An intermediate arrival
is never task completion.

Candidate generation and any locality filtering must be shared. Do not remove
hazardous candidates, provide oracle routes or hide an execution-time shield.
The current retrieval implementation masks targets by hazard clearance, so it
cannot be reused unchanged for this monitor-only condition.

A candidate scorer is:

$$
s_i = v^\top\tanh(Wx_i+b), \qquad
\pi_\theta(i\mid o)=\operatorname{softmax}(s)_i.
$$

The earlier twelve-input, eight-hidden-unit actor has 112 parameters: 96 input
weights, eight hidden biases and eight output weights. A shared scalar output bias
cancels in the softmax and is omitted. Its proposed features were candidate XY,
estimated goal XY, hazard-centre XY, radius, two latent distances, previous-target
XY and recent movement, expressed relatively and normalised.

That exact feature set is a TwoRoom example, not a requirement for PushT or Cube.
Those tasks require information about object configuration, contact or grasp state,
and potentially three-dimensional motion. A body goal position and two latent
distances may discard important task distinctions. Full visual targets could carry
these distinctions, but reachable transitions and the scoring representation still
need qualification.

If retaining memory, record its size, labels, source episodes, acquisition cost and
coverage. Candidate positions are declared labels; any goal estimate comes from a
shared frozen readout, not true query-goal coordinates. Separate the benefit of these
assets from the benefit of evolutionary updates.

### Candidate B: bounded action residual

Let the frozen planner propose a command and let the actor produce a bounded
correction:

$$
a_t=\operatorname{clip}_{\mathcal A}
\bigl(a_t^{\mathrm{base}}+B_e\tanh(u_t)\bigr).
$$

Here the clipping enforces native legal action limits, not hazard avoidance.
The fixed scaling B_e converts the correction to the declared environment action
units. It must leave meaningful behavioural authority without replacing the base
planner by an unrestricted new policy. Its setting and any tuning costs are explicit.

The actor can use frozen current/goal features, causal native observations, the
proposed base command, hazard description and a small declared history. Select
input and output dimensions only after reviewing the three task contracts. The
parameter count may differ for mechanical action dimensions while retaining the
same network family and learning recipe.

A zero correction should recover the base command path. Audit this with matched
RNG handling and recording. Record proposed, corrected and executed actions, action
clipping and the actual replan cadence. Replan from the observed executed state;
do not assume the unmodified plan still predicts a corrected trajectory.

PPO would train a continuous residual distribution with the appropriate log-probability
handling, rather than the categorical distribution of candidate A. Use the same
stochastic policy interpretation for evolutionary and RL training. Choose and freeze
deployment sampling versus deterministic evaluation; changing it changes the policy.

## Main safety condition

A passive monitor checks actual executed motion and records violations without
modifying commands. Audit target masks, planner filtering and fallbacks so they do
not enforce the behaviour being attributed to learning.

Keep initial-state admission, violation termination and original task termination
identical across methods. Report initial violations and engineering failures rather
than relabelling them as controller successes. Training may contain unsafe simulator
episodes; this does not demonstrate safe exploration or real-world safety.

A shared execution-time checker is an optional later deployment condition. Keep its
results separate from the main monitor-only study and do not credit checker-enforced
safety to the adapter.

## Evolutionary objective and optimizer

For candidate parameters theta evaluated over n complete episodes, record:

$$
\widehat J(\theta)=\frac{1}{n}\sum_i R_i,\qquad
\widehat p_V(\theta)=\frac{1}{n}\sum_i
\mathbf 1[\text{episode }i\text{ has any violation}].
$$

Use the verified original task reward by default, shared with RL. The existing
portable wrapper discards the returned native reward, so exposing and auditing it
is implementation work. Task-only shaping, if necessary, is a prospective
development decision shared by methods; it is not a hidden safety penalty.

At a fixed empirical risk target epsilon:

1. Candidates meeting the empirical safety requirement outrank those failing it.
2. Among feasible candidates, maximise mean task return.
3. Among infeasible candidates, minimise safety shortfall, then maximise task return.

Convert this ordering to minimisation ranks for CMA-ES. Preserve genuinely tied
outcomes and qualify ask/tell tie behaviour. If no candidate qualifies, report that
outcome; do not label the least unsafe candidate safe. Safe completion remains the
primary evaluation endpoint even though training uses separate reward and risk.

Use full-covariance CMA-ES as the first candidate optimizer for a compact continuous
actor. It can represent interactions between weights without planner or simulator
gradients. Reconsider its cost if qualification requires a much larger actor.
Start with the implementation's standard population rule and generic scaled
initialisation; choose search scale under a declared development allowance.

Population, update count, package version, repeats and budgets remain to be frozen.
Do not inherit earlier generation counts or successful controller coefficients.
Evolving neural weights is neuroevolution; topology evolution is unnecessary.
Additional genetic algorithms or NSGA-II are not part of the initial comparison.

Finite-panel feasibility is not a population guarantee. Eight episodes give rate
increments of 12.5 percentage points, so the risk target and sample sizes must be
consistent. Size nominee selection and final evaluation separately. Report uncertainty
and condition-specific outcomes; do not pool environments to hide risk in a hard task.
Feasibility-first search can stall at inactivity, so useful authority and feedback
must be established on development data.

## Comparisons and RL semantics

Run the main learned comparison across all three environments:

- **CMA-ES:** the proposed feasibility-based evolutionary adapter procedure.
- **Parameter random search:** the same representation, information, domain,
  candidate evaluation and selection rule without distribution adaptation.
- **PPO-Lagrangian:** the same actor family and formal reward/risk objective,
  with an adaptively updated safety multiplier.

Include frozen-planner, generic-untrained-adapter and expert-designed fixed references.
Define each reference prospectively. If the expert reference has a different
architecture, label it a system-level performance reference rather than a clean
optimizer comparison. Matching it does not prove autonomous discovery if its
successful parameters were supplied to the learned method.

PPO's actor is categorical for target selection or continuous for residual control.
Use reward and cost critics, and count their training. Verify the implementation
supports the chosen distribution and finite-horizon semantics. Its multiplier
initialisation and update settings remain hyperparameters with a development allowance.

Emit cost one at the first actual violation and zero otherwise. The undiscounted
episode sum equals the episode-violation indicator, so its expectation is the
probability of any violation. Counting unsafe steps measures a different quantity;
discounting late violations also changes the objective. Use cost discount one and
correct termination and rollout-buffer handling. If episodes continue after a
violation, preserve the first-event latch and necessary history.

Adapter decisions may span multiple simulator actions. Accumulate reward and cost
over the full executed prefix, monitor at the required cadence and handle termination
partway through it. Primitive-step task discounts must respect actual macro-transition
duration. Do not miss violations between adapter decisions.

Give all methods explicit development and simulator-interaction allowances, shared
feedback and online planning allowances. Report actual actions, resets, model queries,
critic work, discarded candidates, selection and engineering failures separately.
Equal episode counts do not establish equal work. Match training distributions and
use common candidate panels/seeds within evolutionary generations where appropriate.
Reevaluate nominees on a common selection panel rather than comparing historical
scores from different panels. Preserve method-appropriate on-policy PPO sampling.

## Focused ablations

Use TwoRoom for the fuller mechanism analysis, then repeat the essential learning
and safety-selection checks in PushT and Cube. Freeze their scope before final outcomes.

1. **Training contribution:** learned versus generic untrained adapter and frozen
   planner. Match stochastic evaluation semantics.
2. **Safety-selection contribution:** feasibility-based versus task-only evolution,
   keeping the optimizer, actor, feedback and budgets otherwise matched.
3. **Penalty-handling contribution:** feasibility-based versus penalised evolution.
   Choose any penalty grid and development selection prospectively and count it.
4. **Hazard-information contribution:** retrain without hazard inputs to distinguish
   hazard-conditioned behaviour from a generic conservative habit.
5. **Architecture contribution:** remove history or use a simpler actor if these
   components survive qualification.

If using target selection, examine the readout and memory contribution explicitly.
A linear scorer needs candidate-hazard relationships or interactions: features
constant across candidates merely add a common logit offset and cancel in softmax.
Deleting the hidden layer while retaining only additive constant hazard context is
not an adequate simpler safety-conditioned actor.

Retrain structural and information-removal ablations under declared budgets.
Corrupting inputs only at evaluation is a separate reliance or robustness diagnostic.
Keep a fixed-penalty PPO sensitivity study in TwoRoom if resources permit; it
complements, not replaces, PPO-Lagrangian. Avoid a full Cartesian product of actors,
optimizers, penalties, shields, environments and dial settings.

## Safety dial

The preferred dial is the requested episode-violation-probability limit epsilon:

$$
\max_\theta J(\theta)\quad\text{subject to}\quad p_V(\theta)\leq\epsilon.
$$

Hold the hazard, protected geometry, clearance requirement and violation definition
fixed while changing the requested risk limit. A looser limit permits more risk;
it does not require the policy to become unsafe. The risk specification is distinct
from the empirical selection rule and the measured final risk.

First establish the method at one fixed target. Then, initially in TwoRoom, use a
small prospectively chosen set of risk targets to train/select a discrete policy
library. Give compared learning methods the same targets and declared acquisition
allowances. Lock policies using development data before opening final dial outcomes.
If a target yields no qualifying candidate, report that rather than inventing a
qualified fallback. Count all training and library selection.

For each setting, evaluate task return, safe completion, violation frequency,
unfinished episodes and engagement with uncertainty. Check whether stricter requests
produce lower observed risk with useful retained performance. Do not assume
monotonicity, interpolate guarantees between settings or select dial levels after
looking at final curves. Identical policies, uniformly zero violations or increasing
inactivity alone do not establish a useful reward-risk dial.

A single risk-conditioned network and unseen-dial-level generalisation are later
extensions. A clearance-margin dial is also possible, but it changes the safety
specification and must be distinguished from a risk-tolerance dial. Adjustable risk
budgets are established methodology, not a novelty claim by themselves.

## Evaluation, automation and uncertainty

Freeze the learning recipe after development. Train new weights per environment
and evaluate independent learner repeats on reserved source cases without per-case
manual tuning. Keep learner-repeat and source-case variation distinct. Retain host
provenance and contemporary controls; repeated cases are not independent new data.

Report, per environment and declared condition, with denominators:

- Safe completion, the primary joint endpoint.
- All completion, including completion with a violation.
- Episodes with any violation.
- Unfinished episodes and engagement.
- Task return, actual simulator work, model queries and elapsed time.

Separate training violations and acquisition costs from final deployment outcomes.
Use a declared failure-capped completion cost if reporting action efficiency, so
early failure does not appear efficient. Do not pool heterogeneous tasks or shields.
Zero observed violations is not zero population risk.

Define acceptable performance differences and analysis before claiming comparability
or non-inferiority to an expert reference. A nonsignificant difference is not
equivalence. Demonstrate reduced tuning through the frozen cross-condition recipe,
all tuning trials and an inventory of remaining human choices. Do not invent a
retrospective human-time saving from an existing controller.

## Qualification and prospective freeze

Proceed in a staged order:

1. Bind the three task contracts, goals, rewards, action normalisation and protection.
2. Qualify baseline competence, hazard relevance and adapter authority on development cases.
3. Select the interface and small actor under a bounded qualification allowance.
4. Freeze the main objective, actor, risk target, panels, repeats, algorithms,
   selection rules, uncertainty analysis, stopping rules and all resource budgets.
5. Run the main three-environment comparison and the predeclared focused ablations.
6. Open the separately frozen TwoRoom dial study after its prerequisite qualification.

Implementation checks must cover actor agreement across optimizer paths; absence
of hidden hazard masks and shields; original object-goal scoring; actual-action and
dense-monitor consistency; rewards, first-event costs and macro-transition semantics;
ties and complete-panel updates; RNG/checkpoint recovery; and exact accounting.
For residuals, include zero/native parity and correction/clipping provenance.
For target selection, include memory/readout provenance and information boundaries.

All experiment computation and experiment tests belong on authorised rented hosts.
Local document work does not authorise numerical experiments. New studies use fresh
files and outputs; previously inspected panels become development evidence for
future decisions. Historical budgets and rentals do not resume.

## Supporting references

- [LeWM official code and checkpoint suite](https://github.com/lucas-maes/le-wm):
  frozen task models and environment-specific assets.
- [OGBench official implementation](https://github.com/seohongpark/ogbench):
  manipulation task definitions; bind the pinned local variant explicitly.
- [CMA-ES author reference](https://cma-es.github.io/):
  continuous black-box search, covariance adaptation and initialisation.
- [PPO, Schulman et al., 2017](https://arxiv.org/abs/1707.06347):
  the policy-gradient basis of the RL arms.
- [OmniSafe PPO-Lagrangian](https://omnisafe.readthedocs.io/en/latest/saferlapi/lagrange.html):
  candidate implementation reference, requiring distribution and cost-semantic qualification.
- [Feasibility-first constraint handling](https://pymoo.org/constraints/feas_first.html):
  the established candidate comparison principle.
- [Constrained Policy Optimization, Achiam et al., 2017](https://proceedings.mlr.press/v70/achiam17a.html):
  constrained-RL context; no guarantees are inherited by our procedure.
- [Budgeted RL, Carrara et al., NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/4fe5149039b52765bde64beb9f674940-Abstract.html):
  adjustable risk budgets as an established formulation.

The simple two-page outline is in [Rule42.tex](Rule42.tex), compiled to
[Rule42.pdf](Rule42.pdf). This note retains the technical alternatives and unresolved
choices that must be settled before execution.
