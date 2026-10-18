# Phase 3: evolutionary execution adaptation for safe manipulation

**Active working plan, 18 October 2026.** This document consolidates the phase-three
direction and the subsequent methodological review. It is a research plan, not a frozen
experimental protocol or a report of phase-three results. The safety dial, optimizer
details, comparisons, statistical thresholds and total budget still need the decisions
identified below before implementation and main-study execution.

**Latest resource constraint, 18 October 2026.** Sachin requires the first useful
experiment to fit on **one RTX 4090 or RTX 5090 within eight hours of allocated wall
time**, including setup on that machine, exploratory runs and evaluation. The earlier
eight-GPU estimates and staged ManiGuard allocations below do not satisfy this request
and are not the next implementation commitment. Sachin clarified that the experiment
must remain in **ManiGuard with evolutionary search**. Safe CEM with LeWM was an example
of the desired cost and concreteness, not a request to change platforms. Section 21
summarizes the agreed small ManiGuard experiment, including the user-controlled dial.
The comprehensive implementation plan is [phase3Part1.md](phase3Part1.md); its exact
machine-readable run protocol is not yet frozen.

This document is the active research plan and direction. The
[plan history](docs/research/history/README.md) preserves the earlier oracle, ideation,
candidate studies and phase-two plan. The
[archived direction](docs/research/history/20261018/direction.md) records the original
18 October decisions and initial estimates. This document supersedes that record's
prospective experimental details where the review corrects them. The
[phase-two plan](docs/research/history/20261018/phase2.md), its frozen configs and recorded
results remain unchanged. Creating this plan does not resume a paused job, authorize
additional rental or freeze a new run allocation.

## 1. Research question and contribution

The proposed primary question is:

> Under a fixed interaction budget, can evolutionary search discover a compact execution
> controller that improves safe task completion around a frozen manipulation policy more
> reliably than matched random search and ordinary learning?

The proposed mechanism is a bounded, history-dependent execution layer, called the
**safety dial**, optimized using noise-aware evaluation and penalty-free constraint
handling. The pretrained vision-language-action policy, or VLA, stays frozen.

Four claims need separate evidence:

1. **The execution interface helps.** A selected controller improves safe completion over
   the original policy and an identity controller, beyond a simple fixed correction.
2. **Evolution adds value.** Retention and mutation of promising controllers improve
   final performance or interaction efficiency relative to matched random search and
   an appropriately matched ordinary-learning comparison.
3. **Constraint handling helps.** Feasibility-based or multi-objective selection improves
   the measured completion and violation trade-off relative to a scalar-penalty
   alternative, with other relevant factors controlled.
4. **The result generalizes.** Performance persists on reserved scenes and, if claimed,
   families excluded from optimization or method development.

The recommendation is to make evolutionary adaptation under a measured budget the
primary question. Penalty-free selection is a proposed explanation to test through
controlled comparisons. Generalization supports the scope of the result. An improvement
over the original VLA alone does not establish an evolutionary advantage.

Sachin's target remains a convincing evolutionary result for a GECCO 2027 full paper,
within the [allocated topic](docs/allocatedTopic.md). The gates must establish that the
study can answer its question. They must not require an EA win before the comparison is
allowed to count. A null result, trade-off or failure remains evidence; it cannot be
relabelled as superiority or discarded from interaction accounting.

## 2. Decisions retained from 18 October

The original direction records the following decisions:

- Pause the frozen phase-two pilot at a block boundary after evolution/s17 panel 0 and
  the reference block, preserving the partial evidence and void receipts. Do not restart
  it merely because phase-three infrastructure reuses its helpers.
- Run the separately frozen Clutter replicate diagnostic over the eight round-0 genomes,
  three pilot scenes and eight replicates per controller-scene cell.
- Investigate a compact safety-dial execution layer with penalty-free, noise-aware
  evolutionary search.
- Require a heritability probe before a ten-seed main study.
- Use the existing eight-RTX-3090 machine only. The recorded planning rate is about
  USD 53 per day; no second rental was chosen. This is a recorded allocation constraint,
  not a fresh observation of machine availability or rental status. The later single-GPU,
  eight-hour constraint at the top of this document supersedes it for the next pilot.
- Keep genetic programming of sub-goal re-prompting as a gated, optional branch.

The exact algorithms and experimental design listed in the original direction are
candidate choices, subject to the refinements and unresolved decisions in this plan.

## 3. Evidence motivating the change

The figures below are recorded in the [original direction](docs/research/history/20261018/direction.md) and the
[phase-two development log](docs/research/research_notes/ManiGuard%20phase2%20development.md).
They are partial pilot and diagnostic evidence, not a completed optimizer comparison.

### Repeatability

Random search and evolution begin with the same eight round-0 genomes on the same
cases and seeds. Of 22 paired Clutter and Jar episodes inspected in the original
direction, nine disagree in outcome. Initial robot states match, while the first
rendered frame differs in roughly 40 to 70 percent of pixels by one or two levels.
The VLA can consequently diverge from its first proposed action.

This establishes a consequential repeatability concern. It does not yet establish that
rollout noise dominates every useful controller difference, that candidate rankings are
uninformative, or that evolution, random search and PPO must tie. An outcome probability
per controller and case is the appropriate object of measurement.

### Family signal and failure anatomy

Clutter is near 50 percent safe completion in the recorded pilot observations and
provides most of the observed fitness variation. Jar records two safe completions in
22 inspected episodes, both for evolutionary candidates whose identical random-search
twins timed out. Those paired differences are not evidence of optimizer superiority.
Stack records no completion in 25 episodes on the new host and two in eight on the old
host, with most episodes reaching the 4,200-action horizon.

In 22 Jar traces, the lid is closed in nine episodes, commonly by action 300 to 350.
Observed losses include tipping the open jar before grasping and failing to grasp or
lift after closure. These suggest possible execution and sequencing interventions.
They do not establish that caps, smoothing or delayed commands can repair the failures.

The completed [six-hypothesis study](docs/research/reports/ManiGuard%20six-hypothesis%20study.md)
did not establish a repeatable safe-completion gain from its execution adapters. The
completed [UNISafe study](docs/research/reports/UNISafe%20grasp%20maintenance%20and%20evolutionary%20adaptation.md)
found a useful correction without an evolutionary advantage. Phase three must test the
usefulness of its lever rather than assume that a new optimizer will make it useful.

### Throughput

The original direction records median process times under eight-way load of 417 seconds
for Clutter, 626 seconds for Jar and 1,161 seconds for Stack. Its approximate throughput
assumption is 1,400 Clutter episodes per day on eight GPUs, with CPU-bound simulation.
These are planning inputs from that host and workload, not guarantees for a new controller
or a fresh throughput measurement.

The [problem inventory](docs/research/research_notes/Problem%20inventory%2020261018.md)
provides broader motivation: previous results have depended strongly on the quality of
the evaluation signal and the behavioural lever. It explores problems and selects no
replacement method.

## 4. Scope: safe execution, monitoring and learned filtering

The core study is **safe execution adaptation around frozen policies**. ManiGuard's
monitor evaluates simulator-derived predicates and temporal specifications. It does not
predictively modify proposed actions or abort a rollout after a violation. Its labels
support evaluation; they are not a learned action-changing safety filter.
[ManiGuard formulation and evaluation](https://arxiv.org/html/2608.17386v1)

The proposed deployment loop is: observe, query the frozen VLA when required, adapt the
proposed commands, execute them, and update causal controller history. Evolution changes
controller parameters between episodes. Deployment executes one selected controller
without population search at every action.

Monitor predicates, goal flags and supplemental completion labels remain evaluation-only.
The controller is trained against monitor-derived outcomes; it is not monitor-aware at
runtime unless an explicitly permitted monitor input is added in a new protocol. The
same restriction applies to the ordinary-learning critic's observations. Outcome-derived
training targets are distinct from privileged runtime inputs.

No new world model or learned safety model is a dependency of this core plan. A claim
about continuation under learned safety filtering needs a separately specified extension:
a validated predictive filter held fixed across methods, matched adapter/no-adapter and
filter/no-filter comparisons, prediction-error audits, and progress measurements around
intervention. Include acquisition, training and query costs. The original research focus
is not established merely by improving ManiGuard safe completion.

Keep original VLA proposals, adapter outputs, any genuinely filtered outputs, and commands
actually supplied to the environment distinguishable in traces. Do not call the adapter
output a learned-filter output when no such filter is present. Physical dynamics require
executed-action histories. A task critic can intentionally model proposals to the adapted
environment, but that information contract must be explicit and matched.

All safety conclusions concern the audited simulator specifications and evaluated
conditions. Monitor-verified completion is not a certificate of arbitrary physical safety
or a real-robot result.

## 5. The safety-dial search space

A candidate is a bounded parameter vector configuring a shared causal execution
procedure. Parameters stay fixed within an episode; their effects vary with current
observations and episode-local history. The original design suggested approximately
10 to 16 parameters. Section 20 recommends beginning with a four-parameter implementation
and expanding only when the inexpensive diagnostics justify an additional control.

The proposed control groups are:

- Joint-command change caps and smoothing, potentially conditioned on gripper command
  state.
- An additional cap near a defined low-height region while the gripper command is open,
  motivated by pre-grasp tipping.
- A bounded hold period after a close command, potentially with a precisely defined
  motion or lift attenuation.
- A bounded delay before executing a subsequent opening command.
- A VLA query interval and stall-triggered re-query thresholds.

These are candidate mechanisms, not established repairs. An open gripper does not
identify an open jar. A close command does not establish a grasp. End-effector motion
does not establish object progress. Re-querying helps only if the VLA can propose a
useful alternative. Slowing an incorrect approach may retain the failure or cause a
timeout. Command caps do not certify physical velocity, contact force or safety.

### Information and representation contract

Candidate online inputs include measured joints and fingers, end-effector position and
motion, proposed commands, actual environment inputs, tracking error, proposal changes,
gripper command history and time since the last query. Task text and frozen visual
features are possible additional inputs, but their role in a 10-to-16-parameter procedure
must be defined rather than inherited automatically from the phase-two neural controller.

Exclude task-family identifiers, scene names, checkpoint identifiers, simulator object
states, monitor predicates and requested-completion flags from runtime controller and
critic observations. Ordinary learning must not receive additional privileged state.
Each family may retain its own frozen VLA checkpoint; shared adapter parameters do not
mean that there is one generalist backbone.

Before implementation, define each parameter's input signals, units, numerical range,
disabled value, activation condition, reset behaviour and precedence during composition.
Define the coordinate frame and reference surface for low height. Distinguish attenuation
of every joint-command change from selective attenuation of upward end-effector motion;
the latter needs a validated geometric mapping. Define stall using available signals
rather than unobserved task progress. Bound holds, delayed release and repeated re-query
so they cannot silently create indefinite waiting.

Normalize mutation coordinates across quantities with different units. Specify rounding
for discrete durations, boundary handling and how duplicate effective controllers are
identified. Ten to sixteen parameters is an initial design target, not a reason to add
inactive or redundant controls.

### Identity and mechanism checks

An explicit identity setting disables every modification and preserves the original
command and query procedure. First verify identity on identical recorded inputs and
synthetic boundary cases. Then use bounded repeated baseline comparisons; nondeterministic
physics makes exact trajectory equality an unsuitable sole acceptance criterion.

Trace parameter values, relevant causal inputs, active conditions, intermediate command
transforms, re-query reasons and actual environment inputs. Verify that candidate changes
produce distinguishable behaviour. A numerically different genome with the same effective
behaviour is not evidence of a useful search direction.

## 6. Outcome contract and optimization target

Use the existing [requested-sequence completion contract](docs/research/research_notes/ManiGuard%20phase2%20completion%20contract.md)
as the starting point, with source identity and phase-three applicability checked before
execution. Preserve all original benchmark fields alongside supplemental labels.

Let `C_req` denote requested completion, including the family-specific sequence evidence;
let `V` denote the raw recorded `ltl_violated` verdict; and let
`S_req = C_req and not V`. Keep engagement-gated benchmark violations separate from raw
`V`, as the existing contract requires. In Jar, native goal success alone does not
establish closure before lift or closure while carrying.

Report the joint outcomes: safe completion, completion with a violation, unfinished
without a violation, and unfinished with a violation. Also retain benchmark-native
success and unsuccessful termination reasons. Completion, violation and unfinished rates
must not be presented as three mutually exclusive categories. Fewer interventions or
violations alone can reflect inactivity.

Fitness estimates concern a declared distribution of training scenes and rollout
variation. Specify scene sampling, family weighting and aggregation before optimization.
Four repeated rollouts on a few cases estimate a narrower target than general performance
across a family. Report per-family outcomes alongside any pooled score.

Incomplete traces, unavailable predicates, contradictory labels and infrastructure
failures are not ordinary task failures with reward zero. Preserve invalidity, raw
evidence and available costs, and follow the declared blocking rule. Do not substitute
missing evidence with an assumed safe or unsuccessful outcome.

## 7. Constraint handling and the meaning of penalty-free

The existing scalar utility is `U = int(S_req) - 2 * int(V)`. Its coefficient specifies
how safe completions compensate for violating episodes when scores are aggregated. A
violating completion scores -2 under this definition, not -1.

Feasibility-based selection instead compares constraint satisfaction and task performance
separately. [Deb's feasibility-based method](https://www.sciencedirect.com/science/article/pii/S0045782599003898/pdf)
provides established precedent for selection without a tuned penalty coefficient.
It does not remove the need to define safety or an acceptable level of violation risk.

### Stochastic feasibility is an unresolved design choice

Define whether feasibility means zero observed violations, an estimated violation rate
below a declared threshold, an uncertainty-bound criterion, or a different specified rule.
These choices are not interchangeable. Specify tie handling, small samples, insufficient
evidence and how final deployment selection relates to training selection.

For example, over four episodes a controller with no completions and no violations has
mean scalar utility 0. A controller with three safe completions and one violating episode
has mean utility 0.25. The scalar utility prefers the latter; a strict safety-first rule
could prefer the inactive controller. Preventing this failure mode is part of the research
problem, not an automatic property of feasibility ranking.

Zero violations in four independent episodes has an exact one-sided 95 percent binomial
upper bound of about 53 percent. It cannot certify negligible violation probability.
The general rule of thumb, approximately `3/n`, is useful at larger independent sample
sizes and must retain its denominator and scope.

### Scalar and multi-objective comparisons

Use the same ES variation and evaluation machinery for a scalar-penalty ablation so that
the ranking rule is the principal changed factor. Beating the coefficient 2 supports a
comparison with that setting. A broader claim about avoiding penalty tuning needs a
declared penalty-selection procedure and its cost.

For NSGA-II, settle whether the objectives are requested completion and violation rate,
as originally proposed, or safe completion and violation rate. They express different
preferences. Specify dominance under noisy estimates, objective normalization, hypervolume
reference point, archive evaluation and a fixed rule for selecting a deployed controller.
Do not choose a front member after inspecting reserved outcomes.
[Original NSGA-II paper](https://web.njit.edu/~horacio/Math451H/download/2002-6-2-DEB-NSGA-II.pdf)

The core plan does not yet specify a Lagrangian RL baseline. It therefore cannot support
a claim of superiority over Lagrangian safe RL. Calling a method penalty-free also does
not mean that no safety specification, threshold or human preference has been supplied.
ROSARL integration remains an explicit open decision, not an assumed implementation.

## 8. Optimizer and reference comparisons

The initial full comparison set is feasibility-ranked ES, scalar-ranked ES, NSGA-II,
random search and ordinary policy-gradient learning, plus identity and a hand-written
reference. This set is not yet frozen or fully costed.

### Feasibility-ranked evolution strategy

The proposed starting point is a `(mu, lambda)` evolution strategy: selected parents
generate offspring, offspring receive replicated evaluation, and their rankings determine
the next parents. Comma selection does not retain a parent unchanged solely on its old
score. Define `mu`, `lambda`, recombination, initialization, mutation scale and adaptation,
boundary handling, tie breaking, candidate archiving and final selection.

Rank-based selection is insensitive to monotone transformations of fitness. It is not
immune to noise reversing candidate order. The direction's claim that larger populations
are always preferable to replication must be treated as a hypothesis to evaluate under
the measured rollout noise and budget.

### Scalar-penalty evolution strategy

Hold the search space, initial population, variation operators, evaluation allocation
and relevant hyperparameters fixed relative to feasibility-ranked ES. Change the specified
ranking rule. If other factors differ, report the contrast as a combined-method comparison
rather than attributing the result solely to constraint handling.

### Random search

Use the same parameter domain and initial sampling distribution, matched initial
candidates where appropriate, and the same candidate-evaluation and finalist-selection
opportunities. Count all candidates and re-evaluations. Random search must not be
disadvantaged by a weaker selection stage or different prior over useful controllers.

### NSGA-II

NSGA-II answers a trade-off question and normally retains elites through parent-offspring
selection. A blanket rule never to re-evaluate parents can preserve lucky estimates in
that setting. Its noisy dominance, elite re-evaluation and archive policy must therefore
be specified separately and budgeted. A front is not automatically an unqualified safety
improvement or a single-controller deployment result.

### Ordinary learning and PPO

Reusing the existing recurrent PPO unchanged would not automatically match the new dial.
The phase-two actor chooses execution settings over time, while the proposed dial uses
fixed parameters to configure another history-dependent procedure. Same observations and
episode counts do not eliminate this representation difference.

Choose which question the RL arm addresses. An optimizer comparison within the same
controller family requires matching the behavioural parameterization and decision
opportunities. A comparison of practical adaptation systems may allow different internal
architectures, but must expose comparable execution controls and disclose remaining
differences. An episodic learning distribution over dial parameters is one possible
matched representation, not a decision already taken.

Specify training objective, safety feedback, critic information, initialization,
hyperparameter-selection costs and deterministic or stochastic deployment. A task-return-only
PPO comparison does not establish superiority over safe RL. Match information advantages
and resource accounting, not merely the method names.

### References and external adaptation methods

Include the original executor and dial identity checks. Freeze the hand-written rule
using permitted development evidence before reserved testing. Count its selection and
tuning interactions. If one simple rule explains the entire gain, report that result.

The [phase-two related-work review](docs/research/research_notes/ManiGuard%20phase2%20related%20work.md)
recommended a clearly identified joint-space adaptation of Adaptive Action Chunking.
The original phase-three sketch omitted this external execution baseline. Decide whether
to include it, with its extra proposal-generation cost, or explicitly narrow the comparison
claim. A heuristic inspired by a paper must not be labelled a faithful reproduction.

## 9. Literature positioning and limits

The relevant primary work establishes precedent for several ingredients:

- [VLA-Pilot](https://arxiv.org/html/2511.14178v2) refines action proposals through
  evolutionary optimization during deployment. The proposed dial instead spends
  adaptation interactions before deployment to select a reusable controller.
- [DSRL](https://arxiv.org/html/2506.15799v2) adapts frozen diffusion policies through
  latent-noise steering using RL. It is important positioning and a possible systems
  comparator, but not a drop-in optimizer over the proposed execution-dial parameters.
- [Evolution Strategies at Scale](https://arxiv.org/html/2509.24372v3) reports ES results
  in LLM fine-tuning. It motivates testing outcome-based evolutionary adaptation; it does
  not predict an advantage for this manipulation interface or noise process.
- [Adaptive re-evaluation for ES](https://arxiv.org/html/2409.16757v3) studies additive
  Gaussian noise and artificial objective functions. It motivates evaluation allocation,
  but does not validate a particular racing procedure for binary safety outcomes here.
- [Extract-QD](https://arxiv.org/abs/2502.06585) supplies relevant precedent for
  re-evaluation in uncertain search and archives, not a reason to assume an archive is
  reliable without fresh measurement.

These sources were checked during the 18 October review. That review was targeted, not
an exhaustive novelty search. The original direction's absence claims and broad
prescriptions are not established facts. Recheck the nearest primary literature before
claiming novelty or finalizing the paper's comparison set. Substituting an optimizer or
combining existing ingredients is not sufficient novelty by itself.

## 10. Stage A: frozen replicate diagnostic

The existing [replicate protocol](configs/maniguard/phase2/replicate-v1.json) is separate
from the frozen pilot and has its own resource envelope and failure rules. Its full
design is eight existing controllers, three Clutter scenes and eight replicates per
controller-scene cell: **192 episodes**, 24 per controller.

It tests repeated-outcome variability, differences among these controllers and agreement
between repeated rates and the original single-episode fitness vectors. It selects no
controller for deployment or carry-forward. The resumable coordinator, runner and tests
already exist under `experiments/`; their presence is not a completed diagnostic result.

Interpret controller, scene and controller-by-scene effects separately. Interleave
evaluations across time and hardware to avoid assigning a controller effect to a GPU or
execution period. Check effective renderer settings, policy sampling/RNG handling and
other runtime provenance in retained artifacts before attributing the variability to a
particular source. The episode launcher requests the `legacy` source profile; a renderer
configuration mismatch remains something to verify from effective run settings rather
than assume from upstream defaults.

This diagnostic concerns the phase-two neural controller space. It informs noise and
budget decisions, but neither proves nor disproves useful variation in the new dial.
Preserve the frozen descriptive analysis; any supplementary stratified analysis must be
identified as additional analysis rather than silently rewriting its prespecified tests.

## 11. Stage B: dial definition and engineering checks

Settle the dial specification and information contract before building the optimizer.
Reuse validated phase-two interfaces where their semantics match, without changing frozen
sources or historical evidence in place. New reusable code belongs in
`experiments/helpers/`, entry points in `experiments/scripts/`, and configs in a new
phase-three group under `configs/maniguard/`.

Engineering checks should establish identity, bounded and finite commands, causal history,
episode reset, defined composition order, gripper transitions, query scheduling, and
separation of runtime inputs from evaluation labels. Verify actual behavioural activation,
not merely that a parameter changes numerically. Use bounded simulator checks only where
recorded-input or synthetic checks cannot validate the interface.

Declare engineering episode, action, query and elapsed-time limits before execution.
Retain every attempted interaction and invalid run. Tests and integration checks establish
implementation behaviour; they are not success-rate measurements.

## 12. Stage C: heritability and usefulness probe

The proposed design samples 16 uniform parents and one Gaussian mutant per parent. Each
of the 32 controllers receives four replicates on each of two Clutter scenes and two
replicates on one Jar scene. As written, this is **320 episodes**, not the original
estimate of about 250. Identity and hand-rule anchors, if added, require additional
declared episodes or an explicit reallocation.

The probe asks whether the dial changes effective behaviour, whether outcome differences
are meaningful relative to rollout variation, and whether mutation preserves some useful
performance structure. Parent-offspring correlation is a local search diagnostic, not a
biological heritability estimate or proof of eventual optimizer superiority.

Freeze the mutation scale in normalized coordinates, boundary and rounding rules, case
sampling, execution order and analysis before running. There are 16 parent-offspring
pairs, not 32 independent pairs. Include uncertainty on correlation and condition analysis
on scene and family rather than mixing different base rates indiscriminately.

A pass requires prespecified evidence of effective behavioural variation, practically
relevant outcome differences and useful local structure. A small positive sample
correlation alone is insufficient; consistently poor controllers can also correlate.
The exact effect thresholds, uncertainty criterion and treatment of multiple tests remain
to be chosen and costed before execution.

Allow three outcomes:

- **Pass:** the measured structure justifies a bounded optimizer comparison.
- **Fail under the tested design:** the interface or mutation scale lacks useful evidence
  at a declared resolution, so diagnose or redesign before a large study.
- **Inconclusive:** uncertainty remains too large to decide. Report it and choose a
  separately bounded follow-up or stop; do not silently extend sampling.

An inconclusive or failed probe does not prove that evolution cannot work anywhere in
the space. Passing it does not establish a random-search advantage. The recommended next
step is a small, prospectively budgeted ES-versus-random pilot using the same dial and
evaluation procedure before committing to ten seeds. That pilot is a recommendation,
not an already allocated experiment.

## 13. Noise-aware candidate evaluation and selection

The initial main-study sketch gives each candidate two scenes with two replicates each.
Four binary safe-completion observations produce only five possible empirical rates:
0, 25, 50, 75 and 100 percent. Under an independent-binomial model, two successes out of
four give a Wilson 95 percent interval of about 15 to 85 percent. Scene dependence can
make that simple model inappropriate for broader generalization.

Optimization may still accumulate useful information across generations, but close
candidates will be difficult to rank. Four rollouts are an allocation proposal to test,
not a validated estimate of adequate fitness precision. Larger populations and more
replication spend the same finite budget differently. Choose their balance using the
diagnostic and bounded pilot, before reserved results are seen.

Specify a common evaluation procedure for the relevant comparisons: scene panels,
replication, seed and residual-noise handling, tie resolution, re-evaluation allocation,
archive updates and finalist selection. Any shared-randomness design must demonstrate
what is actually shared; equal seeds do not remove renderer or physics nondeterminism.
Avoid confounding method with hardware, run time or changing software.

Finalist evaluation should be stronger than initial screening. If racing is used, freeze
its candidate-entry rule, maximum allocation, stopping rule and uncertainty treatment.
Count every re-evaluation and give random search comparable finalist-selection resources.
Do not retain lucky candidates indefinitely on stale fitness, report only the best noisy
training score, or treat adaptive selection observations as an independent final result.

## 14. Families, shared parameters and generalization

The proposed family roles are Clutter for a usable completion rate, Jar for safety and
sequencing mechanisms, and Lid for transfer with prior design exposure disclosed. Cabinet
and Dusty remain untouched stronger transfer candidates. Stack is excluded from training
because its current completion signal at the existing horizon is very limited; its
recorded outcomes remain visible.

The first major unresolved choice is **one shared controller vector or separate
family-specific controllers**. The original sketch's phrase "Clutter, plus Jar" does
not resolve it. Do not inherit phase two's shared-vector claim without specifying and
budgeting the corresponding joint optimization.

A shared controller needs declared family sampling and weights, common feature and action
semantics, and per-family results so Clutter cannot conceal Jar regression. Separate
controllers support a claim about a shared adaptation method, not zero-shot transfer of
one optimized parameter vector.

Preserve existing scene assignments and grouping of variants. Use training cases for
optimization and permitted engineering cases for design. Lid is not an entirely untouched
research family. If Cabinet or Dusty outcomes inform features, parameters or method choice,
disclose that exposure and narrow any later transfer claim.

Do not replace a nominated family because its observed score is disappointing. Record
asset or integration limitations separately. Adapter-held-out scenes or families may
have been present in the original VLA's training data; the holdout claim concerns adapter
development and optimization unless stronger provenance is available.

## 15. Reserved evaluation and statistical claims

Maintain three distinct stages: training guides optimization, validation selects
finalists, and reserved testing estimates performance after the choices are frozen.
Validation racing does not replace an untouched final stage. Fix the number of finalists
and selection procedure per optimizer run before validation. If a method outputs a front,
freeze how one controller or a declared set is selected for reserved evaluation.

The proposal of ten optimizer seeds per method is a planning choice, not proof of adequate
power or a universal venue requirement. Choose final episode counts using meaningful
effect sizes and the observed between-run, between-scene and within-scene variation.
Use a declared matched evaluation panel where appropriate and preserve all run seeds,
including poor and incomplete runs.

Primary reporting should include safe-completion differences with uncertainty, violation
differences with uncertainty, unfinished and unsafe-completion counts, and interaction
efficiency. Report optimizer-seed distributions and per-family outcomes. Define the
bootstrap or model so it respects repeated controllers, optimizer runs and scenes; these
can be crossed sources of variation. Do not pool all rollout observations as independent
evidence for optimizer superiority. Predeclare primary contrasts, effect-size summaries
and any multiple-comparison adjustment rather than adding tests after inspection.

The claim "improves safe completion without raising violations" needs more than a
nonsignificant violation-rate difference. Specify a meaningful maximum tolerated increase
and enough reserved evaluation to bound that contrast. If that is infeasible, report
the observed difference and uncertainty and use a narrower claim. An absolute risk limit
and noninferiority to baseline are different requirements; choose explicitly.

With zero violations in `n` independent episodes, the approximate 95 percent upper rate
bound is `3/n`, not zero. Preserve counts, denominators and the evaluated distribution.
Many repeats on a small scene set do not establish safety across arbitrary new scenes.

Mechanism evidence should show which dial controls activated, how commands and query
timing changed, and which failure modes improved or persisted. Determine whether one
simple setting explains the gain before attributing it to interacting temporal controls.
Use only appropriately designated development or post-freeze descriptive analyses;
reserved traces must not become an undisclosed tuning set.

## 16. Optional sub-instruction and genetic-programming branch

The original direction proposes a ten-episode Jar probe using a lid-only instruction,
followed conditionally by genetic programming over sub-goal re-prompting programs such
as close, grasp and carry. Mechanical ability to replace a prompt does not establish that
the VLA follows the instruction reliably or that the necessary transition is observable.

Before this probe, define scenes, control instructions, outcome criteria, allowed runtime
signals, budget and the decision it can support. Ten episodes can motivate further
investigation; they do not establish reliable prerequisite control. All interactions
count, including unsuccessful instruction variants.

GP changes the search representation, action interface, evolutionary mechanism and
comparison requirements. It is a separately specified, costed branch, not an automatic
fallback attached to an inconclusive dial result. Preserve a negative dial study rather
than silently selecting whichever representation yields a favourable result.

## 17. Compute, interaction accounting and corrected budget

The original per-run sketch uses population 12, 15 generations, two scenes and two
replicates: `12 * 15 * 2 * 2 = 720` episodes. The frozen run definition must state whether
the initial population is included in those 15 evaluated populations; an additional
initialization population would add cost.

At 720 episodes per run, five methods over ten seeds require **36,000 training episodes**.
At the recorded assumption of 1,400 Clutter-equivalent episodes per day, those training
episodes alone take about **25.7 days** and **USD 1,363** at USD 53 per day. This corrects
the original estimate of 30,000 to 35,000 episodes, about 20 days and about USD 1,100.

The 36,000 figure includes all candidate evaluations within the stated per-run training
allocation, including candidates that are later discarded. It excludes engineering, the
192-episode diagnostic, the 320-episode probe, reference evaluations, any bounded optimizer
pilot, additional Jar training, validation, reserved testing, transfer, instruction probes,
additional ablations and exploration, and failed attempts outside that allocation. Jar and
other families have different episode costs. All figures are planning arithmetic, not a
newly approved total or measured future throughput.

Before main-study execution, cost every stage explicitly in episode attempts, simulator
actions, physics steps where relevant, policy/model queries, GPU allocation time, wall
time and rental cost. Include candidate re-evaluation, normalization or model training,
hyperparameter selection and failed or voided attempts. Do not double-count overlapping
worker and launcher timers. Preserve an explicit boundary between earlier shared costs
and phase-three costs, with both visible.

Equal episodes are not equal simulator interactions: cautious controllers may consume
the full horizon while successful ones terminate early. Declare the primary limiting
resource, complementary caps and learning-curve axes. Report actual actions, queries
and time alongside episode budgets. Do not shorten horizons selectively or discard
timeouts to make a method appear more efficient.

Specify phase-three failure and retry rules prospectively. The standing exception permits
only proven zero-interaction setup failures under its existing conditions and caps;
do not generalize historical explicit void approvals to new interaction-bearing crashes.
Preserve raw artifacts and costs for every attempt. Infrastructure failure is distinct
from valid task failure, and a comparison with invalid evidence must follow its declared
blocking or incompleteness rule.

If the full design exceeds available resources, reduce the number of research questions
and comparison arms before weakening the final evaluation. The recommended first priority
is a well-specified ES, matched random search, scalar ablation and matched ordinary
learning, with identity and the hand rule. NSGA-II needs a distinct trade-off question;
GP remains optional. This prioritization is a recommendation, not a recorded removal of
an arm or approval to enlarge the budget.

## 18. Sequence, artifacts and publication planning

Proceed by evidence rather than the original calendar estimates. Section 20 gives the
recommended inexpensive implementation and mechanism-screen route within this sequence:

1. Resolve the primary claim, shared-versus-separate controller choice, dial semantics,
   stochastic feasibility rule, ordinary-learning match and primary budget unit.
2. Locate and audit completed or partial replicate evidence before scheduling any new
   episodes. Respect the original protocol's stop time and allocation; work outside
   those limits requires a new prospective protocol rather than silently resuming it.
3. Freeze a bounded dial engineering and heritability-probe protocol, including costs,
   analysis and pass/fail/inconclusive decisions, then implement only the needed interface.
4. If warranted, run a separately bounded ES-versus-random pilot to test the complete
   optimization procedure and calibrate candidate-evaluation allocation.
5. Freeze the main method, comparison set, splits, hyperparameter procedures, total budget,
   failure rules, finalist selection and reserved analysis before the seed study.
6. Train and validate within those limits, freeze finalists, then open reserved scene and
   nominated transfer evaluations. Write the result with its actual scope and limitations.

New run paths and protocol identities must distinguish phase three from the paused pilot.
Retain source and asset hashes, environment versions, seeds, scheduling provenance,
proposed/adapted/executed action records, outcome evidence and complete interaction
ledgers. Continue the recorded private evidence-backup arrangement, pinning revisions
and verifying restoration, while keeping credentials, datasets, checkpoints and generated
run directories out of Git. Record any required setup against the intended separate
ManiGuard simulation and policy environments; root dependencies do not own those stacks.

The GECCO 2027 full-paper target remains. The 19 January abstract and 26 January paper
dates are provisional planning assumptions. On the 18 October review, the official
[call page](https://gecco-2027.sigevo.org/Call%2Bfor%2BPapers) still displayed 2026 dates.
Recheck actual 2027 dates and submission rules before using them as binding milestones.
Do not inherit the old mid-November thesis schedule as an experiment gate or promise.
See the dated [venue notes](docs/gecco2027.md).

## 19. Decisions required before the main protocol is frozen

- Primary claim and practically meaningful success and violation effects.
- One jointly optimized controller vector or separate family-specific vectors.
- Exact dial inputs, controls, bounds, composition, identity and mutation distribution.
- Stochastic feasibility comparator, risk interpretation, ties and uncertainty treatment.
- ES details, noisy elitism policy, scalar-penalty comparison and NSGA-II scope.
- Ordinary-learning representation, objective, information access and matching rationale.
- Inclusion or justified omission of a relevant external execution-adaptation baseline.
- Scene/family sampling, weighting, replication, finalist allocation and racing rules.
- Heritability-probe analysis, thresholds, anchors and inconclusive-result handling.
- Training, validation, reserved scenes, family-transfer scope and disclosure of exposure.
- Primary statistical contrasts, uncertainty procedure and safety-claim requirements.
- Stage-by-stage interaction and rental budget, limits, stopping rules and failure handling.
- Whether ROSARL or learned filtering has a defined role beyond the present core scope.

The supervisor discussion should settle whether penalty-free constraint handling is the
main contribution or a mechanism within the broader adaptation question, how much weight
Jar should receive relative to Clutter, which transfer evidence is necessary, and which
GECCO track best matches the final contribution. These choices should follow the evidence
and available budget, rather than a requirement to complete every proposed branch.

## 20. Recommended implementation and inexpensive first experiments

**Planning refinement, 18 October 2026.** The purpose of the first implementation is to
answer a small number of consequential questions cheaply. These are proposed steps and
ceilings, not a frozen run protocol, permission to extend an expired allocation, or new
experimental results. Do not implement the full optimizer comparison before establishing
a usable execution interface and evaluation procedure.

### 20.1 Reuse the validated boundary, with a smaller controller

The existing code already supplies an eight-dimensional absolute-joint command boundary,
robot observations, policy queries, action history, simulator accounting and separate
outcome telemetry. The useful starting points are:

- [maniguard_runtime.py](experiments/helpers/maniguard_runtime.py): proposal, applied-action,
  query, observation and accounting hooks.
- [maniguard_adaptation.py](experiments/helpers/maniguard_adaptation.py): causal robot-frame
  schema and existing pure execution mechanisms.
- [maniguard_phase2_outcomes.py](experiments/helpers/maniguard_phase2_outcomes.py): requested
  completion semantics, kept separate from runtime observations.
- [maniguard_motion_tracking.py](experiments/helpers/maniguard_motion_tracking.py) and
  [maniguard_release_audit.py](experiments/helpers/maniguard_release_audit.py): existing
  trace-based mechanism audits.
- [maniguard_replicate.py](experiments/helpers/maniguard_replicate.py): repeated-outcome
  diagnostic and retained-cost patterns.

The phase-two runtime loads its CLIP encoder and computes features even for fixed
controllers. Reusing fixed mode is therefore not automatically a minimal-compute path.
The initial phase-three adapter should consume robot state and command history directly,
without adding an encoder, a new VLA checkpoint, a learned critic or multiple proposal
samples per query. This removes adapter overhead, not the frozen VLA's own visual work.
Measure any actual saving; simulation was CPU bound on the recorded host.

For a small probe, engineering effort can exceed the value of removing encoder calls.
First reuse an existing validated entry point unchanged if it can express the nominated
comparison with the correct information and outcome contract. Do not rebuild the runtime
solely to remove CLIP before profiling its cost. A new integration becomes necessary
when the causal controls or semantic configuration are not expressible by the old runner;
the intended lean implementation below is for that case.

Use new versioned files, leaving the frozen helpers unchanged. A proposed minimal file
set is `experiments/helpers/maniguard_phase3_adapter.py` for pure controller logic,
`maniguard_phase3_runtime.py` and `maniguard_phase3_source.py` for the checked integration,
`maniguard_phase3_results.py` for the new job/trace audit, and
`experiments/scripts/maniguard_phase3_probe.py` for bounded probe execution and reporting.
Add configs under `configs/maniguard/phase3/` and only the focused tests needed to validate
the changed execution and evidence contracts. These files are proposed, not implemented.

The runtime has import-time study dispatch, so a new phase must not accidentally construct
a legacy study or open a second trace while importing reusable hooks. Resolve that
explicitly in the new integration and test it without simulator work first. Reuse
accounting and outcome semantics rather than blindly inheriting a complete frozen runner.
The phase-two result reader also checks its 692-weight actor, CLIP records and likelihoods;
it cannot certify a semantic dial unchanged. A phase-three reader must bind the semantic
configuration hash and verify its own causal decisions, source identity, complete monitor,
requested completion and cost ledger. Missing evidence remains invalid.

### 20.2 First priority: existing evidence, with no new simulator interaction

Locate the partial pilot, replicate coordinator/report and selected development traces on
the run machine or in the pinned private backups. Restore the smallest useful evidence
first, not the full simulator, model collection or video library. Reuse accepted diagnostic
episodes rather than repeating them. As inspected locally on 18 October, this checkout
does not contain `runs/maniguard/`, `data/maniguard/` or `third_party/ManiGuard`; no fresh
raw-trace analysis or simulator result is implied by this plan.

The first offline deliverable should describe actionable opportunities by episode:
proposal-to-measured-joint displacement, previous-input tracking error, open/close command
transitions, periods of low end-effector motion, query age, first violation and requested
completion. Use available object/goal labels only to diagnose whether the event is relevant,
never as inputs to the deployed controller. Include successful and unfinished traces to
identify triggers that would interfere with useful behaviour. A recording that lacks the
required state is unavailable evidence, not evidence of an absent event.

Inspect command caps against actual measured motion using the existing tracking audit.
The earlier findings already show that target-displacement bounds and physical motion
are different. The [Stack release study](docs/research/reports/ManiGuard%20Stack%20release%20trigger%20comparison.md)
also records 179 delayed opening commands while the goal target was unheld. Repeating
release-delay sweeps on that evidence is low priority. It does not establish that release
is irrelevant on Clutter or that other objects were unheld.

If repeated outcomes are available, replay the selection decision over the measured
controller/case grid: compare one, two and four observations per candidate where the data
permit, keeping some repeats independent of selection. Quantify unstable rankings and
the benefit of spending evaluation on finalists. Use scene-aware resampling and retain
the limited number of controllers and replicates in the interpretation. This can inform
screening allocation; it cannot evaluate unseen mutants, simulate a complete adaptive EA,
or establish optimizer superiority.

An offline controller replay can establish that a trigger fires, when a command would
change, or whether identity is exact on the same inputs. Once commands or query timing
change, the future recorded trajectory is no longer the counterfactual trajectory.
Offline replay cannot establish improved physical outcomes. Count CPU work separately
and charge zero new simulator interactions only when no simulator is actually advanced.

### 20.3 Initial hypothesis: conditional command restraint and re-query timing

The cheapest candidate implementation to investigate jointly is an open-command motion
cap and a persistent-tracking-error query trigger. The reasoning is that abrupt proposed
movement can require restraint, while continuing stale proposals can require a fresh
observation-conditioned action. Their interaction might matter. This is a hypothesis to
check against the trace census, not a diagnosed failure mechanism or an established repair.

A proposed four-parameter vector is an open-command joint-target displacement cap, a
tracking-error threshold, a persistence duration, and a nominal query interval. Identity
disables the cap and event trigger and uses the original eight-action query procedure.
For the first mechanism screen, hold the nominal interval at eight and freeze one
trace-informed setting for the remaining parameters before observing new outcomes.
Explicit disabled values must be serializable and distinguishable from active bounds.

Define tracking error using the previous command actually supplied to the environment
and the current measured joints. The displacement to the next proposed target is demand,
not automatically tracking error. An error persisting for several control actions can
make the query trigger eligible, subject to a fixed minimum query interval. It is not
a detector of task stagnation or failed grasping. Validate the query cooldown and short
chunk handling; avoid a trigger that requests the same ineffective action indefinitely.

For an eligible motion cap, bound the seven proposed joint targets relative to the
current measured joints, preserving gripper commands in this initial experiment. Log
both the unmodified proposal and bounded environment input. Apply no cap in the disabled
state, and do not interpret an open gripper command as a simulator contact or jar-state
label. Freeze action normalization, units, decision frequency and composition order.

Defer closed-command caps, smoothing, release holds, low-height conditions and geometric
lift attenuation until evidence identifies an additional need. Low-height controls need
an audited coordinate/reference-surface convention; selective lift attenuation needs a
validated geometric mapping. A new encoder, world model, sampler modification or sub-goal
language program is not needed for this first probe. If the census does not support the
proposed pair, nominate a different pair explicitly before writing a run protocol.

### 20.4 First physical screen: a small factorial comparison

After offline checks, the proposed engineering allocation is two short initialization and
identity checks of at most 32 control actions each. They validate integration, not task
completion. Initialization and other physics still belong in the ledger. Define separate
wall-time, query and GPU-allocation limits before running.

The proposed first full-episode allocation is two Clutter development scenes, four
replicates per scene and four conditions: identity, conditional cap only, event-triggered
re-query only, and both controls together. This is **32 full episodes**, at most 35,200
policy control actions at the existing 1,100-action horizon, plus engineering checks and
all other attempted interaction. Keep the original horizon and native early termination.
Choose development scenes prospectively, not reserved scenes or cases selected because
one candidate already won. Interleave the four conditions within scene/replicate blocks
and balance available workers. Same-seed episodes still have residual variability.

This design measures each single control and whether their combination is worth further
investigation. Report all four joint outcome categories, counts per scene, query use,
activation times and changed commands. The descriptive interaction contrast is the
combination's rate minus the two single-control rates plus the identity rate; eight
episodes per condition give it substantial uncertainty. It is an engineering comparison,
not a powered safety or optimizer-superiority result.

If a control never activates, explain the missing opportunity or implementation mismatch
before more episodes. If behaviour changes but only produces waiting or extra queries,
do not call that progress. An isolated extra completion is a lead, not a repeatable gain.
If one simple control is sufficient, retain it as a strong reference. If there is credible
activation and possible headroom but no clear rate gain, a bounded search may still be
justified; this screen does not restore the old requirement that a hand-written repair
must win before optimization.

Before making a repeatability claim, a possible second allocation repeats the same four
conditions on two additional development scenes, again four times each: **32 further
episodes**, for 64 full screen episodes in total. Freeze the controls before this second
allocation; changing them creates another exploratory study whose costs remain visible.
This is conditional follow-up, not automatic permission for 64 episodes. A small effect
may remain inconclusive at this budget; do not extend sampling until significance appears.

### 20.5 How the inexpensive evidence changes the trajectory

If raw evidence is missing, first retrieve and audit it. If infrastructure or source
identity is wrong, repair that before interpreting controller outcomes. If the proposed
triggers have no relevant opportunities, remove them from the dial and investigate the
observed failure rather than adding search. If the interface visibly changes relevant
behaviour, use the heritability probe to measure variation and local structure. A useful
fixed control is helpful evidence and a baseline, not the sole route to that probe.

The 320-episode parent-offspring design remains a proposal to cost and freeze after these
checks. It need not be launched as the first experiment. One possible staged allocation
starts with eight parent-offspring pairs, two Clutter scenes and two replicates each:
**64 episodes**. Extending those same prospectively sampled controllers to the full
16-pair, eight-Clutter-plus-two-Jar design can total 320, rather than 64 plus a fresh 320.
Preserve the initial observations and identities. Any intermediate look is descriptive
unless a valid sequential decision rule is frozen; it cannot become an uncorrected
early statistical pass. Specify how fail and inconclusive decisions are handled before
execution. Jar costs and any reference anchors remain explicit.

Only after useful variation and evaluation precision are understood should a small
ES/random-search comparison test the complete adaptation procedure. For scale, two
methods, two search seeds, two evaluated populations, four candidates and four rollouts
per candidate total **128 training episodes**, before independent finalist evaluation.
This gives only one evolutionary transition per run and tests basic search behaviour;
it is not powered evidence of eventual EA superiority. Choose population, generations
and replication from the measured signal, rather than treating this arithmetic as a
frozen optimizer budget. A tiny pilot's null does not prove that longer adaptation fails.

At the recorded planning throughput, 32 Clutter episodes are approximately 33 minutes of
eight-GPU workload-equivalent wall time; 64 are about 66 minutes. These calculations use
1,400 episodes per day and exclude restoration, new integration, queues, changed query
cost and different host/load behaviour. They are not current throughput measurements or
promises. Actual rental accounting includes every allocated GPU, including idle time.

The first implementation order is therefore the evidence audit, pure four-parameter
controller, checked no-encoder runtime and result reader, bounded mechanism screen,
then the appropriate heritability and optimizer pilot. Add PPO, NSGA-II, external
adaptation baselines and further task coverage when the shared execution and outcome
interfaces are stable. They remain necessary for the corresponding final claims, but
they are not dependencies of this initial mechanism test. GP, additional families and
new learned models require separate evidence and budgets rather than automatic expansion.

The existing 192-episode replicate diagnostic and any completed work retain their
original cost scope. The 32/64 screen, 64-to-320 heritability allocation and 128-episode
pilot are distinct proposed allocations; do not add or spend them as one automatically
approved package. A new protocol must include all references, unsuccessful candidates,
re-evaluations, initialization, failures, queries and elapsed time, with an explicit
decision at each allocation boundary.

## 21. Candidate ManiGuard evolution pilot on one GPU

The approved recommendation is expanded into a standalone implementation and experiment
plan in [phase3Part1.md](phase3Part1.md). Its precise Part 1 definitions refine this
initial sketch; the full-study protocol and historical allocations remain separate.

**Clarification from Sachin, 18 October 2026.** Keep ManiGuard and include actual
physical evolutionary search. The earlier Push-T/iCEM suggestion misunderstood the
analogy and is withdrawn. The goal is a small executable experiment with an interpretable
result, not the complete multi-method GECCO study. No run has been launched.

### 21.1 One concrete question

Can a small evolutionary search find execution parameters that improve a frozen
Clutter policy's safe requested completion on fresh episodes, with a user-controlled
safety dial to vary the strength of adaptation?

Clutter is the recommended first family because its recorded pilot includes both safe
completions and failures, and its native horizon is 1,100 actions rather than Jar's 2,000
or Stack's 4,200. This is observed headroom, not proof that execution parameters can
repair the failures. The earlier six-hypothesis study did not establish a repeatable
adapter improvement. The pilot should test that possibility without assuming it.

Use the released, pinned Clutter VLA, unchanged simulator and monitor, and one serial
simulation worker on the same GPU as the policy server. The monitor evaluates physical
outcomes; it does not supply a predictive action-changing filter. Keep privileged object
states and monitor labels out of the execution adapter. Do not train the VLA, a world
model, an encoder or a new safety predictor.

### 21.2 Reuse the existing adapter and evolutionary machinery

Start with the existing `motion` adapter and four parameters: free-motion joint-target
displacement cap, grasp-conditioned displacement cap, smoothing coefficient and policy
execution horizon. These are existing fields in
[maniguard_adaptation.py](experiments/helpers/maniguard_adaptation.py), not a new neural
controller. Grasp conditioning uses the existing finger-based heuristic and proposed
closing command, not privileged target-held state. Fix the other thresholds and settling
behaviour. Freeze finite domains after checking controller units, supported chunk lengths
and a short physical smoke test. The domain must include permissive settings so severe
slowing is not forced on every candidate.

Part 1 adds a user dial `d` with settings zero, one-half and one. Zero reproduces the
original executor; one applies the evolved controller fully; one-half blends the arm
adaptation and query horizon. Search runs at one, then all three settings use the same
frozen anchor for the fresh comparison. This measures intervention strength and its
physical safety/completion effects, not a calibrated failure probability or a guaranteed
monotonic safety response. Exact semantics are specified in [Part 1](phase3Part1.md#44-the-user-controlled-safety-dial).

Use the existing categorical evolutionary implementation as the starting point: a
population of four, three evaluated populations, elitism and mutation. This gives an
initial population and two actual evolutionary transitions. Re-evaluate the retained
elite rather than reusing its previous fitness. It is one optimizer seed, not a repeated
optimizer-superiority experiment. Score against measured requested completion and raw
physical violations, with a prospectively fixed selection rule. Part 1 uses the current
contract's `S_req - 2*V` utility; it does not test penalty-free selection. Preserve the
four joint outcomes separately.

The small launcher should reuse audited search and result logic, not launch the old
phase-two coordinator or alter its frozen allocation. The earlier search validator
requires all three optimizers and its legacy source map lacks Clutter; it is not an
unchanged drop-in entry point for this proposal. Use the pinned phase-two Clutter source
metadata and current completion contract. A new pilot protocol must explicitly support
one method and the semantic adapter. Do not relax the old study's checks globally.

### 21.3 Allocate 24 episodes to search and 24 to checking the result

The proposed full allocation is:

- Search: four candidates, three evaluated populations and two training scenes per
  candidate, for **24 full episodes**. Count elite re-evaluations and failed attempts.
- Independent check: freeze one finalist using search data only, then compare dial
  settings zero, one-half and one on two validation scenes with four repetitions per
  scene, for **24 further full episodes**, eight per setting. Interleave settings within
  scene/repetition blocks. Exclude these scenes from controller selection and tuning.

This totals **48 full episodes**, at most 52,800 policy control actions at the original
horizon, plus initialization, short checks, failures and all other physical interaction.
Use frozen training and validation partitions; keep the reserved main-study scenes
untouched. Repeated simulator roots still have rendering and policy variability.

Prioritize the independent identity comparison over adding several tiny optimizer arms.
This first pilot asks whether the discovered controller is useful. It cannot show that
evolution is more sample-efficient than random search or ordinary learning. A matched
random-search comparison is the next requirement if a promising controller is found;
it is not silently included in this allocation or optional for a later evolutionary
advantage claim. Existing fixed controls remain relevant follow-up baselines too.

Report safe requested completion, unsafe completion, safe unfinished and unsafe unfinished
counts per scene and condition. Include actual control steps, physical interactions,
policy queries and elapsed time, plus intervention frequency and matched videos. Reduced
violations through inactivity are not success. Eight episodes per setting give a
noisy exploratory estimate, not a safety guarantee or a conclusive generalization result.
Select one finalist before checking it and retain null and adverse outcomes.

### 21.4 Single-GPU feasibility and enforced eight-hour ceiling

Historical serial Clutter processes took about 177 to 191 seconds in the early pilot;
under the later eight-way load the recorded median was 417 seconds. At three to seven
minutes per episode, 48 episodes represent approximately **2.4 to 5.6 hours of serial
rollout time**. This arithmetic is not a 4090/5090 benchmark or a guarantee: the hosts,
load, setup and policy-query schedules differ. The historical passive total-memory
sample was 14.3 GB, supporting a single-24-GB-GPU feasibility check, not proving that
memory will stay below that value. Use one policy and one simulator; verify actual
peak memory, renderer configuration, device compatibility and throughput on the run host.

Prepare source and the intended ManiGuard environments before starting the timed pilot
where possible. A preparation template is not an installed research stack. Include all
on-machine setup, initialization, smoke tests, idle allocation and failures in the
single-GPU eight-hour ledger. Do not promise a cold installation plus a complete study
inside eight hours before a real preflight.

Reserve time for setup and final auditing, then choose the episode allocation from
conservative end-to-end timing alone, before viewing comparative outcomes. If 48 episodes
will not fit, the predefined smaller pilot is four candidates, two populations and two
training scenes (**16 search episodes**), followed by the three dial settings on two
validation scenes and two repetitions (**12 checking episodes**), for **28 full episodes**.
This has one evolutionary transition and narrower scope. Do not shorten the
native task horizon merely to claim throughput or claim that incomplete rollouts failed
at the original horizon.

Use a wall-time deadline that stops new launches with enough time to save and audit
records, and an outer hard allocation stop at eight hours. Preserve and account for any
incomplete attempt. If the stack is not ready or measured timings do not support even
the smaller allocation, report that directly. No positive outcome or minimum completed
sample count is guaranteed by the deadline.

The tangible deliverable is a saved evolved parameter vector, search lineage, fresh
three-setting outcome counts, an interaction/time ledger and matched videos.
A useful gain justifies testing evolution against matched random search and identifying
which controller change caused the effect. A short null informs further spending but
does not establish that this interface or a longer search cannot work. Neither outcome
alone establishes a GECCO contribution.
