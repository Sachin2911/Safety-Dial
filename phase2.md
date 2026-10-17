# Phase 2: shared evolutionary adaptation for safe manipulation

**Original research plan: 17 October 2026.** Sachin endorsed moving from individual
ManiGuard repairs to a shared method with evidence across tasks. This document preserves
that direction and the proposed implementation and evaluation programme.

**Implementation checkpoint, 17 October 2026:** the shared controller and supporting
runtime are implemented. Ten short engineering attempts have been consumed: eight valid
Jar/Stack checks, one valid Clutter original-executor check and one failed Clutter identity
initialization. The failure occurred before policy actions and remains invalid with its
costs retained. No full preparation episode, comparative pilot, learning result or reserved
evaluation has run. Clutter's policy startup is repaired; encoder-version provenance and
a bounded recovery within the preparation allocation are being resolved. The [development log](docs/research/research_notes/ManiGuard%20phase2%20development.md)
and [requirements ledger](docs/research/research_notes/ManiGuard%20phase2%20requirements.json)
record the current evidence and limitations. Preparation assignments are separately
frozen; the pilot and main-study numerical budgets below remain planning defaults to
cost and freeze before those evaluations.

## 1. Research objective and intended contribution

> Can evolution discover a compact, history-dependent execution controller that improves
> safe completion of frozen manipulation policies across tasks, and transfers to task
> families excluded from adapter development or optimization?

The object we share is **one adapter with one parameter vector**, operating through the
same observation and action interface. The benchmark may use different frozen VLA
checkpoints for different families. This is not a claim that we train one generalist VLA.
Task instructions can provide context; family-specific branches, parameter tables and
hand-written prerequisite rules are excluded from the primary adapter.

The desired contribution has three independently testable parts:

| Claim | Evidence needed |
| --- | --- |
| Shared adaptation helps | One controller improves safe completion across distinct families on reserved scenes, beyond fixed settings and a simple contextual rule. |
| The adaptation transfers | The frozen controller works on a family excluded from optimization; the stronger test also excludes that family's behavioral outcomes from method development. |
| Evolution adds value | Evolution improves the completion/safety trade-off or interaction efficiency relative to matched random search and ordinary learning over the same controller interface. |

Improvement on every family is an ambition, not a promised result. Report all family
outcomes, including losses. A pooled gain cannot conceal a regression on a hard family.
A shared-controller improvement without an evolutionary advantage remains a behavioral
result, but it does not establish the intended evolutionary contribution.

## 2. What changes after phase 1

The [completed six-hypothesis study](docs/research/reports/ManiGuard%20six-hypothesis%20study.md)
contains 196 full simulator episodes. Jar guarding was the strongest descriptive lead,
but follow-ups did not establish a repeatable useful correction. No ManiGuard optimization
or reserved evaluation ran. These are simulator results, not physical-robot validation.

Retain its validated interfaces, traces, accounting and failure analyses. Preserve original
configs, splits and results. Jar becomes one diagnostic case within a cross-task study;
polishing its closure guard is no longer the central milestone.

For this phase, replace the screening requirement in [fruits.md](fruits.md) that a useful
hand-written correction must precede optimization. A bounded search can discover
combinations that isolated rules miss. Its entry requirements are valid measurements,
a credible behavioral interface, observable controller activation and a costed budget.
A successful script is useful evidence, not a mandatory prerequisite. This changes the
prospective plan; it does not reopen or reinterpret the completed study.

The six tested controls are not known to contain a general solution. If the adapter cannot
change the relevant failure, more search alone is unlikely to help. Missing grasping or
tool-use competence must remain distinguishable from poor execution or coordination.

## 3. Position relative to existing work

The contribution cannot simply be an evolutionary wrapper around a frozen VLA.
[VLA-Pilot](https://arxiv.org/html/2511.14178v2) already evolves action proposals online,
using multimodal reasoning to score them. Our proposed search runs offline over a reusable
controller, and deployment executes that controller without population search at every
action. Whether this distinction plus safety and transfer evidence is sufficient novelty
remains to be established.

| Relevant primary source | Overlap to address |
| --- | --- |
| [Adaptive Action Chunking](https://arxiv.org/abs/2604.04161) | Adaptive chunk duration already has a task-performance literature; it is a comparison ingredient. |
| [CheckVLA](https://arxiv.org/abs/2607.26789) | Action-conditioned verification, intervention and progress retention already motivate execution-time feedback. |
| [Mostly Harmless VLA Steering](https://arxiv.org/abs/2606.12299) | Learned feedback and intervention selection around frozen VLAs already exist; its performance-harm criterion is different from physical safety. |
| [SEAM](https://arxiv.org/abs/2607.04609) | Improving consistency between action chunks already has dedicated methods. |

Before the main study, compare representation, runtime information, training interactions,
deployment cost, physical-safety objectives and transfer tests. Select at least one
compatible execution-adaptation baseline from this literature. Pin its implementation
and explain changes for the joint-action interface. Do not claim a prior method was beaten
if we implemented only a loose approximation. A new EA is not required, but substituting
an optimizer is insufficient by itself.

## 4. Scope: safe execution and learned filtering

The core study optimizes observed safe execution around a frozen VLA. ManiGuard's runtime
monitor observes violations; it is not an action-changing predictive filter.
[ManiGuard formulation](https://arxiv.org/html/2608.17386v1#S3).

This phase broadens the earlier [concept note](docs/researchConceptNote/concept-note.tex)
and [oracle](oracle.md) direction. It does not assume that continuation under a learned
filter has been demonstrated. A new world model or safety-model training-data selection
project is not a dependency of the core pilot.

If learned filtering becomes part of the eventual claim, specify a separate extension:
a validated predictive filter held fixed across methods, adapter/no-adapter comparisons
with and without that filter, and audits of prediction error and progress after
intervention. Include acquisition/training costs and information access. A simulator-derived
diagnostic guard is privileged. Neither a fixed filter nor a good fitness score establishes
unchanged physical risk.

## 5. Family coverage and held-out design

The six families and full horizons are verified against the pinned upstream pipeline
inventory and joint-evaluation configs, referenced by
[the phase-two source manifest](configs/maniguard/phase2/sources.json). Assets and
checkpoints are local for Jar, Stack, Lid and Clutter. Phase-two physical engineering
checks currently cover Jar and Stack only; metadata preparation is not behavioral validation.

| Family | Base tasks | Full action horizon | Current local state | Proposed role |
| --- | ---: | ---: | --- | --- |
| Jar transport | 26 | 2,000 | Assets/policy; four valid phase-two short checks | Development and pilot |
| Stack retrieve | 28 | 4,200 | Assets/policy; four valid phase-two short checks | Development and pilot |
| Lid transport | 30 | 1,900 | Phase-one assets/policy and evaluated runner; no phase-two run | Existing development evidence; later cross-family check |
| Clutter pickup | 55 | 1,100 | Policy verified; original executor checked; encoder initialization repair | Add to pilot for a different task structure |
| Cabinet pickup | 35 | 5,700 | Metadata and policy pin recorded; local assets not prepared | Reserve for transfer |
| Dusty transfer | 26 | 5,100 | Metadata and policy pin recorded; local assets not prepared | Reserve for transfer |

Use Jar, Stack and Clutter for the first joint pilot. Lid has already informed the design;
excluding it from pilot training does not make it an untouched research family. Its
evaluation can measure transfer from the optimized adapter, with that limitation stated.
Cabinet and Dusty are nominated now, before new local behavioral profiling, as the stronger
transfer families. Preserve their outcomes for a frozen-method evaluation.

Check transfer-family asset availability, schemas and checkpoint compatibility without
using task outcomes to tune the method. Log any necessary action-bearing integration
checks separately. If their traces inform features or controller design, disclose the
exposure and narrow the claim. If a family is unavailable, record the infrastructure
limitation rather than replace it because of a disappointing score. Final coverage should
include all six families.

Preserve [the phase-one scene split](configs/maniguard/splits.json) exactly:

| Family | Engineering | Training | Validation | Reserved test |
| --- | ---: | ---: | ---: | ---: |
| Jar | 4 | 6 | 4 | 12 |
| Stack | 4 | 8 | 4 | 12 |
| Lid | 4 | 10 | 4 | 12 |

Create a versioned phase-two manifest referencing these assignments and adding new families
by a deterministic, predeclared procedure. Group every base scene with all its language,
target, location and environment variants. Engineering data informs design; training drives
optimization; validation selects finalists; reserved tests open after the method and
analysis freeze. Fit normalization/projections only on permitted development/training data.

Distinguish new scenes, whole-family transfer and observation shifts. Test base scenes
first; predeclare shift axes before examining their outcomes. Adapter-held-out scenes or
families may have been seen during VLA pretraining/fine-tuning. After transfer results are
inspected, changes require a new evaluation partition or an explicit exploratory label.

## 6. Proposed shared controller

### Observation contract

Use the same extractor, normalization and controller weights for every family. Allow joint
and finger state, end-effector motion, recent action proposals and environment inputs,
tracking error, chunk age, previous adapter choices, and image/task context. Only past and
present information is available. Environment input is distinct from a lower-level actuator
command; current traces log the former.

A common frozen visual/text encoder is the proposed context source. Select and pin it
during the interface audit; the current adapter does not expose these features. Do not
combine unaligned hidden vectors from different VLA checkpoints. Use a fixed, documented
feature projection and account for encoding latency. Task text is allowed; family IDs,
scene names, checkpoint IDs and hand-coded family detectors are not inputs. Simulator
object states, goal flags, violation labels and monitor states remain evaluation/training
labels in the primary observation-limited track.

A privileged diagnostic track must explicitly match information across its methods.
Robot-state-only input is an ablation, not a silent replacement for the proposed
visual/task-conditioned method.

### Starting representation and decisions

The recommended first representation is a fixed-topology recurrent network with eight
hidden units and small categorical output heads. Target fewer than 2,000 trainable
parameters after projection; freeze the exact count before optimization. Evolve its
weights, not the VLA or encoder. This allows an ordinary policy-gradient learner to use
the same observation-conditioned policy class. A finite-state program is an alternative
if necessary, but switching representation creates a new protocol.

Evaluate the adapter every four control actions. Reset memory and buffers each episode.
It chooses from a bounded, composable interface:

| Decision | Proposed initial domain |
| --- | --- |
| Requery | Follow the baseline eight-action schedule; request a new chunk now; or continue the current chunk up to sixteen actions |
| Joint-target adjustment | Identity, or a displacement cap relative to measured joints, initially 0.03 or 0.06 radians |
| Smoothing | Identity or exponential smoothing with coefficient 0.75 or 0.5 |
| Gripper opening | Follow the proposal or delay opening by at most two or four control actions |

These domains are hypotheses, not validated useful settings. All heads must support
identity behavior. Freeze transformation order, first-action initialization, query timing,
gripper semantics and delay resets before running. A delay cannot renew indefinitely.
Query interval is at least four actions, apart from explicitly logged short-chunk
exhaustion. Limit accessible proposals to a common sixteen-action prefix; never pad
missing actions or access future policy queries. Any hard limits remain fixed across
methods and are not optimized as safety definitions.

Caps constrain commanded target displacement, not actual joint velocity or safety. Test
interactions between smoothing, caps and gripper handling: phase one's modes do not
automatically compose correctly. Freeze feature freshness and encoder scheduling across
methods. Preserve each family's pinned camera, grasp mode and controller conventions.

## 7. Evolution and comparisons

Start with a small evolution strategy using selection and Gaussian mutations of shared
controller weights, with layer-scaled mutation magnitudes. Include identity in the initial
evaluation panel, retain an archive and reevaluate survivors on the current panel. Eight
candidate evaluation slots are available per pilot round; elite reevaluation consumes a
slot. Freeze parent selection, mutation schedule and initialization before training.
Architecture search is outside the first pilot.

Construct an explicit identity genotype: constant output heads select baseline querying,
identity motion and immediate proposed gripper behavior. Verify exact equivalence to the
unchanged executor. Evolution and random search share an initial panel containing that
same genotype plus seven samples from a fixed initialization distribution. PPO starts
from that identity-biased actor; its finite logits permit exploration in training, while
its deterministic initial decoding is identity. Freeze the bias and exploration settings.
Before launch, use observation fixtures to verify that sampled/mutated controllers can
make context-dependent decisions rather than always choose one constant mode.

Required comparisons:

1. Unchanged VLA execution with pinned benchmark settings.
2. A fixed adapter and one simple contextual rule, selected on training data only.
3. Random search over the same recurrent-controller weights and initialization distribution
   as evolution, with the same candidate evaluation allocation.
4. Evolution over that controller.
5. Recurrent policy-gradient learning, initially PPO, with the same actor architecture,
   observations, categorical actions, action bounds and outcome reward. A training critic
   is allowed, without a privileged-information advantage.

The existing categorical REINFORCE samples static parameters for an episode group. It is
not the proposed observation-conditioned learner. Retain it only as a labelled
parameter-search baseline if useful. Implementing the recurrent learner is new work.
Do not discretize neural weights into the old Cartesian optimizer; its finalist enumeration
will not scale.

Evolution and random search share the initial candidate panel and scene/seed schedule.
The ordinary learner receives the same full-episode allocation and balanced family
exposure, not a fictitious population budget. Match feature access and horizons.
Use the undiscounted episodic objective (gamma = 1), document advantage estimation, and
weight episode/family contributions consistently with macro utility. Do not substitute
length-discounted terminal rewards or let pooled timestep sampling silently reweight the
families. The within-episode policy-gradient contributions must still represent the
complete trajectory objective. Report
realized physical steps, model/encoder queries, learner updates, memory and wall time.
Equal episode allocations do not imply equal compute or interactions; also plot performance
against cumulative simulator actions and queries. Count tuning, discarded candidates,
failures and retries. Shared physical records count once in actual expenditure and their
reuse is disclosed in each method's allocation.

Use fixed deterministic decoding for evaluation. If ordinary learning samples actions in
training, retain that distinction and evaluate its frozen actor with the same decoding
as the evolved actor. Add a compatible literature baseline for the main study after review;
the pilot is not a state-of-the-art comparison.

## 8. Outcomes, objective and selection

Retain all four original benchmark outcome categories. Additionally define C_req as
attainment of the requested task sequence under a written family-specific completion
checklist, and V as the unchanged benchmark physical-violation indicator. Audit checklists
against the pinned instructions/evaluator before optimization. For example, spatial goal
satisfaction without jar closure is not a completed close-then-carry sequence. Do not
modify the monitor or retroactively relabel phase one.

The supplemental verified safe-completion indicator is S_req = C_req * (1 - V).
Report it alongside benchmark safe success and explain discrepancies. A completion
checklist is an evaluation definition, not a runtime rule. Do not assume terminal
acceptance requirements: verify the pinned temporal semantics and document any stronger
research criterion.

The proposed pilot utility is U = S_req - 2 * V, averaged within each family and then
equally across families. Verified safe completion scores 1, safe noncompletion 0, and
either violating outcome -2. This is a new versioned objective; phase one's
completion-minus-two-violations code is not silently reused. The coefficient is a training
preference, not a safety guarantee. All optimizers use the same utility. Start without
dense task-specific progress rewards; if outcomes are too sparse, record that finding
before introducing a new shared shaping protocol.

Report per-family completion, violations, unfinished episodes, engagement, queries and
time. Include macro averages, the weakest family and every regression. Action changes or
fewer interventions are diagnostics. An invalid monitor or crash is not a safe timeout:
pause affected comparisons, preserve costs/status and repair infrastructure consistently.

Nominate exactly two finalists per run from training evidence only. Select on validation
using family-macro utility, with lower violation rate and then lower query cost as fixed
tie breakers. Report the trade-off even if selection favors higher risk. A claim of
improvement without increased risk requires a separately predeclared statistical safety
comparison in the main study.

## 9. Bounded pilot and decision criteria

The pilot uses Jar, Stack and Clutter. It tests feasibility, not definitive generalization
or evolutionary superiority.

| Item | Allocation | Full episodes |
| --- | --- | ---: |
| Evolution training | 8 candidate slots x 3 rounds x 3 families x 2 search seeds | 144 |
| Random-search training | Same schedule and allocation | 144 |
| Recurrent learning | 72 episodes/run x 2 training seeds, balanced across families | 144 |
| Finalist validation | 2 finalists x 6 runs x 3 families x 2 validation scenes | 72 |
| Reference validation | Identity, fixed adapter and contextual rule on the same 6 cases | 18 |
| **Optimization/selection subtotal** | No reserved test or transfer outcomes | **522** |

Use one prespecified training scene/seed per family per round, rotating across three
panels. All candidates in a round see the same panel. Reevaluate elites; do not rank
historical scores from unequal panels as equivalent. In the final round, each search
method uses two of its eight slots to reevaluate its within-panel winner from each earlier
round; the other six slots follow its search rule. Declare duplicate handling in advance.
This lets random search retain promising earlier draws without comparing incompatible
scores. Take the two search finalists from the final common panel.

The ordinary learner gets 24 episodes on each of the three panels, eight per family,
matching search-method scene/seed exposure. Nominate checkpoints after 48 and 72 episodes.
This checkpoint selection differs from search-candidate nomination and must be reported.
Every finalist/reference receives the same two validation base scenes per family and one
fixed policy seed per scene. Six validation cases are insufficient for a strong claim.

The subtotal excludes integration checks, preparation baselines and training used to
select the fixed/contextual references. Allocate a separate preparation ceiling of 24
full episodes and 12 short checks of at most 32 policy actions. Freeze purposes and
assignments before use; reuse phase-one engineering evidence where appropriate.
The proposed initial ceiling is therefore **546 full episodes plus those short checks**.
Count initialization, failed work and inference diagnostics separately; infrastructure
failure does not create unlimited free retries.

The 522-episode subtotal permits at most 1,270,200 policy actions at the declared
family horizons, before preparation and initialization. Before scheduling, calculate the
full action/query envelope, measure preparation throughput and record a GPU-hour ceiling
and wall-time estimate.
Episode-duration sums are not elapsed wall time. This allocation is not a dollar estimate
or an automatic instruction to launch the workload.

After this one pilot:

- **Proceed to a larger study** if a shared learned controller improves verified safe
  completion over identity on at least two development families in aggregate validation
  counts, utility improves, and the gain is not entirely from one training seed or a
  completion-label loophole. Inspect violation changes for every family. This is a
  screening rule, not statistical proof or a claim of equal risk.
- **Continue adaptation research without an EA claim** if random search or ordinary
  learning performs similarly. Examine sample efficiency and mechanism before expansion.
- **Treat completion gains with more violations as a trade-off**, requiring an explicit
  decision and a revised objective if pursued, not an unqualified safety improvement.
- **Reconsider the interface** if decisions activate but no optimizer improves useful
  completion. Investigate skill deficits, observation quality and control authority.
  Failed finite search does not establish irrecoverability.
- **Repair the experiment** if execution, outcomes, splits or accounting fail validation.

Additional Jar-only rescue batches do not follow automatically. Extensions get a new
protocol, budget and justification based on the completed pilot.

## 10. Main study and transfer evaluation

Freeze a costed main protocol before opening reserved data. Increase training diversity
and independent optimizer seeds; two pilot seeds cannot support an EA-superiority claim.
Choose sample sizes from development variability, a meaningful completion effect and an
explicit violation-risk margin, respecting the number of independent scenes. Record the
precision the available budget can support.

Evaluate one selected adapter per independent training run on the same reserved cases.
Include new scenes in development families, Lid as training-excluded but design-exposed
transfer where applicable, and nominated Cabinet/Dusty transfer families. Keep controller,
preprocessing and deployment rules frozen. Report all six families even if transfer fails.
A later all-family training experiment is separate and cannot retain an untouched transfer
claim for those families.

Use uncertainty estimates clustered by base scene, retaining optimizer-seed variability
and correlations between variants. Matched nominal seeds do not establish identical
physical starts. Not detecting more violations is not evidence of unchanged risk; zero
events in a small sample is not zero risk. If the budget cannot resolve the chosen safety
margin, report an inconclusive safety comparison.

Prioritize history (recurrent versus memoryless), shared versus separately optimized
adapters, and contextual versus constant execution ablations. Then test visual/task
context and individual action heads if the main finding justifies their cost. Match
interaction budgets for efficiency comparisons and disclose the total additional cost
of per-family training. A second backbone or real-robot evaluation is an extension,
not a hidden prerequisite of the first study.

## 11. Milestones and implementation work

| Milestone | Deliverable | Exit condition |
| --- | --- | --- |
| A. Claim and protocol | Related-work matrix; family/exposure registry; scene manifest; completion checklist; cost estimate | Claim, reserved families, measurements and ceilings written before new outcomes |
| B. Shared execution | Controller/features, common encoder, Clutter integration and pinned assets | Valid observations/actions, episode isolation and measured overhead; no task-specific branches |
| C. Matched training | Continuous-weight search, recurrent learner and family coordinator | Same actor interface, balanced schedules and audited costs |
| D. Bounded pilot | All training/validation records and decision report | Explicit proceed/revise/stop outcome, including negative results |
| E. Main evaluation | Frozen finalists, scene/family transfer results, ablations and uncertainty | Claims match breadth, safety trade-offs and optimizer comparisons |

Reuse audited episode execution, policy sampling isolation, source hashing, outcome
reports and ledgers from [the existing stack](configs/maniguard/README.md).
The current search and batch launchers assume one family; add a coordinator over family
sub-batches with verified matching policy servers and equal-weight candidate aggregation.

Jar/Stack use pi0.5 while Lid uses pi0, with different predicted chunk lengths. Their common
joint interface does not align hidden representations. The current Frame exposes no
visual/text features. New downloads, all-family preparation, feature extraction, composable
learned decisions and multi-family optimization are substantive implementation work.

Put reusable code under experiments/helpers/, entry points under experiments/scripts/,
versioned protocols under configs/maniguard/phase2/, and new outputs under
runs/maniguard/phase2/. Reports/audits belong under docs/research/. Store trained adapter
weights outside Git with pinned artifact references; keep code, configs and numerical
summaries in the repository.

Meaningful checks cover identity equivalence; recurrent resets and concurrent isolation;
no future/privileged feature leakage; composition order and bounded delays; checkpoint
routing; split enforcement; measured fitness reconstruction; and costs including failures.
Log proposed, adapted and environment-input actions, feature timestamps and decisions.
Validate state restoration before claiming identical-state branching. Use existing separate
simulator/policy environments and managed services; preserve old runs and environments.

## 12. Choices to freeze before the pilot

The accepted direction is shared adaptation with cross-task evidence. These implementation
choices remain proposed until milestones A/B resolve them:

- Common encoder, projection, recurrent architecture and exact parameter count.
- Action domains, composition rules and ordinary-learning settings.
- New-family checkpoint revisions, scene allocations and integration exposure.
- Completion checklists, objective implementation, mutation/initialization settings,
  training panels and finalist checkpoint schedule.
- Preparation allocation, measured resource estimate and GPU-hour ceiling.
- Closest compatible prior-method baseline and the distinction we will test.

The next deliverable is the shared-controller protocol and implemented interface, followed
by the bounded comparative pilot. A local repair, transfer across tasks and an evolutionary
advantage remain separate empirical outcomes throughout.
