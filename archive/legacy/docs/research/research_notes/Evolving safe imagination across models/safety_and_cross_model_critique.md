# Safe imagination: safety literature, model interfaces and scientific critique

Research cutoff: 7 October 2026. This is a critical literature note for the preferred
candidate, not an adopted method or implementation plan. Official papers and source code
were inspected; no models, checkpoints, environments or timing claims were tested.
Evolutionary algorithm discovery and general learned metareasoning are reviewed by the
other researchers. Novelty cannot be concluded from this note alone.

## Which safety and adaptive-planning papers constrain the idea?

### Takeaway

The components surrounding the proposal are already substantial research areas: safe
latent planning, delayed-cost reasoning, learned rollout stopping, uncertainty adaptation
and deciding when to reuse a plan. A useful study must identify a specific missing
capability and demonstrate its mechanism. Adding evolution or another backend is not
sufficient evidence that this capability is missing.

### Cited Findings

- **SafeDreamer, Huang et al., ICLR 2024.** The online variants use constrained CEM,
  mixing Gaussian action proposals with actor proposals and evaluating learned rewards,
  costs and critics. The feasibility estimate extrapolates a short imagined cost sum to
  episode length. OSRP-Lag adds long-term cost evaluation because a hazard can lie beyond
  the explicit rollout horizon; BSRP-Lag instead trains an actor through background
  imagination. Thus both safe latent planning and addressing delayed hazards are already
  explicit. Its constraint concerns expected accumulated cost, rather than a universal
  guarantee of no failures. The paper credits constrained CEM to Wen and Topcu (2018).
  [Paper, sections 3 to 5](https://arxiv.org/html/2307.07176v3);
  [official acceptance and code record](https://github.com/PKU-Alignment/SafeDreamer).

- **RISE, Lu et al., August 2026 preprint.** A latent evaluator predicts prefix risk and
  the planning benefit of further rollout. A gate weighs these predictions, depth and
  computational cost to choose Roll or Stop. Gate training uses supervised labels from
  evaluating all available depths, with the other modules fixed at that stage. CounterDrive
  supplies verified counterfactual risk examples. Section 5.4.3 integrates the scheduler
  into DAWN without modifying its Predictor or Planner and reports improved driving
  metrics. It does not explicitly establish that evaluator and gate weights transfer
  unchanged without training. Both systems concern driving. This is a direct precedent
  for safety-informed adaptive imagination and a reported second architecture.
  [Full text, sections 4.3, 4.4 and 5.4.3](https://arxiv.org/html/2608.20430).

- **RISE release caveat.** The linked repository currently describes a code/configuration
  release without trained weights or numerical results, and still calls the paper
  forthcoming. That README is inconsistent with the now-public manuscript. It does not
  resolve the DAWN freezing question. Treat reported experiments as author claims, not
  independently reproduced evidence.
  [Official repository](https://github.com/COOWAI/RISE).

- **When to Trust Imagination, Wang et al., May 2026 preprint.** FFDC-WAM verifies whether
  a robot should continue executing an action chunk by comparing imagined visual futures,
  current observations, actions and instruction information. A lightweight verifier
  triggers replanning. Supervision includes successful demonstrations/rollouts, failed
  rollouts and synthetic action corruption. Its evaluation studies manipulation success,
  completion time and model calls. This covers learned stopping of physical execution
  under prediction mismatch; it does not establish a general safety constraint or an
  evolved search-tree controller. The underlying WAM also receives mixture-of-horizon
  training, which matters when interpreting improvements.
  [Full text, sections 3 and 4](https://arxiv.org/html/2605.06222).

- **AdaReP, Cheng et al., ICANN 2026.** The September arXiv revision explicitly records
  acceptance. A training-free wrapper adapts replanning tolerance using deviation from a
  cached rollout and local sensitivity. It studies image-space predictors, TD-MPC2 latent
  control and a physical-state model for a Franka robot. The stated transfer protocol
  permits backbone-level tuning. Its local regret argument relies on perturbation and
  Lipschitz conditions; using latent discrepancies additionally needs a local relationship
  between representation and physical-state distances. This is a close baseline for
  learned choices to reuse, execute or replan, not a general safety guarantee.
  [Full text, sections 2 to 5 and Appendix B](https://arxiv.org/html/2606.23079).

- **Sparse Imagination, Chun et al., ICLR 2026.** The model is trained for token
  sparsification. Appendix B.6 already adapts token retention using changes in rollout
  variance, including preferential retention of high-variance tokens. The authors frame
  this as preliminary and do not claim superiority to the best fixed sparsity setting.
  It makes compute allocation inside prediction another established option, although
  its variance proxy is not a calibrated probability of physical failure.
  [Full text](https://arxiv.org/html/2506.01392);
  [official proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a750d52284ff70c6d6bab8072c392d74-Abstract-Conference.html).

- **UNISafe, CoRL 2025.** The paper's uncertainty mechanism uses a separately
  trained ensemble of latent predictors, rather than treating one RSSM's stochastic
  samples as epistemic uncertainty. Appendix A.3 explicitly warns that RSSM distribution
  variance is not itself calibrated aleatoric uncertainty. Its latent safety filtering
  is relevant inspiration and a check against casual uncertainty claims. The current
  proposal does not need to adopt its HJ or conformal machinery.
  [Full text, section 4 and Appendix A](https://arxiv.org/html/2505.00779).

- **When World Models Lie, Cao and Bansal, September 2026 preprint.** It adapts latent
  safety filtering to observed prediction errors and evaluates a safety value
  pessimistically over uncertainty sets. This directly addresses confidently wrong
  imagination. The theorem controls latent-error set coverage under its assumptions;
  it must not be described as universal collision avoidance. This is evidence that
  error-aware trust is active prior work, not a requirement to introduce HJ or conformal
  methods into this project.
  [Full text, sections III to V](https://arxiv.org/html/2609.34300).

### Inferences

**Safety-specific compute allocation needs a sharper objective than "imagine more near
danger".** An obviously unsafe option can be rejected with little computation. A seemingly
safe option with unresolved delayed consequences may deserve more. The useful quantity
is whether a query is likely to improve the pending decision, including safety and
progress, enough to justify its cost. This is our interpretation, not a verified novel
formulation; RISE and general metareasoning already occupy much of this territory.

Different operations resolve different limitations. Wider search may discover an
alternative; greater depth may expose delayed consequences; repeated stochastic rollouts
may reduce sampling error under a particular model; execution followed by observation
provides actual new environmental information. These should not be conflated. In
particular, repeatedly sampling the same biased model does not supply independent
evidence that the model is correct.

The intellectually defensible target is a discovered, inspectable rule whose decisions
matter under explicitly characterised conditions. It may turn out that a simple fixed or
handwritten adaptive rule captures the useful effect. That would narrow or reject the
algorithm-discovery motivation, rather than justify adding more components to preserve
the original story.

### Gaps

- Frozen cross-family scheduler transfer in RISE is unresolved. The manuscript and
  inspected repository do not specify it sufficiently to credit or deny that result.
- Neither published performance nor available code establishes that this project's
  eventual tasks contain an exploitable planning bottleneck.
- This targeted search cannot justify a claim that safety-aware evolutionary
  metacontrol has never been studied. The complementary algorithm-discovery review is
  essential, especially for GECCO work.

### Search log

Primary sources were opened directly, then searches varied through `safe adaptive
imagination world model`, `SafeDreamer uncertainty planning horizon`, `When to Trust
Imagination`, `When World Models Lie`, `Sparse Imagination ICLR 2026`, `adaptive horizon
safe world model`, and `AdaReP adaptive replanning`. Full method sections were inspected
for SafeDreamer, RISE, FFDC-WAM and AdaReP; selected appendices were inspected for
UNISafe, Sparse Imagination and ROSARL. LeWM and SafeDreamer official source files were
read separately. Secondary summaries supplied leads only. Searches stopped once
additional papers repeated already established mechanisms.

## Can LeWM and SafeDreamer support a common controller?

### Takeaway

An interface is technically plausible, but the models are not interchangeable
off-the-shelf simulators. The difficult work is defining what information and actions the
controller receives without hiding the contribution inside adapters. Compatibility,
re-evolution and frozen transfer require distinct claims.

### Cited Findings

| Interface issue | Verified source evidence |
| --- | --- |
| LeWM state | LeWM encodes images and uses action-conditioned latent prediction. The published predictor uses observation history, with history length three for PushT/Cube and one for TwoRoom. Its planner optimises latent goal distance; it does not supply an intrinsic safety meaning. [Paper](https://arxiv.org/html/2603.19312) |
| LeWM query mechanics | `JEPA.encode`, `predict` and `rollout` expose sequence tensors; rollout batches candidate plans and maintains embedding/action history. `criterion` scores the final predicted embedding against the goal. These are usable building blocks, not an existing generic branch/extend safety API. [Official jepa.py](https://raw.githubusercontent.com/lucas-maes/le-wm/main/jepa.py) |
| LeWM stochasticity | Its predictor contains dropout layers, disabled in evaluation. The inspected inference path is a point-prediction computation without native probabilistic next-state sampling. An uncertainty ensemble or stochastic perturbation would be an additional declared mechanism. [Official module.py](https://raw.githubusercontent.com/lucas-maes/le-wm/main/module.py) |
| SafeDreamer state | Observation assimilation uses the previous latent state, previous action and current encoded observation. `WorldModel.imagine` calls RSSM dynamics and produces continuation-based trajectory weights. A snapshot of one image embedding alone therefore does not specify the rollout state. [Official agent.py](https://raw.githubusercontent.com/PKU-Alignment/SafeDreamer/main/SafeDreamer/agent.py) |
| SafeDreamer randomness | RSSM `obs_step` and `img_step` return recurrent deterministic state, stochastic state and distribution statistics. They explicitly sample with the Ninjax random generator. Reproducible branching must define both state copying and sampling conventions. [Official nets.py](https://raw.githubusercontent.com/PKU-Alignment/SafeDreamer/main/SafeDreamer/nets.py) |
| SafeDreamer proposals and safety | Its planners reuse a shifted action mean, mix actor and Gaussian proposals, and use critics. The code distinguishes learned cost heads from geometry computed on reconstructed coordinate observations. These variants have different information advantages. The inspected online planners return an elite mean action, which differs from the paper's sampled-distribution description. [Official behaviors.py](https://raw.githubusercontent.com/PKU-Alignment/SafeDreamer/main/SafeDreamer/behaviors.py) |
| Time and execution | SafeDreamer configurations vary action repeat by environment/variant; OSRP-Lag's displayed configuration uses repeat four and planner horizon fifteen. Thus fifteen latent transitions do not automatically mean fifteen simulator steps or the same seconds as another backend. [Official configs.yaml](https://raw.githubusercontent.com/PKU-Alignment/SafeDreamer/main/SafeDreamer/configs.yaml) |
| Integration status | The SafeDreamer release includes JAX-based code and task-specific checkpoints. LeWM's official repository loads a PyTorch module through `AutoCostModel`. Availability is not proof that either runs in the current environment or supports a newly chosen task. [SafeDreamer repository](https://github.com/PKU-Alignment/SafeDreamer); [LeWM repository](https://github.com/lucas-maes/le-wm) |

**Action adapters can carry much of the solution.** An October 2026 preprint compares
several skill-action representations under a common world model/search procedure. Its
ablations attribute substantial symbolic-action gains to knowing which actions are
applicable; long-horizon performance is also limited by how partial plans are ranked.
This is concrete evidence that proposal intelligence and search quality should be
separated, though it does not establish how our future interface would behave.
[Flow Policies as Actions of Skill-Level World Models, sections IV and V](https://arxiv.org/html/2610.04767).

### Inferences

#### A defensible interface is a contract, not just function names

The following is a conceptual compatibility sketch, not proposed implementation work:

1. **Assimilate and snapshot.** A backend converts available observation/action history
   into its internal state. The controller receives an opaque handle. Copying a handle
   must preserve all predictor context; mutating one branch must not alter siblings.
2. **Propose executable candidates.** A declared proposal mechanism returns bounded
   physical actions or skills. The same proposals should be available to comparison
   planners. A controller that selects among provided skills makes a narrower claim than
   one that discovers motor-action proposals.
3. **Predict and extend.** Each query returns a successor handle and an explicit physical
   duration. Branch ancestry, sampled action sequence, model version and random-sampling
   convention remain available for diagnosis. Reusing states may save prediction work,
   but the accounting should record what was actually computed.
4. **Read outcomes.** Task progress, safety cost, continuation, critic estimates and
   uncertainty are separate fields with declared provenance. Missing uncertainty is
   missing information, not zero uncertainty. Missing terminal value must not silently
   become an assumption of no future risk.
5. **Select and execute.** The program chooses a physical action and commitment duration.
   Returned action validity and safety are checked using the same declared criteria as
   for other planners. Any fallback is recorded as part of the policy.

A common interface can preserve nontrivial planning through branch topology,
action-specific cost/progress sequences, their changes under extension, feasibility
margins, plan diversity and remaining resources. Scalar terminal reward and cost alone
can discard the temporal pattern that motivates the proposal. Conversely, supplying a
trained estimator of precisely which branch deserves another query may implement most
of the intelligence outside the evolved program. The minimal useful information is an
open design question, not a settled list.

Raw latent distances do not have shared meanings across model families or independently
trained instances. Neither does a numerical reward unit. Scaling a goal-distance change
to match a Dreamer return does not make them semantically equivalent. Safety also needs
an agreed event definition: expected cost per transition, cumulative cost and probability
of at least one violation are different quantities. A valid study might supply the same
external task/safety semantics through separately trained readouts, while acknowledging
that this adds supervision and model-specific error.

Critics require particular care. A strong cost critic can reveal a delayed hazard before
explicit expansion reaches it. If one backend has that information and another lacks it,
the controller is solving different information problems. A useful comparison would
distinguish matched explicit-rollout information from each backend's full native
capabilities, rather than claim that one interface has made all comparisons equal.

Irregular branch-by-branch computation may also lose the efficiency of large batched CEM
rollouts. Equal transition counts do not ensure equal latency, memory, energy or available
physical reaction time. Sequential controller overhead and JAX compilation/batching
constraints are plausible engineering concerns, not measured limitations here. They
belong in feasibility assessment before any claim of compute efficiency.

#### Separate four kinds of portability

- **Compatibility:** the same search framework can call both backends.
- **Re-evolution:** separately optimised controllers work on each backend.
- **Task transfer:** one fixed controller works on held-out tasks, under explicit rules
  about new task-specific world models/readouts.
- **Frozen controller transfer:** the program and its selected constants are unchanged
  when the backend family changes. Any fitted score normalization, critic, proposal,
  threshold or adapter is identified separately.

Even the last result is not automatically a major insight: black-box optimisers already
operate on different objective functions. The informative result would identify what
was learned beyond a generic optimiser and why its decisions remain useful despite a
particular shift. Testing both model families on the same underlying tasks first would
help separate architecture transfer from new environments, sensors, actions and hazards.

### Gaps

- We have not established a common task/checkpoint pair, readout quality or equivalent
  safety labels for LeWM and SafeDreamer.
- No branch-equivalence, batching, stochastic calibration, checkpoint compatibility or
  real-time latency test has been run. Source inspection supports plausibility only.
- Which summaries preserve enough structure for transfer without handcrafting the
  answer remains unresolved. The interface itself could become the dominant contribution.

## What scientific claim could survive the strongest objections?

### Takeaway

The candidate is coherent as a study of how predictive computation changes safety
decisions. It is presently a research theme, not a demonstrated literature gap. The
strongest evidence would connect a discovered computation rule to improved physical
decisions under controlled information, budgets and model errors, with failures that
clearly delimit the claim.

### Cited Findings

**ROSARL is an optional scoring connection, not model validation.** The 2023 formulation
studies bounded-reward, undiscounted stochastic-shortest-path MDPs with unsafe absorbing
states and deterministic proper policies. Its notion of safety minimises unsafe-terminal
probability, which need not be zero; the minmax construction uses controllability and
diameter. Applying it to continuing costs, learned dynamics or approximate program
search needs a separate argument. The later 2026 paper has its own formulation and must
not be silently substituted into a theorem claim from the earlier version.
[ROSARL, sections 2 to 4](https://arxiv.org/html/2306.00035);
[2026 RLJ record](https://rlj.cs.umass.edu/2026/papers/Paper96.html).

### Inferences

#### Strongest objections

**Information cannot be created by querying a fixed model.** Suppose two true situations
require different safe actions but expose identical observations, model predictions and
readouts to the controller. The controller cannot distinguish them through more queries
to that same interface. Deeper imagination helps when relevant consequences are encoded
but unexplored; it cannot reliably recover a systematically omitted hazard. Online real
observations can change that information, but this introduces a distinct feedback
mechanism and additional physical interactions.

**Longer and wider search can worsen decisions.** Greater depth can accumulate model
bias; greater width offers more opportunities to select falsely favourable predictions.
An evolved program selected on actual simulator outcomes may learn a useful response,
but can also memorise the training distribution's error patterns. With selection only
inside imagination, even the outer evaluation may prefer exploits.

**Safety can improve through inactivity.** Fewer violations with lower exposure or
abandoned goals is not the intended result. Progress, success, failure probability,
violation counts, commitment time and fallback frequency should remain visible rather
than collapse into a favourable scalar. The choice of a safety-performance operating
point is a substantive part of the question.

**The outer search is part of the method's cost.** Many simulator-evaluated candidate
programs can make deployment look cheap while development is expensive. Counting only
the final controller's queries would obscure the main tradeoff. The same applies to
reward/cost labels, proposal-policy training, model training, failed candidates and
adaptation to a new backend.

**A model-independent wrapper is not necessarily a discovery about imagination.** A rule
that always uses a fixed depth and selects minimum predicted cost may transfer perfectly.
That would demonstrate a useful heuristic, but not the richer conjecture that evolution
learns when and how to investigate consequences. Inspectable programs are helpful only
if their interpretation is supported by interventions.

#### Conceptual evidence matrix

These are diagnostic comparisons for reasoning about the idea, not instructions to launch
experiments or a commitment to a benchmark.

| Hypothesis | Comparison that would be informative | Observation that weakens it |
| --- | --- | --- |
| Query choice matters beyond query quantity | Compare discovered allocation with fixed, random and simple adaptive schedules using equal model access and matched cost curves | Gains disappear after matching prediction work or runtime |
| The program detects delayed consequences | Change where a predictable hazard appears relative to the initial planning horizon; inspect which continuation changes the chosen action | Improvement is explained entirely by a cost critic or uniformly longer horizon |
| Branching discovers useful alternatives | Hold the action proposal bank fixed and remove or replace the branch-selection rule | A better proposal module explains all improvement |
| Predictions cause the improved decisions | Remove predictions or disrupt their assignment to candidate actions while retaining comparable marginal score statistics | Performance is unchanged, suggesting a reactive or distribution-specific shortcut |
| The method manages model error | Contrast accurate models, locally noisy models and systematic false-safe models while keeping task semantics fixed | The effect survives only with an oracle uncertainty signal or fails under modest bias |
| The discovered rule transfers | Freeze the complete program, constants and normalization policy; change backend while initially holding task fixed | Backend-specific retuning or safety-readout changes explain the result |
| Safety improves usefully | Compare at meaningful progress levels and report exposure/fallbacks | Low cost is obtained by waiting, refusing or avoiding task completion |

An oracle dynamics/readout condition would be a diagnostic information advantage, clearly
labelled. It could separate inability to search from inability to predict. If a controller
cannot outperform simple allocation even with reliable predictions, a more sophisticated
world model is unlikely to rescue its central search mechanism. Conversely, improvement
only with an oracle would not establish practical safe control.

The most defensible provisional question is therefore conditional: **under what
predictive-information and compute conditions can a discovered allocation of model queries
improve actual safe task performance, and which part of that allocation transfers?** This
is a way to sharpen the investigation, not a claim of firstness. If existing methods
already answer it adequately in the chosen setting, the correct outcome of this review
is to revise or abandon the candidate before building it.

### Gaps

- No evidence yet distinguishes an evolutionary advantage from using RL, imitation,
  direct tuning or a short handwritten rule to control the same operations.
- No chosen task yet proves that safety-critical information is available but allocated
  poorly by established planners. This is the central empirical premise to establish.
- A finite evaluation budget cannot certify zero violation probability. Rare-event
  claims need counts, uncertainty and a declared distribution, independently of how
  the planning program was discovered.
- ROSARL, uncertainty estimators and multiple models should remain optional until they
  resolve a concrete scientific question. Combining every interest would enlarge the
  project without necessarily strengthening its contribution.
