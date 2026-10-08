# Documented open problems in safe manipulation with world models

Research date: 8 October 2026. This inventory starts from observed problems and explicit
limitations, not a preferred evolutionary algorithm. "Remaining" means unresolved in the
inspected evidence, not an exhaustive absence claim. The five areas below are candidates
for modest research contributions, not five proposed new methods.

## Which agent-level outcomes still need improvement?

### Takeaway

Two concrete problems deserve early consideration: completing a task after the safety
mechanism intervenes, and distinguishing useful warnings from harmless novelty. Both have
direct manipulation evidence. Neither requires assuming that a sophisticated planner is
the missing component.

### Cited Findings

#### 1. The robot avoids a failure but then cannot finish the task

**Problem:** safety intervention can leave the task controller unable to make useful
progress. **Evidence type: author-stated explanation plus residual outcomes.**

- UNISafe's hard block-plucking evaluation changes size, weight and friction. With a
  diffusion task policy, Table 15 reports 38% safe success, 31% failure and 31%
  incompletion. These aggregate outcomes do not identify which component caused each
  failure or timeout. [UNISafe, Appendix E.2](https://arxiv.org/html/2505.00779)
- The newer *When World Models Lie* explicitly attributes some remaining incompletions
  to a nominal policy that was not trained to resume after a safety intervention.
  Its online adaptation improves handling of world-model mismatch but does not remove
  this task-policy dependency. [Section V-B](https://arxiv.org/html/2609.34300)
- **Already addressed:** LatentCBF smooths interventions and trains its safety value
  function on mixed nominal/safety-policy data. In hardware, aggregate success improves
  from 38% for switching baselines to 80%, with 20% stalling. Its limitations still note
  filter-induced OOD states and nominal-policy dependence. Mixing data for the safety
  critic is not equivalent to teaching the nominal task policy to resume.
  [LatentCBF, Table 2 and section 7](https://arxiv.org/html/2511.18606)

#### 2. A warning can mean harmless novelty, while confident predictions can still be wrong

**Problem:** a monitor must distinguish actionable danger from changes that merely look
unfamiliar. **Evidence type: explicit limitations and measured model behaviour.**

- A world-model failure detector for bimanual cable manipulation acknowledges false
  alarms from benign background changes and unknown pretrained-tokenizer biases. It
  also lists longer-term failure detection and fully autonomous-policy evaluation as
  future work. Its hardware test contains seven nominal and nine failure trajectories,
  so the strong reported detection result does not establish broad deployment coverage.
  [Foundational World Models Accurately Detect Bimanual Manipulator Failures, sections
  V-B and VI](https://arxiv.org/html/2603.06987)
- **Already addressed:** that detector learns from successful trajectories and compares
  several anomaly scores. UNISafe already uses explicit ensemble disagreement to reject
  unfamiliar actions. A new uncertainty score or nominal-data detector is therefore not
  an untouched contribution. [Bimanual detector, section IV](https://arxiv.org/html/2603.06987);
  [UNISafe, section 4](https://arxiv.org/html/2505.00779)
- A complementary warning comes from *Biased Dreams*: prolonged RSSM rollouts can have
  decreasing ensemble disagreement while error increases, apparently due to attraction
  toward familiar latent regions. The authors state that trajectory-level reliability
  does not follow from local uncertainty quality. This is mechanistic world-model
  evidence, not a demonstration of every manipulation filter failing.
  [Sections 5-6](https://arxiv.org/html/2604.25416)
- *When World Models Lie* already uses observed prediction error for online adaptation.
  Its authors retain a reactive-response limitation and warn that calibration may not
  cover actions introduced by the safety mechanism. [Sections IV and VII](https://arxiv.org/html/2609.34300)

### Inferences

For **area 1**, what remains is the interaction between task progress and safety action:
which otherwise feasible episodes stall after intervention, and whether this can be
improved without restoring failures. A modest initial scope is one existing arm task,
one frozen world model and a few nominal policies. This is a credible agent-side problem
with moderate implementation effort once the released stack runs. It does not require
irreversibility labels or proofs about recovery.

For **area 2**, the remaining practical question is reliable warning under a declared
set of benign and dangerous changes, including warning lead time and false interventions.
This is relatively approachable if observations, trajectories and safety labels are
available. Offline evaluation is cheaper than full policy training, but a monitor-only
result must be described as detection rather than prevention. We have not selected an
algorithmic fix for either area.

### Gaps

- **Area 1:** a timeout alone does not prove a bad handover. The model, nominal policy,
  safety estimates, episode horizon or impossible initial configuration can also matter.
  The published explanations must be checked in the chosen runnable environment.
- **Area 2:** benign-shift invariance can hide real dangers if the removed signal matters.
  The desired distinction needs task-specific physical labels. Detection after an object
  falls is different from warning early enough to affect the outcome.
- Neither paper family establishes that evolution is necessary. Strong simple alternatives
  and existing learned-filter improvements must remain eligible baselines.

## Which environment and information gaps remain concrete?

### Takeaway

Generalization to changed manipulation conditions and incomplete safety information are
documented weaknesses. The former offers a bounded robustness study; the latter may
require substantial new environment or sensor work and is less obviously low effort.

### Cited Findings

#### 3. Safety established on one physical setup does not automatically survive changed setups

**Problem:** the same visual task may behave differently with another object arrangement
or contact condition. **Evidence type: author-stated scope limit.**

- UNISafe fixes tower configurations and explicitly says its model cannot reliably
  filter entirely different configurations, although they can be detected as OOD. Its
  success-only ablation also performs poorly in complex block plucking, demonstrating
  that unfamiliarity detection does not replace suitable failure coverage.
  [Appendix D.2 and section 6.2](https://arxiv.org/html/2505.00779)
- **Already addressed:** *Safety filtering of robotic manipulation under environment
  uncertainty* evaluates a bimanual handover with uncertain mass and friction. It uses
  a calibrated physics simulator, sparse evaluation at critical transitions and physical
  probing to reduce uncertainty. Its discussion acknowledges reliance on an accurate
  simulator and a separate safe policy. These are substantial existing tools, not
  proposed innovations for this project.
  [Sections III-IV and VI](https://arxiv.org/html/2509.12674)

#### 4. The observations may omit the state needed for a safe decision

**Problem:** a robot can recognise a failure after it happens but lack information to
anticipate it. **Evidence type: controlled hardware result and explicit future work.**

- MultiSafe distinguishes an estimation gap, such as unknown temperature, from a
  prediction gap, such as unknown contents of an opaque bottle. Its calibrated RGB-only
  pouring filter permits safe empty-bottle tilts in 18% of trials, compared with 60%
  using multimodal information. This gives a concrete cost of missing information,
  beyond generic poor prediction. [Sections 3 and 5.2, Table 4](https://arxiv.org/html/2510.06492)
- **Already addressed:** privileged multimodal training and risk calibration improve
  RGB-only safety in its case studies. The authors nevertheless state that how to
  inject missing safety information and diagnose observability in complex environments
  remains unclear. Simply adding a probe, privileged signal or threshold would repeat
  known approaches. [Sections 5-6](https://arxiv.org/html/2510.06492)

### Inferences

For **area 3**, a tractable remaining question is robustness within one task family and
a small, physically meaningful range of variations. It is not necessary to claim
open-world generalization. A useful contribution could establish which changes break
safety and whether the limitation belongs to perception, predicted dynamics or the
controller. A released frozen model reduces training cost, but new layouts may fall
so far outside its experience that useful control is impossible without new data.

For **area 4**, distinguish missing information from an algorithm failing to use available
history. Identical observations with different hidden contents impose a real information
limit. Any approach must obtain information, exploit valid prior knowledge or tolerate
more cautious behaviour. This is a substantial scientific problem, but reproducing
specialised thermal/tactile hardware is not a small implementation. A carefully designed
simulation case could bound scope; it would still need a convincing safety definition
and observability control. No new sensing strategy is assumed here.

### Gaps

- **Area 3:** there is no verified comparison across all robustness methods and latent
  filters. A parameter sweep alone is not a novel control method. The contribution must
  be more informative than demonstrating that an out-of-distribution model can fail.
- **Area 4:** MultiSafe's project lists code as forthcoming. Availability of its trained
  models and hardware data requires a separate asset check. Its illustrative simulator
  is a thermal unicycle, not a ready-made arm-manipulation benchmark.
  [Project](https://cmu-intentlab.github.io/multisafe/);
  [Appendix 7.2](https://arxiv.org/html/2510.06492)

## Which compute issue matters, and which opportunities should be inspected first?

### Takeaway

The fifth area is timely safety decisions under bounded computation. It is measurable
on modest hardware, but often an engineering concern until tied to a substantive safety
or task-progress result. Areas 1-3 currently provide the clearest starting points.

### Cited Findings

#### 5. Safety evaluation must finish before its decision becomes stale

**Problem:** evaluating enough candidate actions or robust predictions can be expensive.
**Evidence type: measured cost and an explicit limitation.**

- **Already addressed:** LatentCBF avoids repeated world-model calls by scoring actions
  with a learned critic. In its particular implementation, model-based filtering runs
  out of memory at 50 action samples on a 48 GB GPU, whereas the critic evaluates
  7,600 samples in about 9.6 ms. These figures concern its DINO-WM pipeline and cannot
  be treated as a universal latent-model limit. [Table 3](https://arxiv.org/html/2511.18606)
- The newer error-adaptive filter reports 39.59 ms per step on an RTX 5090, versus
  17.42 ms for UNISafe, within its 15 Hz hardware budget. Its limitations still list
  added optimization cost. This is a different three-camera dustpan task, not a
  benchmark of the released block-plucking checkpoint.
  [When World Models Lie, sections VI-VII](https://arxiv.org/html/2609.34300)

### Inferences

For **area 5**, the remaining question is the safety/progress attainable at a declared
latency or query budget for a chosen controller and task. Measuring this frontier can
expose an actionable bottleneck. Batching or lowering precision alone may be routine
engineering rather than a research contribution. No evolved computation scheduler is
being presumed as the answer.

My evidence-led priority is **task progress after intervention**, **useful warnings under
benign versus dangerous changes**, then **bounded physical robustness**. These can begin
with existing predictive components. Missing-information studies are attractive but
carry more environment and data risk. Compute-aware studies are a supporting option
unless a clear task-level failure caused by latency is established.

### Gaps

This bounded search inspected UNISafe, its safety-filter successors, MultiSafe, a bimanual
failure detector and a physics-based uncertainty method, following references to newer
work. UNISafe is CoRL 2025. The bimanual detector's March 2026 arXiv record reports ICRA
2026 acceptance. MultiSafe was revised 6 June 2026; LatentCBF is a November 2025 preprint;
*When World Models Lie* is a 28 September 2026 preprint. Unverified venue acceptance is
not implied. [UNISafe record](https://arxiv.org/abs/2505.00779);
[detector record](https://arxiv.org/abs/2603.06987);
[MultiSafe record](https://arxiv.org/abs/2510.06492);
[LatentCBF record](https://arxiv.org/abs/2511.18606);
[adaptive-filter record](https://arxiv.org/abs/2609.34300)

No area is certified as unstudied. Releases, reproduction effort and evolutionary fit
are separate checks. The purpose is to choose a real, approachable problem before
deciding what should evolve.
