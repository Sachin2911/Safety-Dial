# Evidence and experimental logic for the Safety Dial project

## What did the comparative review and UNISafe compute audit contribute to the current direction?

### Takeaway

The review supplied the logic for asking a narrow behavioral question and testing evolution
against ordinary learning and simple corrections. The compute audit supplied feasibility
bounds and warned that model queries, training updates, simulator interactions and elapsed
time are different costs. Neither document proved that the present ManiGuard method works.

### Cited Findings

- **Tested fact:** The completed UNISafe adaptation study found a large behavioral gain over
  the frozen baseline, but not an evolutionary advantage. On 64 reserved starts, completion
  was 2 for native, 55 for a fixed hold rule, 56 for random search, and 54 for both policy
  gradient and evolution. Evolution also had 8 violating starts against 6 for native.
  [UNISafe adaptation study](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/UNISafe grasp maintenance and evolutionary adaptation.md:97>)
- **Tested fact:** Diagnosis showed that the UNISafe task actor opened its gripper before the
  first safety replacement. A bounded hold rule largely retained its benefit without the
  filter. The supported explanation is therefore basic grasp maintenance, not a demonstrated
  solution to filter-induced stalling.
  [UNISafe adaptation study](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/UNISafe grasp maintenance and evolutionary adaptation.md:27>)
- **Tested fact:** The completed UNISafe study charged elite reevaluations and discarded
  candidates, and reported that equal episode caps produced different realized control-step
  costs. Its adaptation study used 1,424 episodes, 166,697 control steps and 833,485 physics
  steps when engineering, development, search, validation and reserved evaluation were all
  included.
  [UNISafe design](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/UNISafe grasp maintenance and evolutionary adaptation.md:51>),
  [UNISafe accounting](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/UNISafe grasp maintenance and evolutionary adaptation.md:178>)
- **Tested fact:** The earlier ManiGuard dial pilot produced an adverse result. The evolved
  endpoint obtained 4/8 safe completions and 2/8 violating episodes, while identity obtained
  5/8 safe completions and 0/8 violating episodes. The experiment had no matched random-search
  or ordinary-learning arm.
  [ManiGuard pilot](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/ManiGuard evolutionary safety dial pilot.md:8>),
  [pilot interpretation](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/ManiGuard evolutionary safety dial pilot.md:177>)
- **Review finding, not a new experiment:** The comparative review distinguishes task-policy
  incompetence, repeated rejected proposals, an overly conservative filter, prediction error,
  missing information and an inadequate horizon as different possible causes of a timeout.
  It consequently treats a stall as an outcome that still needs diagnosis.
  [Comparative review](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/Safe manipulation progress and robustness.md:13>)
- **Review finding, not a new experiment:** The review identifies ordinary task-policy
  learning with the same filter present as an essential comparator. It also identifies
  compact residual adaptation, random search and improved action correction as credible
  alternatives to evolutionary adaptation.
  [Comparative review](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/Safe manipulation progress and robustness.md:19>)
- **Audit finding, not a runtime measurement:** The UNISafe audit verified checkpoint shapes
  and parameter counts and judged one-GPU investigation plausible. It found about 40 million
  world-model parameters and about 151 million uncertainty-ensemble parameters, but it did
  not run a GPU benchmark, simulator or training job.
  [UNISafe compute audit](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/UNISafe inference and training compute.md:3>),
  [audit conclusion](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/UNISafe inference and training compute.md:168>)
- **Audit finding:** The same audit shows why costs must be named precisely. One published
  training batch has 1,024 observed time positions and 15,360 imagined transitions, but the
  latter are reused observations and model queries, not new simulator interactions.
  [UNISafe compute audit](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/UNISafe inference and training compute.md:62>)
- **Current proposal:** The present direction responds to these lessons by prioritizing a
  shared, bounded, observation- and instruction-conditioned controller that steers each
  frozen ManiGuard policy through its native flow-noise input and is evaluated using actual
  simulator outcomes. This is explicitly a recommendation, not an experimentally discovered
  winner.
  [Current direction](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:3>)
- **Scope boundary:** The current direction does not assume the complete UNISafe safety
  filter or a newly trained world model. The compute audit itself says that its feasibility
  result does not require adopting the complete filter or expanding the project into model
  retraining.
  [UNISafe compute audit](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/UNISafe inference and training compute.md:168>)

### Inferences

- The evidence motivates studying whether a compact adapter can recover useful alternatives
  from a frozen policy, but it gives no reason to expect evolution to beat a fixed rule,
  random search or ordinary learning. That advantage is the main fact still to be earned.
- The previous cap and timing controller is a weak research bet because its adverse pilot
  and missing baselines leave both the representation and the optimizer explanation
  unresolved. Native-noise steering is a new physical hypothesis, not a conclusion inherited
  from those results.
- The broad allocated topic asks whether evolutionary algorithms can balance reward and
  violations without relying on hand-crafted penalties, but the current proposed objective
  explicitly penalizes violations. The immediate study should therefore make a narrower
  empirical claim about safe completion per interaction, not a penalty-free learning claim.
  [Allocated topic](</home/sachin/Desktop/Research/Safety-Dial/docs/allocatedTopic.md:16>),
  [current objective](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:103>)

### Gaps

- No current experiment has shown that the explicit noise input changes behavior usefully on
  the pinned ManiGuard checkpoints.
- No current result establishes a cross-family shared controller, an evolutionary advantage,
  a safety improvement, or transfer to a real robot.
- The runtime and interaction budget of the new ManiGuard stack remains to be measured on the
  actual host. The UNISafe compute figures answer a different system question.

## What must be frozen before experiments begin?

### Takeaway

Freeze the question, the controller, the evidence split and the spending rules before seeing
comparative outcomes. Otherwise every disappointing result creates an opportunity to change
the method, the cases or the budget, and the final number no longer tests one identifiable
hypothesis.

### Cited Findings

- **Required protocol:** Before launch, freeze a versioned protocol and stage schedule. It
  must state development and evaluation partitions, controller and feature definitions,
  parameter bounds, objective, baselines, seeds, validity rules, interaction limits, time
  allocation, and finite recovery allowances.
  [Current direction](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:13>)
- **Required evidence separation:** Method changes belong on development cases. Independent
  audit and held-out-family cases must stay untouched. Once a case is inspected, it is exposed
  and cannot be presented as untouched confirmation of a revised method.
  [Current direction](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:35>)
- **Required mechanism contract:** Freeze the exact observation features, normalization,
  feature latency, noise basis, gripper permissions, bounds, parameter count and controller
  architecture. At zero controller weights, the native noise distribution, random-number
  advancement and numerical inference path must be preserved. A zero noise tensor is not the
  native policy.
  [Current direction](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:97>)
- **Required benchmark contract:** Pin the source revision, assets, checkpoint, prompts,
  camera mapping, renderer, grasp mode and native hard horizons. A rendering-profile change
  can invalidate attribution even when the code still runs.
  [Current direction](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:133>)
- **Required selection contract:** Predeclare the primary task family and fallback before
  reading comparative scores. Jar is the proposed scientific target and Clutter the capacity
  fallback. The family, pilot size and fallback rule must be selected before outcomes are
  known.
  [Current direction](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:109>)
- **Required statistical contract:** After the pilot, freeze the main-study effect regarded as
  meaningful, the acceptable violation increase or safety noninferiority margin, uncertainty
  analysis and sample allocation before the independent study.
  [Current direction](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:121>)
- **Required recovery contract:** Save optimizer state, random generator states, ancestry,
  completed and pending evaluations, consumed budget and failure status. Interrupted episodes
  remain charged, and valid poor outcomes or timeouts are not infrastructure replacements.
  [Current direction](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:15>),
  [comparison preservation](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:35>)

### Inferences

- Freezing is not bureaucracy. It makes a negative result informative by identifying exactly
  which controller, search process and evidence population failed.
- The protocol needs separate development, selection and final checking blocks because using
  final cases to choose the controller turns evaluation into additional training.
- The identity condition must be tested as an implementation invariant before performance
  comparisons. Without it, an apparent gain or loss could come from changed random sampling,
  preprocessing, cadence or output transforms rather than the learned controller.
- A new representation, larger budget or repaired implementation after a valid comparison is
  a new cohort. Its costs and matched baselines must be declared again, rather than silently
  replacing the original evidence.

### Gaps

- The repository contains an illustrative pilot allocation, but the new protocol, comparison
  design and approved budget have not yet been frozen for execution.
- The meaningful safe-completion effect, acceptable violation margin and adequately powered
  sample size remain unspecified.
- The exact family partitions and whether enough genuinely unexposed tasks remain are not yet
  established.

## Which baselines and matched-budget comparisons are essential?

### Takeaway

Each comparator answers a different Feynman-style question: did the interface help, did
conditioning matter, did evolution matter, and did the result transfer? Matching the
information, tuning effort and physical interaction allowance is what makes the answers
causal rather than cosmetic.

### Cited Findings

- **Native checkpoint:** Contemporaneous native execution is the identity reference. It asks
  whether any steering improves the frozen policy at all.
  [Current direction](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:121>)
- **Tuned fixed correction:** A genuinely tuned constant bias or fixed setting asks whether a
  simple reusable rule explains the whole gain. This is essential because the UNISafe fixed
  hold rule slightly exceeded evolution on reserved completion.
  [Current direction](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:121>),
  [UNISafe result](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/UNISafe grasp maintenance and evolutionary adaptation.md:97>)
- **Matched random search:** Random search must receive the same initial data, bounded domain,
  candidate evaluations, reevaluations and final confirmation. It asks whether evolutionary
  inheritance and selection help beyond simply trying the same number of alternatives.
  [Pilot design](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:115>)
- **Ordinary learning:** Learning with the same observations and output domain asks whether
  the evolutionary update is practically useful. Episodic REINFORCE over a tiny shared
  parameterization is a narrow check, not a substitute for a competent state-conditioned
  latent RL baseline such as a DSRL-style SAC learner.
  [Paper-level baselines](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:121>)
- **Representation control:** If the claim is that latent noise steering is a better
  adaptation interface, compare it with a bounded direct-action residual learned using an
  appropriate ordinary RL method. Otherwise an improvement only supports adaptation in
  general, not the latent representation.
  [Paper-level baselines](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:121>)
- **Conditioning control:** Fixed settings versus a shared conditional controller ask whether
  observation and instruction conditioning is useful, rather than whether one global bias
  happened to work.
  [Separated conclusions](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:125>)
- **Allocation controls:** If adaptive racing or noise-aware evaluation allocation is claimed
  as the contribution, compare it with one-shot selection, fixed resampling and equally
  funded adaptive allocation for competitors under the same physical budget and finalist
  protection.
  [Separated conclusions](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:125>)
- **Transfer controls:** Distinguish per-family refitting, a frozen controller on new tasks
  from trained families, and the same frozen controller on an adaptation-held-out family.
  Only the last tests the stronger cross-family adapter target.
  [Transfer design](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:129>)
- **Information matching:** Dense rewards, privileged critic inputs, additional encoders or
  extra outcome labels give one method more information. Such arms must be matched or clearly
  labeled as a different comparison.
  [Paper-level baselines](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:121>)
- **Budget matching:** Equal episode counts are insufficient because methods can terminate at
  different times. Compare learning curves at common executed control counts, charge all
  candidates, repeated evaluations, decoder probes and training data, and separately report
  policy or model queries and elapsed time.
  [Main-study controls](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:127>)

### Inferences

- Native versus fixed versus conditional isolates the value of the interface and conditioning.
  Random search and ordinary learning then isolate the optimizer. Without that ladder, the
  phrase "evolution improved safety" bundles several different causes.
- A fair ordinary-learning baseline must have the same action authority as the EA. Giving RL
  richer observations or a larger output space would compare systems, not optimizers.
- Matching total simulator interaction is the primary fairness rule because physical
  experience is the scarce resource. Model queries and elapsed time remain important
  secondary costs because two methods can consume the same interactions while requiring very
  different computation.
- Optimizer replication is necessary because one successful evolutionary run could be lucky
  selection. Finalist testing alone estimates one selected controller, not the distribution
  of outcomes produced by the optimizer.

### Gaps

- The minimum 64-attempt pilot explicitly cannot fit every paper-level baseline and leaves
  ordinary learning outstanding if a meaningful arm cannot fit within capacity.
  [Pilot limitation](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:119>)
- The exact ordinary-learning algorithm, its tuning allowance and a fair compute budget have
  not been frozen.
- The number of optimizer replicates and independent task blocks needed for a paper-level
  claim has not been determined by a power calculation.

## Which metrics prevent misleading conclusions?

### Takeaway

The central outcome is not merely "success" and not merely "no violation." Report whether
the requested task finished, whether any physical violation occurred, and whether the episode
ended unfinished, then report the cost required to obtain that outcome.

### Cited Findings

- **Safe completion:** The proposed objective defines requested safe completion as requested
  completion with no violation. A violating completion is therefore not counted as safe
  success. The raw outcomes must still be retained even when a scalar objective is used for
  search.
  [Current objective](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:103>)
- **Four outcome cells:** Report safe complete, unsafe complete, safe unfinished and unsafe
  unfinished. The earlier ManiGuard pilot demonstrates why: full steering completed six of
  eight episodes, but two completions were unsafe, leaving only four safe completions.
  [ManiGuard pilot](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/ManiGuard evolutionary safety dial pilot.md:14>)
- **Unfinished episodes:** Completion, raw physical violations and unfinished episodes must
  be reported together. Fewer violations alone can reward a controller that does nothing or
  makes weaker progress.
  [Comparative review](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/Safe manipulation progress and robustness.md:68>)
- **Timeout semantics:** A valid timeout is an outcome, not proof that the filter is too
  restrictive, the situation is irreversible, or the controller lacks authority. Those
  explanations require trajectory evidence.
  [Comparative review](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/Safe manipulation progress and robustness.md:13>)
- **Task meaning:** Retain requested-sequence completion, especially Jar closure, as well as
  benchmark completion. This prevents a controller from exploiting a benchmark goal while
  skipping a requested prerequisite.
  [Benchmark contract](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:133>)
- **Denominators:** Report native success, engagement-conditioned metrics and their exact
  denominators, rather than only percentages. Different controllers can engage or intervene
  on different subsets, so conditional success alone is descriptive.
  [Benchmark contract](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:133>),
  [comparative review](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/Safe manipulation progress and robustness.md:13>)
- **Action provenance:** Log proposed, filtered and finally executed actions separately. A
  task critic may intentionally learn the consequence of proposing an action to a composed
  filtered environment, while a physical dynamics model must associate the next observation
  with the action actually executed.
  [Comparative review](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/Safe manipulation progress and robustness.md:29>)
- **Current-system nuance:** ManiGuard's current monitor labels physical violations but does
  not alter actions. It is therefore an outcome source, not evidence that a deployed safety
  filter caused the trajectory.
  [ManiGuard pilot](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/ManiGuard evolutionary safety dial pilot.md:30>)
- **Interaction accounting:** Count simulator interactions used for adaptation, discarded
  candidates, reevaluations, engineering probes and final checking. Report executed control
  steps, VLA or model queries and elapsed time separately.
  [Main-study controls](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:127>)
- **Statistical caution:** With zero violations in `n` independent episodes, the approximate
  95 percent upper rate bound is `3/n`, not zero. The earlier pilot's 0/8 identity result
  therefore has an approximate upper bound of 37.5 percent even before accounting for its
  repeated scenes.
  [Main-study controls](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:127>),
  [pilot interpretation](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/ManiGuard evolutionary safety dial pilot.md:177>)
- **Inference unit:** Cluster evaluation by base task and replicate optimizer runs. Frames
  from one trajectory are not independent safety trials, and variants of one base scene do
  not automatically supply many independent task examples.
  [Main-study controls](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:127>)
- **Safety claims:** A nonsignificant difference in violations is not evidence of equal safety.
  A safety noninferiority statement needs a prospectively declared margin and enough evidence
  to support it.
  [Main-study controls](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:127>)

### Inferences

- A useful one-line scorecard is: "Of all independent episodes, how many safely completed the
  requested sequence, how many violated, how many remained unfinished, and how many executed
  control steps did the method spend during adaptation and evaluation?"
- The four outcome cells are preferable to one reward because the same scalar can hide very
  different trade-offs. An unsafe completion and a safe timeout are both failures of safe
  completion, but for different scientific reasons.
- Executed-action histories are the physical causal record. Proposed actions remain necessary
  when studying what a controller asked a filter or environment to do. Mixing those meanings
  can train the wrong model and invalidate diagnosis.
- Violation-free pilot runs are useful screening evidence, not certification. The upper-bound
  calculation makes the residual uncertainty visible.

### Gaps

- The project still needs a final, versioned outcome contract for every selected family,
  including which predicates count as physical violations and how simultaneous completion
  and violation are classified.
- The dependence structure among task variants and the proper number of independent base tasks
  remain to be fixed.
- No accepted violation increase or noninferiority margin has yet been justified.

## What claims would the planned experiments support, and which would they not support?

### Takeaway

The planned study can support a narrow, evidence-based statement about a compact adapter's
safe-completion benefit per interaction and whether evolution contributed to that benefit.
The strength of the claim must match the exact comparison and transfer stage that completed.

### Cited Findings

- **Pilot-level support:** The proposed protected pilot can test whether the noise interface
  has behavioral authority, whether bounded changes cause adverse physical effects, and
  whether there is an effect-size lead worth studying. Four audit episodes per condition and
  one optimizer seed cannot establish superiority.
  [Pilot interpretation](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:109>)
- **Interface claim:** Native versus adapted execution can show that a frozen ManiGuard
  checkpoint contains useful alternatives accessible through the selected steering
  interface, provided the identity path and independent outcomes are valid.
  [Separated conclusions](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:125>)
- **Conditioning claim:** A shared conditional controller beating a tuned constant correction
  can support the claim that observations and instructions add value beyond one fixed bias.
  [Separated conclusions](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:125>)
- **Evolution claim:** Evolution beating both matched random search and competent ordinary
  learning under common information, tuning and interaction budgets can support a practical
  evolutionary advantage. If the optimizers tie, the result supports adaptation but not
  evolutionary superiority.
  [Separated conclusions](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:125>)
- **Mechanism claim:** If adaptive evaluation allocation is proposed as the distinctive
  mechanism, it can be credited only after matched comparison with one-shot selection, fixed
  resampling and competitor racing. A standard EA result by itself does not demonstrate that
  allocation was responsible.
  [Separated conclusions](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:125>)
- **Transfer claim:** One frozen shared controller working on an adaptation-held-out family
  supports adapter transfer across family-specific pretrained checkpoints. It does not show
  transfer of one foundation policy to an unseen family, because the held-out family still
  uses its own pretrained checkpoint.
  [Transfer design](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:129>)
- **Scope claim:** A reserved Jar improvement can be a valid narrow contribution. It does not
  establish domination across the ManiGuard suite; a suite-wide claim needs fixed evaluation
  across all relevant families, including null or adverse results.
  [Benchmark contract](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:133>)
- **Novelty limit:** Noise steering, frozen-policy adaptation and evolutionary frozen-VLA
  improvement already have close precedents. The proposed contribution is therefore the
  tested combination of a wholly frozen base, reusable compact controller, physical
  specification outcomes and explained safe-completion advantage per total interaction, not
  the invention of low-dimensional steering or evolutionary optimization.
  [Current direction](</home/sachin/Desktop/Research/Safety-Dial/Evolutionary improvement on ManiGuard.md:103>)
- **Venue relevance:** GECCO requires evolution to remain central to the paper's contribution.
  The venue notes identify Neuroevolution and Evolutionary Machine Learning as the closest
  tracks, but publication suitability still depends on the eventual evidence.
  [GECCO notes](</home/sachin/Desktop/Research/Safety-Dial/docs/gecco2027.md:47>),
  [comparative review](</home/sachin/Desktop/Research/Safety-Dial/docs/research/reports/Safe manipulation progress and robustness.md:68>)

### Inferences

- The cleanest final claim has four nested levels: the interface helped; conditioning helped;
  evolution helped compared with random and ordinary learning; the same controller transferred.
  Evidence for an earlier level does not automatically support a later one.
- A positive Jar result without transfer is still useful, but it should be described as a
  family-specific safe-execution result. A controller that requires per-family outcome-based
  refitting supports reusable methodology, not one shared transferable controller.
- A negative evolutionary result is scientifically complete if the comparison was valid. It
  can show that a fixed rule, random search or ordinary learning explains the improvement more
  simply.
- None of these simulator studies alone supports a real-world robot safety guarantee, formal
  safety certification, universal alignment, or irreversibility claims.
- The monitor provides labels rather than action protection, so a gain in ManiGuard would be
  evidence about learned execution under measured constraints, not evidence that a safety
  filter guarantees safe actions.

### Gaps

- Behavioral authority, evolutionary advantage, ordinary-learning competitiveness,
  cross-family transfer and statistical power all remain untested for the proposed native
  noise controller.
- The exact independent main-study allocation has not been approved, so the strongest
  attainable claim is not yet known.
- Physical transfer, formal guarantees, whole-suite aggregation and novelty priority would
  require evidence beyond the currently proposed first pilot.
