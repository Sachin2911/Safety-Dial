# The current Safety Dial project: its scientific core

## 1. What problem is the project trying to solve, and why?

### Takeaway

**One-sentence version:** The project asks whether a small controller can learn to steer the
random choices inside several already-trained robot policies so that they complete physical
manipulation tasks safely more often, without retraining those policies and without receiving
more simulator experience than the methods it is compared with.

The Feynman-style picture is this: imagine that ManiGuard supplies several trained robot
drivers. Each driver knows one family of jobs, but sometimes chooses a poor approach, tips an
object, misses a grasp or simply runs out of time. The project does not rewrite the drivers. It
adds one small navigator that looks at what the robot sees, the instruction and what has just
happened, then gently biases the driver's built-in random starting point. Evolution tries many
bounded versions of that navigator, keeps the versions whose real episodes go better, and
mutates them. The scientific question is not merely whether one evolved navigator gets a good
score. It is whether this mechanism produces a reliable, safe-completion advantage over the
unchanged driver, a simple fixed bias, random search and ordinary learning when all methods pay
for all of their experience.

### Cited Findings

- ManiGuard is being used as a simulated manipulation benchmark with task families such as
  Jar, Lid, Stack, Clutter, Cabinet and Dusty. It supplies separately fine-tuned family
  checkpoints and physics-grounded temporal safety specifications that are distinct from task
  completion. The reported low safe-completion rates create headroom, but weak completion is
  not automatically a safety failure. For example, the published main simulation results had
  no measured Clutter violations despite variable completion. [Research screening](../../docs/research/reports/Evolutionary%20opportunities%20for%20safe%20robotics.md)

- The current research target is explicitly “one shared observation- and
  instruction-conditioned controller” across the released pi0.5 family-specific frozen
  policies. Jar is the first scientific target and Clutter is the predeclared capacity
  fallback. This is a research recommendation, not an experimentally discovered winner.
  [Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)

- The project moved to a deeper behavioral lever because earlier execution-side adjustments
  were not convincingly effective. A six-hypothesis ManiGuard study completed 196 engineering
  episodes, but replanning, motion moderation, prerequisite guards, release delay, retry and
  return after intervention did not establish a repeatable safe-completion repair. A later
  uniform motion cap affected only 55 of 18,657 recorded controls, while a feedback trigger was
  eligible on only 3 of 482 diagnostic prefixes. [Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)

- Earlier UNISafe work gives the central warning against an easy evolutionary story. Evolution
  greatly raised completion, but a fixed rule and random search matched or exceeded it, and the
  improvement did not establish improved safety. The lesson retained by the current project is
  to compare against a strong fixed rule and explain what actually changed before crediting the
  optimizer. [Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)

- Completion, violation and unfinished status must be reported together. An inactive robot can
  avoid violations while failing the task, and a completed task can still be unsafe. The
  supplemental requested-sequence label also prevents loopholes such as carrying an open jar
  to the goal and calling that successful. [Requested-sequence contract](../../docs/research/research_notes/ManiGuard%20phase2%20completion%20contract.md)

### Inferences

- “Safety Dial” should not be understood as a single knob that simply makes the robot more
  cautious. The present object is a small conditional policy whose output may change with the
  observation, instruction and recent history. A useful controller must improve the joint
  outcome “did the requested job without a physical violation,” not merely slow the robot or
  reduce monitor alarms.

- The project is really testing three nested ideas: the frozen policy contains useful alternate
  behaviors; a compact controller can select among them from available observations; and
  evolution is a better way to learn that controller than fair alternatives. A positive result
  at one level does not prove the next level.

### Gaps

- The repository does not yet contain a completed, independently evaluated comparison proving
  that evolution improves physical safe completion. The dated 20 October pilot report says the
  full Jar comparison is running and that its results so far are development evidence only.
  [Native-noise pilot](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md)

- The present local evidence does not establish a whole-suite improvement, a hardware-robot
  result or a universal safety guarantee. The current study is a simulation study whose first
  funded comparison is on Jar. [Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md);
  [native-noise pilot](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md)

## 2. What are the important pieces, in plain language, and what exists now?

### Takeaway

ManiGuard is the test course; each frozen VLA is a trained family-specific driver; the monitor
is a passive referee; the shared controller is the small navigator; native-noise steering is
the navigator's steering wheel; and a matched interaction budget is the rule that every method
gets the same amount of costly practice and must disclose all of it.

### Cited Findings

- **ManiGuard:** A benchmark episode gives the robot camera views and a language instruction,
  lets a family-specific policy act in a physics simulator, checks whether the task goal was
  reached and tracks temporal safety conditions. The family checklists make the jobs concrete:
  Jar means close before lifting and keep it closed while carrying; Lid means put the lid on
  before lifting; Stack and Clutter require a held target in the goal while retaining the
  benchmark's safety verdict; Cabinet requires opening before insertion and ending closed; and
  Dusty requires cleaning before food transfer. [Requested-sequence contract](../../docs/research/research_notes/ManiGuard%20phase2%20completion%20contract.md)

- **VLA:** “Vision-language-action” means a model that uses images and a text instruction to
  produce a chunk of robot actions. In the present pi0.5 setup, the model works with a
  16-position action chunk and 32 model coordinates, while the physical command is seven arm
  joint targets plus a gripper command. The model-coordinate width and robot-action width are
  therefore not interchangeable. [Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)

- **Family-specific frozen policies:** Different task families use separately fine-tuned
  pi0.5 checkpoints. “Frozen” means their learned weights, preprocessing, output transforms and
  normal action-query rhythm remain unchanged while the small controller is adapted. The
  stronger transfer target is to use one learned controller with each family's own pretrained
  checkpoint, not to use one foundation checkpoint on a wholly unseen family.
  [Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)

- **Safety monitor:** The standard monitor is a judge, not a bodyguard. It observes the episode
  and reports temporal-specification violations after actions are executed. It does not block,
  replace or repair an action. The live monitor's state is excluded from the controller's input
  in the current implementation; its verdict is used as an outcome label. [Initial adapter
  screens](../../docs/research/reports/ManiGuard%20initial%20adapter%20screens.md); [native-noise
  pilot](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md)

- **Shared controller:** In symbolic form, the proposal is
  `z' = epsilon + alpha B tanh(W phi)`. Here `epsilon` is the policy's normal random noise,
  `phi` is a compact description of the present observation, instruction and history, `W` is
  the small learned matrix, `tanh` keeps its output bounded, `B` maps a few controller outputs
  into chosen noise directions, and `alpha` limits their strength. “Shared” means the same `W`
  is intended to work across tasks and eventually across family checkpoints, without a family
  identifier or Jar-specific rule. [Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)

- **Native-noise steering:** A flow-based VLA normally starts action generation from Gaussian
  noise and turns that noise into an action chunk. Instead of clipping the finished joint
  command, the controller makes a small bounded change to this native input noise. The hope is
  that the frozen policy itself decodes that nudge into a coordinated alternate approach,
  grasp or contact behavior. This does not guarantee that the alternate is useful or safe.
  [Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)

- **What is now implemented:** The current controller uses `W` with shape 4 by 44, so it has
  176 bounded weights. Its features combine measured robot state, executed-action history and
  frozen CLIP projections of two real camera images and the actual instruction. Its four
  steering directions act on the seven arm coordinates across the 16-position chunk. Family,
  task identity, goal state and monitor state are excluded. The zero controller follows the
  unchanged native inference path. [Native-noise pilot](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md)

- **What the authority audit established:** All 320 allowed decoder calls completed. In both
  Jar and Clutter, perturbing all four directions changed the decoded arm prefix enough to pass
  the declared influence test, while same-noise identity pairs matched. This demonstrates that
  the controller can influence outputs through the intended interface on saved observations.
  It does not show that the changes improve a future physical trajectory, completion or safety.
  [Native-noise pilot](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md)

- **Matched interaction budget:** Fair comparison requires more than the same number of final
  episodes. All simulator controls used for adaptation, discarded candidates, failed and
  partial attempts, elite reevaluation, basis-response probes and training data must be charged.
  Because episodes can have different lengths, methods should also be compared at common
  control counts. VLA queries and elapsed time are reported separately. Information access,
  observation features, controller output domain, tuning care and finalist evaluation also
  need to be matched. [Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md);
  [repository controls](../../AGENTS.md)

- **Current funded comparison:** The frozen pilot gives evolutionary search and matched random
  search the same eight initialization episodes, then gives each 16 descendant episodes. It
  seals the resulting finalists and evaluates native, fixed, evolutionary and random
  conditions in 16 independent checking episodes. The overall plan is 60 valid episodes with
  at most four infrastructure replacements. Ordinary learning is implemented only as an
  unfunded interface and is outside this minimum comparison. [Native-noise pilot](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md)

### Inferences

- The best everyday analogy for native-noise steering is not “grab the robot's wrist after it
  moves.” It is “slightly bias the rough sketch from which the trained driver plans its next
  short motion.” This may preserve more of the driver's learned coordination than editing each
  finished joint target, but that preservation is still a hypothesis.

- One shared controller is a deliberately stronger claim than one custom adapter per family.
  It asks whether the families expose sufficiently compatible steering directions and common
  visual/language features. Equal tensor sizes alone do not imply that the same direction means
  the same behavior in separately fine-tuned checkpoints.

### Gaps

- The 320-call audit proves interface authority only. It has not established stable causal
  effects over closed-loop trajectories, physical safe-completion improvement or held-out
  family transfer.

- The simple episodic `ParameterReinforce` implementation is not the competent
  state-conditioned latent SAC baseline needed for a strong paper-level claim about evolution
  versus ordinary learning. Its mean controller has no physical evidence.
  [Native-noise pilot](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md)

## 3. What hypotheses are being tested, what is only proposed, and what would success or failure mean?

### Takeaway

The project is staged so that each claim has to earn the next one. First show controllability,
then physical benefit, then an evolutionary advantage, then reuse and transfer. At the current
repository snapshot, the interface and bounded output influence exist; the full evolutionary
comparison, ordinary-learning comparison, latent extensions and cross-family transfer result
do not yet exist.

### Cited Findings

- **Hypothesis 1, useful authority:** Nearby bounded noise perturbations should produce
  meaningful alternate behavior, not merely microscopic numerical differences. This first
  stage has partial support because the decoder audit found repeatable output influence in Jar
  and Clutter. Whether those differences change physical outcomes usefully remains untested.
  [Native-noise pilot](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md)

- **Hypothesis 2, evolvability:** Nearby controllers should create useful, partly heritable
  behavioral differences. Selection should retain beneficial differences and mutation should
  improve safe completion efficiently despite noisy episode outcomes. The frozen illustrative
  score is `int(safe requested completion) - 2 * int(violation)`, but every raw completion,
  violation and unfinished outcome is retained. [Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)

- **Hypothesis 3, conditioning matters:** A controller that changes its steering from the
  current images, instruction, robot state and history should outperform one constant bias.
  Otherwise, a simple fixed correction explains the effect and the conditional-controller
  claim fails. [Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)

- **Hypothesis 4, evolution matters:** The evolutionary update must beat matched random search
  and appropriate ordinary learning using the same observations, output space and total
  interaction allowance. If adaptation helps but the optimizers tie, the adaptation mechanism
  may still be useful, but evolutionary superiority should be rejected. [Current
  direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)

- **Hypothesis 5, reuse and transfer:** A single frozen controller should work on new tasks in
  development families and, for the strongest claim, on a family excluded from adaptation.
  Per-family refitting would support only the weaker claim that the method is reusable. The
  separately budgeted cross-family bridge remains untested. [Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md);
  [native-noise pilot](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md)

- **Proposed native-noise extensions, all currently untested:** adjust both noise bias and
  scale; make some steering persist between policy queries; construct a compact basis from
  directions to which the decoder responds strongly; and evolve small gates on internal VLA
  activations. Response magnitude would show influence, not usefulness or safety. Activation
  association would show correlation with outcomes, not a causal safety direction.
  [Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md); [pilot status](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md)

- **Proposed fallbacks and later avenues:** A bounded direct-action residual is the designated
  representation fallback if native noise lacks useful authority. Guarded subgoal programs,
  noise-aware evaluation allocation, a quality-diversity controller repertoire, evolutionary
  scene acquisition and online action search with a learned scorer or world model have higher
  prerequisites and are not assumed parts of the current experiment. [Current
  direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)

- **Explicitly not assumed:** The current controller does not depend on a new learned world
  model, ROSARL, a learned safety filter that changes actions or continued task execution under
  such a filter. The monitor supplies labels but does not intervene. Base-policy fine-tuning is
  also outside the frozen-policy question. [Repository authority](../../AGENTS.md); [current
  direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)

- **Latest empirical boundary:** In the completed eight-episode shared initialization panel,
  none of the native, positive-bias, negative-bias or random-conditional runs achieved requested
  completion. Their violation counts were 1/2, 1/2, 0/2 and 2/2 respectively. The negative bias
  led the frozen score only because both runs avoided violations while remaining unfinished;
  this is not improved safe completion. Evolutionary descendants and protected final checking
  were still pending at the report snapshot. [Native-noise pilot](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md)

### Inferences

- **What success would mean:** Reserved physical episodes show more requested tasks completed
  without violations, not merely fewer violations or more benchmark goal flags. The gain
  survives comparison with native execution and a tuned fixed bias. The evolutionary method
  then also beats matched random search and competent ordinary learning at the same total cost,
  and the mechanism remains useful on reserved tasks and an adaptation-held-out family. The
  project should be able to explain what behavioral change produced the gain and report any
  family regressions.

- **What a useful partial success would mean:** Native-noise adaptation beats native execution,
  but a fixed bias, random search or ordinary learning performs just as well. That would support
  the steering representation or adaptation idea, while rejecting the stronger evolutionary
  claim. Likewise, per-family controllers that work while one shared controller fails would
  support reusable methodology but not cross-family adapter transfer.

- **What failure would mean:** Several outcomes are scientifically different. No meaningful
  decoder effect means the interface lacks authority. Decoder effect without changed outcomes
  means numerical control is behaviorally weak. Widespread unfinished episodes may indicate a
  frozen-policy competence floor. Increased violations indicate harmful steering. EA losing to
  random or ordinary learning rejects its optimizer advantage. Success only on inspected tasks
  rejects the transfer claim. A timeout by itself diagnoses none of these.

- A negative result can still complete the study if the protocol and comparisons finish and all
  costs and outcomes are preserved. The research objective is a defensible answer, not an EA
  victory.

### Gaps

- The running Jar comparison must finish before the repository can say whether evolution,
  random search, the fixed bias or native execution wins its protected pilot. Four audit
  episodes per condition and one optimizer seed would still be exploratory evidence, not a
  definitive superiority claim. [Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)

- Paper-level evidence still needs a separately funded competent ordinary-learning baseline,
  replicated optimizer runs, enough reserved trials for uncertainty, and a cross-family bridge.
  Zero observed violations would not mean zero risk; with `n` independent episodes, the rough
  95 percent upper bound is `3/n`. [Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)
