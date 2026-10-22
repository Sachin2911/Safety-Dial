# Feynman-style teaching narrative for the Safety Dial project

## What is the clearest everyday analogy for the policy, monitor, controller, native noise and evolution?

### Takeaway

Imagine a skilled but inconsistent apprentice chef. The frozen ManiGuard policy is the
chef, native noise is the chef's own small roll of the dice when deciding exactly how to
move, the monitor is a hygiene inspector who only writes down violations, the controller
is a tiny recipe card that gently biases the dice, and evolution is the process of trying
many recipe cards, keeping the better ones and making small variations. The project asks
whether one small recipe-card maker can help several specialist chefs finish tasks safely,
without retraining any chef.
([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

### Cited Findings

- At the highest level, the supervisor's allocated topic is "Safe AI via Evolutionary
  Algorithms." It asks whether black-box optimization can balance reward and the
  probability of safety violations, especially when ordinary reward penalties and cost
  functions are hard to specify. This is the intellectual umbrella, not a claim that the
  current implementation has removed all hand-crafted trade-offs.
  ([Allocated topic](../../docs/allocatedTopic.md))

- The current concrete question is narrower: can a reusable bounded controller evolve how
  frozen family-specific pi0.5 vision-language-action policies sample their actions, using
  the policy's native flow-noise input and real simulator outcomes? Jar is the first
  scientific target and Clutter is the predeclared capacity fallback.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- Each ManiGuard family has its own frozen policy. The controller is meant to be shared
  across those policies and conditioned on observations and instructions. The target claim
  rules out a Jar-specific deployed controller, scene identifiers and family one-hot inputs.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- The current implementation uses the robot state, executed-action history and frozen
  projections of two camera images and the instruction. It excludes family identity, task
  identity, goal state and monitor state. The deployed controller has a 4 by 44 weight
  matrix, hence 176 bounded weights, rather than one literal scalar dial.
  ([Current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md);
  [controller implementation](../../experiments/helpers/maniguard_latent.py))

- The policy is a good analogy for the apprentice chef because it already understands the
  cameras and instruction and already proposes coordinated action chunks. It is not being
  relearned in this study. The controller only influences which behaviour the frozen
  generator samples.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- Native noise is best explained as the frozen policy's own private dice. The flow-based
  policy starts generation from Gaussian noise. The adapter preserves that native random
  draw and adds a bounded structured offset before the frozen policy decodes the action
  chunk. Zero controller weights use the unchanged native inference path, not a zero-noise
  policy.
  ([Policy wrapper](../../experiments/helpers/maniguard_latent_policy.py);
  [current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- The controller is a steering influence, not a replacement driver. It converts current
  features into four bounded coefficients, maps them through four fixed arm-supported
  directions, and adds the resulting offset to the policy's sampling noise. The frozen
  policy still performs the full image, language and action-generation work.
  ([Controller implementation](../../experiments/helpers/maniguard_latent.py);
  [current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

- The monitor is best explained as a referee with a clipboard. It observes the physical
  trace and labels temporal specification violations, but it does not alter the action.
  The current controller does not read the monitor during execution. Monitor labels and
  requested completion are used after the episode to evaluate a candidate.
  ([Six-hypothesis report](../../docs/research/reports/ManiGuard%20six-hypothesis%20study.md);
  [runtime implementation](../../experiments/helpers/maniguard_latent_runtime.py))

- Evolution is the recipe-card breeding process. Four candidate controllers are evaluated
  on physical cases. The evolutionary arm retains its best current candidate and mutates
  descendants. The matched random-search arm retains a best candidate but otherwise draws
  fresh controllers from the same bounded domain. Both start from the same four initial
  controllers and consume matched episode information.
  ([Candidate search implementation](../../experiments/helpers/maniguard_latent.py);
  [current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

#### A child-level version

There is already a robot brain that can see a scene, read an instruction and choose how to
move. Sometimes it succeeds, sometimes it breaks a rule and sometimes it simply runs out of
time. We do not open up and retrain that big brain. We add a tiny helper that nudges the
brain's own choice-making dice. We try different helpers in the simulator, keep the helpers
whose robots finish safely more often and change them a little to make new helpers. A
separate referee watches and records what happened.
([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md);
[current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

#### A secondary-school version

ManiGuard supplies specialist robot policies and a safety monitor. The policy maps cameras
and a language command to a short sequence of robot actions. The monitor checks the
resulting physical trajectory but is passive. The project adds a small conditional
controller before action decoding. That controller sees the robot, recent executed motion,
images and instruction, then nudges a few directions in the policy's random sampling
space. An evolutionary algorithm searches the controller weights using completion and
violation outcomes from complete simulator episodes.
([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md);
[current controller report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

#### An undergraduate version

Let `phi_t` be a 44-dimensional causal feature vector available immediately before a policy
query, `W` a bounded 4 by 44 controller matrix, `B` a fixed low-dimensional basis in the
policy's sampling space, and `epsilon_t` the policy's ordinary Gaussian flow-noise sample.
The controller supplies

`z'_t = epsilon_t + alpha B tanh(W phi_t)`, with `alpha = 0.35` in the pilot.

The frozen VLA decodes `z'_t`, the cameras and the instruction into an action chunk. Only a
prefix is executed before the next query. The episode is later scored from requested
completion and the raw monitor violation. Evolution changes `W`; it does not change the VLA
weights, monitor or task definition.
([Controller implementation](../../experiments/helpers/maniguard_latent.py);
[current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

### Inferences

- The deepest Feynman-style summary is: "We are testing whether a small reusable steering
  rule can reliably select better behaviour already latent inside several frozen robot
  policies." This wording separates the scientific hypothesis from the chosen optimizer
  and from any unproven claim of safety.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- The chef analogy should explicitly say that the inspector does not grab the chef's hand.
  Otherwise a listener will naturally but incorrectly infer that "ManiGuard" is an active
  runtime guard in this experiment.
  ([Six-hypothesis report](../../docs/research/reports/ManiGuard%20six-hypothesis%20study.md))

- "Dice steering" is clearer than "adding action noise." The intervention changes the
  policy's internal starting sample before decoding. It is not an arbitrary random force
  added directly to the robot after the policy speaks.
  ([Policy wrapper](../../experiments/helpers/maniguard_latent_policy.py))

### Gaps

- The present evidence shows that the noise interface can measurably alter decoded action
  prefixes in Jar and Clutter, but it does not yet show that the available alternatives are
  useful for safe completion. That is the purpose of the physical comparison.
  ([Current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

- Transfer of one controller across separately fine-tuned family policies remains untested.
  Equal latent tensor shapes do not imply that useful directions align across checkpoints.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

## How does information flow during one action and during adaptation across episodes?

### Takeaway

There are two loops. The fast loop repeatedly turns the latest observation and instruction
into a bounded noise nudge, asks the frozen VLA for an action chunk and executes a prefix.
The slow loop evaluates complete episodes, then evolution keeps or changes the controller
weights for future episodes. Safety labels travel through the slow evaluation loop, not
through the deployed controller's fast action loop.
([Runtime implementation](../../experiments/helpers/maniguard_latent_runtime.py);
[candidate search implementation](../../experiments/helpers/maniguard_latent.py))

### Cited Findings

#### Fast loop: one policy query and its executed actions

1. The simulator supplies the current camera images and measured robot state. The system
   also has the language instruction, the previous executed action and recent end-effector
   motion. These are causal inputs available before the next query.
   ([Runtime implementation](../../experiments/helpers/maniguard_latent_runtime.py);
   [feature implementation](../../experiments/helpers/maniguard_latent.py))

2. A frozen context encoder projects the overview image, wrist image and instruction. Those
   projections are concatenated with normalized joint positions, finger positions, the
   last executed arm action relative to the current joints, recent end-effector motion and
   a constant bias feature. The result has 44 entries.
   ([Feature implementation](../../experiments/helpers/maniguard_latent.py))

3. The small controller computes four bounded codes, `c_t = tanh(W phi_t)`. A fixed basis
   maps those four codes into the first seven model coordinates, repeats the shift across
   the policy's 16-position chunk and leaves other sampling-noise coordinates native.
   Model coupling can still indirectly change other decoded outputs.
   ([Controller implementation](../../experiments/helpers/maniguard_latent.py))

4. The policy service reproduces the native Gaussian sample and adds the structured offset
   for a nonzero controller. For the identity controller it calls the untouched native
   inference path, preserving the normal random-number advancement and numerical path.
   ([Policy wrapper](../../experiments/helpers/maniguard_latent_policy.py))

5. The frozen VLA sees its ordinary observation and the modified noise and returns an action
   chunk. The pilot retains a native model shape of 16 positions by 32 model coordinates,
   while the physical interface is seven arm joints plus gripper. These spaces must not be
   confused.
   ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md);
   [policy wrapper](../../experiments/helpers/maniguard_latent_policy.py))

6. The current pilot executes an eight-action prefix and then queries again. Executed action
   history, rather than only proposed action history, is fed into the next controller
   feature vector.
   ([Current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md);
   [runtime implementation](../../experiments/helpers/maniguard_latent_runtime.py))

7. The passive monitor observes the evolving simulator state and records violation
   telemetry. Its state is explicitly absent from the controller feature vector. At the
   episode end, requested-sequence completion and raw violation labels produce the outcome.
   ([Runtime implementation](../../experiments/helpers/maniguard_latent_runtime.py);
   [completion contract](../../docs/research/research_notes/ManiGuard%20phase2%20completion%20contract.md))

#### Slow loop: adaptation across episodes

1. Each candidate is one bounded matrix `W`, not a modification of the large VLA. The same
   four initial candidates and their physical observations seed EA and random search.
   ([Candidate search implementation](../../experiments/helpers/maniguard_latent.py);
   [current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

2. A candidate receives two physical case scores. The pilot score is
   `int(requested safe completion) - 2 * int(raw violation)`. Safe completion scores 1,
   a safe unfinished episode scores 0, and any violating episode scores -2, including a
   violating completion. The raw categories remain recorded and should not be collapsed
   into the scalar score when reporting results.
   ([Controller implementation](../../experiments/helpers/maniguard_latent.py);
   [completion contract](../../docs/research/research_notes/ManiGuard%20phase2%20completion%20contract.md))

3. In the EA arm, the best controller is retained and the other descendants are Gaussian
   mutations of the best two parents. In matched random search, the best is retained but
   the other candidates are fresh uniform draws from the same bounded weight domain. A
   reevaluated elite's latest score replaces its earlier noisy score.
   ([Candidate search implementation](../../experiments/helpers/maniguard_latent.py))

4. Finalists must be frozen before independent checking. The minimum comparison evaluates
   native, a tuned fixed correction, EA and random-search finalists. Ordinary learning is
   implemented as an interface but remains outside the current funded minimum comparison,
   and a stronger paper-level comparison still needs a competent latent-learning baseline.
   ([Current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md);
   [current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

5. Every simulator interaction matters, including unsuccessful candidates, elite
   reevaluations, interrupted attempts and development probes. Model queries and elapsed
   time are reported separately. Equal episode counts do not automatically imply equal
   interaction cost.
   ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md);
   [repository controls](../../AGENTS.md))

6. The bounded decoder audit has completed 320 calls. It established that all four tested
   directions changed decoded prefixes in both Jar and Clutter and that identity pairs
   matched. This proves interface authority and reproducibility, not safe physical benefit
   or cross-family transfer.
   ([Current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

7. At the latest report state, the full Jar comparison was still running. Shared
   initialization had completed with zero requested completions in all eight episodes, and
   no evolutionary or transfer benefit was claimed. These are development observations,
   not a final comparison.
   ([Current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

### Inferences

- A compact verbal flow diagram is:
  `cameras + instruction + robot/history -> 44 features -> four steering numbers ->
  modified native noise -> frozen VLA -> action chunk -> executed prefix -> simulator ->
  monitor and completion labels`. Across episodes, add:
  `labels -> candidate score -> retain/mutate or resample -> next controller`.
  ([Runtime implementation](../../experiments/helpers/maniguard_latent_runtime.py);
  [candidate search implementation](../../experiments/helpers/maniguard_latent.py))

- The separation between fast and slow loops is scientifically important. If live monitor
  labels entered the actor, the controller would receive a privileged information advantage
  and the project would become a different form of safety-aware feedback control.
  ([Completion contract](../../docs/research/research_notes/ManiGuard%20phase2%20completion%20contract.md);
  [repository controls](../../AGENTS.md))

- The score is a selection tool, not the complete result. Scientific reporting must retain
  safe completion, unsafe completion, safe unfinished and unsafe unfinished counts because
  one scalar can hide inactivity or violating success.
  ([Completion contract](../../docs/research/research_notes/ManiGuard%20phase2%20completion%20contract.md);
  [current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

### Gaps

- The current pilot is deliberately small. Four audit episodes per condition and one
  optimizer seed cannot establish superiority. A later main study needs replicated
  optimizers, appropriate uncertainty analysis, ordinary learning and separately budgeted
  cross-family evaluation.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- The current report records differing initial camera images even for nominally matched
  cases and seeds. The cause and impact of that variability remain part of the evaluation
  uncertainty and should not be hand-waved away.
  ([Current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

## How does this project relate to safe RL, world models, evolutionary algorithms and ROSARL, and which are context rather than implemented components?

### Takeaway

Evolutionary adaptation is the current optimization method, and safe completion in
ManiGuard is the current empirical target. Safe RL is the broader research field and an
essential source of ordinary-learning comparisons. Latent world models and ROSARL belong to
the project's intellectual history and surrounding research context, but neither a new
world model nor ROSARL is part of the current ManiGuard implementation.
([Repository authority](../../AGENTS.md);
[current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

### Cited Findings

#### Intellectual lineage

- The supervisor's starting question was broad: what can evolution teach us about AI that
  better respects human values, and can black-box optimization learn safe behaviour without
  relying only on delicately tuned reward penalties or cost functions?
  ([Allocated topic](../../docs/allocatedTopic.md))

- Earlier work in the repository investigated latent world models, safe filters and safe
  control in Push-T, Walker and UNISafe. That work is preserved as evidence and history,
  but the archived Walker methods, old run gates and old schedules are not the current
  ManiGuard plan.
  ([Repository authority](../../AGENTS.md);
  [plan history](../../docs/research/history/README.md))

- UNISafe produced a crucial causal lesson. Evolution improved completion from 2/64 for the
  native controller to 54/64, but a fixed hold achieved 55/64 and random search 56/64, with
  broadly similar violation counts. The simple grasp-maintenance rule, not evolutionary
  uniqueness, explained the major effect, and much of the improvement remained without the
  safety filter.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md);
  [UNISafe study](../../docs/research/reports/UNISafe%20grasp%20maintenance%20and%20evolutionary%20adaptation.md))

- The subsequent six-hypothesis ManiGuard study tested replanning, motion moderation,
  prerequisite guarding, grasp persistence, retreat and hand-back across 196 full episodes.
  It found useful implementation and evaluation problems but no repeatable simple repair
  that justified an optimizer study on those same knobs.
  ([Six-hypothesis report](../../docs/research/reports/ManiGuard%20six-hypothesis%20study.md))

- The earlier literal safety-dial pilot also did not improve fresh safe completion. The
  evolved full setting obtained 4/8 safe completions and 2/8 violating episodes, versus
  identity's 5/8 and 0/8. It lacked matched random search and ordinary learning.
  ([Earlier dial pilot](../../docs/research/reports/ManiGuard%20evolutionary%20safety%20dial%20pilot.md))

- Those failures motivate the current missing lever: influence action generation itself,
  through native sampling noise, instead of mostly clipping or delaying an action that has
  already been chosen.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

#### Safe RL

- The current study is safety-related reinforcement learning research because it evaluates
  task completion jointly with physical violations and compares an adaptive policy-side
  mechanism under interaction budgets. It is not currently training the base policy with a
  standard RL algorithm.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md);
  [current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

- Ordinary learning is not optional at paper level. Episodic REINFORCE on the same small
  parameterization asks whether the evolutionary update matters locally. A competent
  state-conditioned latent-learning method asks the stronger practical question of whether
  evolution is preferable to established learning in the same interface.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- The current objective includes a fixed violation penalty of -2. Therefore the study must
  not be advertised as penalty-free. Its more defensible feature is transparent selection
  from direct physical specification outcomes, with raw categories retained.
  ([Controller implementation](../../experiments/helpers/maniguard_latent.py);
  [current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

#### World models

- World models were central in earlier LeWM and UNISafe directions, where learned dynamics,
  latent prediction and safety evaluators had to be separated from readout error and
  physical outcomes. Those results remain methodological context.
  ([Repository authority](../../AGENTS.md);
  [safe manipulation review](../../docs/research/reports/Safe%20manipulation%20progress%20and%20robustness.md))

- The present phrase "latent steering" does not mean that the current method learns or
  rolls out a predictive world model. It manipulates the frozen policy's sampling inputs
  and uses frozen image/text representations. The current direction explicitly says these
  extensions do not require a new world model.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- Online evolutionary action search with a learned scorer or world model is listed as the
  eighth and least attractive immediate avenue because it adds a trustworthy-ranking
  dependency and repeated model queries. It is a prospective alternative, not the primary
  implemented route.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

#### Evolutionary algorithms

- Evolution is central because the simulator, contacts, temporal labels and complete
  closed-loop controller form a black-box objective, and candidate controllers can be
  evaluated without differentiating through the simulator. Ordinary RL also handles
  simulator nondifferentiability, so this motivation establishes relevance, not superiority.
  ([Safe manipulation review](../../docs/research/reports/Safe%20manipulation%20progress%20and%20robustness.md))

- The proposed evolutionary explanation is that nearby bounded controllers may create
  useful partly heritable behavioural differences. Selection can retain a useful difference
  and mutation can refine it under noisy episode feedback. This hypothesis needs direct
  evidence from lineage, behavioural changes and matched alternatives.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

#### ROSARL

- ROSARL is relevant background because the project sits in safe RL and the supervisor is
  connected to that line of work. Its result concerns reward design for unsafe terminal
  outcomes in an undiscounted stochastic-shortest-path setting under specific assumptions.
  It does not predict hidden physical hazards or guarantee safe intermediate exploration.
  ([Evolutionary opportunities report](../../docs/research/reports/Evolutionary%20opportunities%20for%20safe%20robotics.md))

- No measured ROSARL-specific manipulation deficit currently justifies adding it. The
  repository authority explicitly states that ROSARL integration is not an assumed
  component of the current implementation.
  ([Repository authority](../../AGENTS.md);
  [Evolutionary opportunities report](../../docs/research/reports/Evolutionary%20opportunities%20for%20safe%20robotics.md))

### Inferences

- A useful family-tree explanation is:
  `safe AI question -> safe RL and latent-model investigations -> UNISafe causal lesson ->
  ManiGuard execution-adapter failures -> current native-noise steering hypothesis`.
  Evolution remains central, but the current project moved from "find an optimizer" to
  "find a consequential behavioural lever, then test whether evolution uses it better."
  ([Allocated topic](../../docs/allocatedTopic.md);
  [current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- The project is not abandoning world-model or safe-RL interests. It is postponing extra
  predictive machinery until direct physical adaptation has a demonstrated useful signal.
  This makes the component attribution cleaner.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- GECCO is a plausible venue because the evolutionary mechanism must remain central, but
  venue fit depends on demonstrating an evolutionary explanation rather than merely using
  an EA library. The existing venue note itself flags that relevance to evolutionary
  computation must be explicit.
  ([GECCO notes](../../docs/gecco2027.md))

### Gaps

- The current minimum pilot does not include the competent state-conditioned ordinary
  learning comparison required for the strongest paper claim. The implemented episodic
  parameter-gradient interface is explicitly unfunded and is not equivalent to the stronger
  latent-learning baseline.
  ([Current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

- No current evidence shows that ROSARL would repair the observed ManiGuard failures. Adding
  it now would expand rather than complete the present scientific question.
  ([Repository authority](../../AGENTS.md);
  [Evolutionary opportunities report](../../docs/research/reports/Evolutionary%20opportunities%20for%20safe%20robotics.md))

## What is the intended scientific contribution and why is optimizer substitution alone insufficient novelty?

### Takeaway

The intended contribution is not "we used evolution on a robot." It is an explained,
budget-matched demonstration that one compact controller can steer wholly frozen specialist
VLAs toward better physical safe completion, and that evolutionary retention and mutation
provide an advantage over fixed steering, random search and ordinary learning per total
interaction. Every part of that sentence is a claim that must be tested separately.
([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

### Cited Findings

- Low-dimensional noise steering, state-conditioned noise for frozen generators,
  frozen-pi0.5 steering, evolutionary frozen-VLA action search and critic-free evolutionary
  VLA improvement all have close precedents. Therefore neither "steer latent noise" nor
  "use evolution with a frozen VLA" is sufficient novelty on its own.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- The proposed distinguishing combination is a wholly frozen base policy, one reusable
  compact observation- and instruction-conditioned controller, direct physical
  specification outcomes, and an explained safe-completion advantage per total interaction.
  The direction document describes this as a contribution to establish, not an already
  proven novelty claim.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- Four conclusions must remain separate: whether the steering interface improves the native
  checkpoint, whether conditioning improves on a constant bias, whether evolution improves
  on random search and ordinary learning, and whether the effect generalizes.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- The mandatory comparison logic follows directly. Native tests whether adaptation helps;
  a tuned fixed correction tests whether a simple rule explains the gain; matched random
  search tests whether evolutionary inheritance helps; ordinary learning tests whether the
  optimizer choice is practically useful; an independent audit tests whether development
  selection survives fresh cases.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- If the fixed setting explains the gain, that is a useful adaptation result but not an EA
  advantage. If EA and ordinary learning tie, the adapter may still be valuable but
  evolutionary superiority must be rejected. A positive EA result is not required for the
  study to be scientifically complete.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- Generalization has levels. Reusing the method while fitting separate controllers per
  family is weaker than freezing one controller across new tasks within trained families.
  The strongest stated target freezes that same controller and evaluates it with a
  family-specific base checkpoint from a family held out of adaptation. This is adapter
  transfer, not transfer of one universal base policy.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- The current 64-attempt pilot is a capacity-bounded screen, not a definitive superiority
  study. It can reject a broken interface, expose harmful physical effects or give an
  effect-size lead. Four audit episodes per condition and one optimizer seed are not enough
  for a final superiority claim.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- A standard EA can still support a worthwhile empirical paper when it reveals a
  reproducible failure, improves actual safe completion, survives strong comparisons and
  exposes its limits. Merely swapping the optimizer, improving imagined scores or finding
  prediction error leaves the central behavioural question unanswered.
  ([Safe manipulation review](../../docs/research/reports/Safe%20manipulation%20progress%20and%20robustness.md))

### Inferences

- The contribution should be stated as a causal ladder rather than one headline number:
  `the interface has authority -> it changes relevant behaviour -> conditioning matters ->
  evolution uses inheritance efficiently -> the result survives independent checking ->
  some part transfers`. A broken rung limits the claim but does not erase the evidence below
  it.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- The title "Safety Dial" should be treated as a project name, not as a promised calibrated
  knob from 0 to safe. The earlier three-setting dial was neither monotonic nor safer at
  higher strength, while the current controller is conditional and 176-dimensional.
  ([Earlier dial pilot](../../docs/research/reports/ManiGuard%20evolutionary%20safety%20dial%20pilot.md);
  [current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

- Complete accounting is part of the scientific contribution because an optimizer can look
  efficient if discarded candidates, reevaluations, probes or shorter episodes are hidden.
  Comparing performance at common control counts is more meaningful than quoting equal
  episode counts alone.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md);
  [repository controls](../../AGENTS.md))

### Gaps

- The full current optimizer comparison and its independent audit were not complete in the
  latest report, so there is no established evolutionary advantage yet.
  ([Current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

- Behavioural mechanism, ordinary-learning performance and cross-family transfer all remain
  open empirical questions. The decoder audit establishes influence but not usefulness.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md);
  [current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

## What misconceptions should be corrected, and which simple self-test questions reveal understanding?

### Takeaway

The safest explanation repeatedly distinguishes generating an action from monitoring its
outcome, completion from safety, useful steering from mere controllability, and adaptation
from optimizer superiority. A listener who can answer the self-test questions below without
using jargon has understood the project.
([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md);
[repository controls](../../AGENTS.md))

### Cited Findings

#### Misconceptions to correct explicitly

- **"ManiGuard blocks unsafe actions in this experiment."** False. The current native
  monitor supplies labels but does not change actions. Learned filtering was part of earlier
  UNISafe work and is not assumed in the current implementation.
  ([Repository authority](../../AGENTS.md);
  [six-hypothesis report](../../docs/research/reports/ManiGuard%20six-hypothesis%20study.md))

- **"The controller sees a warning and reacts."** False for the current primary route.
  Monitor state, goal flags, family identity and task identity are excluded from its inputs.
  It acts from robot/history, images and instruction.
  ([Current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

- **"Evolution retrains the robot's big neural network."** False. The family VLA remains
  frozen. Evolution searches only the compact controller matrix `W`.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md);
  [controller implementation](../../experiments/helpers/maniguard_latent.py))

- **"Noise steering means adding random shaking to the robot."** False. The adapter adds a
  bounded, observation-conditioned low-dimensional offset to the policy's native internal
  sampling noise before the frozen model decodes an action chunk.
  ([Policy wrapper](../../experiments/helpers/maniguard_latent_policy.py);
  [controller implementation](../../experiments/helpers/maniguard_latent.py))

- **"Zero weights mean passing zero noise."** False. The identity condition omits explicit
  noise and uses the untouched native inference call so that the baseline distribution and
  random-number path are preserved.
  ([Policy wrapper](../../experiments/helpers/maniguard_latent_policy.py))

- **"The 32-dimensional model action is the 8-dimensional robot command."** False. The
  current policy generates a 16 by 32 model-space chunk, while the physical command is seven
  arm joints plus gripper. Mapping and coupling must be audited.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- **"Completion means safe completion."** False. An episode can complete after a violation,
  and an unfinished episode can also violate. The earlier dial's full setting completed six
  tasks but only four safely. All four completion/safety categories must remain visible.
  ([Earlier dial pilot](../../docs/research/reports/ManiGuard%20evolutionary%20safety%20dial%20pilot.md);
  [repository controls](../../AGENTS.md))

- **"The benchmark goal always means the user-requested sequence was followed."** False.
  Some Jar episodes reached the spatial goal without first closing the jar. The project
  therefore retains a stricter requested-sequence completion label alongside native success.
  ([Completion contract](../../docs/research/research_notes/ManiGuard%20phase2%20completion%20contract.md);
  [six-hypothesis report](../../docs/research/reports/ManiGuard%20six-hypothesis%20study.md))

- **"No violation means the controller was safely competent."** False. The robot can remain
  inactive or fail to engage, yielding no violation and no completion. With zero violations
  in `n` independent episodes, the approximate 95 percent upper rate bound is still `3/n`.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md);
  [repository controls](../../AGENTS.md))

- **"The decoder audit proves physical safety."** False. It proves that bounded noise
  directions influence decoded action prefixes and that the interface is reproducible. It
  does not prove useful future trajectories, safety or transfer.
  ([Current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

- **"A timeout diagnoses the safety mechanism."** False. Timeout can reflect weak task
  competence, restrictive control, missing information, prediction error or an insufficient
  horizon. It is an outcome that requires diagnosis.
  ([Repository controls](../../AGENTS.md);
  [safe manipulation review](../../docs/research/reports/Safe%20manipulation%20progress%20and%20robustness.md))

- **"Matched seeds create identical counterfactual starts."** False. The current pilot found
  matching initial robot inputs but different overview and wrist images in six matched
  comparisons. Nominal case and seed matching does not itself prove identical observations.
  ([Current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

- **"Evolution must win for the project to succeed."** False. A fixed bias, random search or
  ordinary learner may be the correct winner. A defensible negative result can still locate
  the useful interface or reject the evolutionary explanation.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- **"This is already a world-model or ROSARL system."** False. Latent noise steering uses a
  frozen policy's sampling variables and representations. A new world model, a learned
  safety filter and ROSARL are not assumed current components.
  ([Repository authority](../../AGENTS.md);
  [current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- **"The old Phase 3 plans tell us what to run next."** False. They preserve earlier methods,
  results and stop decisions as history. The root direction supersedes their prospective
  instructions.
  ([Plan history](../../docs/research/history/README.md);
  [repository authority](../../AGENTS.md))

#### Simple self-test questions with expected answers

1. **In one sentence, what is the project?** It tests whether evolution can learn one small
   observation- and instruction-conditioned rule that steers the native sampling of frozen
   specialist robot policies toward more safe completions under a fixed interaction budget.
   ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

2. **Who acts and who only judges?** The frozen VLA plus compact controller generate actions;
   the ManiGuard monitor observes the resulting physical trajectory and labels violations.
   ([Runtime implementation](../../experiments/helpers/maniguard_latent_runtime.py))

3. **What exactly evolves?** The 176 bounded weights of `W`, not the base VLA, monitor,
   camera encoder or task definition.
   ([Current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

4. **Why steer native noise instead of clipping actions afterward?** It may expose different
   coordinated behaviours during action generation, whereas the previous cap and timing
   controls mostly constrained choices already made and rarely changed commands.
   ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

5. **What does the controller know at deployment?** Robot state, recent executed motion,
   two camera-derived feature vectors and the instruction. It does not know the family,
   task ID, goal flag or monitor verdict.
   ([Current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

6. **What are the four episode outcomes that must be reported?** Safe complete, unsafe
   complete, safe unfinished and unsafe unfinished, together with their denominators.
   ([Earlier dial pilot](../../docs/research/reports/ManiGuard%20evolutionary%20safety%20dial%20pilot.md);
   [repository controls](../../AGENTS.md))

7. **What did the 320-call decoder audit prove?** The selected bounded directions can alter
   decoded prefixes reproducibly in Jar and Clutter. It did not prove physical benefit or
   transfer.
   ([Current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

8. **Why are fixed bias and random search essential?** Fixed bias tests whether a simple
   constant rule explains the gain. Random search tests whether inheritance and mutation add
   value beyond sampling the same domain.
   ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

9. **Why is ordinary learning essential?** It tests whether the same observations and action
   domain can be learned as well or better without evolutionary selection.
   ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

10. **What would count as the strongest transfer result?** The same frozen controller,
    developed without one family, improves reserved tasks when paired with that held-out
    family's own frozen checkpoint. That is adapter transfer, not a universal checkpoint.
    ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

11. **Why is one safe run weak evidence?** Episode outcomes are noisy, and zero observed
    violations does not imply zero risk. Independent repetitions and uncertainty are needed.
    ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

12. **When would the project use the residual fallback?** If the audited native-noise
    interface cannot produce relevant bounded behavioural alternatives within its declared
    engineering allowance, a separately frozen bounded direct-action residual study is the
    designated fallback.
    ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

### Inferences

- A useful Feynman check is to ask Sachin to explain the project without saying "latent,"
  "VLA," "EA" or "safe RL." A sound answer is: "I keep the robot's main brain fixed, learn
  a small rule that nudges how it chooses among its own possible moves, and test whether
  keeping and varying good rules beats simple alternatives on both finishing and breaking
  rules."
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

- A second check is to ask him to point to every information boundary. If he cannot say
  which signals reach the controller, which reach only the evaluator, and which costs are
  counted, then he cannot yet defend the experimental comparison.
  ([Repository controls](../../AGENTS.md);
  [runtime implementation](../../experiments/helpers/maniguard_latent_runtime.py))

- A third check is counterfactual: "If fixed bias wins, what did we learn?" The correct
  answer is that native-noise steering may still be useful, but conditioning and
  evolutionary inheritance were not supported. This tests whether claims remain separated.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))

### Gaps

- The current report predates completion of the full comparison, so any spoken explanation
  must clearly separate implemented mechanism, completed authority evidence, running
  comparison and untested future claims.
  ([Current pilot report](../../docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md))

- The project does not yet know whether one shared conditional controller can transfer
  across family-specific checkpoints, whether persistence or response-informed coordinates
  help, or whether activation gating supplies a better interface. These are prospective
  extensions, not parts of the established method.
  ([Current direction](../../Evolutionary%20improvement%20on%20ManiGuard.md))
