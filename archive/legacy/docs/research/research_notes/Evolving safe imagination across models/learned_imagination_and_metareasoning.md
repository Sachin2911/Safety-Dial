# Learned imagination and metareasoning

Research assessment dated 7 October 2026. Scope: learned planning operations, computation
allocation, transferable deliberation, and the novelty boundary for the preferred
agent-focused candidate. This is a bounded primary-source investigation, not a systematic
review or an implementation proposal. The first five sources below were inspected beyond
their abstracts, including their methods and evaluation descriptions. This note does not
change the project's adopted direction.

## What can existing imagination controllers already choose?

### Takeaway

Almost the entire proposed operation vocabulary already exists in learned planners.
The closest references are IBP for flexible imagination trees, Thinker for learning
black-box model interactions, and RISE for safety-sensitive stopping in latent imagination.
These precedents constrain the claim; they do not establish that every useful procedure
within this space has already been discovered. [IBP](https://arxiv.org/html/1707.06170);
[Thinker](https://arxiv.org/html/2307.14993);
[RISE](https://arxiv.org/html/2608.20430)

### Cited Findings

| Closest work and status | Decisions learned | Training, evaluation, and precise overlap |
| --- | --- | --- |
| Pascanu et al., **Learning model-based planning from scratch**, 2017 preprint | Act or imagine; which previous real or imagined state to expand; proposed action; aggregation into plan memory | Continuous spaceship control uses a learned interaction-network model, REINFORCE for the manager and stochastic value gradients for controller/memory. Real task loss and imagination cost train the agent. A discrete maze study uses a perfect model. Branching, backtracking, adaptive depth, stopping and execution are already explicit. Methods §§2-3 and maze §4. [Full text](https://arxiv.org/html/1707.06170) |
| Hamrick et al., **Metacontrol for Adaptive Imagination-Based Optimization**, 2017 preprint | Number of internal optimisation steps; expert/model to query; whether to execute | A contextual-bandit spaceship problem, with one physical control followed by dynamics. Experts differ in reliability and expense. A REINFORCE manager selects experts; controller/memory use a differentiable critic; experts learn from on-policy outcomes. Selection among multiple predictive models and cost-aware stopping are established, although this is not sequential robot control. Methods §2 and experiments §3. [Full text](https://arxiv.org/html/1705.02670) |
| Chung, Anokhin and Krueger, **Thinker: Learning to Plan and Act**, NeurIPS 2023 | Imaginary actions, resetting to the root, and the eventual physical action | Transforms an MDP into one with model-interaction actions, then trains IMPALA. Typically 19 imaginary steps precede a real step, with maximum depth five. The model is a black-box tool to the actor. Observations include tree statistics and model hidden state; model/value/policy components are learned. Evaluation covers Sokoban and Atari. Learning actual planning behavior through external reward is already demonstrated. Methods §§4-5. [Full text](https://arxiv.org/html/2307.14993); [proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/file/4761fab863f0900d90cf601fce6d5155-Paper-Conference.pdf) |
| Xia, Wang, Chung and Greenwald, **Dynamic Thinker: Optimizing Decision-Time Planning with Costly Compute**, ALA workshop 2025 | Thinker search operations plus stop-search at an arbitrary internal step | Adds an explicit computation penalty and learns variable search length. Experiments use three small knapsack variants, compared with fixed-length Thinker, AlphaZero-style search and model-free IMPALA. Behavior changes with task incentives, including spending search earlier or later in an episode. This establishes contextual stopping and interpretable emergent allocation, not learned-model robotic safety. Methods §§2-4. [Workshop paper](https://ala-workshop.github.io/papers/ALA2025_paper_44.pdf) |
| Lu et al., **RISE: Adaptive Imagination for World Action Models**, August 2026 preprint | Sequential Roll/Stop, conditioned on predicted risk, benefit of continuing, and compute cost | A latent evaluator estimates risk and future planning gain. A supervised gate uses targets from exhaustive training-time depth evaluation. Driving results use nuScenes/NAVSIM; the scheduler is also integrated into DAWN without modifying its predictor/planner. This is direct prior for safety-sensitive latent depth allocation and multiple architectures. It does not search arbitrary imagination trees. Methods §§4.3-4.4, transfer §5.4.3. [Full text](https://arxiv.org/html/2608.20430) |

- **I2A, Imagination-Augmented Agents for Deep Reinforcement Learning**, 2017, learns
  how to interpret predicted trajectories using a rollout encoder. Its action-wise
  rollouts have prescribed construction rather than arbitrary learned tree traversal.
  Experiments include inaccurate models, unseen Sokoban box counts, and reusing a dynamics
  model for different MiniPacman rewards. The inaccurate-model experiment shows that
  learning to use imperfect predictions is itself established. Reuse of one dynamics
  model is different from transferring a frozen planner between model families.
  [Paper, §§4.2, 4.5, 5](https://arxiv.org/html/1707.06203)

- **MCTSnets, Learning to Search with MCTSnets**, Guez et al., ICML 2018, learns
  simulation traversal, evaluation and backup operations using vector memory in search
  nodes, trained by gradient optimisation. Its Sokoban results compare learned search
  against conventional MCTS. Consequently, claiming novelty from learning how to combine
  tree evidence, rather than only choosing a depth, would also need a more precise
  distinction. [Proceedings and paper](https://proceedings.mlr.press/v80/guez18a.html)

### Inferences

- The candidate's vocabulary is a useful design space, but not a contribution by itself.
  A convincing claim must name the procedure discovered, the reason existing procedures
  fail in the relevant setting, or a new discovery mechanism with substantive advantages.
  The table supplies the specific nearest comparisons.
- Replacing an RL optimiser with CMA-ES while preserving the same recurrent controller
  is a weak starting claim. It might become valuable if the evolutionary representation
  exposes a useful structural discovery, improves a measured search tradeoff, or makes
  a previously intractable objective manageable. Those benefits are currently hypotheses.
- Branching versus stopping is a meaningful distinction between methods. It would be
  misleading to describe every comparator as only a fixed-depth rollout method.
- Fixed model access does not imply fixed information access. A policy receiving
  learned values, proposal probabilities or rich latent features may have considerably
  more useful information than a comparator receiving only scalar risk estimates.

### Gaps

- No global absence claim is justified. Classical planning, meta-RL and evolutionary
  search each have wider literatures than this bounded review.
- RISE does not state whether scheduler/evaluator weights remain frozen during DAWN
  integration, so zero-shot transfer remains unresolved.
  [RISE §5.4.3](https://arxiv.org/html/2608.20430)
- RISE's arXiv history says 20 August 2026, versus 24 August on its title page;
  peer-reviewed venue unverified.
  [Submission record](https://arxiv.org/abs/2608.20430)

## What is already established about reusable planning and value of computation?

### Takeaway

Choosing computations by their expected effect on the eventual decision is an old,
formal research problem. The useful distinction for this candidate is between learning
to search more efficiently within a model and learning to make safer decisions when
that model is fallible. A shared model API establishes compatibility, while frozen
transfer is an empirical question. [Hay et al.](https://proceedings.mlr.press/r10/hay12a.html);
[Sezener and Dayan](https://arxiv.org/html/2002.04335)

### Cited Findings

- **Selecting Computations: Theory and Applications**, Hay, Russell, Tolpin and Shimony,
  UAI 2012, formalises selecting simulations using their expected improvement in action
  choice. Its Bayesian selection treatment distinguishes the value of information for
  choosing an action from conventional bandit reward collection. It develops stopping
  results and approximation heuristics. This is a foundation for query selection,
  not a learned-latent-model safety theorem. The publisher page is a 2026 reissue of the
  2012 work, not a new 2026 contribution.
  [Official proceedings](https://proceedings.mlr.press/r10/hay12a.html)

- **Learning to select computations**, Callaway et al., 2017 preprint, gives Bayesian
  metalevel policy search. Computations change a belief state; terminating computation
  obtains the value of the selected physical decision. The policy approximates
  computation value using information-value features and cost, with a small weight
  vector optimised using Bayesian optimisation. Its tests include stopping, allocation
  among alternatives, planning and emergency management. Thus even compact,
  interpretable black-box search over computation policies has strong precedent.
  [Methods §§2-3](https://arxiv.org/html/1711.06892)

- **Static and Dynamic Values of Computation in MCTS**, Sezener and Dayan, 2020
  preprint, addresses effects of later computations and later actions, rather than
  only one-step information gain. Crucially, §3.1 assumes known transition and reward
  dynamics: uncertainty comes from unfinished computation. Its guarantees cannot
  automatically be applied to epistemic error in a learned world model.
  [Methods §§3-4](https://arxiv.org/html/2002.04335)

- **On Learning to Think**, Schmidhuber, 2015 technical report, proposes a controller
  learning programs that query and exploit a recurrent world model, including abstract
  planning beyond ordinary forward simulation. It discusses evolutionary controllers
  and continued learning across tasks. This establishes conceptual precedence rather
  than an empirical cross-model safety result.
  [Report §5](https://arxiv.org/html/1511.09249)

- **World Models**, Ha and Schmidhuber, 2018, explicitly connects its evolved controllers
  to that earlier programme. Its future-work discussion describes controllers accessing
  world-model computations for more abstract reasoning. Therefore, a broad claim that
  evolution can discover how to think with a world model is not a new conceptual
  proposition. [Authors' article, related/future work](https://worldmodels.github.io/)

- **When and How Much to Imagine**, Yu et al., author project page labelled NeurIPS
  2026, describes AVIC-R: a GRPO-trained policy deciding whether and how much to query
  a world model, with QA correctness and imagination-cost feedback while the world model
  and QA model remain frozen. It concerns visual spatial reasoning, rather than robot
  safety. The project page was inspected; its venue label was not independently
  verified. [Author project](https://adaptive-visual-tts.github.io/)

- Two adjacent 2026 preprints show further crowding. Deng et al.'s **Efficient Agentic
  Reasoning Through Self-Regulated Simulative Planning**, submitted 21 May, uses an
  LLM world model and learned configurator for when/how deeply to plan.
  [Primary record](https://arxiv.org/abs/2605.22138) Liu et al.'s **Imagine-then-Plan**,
  submitted 13 January and revised 3 September, presents adaptive lookahead and both
  training-free and RL-trained variants. These were checked at abstract/record level
  only and are leads, not fully audited comparators.
  [Primary record](https://arxiv.org/abs/2601.08955)

### Inferences

- A computation is valuable because it changes a decision for the better. High
  predicted danger alone does not imply high computation value: an obviously bad action
  may be rejected immediately. Conversely, an apparently benign option may deserve
  investigation if its delayed consequences remain unresolved. This distinction is a
  useful conceptual test for any proposed safety-aware procedure.
- More samples from a systematically wrong model can increase confidence in a wrong
  decision. An investigation should distinguish uncertainty that model queries can
  reduce from missing knowledge that those queries cannot supply. No new theorem is
  claimed here.
- A portable procedure over reward, cost and tree summaries could avoid dependence on
  latent coordinates. However, standard tree search already operates through generic
  interfaces. Portability would need a demonstrated benefit attributable to the learned
  procedure, rather than merely successfully calling two backends.
- A richer adapter may improve portability by solving important inference problems
  itself. Keeping that intelligence visible is essential when explaining what evolution
  has discovered. A module translating every branch into a calibrated risk summary is
  already performing part of the decision task.
- It may be scientifically cleaner to distinguish generalisation to new task instances,
  new models of the same task, new model architectures, and new task families. These
  changes test different failure modes; a single transfer score would conceal them.

### Gaps

- The inspected literature does not establish whether a frozen safety-sensitive
  imagination procedure transfers between LeWM and Dreamer-family latent models with
  a strictly controlled interface. That is a bounded finding, not proof that nobody has
  attempted it.
- None of the inspected methods provides an automatic guarantee of real safety merely
  from observing safe imagined branches. The separate safety-model review must determine
  which additional assumptions could support such a claim.

## Which research claims remain defensible, and what needs further investigation?

### Takeaway

The broad concept is established; the open intellectual work is to identify a particular
failure in safety-relevant deliberation and a procedure that resolves it. The most
promising framing currently concerns the choice and use of counterfactual evidence
under limited computation and imperfect prediction. This is a candidate question,
not a verified novelty claim or recommendation to start implementation.

### Cited Findings

- The nearest baselines span different abilities: flexible tree navigation (IBP),
  black-box learned planning (Thinker), contextual stopping (Dynamic Thinker), and
  risk/benefit-sensitive latent depth (RISE). A novelty argument must cover this union
  rather than comparing only against CEM.
  [IBP](https://arxiv.org/html/1707.06170);
  [Thinker](https://arxiv.org/html/2307.14993);
  [Dynamic Thinker](https://ala-workshop.github.io/papers/ALA2025_paper_44.pdf);
  [RISE](https://arxiv.org/html/2608.20430)

- Learned interpretation of imperfect rollouts and structural search components are
  also precedents. Neither ignoring unreliable predictions nor learning a search
  backup rule is sufficient as a standalone novelty claim.
  [I2A](https://arxiv.org/html/1707.06203);
  [MCTSnets](https://proceedings.mlr.press/v80/guez18a.html)

### Inferences

**Claims to avoid at this stage:** first agent to learn how to imagine; first learned
branching/stopping policy; first risk-sensitive imagination scheduler; first use of
multiple predictive models; first learned or evolved general planning procedure. A
different optimiser, an added safety penalty, and another compatible backend are
ingredients rather than a complete scientific argument.

**Three bounded questions worth retaining:**

1. **Which imagined evidence changes unsafe decisions?** A procedure might allocate
   effort to extending an apparently successful plan, testing an alternative action,
   or checking a branch with unresolved consequences. The contribution would need to
   identify the discovered rule and show why its choice of evidence matters. Simply
   showing longer rollouts near hazards would be weak.
2. **Can useful deliberation be separated from a particular model's errors?** A
   discovered procedure could exploit idiosyncratic prediction errors or risk-score
   scales. Holding the procedure fixed while changing the predictive backend could
   reveal that dependence. Success would be especially informative if stronger
   conventional search through the same interface did not obtain the same benefit.
3. **Does the procedure improve the safety-performance-computation frontier?** An
   apparent benefit could arise from reduced motion, superior action proposals, extra
   queries, better readouts or more outer-loop training. The research question should
   survive these alternative explanations before a complex agent architecture is
   justified.

These are separable questions. It would be premature to require every one in a first
paper or combine learning objectives, model selection, planning, readout calibration and
evolution into one simultaneous optimisation problem. The user is exploring freely;
scope should follow the strongest identifiable problem rather than an inherited deadline.

Evidence of an interesting mechanism would involve repeatable internal behavior,
interventions showing that behavior changes physical decisions, and useful performance
on held-out circumstances. Attractive rollout visualisations alone cannot establish
that a procedure caused a safety improvement. Similarly, a complex discovered program
is not automatically better than a short threshold rule.

The evolutionary-discovery review is necessary before deciding whether program search
adds something substantial. This note does not independently assess all genetic
programming and automated algorithm discovery precedents.

### Gaps

**Search record and limits.** Primary-source searches conducted on 7 October 2026
included exact titles/IDs for IBP, Hamrick metacontrol, Thinker, RISE, Dynamic Thinker,
I2A, Learning to Think, MCTSnets and metalevel policy search. Additional queries combined
learned planning with transfer/world models, metareasoning with safety/model error,
adaptive planning with 2026 world models, and RISE with frozen scheduler transfer.
References and related-work passages supplied backward links; searches supplied
Dynamic Thinker and the 2026 adjacent leads. No citation-index-wide forward sweep was
performed. Eleven batched web calls were used, with primary methods inspected for the
five closest sources and selected formal foundations.

The searches found substantial positive prior art. They did not establish comprehensive
absence of an evolved safe metaplanner or a frozen cross-architecture transfer study.
The next literature question should be tied to a precise failure mechanism and proposed
representation, because the broad keywords now return several mature research lines.
