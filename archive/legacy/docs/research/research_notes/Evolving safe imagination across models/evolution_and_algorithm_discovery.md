# Evolution and algorithm discovery for safe latent imagination

Research date: 7 October 2026. This note critiques the preferred candidate in
[the brainstorming record](../evolving-safe-imagination-20261007.md). It does not adopt
a new project direction or claim that the remaining questions are novel. Scope: the
evolutionary-computation contribution, reusable planner discovery, and the evidence needed
to distinguish structural discovery from algorithm configuration. Model integration and
non-evolutionary imagination agents are covered by the other research notes.

## What does evolutionary literature already cover?

### Takeaway

Evolving an agent's reusable search logic has a substantial history. Recent work also
discovers algorithms that use learned world models. The proposed combination requires a
more specific scientific contribution than evolution, latent planning, inspectable rules,
or transfer considered separately.

### Cited Findings

- **EVOCK, Aler, Borrajo and Isasi, Evolutionary Computation 9(4), 2001.** This paper
  explicitly separates evolving individual plans, entire planning programs, and control
  heuristics for an existing planner. EVOCK evolves the latter for PRODIGY4.0, allowing
  reuse across problems in a domain. Blocks-world and logistics results are reported.
  Its instance-based crossover injects material derived from planner traces. Thus,
  distinguishing an evolved planning procedure from an evolved action sequence is
  important, but that distinction itself is old. This study uses symbolic planning
  operators, rather than learned latent dynamics or safety costs. The introduction and
  method framing were inspected; the entire 34-page experimental analysis was not
  audited. [Author manuscript](https://e-archivo.uc3m.es/rest/api/core/bitstreams/1c55d9d2-574e-42b3-a8f3-c41b6ded31b0/content),
  [author bibliography confirming year and venue](https://plg.uc3m.es/daniel-borrajo/index.html).

- **Evolving Monte-Carlo Tree Search Algorithms, Tristan Cazenave, 2007 author
  manuscript.** GP constructs mathematical tree-development heuristics from arithmetic,
  statistics about moves and simulations, and constants. Invalid expressions that lack
  relevant move information are rejected. Individuals compete through Swiss tournaments.
  The paper reports improvements over UCT and RAVE, and evaluates a simplified evolved
  expression on larger Go boards and larger simulation budgets. This is direct precedent
  for reusable, inspectable rules controlling where a planner searches. It does not
  establish their utility with approximate latent dynamics or explicit safety events.
  Sections 4 to 6 were inspected. The accessible document is a dated manuscript, so a
  journal venue should not be invented.
  [Full text](https://www.lamsade.dauphine.fr/~cazenave/papers/evolveMC.pdf).

- **Towards Understanding the Effects of Evolving the MCTS UCT Selection Policy,
  Valdez Ameneyro and Galvan, SSCI 2022, arXiv 2023.** The study evolves alternative
  selection expressions online and examines function landscapes with different
  characteristics. Its conclusion is conditional: the evolved alternatives can help on
  multimodal or deceptive cases, while UCT remains strong on unimodal cases. The useful
  precedent is an explanatory experiment about when evolution helps, rather than assuming
  an evolved rule should dominate everywhere. Its online adaptation differs from a
  procedure evolved once and frozen for later deployment. Method section III and the
  conclusion were inspected.
  [Paper](https://arxiv.org/pdf/2302.03352),
  [author thesis bibliography confirming SSCI 2022](https://mural.maynoothuniversity.ie/id/eprint/18627/1/Fred_Valdez_Ameneyro_PhD_thesis.pdf).

- **GP2S: Search Strategy Generation for Branch and Bound Using Genetic Programming,
  AAAI 2025, preprint December 2024.** A node-scoring expression is evolved inside a
  fixed best-first search framework in SCIP. Terminals include depth, bounds and problem
  dimensions; operators are arithmetic with protected division. A size-sensitive
  tournament discourages program growth. Fitness uses solving time or solution gap.
  Experiments compare default and handcrafted strategies and a learned GNN strategy,
  including held-out larger instances. This establishes that small symbolic search rules,
  controlled overhead and generalisation are already legitimate algorithm-design goals.
  The authors explicitly classify their approach as learning algorithm configuration
  across instances. It is not latent control, and larger-instance transfer is not
  transfer across predictive models. Sections 4 and 5 were inspected.
  [Full method](https://arxiv.org/html/2412.09444),
  [AAAI proceedings paper](https://ojs.aaai.org/index.php/AAAI/article/download/33229/35384).

- **Evolutionary Planning in Latent Space, Olesen et al., November 2020 preprint.**
  EPLS learns a VAE and recurrent mixture-density model, with reward and termination
  predictions. Random-mutation hill climbing evolves an action sequence at each physical
  decision. Shift buffering carries the remaining sequence to the next decision.
  The CarRacing study also iteratively improves the world model using collected
  experience, and examines horizon and generation counts. The evolving object is the
  current plan, rather than reusable rules deciding how to construct plans. This is a
  necessary evolutionary latent-planning comparator, but the GP studies above are closer
  to the proposed discovery level. Sections 3 to 5 were inspected. This note cites the
  preprint; its later conference bibliographic details were not independently checked.
  [Full method](https://arxiv.org/html/2011.11293),
  [author implementation](https://github.com/two2tee/WorldModelPlanning).

- **World Models, Ha and Schmidhuber, 2018, and Deep Neuroevolution of Recurrent and
  Discrete World Models, Risi and Stanley, 2019.** The former evolves a compact controller
  using learned representations and recurrent dynamics, including training within an
  imagined environment. The latter evolves the world-model components from task fitness.
  These are important precedents for evolution integrated with latent models, but they
  should not be cited as identical to discovery of an explicit runtime query procedure.
  The World Models discussion also links to earlier work on a controller actively using
  a model for abstract reasoning.
  [World Models](https://worldmodels.github.io/),
  [Deep Neuroevolution paper](https://arxiv.org/abs/1906.08857).

- **Evolved Policy Gradients, Houthooft et al., NeurIPS 2018.** An outer evolutionary
  process learns a differentiable policy-training loss, with ordinary gradient learning
  inside each task. The loss uses temporal convolutions over experience and is evaluated
  by the resulting policy performance. Out-of-distribution task evaluation is reported.
  The reusable object is a learning rule. This makes it a closer precedent for the
  alternative model-learning idea than for a frozen agent's runtime planning logic.
  [Proceedings](https://proceedings.neurips.cc/paper/2018/hash/7876acb66640bad41f1e1371ef30c180-Abstract.html).

- **Evolving Reinforcement Learning Algorithms, Co-Reyes et al., ICLR 2021.** Typed
  computation graphs define a loss for value-based RL. Regularized evolution changes
  graph operations and wiring; a functional hash and a preliminary environment screen
  avoid some expensive evaluations. The work supports discovery from scratch or
  modification of DQN, and evaluates generalisation beyond the search environments.
  This is strong prior for an interpretable, transferable program genotype and for
  controlling the cost of outer search. The evolved object is an update rule, not the
  execution-time model-query policy. Sections 3.2 and 3.3 were inspected.
  [Paper](https://arxiv.org/html/2101.03958),
  [research-team account and venue](https://research.google/blog/evolving-reinforcement-learning-algorithms/?m=1).

- **AutoML-Zero, Real et al., ICML 2020.** Candidate algorithms are short programs
  operating on typed scalar, vector and matrix memory, with Setup, Predict and Learn
  functions. Their low-level operation vocabulary intentionally excludes ready-made ML
  algorithms. Evolution, functional-equivalence checks and small proxy tasks make the
  search more tractable. Search, candidate selection and final testing are separated;
  evaluation also changes data domains and dimensionality. Thus, primitive design,
  restricted languages and proxy-to-target transfer have established precedents.
  The method is supervised-learning discovery, so it does not settle which language is
  appropriate for sequential safety decisions. Sections 3 and the evaluation structure
  were inspected. [ICML paper](https://proceedings.mlr.press/v119/real20a/real20a.pdf).

- **Evolutionary Discovery of Reinforcement Learning Algorithms via Large Language
  Models, Sygkounas, Loutfi and Persson, GECCO 2026 EML.** Conference acceptance is
  independently verified by the official programme. The examined methods version is the
  March 2026 arXiv manuscript; publisher full text was unavailable through this search.
  [Official accepted papers](https://gecco-2026.sigevo.org/Accepted%2BPapers),
  [publisher DOI](https://doi.org/10.1145/3795095.3805180).

  LLM mutation and crossover evolve executable update rules. Its reported CG-FPD
  algorithm uses a learned latent model, CEM-refined plans, reward/termination scoring
  and a consistency term to train a feedforward execution policy. DF-CWP-CP uses
  confidence-weighted model rollouts for learning. These are evaluated algorithms,
  rather than appendix suggestions. Search uses five environments and five training
  seeds per candidate/environment. It reports ten generations, 24 candidates per island
  per generation, four A100s and about 30 hours per generation. Final evaluation adds
  five environments, but includes post-evolution hyperparameter tuning and peak
  checkpoint selection. That is not frozen runtime-planner transfer. The central
  distinction is training-time planning-derived learning versus deployment-time query
  control; latent imagination and confidence-aware scoring already occur in the former.
  Sections 3 to 5 and Appendix B were inspected.
  [Full method](https://arxiv.org/html/2603.28416).

### Inferences

The important taxonomy is what one individual represents: an action sequence, a physical
policy, a vector configuring a fixed planner, a rule selecting among planning operations,
or an executable learning algorithm. The proposed idea is most naturally a learned
hyper-heuristic or metacontroller. It should be evaluated at that level. Moving from
parameters to a syntax tree does not by itself turn configuration into a fundamentally
new algorithm. The distinction becomes meaningful when the language permits different
control flow and those differences explain a useful result.

Safety changes the desired outcome and may change the value of particular computations.
It does not erase the close algorithmic overlap with evolved search heuristics or learned
imagination controllers. Likewise, a common model API allows ordinary CEM and tree search
to operate across model implementations. Portability is evidence to demand, not a
novelty claim to accept automatically.

### Gaps

The papers checked do not establish that this exact candidate has already been evaluated,
but that absence is not evidence of an open gap. The GECCO 2026 article is particularly
important to compare against the eventual method. No author implementation was located
in its manuscript or the focused code search. Its final publisher version was not
retrieved. The inspected text leaves the comparative total discovery/HPO cost unresolved;
policy-network matching does not imply matching auxiliary-model or planning compute.

## Is grammar-based evolution justified, and what could its contribution be?

### Takeaway

A restricted program language is a plausible experimental choice if the question concerns
discovering and understanding planning structure. It is not justified solely by the need
to include an EA. The representation and evaluation must let us test whether structure
search contributes beyond numerical tuning or training a neural metacontroller.

### Cited Findings

- Prior GP planner studies expose a narrow learned component within a working search
  framework. GP2S keeps best-first selection fixed while evolving scores; Cazenave keeps
  MCTS while evolving development heuristics. These examples establish useful precedent
  for scoped rather than unrestricted planner synthesis.
  [GP2S](https://arxiv.org/html/2412.09444),
  [Cazenave](https://www.lamsade.dauphine.fr/~cazenave/papers/evolveMC.pdf).
- Learned algorithms also need not be symbolic. Learned Policy Gradient discovers a
  neural update rule and tests it beyond the small training environments. Consequently,
  symbolic interpretability and transfer must be demonstrated, rather than inferred from
  choosing evolution. [Oh et al., NeurIPS 2020](https://papers.nips.cc/paper_files/paper/2020/hash/0b96d81f0494fde5428c7aea243c9157-Abstract.html).

### Inferences

**Genotype choices have different scientific meanings.** A small arithmetic expression
ranking imagined nodes is tractable and interpretable, but much of the planner remains
hand-designed. A bounded typed program can choose branch, extend, reconsider and stop;
this better matches the motivating idea, with harder search and credit assignment.
A recurrent neural controller can condition on history, but makes mechanism extraction
harder and invites a direct comparison with RL-trained metacontrol. A fixed planner's
thresholds and horizon settings provide the necessary configuration baseline. None is
the default winner before the desired experiment is specified.

**The primitive language is part of the hypothesis.** If a primitive already means
"extend this trajectory until delayed danger is ruled out", the experiment has supplied
the purported discovery. More defensible operations would manipulate model-state
handles and perform one explicitly costed prediction, with conditions constructed from
available summaries. Even then, the designer supplies action proposals, reward/cost
readouts, memory operations and available observations. The report must identify those
choices and show what the evolved program added.

The language also needs bounded execution and defined behaviour when a calculation is
invalid or a model cannot extend a state. Type checks prevent a plan handle from being
used as a probability; they do not make the plan safe. An attractive evolved formula
can remain brittle under a change in cost scale, action duration or latent uncertainty.
Protected arithmetic is an implementation convention whose behavioural consequences
must be included in the assessment.

**Credit assignment and search cost are central.** An outer fitness measured from actual
episodes attributes a whole outcome to many internal decisions. Rare violations make
ranking noisy. Selecting from many programs can reward lucky evaluations, even with
fixed predictive models. Evaluation cases should be shared across candidates where
appropriate, with separate new cases for model selection and final testing. Repeated
simulator use remains charged. A cheap screening task is useful only if its rankings
predict final-task quality; screening can otherwise reject the very delayed reasoning
that motivates the project.

**Three evidential burdens follow.** First, compare against configuration of an existing
planner and random search in the same valid program space, at comparable outer budgets.
Second, compare against RL-trained metacontrol with the same observations, operations
and runtime computation. Third, simplify the selected program and intervene on its
important clauses. If removing branching logic does not affect decisions, the correct
claim concerns the remaining scoring rule. If gains come from a better proposal policy
or cost readout, they do not establish better imagination control.

A substantial GECCO result could be a new search representation/operator that addresses
a demonstrated difficulty of evolving planning procedures, or a carefully explained
discovery about how evolved procedures allocate computation under prediction error.
Neither requires claiming that evolution invented reasoning. Merely replacing RL with
CMA-ES, applying GP to a new benchmark, or wrapping two model families would need much
stronger evidence before being presented as the contribution.

### Gaps

There is currently no defined program space, no estimate of fitness variance, and no
evidence that program search beats configuration in this setting. Search-cost numbers
from other papers should not be treated as estimates for this project. The relevant
comparison is total discovery and deployment cost at a useful safety-performance level,
including failed candidates, adapter fitting and world-model learning.

## Which bounded research questions are worth testing against the literature?

### Takeaway

Two questions preserve the intellectual interest while exposing the proposal to failure.
They are candidate questions, not verified novelty claims or instructions to begin runs.

### Cited Findings

- Evolved MCTS studies already show that benefits depend on the search landscape, while
  EPLS shows that simply extending imagined horizons can cease helping. This motivates
  controlled questions about the circumstances in which a computation is useful.
  [MCTS analysis](https://arxiv.org/pdf/2302.03352),
  [EPLS](https://arxiv.org/html/2011.11293).
- Transfer of learned search rules and algorithmic programs already appears in GP2S,
  AutoML-Zero and evolved RL algorithms. Their existence raises the standard for claiming
  that a shared interface alone provides a new contribution.
  [GP2S](https://arxiv.org/html/2412.09444),
  [AutoML-Zero](https://proceedings.mlr.press/v119/real20a/real20a.pdf),
  [Evolving RL Algorithms](https://arxiv.org/html/2101.03958).

### Inferences

**Question 1: When does discovering query structure improve safe decisions beyond
tuning a planner's numerical settings?** Define a bounded planning language that can
express both existing search patterns and meaningful alternatives. Compare structural
search to a parameter-only subspace under equal discovery and runtime budgets, using
the same model, action proposals and safety information. Vary a small number of
interpretable task properties, such as delay before a hazard becomes predictable and
the concentration of hazards among promising plans. The potential finding is a
conditional relationship between those properties and the useful evolved structure.

The decisive evidence is more than a lower aggregate violation rate. We need to see
which model queries changed, whether they changed the physical decision, and whether
removing the responsible program component removes the benefit. Compare against a
learned metacontroller and simple state-dependent rules, including the relevant
adaptive-depth prior work covered elsewhere. Failure is informative: if a tuned horizon
or threshold explains the gain, structural discovery is unnecessary for that setting.
If an RL metacontroller discovers the same behaviour at lower cost, evolution needs a
different demonstrated advantage to remain central.

**Question 2: Which properties of an evolved planning rule survive changes in the
predictive model, and which depend on its errors?** Treat cross-model evaluation as a
test of mechanism. Hold task semantics and evaluation outcomes fixed while changing
predictive architecture or model quality through a declared interface. Freeze the
procedure and separate permitted adapter calibration from search. Compare its change
in useful performance against fixed and adaptive black-box planners that also transfer
through that interface.

The strongest result would identify a specific rule whose benefit survives relevant
changes, and explain a boundary where it fails. For example, an advantage from revisiting
an uncertain branch should diminish when the supplied uncertainty has no predictive
value. That is a falsifiable relationship, not a promise that an ensemble score is
calibrated. If rescaling inputs or replacing a backend silently changes the decision
semantics, apparent failure could be an interface artefact. If most intelligent choices
sit in the adapter, apparent success could be an adapter contribution. A single tuned
procedure per backend establishes compatibility, not transfer. Even successful frozen
transfer remains supporting evidence for the rule's mechanism, rather than novelty
by itself.

### Gaps

**Search record and limits.** Searches used web-accessible primary manuscripts,
proceedings, author pages and official venue records. Secondary search hits were used
only to locate primary sources. The targeted queries included:

- `Evolutionary Planning in Latent Space`; `genetic programming MCTS selection`;
  `evolving planning algorithms genetic programming`.
- `Towards understanding evolving MCTS`; `Evolving Monte Carlo Tree Search genetic`;
  `Learning to search genetic programming planning`.
- `Learning to Solve Planning Problems Efficiently`; `Search Strategy Generation for
  Branch and Bound genetic programming transfer`.
- `Discovering reinforcement learning algorithms evolution programs`;
  `Evolving Reinforcement Learning Algorithms Co-Reyes`; `Evolved Policy Gradients`.
- `genetic programming planning world model 2026`; `evolving metareasoning`;
  `genetic programming imagination planning`; `algorithm discovery planning evolution 2026`.
- Exact-title checks for the Sygkounas paper plus `github`, `GECCO`, and publisher
  identifiers established conference status and failed to locate an author code release.

A recent title that sounds close, **PathWise: Planning through World Model for Automated
Heuristic Design via Self-Evolving LLMs** (January 2026 preprint), uses LLM agents to
generate combinatorial-optimisation heuristics. Its world-model terminology describes
heuristic derivation in a reasoning system. Its abstract was screened, and it was not
treated as direct evidence of a physical agent operating through a latent dynamics
model. [Primary abstract](https://arxiv.org/abs/2601.20539).

This was a focused novelty-boundary review, not an exhaustive systematic review.
Detailed methods were inspected for EPLS, Cazenave, GP2S, evolved RL computation graphs,
and the GECCO 2026 discovery paper; neighbouring works received narrower screening.
Follow-up should check the final GECCO article and code, trace forward citations of
EVOCK and evolved MCTS, and compare any concrete candidate against the separate learned
imagination and safety reviews before declaring a research gap.
