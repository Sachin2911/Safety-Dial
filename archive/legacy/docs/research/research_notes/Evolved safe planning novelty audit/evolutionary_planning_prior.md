# Evolutionary discovery of reusable planning procedures

Research date: 8 October 2026. Target: a reusable procedure that controls branching,
extension, uncertainty queries and stopping in a frozen latent world model, selected
using simulator task and safety outcomes under explicit computation budgets. This note
updates yesterday's review; it does not adopt an implementation or claim completeness.

## Has evolution already discovered planning strategies rather than action sequences?

### Takeaway

Yes. The broad distinction between evolving a plan and evolving a reusable planner or
search controller is established. The closest precedents include classical planning
control, evolved MCTS heuristics, and grammar-based discovery of entire search procedures.

### Cited Findings

- **EVOCK, Aler, Borrajo and Isasi (2001).** Individuals are sets of control rules for
  the symbolic PRODIGY4.0 planner, rather than action sequences. Its fitness actually runs
  the planner on training problems. The hierarchical performance criterion prioritizes
  problems solved more efficiently than the baseline, total problems solved, and expanded
  nodes; further criteria favour suitable, compact rules. Testing includes harder and
  larger blocks-world and logistics problems. Runtime uses STRIPS domain operators,
  not learned latent predictions. This is direct precedent for outer evolution selecting
  reusable internal decision logic by its effect on task solving and search cost.
  Inspected sections 1, 4.4 and 5.1. [Author manuscript](https://e-archivo.uc3m.es/rest/api/core/bitstreams/1c55d9d2-574e-42b3-a8f3-c41b6ded31b0/content).

- **Cazenave, Evolving Monte-Carlo Tree Search Algorithms (2007 manuscript).** GP evolves
  arithmetic expressions controlling MCTS tree development using move and parent
  statistics, including reward means, variability and visit counts. Fitness comes from
  games between candidate agents in Swiss tournaments. Search remains MCTS/RAVE; evolution
  changes its heuristic. Experiments use fixed playout counts and test a simplified
  evolved rule on larger Go boards and increased budgets. Thus inspectability, reuse,
  outer evaluation through complete behaviour, and transfer across budgets already occur.
  The examined setup uses game simulation, without the learned latent dynamics or
  physical safety costs in our proposal. Inspected sections 4-7. The author's bibliography
  dates the manuscript to 2007; no journal venue is inferred.
  [Paper](https://www.lamsade.dauphine.fr/~cazenave/papers/evolveMC.pdf),
  [author bibliography](https://www.lamsade.dauphine.fr/~cazenave/index.php).

- **GP2S (AAAI 2025).** GP evolves an arithmetic node-scoring expression inside a fixed
  best-first branch-and-bound framework in SCIP. Fitness uses actual solving time, or
  solution quality under a time limit. Inputs include node depth, bounds and problem
  dimensions. Size-sensitive selection controls expression growth. Evaluations include
  held-out larger instances and longer solving budgets. These are mathematical
  optimization problems, not trajectories in learned dynamics; optimization feasibility
  should not be equated with physical safety. Reusable symbolic search rules and
  computation-sensitive evaluation are established contributions, not new merely because
  we use latent nodes. Inspected method and evaluation.
  [Paper](https://arxiv.org/html/2412.09444),
  [proceedings](https://ojs.aaai.org/index.php/AAAI/article/download/33229/35384).

- **Maes, Lupien St-Pierre and Ernst (2012), Monte Carlo Search Algorithm Discovery
  for One Player Games.** This especially close structural precedent composes procedures
  using `simulate`, `repeat`, `lookahead`, `step` and `select`. Discovery uses multi-armed
  bandits, rather than evolution. Candidates are evaluated by solution quality across
  training problems under explicit trajectory-evaluation budgets; an experiment also
  fixes CPU time. Tests change problem distributions and budgets. The runtime problems
  have provided transition and reward functions, not learned latent models. Section VI
  explicitly distinguishes composing search algorithms from earlier expression-only GP.
  A bounded language that composes planning operations therefore needs a contribution
  beyond its existence. Inspected sections II-IV and V-VII.
  [Paper](https://arxiv.org/pdf/1208.4692).

- **Rollout-policy precedents also matter.** EvoMCTS (Benbassat and Sipper, 2013)
  evolves board-state evaluators used to guide MCTS rollouts, with fitness from complete
  games and comparisons using equal search amounts. Fast Evolutionary MCTS (Lucas,
  Samothrakis and Perez, 2014) instead updates rollout-policy weights from individual
  simulated returns during search, and also tests reusing pre-evolved weights across
  runs. These are different evolutionary timescales, not interchangeable precedents.
  The latter's knowledge-based extension works under a 40 ms game decision budget.
  [EvoMCTS](https://www.moshesipper.com/pubs/evomcts_players.pdf),
  [Fast Evolutionary MCTS](https://repository.essex.ac.uk/11981/1/Lucas.pdf),
  [knowledge-based extension](https://repository.essex.ac.uk/11983/1/KBFastEvoMCTS_CIG2014.pdf).

### Inferences

- Our genotype must be described precisely: scoring expression, fixed-planner settings,
  rollout policy, or program composing operations. Those objects face different prior art.
- Neither complete-episode fitness nor retaining the resulting procedure at deployment
  separates the broad proposal from evolved search heuristics.

### Gaps

No claim is made that these papers evolve every MCTS component, particularly backup
operators, or match all proposed uncertainty-query actions. Their demonstrated overlap
is already sufficient to reject broad novelty.

## How close are latent-world-model and recent algorithm-discovery papers?

### Takeaway

Evolution combined with latent planning is established, and GECCO 2026 already includes
discovered algorithms using learned world models. Deployment-time query control remains
a meaningful distinction from these particular model-based learning algorithms.

### Cited Findings

- **Evolutionary Planning in Latent Space, Olesen et al. (2020 preprint).** A VAE and
  recurrent mixture-density model predict latent dynamics, rewards and termination.
  Random-mutation hill climbing searches action sequences using predicted cumulative
  reward. Shift buffering carries a partial plan between physical decisions. CarRacing
  evaluation uses random tracks; the study varies planning horizon/generations and
  improves the model with further experience. Its evolving individual is the current
  plan, not reusable planning logic. The reported objective is task reward, rather than
  a separate physical-safety constraint. Inspected sections 3-5.
  [Paper](https://arxiv.org/html/2011.11293).

- **Sygkounas, Loutfi and Persson, GECCO 2026.** LLM mutation/crossover evolves
  executable training rules. Fitness aggregates best-checkpoint returns over five
  environments and five seeds. Its CG-FPD algorithm learns latent dynamics and uses
  CEM-refined plans as a teacher; execution during training and evaluation is a
  feedforward policy. DF-CWP-CP uses confidence-weighted differentiable model rollouts
  for learning; Appendix B specifies dynamics directly in observation space. It should
  not automatically be labelled a second latent planner. Final evaluation adds five
  environments after further hyperparameter tuning, so this is not frozen planner
  transfer. Explicit safety constraints and budgeted runtime query control are not the
  inspected evaluation's target. Search uses four A100s and reports approximately
  30 hours per generation, showing why its discovery budget is no estimate for ours.
  Method evidence is the March arXiv v1; acceptance is independently verified.
  [Method](https://arxiv.org/html/2603.28416),
  [official programme](https://gecco-2026.sigevo.org/Accepted%2BPapers).

- **HeurEvo, 28 September 2026 preprint.** It co-evolves high-level algorithm
  compositions, executable implementations and reusable components, including stage
  ordering, stopping rules and allocation of runtime. Candidates are evaluated on
  optimization instances under a two-minute execution budget. Synthetic-task evaluation
  includes held-out instances; MIPLIB evaluation has no reported validation/test split.
  Its operations include heuristics and mathematical solvers. This is not a latent robot
  planner, but adds recent direct overlap with reusable budget-allocation programs.
  Inspected sections 1, 3 and 4.1.
  [Preprint](https://arxiv.org/html/2609.36303).

- **Cazenave (2026)** systematically generates small PUCT/SHUSS exploration expressions
  and tests them in Go. Only the publisher abstract was inspected, so operator,
  transfer and safety details remain unverified. It reinforces that automated search-rule
  design remains active work. [Publisher](https://doi.org/10.1177/13896911261463788).

### Inferences

- Evolving an algorithm that uses uncertainty or a learned model is insufficient as a
  novelty claim. Our sharper object is the runtime allocation of queries to an imperfect
  model, judged by physical consequences.
- Adding safety, robotics or a new model family can motivate a new problem without
  establishing a new solution or a useful evolutionary contribution.

### Gaps

The GECCO publisher page remained inaccessible through DOI and direct ACM access.
Focused title, acronym and author searches did not verify released algorithm code.
The author's public GitHub profile was found, but no repository could be established as
this paper's implementation. Do not claim the final article or code has been audited.

## What novelty claim survives this targeted check?

### Takeaway

The broad idea is not novel. A specific contribution remains possible, but is unverified:
discover and explain a useful way to allocate imperfect latent predictions for safer
physical decisions, with evidence that structural evolutionary discovery matters.

### Cited Findings

The strongest four sources to put beside a concrete method are EVOCK, Cazenave's
evolved MCTS, Maes's compositional search discovery, and GECCO 2026's evolved
model-based learning algorithms. They jointly cover reusable control logic, outer task
evaluation, search-program composition and evolution with learned models.
[EVOCK](https://e-archivo.uc3m.es/rest/api/core/bitstreams/1c55d9d2-574e-42b3-a8f3-c41b6ded31b0/content),
[MCTS](https://www.lamsade.dauphine.fr/~cazenave/papers/evolveMC.pdf),
[compositional discovery](https://arxiv.org/pdf/1208.4692),
[GECCO 2026](https://arxiv.org/html/2603.28416).

### Inferences

Evidence worth seeking would isolate when additional model queries change an imminent
decision, whether the change prevents simulator violations without eliminating useful
progress, and why the discovered structure succeeds. Compare against tuning a fixed
planner, simple adaptive rules, random search in the same program space, and a learned
metacontroller with the same observations and operations. Separate discovery cost from
decision-time cost. If a simple tuned rule explains the gain, report that finding rather
than claiming discovery of a new planning paradigm.

### Gaps

Searches included: `"genetic programming" "planning" "world model"`,
`evolve "imagination" "planning" controller`, `"evolution" "metareasoning" planning`,
exact EVOCK/GP2S/EPLS/MCTS titles, `"Fast Evolutionary Adaptation for Monte Carlo Tree
Search"`, `Maes grammar Monte Carlo tree search algorithms automatic design 2012`, and
the GECCO title/CG-FPD plus author/GitHub queries. HeurEvo was a collaborator lead.

No exact match to all proposed components was established. This focused search cannot
prove none exists, and the idea still lacks a chosen genotype and mechanism. Adjacent
learned-imagination, active edge-evaluation and safety metareasoning literature must be
combined with this audit before claiming novelty.
