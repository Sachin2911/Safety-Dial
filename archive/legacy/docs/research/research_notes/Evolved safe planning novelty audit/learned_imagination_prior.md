# Learned imagination, metareasoning and the novelty boundary

Audit dated 8 October 2026. This bounded update checks the proposed reusable controller
over latent-model queries, with evolution scored by simulator safety, progress and
computation. It does not establish novelty for an unspecified implementation.

## Which proposed planning capabilities are already established?

### Takeaway

Learning a reusable strategy for deciding how to imagine is established. Even continuing
to inspect a promising plan until an adverse prediction appears has a close empirical
precedent in Thinker. Evolution therefore needs a more specific contribution than replacing
the optimizer of an otherwise familiar metacontroller.
[Thinker, Appendix D.1](https://arxiv.org/html/2307.14993)

### Cited Findings

- **Learning model-based planning from scratch, IBP, 2017:** the manager chooses whether
  to imagine or act, and where imagination starts. The general formulation permits any
  previous real or imagined state. Crucially, the continuous experiments use the restricted
  choices act, root and latest imagined state, rather than experimentally demonstrating
  arbitrary tree-node selection. The manager is trained with REINFORCE using task loss
  plus imagination cost. A learned interaction-network model is trained separately from
  the manager; a maze experiment uses a perfect model. Flexible depth, branching through
  root resets and computation-aware execution are therefore clear precedents.
  [Methods sections 2-4](https://arxiv.org/html/1707.06170)

- **Metacontrol for Adaptive Imagination-Based Optimization, ICLR 2017:** a manager
  chooses which predictive expert to consult and when to execute, accounting for expert
  reliability and computational cost. Experts can be inaccurate learned predictors.
  The task is a contextual spaceship decision, not recurrent manipulation control.
  This already covers allocating queries among predictors with different costs.
  The previous local note called it a preprint; the author's publication page confirms
  ICLR 2017. [Methods](https://arxiv.org/html/1705.02670);
  [author's publication record](https://www.jesshamrick.com/publication/hamrick-2017-metacontrol/)

- **Thinker, NeurIPS 2023:** an RL agent learns model-interaction actions, including
  imaginary actions and resets to the real-state root, before executing a physical
  action. Original experiments typically fix the number of internal steps and impose
  a depth cap. More decisively, Appendix D.1 finds deep, narrow planning: agents tend
  to continue promising rollouts and reset when predicted quality deteriorates,
  including predictions of irreversible Sokoban states. This is directly relevant to
  the proposed illustrative plan-scrutiny behaviour. Its learned planning is exercised
  at decision time, not merely used to train an actor offline.
  [Methods section 4 and Appendix D.1](https://arxiv.org/html/2307.14993)

- **Dynamic Thinker, ALA workshop 2025:** adds a stop-search action and computation
  penalties to Thinker, learning variable search length. Experiments are small knapsack
  settings, where different task incentives produce different allocation patterns.
  Contextual stopping under explicit computation cost is already demonstrated.
  [Sections 2-4](https://ala-workshop.github.io/papers/ALA2025_paper_44.pdf)

- **Learning to select computations, 2017:** Bayesian metalevel policy search learns
  computation policies from value-of-information features, using Bayesian optimization
  over a compact weight vector. Selecting a computation for its effect on the eventual
  decision, rather than uncertainty alone, is established metareasoning.
  [Methods sections 2-3](https://arxiv.org/html/1711.06892)

### Inferences

- A vocabulary of branch, extend, inspect and stop does not establish novelty. Neither
  does the illustrative rule "investigate promising plans and stop when they become bad."
  We need a precise distinction from the existing controller and observed behaviour,
  particularly Thinker. [Thinker](https://arxiv.org/html/2307.14993)
- Optimizing the same controller with CMA-ES instead of RL is an optimizer substitution.
  A stronger claim would explain a useful structural discovery or a measurable capability
  that this search method enables. That judgement is an inference, not a publication rule.

### Gaps

The older methods do not by themselves establish the proposed combination of an evolved
external planning program, frozen fallible visual latent dynamics and simulator-measured
safety. Their differences cannot be converted into a claim that no later work combines
those ingredients.

## Do recent safety and latent-model methods close the remaining space?

### Takeaway

Risk-sensitive latent stopping is already close prior art, and a new September 2026
paper directly studies allocating latent computation where it changes candidate rankings.
These approaches have distinct control interfaces, so they constrain the claim without
being exact implementations of the proposal.
[RISE](https://arxiv.org/html/2608.20430);
[DeepJEPA](https://arxiv.org/html/2610.00368)

### Cited Findings

- **RISE, August 2026 preprint:** its latent evaluator estimates revealed risk and
  the planning gain obtainable from longer future prefixes. A gate chooses Roll/Stop.
  The gate is supervised using exhaustive depth-wise planning scores, with targets
  equal to the best remaining gain after deducting computation cost. Other modules
  are frozen during gate training. This is direct prior for decision-benefit-sensitive,
  safety-relevant latent imagination allocation. Its operation is sequential prefix
  continuation, not arbitrary external branch/program search. Section 5.4.3 adds the
  scheduler to DAWN without modifying DAWN's predictor/planner, but does not establish
  frozen scheduler-weight transfer. [Methods sections 4.3-4.4 and 10.3; transfer section
  5.4.3](https://arxiv.org/html/2608.20430)

- **DeepJEPA: Scaling World Models from Within, 30 September 2026 preprint:** a learned
  head controls how many recurrent updates to apply within each candidate's single
  imagined transition. It retains the outer CEM candidate budget, horizon and search
  procedure. The actual training target is marginal improvement in latent prediction
  error, not safety, task reward or measured decision value. A value-of-computation
  formulation motivates this proxy; the authors explicitly do not claim the head
  estimates true computation value. Experiments examine contact-aligned updates,
  candidate ranking changes and action changes. Thus decision-relevant latent compute
  is already a motivation and analysed mechanism, but this is internal predictor
  refinement rather than an evolved external search procedure. The arXiv record and
  HTML header agree on the 30 September date despite the October identifier.
  [Methods sections 3.1-3.3 and Appendix B](https://arxiv.org/html/2610.00368);
  [submission record](https://arxiv.org/abs/2610.00368)

- **When to Plan, July 2026 preprint:** PPO learns whether to use a reactive policy or
  invoke A* at one of several horizons, conditioned on reactive-policy uncertainty.
  Evaluations include Fetch reaching around obstacles. Methods explicitly use an
  accurate supplied world model; the uncertainty is about reactive competence, not
  uncertainty in learned latent dynamics. It arbitrates among action-generation options
  rather than choosing individual branches inside a latent search.
  [Sections 5-6](https://arxiv.org/html/2607.16421)

- **Effort Allocation for Deadline-Aware Task and Motion Planning, 2024:** a metalevel
  MDP allocates each planning time step to a plan skeleton. Model-based approaches
  and PPO are evaluated in navigation and manipulation. The learned uncertainty concerns
  planning/execution times, not visual prediction error. This nevertheless rules out
  a broad "first robot to learn which plan to investigate" claim.
  [Sections III, VI-VIII](https://arxiv.org/html/2410.05828)

### Inferences

There are three different allocation sites: external search-tree operations, temporal
rollout stopping, and computation inside one transition. Comparing them as if all only
choose a scalar rollout depth would conceal important prior capabilities. A distinctive
external search policy remains possible, but its benefit must survive stronger comparators
than uniform depth alone.

### Gaps

RISE's public README was reachable but contained no DAWN or transfer protocol. GitHub
repository/API retrieval failed, and local network retrieval was unavailable; no complete
code audit was possible. Frozen scheduler transfer remains unknown, not disproved.
[Public README](https://raw.githubusercontent.com/COOWAI/RISE/main/README.md)

## What novelty assessment is justified now?

### Takeaway

The defensible verdict is **established research family, potentially new specific method,
novelty not yet demonstrated**. Broad learned planning, safety-sensitive stopping and
decision-relevant compute allocation are already occupied. The evolutionary-discovery
review must separately establish what program evolution adds.

### Cited Findings

- Real task outcomes combined with internal computation costs already train a planning
  manager. That evaluation structure is sound, but is not itself a novelty claim.
  [IBP section 3.2](https://arxiv.org/html/1707.06170)
- Freezing predictive components while training a stopping mechanism already occurs.
  Frozen-model use alone does not distinguish the proposal. [RISE section 4.4.3](https://arxiv.org/html/2608.20430)
- Learned reactive/planning arbitration can use images in a robot-arm task without
  confronting learned-model error. We should explicitly identify whether our proposed
  advantage comes from handling fallible predictions. [When to Plan section 6](https://arxiv.org/html/2607.16421)

### Inferences

A stronger candidate claim would identify a particular failure of conventional allocation
under imperfect safety predictions, and a discovered sequence of computations that
reduces actual violations at matched progress and cost. For example, it might explain
when comparing a rival plan is more useful than extending the current favourite, or when
an additional uncertainty query should change an action. These are questions to test,
not presently established contributions.

Useful evidence would include equal-information, equal-budget comparisons; interventions
on the discovered rule; held-out situations; and separation of improvements due to
proposals, readouts or extra training from improvements in allocation. A new robot task
or a combination of familiar ingredients can support a contribution but cannot alone
demonstrate it. Physical observation gathering is not assumed part of the selected idea.

### Gaps

Searches updated the prior 7 October review using exact title/identifier checks plus
combinations of robot safety, metareasoning, model queries, uncertain dynamics and
computation allocation. They found DeepJEPA, When to Plan and deadline-aware TAMP as
additional neighbours. Primary methods were inspected for their precise interfaces and
objectives. This was not a citation-index-wide systematic review; no comprehensive
absence claim is supported. Adjacent LLM, task-planning and active-sensing results were
not treated as exact anticipations merely because their abstracts mention safety or
reasoning. Evolutionary program search is covered by the separate audit.
