# Claim boundaries for evolved safe plan scrutiny

## Which broad claims are already established?

### Takeaway

Assessed on 8 October 2026: the broad idea is not novel. Learning which parts of a promising plan to investigate, controlling the balance between search and expensive checks, and evolving reusable computation strategies all have substantial precedents. A narrower contribution involving fallible latent predictions remains possible, but is not established by combining these ingredients.

### Cited Findings

- **Learned scrutiny of a promising plan:** *Leveraging Experience in Lazy Search* learns an edge-selection policy inside LazySP. The planner first finds the shortest potentially feasible path, then selects an edge to check and updates its search. Useful checks invalidate unattractive paths efficiently, including by eliminating future alternatives. STROLL learns through imitation of training-time oracles. The original preprint was submitted **16 July 2019**; a separate extended preprint was submitted **10 October 2021**, explicitly marked as accepted by *Autonomous Robots*. The inspected extended formulation receives binary edge-validity outcomes, rather than learned latent forecasts. [Original record](https://arxiv.org/abs/1907.07238), [extended record](https://arxiv.org/abs/2110.04669), [extended method, sections 2 and 4](https://arxiv.org/html/2110.04669)
- **Decision-relevant expensive checks:** *Bayesian Active Edge Evaluation on Expensive Graphs*, IJCAI 2018, chooses edge evaluations using their expected value for determining path feasibility. It exploits correlations between edges instead of merely checking them in order. Applications include manipulators and helicopters. Thus, allocating checks according to what they can establish about candidate plans is older than latent world models. [Official proceedings](https://www.ijcai.org/proceedings/2018/0679.pdf), [author institution record](https://publications.ri.cmu.edu/bayesian-active-edge-evaluation-on-expensive-graphs)
- **Search versus verification allocation:** *Generalized Lazy Search*, published **5 July 2019** in ICAPS, explicitly toggles between expanding search and evaluating edges. Its objective accounts for both forms of computation; it includes results in 7-DoF manipulation. [Official proceedings record](https://ojs.aaai.org/index.php/ICAPS/article/view/3543)
- **Safety metareasoning:** Svegliato and colleagues' *Metareasoning for Safe Decision Making in Autonomous Systems* uses parallel safety processes and an arbitration procedure that modifies task behaviour while balancing hazard severity and interference. This is not an evolved imagination controller, but it prevents claiming that adding safety to meta-level control is itself new. [Author manuscript, sections III and IV](https://static1.squarespace.com/static/5ea7560899ee71027ae31962/t/627298a27ee21b5d82e34bb2/1651677359982/ICRA22_Safe_Metareasoning.pdf)
- **Evolution of computation structure:** The **28 September 2026 preprint** *HeurEvo* evolves reusable hybrid algorithms, including their component choices, execution order and runtime allocations. Its domain is mathematical optimization, not robot control. This is a secondary precedent against treating evolved composition and runtime allocation as an untouched general idea. [Preprint](https://arxiv.org/abs/2609.36303), [method](https://arxiv.org/html/2609.36303)
- **Decision-relevant latent computation:** *DeepJEPA*, submitted **30 September 2026**, learns whether another recurrent update is worth making for each candidate and rollout step. Its stated motivation is that additional computation matters where latent refinements can change candidate ranking and action choice. Another researcher audits the mechanism and fixed outer planner in detail; this source already rules out claiming the general motivation as new. [Author abstract and date](https://arxiv.org/abs/2610.00368)
- **Observed selective scrutiny:** Thinker's behavioural analysis reports that learned agents continue promising rollouts deeply and reset after poor predicted outcomes. This is empirical precedent for the proposed motivating behaviour, not merely an overlap in available operations. The currently retrieved HTML labels this **Appendix F**, especially its reset-action analysis. [Thinker](https://arxiv.org/html/2307.14993)

### Inferences

- The phrase “scrutinise promising plans” is a helpful explanation of the idea, but it does not isolate a literature gap. Lazy search is an especially direct conceptual predecessor.
- A population of reusable planning procedures differs from a population of action sequences. That distinction positions the work in algorithm discovery and metareasoning; it does not establish novelty within those fields.
- An application of established ideas can still contribute valuable evidence. However, “use genetic programming instead of RL, then add robot safety penalties” would presently read as an optimizer/application substitution, without a demonstrated methodological or explanatory advance.

### Gaps

- This focused intersection search did not identify an exact implementation of all the proposed ingredients. That is a search result, not evidence that no such implementation exists.
- The present candidate has no fixed operation vocabulary, genotype or discovery mechanism. An exact novelty comparison is therefore not yet possible.

## Where might a substantive contribution remain?

### Takeaway

One potential distinction from geometric lazy search is the evidence being purchased: latent predictions and uncertainty estimates can both be wrong. Thinker already reasons through learned predictions, so fallibility alone is not a gap. A useful contribution would need a specific unresolved mechanism or explanatory result about safety under that fallibility.

### Cited Findings

- In STROLL's inspected formulation, checking an edge reveals its validity under the current world. Its Bayesian analysis also states realizability limitations. Replacing this operation with a noisy prediction changes what a check can establish, rather than merely changing its computational implementation. [STROLL, sections 2 and 6.5](https://arxiv.org/html/2110.04669)
- UNISafe explicitly targets optimistic predictions of unobserved failure. Its separate ensemble estimates epistemic uncertainty, which enters the latent safety construction. It therefore already addresses unreliable latent safety evidence, although its contribution is a safety filter rather than evolutionary discovery of runtime investigation logic. [UNISafe, sections 4 and 5](https://arxiv.org/html/2505.00779)
- Generalized Lazy Search accounts for both search and evaluation costs. Consequently, counting heterogeneous operations is necessary for interpreting a result, but doing so alone would not establish a new problem. [GLS](https://ojs.aaai.org/index.php/ICAPS/article/view/3543)

### Inferences

Two candidate contribution formulations are defensible **hypotheses**, not novelty claims:

1. **Discovery of a specific mechanism for handling conflicting safety evidence.** “An evolved procedure reconciles attractive nominal forecasts with inconsistent or unreliable uncertainty evidence, and its particular ordering of additional checks explains fewer physical failures at comparable progress and cost.” The unresolved object is that reconciliation mechanism, not generic deep investigation of promising plans. Its structure is currently unspecified; a learned risk score, conventional metacontroller or tuned threshold might reproduce any observed benefit. Evolution would need to discover an identifiable feature beyond behaviour already reported by Thinker.
2. **An explanatory result about when safety changes optimal computation allocation.** “For a fixed imperfect world model, procedures selected on physical task outcomes allocate investigation differently from procedures selected on prediction accuracy or raw uncertainty, and this difference explains which failures they avoid.” This could be a useful scientific contribution even without a wholly new evolutionary optimizer. Its limit is that a conventional constrained metacontroller may produce the same behaviour; the EC contribution would then need honest positioning rather than assuming that evolution is essential.

Neither requires a claim of guaranteed safety. Nor does either imply that querying a model repeatedly can expose errors that all its predictions consistently conceal. Transferring a frozen procedure to another task-matched model might strengthen the evidence for reusable planning logic, but a shared interface or separate retraining for each backend would not establish that result.

### Gaps

- We do not yet know whether accessible uncertainty signals distinguish the dangerous attractive plans from the harmless ones in the intended environment.
- There is no demonstrated reason yet that evolutionary search finds better planning logic than other discovery methods in this proposed space. This is an unanswered research question, not an objection that can be resolved by terminology.

## What must be specified before calling the idea novel?

### Takeaway

The next conceptual step is a concrete definition of one evolving individual and one decision it can make. More combinations of model names, safety penalties or environments will not resolve the current ambiguity.

### Cited Findings

- GLS separates an event rule, which determines when evaluation interrupts search, from an edge selector. This illustrates why saying “a planning strategy evolves” is insufficient to identify the changed algorithmic component. [GLS](https://ojs.aaai.org/index.php/ICAPS/article/view/3543)
- UNISafe supplies its own uncertainty-informed safety machinery. If used as a backbone, its predictive evidence and safety decisions must be distinguished from those newly produced by a planning controller. [UNISafe](https://arxiv.org/html/2505.00779)

### Inferences

The missing specifications are:

- **Operations:** whether the individual merely sets horizon/threshold values, selects among supplied planners, ranks branches, or constructs a conditional sequence of primitive computations.
- **Information:** the scores, prediction history, uncertainty signals and memory it can access; especially whether an operation already reveals the answer the controller is supposed to discover.
- **Action authority:** how the robot's final action is selected, and whether an existing shield can override it independently of the evolved strategy.
- **Evolution's contribution:** what structural changes mutation/crossover can express, and whether those changes are responsible for the eventual behaviour.
- **Claim and accounting:** what counts as task success, failure and computation, including both discovery cost and decision-time cost. ROSARL can be a principled choice of objective without becoming a novelty claim about the planning procedure.

These are requirements for defining the idea, not a request to commit to a large experiment plan. The candid answer to the user is: **promising research direction, established broad ingredients, and an unresolved specific contribution.**

### Gaps

Targeted search log, 8 October 2026: searched combinations of `evolutionary metareasoning planning safety`, `genetic programming metareasoning`, `latent planning hyper-heuristic safety`, `robot planning metareasoning selective collision checking`, `learning edge evaluation planning lazy`, and `safe value of computation planning`. Broad intersection queries were sparse; edge-evaluation and lazy-search terminology exposed the strongest scrutiny precedents. Followed these to original arXiv records, author-hosted manuscripts, IJCAI/ICAPS records and UNISafe's method. Stopped once decisive overlaps were established. This was not an exhaustive systematic review, and earlier literature notes plus the other researchers' direct learned-imagination and evolutionary-planner audits remain necessary context.
