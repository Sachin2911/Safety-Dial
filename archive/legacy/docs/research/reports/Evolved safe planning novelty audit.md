# Evolved safe planning novelty audit

**The broad idea is established, and the current proposal is not yet a demonstrated
novel contribution.** Learning how to investigate imagined futures, evolving reusable
search procedures, and allocating computation according to risk or decision benefit all
have close precedents. Thinker even reports behaviour resembling our motivating example:
continuing promising rollouts and resetting after poor predicted outcomes.
([Thinker](https://arxiv.org/html/2307.14993)) The narrower combination of evolutionary
discovery, runtime latent-model queries and simulator-measured safety remains a plausible
research setting. This audit did not establish an exact match to every proposed ingredient,
but that does not establish originality. The missing contribution is a **specific mechanism,
representation or explanatory finding**, with a reason it improves on existing approaches.
The field remains worth investigating; the broad pitch needs sharpening before adoption.

Assessed **8 October 2026**, using the
[provisional plan-scrutiny idea](../research_notes/evolving-plan-scrutiny-20261008.md).
No experiments or GPU runs were needed. This report preserves the idea as brainstorming
and does not replace the current project plan.

## Reusable planning procedures already have close precedents

The proposed evolutionary individual is a procedure that decides which imagined branch
to extend, whether to investigate another candidate, when to query uncertainty and when
to act. Its fitness comes from the robot's simulator behaviour, while the predictive model
initially stays fixed. This differs from evolving the next action sequence. However, the
distinction places the idea in established planning-control and algorithm-discovery
research. EVOCK already evolved reusable rules for a symbolic planner, evaluating them
through solved problems and search effort. Cazenave evolved MCTS development heuristics
using performance in complete games. ([EVOCK](https://e-archivo.uc3m.es/rest/api/core/bitstreams/1c55d9d2-574e-42b3-a8f3-c41b6ded31b0/content),
[evolved MCTS](https://www.lamsade.dauphine.fr/~cazenave/papers/evolveMC.pdf))

The nearest comparisons occupy different levels of the system:

| Prior work | Established overlap | Important boundary |
| --- | --- | --- |
| EVOCK (2001), evolved MCTS (2007 manuscript) | Evolution selects reusable search-control rules through downstream behaviour. | Symbolic planning or game simulation, rather than our proposed safety problem. |
| [Maes et al. (2012)](https://arxiv.org/pdf/1208.4692) | Composes whole search procedures from simulate, repeat, lookahead, step and select; evaluates with explicit computation budgets. | Discovery uses bandits and supplied transition functions. |
| [Thinker (NeurIPS 2023)](https://arxiv.org/html/2307.14993) | Learns imaginary actions, root resets and physical decisions; observes selective continuation of promising rollouts. | Original experiments generally fix the internal-step budget. |
| [RISE (August 2026 preprint)](https://arxiv.org/html/2608.20430) | Chooses Roll/Stop using predicted risk, further planning benefit and computation cost. | Sequential prefix continuation, rather than arbitrary branch/program search. |
| [STROLL, extended version (2021)](https://arxiv.org/html/2110.04669) | Learns which edge of a promising path to check next. | Checks reveal edge validity; learned latent predictions provide fallible evidence. |
| [GECCO 2026 algorithm discovery](https://arxiv.org/html/2603.28416) | Evolution discovers learning algorithms using world-model planning and confidence information. | Inspected planning supports policy learning, rather than our proposed runtime query controller. |
| [DeepJEPA (September 2026 preprint)](https://arxiv.org/html/2610.00368) | Allocates latent computation per candidate and transition; analyses effects on ranking and action choice. | Refines computation inside the predictor while retaining the outer planner. |

Maes and colleagues are particularly relevant if we choose a small planning language.
Their grammar changes the composition of search operations, rather than only numerical
parameters. Evaluation includes changed problem distributions and computation budgets.
Thus a program representation, reusable structure and budget sensitivity do not by
themselves supply our contribution. ([Maes et al.](https://arxiv.org/pdf/1208.4692))
More recent algorithm discovery reinforces this boundary: GP2S evolves node scores, while
the September 2026 HeurEvo preprint evolves component composition and runtime allocation.
([GP2S](https://arxiv.org/html/2412.09444),
[HeurEvo](https://arxiv.org/html/2609.36303))

## Selective scrutiny is a useful motivation, not an untouched mechanism

**Thinker's observed behaviour is stronger prior art than a superficial similarity in
terminology.** Its behavioural analysis describes deep, narrow planning that continues
promising rollouts and resets when predictions deteriorate, including predictions of
irreversible Sokoban states. That resembles our illustrative robot investigating an
attractive pulling motion until a later consequence changes its assessment.
([Thinker](https://arxiv.org/html/2307.14993)) The robot example makes the idea concrete,
but its narrative alone cannot distinguish the method.

Stopping is also established. The 2017 Imagination-based Planner learns whether to imagine
or act using task loss and imagination cost. Its general formulation permits flexible
starting states, although the continuous experiments use a narrower action set. Dynamic
Thinker later adds variable stopping and computation penalties.
([IBP](https://arxiv.org/html/1707.06170),
[Dynamic Thinker, ALA 2025 workshop](https://ala-workshop.github.io/papers/ALA2025_paper_44.pdf))
RISE more directly combines safety-relevant predictions with expected benefit from further
imagination. Its gate uses supervised targets from depth-wise planning evaluations, with
other modules frozen at that stage. We therefore cannot claim that freezing a world model
or learning safety-sensitive stopping is new. ([RISE](https://arxiv.org/html/2608.20430))

Robot motion planning supplies another close analogy. STROLL learns an edge-checking
policy within lazy search: first identify a potentially feasible attractive path, then
choose checks that efficiently resolve its validity. Bayesian Active Edge Evaluation
already values checks by their usefulness for path feasibility, and Generalized Lazy
Search allocates effort between expanding search and evaluating edges.
([STROLL](https://arxiv.org/html/2110.04669),
[active edge evaluation](https://www.ijcai.org/proceedings/2018/0679.pdf),
[Generalized Lazy Search](https://ojs.aaai.org/index.php/ICAPS/article/view/3543))

The evidence differs materially in our setting. A collision check reveals validity under
the checked model or world; another latent prediction can still be wrong. An uncertainty
estimate can also be misleading. This changes what a computation can establish, but
**fallibility itself is not a new problem**: Thinker uses learned predictions, and UNISafe
explicitly addresses optimistic latent forecasts of unobserved failure.
([Thinker](https://arxiv.org/html/2307.14993),
[UNISafe](https://arxiv.org/html/2505.00779)) Our contribution would have to identify a
particular unresolved consequence of that fallibility and a useful way to handle it.

## Recent world-model papers narrow the claim without duplicating it

The GECCO 2026 discovery paper matters because it prevents a broad claim about evolution
inventing algorithms that use world models. The inspected CG-FPD procedure uses latent
CEM planning as a training teacher and executes a feedforward policy. Its other reported
algorithm, DF-CWP-CP, uses confidence-weighted model rollouts, with dynamics specified
in observation space. Their evaluation on additional environments includes further
hyperparameter tuning. This is different from freezing a discovered runtime query
procedure and testing it elsewhere.
([Inspected manuscript](https://arxiv.org/html/2603.28416),
[official acceptance](https://gecco-2026.sigevo.org/Accepted%2BPapers))

DeepJEPA requires another precise distinction. Its controller chooses additional recurrent
updates **inside one predicted transition**. The outer CEM search, candidates and horizon
remain fixed. Training targets measure marginal latent prediction-error improvement,
rather than actual safety or task reward, and the authors do not equate this proxy with
true decision value. It nevertheless already investigates where additional latent
computation changes candidate rankings and actions. Our external search controller would
operate at a different level, while sharing that motivation.
([DeepJEPA](https://arxiv.org/html/2610.00368))

These distinctions support a careful comparison, not a claim that the unoccupied space
must be novel. Nor would multiple model backends settle the issue. A common interface
establishes compatibility; separately evolving procedures establishes applicability.
Keeping a procedure fixed across a model change could challenge its explanation, but
transfer becomes useful evidence only after specifying what was learned and what
adaptation the interface permits. RISE's reported integration with DAWN does not resolve
whether its scheduler weights transfer unchanged. ([RISE](https://arxiv.org/html/2608.20430))

## A contribution needs an identifiable decision mechanism

The present candidate leaves three central objects unspecified: what one individual can
change, what information it receives, and which failure of existing planning it resolves.
A vector of thresholds, an expression ranking nodes, a conditional program and a recurrent
controller are different scientific proposals. Calling all of them a planning strategy
conceals the actual comparison.

Our assessment is that replacing RL with CMA-ES, adding a safety penalty, or applying a
familiar controller to an arm would currently be an optimizer or application change.
Such work can produce valuable evidence, but a stronger claim would explain why a
particular representation or discovery process exposes useful behaviour that simpler
choices miss. Evolution need not be universally superior; its contribution must be
visible in the result being claimed.

The information contract matters just as much. If an input module already predicts which
branch deserves scrutiny, much of the solution sits outside evolution. If an existing
safety filter overrides dangerous actions, improvements must be attributed between that
filter and the planning procedure. The model, proposals, readouts, memory and final action
authority need clear roles before novelty can be assessed. UNISafe supplies substantial
safety machinery of its own; adopting its components does not make their capabilities a
new planner contribution. ([UNISafe](https://arxiv.org/html/2505.00779))

Two formulations remain useful **hypotheses with open novelty**. The first concerns a
specific mechanism for handling conflicting safety evidence: an evolved procedure orders
additional predictions or uncertainty checks so that it rejects dangerous attractive
plans while retaining useful alternatives. The unknown is the actual reconciliation
mechanism and why a tuned threshold or learned metacontroller would not capture it.

The second concerns an explanatory result: identify conditions under which selecting
planning procedures for physical safety and progress produces different useful computation
patterns from selecting them for prediction accuracy or raw uncertainty. That could
clarify when safety changes which computation is worth buying. Its contribution would
depend on demonstrating the relationship and evolution's role, rather than claiming the
general value-of-computation principle as new.

Both formulations need a concrete decision where existing allocation fails despite useful
evidence being available. Equal-information comparisons, matched computation, and removing
the distinctive part of a discovered rule would help distinguish a mechanism from extra
queries, stronger readouts or better proposals. Discovery cost and deployment cost remain
separate. Reduced violations through abandoning the task would not support the intended
claim. These are criteria for interpreting the idea, not an experiment plan.

## The area is viable, but the research claim remains open

The productive next step is to describe one controlled decision, the evidence each query
could reveal, and the reason a strong existing method would allocate its effort poorly.
That description would give the evolutionary search a substantive target. The present
evidence supports continued ideation around that target, while ruling out selling the
broad concept as original.

This was a focused primary-source audit, not an exhaustive systematic review. The final
GECCO publisher text and implementation were not verified; its method comparison rests
on the March manuscript. RISE scheduler transfer remains unresolved, and recent preprints
are author-reported evidence. Detailed sources and search limits are preserved in
[evolutionary planning prior](../research_notes/Evolved%20safe%20planning%20novelty%20audit/evolutionary_planning_prior.md),
[learned imagination prior](../research_notes/Evolved%20safe%20planning%20novelty%20audit/learned_imagination_prior.md),
and [claim boundaries](../research_notes/Evolved%20safe%20planning%20novelty%20audit/claim_boundaries.md).
