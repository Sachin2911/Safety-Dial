# Evolving how an agent uses latent imagination

Recorded 7 October 2026.

**Status: preferred research candidate from the current brainstorming.** The student
explicitly prefers this agent-focused idea to
[evolving the model's learning objective](evolving-latent-learning-objectives-20261007.md).
The student also proposes investigating multiple world-model families, including LeWM
and the model used by SafeDreamer. This records that preference and the proposed research
question. Novelty, feasibility and an executable experiment plan remain to be established.

## Research question

**Can evolution discover a reusable procedure for using latent imagination that enables
safer decisions across tasks and world-model families?**

The scientific focus is how the agent decides which possible futures to examine and how
to use those predictions when acting. An initial study would freeze the world model and
evolve the planning procedure. The exploration is open to new environments and
infrastructure, without being constrained by the previous experiments or November thesis
deadline.

## What evolution would change

A candidate agent can perform a bounded sequence of internal planning operations before
executing a physical action. Possible operations include:

- Propose an action sequence and predict its consequences.
- Extend an imagined trajectory further into the future.
- Branch from an earlier imagined state to investigate another action sequence.
- Compare candidate futures and decide whether to imagine further or act.

Evolution would search over the logic that selects and combines these operations.
Restricted programs evolved with genetic programming are one concrete possibility.
Small recurrent controllers are another; neither representation has been selected yet.
Branches must preserve the model's required state and history and use its action-
conditioned dynamics. Arbitrary latent vectors do not automatically describe achievable
physical states.

Selection would use actual simulator outcomes, measuring safety and useful task progress
under a declared imagination budget. ROSARL is a possible outer scoring rule, with its
assumptions checked for this setting. Predicted safety remains separate from actual
safety. The agent still needs an externally specified safety criterion and appropriate
feedback; evolution does not establish what should count as dangerous.

An illustrative hypothesis is that an evolved planner learns to investigate delayed
consequences after apparent task success, such as instability after a robot picks up an
object. The aim would be to identify a repeatable planning strategy, its causal effect on
decisions, and the conditions under which it transfers. These are hypotheses, not results.

## Using different world models

The proposal can be designed around an interface for querying predictive models. Each
backend retains its own latent representation and state bookkeeping.

- [LeWM](https://le-wm.github.io/) provides an encoder and action-conditioned latent
  predictor. Its published planner optimises action sequences against latent goal
  distance with CEM. Our proposed safety study would additionally need a defined safety
  readout and task-scoring interface.
- [SafeDreamer](https://arxiv.org/html/2307.07176v3) is a complete safe RL method built on
  DreamerV3, with RSSM dynamics and learned reward and cost predictions. Its online
  planning variants provide relevant baselines. The proposed backend is its learned
  world-model component; using that component does not mean the evolved planner is the
  original SafeDreamer algorithm.

A shared planning interface could expose operations such as propose, predict, extend,
branch and select, together with summaries of predicted progress, safety costs and the
remaining computation budget. The latent state could remain an opaque handle managed by
the backend. Action proposals can be produced through task-specific modules whose
training cost and information are declared and shared with the relevant baselines.

This interface is a design proposal, not an existing validated implementation. It must
handle differences in action dimensions, action blocks, physical time, stochastic
predictions, history, termination and score scales. A cost prediction is not automatically
a calibrated violation probability. Stochastic latent samples do not by themselves
measure epistemic uncertainty. Any additional uncertainty mechanism needs a separate,
explicit specification.

The adapters and safety readouts must be fitted using declared data, without tuning to
final evaluation outcomes. They must not quietly implement the planning decisions that
the experiment attributes to evolution. Every backend needs a model trained for the
chosen task family; released checkpoints are not assumed to support arbitrary tasks.

## Three distinct levels of evidence

1. **Compatibility:** the evolutionary method can train a planning procedure separately
   for each model family. This establishes breadth of applicability.
2. **Transfer across tasks:** a frozen evolved procedure works on held-out goals, layouts
   or tasks through the declared interface, without further evolution on those tests.
3. **Transfer across model families:** a procedure evolved using one family remains
   useful with another family, without re-evolving the procedure. The new model and any
   permitted adapter training are disclosed separately. This is a stronger empirical
   claim than compatibility and is not implied by a shared interface.

Raw latent coordinates do not share semantics across separately trained models. A
controller tied directly to one model's latent coordinates therefore needs a justified
transfer mechanism. Operating on common planning operations and meaningful summaries is
one proposed way to investigate transfer, with possible loss of useful information as a
tradeoff to test.

## Evaluation that isolates the agent contribution

Begin with fixed predictive models. Within each model family, compare planning methods
using the same checkpoint, readouts, action-proposal information and task distribution.
Assess actual violations and task progress together. Inspect imagined branches to test
which discovered operations change physical decisions.

Relevant comparisons include the model's standard planner, safety-aware CEM, simple
adaptive planning rules, a planning controller learned through RL, and a reactive
controller. Preserve the original SafeDreamer system as a separate end-to-end comparison
where feasible. A fixed-checkpoint comparison isolates planning; an end-to-end comparison
also includes differences in training and acquired experience.

Control model-query budgets within a backend and report latency and computation as well.
One prediction call from each architecture need not cost the same or advance the same
physical time. Across families, report matched-time comparisons or performance against
compute budgets. Count all simulator interactions used for model learning, readouts,
evolutionary search and discarded candidates. Record model training, imagined steps,
search compute and wall-clock time separately.

Use held-out tasks for final evaluation, with no repeated search or adapter tuning on
them. Ablations should test dependence on model predictions and the evolved branching or
stopping rules. Low violation rates achieved by abandoning useful progress do not
establish the intended contribution. A planner cannot reliably discover a danger that
its predictive model consistently fails to represent, and no guarantee of safety is
implied by zero observed violations.

## Prior work and the novelty boundary

- [Learning model-based planning from scratch](https://arxiv.org/abs/1707.06170) already
  learns how to construct, branch and execute imagined plans.
- [Metacontrol for Adaptive Imagination-Based
  Optimization](https://arxiv.org/abs/1705.02670) learns allocation of internal simulation
  and choice among models with different reliability and computation costs.
- [Evolutionary Planning in Latent Space](https://arxiv.org/abs/2011.11293) searches
  action sequences through a learned world model using an evolutionary method.
- [RISE](https://arxiv.org/abs/2608.20430), an August 2026 preprint, adapts rollout depth
  using estimated risk, planning benefit and computation cost.
- SafeDreamer's constrained CEM planner is credited prior work, with its constrained
  cross-entropy foundation attributed to Wen and Topcu (2018) in the paper.

Adaptive imagination, evolutionary planning and safety-aware planning are established
ingredients. A plausible contribution would require a specific discovery method and
evidence of a useful, explainable planning procedure whose benefit transfers. Swapping
the training optimiser alone does not establish a substantive contribution. A focused
novelty review is still needed before choosing a final claim.

## Next decisions

Define a small but expressive space of planning procedures and an interface that can be
implemented for two genuinely different model families. Select a task family with clear
safety events and useful progress, then design the smallest experiment that distinguishes
a better planning strategy from extra computation, a better readout, or an idle policy.

No implementation, experiment, model compatibility result or supervisor approval is
claimed by this note.
