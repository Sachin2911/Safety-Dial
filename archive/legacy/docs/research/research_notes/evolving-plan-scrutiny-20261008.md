# Evolving selective investigation of imagined plans

Recorded 8 October 2026.

**Status: provisional brainstorming.** This develops the preferred
[agent-focused imagination idea](evolving-safe-imagination-20261007.md). The specific
mechanism remains open while Sachin reads UNISafe. It does not replace the current project
plan or establish a novelty claim.

## Research question

**Can evolution discover which additional predictions are worth making to prevent a bad
physical decision?**

The proposed agent chooses how to investigate possible futures through a latent world
model before acting. Evolution searches for a reusable procedure that allocates those
computations. The motivating behaviour is a robot changing its intended action because
it chose a useful question to ask its model.

Robotic manipulation is now a plausible setting to investigate. The
[UNISafe compute audit](../reports/UNISafe%20inference%20and%20training%20compute.md)
supports considering a single-GPU study, with software migration and actual throughput
still to be checked. This does not commit the idea to UNISafe or to its complete safety
filter.

## A concrete robot example

In UNISafe's simulated block-plucking task, a robot removes the middle block from a stack
of three while keeping the top block on the bottom one. This is distinct from its physical
Jenga experiment. [UNISafe, section 6.2](https://arxiv.org/html/2505.00779#S6.SS2)

Suppose a proposed pulling motion looks successful over its first few predicted steps:
the target moves and nothing falls. The agent has a limited computation budget and could
choose among several operations:

| Operation | What it could reveal |
| --- | --- |
| Extend the imagined trajectory | Whether instability appears after the target moves out |
| Branch into another pulling motion | Whether a different approach offers useful progress with less predicted danger |
| Evaluate uncertainty along a candidate trajectory | Whether apparent success depends on unreliable predictions |
| Execute a short action and observe again | How the physical scene responds, adding evidence beyond model predictions |

These are proposed choices, not capabilities already demonstrated by our agent. Physical
actions incur interaction cost and may themselves cause failure; a fresh observation is
not a free or automatically safe alternative to imagination.

## What evolution would discover

A candidate would be a small program or controller deciding which operation to perform,
which imagined branch to investigate, and when to select a physical action. Genetic
programming and small recurrent controllers remain possible representations. The world
model would initially stay frozen so that changes in decisions can be attributed to the
planning procedure.

An illustrative discovered strategy might explore several actions briefly, discard poor
options, investigate unresolved consequences of a promising option, and switch plans when
that investigation changes the choice. This is a hypothesis about possible behaviour.
Encoding that entire strategy as a supplied operation would give evolution the answer.

Selection would use safety and useful task progress measured in the actual simulator,
with computation accounted for. Imagined safety would remain separate from measured
physical outcomes. Safety criteria and their readouts still need to be supplied; the
proposed discovery concerns how to use predictions, not how to define danger.

## Why selective investigation could matter

The most uncertain prediction might concern an action the robot would never choose.
Investigating it could consume the budget without improving the decision. Conversely,
an attractive action might deserve scrutiny because one additional prediction could
expose a delayed failure and change what the robot does.

The hypothesis is that a discovered procedure can allocate computation according to its
effect on the eventual decision. Its benefit would need to exceed that of a simple rule
such as always using a longer horizon or checking uncertainty on the current best plan.

More queries cannot reliably reveal a danger that every available model response
consistently misses. Longer imagination may also accumulate error. Deciding when to seek
another physical observation is therefore a related, potentially broader research
question, not an established solution to model error.

## Prior work and the evidence needed

[Thinker](https://arxiv.org/abs/2307.14993) already learns how to interact with a world
model before selecting physical actions. [RISE](https://arxiv.org/abs/2608.20430) already
allocates imagination using estimated risk, expected planning benefit and computation
cost. The broader
[literature review](../reports/Evolving%20safe%20imagination%20across%20models.md)
also identifies evolutionary discovery of planning rules as established prior work.

A possible contribution would require a particular discovered investigation strategy,
an explanation of why it improves decisions, and evidence for the usefulness of evolution
in discovering it. Relevant comparisons include simple tuned planning rules, random
search over the same controller space, and an RL-trained controller with the same
operations and information.

Comparisons should share the model, readouts and action proposals, and account for both
model queries and elapsed time. A latent transition and an uncertainty-ensemble call
need not cost the same. Evaluation should keep task completion and violations visible,
use held-out cases, and test whether removing the distinctive investigation behaviour
changes decisions. All simulator interactions used for training and selection count.

## Choices still open

- **Main behaviour:** selectively scrutinise promising plans, or decide when imagination
  is insufficient and another physical observation is worth obtaining. Sachin has not
  selected between these narrower emphases.
- **Representation:** determine what a program or controller can observe and which basic
  operations it can combine without supplying the desired strategy in advance.
- **Transfer:** later test whether the same frozen procedure remains useful with another
  task-matched model. LeWM and the world-model component of SafeDreamer remain interests;
  a common interface alone would not demonstrate transfer.
- **Objective:** ROSARL remains a possible connection for selection, subject to checking
  its assumptions for the eventual task. It is not yet a required ingredient.

## Reading UNISafe

Paper: **Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution
Failures** by Junwon Seo, Kensuke Nakamura and Andrea Bajcsy.
[Online paper](https://arxiv.org/html/2505.00779),
[PDF](https://arxiv.org/pdf/2505.00779),
[existing local copy](../../papers/myPapers/unisafe.pdf).

A useful reading order for this discussion is:

1. **Figure 1 and section 6.2:** understand the overall system and simulated manipulation
   task; section 6.3 shows the separate physical-robot experiment.
2. **Section 3:** distinguish the world model, failure readout, safety value and fallback
   policy, and see how filtering affects the task policy.
3. **Sections 4 and 5:** understand how uncertainty enters the safety method.
4. **Appendix A.1 and A.2:** inspect the latent world model and separate uncertainty
   ensemble that motivated the compute audit.

While reading, consider which safety judgements are learned ahead of time and which
computations occur when the robot acts. That distinction will help clarify where a
discovered imagination procedure could fit and what it would have to add.
