# Evolving learning objectives for latent world models

Recorded 7 October 2026.

**Status: candidate research idea that the student finds exciting.** This note preserves
the discussion for further investigation. It is not an adopted experiment plan, a
validated novelty claim, or a replacement for the existing research records. The next
discussion also explores a contribution focused on how an agent uses a latent model.

## Motivation and scopez

The scientific centre is an agent that learns an internal model of the world, imagines
consequences, and uses that understanding to act safely. The student's main inspirations
are Yann LeCun's vision of predictive latent world models, the LeWM release, and UNISafe's
use of latent prediction and uncertainty for robotic safety.

Evolution must have a substantive role: it is both a research requirement and an area of
personal interest, with GECCO the target venue. ROSARL provides relevant theory and
supervisor expertise; its precise role should follow the scientific question.

The exploration is open to new environments, architectures, methods and infrastructure.
The November thesis deadline and reuse of Walker or the existing LeWM assets do not
constrain this search for a conference direction. Previous experiments provide useful
background without defining the space of possible contributions.

## Central question

**Can evolution discover how a latent world model should learn, so that its imagination
becomes useful for safe planning?**

The concrete proposal is to evolve a compact training objective or learning rule for a
latent world model. The resulting model is trained with gradient descent. Evolution
evaluates the learning rule through the task performance and safety of an agent that
plans using the trained model.

The potential output is a transferable way of learning an internal predictive model.
An especially valuable result would explain which kinds of prediction mistakes the
discovered rule emphasises, and why those changes improve decisions.

## Intuition

Consider a robotic arm moving an object through clutter. Useful imagination may need to
preserve whether an object is supported, whether a movement will destabilise it, and
whether two routes have different consequences. A compact representation can be useful
for average prediction without necessarily preserving every distinction needed for a
safety decision.

That possibility motivates the study. It is a hypothesis to test, not a claim that every
JEPA necessarily discards safety information. Nor does a readable safety feature alone
establish reliable imagined dynamics or safe closed-loop behaviour.

## Proposed learning loop

1. Define a restricted space of numerically stable, differentiable learning objectives
   for a fixed latent world-model architecture. Candidates might change how prediction
   errors across time, separation of action-conditioned futures, and supplied safety
   feedback contribute to training. The exact search space remains open.
2. Evolution proposes a population of candidate rules. Each rule trains a model with
   ordinary gradient descent on the same data and with the same training budget.
3. A fixed planner uses each trained model. Score task performance and violations in the
   actual environment or simulator, separately from predictions made in imagination.
4. Evolution selects and modifies the learning rules according to those downstream
   outcomes. Development tasks support this search; final evaluation tasks remain
   excluded from rule selection.
5. Freeze the discovered rule, then use it to train fresh models on held-out tasks.
   Evaluate whether its benefits transfer across tasks and, if practical, architectures
   and planners.

Transfer here means transfer of the learning rule. Fresh models may still require task
data and training; the proposal does not imply zero-shot policy transfer.

## Roles of the four interests

- **Latent world models:** the representations and predictive dynamics being learned.
- **Evolutionary algorithms:** discovery of the model's learning objective or update rule.
- **Safe RL and control:** downstream behaviour used to judge whether the model is useful.
- **ROSARL:** a possible principled way to score unsafe outcomes in the outer evaluation.
  Its applicability needs checking, and any practical adaptation must be identified.

The world-model training objective and the agent's reward or safety penalty are distinct
objects. Evolving the former does not by itself discover what humans regard as unsafe.
Specified safety criteria or failure feedback are still required. Any physical-state
labels used during training must be declared as privileged supervision.

## What would make the contribution substantive

The proposed claim is that a discovered learning rule yields models that support safer,
useful decisions across tasks. Demonstrating this would require more than selecting one
fortunate checkpoint or tuning a coefficient on one benchmark.

Useful comparisons include the original model objective, deliberately designed safety
or decision-aware objectives, and ordinary hyperparameter search with a matched search
budget. A fixed planner initially isolates the model-learning effect. Later testing with
another planner can reveal whether the rule is specific to the evaluator used to evolve
it.

The scientific analysis should examine what the rule changes: representation content,
multi-step prediction, risk ordering of candidate actions, or another identifiable
mechanism. Task progress matters alongside safety, including whether a method achieves
few violations by refusing useful action.

All outer-search evaluations, model-training experience and discarded candidates count
toward cost. Report simulator interactions, imagined computation, model-training compute
and wall-clock time separately. Reusing source data does not create independent evidence.

## Literature anchors and novelty limits

These sources were checked during the discussion. The list is a starting point for a
focused novelty review, not an exhaustive review.

- [LeWorldModel](https://arxiv.org/abs/2603.19312): a compact JEPA trained through embedding
  prediction and latent regularisation, providing a concrete model-learning starting
  point.
- [UNISafe](https://arxiv.org/abs/2505.00779): uses latent dynamics and epistemic uncertainty
  to construct a safety filter. It motivates the desired anticipatory behaviour.
- [Evolved Policy Gradients](https://arxiv.org/abs/1802.04821): evolves a differentiable
  loss for policy learning. Evolving a learning objective is established prior work.
- [Learning Symbolic Model-Agnostic Loss Functions via Meta-Learning
  (EvoMAL)](https://arxiv.org/abs/2209.08907): combines evolutionary discovery of symbolic
  loss structures with gradient-based parameter optimisation for supervised learning.
- [Deep Neuroevolution of Recurrent and Discrete World
  Models](https://arxiv.org/abs/1906.08857): evolves world-model system components through
  behavioural fitness. Evolution of world-model systems is also established prior work.
- [PhyLatent](https://arxiv.org/abs/2608.05720): proposes objectives that preserve physical
  state relationships and distinct action-conditioned futures in JEPA models, improving
  planning in its experiments.
- [Supervise What Decides Success](https://arxiv.org/abs/2610.01224): uses physical success
  criteria as auxiliary training targets for latent models. This is a recent preprint.
- [ROSARL](https://arxiv.org/abs/2306.00035) and its
  [2026 value-range formulation](https://rlj.cs.umass.edu/2026/papers/Paper96.html): relevant
  penalty formulations whose assumptions and differences require attention when
  designing the outer fitness.

The broad ingredients are therefore not a novelty claim. The opening to investigate is
a particular method for discovering model-learning rules whose downstream safety benefit
transfers, together with an explanation of why it works.

## Open questions before adopting the idea

- What restricted rule space offers meaningful discovery beyond tuning known losses?
- What information can the training objective use, and what supervision does each
  baseline receive?
- How do we preserve a useful predictive representation and avoid degenerate objectives?
- Which task family can expose safety-relevant distinctions with affordable inner
  training runs and clear ground truth?
- How should outer fitness express safety and useful progress, and where does ROSARL fit?
- How well does selection after short training predict the result of full training?
- Does the discovered rule transfer across tasks, model seeds, planners and model families?
- What does it teach us about latent imagination beyond a benchmark improvement?

No experiment, novelty confirmation or supervisor approval is claimed by this note.
