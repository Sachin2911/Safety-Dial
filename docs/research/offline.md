# Frozen-policy adapters and safe task completion

29 October 2026. Related work and exploratory ideas from discussion with Sachin.

This note records research on improving an existing robot policy while keeping all
of its original weights frozen. It does not change
[firstPrinciples.md](../../firstPrinciples.md) or freeze an experimental protocol.
The separate [Online-ES idea](onlineES.md) allows action-expert updates and therefore
addresses an adjacent question.

## Main finding

Frozen-policy adaptation is an established architectural pattern. Existing methods
steer policy inputs, select candidate actions, add residual corrections, modify
execution or route control to specialists. Freezing the policy alone is unlikely to
constitute a research contribution. The contribution would need to concern the
adapter's mechanism, its safety and completion outcomes, or an evidenced advantage
of its learning procedure.

Most of the closest recent papers below are September or October 2026 preprints.
Their reported results are evidence to examine, not independent replications or
proof that the same mechanism will work for ManiGuard Jar.

## Closest related work

### VLA-Stem: learned execution around a frozen VLA

[VLA-Stem: A Lightweight Safety Mechanism for Vision-Language-Action Models via
Reinforcement Learning](https://arxiv.org/abs/2610.37378), October 2026.

VLA-Stem keeps a task-trained pi0 or pi0.5 policy frozen. It converts action chunks
into motion references, then trains a separate controller to adjust motion and
timing around obstacles while preserving task progress. The controller conditions
on the reference, robot proprioception and local obstacle information. Training
uses RL, a curriculum and a privileged teacher, followed by student distillation.

This is the closest safety precedent in this search. It addresses the difficulty
of recovering after a safety intervention changes the robot's state. Its collision
avoidance results do not establish Jar's temporal requirements, such as closure
before carrying, or safety guarantees for a learned controller.

### VLaRL: bounded action corrections

[VLaRL: Augmenting Vision-Language-Action Models with Simulation-Trained
Latent-Conditioned Residual RL](https://arxiv.org/abs/2609.30868), September 2026.

The frozen VLA supplies nominal actions and internal representations. A separate
residual policy adds bounded corrections:

`executed action = frozen policy action + scaled residual correction`

The residual receives VLA features, nominal actions, robot state and force
information. The authors report gains on contact-rich manipulation with Flower
and GR00T N1.7. Training uses demonstrations, a sim-to-real latent mapper and
privileged simulator information for the critic. These costs and information
sources matter when designing a fair comparison. The reported task gains are not
evidence of strict safe completion under ManiGuard's rules.

### V-GPS: select among candidate actions

[Steering Your Generalists: Improving Robotic Foundation Models via Value
Guidance](https://arxiv.org/abs/2410.13816), CoRL 2024.

V-GPS samples multiple candidate actions from a frozen pretrained policy and ranks
them using a value function trained with offline RL. The base policy can remain a
black box. This improves action selection without requiring access to its weights
or internal representations.

The mechanism depends on useful alternatives appearing among the candidates. It
does not directly add a new corrective behavior outside them. Critic training,
training data and additional policy queries must be included in the accounting.

### RouteRLT: selective specialist intervention

[RouteRLT: Learning When and Which RL Specialist Should Control a
Vision-Language-Action Policy](https://arxiv.org/abs/2609.26467), September 2026.

A frozen VLA handles general manipulation while learned RL specialists handle
particular phases. A router selects the active controller, with mechanisms to
stabilize switching and manage action chunks when control changes.

This is close to an adapter that intervenes only around difficult contact or
precision phases. The specialists and their representations require training;
the router's parameter count does not describe the whole method's cost. The
real-world evaluation uses operator-aligned handoffs, which limits what it
establishes about autonomous routing on hardware.

### FailPatch: gated internal residual corrections

[FailPatch: Failure Residual Patching for Vision-Language-Action
Models](https://arxiv.org/abs/2609.34175), September 2026.

FailPatch freezes all original VLA weights and learns a router and residual expert
bank in the action-hidden representation. The router can preserve the original
representation or select a correction. Training uses failure information and
corrective supervision.

This requires access inside the model rather than only an external action
interface. Its adapters contain 17.3 million trainable parameters, although that
is only 0.52% of the model. Parameter efficiency relative to a VLA does not
necessarily imply a compact controller suitable for our intended search budget.

## Other relevant evidence

[DSRL: Steering Your Diffusion Policy with Latent Space Reinforcement
Learning](https://arxiv.org/abs/2506.15799) learns to choose input noise for a frozen
diffusion policy. This is another adaptation interface, already investigated in
our project. Its existence does not resolve the unsuccessful or incomplete Jar
studies recorded in the [current evidence](../../firstPrinciples.md).

[Retrieval-triggered Action Projection Adjustment for Frozen VLA Policies: An
Empirical Study and Failure Analysis](https://arxiv.org/abs/2610.24573), October
2026, provides a useful negative result. Its external-memory wrapper corrects
frozen OpenVLA translation commands. A reduction in a depth-based near-collision
or stall proxy in the primary cohort did not increase task successes and did not
reproduce independently. Retrieved correction directions were not distinguishable
from random same-family directions. The study does not establish physical safety
or a memory-specific benefit.

## Candidate question for our project

Can a compact controller use task progress and recent history to make corrective
interventions that improve safe completion while leaving the base policy frozen?

For Jar, blocking an early lift is insufficient. A useful adapter must help the
robot finish closing the jar and then resume carrying it. Task 19 reduced observed
violations but retained 32/107 strict safe completions in each arm, with more
post-closure stalls. The [consolidated record](../../principia.pdf) retains the
result and its limitations.

The first uncertainty is the correction mechanism:

- Can selecting another policy proposal expose an already available safe action?
- Can a bounded residual correct the local contact or movement error?
- Does recovery require a behavior that the base policy cannot supply reliably?

A controller could use task-stage memory to decide when to trust the frozen
policy and when to apply a limited recovery correction. This is a hypothesis to
investigate, not an established novelty claim. Existing residual and routing
methods already overlap that description. Enlarging the search space does not
establish that productive corrections exist.

## Implications for the research process

VLA-Stem is the first reading priority for preserving progress during safety
interventions. VLaRL is the next priority for designing residual corrections.

The current sequence remains appropriate: establish a completion lever, establish
stable evaluation, then compare optimizers. Before claiming an evolutionary
advantage, compare against native execution, a tuned fixed setting, parameter
random search and ordinary learning using the same information and matched
interaction budgets. Count auxiliary training, demonstrations, simulator work,
discarded candidates and tuning; report model queries and elapsed time separately.

Evaluate safe completion alongside violations, unfinished episodes and engagement.
Keep benchmark and requested-sequence scoring distinct. These precedents motivate
studying an adapter, but do not establish that evolution is the best way to learn
one or that collision avoidance transfers to Jar's temporal safety requirements.
