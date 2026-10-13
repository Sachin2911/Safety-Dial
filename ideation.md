# Research ideation: safe task continuation through evolutionary adaptation

**Discussion notes, 13 October 2026.** This document develops the direction in
[oracle.md](oracle.md), which remains the central authority for research decisions.
The ideas below are contribution possibilities, not a settled method or experiment plan.

## Ambition

The project is Sachin Mohan's BSc Honours research at the University of the Witwatersrand,
supervised by Geraud Nangue Tasse. It must support an end-of-year honours submission,
with the ambition of presenting the research at GECCO 2027.

The aim is a genuine contribution at the intersection of safe reinforcement learning,
evolutionary algorithms and world models. Combining those components establishes the
setting; the contribution must explain something consequential about their interaction.

The guiding question is:

> What would someone learn from this project that they could not learn merely by
> assembling these three components?

## Promising research question

> When a learned safety filter disrupts a pretrained manipulation policy's task behaviour,
> can evolutionary adaptation improve safe completion under a limited adaptation budget?

A policy may be competent in familiar conditions yet struggle with the consequences of
safety intervention. It might repeatedly propose rejected actions or fail to continue
from configurations reached after the filter redirects it. Adaptation could help the
policy resume progress, anticipate an intervention or choose an earlier approach that
works more effectively with the filter.

These are hypotheses. Meaningful, avoidable stalls have not yet been established in the
system we would use. An unfinished episode could instead reflect an overly restrictive
filter, prediction error, missing information or insufficient time. Establishing that
better task behaviour has room to help is central to the research motivation.

## Substantive role of each area

- **Safe RL:** study task learning or adaptation while actions pass through a learned
  safety mechanism. Evaluate completion alongside physical violations and unfinished
  episodes.
- **Evolutionary algorithms:** search behavioural changes according to the performance
  of the complete filtered system. Investigate whether this search offers useful
  advantages under a declared adaptation budget.
- **World models:** supply learned representations and predictions that inform safety
  intervention. Understand how prediction quality and the states reached after
  intervention affect what behavioural adaptation can achieve.

Evolutionary fitness could come from simulator outcomes while the world model supplies
representation and safety assessment. If imagined outcomes guide selection, their
accuracy and their relationship to observed physical outcomes require separate checks.

## What the contribution could be

The promising contribution is understanding and improving the interaction between task
behaviour and imperfect learned safety assessment. Two routes remain open:

- **Methodological:** develop a justified adaptation mechanism and demonstrate why it
  improves safe task completion.
- **Empirical:** identify a reproducible difficulty, explain its mechanism and establish
  when evolutionary adaptation helps relative to ordinary learning with the same filter.

A standard EA could support the empirical route. An optimizer substitution alone would
not establish a contribution. Ordinary RL can also learn through a nondifferentiable
filter, so nondifferentiability alone does not explain an evolutionary advantage.
Novelty and publication suitability need checking against primary literature as the
specific question develops.

## Evidence that would make the story convincing

We need a concrete manipulation setting in which consequential stalls occur and evidence
that better task behaviour can avoid some of them under the same safety constraints.
Then we need to explain what adaptation changes and compare it with ordinary task-policy
learning with the same filter, using comparable information and interaction budgets.

Report safe completion, physical violations and unfinished episodes together. Reduced
intervention counts alone can reward inactivity. Account for all simulator interactions,
including adaptation and discarded candidates, and report model queries and elapsed time
separately. A fixed filter does not guarantee an unchanged violation rate after adaptation.

## Scope and open choices

For honours, aim for one well-explained failure and one thoroughly evaluated response.
GECCO could build on that through stronger methodological development or broader evidence,
depending on what the initial work reveals. This is an ambition, not a publication promise
or a requirement to invent a new EA.

UNISafe remains a candidate foundation. The environment, world model, behavioural search
space, evolutionary algorithm, role of ROSARL and experiment budgets remain open.
Freezing the world model and filter initially could aid attribution, but is still an
experimental choice. The next useful step is diagnosing the available system before
committing to an adaptation mechanism.

Supporting evidence is in the
[manipulation comparison](docs/research/reports/Safe%20manipulation%20progress%20and%20robustness.md)
and [UNISafe compute audit](docs/research/reports/UNISafe%20inference%20and%20training%20compute.md).
