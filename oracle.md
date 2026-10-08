# Oracle: task continuation under safety filtering

**Central reference for the research rethink. Updated 8 October 2026.**

We have chosen to focus on helping a robotic manipulation agent complete its task when
a learned safety filter intervenes. This choice remains conditional on finding meaningful,
avoidable stalls in an accessible task. The specific method and experiment plan remain open.

Use this document to develop the current focus. Earlier proposals and experiment plans
remain records of their respective studies; their methods, deadlines and gates are not
decisions for this research rethink.

## The research question

> Can evolutionary adaptation help a pretrained manipulation policy work more effectively
> with its safety filter, improving safe task completion?

The central problem is the interaction between task competence and safety intervention.
A filter can redirect a dangerous action while leaving the task policy in a configuration
it does not know how to handle. The robot may remain protected but fail to finish its job.

For example, imagine a robot extracting a block from a tower. Its filter interrupts a pull
that appears dangerous. The robot then repeatedly attempts a rejected movement or cannot
continue from its new position. Useful adaptation might improve what it does next, or
change its earlier approach so that this interaction no longer prevents completion.
This is an illustrative failure, not an outcome we have reproduced in UNISafe.

## How the components fit

| Component | Role in the direction |
| --- | --- |
| Task policy | Proposes actions to complete the manipulation task. |
| Latent world model | Supplies representations and predictive dynamics that can support safety assessment. |
| Safety filter | Evaluates proposed behaviour and modifies or replaces actions when necessary. |
| Evolutionary algorithm | Adapts task behaviour according to how the complete filtered system performs. |

Evolution could adapt a compact part of an existing policy or another limited behavioural
interface. The representation, search space and optimizer should follow the diagnosed
problem. A standard evolutionary method may be sufficient for a useful empirical
contribution; a new planning architecture is not a prerequisite.

Candidate behaviour can be evaluated through actual simulator outcomes while the world
model supplies latent state and safety assessment. If imagined outcomes are used for
selection, they must remain distinct from observed simulator outcomes. Improvement inside
the model does not establish improved physical safety.

## What must make this problem worth solving

The initial question is whether the available system has consequential stalls that better
task behaviour could avoid under the same safety constraints. An incomplete episode alone
does not establish that diagnosis. Similar symptoms could come from an overly restrictive
filter, inaccurate predictions, missing information or insufficient time.

We therefore need to understand what happens after an intervention and what successful
continuation would require. Failed attempts do not prove that completion is impossible.
The strongest continuation motivation in the reviewed literature comes from a different
manipulation setup; its explanation still needs to be checked in the task we choose.

Ordinary task-policy learning with the same filter present is an essential comparison.
The broad idea of learning through safety interventions already exists. The potential
contribution is to explain a remaining difficulty and demonstrate effective evolutionary
adaptation under comparable information and interaction budgets.

The intended result is improved safe task completion. Report task success, physical
violations and unfinished episodes together, with useful progress and completion time
where relevant. Fewer interventions alone could reward inactivity. Keeping a filter fixed
does not establish that a changed policy has the same violation rate, and zero observed
violations does not establish zero risk. Simulator interactions, model queries and elapsed
time should be accounted for separately, including adaptation and candidate evaluation.

## Decisions still open

UNISafe is a candidate starting point because its release contains relevant task-policy,
world-model and safety-filter components. Its runtime baseline, avoidable-stall prevalence
and evaluation cost still need to be established. The environment and world model are
not yet fixed.

We have also left open the behavioural parameters to adapt, the evolutionary algorithm,
the role of ROSARL, and the training and evaluation budgets. Holding the world model and
filter fixed initially could help attribute changes to task behaviour, but this remains
an experimental choice. The exploration is not restricted to earlier experiments or the
previous thesis schedule.

Physical robustness can become a later extension if changing conditions helps explain
the same continuation problem. The immediate focus is understanding task progress under
safety intervention in one concrete manipulation setting.

## Research trail

- [Research archive](archive/README.md): preserved Push-T Safe CEM, Hopper/Walker and
  earlier LeWM studies, with their original code, results and demonstrations.
- [Chosen-focus record](docs/research/research_notes/task-continuation-focus-20261008.md).
- [Comparative literature review and reading guide](docs/research/reports/Safe%20manipulation%20progress%20and%20robustness.md).
- [Detailed task-continuation literature](docs/research/research_notes/Safe%20manipulation%20progress%20and%20robustness/task_progress_after_intervention.md).
- [Available assets and evolutionary scope](docs/research/research_notes/Safe%20manipulation%20progress%20and%20robustness/feasibility_and_evolutionary_scope.md).
- [UNISafe compute and compatibility audit](docs/research/reports/UNISafe%20inference%20and%20training%20compute.md).
