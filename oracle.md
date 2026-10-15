# Oracle: task continuation under safety filtering

**Central reference for the research rethink. Updated 15 October 2026.**

We have chosen to focus on helping a robotic manipulation agent complete its task when
a learned safety filter intervenes. This choice remains conditional on finding meaningful,
avoidable stalls in an accessible task. The specific method and experiment plan remain open.

On 15 October 2026, Sachin also selected three adjacent ideas for further exploration:
evolutionary active perception, selection of safety-model training data, and evolved
temporal safety monitors. These are candidate supporting studies or alternative directions.
Task continuation remains the chosen conditional focus; no switch or combined experiment
plan has been decided.

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

## Adjacent directions to explore

The following questions broaden the exploration across sensing, safety-model learning and
monitoring. They do not depend on first establishing avoidable continuation stalls, although
each could help explain or address lost progress under filtering. Their usefulness,
implementation burden and specific contribution still need investigation. The linked work
provides foundations, not evidence that these questions are novel or already solved.

### Evolutionary active perception

> Can evolution discover short sensing motions that improve safety predictions and
> subsequent manipulation?

A safety filter may intervene because a contact or object is poorly observed. Evolution
could search a compact description of camera poses or inspection movements that gather
useful information before manipulation. A world model could supply representations,
predictions and uncertainty estimates for safety assessment.

A bounded first study could use one task with controlled occlusion and compare fixed
observation, a scripted inspection and evolved inspection. Evaluate prediction errors,
safe completion, physical violations and time spent gathering information. Lower predicted
uncertainty alone could reflect misplaced confidence. Inspection actions must remain under
safety filtering, and their interactions count toward the budget. A pretrained model's
usefulness under changed viewpoints needs checking; viewpoint changes may require training.

[Active Perception and Representation for Robotic Manipulation](https://arxiv.org/abs/2003.06734)
provides a manipulation foundation. The proposed emphasis is gathering information that
makes learned safety assessment more useful.

### Evolutionary selection of safety-model training data

> Can evolutionary selection produce a better safety model from the same amount of
> training data?

The coverage of a safety model's training examples may matter more than their raw number.
Evolution could select trajectory subsets or sampling proportions from a fixed pool,
seeking coverage of different failure mechanisms. One bounded implementation could freeze
the visual encoder and train a small failure predictor for each candidate selection.

Compare with random, class-balanced and uncertainty-based selection at matched data and
training budgets. Keep trajectory-level training, selection-validation and final test splits
separate. Measure missed failures and false alarms on untouched trajectories, then assess
closed-loop completion and physical violations. Filter interventions are not ground-truth
physical failure labels. Account for acquisition of the original pool and all candidate
training and evaluation, including discarded selections.

[UNISafe](https://arxiv.org/html/2505.00779) supplies a relevant foundation through its use of
successful and failed trajectories. The proposed question concerns safety-data coverage and
generalization; it does not assume that evolutionary subset selection outperforms simpler
sampling methods.

### Evolved temporal safety monitors

> Can genetic programming discover short temporal rules that predict manipulation
> failures early enough for useful intervention?

Warning signs such as rising contact force, gradual slipping or persistent object tilting
develop over time. Genetic programming could construct interpretable rules over observation
histories, using signals available to the deployed robot. Learned visual or world-model
features are possible inputs; additional force or tactile sensing is an open choice.

Begin with recorded trajectories and evaluate missed failures, false alarms and warning
time on held-out episodes. Compare with single-step predictors, simple temporal rules and
a learned sequence predictor using comparable information. A later closed-loop study would
test whether responding to warnings improves actual safety and completion. Physical
violation definitions remain fixed while the monitor is learned. A predictive warning is
not a certified barrier function or a guarantee of safe intervention.

[Interpretable Early Failure Detection via Machine Learning and Trace Checking-based Monitoring](https://arxiv.org/abs/2508.17786)
already uses genetic programming to learn temporal properties from historical traces.
The manipulation-specific monitoring and intervention contribution would need establishing.

## Research trail

- [Research archive](archive/README.md): preserved Push-T Safe CEM, Hopper/Walker and
  earlier LeWM studies, with their original code, results and demonstrations.
- [Chosen-focus record](docs/research/research_notes/task-continuation-focus-20261008.md).
- [Comparative literature review and reading guide](docs/research/reports/Safe%20manipulation%20progress%20and%20robustness.md).
- [Detailed task-continuation literature](docs/research/research_notes/Safe%20manipulation%20progress%20and%20robustness/task_progress_after_intervention.md).
- [Available assets and evolutionary scope](docs/research/research_notes/Safe%20manipulation%20progress%20and%20robustness/feasibility_and_evolutionary_scope.md).
- [UNISafe compute and compatibility audit](docs/research/reports/UNISafe%20inference%20and%20training%20compute.md).
