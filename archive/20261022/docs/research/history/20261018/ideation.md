# Research ideation: evolutionary adaptation for safe manipulation

The next phase is now planned in [phase2.md](phase2.md): a shared execution controller,
cross-task optimization and reserved transfer evaluation. The candidate discussion below
is retained as its research background; the phase-two plan records the revised scope.

**Updated 17 October 2026.** Sachin selected two ManiGuard ideas for initial exploration:
an execution adapter around a frozen vision-language-action policy (VLA), and a compact
controller that coordinates task stages and recovery. These discussion notes develop the
research rethink alongside [oracle.md](oracle.md). The two ideas are candidates to
investigate, not a settled architecture, evolutionary method or experiment budget.

## Ambition

The project is Sachin Mohan's BSc Honours research at the University of the Witwatersrand,
supervised by Geraud Nangue Tasse. It must support an end-of-year honours submission,
with the ambition of presenting the research at GECCO 2027.

The broader interest remains the intersection of world models, learned safety filters,
evolutionary algorithms and robotic manipulation. Combining those components establishes
the setting; the contribution must explain a consequential difficulty and show what the
chosen adaptation changes. The first investigation should target substantial, repairable
headroom whose cause can be understood before investing in a larger learning system.

The guiding question is:

> What would someone learn from this project that they could not learn merely by
> assembling these three components?

## What changed after UNISafe

The earlier motivation depended on safety filtering disrupting task progress. Our
[completed UNISafe study](docs/research/reports/UNISafe%20grasp%20maintenance%20and%20evolutionary%20adaptation.md)
identified premature gripper release and a useful grasp-maintenance correction. Across
64 reserved starts, baseline produced two successes, six physical failures and 56 timeouts.
Evolution produced 54 successes, seven physical failures, one boundary exit and two
timeouts. Simpler comparisons matched or exceeded evolutionary completion, and most of
the improvement persisted without filtering. Filter-induced loss of progress, an
evolutionary advantage and unchanged safety risk were not established.

This result remains useful evidence. It motivates reopening problem selection and checking
simple corrections early. It does not prove that safety filters never hamper completion.
Sachin has set aside safety-model training-data selection and failure-data acquisition as
research directions for this exploration, following the earlier Walker experience.

## Working research question

> Can compact evolutionary adaptation improve the safe execution of a pretrained
> manipulation policy, and when does it add value beyond simple controls and ordinary
> learning?

Poor execution, weak individual skills, missing observations and unsafe coordination are
different possible causes of low safe completion. The first task is to establish which
cause offers avoidable failures and an appropriate behavioural search space.

## Why ManiGuard is a candidate

[ManiGuard](https://arxiv.org/html/2608.17386v1) provides manipulation tasks, explicit
safety specifications, pretrained VLA checkpoints and separate task-success and safety
measurements. Its runtime monitor records physical specification violations; a predictive
filter that changes proposed actions would be an additional component.

Jar is a promising first family to inspect. The task requires closing a hinged jar before
lifting it, keeping it upright and carrying it to a goal region. The paper reports 21
safe completions, four unsafe completions, 27 safe unsuccessful episodes and 26 unsafe
unsuccessful episodes for fine-tuned pi0.5 across 26 tasks and three policy-sampling seeds.
These outcomes show some competence and substantial remaining difficulty, without yet
identifying its cause. See the [Jar task description](https://nu-ideas-lab.github.io/ManiGuard/docs/pipelines/jar_transport/).

ManiGuard uses OmniGibson on Isaac Sim. Its released simulator and policy interfaces need
their own compatibility and baseline checks on the run machine. Our existing Isaac Lab
work is useful experience, but does not establish that ManiGuard is already runnable.

## Candidate 1: execution adapter around a frozen VLA

> Can a small evolved execution adapter improve safe completion while the pretrained
> VLA remains fixed?

ManiGuard's [policy interface](https://nu-ideas-lab.github.io/ManiGuard/docs/evaluation/)
returns action chunks. A bounded adapter could control how much of a chunk executes before
replanning, limit joint-target changes, smooth motion or apply a few corrective offsets.
The diagnosed failure should determine which parameters are useful; these examples are
not a chosen search space. Changes must respect the policy's
[action and controller conventions](https://nu-ideas-lab.github.io/ManiGuard/docs/fine_tuning/end_to_end/).

Evolution would evaluate complete episodes across multiple scenes and search for settings
that improve measured safe completion. Changing the execution horizon also changes VLA
query frequency and cost, so separate that effect from the benefit of the optimizer.
Record original proposals, adapter outputs and executed actions distinctly.

This is the lower-cost screening idea. Compare the unchanged policy, a simple fixed
correction, random search and evolution using the same adapter family and observations.
An ordinary-learning comparison within that family is needed for an evolutionary claim.
If a fixed setting or random search closes the gap, retain the behavioural finding and
reconsider whether further evolutionary development is justified. Training a new VLA,
world model or safety filter is not a prerequisite for this initial comparison.

## Candidate 2: controller for task stages and recovery

> Can an evolved compact controller coordinate task stages, retries and corrections more
> effectively than simple sequencing or ordinary learning?

A finite-state controller or behaviour tree could decide when to continue the VLA,
advance to another subgoal, retry, change approach or execute a short correction. The
potential evolutionary role is searching combinations of transitions and alternative
behaviours that respond to different situations.

In Jar, closing, grasping, lifting and transporting must work together. The obvious order
of closing before lifting needs no evolutionary discovery. The useful question is whether
different off-nominal situations require different retries or corrections, and whether
those choices can be made from available observations and history.

First establish that component behaviours and proposed corrections work from valid states.
Compare their full sequence with a simple scripted controller using the same information.
The benchmark does not automatically provide a bank of reliable learned primitives. If
most failures occur within one primitive, or a short scripted dispatcher resolves them,
the proposed coordination contribution needs reconsideration.

If complementary alternatives exist, compare evolution with fixed sequencing, a simple
retry dispatcher, random search and ordinary learning over the same controller interface.
Failed finite recovery search does not prove that a state is irrecoverable. Use full
episodes or validated prefix replay before assuming exact mid-episode restoration.

## Roles for world models and safety filters

The two ideas allow us to test useful behavioural search before committing to a full
model-based stack. A world model could later predict the consequences of candidate
corrections or task-stage choices. A learned safety filter could assess proposals before
execution and select a tested alternative when risk is predicted. Those roles should
follow a measured need, with new training and integration costs included.

ManiGuard's physical monitor can provide evaluation labels. Simulator-only object states
or monitor inputs must not silently become privileged runtime observations for one method.
If a privileged-state prototype is useful for diagnosis, label it explicitly and match
information across its comparisons. Temporal requirements also need appropriate history
or specification-state information; a current-image hazard score may be insufficient.

Evolutionary fitness can come from actual simulator outcomes. If imagined outcomes guide
selection, validate their relationship to executed physical outcomes separately. The
role of ROSARL remains open and is not a dependency of either first pilot.

## Evidence that would make the story convincing

Reproduce one pretrained family policy, inspect representative safe and unsafe outcomes,
and identify a failure mechanism before choosing an adapter or controller. Jar is a
starting candidate; a richer family such as Cabinet could follow if its component skills
and integration cost support a meaningful study.

Explain what adaptation changes and whether the benefit generalizes across reserved
scenes. Keep model training, adaptation, selection and final evaluation separate, grouping
correlated trajectories and variants of the same scene appropriately. Distinguish held-out
adaptation cases from cases unseen during the released policy's pretraining or fine-tuning.

Report safe completion, violations and unfinished episodes together, retaining ManiGuard's
safe and unsafe outcome categories. Include engagement and completion time where useful;
inactivity can look safe. Count all simulator interactions, including discarded candidates,
retries and diagnostic witnesses, and report model queries, training and elapsed time
separately. A fixed filter does not guarantee unchanged risk after behavioural adaptation.

A standard EA can support a useful empirical contribution, but an optimizer substitution
alone is insufficient. Ordinary RL can learn through a nondifferentiable filter, so
nondifferentiability does not establish an evolutionary advantage. Recheck primary
literature before claiming novelty or publication suitability.

## Scope and open choices

For honours, aim for one well-explained failure and one thoroughly evaluated response.
GECCO could build on that through stronger methodological development or broader evidence,
depending on what the initial work reveals. This is an ambition, not a publication promise
or a requirement to invent a new EA.

The immediate preference is to inspect ManiGuard and test a small behavioural interface
before building a larger learning system. Candidate 1 is the cheaper screening experiment;
Candidate 2 has a more structured evolutionary search role if meaningful coordination
and recovery difficulties are demonstrated. Neither is an established contribution yet.
The final task family, runtime signals, search representation, optimizer, world-model and
filter choices, ROSARL role and experiment budgets remain open.

Supporting context is in the
[opportunity review](docs/research/reports/Evolutionary%20opportunities%20for%20safe%20robotics.md)
and [earlier manipulation comparison](docs/research/reports/Safe%20manipulation%20progress%20and%20robustness.md).
The opportunity review records the broader screening before Sachin selected these two
behavioural ideas; its initial ranking is not the current preference.
