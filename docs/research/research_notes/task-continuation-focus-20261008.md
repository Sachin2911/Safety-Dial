# Task continuation under learned safety filtering

Recorded 8 October 2026.

**Decision: chosen focus for the research rethink.** Sachin selected task continuation
after safety intervention from the two candidates in the
[comparative review](../reports/Safe%20manipulation%20progress%20and%20robustness.md).
The choice remains conditional on finding meaningful, avoidable stalls in an accessible
manipulation task. The method, environment and executable experiment plan remain open.

**Research question:** How can evolutionary adaptation help a pretrained manipulation
agent keep making useful progress when its learned safety filter intervenes?

The target outcome is improved safe task completion at a comparable adaptation budget.
Task success, physical violations and unfinished episodes must remain visible together.
Fewer interventions alone could reflect inactivity, and leaving the filter unchanged does
not establish that a changed policy has the same physical violation rate.

An illustrative failure is a robot redirected away from a dangerous pull, then repeatedly
attempting a movement that its filter rejects. Useful adaptation could change its next
action or its earlier approach so that this interaction no longer prevents completion.
This example is a motivating possibility, not a reproduced UNISafe result.

The scientific focus is the interaction between task behaviour and safety intervention.
A stalled episode could instead arise from an overly restrictive filter, inaccurate
predictions, missing information or insufficient time. The first unresolved issue is
whether poor task continuation is a substantial cause in the available system, and
whether useful progress remains possible under the same constraints. Failed attempts
alone do not establish that completion is impossible.

Evolution could adapt a compact part of existing task behaviour, with fitness measured
through the complete controller in the simulator. The world model can supply latent
state, predictive dynamics and safety assessment. This leaves the representation and
optimizer open; a new planning architecture is not needed to define the problem.

Ordinary task-policy learning with the same filter present is an essential comparison.
The literature already establishes this approach. A contribution would need to explain
what remains difficult and show a useful evolutionary adaptation result under comparable
information and interaction budgets. Improved imagined scores are not evidence of safer
physical outcomes.

UNISafe is a candidate starting point because it releases relevant components. Avoidable
stalling, a faithful runtime baseline and evaluation cost still need to be established.
Physical robustness remains a possible later extension if it helps explain the same
continuation problem.

Detailed evidence and existing remedies are in the
[task-continuation review](Safe%20manipulation%20progress%20and%20robustness/task_progress_after_intervention.md)
and [asset audit](Safe%20manipulation%20progress%20and%20robustness/feasibility_and_evolutionary_scope.md).
