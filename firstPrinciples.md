# First principles plan

Sachin Mohan. Established 22 October 2026; current status updated 25 October 2026.
The goal remains a compact evolved controller that raises safe task completion of
frozen ManiGuard policies under matched interaction budgets. The research controls
in [AGENTS.md](AGENTS.md) apply throughout. Jar is the chosen family; Stack is the backup.

## Current decision

Tasks 1 through 6 are complete. The baseline has useful variance, but the tested
retry and execution-feedback presets have not established a useful lever. Sachin
selected DSRL noise steering, four RTX 4090 GPUs and a hard $15 cap, then authorized
implementation and execution. Task 6 completed four native engineering episodes
and its decoder probe, then stopped at failed numerical integration gates. It did
not train or evaluate a DSRL controller. See the
[result](progress/receipts/06-dsrl-results.md). A useful learned lever remains
unestablished.

This decision replaces the original scalar-only restriction on stage 2. Ordinary
learning is now the next way to test whether the frozen policy offers useful steering.
It does not skip the requirement to demonstrate a lever before an evolutionary search.
The earlier plan remains in git history; its budgets and immediate-action text are
not current launch instructions. Use the [progress index](progress/README.md) for status.

## What must be established in order

1. **Variance.** The native policy sometimes completes safely and sometimes does not.
2. **A lever.** A controller changes safe completion, rather than merely changing actions.
3. **Stable evaluation.** Enough episodes and repeats to distinguish improvement from noise.
4. **An optimizer advantage.** Evolution improves on appropriate fixed, random and ordinary-learning controls.

Each stage has its own frozen protocol. A negative result belongs in the record and
does not authorize an unplanned extension, another seed or a restarted archived pilot.

## Stage 1 Baseline and variance

Completed on 23 October. The paper-protocol Jar reproduction on machine 54514 ran
104 episodes: 24 safe successes, 7 unsafe successes, 37 safe failures and 36 unsafe
failures. Counted violations occurred in 43/104 episodes and engagement in 103/104.
The observed safe-success and violation rates were compatible with the published
Jar rates. Two repeats of one seed agreed on the four-class outcome for 13/26 tasks.
See [Jar replication and noise floor](docs/research/research_notes/Jar%20replication%20and%20noise%20floor.md).

The frozen [panels](progress/panels.json) separate development, diagnostic pool and
held-out tasks. Held-out tasks already have baseline and descriptive evidence;
they receive no new intervention episodes before the final comparison.

## Stage 2 Find a useful controller

The retry controller produced 16/96 safe successes against 17/96 native, with no
observed improvement. The native tracking diagnostic found no general tracking
deficit, although one contact-associated discrepancy kept a narrow feedback hypothesis
open. The subsequent bounded-adapter pilot produced safe requested completion of
1/12 native, 1/12 fixed and 0/12 adaptive. These are separate studies with separate
controls, not a pooled estimate. [Retry results](progress/receipts/03-results.md),
[tracking results](progress/receipts/04-tracking-results.md),
[adapter results](progress/receipts/05-adapter-results.md).

Task 6 implemented a DSRL noise actor around the frozen pi0.5 checkpoint using the
author's SAC code. GPU SAC parity and cross-lane native decoder replay failed their
numerical criteria. All 128 decoder queries were used; learning was not admitted.
Descriptive sensitivity in one of eight inputs cannot establish absence or efficacy
under an invalid identity gate. The [plan](progress/06-dsrl-plan.md) and stage freezes
are historical launch evidence. A revised integration protocol or another lever
needs a new decision; unused allocations do not authorize extensions.

A future valid pilot could support another development replication, not an
evolutionary or generalization claim. A fixed-zero control is not a tuned best fixed setting, and
random noise per query is not parameter random search. Those comparisons must be
added before attributing an advantage to an optimizer. The actor's parameter count
and the cost of any learned feature encoder must be visible when defining a later
compact evolutionary search space.

## Stage 3 Size a stable comparison

After a useful lever is observed and replicated, use its per-task variance, learner-seed
variance and measured rollout speed to fix the comparison. Freeze the controller
domain, evaluation panel, training and selection budgets, outcome definitions, number
of optimizer seeds and analysis before running searches. Historical episode seeds do
not guarantee identical trajectories. Keep new control and intervention results
blocked by physical host.

The final held-out task panel stays unavailable for method selection. A small
development screen is not a power calculation. Use uncertainty appropriate to task
and training-seed replication, rather than treating every correlated rollout as an
independent experiment.

## Stage 4 Test the evolutionary contribution

Compare evolution with parameter random search, a tuned fixed controller, native
execution and an ordinary-learning controller using the same allowed observations,
action/noise interface and reward information. Charge shared representation learning,
tuning, rejected candidates and validation interactions explicitly. Match simulator
interaction budgets and report model queries, learner updates and wall time separately.

An evolutionary claim requires replicated improvement on reserved evaluations with
safe completion up and violations not increased within the stated uncertainty. A
positive DSRL result establishes a comparator and a candidate mechanism; it does not
by itself establish novelty or an evolutionary advantage. Transfer and broader safety
tradeoff studies need their own justified protocols.

## Controls that remain in force

- Preserve the benchmark protocol and label every intervention-specific deviation.
- Report benchmark and requested-sequence completion, raw and counted violations,
  unfinished episodes and engagement together, with denominators.
- A monitor labels outcomes; it never vetoes commands. Distinguish proposed,
  clipped and executed actions.
- Count every simulator interaction, including engineering, failures, discarded
  attempts and selection. Report queries, gradient updates and elapsed time separately.
- Freeze each stage before its first eligible episode. Do not use held-out outcomes
  to select the next method, reward or budget.
- Preserve original evidence and use fresh output paths for new experiments.
- Do not restart the superseded twelve-mechanism screen or any pre-22 October pilot.

## Immediate next action

Review the [task 6 result](progress/receipts/06-dsrl-results.md) before choosing new
work. The evidence distinguishes numerical integration failures from descriptive
decoder sensitivity. Same-lane replay and deterministic learner kernels are possible
engineering questions for a separately frozen follow-up; neither was tested as a
new protocol in this run. Development and held-out tasks received no new episodes.
The next allocation, integration amendment or method needs its own concrete plan.
Do not resume the stopped task 6 queue.
