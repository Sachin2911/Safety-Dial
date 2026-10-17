# ManiGuard controller-limit guard comparison

The twelve-trial correction study finds no safe-completion gain. Both corrected variants
have zero safe completions in three trials. Joint-limit-aware prediction repairs a
specific calculation error in our prototype guard, but these trials do not establish a
useful physical recovery or justify an optimizer comparison.

## Complete outcomes

| Condition | Safe complete | Unsafe complete | Safe unfinished | Unsafe unfinished | Queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original plain guard | 0 | 1 | 1 | 1 | 641 |
| Corrected plain guard | 0 | 1 | 2 | 0 | 705 |
| Original hand-back guard | 1 | 0 | 1 | 1 | 3,183 |
| Corrected hand-back guard | 0 | 1 | 2 | 0 | 2,604 |

![Twelve correction-trial outcomes and query costs](../figures/ManiGuard/jar-controller-limit-outcomes.png)

All three plain-guard pairs have no safe-completion gain or loss; corrected prediction
has one fewer violating episode. All three hand-back pairs have no gain, one loss and
equal violation counts. The sole benchmark safe completion, original hand-back/root two,
never records jar closure. It satisfies the spatial held-intersection goal while still
supported, with a monitor state that is neither accepting nor doomed. The table preserves
the benchmark label; it does not establish the requested close-then-carry sequence.

## Why this correction was tested

A completed Jar repeat exposed a mismatch between the prototype guard's requested-target
lift prediction and the robot controller's joint-limit clipping. At the recorded failure,
the requested-target prediction is -5.18 mm, the reconstructed feasible-target prediction
is +14.60 mm, and measured end-effector rise is +13.67 mm. The open, held jar loses support.
The [original boundary audit](../research_notes/ManiGuard%20fruits%20Jar%20guard%20lift%20prediction%20mismatch.json)
preserves the calculation, controller code and robot-asset provenance.

The corrected variant estimates lift from the joint target after applying live controller
position limits. It preserves both predictions in its trace and leaves accepted policy
actions on their original execution path. The option defaults off for historical
compatibility and is enabled explicitly by the corrected configs. It passes 90 focused
software tests, including the recorded failure, plus exact legacy replay of 41,913 actions
with the correction disabled. A fresh 32-action physical reference checks live limits and
prediction reconstruction; six strict interleaved policy replay queries also pass.

The declared physical plan uses the diagnosed engineering scene, Jar task_0008/base,
roots zero, one and two, and original/corrected versions of plain guard and hand-back.
Within each pair only the prediction option changes. Every trial keeps the 2000-action
task horizon, execution horizon eight, 1 mm guard trigger, privileged prerequisite and
engagement signals, raw joint-position control, assisted grasping, 20 Hz actions and
120 Hz physics. Hand-back retains four settling holds and fresh querying. The Jar policy
remains frozen at step 7400, revision `1d84eda070313a202a595e449fcf41a1a1e8a546`.

Original controls were rerun because repeated scene/root settings do not yield identical
physical starts. The six-pair [input audit](../research_notes/ManiGuard%20fruits%20controller-limit%20initial%20inputs.json)
checks configuration equality apart from the intended option, shared code/checkpoint,
and initial state, images and first proposals. These are small physical engineering
comparisons, not identical-state counterfactuals or a population estimate. There is no
unguarded identity condition in this correction study.

## How often the correction affected decisions

Across all six corrected runs, 3408 proposals request an arm target outside a joint limit.
Clipping changes fourteen comparisons against the lift threshold, but only two eligible
blocking decisions. Both occur with the goal target unheld:

- Plain guard/root one adds a block at proposal 1165: requested prediction -3.496 mm,
  feasible-target prediction +1.092 mm. The episode never closes or holds the jar.
- Plain guard/root two removes a block at proposal 548: requested prediction +1.294 mm,
  feasible-target prediction -0.073 mm. This episode also never closes or holds the jar.

The corrected plain-guard/root-zero trial has no outside-limit target. None of the three
corrected hand-back trials changes an eligible blocking decision because of clipping.
Consequently, this study verifies the corrected calculation in live execution but does
not demonstrate interception of the original held-jar clipping failure. Different outcome
counts cannot be attributed to a correction that made no different decision on a recorded
trajectory.

The corrected plain guard's two safe unfinished episodes have no closure or target grasp.
Corrected hand-back/root zero closes at observation 590 but never grasps; root one neither
closes nor grasps. These are not successful continuations. Root two holds an open jar for
943 observations, violates at 1376, and completes unsafely at 1387 actions despite 893
changed commands with the target held.

## Remaining failure mechanisms

All five violating episodes have boundary targets within joint limits. The
[complete boundary audit](../research_notes/ManiGuard%20fruits%20controller-limit%20violation%20boundaries.json)
separates their observed failures:

- Corrected plain guard/root zero allows a predicted +0.830 mm rise below the 1 mm trigger.
  Measured rise is +1.264 mm, and the held, open jar loses support.
- Original hand-back/root zero and corrected hand-back/root two lose support while the
  end effector moves slightly downward. Both estimates are also downward. End-effector
  height alone is not a complete predictor of the jar's support/contact state; these
  observations do not identify the precise contact dynamics.
- Original plain guard/root zero tips the jar while it is still unheld and open. Root two
  tips after closure while carrying. A closure-before-upward-motion rule does not guard
  against every physical specification violation.

These are residual limitations of the prototype, not evidence that applying joint limits
is mathematically wrong. The study retains the declared threshold and safety definitions;
it does not tune them after observing these outcomes.

## Costs, validation and decision

The twelve episodes consume 19,983 policy actions, 120,930 physics ticks and 7133 policy
queries. The two-worker batch takes 2094.72 seconds; episode ledger durations sum to
3966.07 seconds and are not wall time. Recorded query latency sums to 2069.35 seconds.
Reference and replay costs remain separate and enter the final whole-study inventory.

Every saved source/runtime hash, policy metadata record and common evaluation setting
passes validation. Proposals and environment input actions are complete, ordered and
exactly equal after adaptation; gripper commands are unchanged. Every lift prediction
and blocking decision is reconstructed from its recorded frame. The managed batch exits
normally with all twelve outcomes and cost records present.

Retain the controller-limit-aware estimator as an explicit, tested implementation option.
The physical study does not establish a repeatable simple correction that improves safe
task completion. No evolutionary optimization, ordinary-learning comparison, selection
or reserved evaluation is warranted by this screen's results. This is a conclusion about
the tested mechanisms and settings, not proof that the broader hypotheses cannot work.

## Evidence

- [Complete outcomes and costs](../research_notes/ManiGuard%20fruits%20controller-limit%20guard%20complete.json).
- [All paired comparisons](../research_notes/ManiGuard%20fruits%20controller-limit%20guard%20comparison.json).
- [All twelve command, prediction and timeline audits](../research_notes/ManiGuard%20fruits%20controller-limit%20guard%20progress.json).
- [Final completion-semantics audit](../research_notes/ManiGuard%20fruits%20completion%20semantics%20final.json).
- [Declared plan](../../../configs/maniguard/jar-controller-limit-screen.json) and
  [setup/reproduction commands](../../../configs/maniguard/README.md).
