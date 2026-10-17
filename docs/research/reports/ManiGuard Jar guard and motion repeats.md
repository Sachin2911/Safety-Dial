# ManiGuard Jar guard, motion and return repeats

**Completed 17 October 2026.** All 48 declared engineering trials finished normally and
passed outcome, action-record, source/configuration and cost audits. The fixed privileged
guard safely completes 4/12 trials versus identity's 2/12, with two versus four episodes
containing violations. This is a useful local lead. Smoothing adds a completion/safety
tradeoff, and settling/requerying does worse than the plain guard at much higher query cost.
A newly diagnosed mismatch between the guard's prediction and the controller's joint-limit
clipping must be corrected and tested before optimizer comparisons.

## Protocol and outcomes

The comparison uses all four frozen engineering Jar scenes and root seeds zero, one and
two, with one trial per condition/case. Every episode retains the original 2000-action
horizon, eight-action execution chunks, 20 Hz actions and 120 Hz physics. The pinned
pi0.5 Jar checkpoint at step 7400 remains frozen. The raw joint-position controller and
assisted-grasp mode match the declared setup. A fresh 32-action reference and six-query
interleaved replay passed before the batch. All 48 saved source snapshots, runtime hashes,
policy metadata and evaluation settings were checked; scene, seed and output directory
are the allowed evaluation differences.

| Condition | Safe complete | Unsafe complete | Safe unfinished | Unsafe unfinished | Policy queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identity | 2 | 1 | 6 | 3 | 2627 |
| Plain prerequisite guard | 4 | 0 | 6 | 2 | 2746 |
| Smoothing with default caps | 3 | 1 | 4 | 4 | 2520 |
| Guard with settling/requery | 3 | 0 | 7 | 2 | 10452 |

![All 48 outcomes and query totals](../figures/ManiGuard/jar-guard-motion-repeat-outcomes.png)

Every condition now has twelve completed same-scene/root comparisons against identity.
Guard has three safe-completion gains and one loss; settling/requery has three gains and
two losses; smoothing has two gains and one loss. Both guard variants have two fewer
violating episodes than identity, while smoothing has one more. Against the same plain
guard, adding settling/requerying produces one gain and two losses, the same number of
violations, and 10,452 versus 2,746 queries.

These are descriptive engineering comparisons across twelve cases, not twelve independent
scene replications. The scenes were chosen for development, and initial camera images and
policy proposals can differ despite equal root seeds and initial joint state. No reserved
generalization, deterministic causal repair or evolutionary advantage is established.
The guard variants use labelled privileged prerequisite/contact/holding signals; smoothing
uses robot observations and history. Safe unfinished means no observed violation, not
completion of an unmet temporal prerequisite.

## What the traces establish

All 434 plain-guard modifications occur while the goal target is explicitly unheld.
In each of its four successful runs, the changes finish before first target holding.
The gains are task_0013/root zero and task_0008/roots one and two. Identity remains
unfinished in the first two and completes unsafely in the third. The guard loses the
safe completion on task_0002/root zero and tips instead. None of the three task_0017
guard repeats reproduces that scene's original successful screen run.

Smoothing gains task_0013/root one and task_0008/root two, but loses task_0002/root zero.
Its first gain never activates the target caps; the second activates them on 17 actions
after first target holding. That second result combines smoothing and clipping and cannot
be attributed to smoothing alone. The final task_0013/root-two trial reaches the goal
unsafely after tipping while closed during transport. None of the three task_0017 smoothing
repeats reproduces its original success. Actual joint motion is not bounded by a target
step cap; submitted actions and measured movement remain distinct.

Settling/requerying completes task_0017/root one where the plain guard remains unfinished,
but loses the plain guard's successes on task_0002/root two and task_0008/root one. On all
three task_0002 roots, it never closes or holds the jar and spends 3879 queries versus
identity's 528; identity safely completes two of those cases. Its successful interventions
all end before target grasp. The added return behavior has no aggregate benefit over the
plain guard in this complete comparison.

## A concrete guard prediction mismatch

The task_0008/root-zero settling trial does hold the target, despite never closing it.
It first violates at observation 1873 when the open jar leaves support. Across the run,
497 guard-blocked proposals occur while the target is held; frequent intervention does
not ensure the prerequisite is protected.

At proposal 1872, the guard allows the action because its linearized estimate predicts
5.18 mm downward. The requested second-joint target is -1.853714 radians, below the pinned
robot asset's lower limit of approximately -1.762800. The installed position controller
clips its output to joint limits even though raw command normalization is disabled.
Recomputing the same estimate with that target clipped to the asset limits predicts
14.60 mm upward, exceeding the guard's 1 mm trigger. The measured end-effector rise is
13.67 mm, and the linearization of the actual joint change is 13.87 mm upward.

This is a prediction-interface mismatch with a specific correction to test. The saved
trace contains the submitted environment command, not a direct log of the controller's
clipped actuator target. The clipped prediction is an offline reconstruction supported by
the pinned asset and controller source. It is not a physical test of a corrected guard,
and clipping-aware linearization still would not provide a safety guarantee. The
[boundary audit](../research_notes/ManiGuard%20fruits%20Jar%20guard%20lift%20prediction%20mismatch.json)
preserves source hashes, limits, commands, measured states and monitor chronology.

## Costs and remaining work

These 48 trials consume **83,072 policy actions, 502,560 physics ticks and 18,345 policy
queries**. The two-worker batch takes **6605.96 seconds** (110.10 minutes). Summed episode
ledger times are 12,802.20 seconds and summed process times are 12,967.83 seconds; neither
is batch wall time. Summed policy-query latency is 5056.11 seconds and can include server
queueing. Reference episodes, inference checks, failed setup attempts and other batches
are additional study costs, not included in these batch totals. Reused controls must not
be added again when reporting alternate-control comparisons.

The unchanged sixteen-trial Lid guard comparison started after this batch's normal exit,
with a fresh reference and strict replay passing first. Preserve that declared comparison.
Next, correct and test the controller-limit mismatch using a separately identified guard
variant and matched controls. Reassess useful correction headroom before evolving an
adapter around the guard. No optimizer, selection-validation or reserved evaluation has run.

Evidence: [complete setup and cost audit](../research_notes/ManiGuard%20fruits%20Jar%20repeat%20complete.json),
[identity comparisons](../research_notes/ManiGuard%20fruits%20Jar%20repeat%20comparison.json),
[plain-guard comparisons](../research_notes/ManiGuard%20fruits%20Jar%20repeat%20handback%20versus%20guard%20complete.json),
[all diagnoses and timelines](../research_notes/ManiGuard%20fruits%20Jar%20repeat%20progress.json),
and [development record](../research_notes/ManiGuard%20fruits%20development.md).
