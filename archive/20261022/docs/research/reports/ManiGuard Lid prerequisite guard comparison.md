# ManiGuard Lid prerequisite guard comparison

The fixed prerequisite guard does not improve safe completion in this sixteen-trial
engineering comparison. Neither condition safely completes any trial; each violates
once. Guarding changes the sole unsafe completion into an unsafe unfinished episode
in the same scene/root pairing and uses more policy queries. This is not a useful
continuation repair.

## Complete outcomes

| Condition | Safe complete | Unsafe complete | Safe unfinished | Unsafe unfinished | Policy queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged policy | 0 | 1 | 7 | 0 | 1,728 |
| Prerequisite guard | 0 | 0 | 7 | 1 | 2,887 |

![Complete Lid outcomes and query costs](../figures/ManiGuard/lid-guard-outcomes.png)

All eight same-scene/root pairs are complete: zero safe-completion gains, zero losses
and equal violation counts. The seven remaining pairs are safe unfinished under both
conditions. Only the two task_0003/root-two trials ever hold the goal container. Fewer
violations in inactive episodes would not establish safe task competence.

## Protocol and verification

The declared plan covers engineering scenes task_0003, task_0006, task_0008 and task_0014,
roots one and two, under identity and the original requested-target guard. Every trial
uses the original 1900-action horizon, execution horizon eight, sticky grasping, raw
joint-position targets, 20 Hz actions and 120 Hz physics. The guard has privileged
prerequisite/contact information and a 1 mm predicted-lift trigger. It is a prototype
intervention rule, not a certified safety filter.

The pi0 Lid checkpoint is step 8250, pinned to revision
`5bff6f0cd4d232874f86a86bdb0a0c1662f95226`. All sixteen saved code snapshots, instrumented
runtime hashes, policy metadata and evaluation settings agree; scene, seed and output
path are the declared variable evaluation fields. Proposal and execution records are
complete and ordered, adapted commands exactly match environment inputs, gripper commands
are unchanged, and intervention counts agree with the cost ledgers. Environment input
records do not expose lower-level controller output clipping. Equal scene/root settings
do not make separate physical starts identical counterfactuals.

Before the batch, a fresh reference completed 32 actions and four policy queries, and
six strict interleaved replay queries passed, including agreement with the reference's
first proposal. The batch and its managed supervisor exited normally before any runtime
code or policy family was changed.

## Why the guard failed

On task_0003/root two, identity first holds at observation 145, loses support before lid
placement at 146, and finishes unsafely after 489 actions and 62 queries. Guarding first
holds at 153, starts blocking at proposal 163, violates at observation 171 and remains
unfinished after 1900 actions and 1220 queries. All 1198 changed commands occur while
the target is held. Neither trial records lid placement.

At proposal 170 the guard permits a predicted upward movement of 0.676227 mm because
it is below the fixed 1 mm trigger. Measured end-effector rise is 0.669837 mm, and support
becomes false while the open container is held. Both adjacent proposals are blocked.
Clipping this requested target to the robot's joint limits leaves the prediction
unchanged. This is a trigger-tolerance limitation, separate from the Jar joint-limit
prediction defect. The boundary records and calculation are preserved in the
[tolerance audit](../research_notes/ManiGuard%20fruits%20Lid%20guard%20tolerance%20boundary.json).

The only other guard changes are two commands on task_0014/root one, both while the
goal target is unheld. That trial briefly records lid placement but never holds the
container. Six guard trials make no changes at all. Inactivity or occasional placement
without transport does not demonstrate that the guard repairs an unsafe approach.

## Interaction costs and scope

The sixteen episodes consume 28,989 policy actions, 175,310 physics ticks and 4,615
policy queries. Batch wall time is 1809.14 seconds with two workers; episode ledger
elapsed times sum to 3515.74 seconds and are not wall time. Recorded query latency sums
to 1107.87 seconds. Reference and replay work are retained separately in the complete
audit and belong in whole-study accounting.

This completes the declared Lid comparison. It supplies no basis for optimizing this
fixed rule on Lid. The separately declared twelve-trial Jar comparison tests whether
controller-limit-aware prediction repairs the diagnosed Jar interface defect, keeping
the 1 mm trigger fixed. No evolutionary optimization, selection-validation or reserved
evaluation has run in this ManiGuard study.

## Evidence

- [Complete outcomes, costs and source audit](../research_notes/ManiGuard%20fruits%20Lid%20guard%20complete.json).
- [Eight paired comparisons](../research_notes/ManiGuard%20fruits%20Lid%20guard%20comparison.json).
- [All sixteen diagnostics and intervention timelines](../research_notes/ManiGuard%20fruits%20Lid%20guard%20progress.json).
- [Figure input audit](../research_notes/ManiGuard%20fruits%20Lid%20guard%20plot%20audit.json).
- [Declared plan](../../../configs/maniguard/lid-guard-screen.json).
