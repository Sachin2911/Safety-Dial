# Project progress

Updated 25 October 2026. Task 7 is in progress under the authorized
[DSRL follow-up](07-dsrl-repair.md), retaining the cumulative $15 cap including
task 6. Fixed-batch deterministic SAC comparisons pass exactly, as do all
same-process decoder repeats and native-draw passthrough checks. Cross-process
differences are recorded separately. At 08:41 UTC, installation is complete and the
first live engineering episode is running. Development learning has not run. See the
[run record](receipts/07-dsrl-run.md) and current
[source freeze v3](receipts/07-dsrl-stage-a-v3.json). Numerical integration does not
establish safe-completion efficacy.

Tasks 1 through 6 are complete.
[Task 6 DSRL engineering](receipts/06-dsrl-results.md) stopped at failed numerical
integration gates after four native pool episodes and the full decoder probe.
No development learning or evaluation ran. All 166 local tests passed; real GPU
gates remained strict. Instance 57575920 is destroyed and provider absence is
verified. All 1,623 primary evidence files were downloaded from pinned private
revisions and their hashes matched. The provider charge snapshot is $3.332; the
conservative estimate is $3.54 against the $15 cap. Final preservation, cleanup
and billing proofs are in the [completion receipts](receipts/06-dsrl-completion/README.md).
See the [run record](receipts/06-dsrl-run.md) and
[rental record](rental-dsrl06-20261025.json).

The [static shape audit](receipts/06-dsrl-shape-audit.json) confirms 16 by 32 without
a GPU. The [decoder probe specification](receipts/06-dsrl-decoder-probe.json) now
required magnitude-1 gripper-command authority before learning. Cross-lane replay
failed identity. Descriptive sensitivity in one of eight inputs is not a valid
authority or absence result. A new experiment needs its own frozen plan.

## Current reading order

1. [First principles](../firstPrinciples.md): research direction and stage boundaries.
2. [DSRL follow-up plan](07-dsrl-repair.md): current numerical repair, live validation and learning allocation.
3. [Current run](receipts/07-dsrl-run.md): measured progress and source amendments.
4. [Task 6 result](receipts/06-dsrl-results.md) and [original plan](06-dsrl-plan.md): historical outcome and the retained learning design.
5. [The Odyssey](theOdessy.pdf): short entries for completed work, with [LaTeX source](theOdessy.tex).

## Completed experiments

| Task | Finding | Evidence |
| --- | --- | --- |
| 1, Jar replication | 24 safe successes and 43 counted violations in 104 episodes on machine 54514; substantial repeat variance | [Replication and noise floor](../docs/research/research_notes/Jar%20replication%20and%20noise%20floor.md) |
| 2, behavior audit | Contact and continuation distinguish many successes from failures | [Timelines](receipts/02-timelines.md), [paired videos](receipts/02-pairs.md) |
| 3, retry controller | Safe success 17/96 native versus 16/96 retry; no useful gain established | [Results](receipts/03-results.md) |
| 4, native tracking diagnostic | Safe requested completion 2/12; no general tracking deficit established | [Protocol](04-tracking-check.md), [results](receipts/04-tracking-results.md) |
| 5, bounded execution adapter | Safe requested completion 1/12 native, 1/12 fixed, 0/12 adaptive; no useful gain established | [Protocol](05-execution-adapter.md), [results](receipts/05-adapter-results.md) |
| 6, DSRL engineering | Native replay and GPU SAC numerical gates did not pass; no development learning or efficacy estimate | [Result](receipts/06-dsrl-results.md), [decoder audit](receipts/06-dsrl-decoder-audit.md), [learner audit](receipts/06-dsrl-learner-runtime.md) |

Completed protocols remain at their recorded paths because receipts and source
manifests refer to them. Their original authorization text does not authorize new runs.
Task 5's executed machine-readable protocol is
[05-adapter-plan-v4.json](receipts/05-adapter-plan-v4.json); earlier versions remain evidence.
The [receipt index](receipts/README.md) identifies the final records for each task.

## Panels and rental records

[panels.json](panels.json) remains frozen. Development tasks are 0, 3, 6, 9, 15 and 17.
Pool tasks are for diagnostics. Held-out tasks receive no new episodes before the
final comparison. Baseline and task 2 evidence already include those tasks; they
are not an unseen dataset.

Rental files stay at their existing paths so the preserved scripts and manifests
continue to resolve them. A record is historical evidence, not an instruction to rent
or restart its instance.

| Record | Scope |
| --- | --- |
| [rental-20261022.json](rental-20261022.json) | Earlier Vast allocations, including task 3 |
| [rental-runpod-20261023.json](rental-runpod-20261023.json) | Task 2 analysis host |
| [rental-tracking04-20261024.json](rental-tracking04-20261024.json) | Task 4 allocation |
| [rental-adapter05-20261024.json](rental-adapter05-20261024.json) | Task 5 allocations |
| [rental-dsrl06-20261025.json](rental-dsrl06-20261025.json) | Completed task 6 allocation |
| [rental-dsrl07-20261025.json](rental-dsrl07-20261025.json) | Active task 7 allocation |

## Superseded material

The unexecuted twelve-mechanism screen and its launch prompt now live under
[superseded/](superseded/README.md). Notices at their old paths preserve existing links.
The pre-22 October archive is unchanged. Unrelated research drafts are unchanged.
