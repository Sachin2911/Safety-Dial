# Project progress

Updated 26 October 2026. One row per task. Each task's evidence lives in
[receipts/](receipts/README.md) under its number prefix. The [Odyssey](theOdessy.pdf)
holds a short narrative entry per finished task, with [LaTeX source](theOdessy.tex).
Research direction and stage boundaries are in [first principles](../firstPrinciples.md).

## Open

- **Task 15, regrasp adapter with a safety dial on Jar: running.** The live allocation
  is recorded in [its rental file](rental-regrasp15-20261025.json). See the
  [plan](15-regrasp-dial-plan.md) and the running [results record](receipts/15-regrasp-results.md).
- **Task 7, DSRL follow-up: experiment complete, closeout records pending.** Its
  [protocol](07-dsrl-repair.md), [rental record](rental-dsrl07-20261025.json) and the
  [task 6 plan](06-dsrl-plan.md) it reused stay here until the closeout is written,
  because its frozen source manifests, preservation script and tests read them.

## Tasks

| Task | Finding | Evidence |
| --- | --- | --- |
| 1, Jar replication | 24 safe successes and 43 counted violations in 104 episodes on machine 54514; substantial repeat variance | [Replication and noise floor](../docs/research/research_notes/Jar%20replication%20and%20noise%20floor.md) |
| 2, behavior audit | Contact and continuation distinguish many successes from failures | [Timelines](receipts/02-timelines.md), [pairs](receipts/02-pairs.md) |
| 3, retry controller | Safe success 17/96 native against 16/96 retry; no useful gain | [Results](receipts/03-results.md) |
| 4, native tracking | Safe requested completion 2/12; no general tracking deficit | [Results](receipts/04-tracking-results.md) |
| 5, execution adapter | Safe requested completion 1/12 native, 1/12 fixed, 0/12 adaptive; no useful gain | [Results](receipts/05-adapter-results.md), [executed plan v4](receipts/05-adapter-plan-v4.json) |
| 6, DSRL engineering | Numerical integration gates failed; no learning or efficacy estimate | [Result](receipts/06-dsrl-results.md), [completion](receipts/06-dsrl-completion/README.md) |
| 7, DSRL follow-up | Frozen completion screen failed: native 3/12 strict safe completions against 0/12 DSRL, both 4/12 violations | [Result](receipts/07-dsrl-results.md), [run](receipts/07-dsrl-run.md) |
| 8, execution-knob screen | Target scaling's screening gain did not replicate: 2/12 for both slow and native | [Result](receipts/08-screen-results.md), [completion](receipts/08-screen-completion/README.md) |
| 9, UNISafe mechanism and dial | Gain over random search not confirmed (54/64 against 51/64); grasp-sensing ablation confirmed (history-only lost 29.7 points); no speed/completion dial | [Result](receipts/09-unisafe-results.md), [report](../results/unisafe-follow-up-results.pdf), [completion](receipts/09-unisafe-completion/README.md) |
| 10, SafeCEM dial search | Searching the constraint interface beat the best fixed dial on the search layout, but not on the held-out layout | [Stage C2](receipts/10-safecem-C2-search.md), [preservation](receipts/10-safecem-preservation.md) |
| 11, DSRL improvement | Stopped after stage A: critic inflation reproduced, engineering episodes died in a simulator start-up race; no efficacy estimate | [Result](receipts/11-dsrl-results.md), [completion](receipts/11-dsrl-completion/README.md) |
| 12, TwoRoom planner search | Random search beat NSGA-II and the grid in 9 of 9 replicate pairs; planner horizon was the lever | [Stage C](receipts/12-tworoom-C-search.md) |
| 13, Reacher planner search | NSGA-II and random search tied; both beat the grid; gains did not carry to the held-out layout | [Stage C](receipts/13-reacher-C-search.md) |
| 14, Cube planner search | NSGA-II lost to random search; no search beat the best fixed dial on the search layout | [Stage C](receipts/14-cube-C-search.md) |

Full per-run evidence for every closed task is in the private dataset
`Sachioster/maniguard-jar-replication-20261022`, at the pinned revisions its preservation
receipt records. Tasks 10 and 12 to 14 also have packaged datasets
(`Sachioster/SafetyDial-PushT`, `-TwoRoom`, `-Reacher`, `-Cube`) that include their protocols.

## Panels

[panels.json](panels.json) is frozen. Development tasks are 0, 3, 6, 9, 15 and 17.
Pool tasks are for diagnostics. Held-out tasks receive no new episodes before the final
comparison. Baseline and task 2 evidence already include them, so they are not an unseen
dataset.

## Removed on 26 October 2026

Completed task protocols, handovers, agent prompts, the superseded twelve-mechanism
screen and the host rental records for closed tasks were removed. The committed ones are
in git history at `1291199`; links in receipts point there. The task 10 and 12 protocols
are inside their packaged datasets. The task 9 protocol and handover were never committed;
local copies are under `outputs/unisafe09/kept-unique/documents/`. Task 2's image sheets
are in the private dataset (see [02-preserve.md](receipts/02-preserve.md)). A historical
protocol or host record is never launch authority.
