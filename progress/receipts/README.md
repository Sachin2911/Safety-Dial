# Experiment receipts

Receipts describe what was planned, executed, observed and preserved. Keep their
contents and paths stable. A historical plan or host record is not launch authority.
Use the [progress index](../README.md) to find the next approved work.

| Task | Main records |
| --- | --- |
| 1 | `01-*`, `01b-*`, `01c-*`; [paper evaluation configuration](01-jar-eval.yaml); [excluded instance B](excluded-instance-B/README.md) |
| 2 | [Timelines](02-timelines.md), [pairs](02-pairs.md), [task notes](02-task-notes.md); image sheets are in the private dataset (see [preservation](02-preserve.md)) |
| 3 | [Frozen plan](03-plan.json), [results](03-results.md), [preservation](03-preserve.md) |
| 4 | [Frozen plan](04-tracking-plan.json), [results](04-tracking-results.md), [preservation](04-tracking-preserve.md), [completion](04-tracking-completion.json) |
| 5 | [Executed plan v4](05-adapter-plan-v4.json), [results](05-adapter-results.md), [preservation](05-adapter-preserve.md), [completion verification](05-adapter-completion-verification.json), [billing](05-adapter-billing.json) |
| 6 | [Result](06-dsrl-results.md), [run](06-dsrl-run.md), [executed source v4](06-dsrl-stage-a-v4.json), [decoder audit](06-dsrl-decoder-audit.md), [learner audit](06-dsrl-learner-runtime.md), [independent accounting](06-dsrl-independent-audit.json), [completion](06-dsrl-completion/README.md) |
| 7, experiment complete; preservation closing | [Result](07-dsrl-results.md), [analysis](07-dsrl-analysis.json), [run](07-dsrl-run.md), [final source freeze](07-dsrl-stage-b-v2.json), [critic diagnostic](07-dsrl-critic-diagnostic.json), [reserve handover](07-dsrl-reserve-handover.json) |
| 8 | [Result](08-screen-results.md), [confirmation](08-screen-confirmation-results.json), [final audit](08-screen-audit.json), [protocol](https://github.com/Sachin2911/Safety-Dial/blob/1291199e665ed574050de42a7708c19965ecda35/progress/08-knob-screen.md), [frozen source and schedule](08-screen-plan.json), [run](08-screen-run.md), [completion](08-screen-completion/README.md) |
| 9 | [Result](09-unisafe-results.md), [run](09-unisafe-run.md), [C freeze](09-unisafe-C.json), [validation](09-unisafe-C-combined-validation.json), [analysis](09-unisafe-C-combined-analysis.json), [post hoc contrasts](09-unisafe-C-descriptive-contrasts.json), [accounting](09-unisafe-final-accounting.json), [runtime](09-unisafe-runtime.json), [fallback decision](09-unisafe-parallel-C-admission-failure.json), [remaining admission](09-unisafe-C-remaining-admission.json), [final preservation](09-unisafe-final-preservation.json), [completion](09-unisafe-completion/README.md); historical handover |
| 11, stopped after stage A | [Result](11-dsrl-results.md), [run](11-dsrl-run.md), [launch readiness](11-dsrl-launch-readiness.md), [stage-A build](11-dsrl-stage-a-build.md), [frozen source](11-dsrl-stage-a.json), [setup manifest](11-dsrl-setup.json), [completion](11-dsrl-completion/README.md) |

For task 5, `05-adapter-plan.json`, `05-adapter-plan-v2.json` and
`05-adapter-plan-v3.json` precede the executed v4 plan. Read the three amendment
receipts for the changes. Preserve those versions and their hashes; do not rename,
rewrite or execute them. The failed-provisioning receipts account for the first
task 5 rental separately from its successful replacement.
