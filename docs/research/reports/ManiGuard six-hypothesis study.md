# ManiGuard six-hypothesis study

**Status: correction trials in progress.** All six hypotheses in [fruits.md](../../../fruits.md)
have executable adapters and completed physical mechanism screens. The initial screens
and four follow-up batches contain 184 validated full episodes. Twelve additional Jar
trials are testing a diagnosed guard prediction defect; eight have completed, including
all three original/corrected plain-guard pairs. This report remains provisional
until those trials and the final accounting audit finish.

The strongest lead so far is prerequisite guarding on Jar. Its repeat batch has four safe
completions in twelve trials versus two for the unchanged policy, with two versus four
violating episodes. However, the original promising scene did not reproduce its success,
some gains occur before the target is grasped, and an observed guard failure exposed a
joint-limit prediction error. These observations justify checking the correction; they
do not yet demonstrate a reliable intervention mechanism or an evolutionary advantage.

## What was implemented and tested

All adapters operate around frozen family-specific VLA checkpoints. The shared
[controller implementation](../../../experiments/helpers/maniguard_adaptation.py) receives
recorded robot state and action history. Guard variants additionally use explicitly
labelled privileged object predicates and target engagement. The upstream temporal
monitor measures physical violations; it does not itself prevent actions.

| Hypothesis | Implemented comparison | Physical evidence and interpretation |
| --- | --- | --- |
| Adaptive replanning | Fixed chunk horizons 4, 8 and 16, plus tracking/gripper-triggered early replanning | No consistent safe-completion gain. A longer chunk reduces queries on one successful Jar case, but other cases fail or remain unfinished. The adaptive Jar rule makes 361 early replans without aggregate benefit. |
| Motion moderation | Joint-target displacement caps and exponential smoothing | Stack repeats give each of two caps one safe completion in six trials versus zero for identity, but each also has three violations versus two. Jar smoothing gives 3/12 safe completions and five violations versus identity's 2/12 and four. The tested settings introduce a completion/safety tradeoff. |
| Prerequisite guarding | Block predicted upward movement before jar closure or lid placement; retain original gripper commands | Jar repeats are the strongest descriptive lead. Lid gives zero safe completions and one violating episode under each condition, with more queries under guarding. Controller-limit correction trials remain outstanding. |
| Grasp persistence | Bounded release delay using finger aperture and recent closing commands; broader trigger follow-up | All 179 delayed openings in the Stack follow-up occur while the goal target is unheld, with no safe completion. The rule activates but does not demonstrate useful target-grasp preservation. |
| Bounded retry | Detect limited end-effector progress under persistent command demand, then execute a bounded physical retreat | Five retreats across three Jar trials move 14.56–14.90 mm against a 15 mm request, but none yields a safe completion. One begins after a violation. The detector can also become eligible during successful behaviour. |
| Return after intervention | Four settling holds and a fresh query after guard release, compared against the same plain guard | Across twelve repeat pairs, settling gains one safe completion and loses two, with equal violation counts. It uses 10,452 queries versus the plain guard's 2,746. The tested return rule is not a useful improvement over immediate return. |

These are tests of particular mechanisms and parameter settings. Negative screens do
not establish that the entire hypothesis space is impossible. Likewise, a rule that
changes actions without improving task outcomes is not a demonstrated repair.

## Experimental coverage

| Stage | Full episodes | Detailed evidence |
| --- | ---: | --- |
| Original policy baselines, initial Jar/Stack screens and additional Lid baselines | 84 | [Initial screens](ManiGuard%20initial%20adapter%20screens.md) |
| Stack motion repeats | 24 | [Motion report](ManiGuard%20Stack%20motion%20repeats.md) |
| Stack release-trigger comparison | 12 | [Release report](ManiGuard%20Stack%20release%20trigger%20comparison.md) |
| Jar guard, smoothing and settling repeats | 48 | [Jar repeat report](ManiGuard%20Jar%20guard%20and%20motion%20repeats.md) |
| Lid prerequisite-guard comparison | 16 | [Lid report](ManiGuard%20Lid%20prerequisite%20guard%20comparison.md) |
| Jar requested-target versus controller-limit-aware guards | 8/12 complete at this snapshot | [Declared correction plan](../../../configs/maniguard/jar-controller-limit-screen.json) |

The 184 completed episodes in the first five rows consume 446,508 policy actions,
2,694,872 physics ticks and 69,823 policy queries. These are stable batch-subset totals,
not final whole-study costs: short references, inference diagnostics, failed setup work
and the correction trials must also be counted. Batch reports distinguish wall time
from summed episode durations. Query latency includes any time waiting behind another
policy client.

All physical comparisons retain original task horizons: 2000 actions for Jar, 1900 for
Lid and 4200 for Stack. Actions run at 20 Hz and physics at 120 Hz. Jar uses assisted
grasping; Lid and Stack use sticky grasping. Each episode saves source/configuration
hashes, proposals, adapted commands, environment input actions, observations, monitor
results and costs. The environment input labelled `executed` is not a direct record of
the controller's internally clipped joint target.

Scene partitions were frozen before inspecting development outcomes, with variants of
a task kept together. The [requirements audit](../research_notes/ManiGuard%20fruits%20requirements%20audit.json)
revalidated all 184 completed episodes and found that all 198 recorded jobs at its snapshot,
including references, failed startup and active correction trials, used engineering scenes.
Training, selection-validation and reserved scenes have not been used. This does not
establish exclusion from the released VLA's original training data.

## Two concrete guard limitations

The Jar audit identifies an action-interface mismatch. In a hand-back trial the guard
predicts a 5.18 mm downward movement from a requested joint target below the robot's
lower limit. Reconstructing the controller's joint-limit clipping instead predicts a
14.60 mm upward movement; measured end-effector rise is 13.67 mm, and the open jar loses
support. The [boundary evidence](../research_notes/ManiGuard%20fruits%20Jar%20guard%20lift%20prediction%20mismatch.json)
retains the source, asset and trace records. The reconstruction is not a logged actuator
command.

The correction uses live robot controller bounds when estimating proposed lift. It leaves
unblocked policy actions on the original execution path, retains both prediction values,
and fails explicitly if required bounds are absent. It passes 90 focused tests and exact
legacy replay of 41,913 actions with the new option disabled. A fresh 32-action physical
reference verifies live limits and prediction reconstruction, followed by six successful
strict replay queries. These establish implementation behaviour, not task improvement.

The twelve comparison trials rerun original and corrected plain/hand-back guards on the
diagnosed Jar scene at three roots. Within each pair only the prediction mode changes.
The 1 mm trigger remains fixed. Original controls are rerun because identical scene/root
settings do not reproduce identical physical starts. The test cannot establish broad
improvement over the unchanged policy, which is not a condition in this correction batch.

The Lid failure has a different cause. The guard allows a predicted 0.676 mm rise below
its 1 mm threshold; measured rise is 0.670 mm and the held container loses support before
lid placement. Joint-limit clipping leaves that prediction unchanged. This
[tolerance boundary](../research_notes/ManiGuard%20fruits%20Lid%20guard%20tolerance%20boundary.json)
is not claimed to be repaired by the Jar correction.

## Benchmark completion and prerequisite attainment

The [completion-semantics audit](../research_notes/ManiGuard%20fruits%20completion%20semantics.json)
examines all 186 full episodes completed at its snapshot. Two of the 28 Jar episodes
labelled safe completion never record jar closure: the initial plain guard on task_0008,
root zero, and the original hand-back control in the correction batch on task_0008,
root two. Both finish with the target held and intersecting the goal sphere, support
still true, and a monitor state that is neither accepting nor doomed.

The evaluator's spatial goal and its absence-of-detected-violation label are separate
checks. These two episodes do not demonstrate the requested close-then-carry sequence
or satisfaction of the strong-until prerequisite on a finished trace. All published
counts here retain the benchmark labels; this audit supplements them with prerequisite
attainment rather than silently changing the safety definition. The twelve-trial Jar
repeat guard condition's four safe completions all did record closure.

## Correction results so far

All three plain-guard pairs are complete. Neither version has a safe completion.
Original prediction yields one unsafe completion, one safe unfinished and one unsafe
unfinished episode; corrected prediction yields one unsafe completion and two safe
unfinished episodes. Corrected prediction therefore has one fewer violating episode,
but its two safe unfinished runs never close or hold the jar. Queries are 705 corrected
versus 641 original. This is no demonstrated completion benefit.

Across all three corrected plain-guard traces, joint-limit clipping changes only two
eligible blocking decisions: it adds one at root one and removes one at root two, both
while the target is unheld. Root zero has no outside-limit target and no prediction
change. Same-case original and corrected runs have different initial images and first
policy proposals, despite identical recorded initial joint vectors in the first two pairs.
The condition counts remain descriptive physical comparisons.

The corrected plain guard at root one remains safe unfinished after 2000 actions and
280 queries, without closure or grasp. Of 1185 requested targets outside a joint limit,
five change the lift-threshold comparison; only one changes a blocking decision while
the guard is eligible. At proposal 1165 the requested prediction is -3.496 mm and the
limit-aware prediction is +1.092 mm, causing an added block while the goal target is
unheld. All 47 changes in this episode occur before a target grasp, which never happens.
This confirms a decision effect without demonstrating a useful task repair.

The original hand-back control at root two receives the safe-completion label after
1223 actions and 470 queries. It first holds at observation 1214, never closes or leaves
support, and satisfies held-intersection for ten observations. This is one of the
completion-semantics cases above. The hand-back counterpart remains pending. The original plain guard at root one now
also finishes safe unfinished: 2000 actions and 268 queries, versus the corrected guard's
280 queries. It closes at observation 319 but never grasps; the corrected run never
closes. That closure contrast precedes the corrected run's sole decision changed by
clipping at proposal 1165, so it is not evidence of closure being obstructed by that
intervention. The first completed pair has no completion or violation-count difference.

A third trial, corrected plain guard at root zero, completes unsafely after 692 actions.
It first holds at observation 683 and loses support at 684 without closure. At proposal
683 both predictions are +0.830 mm, below the unchanged 1 mm trigger; measured rise is
+1.264 mm. No requested arm target exceeds a joint limit anywhere in this 692-action trial, so
the correction changes no prediction or blocking decision on its recorded trajectory. Subsequent blocking cannot erase the violation. The
[recorded boundary](../research_notes/ManiGuard%20fruits%20corrected%20guard%20tolerance%20boundary.json)
shows residual trigger tolerance and prediction error after correcting the separate
joint-limit mismatch. It does not demonstrate a useful physical repair.

## What this means for evolution

The implemented search runner supports evolution, uniform random search and categorical
policy-gradient learning over the same adapter family, with matched initial populations,
physical candidate budgets and information access. Ordinary learning here would learn
adapter parameters; it would not fine-tune the VLA. That distinction matters when comparing
methods.

No ManiGuard optimizer has consumed training episodes, and no selected controller has
been evaluated on reserved scenes. The gate in `fruits.md` is a repeatable failure with
a useful simple correction before optimizer comparisons. Replanning, release delay,
retries and the tested settling rule have not met that gate. Motion moderation has mixed
outcomes. Guarding remains under assessment through the declared correction study. An
optimizer advantage cannot be inferred from working search code or isolated successes.

## Reproduction and remaining work

[Setup and command documentation](../../../configs/maniguard/README.md) covers pinned
sources, separate simulator/policy environments, full episodes, managed batches, replay,
plots and cost inventory. Use new output paths to preserve prior evidence. The active
correction plan requires the Jar policy with independent sampling streams for two workers.
The batch-report and accounting tools validate local raw records; the tracked report
artifacts preserve the numerical findings on the research branch.

Remaining work is to complete and audit the twelve correction trials, interpret their
mechanism and outcome evidence, refresh whole-study accounting, reassess the optimizer
gate, and push the final synthesis. Raw local results are not yet mirrored to Hugging Face;
the separate credential/destination approval remains pending. No new VLA checkpoint has
been trained in this study.
