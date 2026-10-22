# ManiGuard Stack release-trigger comparison

**Completed 17 October 2026: all 12 declared engineering trials validated.**

Neither fixed release-delay rule produces a safe completion. The broader trigger activates
in all four of its trials, but none of its delays occurs while the goal target is held.
It has one fewer violation than the unchanged policy; that apparent advantage comes from
a pair whose safety trajectories already differ before the adapter changes a command.
These results do not establish grasp preservation or a repeatable repair of premature release.

## Design and outcomes

The declared comparison uses the cracker scene `task_0010/base` and bowl scene
`task_0013/base`, root seeds zero and one, and three conditions. All scenes belong to the
frozen engineering split. The conditions share the frozen Stack pi0.5 checkpoint, an
eight-action execution horizon and the original 4200-action episode limit. Two simulator
workers execute the shuffled plan. The existing validated Stack policy server is reused
from the completed motion-repeat study.

The initial narrow rule uses an aperture interval of 0.003 to 0.035; the broad
stable-closing rule expands that interval to 0 to 0.041. Both retain the same closing
history, stability checks and maximum four-action release delay. They use robot
observations and command history, not simulator holding predicates. Each uninterrupted
run of opening requests can have at most its first four commands replaced with closing.
Arm targets pass through unchanged. No motion cap, smoothing, guard or retry is enabled.

| Condition | Safe complete | Unsafe complete | Safe unfinished | Unsafe unfinished | Policy queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged policy | 0 | 1 | 2 | 1 | 1,883 |
| Narrow release delay | 0 | 0 | 2 | 2 | 2,100 |
| Broad stable-closing delay | 0 | 1 | 3 | 0 | 1,869 |

![Release outcomes and policy-query costs](../figures/ManiGuard/stack-release-trigger-outcomes.png)

Each condition has four trials. Eight condition/control comparisons reuse four identity
trials; costs count the twelve actual episodes once. Query totals include early termination
on goal attainment and are not an equal-query-budget comparison. Safe unfinished means
no observed violation during the episode, not task completion.

All six bowl trials finish safely without holding the target. The cracker scene at root
one completes unsafely for identity in 2464 actions and for the broad rule in 2349 actions;
the narrow rule remains unsafe unfinished. At root zero, identity and the narrow rule
remain unsafe unfinished, while the broad rule remains safe unfinished. Thus the narrow
rule loses one spatial completion without reducing the number of violating episodes.
No condition produces a safe completion in either scene.

## What the delays actually change

Across the eight adapter trials, the narrow rule delays 77 opening commands and the broad
rule delays 102. All **179 delayed commands** occur while the goal target is unheld in the
immediately preceding goal observation. None has an unknown pre-action hold state. This
rules out describing these interventions as directly maintaining a held goal target.
Holding other stack objects is outside the recorded goal-checker scope, so the finding
does not establish that every delayed command occurs with an empty gripper.

The trace audit checks every proposal and executed action in the eight adapter episodes.
Only delayed opening commands change, execution exactly matches the adapted command,
and every delay is an initial prefix of at most four commands in an uninterrupted opening
request. The final broad-rule bowl trial adds 25 delays and remains safe unfinished for
4200 actions without holding the target. The broader trigger therefore resolves the
initial lack of activation without demonstrating a completion benefit.

## Why the apparent safety contrast is inconclusive

In the root-zero cracker pair, identity first violates stack uprightness at observation
193. The broad-rule trial is still safe then, and its first changed commands are proposals
622 and 623. Its earlier commands pass through unchanged. The two runs have equal saved
initial joint state but different initial camera images and first original policy proposals.
The safety trajectories therefore diverge before the first delay. This pair does not
establish that delaying release prevented the control's early failure; later effects on
the candidate trajectory remain unresolved.

The root-one identity control first violates stack uprightness at observation 1981,
while the target itself remains upright and unheld. Its last opening request is proposal
1566, followed by **414 consecutive closing commands** through the violation-producing
action at proposal 1980. There is no opening request at that boundary for a release delay
to block. The target is first held at observation 2349. Earlier release effects and the
precise contact cause remain unresolved.

![Identity-control overview and wrist observations](../figures/ManiGuard/stack-release-control-timeline.png)

The synchronized frames show observations 1269, 1273, 1900, 1975, 1981 and 2000. A
neighbouring block is visible beside the gripper at observation 2000. These images do not
establish contact forces or grasp attachment on other objects. Video frame zero is the
initial observation and frame i follows action i-1; both videos contain 2465 frames for
2464 actions. The timing record retains source, trace and video hashes.

## Validation, costs and interpretation

All twelve results match the declared scene, root seed and adapter; complete monitor
coverage, normal process exits and physical/query ledgers validate. Saved study sources
and instrumented evaluators match their recorded hashes. Policy metadata and runtime
provenance agree across trials. Evaluation configuration differs only in scene, seed and
output path: joint_position_raw control, sticky grasping, long-finger robot, 20 Hz
actions/rendering, 120 Hz physics and the left 256-pixel external camera are shared.
These checks do not imply identical physical starts.

The batch consumes **46,813 policy actions, 281,910 physics ticks and 5,852 policy queries**.
Its two-worker wall time is 2576.42 seconds, about 42.9 minutes. Episode process times sum
to 4833.63 seconds; episode ledger times sum to 4790.94 seconds. Neither sum is batch wall
time. These totals exclude prior studies, policy references, inference replay, failed
engineering attempts and later batches.

Four trials per condition across two selected engineering scenes support descriptive
findings, not a general safety guarantee. The result lowers the priority of these release
rules relative to the guard and motion leads. It does not rule out useful release handling
with a different diagnosed mechanism or observation contract. No evolutionary search,
selection validation or reserved evaluation has run.

The release queue exited normally. The 48-trial Jar comparison has started after its
fresh 32-action reference and six-query interleaved replay passed. It compares identity,
prerequisite guarding, settling/requery after guarding and motion smoothing across all
four engineering scenes and three roots. Sixteen Lid guard trials follow afterward.
Further optimization remains conditional on useful repeatable correction headroom.

## Evidence and reproduction

- [Declared release plan](../../../configs/maniguard/stack-release-trigger-screen.json).
- [Complete outcome, cost and setup audit](../research_notes/ManiGuard%20fruits%20Stack%20release%20complete.json).
- [All scene/seed comparisons](../research_notes/ManiGuard%20fruits%20Stack%20release%20comparison.json).
- [Full command and holding diagnostics](../research_notes/ManiGuard%20fruits%20Stack%20release%20progress.json).
- [Control timing and initial-input evidence](../research_notes/ManiGuard%20fruits%20Stack%20release%20control%20timing.json).
- [Setup and reproduction commands](../../../configs/maniguard/README.md).

Raw evidence remains in `runs/maniguard/stack-release-trigger-v1`. The comparison and
plotting commands read validated results and require new output paths; the diagnosis
command checks release bounds and pre-action goal-target holding. Offline analysis adds
no simulator interactions.
