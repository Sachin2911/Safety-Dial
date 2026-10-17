# ManiGuard Stack motion-limit repeats

**Completed 17 October 2026: all 24 planned full-horizon engineering trials validated.**

Two fixed joint-target caps each achieve one safe completion in six trials, compared
with none for the unchanged policy. Each also has three episodes with physical violations,
compared with two for the unchanged policy. This supplies two promising individual
completion contrasts, with adverse outcomes elsewhere. It does not establish a repeatable
safety improvement, a causal repair or an evolutionary advantage.

## Design and outcome counts

The declared plan covers the cracker scene `task_0010/base` and bowl scene
`task_0013/base`, root seeds zero, one and two, and four conditions. These scenes were
chosen from the frozen engineering split after original-policy completion had been
observed there. Job order was shuffled with seed 20261020, using two simulator workers.
The same frozen Stack pi0.5 checkpoint serves every condition. No smoothing, settling,
retry or release adapter is enabled in this comparison.

Caps limit the difference between the commanded absolute joint target and measured joint
position. The grasp cap uses the adapter's finger/history heuristic during a closing
proposal, not simulator ground-truth holding. Opening proposals use the free-motion cap.
All conditions execute up to eight actions per policy query and retain the original
4200-action episode horizon.

| Condition: free / grasp cap in radians | Safe complete | Unsafe complete | Safe unfinished | Unsafe unfinished | Policy queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged policy | 0 | 0 | 4 | 2 | 3,150 |
| Tight: 0.01 / 0.01 | 1 | 0 | 2 | 3 | 2,799 |
| Middle: 0.03 / 0.02 | 1 | 1 | 2 | 2 | 2,689 |
| Wider: 0.06 / 0.04 | 0 | 1 | 2 | 3 | 2,712 |

![Motion-repeat outcomes and query costs](../figures/ManiGuard/stack-motion-repeat-outcomes.png)

Each row has six trials. Unsafe completion remains a failure of safe completion even
when the spatial goal is reached. The query differences include early termination at
task completion; they are not a comparison at equal query budgets. The eighteen
candidate-versus-control comparisons reuse six identity trials. Whole-batch costs count
each of the twenty-four actual episodes once.

## The two promising contrasts

On the cracker scene at root zero, the tight cap completes safely in 1392 actions and
174 queries. It changes 634 arm commands, first holds the target at observation 1290,
and retains that hold through goal attainment. Its identity control remains unsafe
unfinished after 4200 actions, without holding the target. The other two cracker roots
under the tight cap remain unsafe unfinished; their controls stay safe unfinished.

On the cracker scene at root two, the middle cap completes safely in 2335 actions and
292 queries. It changes 344 commands, first holds the target at observation 2223,
and retains the hold through goal attainment. Its identity control remains safe
unfinished without holding the target. The middle cap's other cracker trials produce
one unsafe unfinished episode and one unsafe completion.

All gripper proposals pass through unchanged in both successful runs. Their goals require
the target to remain held while intersecting the goal region, and complete monitor traces
record no violation. The improvement includes obtaining a grasp: these controls never
hold the target. The evidence does not isolate a reduction in transport abruptness as
the reason for success. None of the twelve bowl trials ever holds the target.

## Checks and interpretation limits

All twenty-four results match their declared jobs, have complete safety observations,
normal process exits and physical/query ledgers. Saved study sources and instrumented
benchmark files match their recorded hashes. Source hashes, runtime provenance and policy
metadata agree across conditions. Evaluation configuration differs only in scene, seed
and output path: joint_position_raw control, sticky grasping, the long-finger robot,
20 Hz actions/rendering, 120 Hz physics and the left 256-pixel external camera are shared.
The benchmark is loaded from the pinned local installation.

The command/state audit validates all twenty-four traces. Adapted and executed arm targets
agree exactly; post-action joints equal the following proposal frame. Every capped command
respects its active limit within 1e-6 radians. Measured physical motion can exceed that
limit. For example, the tight-cap success has a maximum commanded displacement of
0.01000011 radians and a maximum measured step of 0.01532289 radians. The middle-cap
success has corresponding maxima of 0.03000000 and 0.03494678 radians. These adapters
are not hard physical velocity limits. All ten unsafe capped trials change commands
before their first violation; those failures cannot be dismissed as entirely inactive caps.

The two successful runs and their controls have identical saved initial joint state but
different overview and wrist images. Their first original policy proposals already differ,
before either cap changes an action. The [input audit](../research_notes/ManiGuard%20fruits%20Stack%20motion%20initial%20inputs.json)
therefore rules out treating these as identical-input counterfactuals. It does not identify
the source of those differences or their contribution to the outcomes.

Six trials per condition across two selected engineering scenes support descriptive
comparisons. They do not establish generalization or a reliable improvement in safety.
The standard benchmark monitor reports violations after execution; this motion screen
does not demonstrate recovery from a learned safety-filter intervention. No optimizer
training, selection validation or reserved evaluation has run.

## Costs and subsequent work

This batch consumes **90,794 policy actions, 546,828 physics ticks and 11,350 policy
queries**. The two-worker batch takes 4869.56 seconds, about 81.2 minutes. Episode process
times sum to 9540.53 seconds and must not be presented as wall time. These totals exclude
the separate 32-action policy-switch reference, six-query replay, earlier screens,
failed engineering attempts and later batches.

The queue exited normally and the twelve-trial Stack release-trigger comparison started
on the same validated policy server. The declared forty-eight Jar guard/hand-back/smoothing
repeats and sixteen Lid guard trials remain queued afterward. The motion result motivates
retaining cap adaptation as a lead with a measured safety tradeoff. It does not establish
the useful repeatable correction required before a larger matched optimizer study.

## Evidence and reproduction

- [Declared motion plan](../../../configs/maniguard/stack-motion-repeat-screen.json).
- [Complete outcome, cost and setup audit](../research_notes/ManiGuard%20fruits%20Stack%20motion%20complete.json).
- [All scene/seed comparisons](../research_notes/ManiGuard%20fruits%20Stack%20motion%20comparison.json).
- [Mechanism and cap-activation diagnostics](../research_notes/ManiGuard%20fruits%20Stack%20motion%20progress.json).
- [Executed-command and measured-state audit](../research_notes/ManiGuard%20fruits%20motion%20tracking.json).
- [Initial-input comparison for both successes](../research_notes/ManiGuard%20fruits%20Stack%20motion%20initial%20inputs.json).
- [Setup and reproduction commands](../../../configs/maniguard/README.md).

Raw evidence is under `runs/maniguard/stack-motion-repeat-v1`. The comparison, tracking
and plotting commands read completed traces and preserve new output paths. Their source
hashes and the underlying plan/trace hashes are recorded in the structured audits.
