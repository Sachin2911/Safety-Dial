# Review of gates and environment choice, 4 October 2026

Recommendation: retain Walker2d while separating the strict controller target
from the question of whether the selection-amplification study is experimentally
informative. Do not silently lower a threshold or relabel a historical failure.
No gate, environment, model checkpoint or root bank was changed by this review.

## Evidence available now

The rollout infrastructure passes 643 tests. The latest clean-target controller
and validation-selected ensemble each fail 4/24 development segments. Full temporal
context failed 5/24. The development roots have been repeatedly inspected, and
these counts are not fresh confirmation. The latest ensemble has 3/64 failures
on the validation roots used to choose it. This is selection performance, not an
independent estimate of generalisation. None has a new 256-root Gate 0 verdict.

Exact source teachers with feedback every simulator step had 0/24 failures. Holding
a teacher action for ten steps failed 22/24. Predicting ten distinct actions is
better than holding one, but is still a demanding interface: the visual controller
gets another image only after all ten actions execute. Clean teacher relabelling
and broader temporal context did not establish a repair. This suggests inspecting
the observation/control interface as well as the imitation fit. It is not an
impossibility result about visual control or Walker2d.

## What Gate 0 currently asks

The committed stage2.yaml requires at most max(2 times recorded failures, 3)
violations in 256 short segments, plus mean progress of at least half the recorded
reference. Recorded reference segments have zero failures partly by construction:
roots need a full surviving 100-step continuation in a recording environment that
terminates on the health rule. Consequently the count floor controls this gate,
not an informative nonzero reference failure rate. Three of 256 is a 1.17% observed
failure rate, a demanding near-zero target over the conditional root distribution.

The original comment invokes the approximate 3/n upper bound for zero events.
That statistical bound does not itself justify allowing three observed failures.
Moreover, the 256 roots share 64 source episodes, so they are not 256 independent
trajectory samples. Uncertainty needs source-episode clustering; a count threshold
and a confidence claim must remain distinct. These limitations do not invalidate
the recorded Gate 0 failures, nor prove that current controllers would pass a
better justified gate.

The old held-action and linear full-block runs inspected the 256 test-role roots.
A future confirmation should freeze its candidate and protocol first and use a
new whole-episode bank if an untouched-test claim is needed. Resampling sibling
roots from the same inspected episodes does not create independent new evidence.

## A prospective revision worth considering

Keep the existing gate as the strict visual-controller competence target. Add a
separately named research-readiness protocol only if the research objective is to
measure how search changes an already imperfect controller's imagined-real gap.
A positive baseline violation rate does not mathematically prevent that comparison,
provided its baseline is measured, progress remains useful, and the experiment can
resolve the increment attributable to selection. Passing such a protocol would
not make the controller safe or pass the original Gate 0.

Before adopting it:

1. Freeze the intended estimand: paired change in dense real-minus-imagined health
   violations under increasing selection pressure, alongside forward progress.
2. Specify the smallest scientifically useful effect and the allowable progress
   loss independently of whichever setting happens to pass. Use development-only
   source-episode resampling to plan uncertainty and run sizes.
3. Declare candidate sampling, budgets, baseline, noise calibration, episode roles
   and the rule for continuing before collecting new confirmatory outcomes.
4. Evaluate on new whole episodes, retain the original failed gates, and report
   baseline unsafety explicitly. A failed or absent amplification effect is a valid
   negative finding, not a reason to change the endpoint after looking.

Gate 1's proposed correlation of at least 0.5 should also be reviewed as a routing
decision. Safety predictions that are all equal have undefined rank correlation,
not evidence of good transfer. Poor transfer may be central to the model-error
question while also undermining the proposed safety fix. Those two implications
need separate interpretation. Gate 2 should distinguish evidence supporting the
amplification claim from successful completion of an honest experiment; no positive
result is guaranteed. The current stage order remains in force until a prospective
revision is explicitly adopted.

## Environment choices

| Choice | Benefit | Cost or limitation | Recommendation |
|---|---|---|---|
| Walker2d with existing assets | Frozen world model, probes, teachers, data and exact snapshot harness already validated | Difficult image-to-action-block control; current controllers still fail | Retain for now, with a bounded repair budget and explicit next decision |
| Hopper | Three action dimensions and an 11-dimensional state observation in standard Gymnasium | Still an unstable contact/balance task; different health limits; no prepared Hopper model, probes or banks in the current asset inventory | Do not assume fewer joints solves the observed problem |
| A small balancing task such as CartPole | Simpler state and action spaces make a diagnostic pipeline easier to inspect | Standard CartPole has discrete actions; new visual model, safety/task definition, data and harness required | Useful as a separate mechanism demonstration if a new track is chosen, not a substitute for Walker success |
| Existing direct-simulator fallback | The plan already allows state-based real-simulator evolution | It answers a different question and cannot establish world-model exploitation | Keep explicitly labelled as the fallback, not a visual-controller repair |

Hopper's dimensions and healthy-state termination are documented by
[Gymnasium](https://gymnasium.farama.org/environments/mujoco/hopper/).
CartPole's discrete action interface and termination conditions are documented by
[Gymnasium](https://gymnasium.farama.org/environments/classic_control/cart_pole/).
These are standard environments; the project's vendored SafetyWalker implementation
and local configuration control its actual runs. The repository's existing plan
already places Hopper after the Walker study and explicitly requires new policies,
data, LeWM and probes. No prepared Hopper, CartPole or Pendulum asset was found in
the current data/hf and configs inventory.

## Decision after this review

The clean-target and ensemble tests are complete and still do not establish the
requested reliable starting controller. The latest two runs used 560,660 new
simulator steps in total. Do not continue architecture and seed searches indefinitely
on the same 24 development roots. The next useful decision is between a bounded
control-interface repair under the existing scope and a declared research-readiness
revision or separate simpler-environment track. That choice should preserve the
original scientific question and distinguish a working research harness from a
reliable visual controller. No goal-completion claim is supported yet.

## User decision and prospective baseline protocol

After this review, the user accepted the recommendation to retain Walker2d,
preserve the strict Gate 0, and introduce a separate research-readiness assessment.
The [new baseline protocol](../../../configs/evo/stage2_readiness_baseline.yaml)
freezes the selected ensemble before any fresh episode generation. It requests
256 roots from 256 new episodes, drawn from the original competent source-actor
pool and action-noise mixture. The source-actor competence filter is frozen from
the historical S1 ladder, not refitted after seeing the new outcomes. Rejected
short episodes and all their simulator steps remain recorded and charged. Root
sampling preserves the original representative bank's healthy-future conditioning,
which remains an explicit limit on interpretation rather than an unreported change
of task distribution.

The audit retains useful progress by requiring the lower paired-bootstrap bound
on the ratio of mean progress to reach 0.5. For measurement feasibility, it requires
the upper bound on baseline failure probability below 0.90, leaving room for a
predeclared 10 percentage-point increase, and a baseline gap interval no wider than
20 percentage points. These are coarse criteria for permitting a bounded transfer
and power pilot, not safety limits or a demonstration of power for the final paired
search effect. The 10-point effect is a prospective planning choice, not a detected
result. Power for that effect must include both model/real differences and search
seed variability before a confirmatory study is sized.

This bank becomes development evidence after the audit. It cannot then be called
an untouched final search test. A pass will not certify safe control, pass the old
Gate 0, or authorize the full fix grid. A failure will not trigger automatic
threshold changes. The original controller goal remains incomplete.
