# ManiGuard six-hypothesis study

All six hypotheses in [fruits.md](../../../fruits.md) have executable adapters and completed
physical mechanism screens. The declared study contains **196 validated full episodes**.
It finds useful implementation and evaluation problems, but no repeatable simple repair
that currently supports an evolutionary-optimization study. No ManiGuard optimizer was
trained, no reserved scene was evaluated, and no evolutionary advantage is claimed.

Prerequisite guarding on Jar was the strongest descriptive lead: the repeat batch has
four safe completions in twelve trials versus two for the unchanged policy, with two
versus four violating episodes. The original promising scene did not reproduce its
success, however, and the subsequent controller-limit correction study produces no safe
completion under either corrected variant. The detailed findings below distinguish
benchmark outcomes, diagnosed mechanisms and unproven explanations.

## Findings across the six hypotheses

All adapters wrap frozen family-specific VLA checkpoints. The shared
[controller implementation](../../../experiments/helpers/maniguard_adaptation.py) receives
robot state and action history. Guard variants additionally use explicitly labelled
privileged object predicates and target engagement. The upstream temporal monitor
measures violations; it does not itself prevent actions.

| Hypothesis | Implemented and physically tested | Finding |
| --- | --- | --- |
| Adaptive replanning | Fixed chunk horizons 4, 8 and 16, plus tracking/gripper-triggered replanning | No consistent safe-completion gain. A longer chunk halves queries on one successful Jar case, but other cases fail or remain unfinished. The adaptive Jar rule makes 361 early replans without aggregate benefit. |
| Motion moderation | Joint-target displacement caps and smoothing | In Stack repeats, two caps each gain one safe completion in six trials, but each has three violations versus identity's two. Jar smoothing gives 3/12 safe completions and five violations versus identity's 2/12 and four. The tested settings introduce a completion/safety tradeoff. |
| Prerequisite guarding | Block predicted upward movement before closure/lid placement; test original and controller-limit-aware estimates | The Jar repeat lead does not become a demonstrated repeatable repair. Lid has no safe completion under either condition. The final correction study has no safe completion under either corrected guard variant. |
| Grasp persistence | Bounded release delay and a broader trigger follow-up | All 179 delayed openings in the follow-up occur while the goal target is unheld, with no safe completion. Activation does not establish useful target-grasp preservation. |
| Bounded retry | Observable stall detection followed by a bounded physical retreat | Five retreats across three Jar trials move 14.56–14.90 mm against a 15 mm request, but none produces safe completion. One starts after a violation, and the detector can also become eligible during successful behaviour. |
| Return after intervention | Four settling holds and fresh querying, compared against the same plain guard | Twelve repeat pairs contain one safe-completion gain and two losses, with equal violations. Settling uses 10,452 queries versus the plain guard's 2,746. The tested return rule is not a useful improvement. |

These are bounded tests of particular mechanisms and settings. Negative screens do not
prove that the broader hypotheses cannot work. Likewise, changing commands, reducing
query counts or remaining safe through inactivity does not demonstrate task recovery.

## Completed experimental coverage

| Stage | Full episodes | Detailed report |
| --- | ---: | --- |
| Original baselines, initial Jar/Stack screens and additional Lid baselines | 84 | [Initial screens](ManiGuard%20initial%20adapter%20screens.md) |
| Stack motion repeats | 24 | [Motion report](ManiGuard%20Stack%20motion%20repeats.md) |
| Stack release-trigger comparison | 12 | [Release report](ManiGuard%20Stack%20release%20trigger%20comparison.md) |
| Jar guard, smoothing and settling repeats | 48 | [Jar repeat report](ManiGuard%20Jar%20guard%20and%20motion%20repeats.md) |
| Lid prerequisite-guard comparison | 16 | [Lid report](ManiGuard%20Lid%20prerequisite%20guard%20comparison.md) |
| Jar controller-limit correction comparison | 12 | [Correction report](ManiGuard%20controller-limit%20guard%20comparison.md) |

The final correction comparison is complete:

| Condition, three trials each | Safe complete | Unsafe complete | Safe unfinished | Unsafe unfinished | Queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original plain guard | 0 | 1 | 1 | 1 | 641 |
| Corrected plain guard | 0 | 1 | 2 | 0 | 705 |
| Original hand-back | 1 | 0 | 1 | 1 | 3,183 |
| Corrected hand-back | 0 | 1 | 2 | 0 | 2,604 |

Plain-guard correction has one fewer violating episode without a safe-completion gain.
Hand-back correction has one safe-completion loss and equal violations. The sole safe
completion in this table does not include jar closure, as explained below. This
single-scene correction study has no unguarded identity condition and does not estimate
broad improvement over the original VLA.

## Concrete implementation and evaluation findings

Our prototype guard initially estimates lift from a requested joint target without
accounting for downstream joint-limit clipping. In one recorded failure it predicts
-5.18 mm, whereas reconstructing the feasible target predicts +14.60 mm and measured
end-effector rise is +13.67 mm. The open, held jar loses support. The
[boundary audit](../research_notes/ManiGuard%20fruits%20Jar%20guard%20lift%20prediction%20mismatch.json)
preserves the controller/asset evidence; the feasible target is an offline reconstruction,
not a logged actuator command.

The opt-in correction uses live controller limits and retains both estimates. It passes
90 focused software tests and exact legacy replay of 41,913 actions with the new option
disabled. A fresh 32-action reference validates live limits and prediction reconstruction,
and six strict interleaved replay queries pass. Across the six corrected full trials,
clipping changes fourteen lift-threshold comparisons but only two eligible blocking
decisions, both while the target is unheld. It changes no eligible blocking decision in
the corrected hand-back trials. The implementation correction is verified; useful
physical recovery from it is not demonstrated.

Remaining failures include small permitted rises below the 1 mm trigger, tipping outside
the guard's prerequisite scope, and loss of support despite downward end-effector motion.
End-effector height is not a complete predictor of object support/contact state. The
[correction report](ManiGuard%20controller-limit%20guard%20comparison.md) retains all five
violation boundaries. This study keeps its declared trigger and physical safety definitions
fixed rather than tuning them after seeing the outcomes.

The [final completion-semantics audit](../research_notes/ManiGuard%20fruits%20completion%20semantics%20final.json)
checks all 196 full episodes. Two of 28 Jar episodes labelled safe completion never
record closure: initial plain guard task_0008/root zero and correction-batch original
hand-back task_0008/root two. Both satisfy held-intersection with the goal sphere while
remaining supported; their final monitor states are neither accepting nor doomed.
The pinned evaluator's spatial goal and absence-of-detected-violation label are separate
checks. These two runs do not establish the requested close-then-carry sequence or the
strong-until prerequisite on a finished trace. All tables retain the benchmark labels
and supplement them with prerequisite attainment. The Jar repeat plain guard's four
safe completions all do record closure.

## Controls and validation

Every full episode retains the original task horizon: 2000 actions for Jar, 1900 for Lid
and 4200 for Stack. Actions run at 20 Hz and physics at 120 Hz. Jar uses assisted grasping;
Lid and Stack use sticky grasping. Policy weights and physical specification thresholds
remain fixed. The recorded `executed` action is the environment input, not lower-level
controller output telemetry.

The [final execution-integrity audit](../research_notes/ManiGuard%20fruits%20final%20execution%20integrity.json)
verifies all 196 full episodes: 466,491 ordered finite proposal/environment-input action
pairs, complete valid observation and goal records, matching query/intervention counts,
pinned policy metadata and 1,568 saved source files. Adapted commands and environment
inputs match exactly. This verifies the records, not a safety guarantee.

Scene partitions were frozen before development outcomes were inspected, with variants
of a task kept together. All recorded jobs use engineering scenes; training, validation
and reserved partitions remain unused and disjoint. Released VLA pretraining overlap is
unknown. Same-scene/root trials can differ in initial images and policy proposals, so
these small engineering comparisons do not establish population effects or identical-state
counterfactuals. The [requirements audit](../research_notes/ManiGuard%20fruits%20requirements%20audit.json)
maps all six implementations to their physical evidence and checks the declared coverage.

## Complete recorded interaction costs

The 196 full episodes consume **466,491 policy actions, 2,815,802 physics ticks and
76,956 policy queries**. The final raw inventory also includes ten short physical
references, two failed setup attempts and twelve inference diagnostics:

| Recorded total across all retained work | Count |
| --- | ---: |
| Policy actions | 466,811 |
| Physics ticks | 2,818,668 |
| Policy queries | 77,065 |
| Environment steps, including initialization | 468,881 |

Recorded query latency sums to 20,972.44 seconds. Episode/diagnostic elapsed durations
sum to 59,638.68 seconds; this is not study wall time because work overlaps. Individual
reports retain batch wall times. The
[final accounting audit](../research_notes/ManiGuard%20fruits%20final%20accounting.json)
counts failed work and excludes duplicated derived reports. It preserves two historical
gaps: a native startup failure before physics accounting attached and a counted failed
query whose latency was not recorded. These are recorded totals, not a claim that every
unmeasured startup cost is known.

## Optimizer gate and deliverables

Evolution, uniform random search and categorical policy-gradient learning are implemented
over the same adapter interface, with matched initial populations, candidate budgets and
information access. Ordinary learning here would learn adapter parameters, not fine-tune
the VLA. No optimizer consumed ManiGuard training episodes.

The gate in `fruits.md` is a repeatable failure with a useful simple correction before
optimizer comparisons. The tested replanning, release, retry and settling rules do not
meet it; motion moderation has mixed outcomes; the strongest guard lead does not yield
safe completion in the correction study. We therefore finish this bounded mechanism
study without selecting or optimizing a controller. The broader evolutionary research
question remains open. Working search code is not evidence of an evolutionary advantage.

Code, configs, reports, figures and structured findings are maintained on the authorized
research branch. [Reproduction documentation](../../../configs/maniguard/README.md)
covers pinned assets, separate simulator/policy environments, full episodes, managed
batches, replay and auditing. The [296.5 MB final numerical-results archive](../research_notes/ManiGuard%20fruits%20final%20backup.json)
is verified locally, with all 2088 result files read back and hash-checked. Optional
off-instance raw-data storage remains separate; Hugging Face upload still requires the earlier
pending token/destination approval. No new VLA checkpoint was trained.

The [preserved audit programs](../research_notes/ManiGuard%20audit%20sources/README.md)
make the analysis-source hashes in the structured evidence independently inspectable.
