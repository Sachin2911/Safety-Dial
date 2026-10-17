# Low-hanging fruit for evolutionary adaptation in ManiGuard

**Recorded 17 October 2026.** This is a growing list of accessible research candidates
alongside [ideation.md](ideation.md). Sachin endorsed adaptive replanning as the first
candidate to record, followed by five further candidates from the same discussion.
These are possibilities to investigate, not commitments to an architecture, optimizer or
experiment budget. No general ManiGuard improvement has been demonstrated locally.

Implementation and experiment records are maintained in the
[ManiGuard study log](docs/research/research_notes/ManiGuard%20fruits%20development.md).
All six initial adapters have been implemented. Twelve original-policy baselines,
24 Stack screen episodes and 40 Jar screen episodes are complete and validated.
The [initial-screen report](docs/research/reports/ManiGuard%20initial%20adapter%20screens.md)
records all outcomes, costs and remaining limitations.

Prerequisite guarding and motion smoothing are the strongest early leads: each safely
completed one Jar scene where the contemporaneous unchanged policy transported an open
jar. These are single-trial contrasts. Across four Jar scenes, the guard and unchanged
policy each complete three safely, smoothing completes two, and no overall improvement
is established. The Stack screen also found no safe-completion gain. The release rule
rarely activates, bounded retries execute the intended retreat without a demonstrated
recovery, and the tested settling/requery rule increases cost without improving completion.

Eight extra Lid baselines are also complete: one unsafe completion, two unsafe unfinished
and five safe unfinished. Three scenes now show lifting before lid placement; three other
trials record lid placement but never hold the target container. This supplies clearer
evidence of premature transport to test, not a demonstrated correction. A 16-episode Lid comparison
of identity and the fixed guard is queued across all four engineering scenes and two seeds.
See the [Lid follow-up evidence](docs/research/research_notes/ManiGuard%20fruits%20Lid%20followup.json).

The 24 Stack motion-limit repeats are running. Follow-up queues also contain 12 Stack
release-trigger comparisons and 48 Jar repeats comparing identity, guard, hand-back
and smoothing across all four engineering scenes and three root seeds.
These remain engineering experiments. No evolutionary search or reserved evaluation has
run, and no evolutionary advantage has been demonstrated.

## Candidate overview

Each row is a hypothesis to screen, not an established failure mechanism or improvement.
The detailed sections below describe what evolution could tune and the evidence needed.

| Candidate | Potential headroom | First simple comparison |
| --- | --- | --- |
| Adaptive replanning | Continuing stale actions after contact or a state change | Original policy vs fixed execution horizons |
| Motion moderation | Abrupt movements destabilizing otherwise useful manipulation | Original motion vs fixed joint-target limits and smoothing |
| Prerequisite guarding | Lifting or transporting before closure or lid placement | Original policy vs a fixed prerequisite guard, including timeouts |
| Grasp persistence | Opening prematurely and losing a useful grasp | Original release commands vs a short bounded release delay |
| Bounded retries | Repeating an ineffective approach without recovering progress | Continued execution vs one scripted physical retreat and retry |
| Return after intervention | Repeating blocked proposals or switching disruptively | Immediate return vs a settling period and fresh policy query |

## Where to look for headroom

Prioritize cases where the policy already performs much of the task correctly, but an
execution decision makes it unsafe or prevents completion. The paper reports the following
in-distribution outcomes for family-specific fine-tuned checkpoints, pooled across three
policy-sampling seeds:

| Policy and task | Safe completions | Unsafe completions | Total episodes |
| --- | ---: | ---: | ---: |
| pi0.5 on Jar | 21 | 4 | 78 |
| pi0.5 on Stack | 17 | 7 | 84 |
| pi0 on Lid | 2 | 9 | 90 |

Unsafe completions show that goal achievement and safety can come apart. A small correction
might preserve the useful behaviour while avoiding a violation, but these counts do not
establish that any particular failure is locally repairable. Policy and task must be
considered together: pi0.5 on Lid has nine safe completions and one unsafe completion in
90 episodes, so its main difficulty appears different from pi0's. These are published
results, not locally reproduced measurements.
[ManiGuard, Tables 12 and 13](https://arxiv.org/html/2608.17386v1).

## 1. Evolve when a frozen VLA replans

> Can a compact evolved replanning rule improve safe completion of a frozen
> vision-language-action policy?

### Opportunity and hypothesis

ManiGuard's policy interface returns chunks of actions, and its evaluator controls how
many actions execute before querying the policy again through `execute_horizon`.
The paper's evaluation executes eight actions at 20 Hz between predictions, corresponding
to 0.4 seconds of simulated execution. This describes the published protocol; confirm the
selected checkpoint and release configuration before reproducing it.
[Evaluation interface](https://nu-ideas-lab.github.io/ManiGuard/docs/evaluation/),
[paper, Appendix C](https://arxiv.org/html/2608.17386v1).

The hypothesis is that some failures occur because the robot continues an action chunk
after contact or another change has made the remaining predictions inappropriate.
Earlier replanning might repair those failures. This mechanism has not been diagnosed
in our ManiGuard setup, and more frequent replanning need not improve performance.

The entry cost is relatively low once the simulator and pretrained policy work: vary an
existing execution control while keeping the VLA weights fixed. A new world model or
learned safety filter is not required for this screening experiment. ManiGuard's runtime
monitor measures violations; it does not intervene to prevent them.
[Paper](https://arxiv.org/html/2608.17386v1).

### First screen: fixed execution horizons

Start with one pretrained policy and a small development set from Jar, subject to runtime
and baseline validation. Jar is a candidate because closing, lifting and carrying offer
different interaction demands. Whether timing explains any failures remains open.

1. Reproduce the unchanged policy and inspect successful, unsafe and unfinished episodes.
2. Compare a few fixed execution horizons within the policy's supported chunk length.
   Keep the simulator timestep, task horizon, observations and action conventions fixed.
3. Check whether the changes repair a recurring failure, merely spend more policy queries,
   or introduce new failures. Record actual query counts and elapsed time.

If one fixed horizon resolves the problem, retain that behavioural finding and use it as
a strong baseline. A one-parameter sweep alone offers little reason to prefer evolution.
If the policy lacks the underlying skill, such as reliably closing the lid, execution
timing may offer little useful headroom.

### Where evolution could add value

A more interesting result would be that different situations benefit from different
replanning intervals. Evolution could then search a small rule that decides when to
discard the remaining chunk and request a fresh prediction.

Possible inputs include measured joint tracking error, gripper state changes and recent
action history. Their availability and predictive usefulness must be checked first.
An initial representation might contain a normal execution horizon, an early-replanning
threshold and a minimum interval between queries. These are examples, not selected genes.
Keep simulator-only object predicates and safety labels out of runtime inputs unless the
study explicitly uses and labels a matched privileged-information setting.

Evaluate candidate rules through complete simulator episodes across development scenes.
Compare the unchanged policy, the best selected fixed horizon, a simple hand-designed
adaptive rule, random search, evolution and ordinary learning over the same rule family.
Separate any benefit of adaptation from the benefit of additional VLA queries, including
comparisons at comparable query budgets. The evolutionary algorithm remains open.

### Evidence to retain

Report safe completion, physical specification violations and unfinished episodes
together, with engagement and completion time where useful. Preserve the benchmark's
safe/unsafe outcome categories. Increased safe completion alone does not establish an
unchanged violation rate.

Separate development, candidate selection and final evaluation by scene, grouping variants
of the same base scene appropriately. Reserved scenes establish generalization within this
adaptation study, not necessarily novelty relative to the VLA's original training data.
Account for all simulator interactions, including discarded candidates, and report VLA
queries and elapsed time separately. Log predicted chunks, executed prefixes and reasons
for replanning so that improvements can be explained.

Continue toward a larger study if a recurring, avoidable failure responds to replanning
and an adaptive rule has useful headroom over fixed settings. If simple controls match
evolution, preserve that result without claiming an evolutionary advantage. Novelty and
publication suitability require a separate primary-literature review.

## 2. Moderate motion during delicate interactions

> Can a small evolved motion adapter preserve task progress while reducing destabilizing
> movements?

Jar requires upright transport; Stack requires extracting the bottom object without
toppling or dropping the others. The hypothesis is that some useful movements are executed
too abruptly. Excessive speed or abruptness has not been established as the cause of the
reported violations.
[Jar](https://nu-ideas-lab.github.io/ManiGuard/docs/pipelines/jar_transport/),
[Stack](https://nu-ideas-lab.github.io/ManiGuard/docs/pipelines/stack_retrieve/).

Evolution could adjust joint-target step limits, smoothing strength and brief settling
periods, potentially with different settings during approach and carrying. A basic adapter
could use joint state, gripper state and recent commands; any additional contact or object
signals need an explicit observation contract.

**First test:** compare a few fixed motion limits and inspect whether they repair a
recurring failure. Match the checkpoint's intended controller configuration first, since
ManiGuard documents soft joint tracking as a possible setup mismatch. Apply moderation to
joint-target differences; multiplying absolute joint targets by a speed factor changes the
target configuration.
[Controller and action conventions](https://nu-ideas-lab.github.io/ManiGuard/docs/fine_tuning/end_to_end/).

This is a relatively cheap additional screen after replanning. Evolution becomes more
interesting if a single cautious setting is inadequate and different situations need
different settings. Slowing an incorrect path may not help, and motion limits alone do
not guarantee safety.

The first completed repeated Stack comparison on the same scene and root seed leaves
both the unchanged policy and the tight cap safely unfinished, without a target grasp.
A [trace audit](docs/research/research_notes/ManiGuard%20fruits%20motion%20tracking.json)
confirms that the limits reach the executed joint targets. Actual joint motion can still
exceed the commanded target displacement, so this adapter is not a hard velocity limit.
The remaining repeated comparisons are in progress.

## 3. Prevent premature transitions between task stages

> Can a compact evolved controller prevent unsafe stage transitions while still enabling
> the policy to complete the prerequisite and continue?

Jar requires closing before lifting; Lid requires placing the lid before transporting
the container. A policy might execute transport while skipping the prerequisite. A guard
could interrupt premature transport, with evolution adjusting confirmation dwell time,
retry conditions and the return to normal policy execution. Keep the benchmark's safety
specification unchanged.
[Jar specification](https://nu-ideas-lab.github.io/ManiGuard/docs/pipelines/jar_transport/),
[Lid specification](https://nu-ideas-lab.github.io/ManiGuard/docs/pipelines/lid_transport/).

**First test:** use a fixed prerequisite guard and check whether it converts unsafe
completions into safe completions or merely creates timeouts. pi0 on Lid is a candidate
because its published outcomes include more unsafe than safe completions. Inspect which
constraints those episodes violate before selecting the guard.

This is a stronger direct safety-filter candidate, with more integration work than motion
limits. Detecting closure and impending lifting requires suitable signals and action
interpretation. With joint-target actions, a lift check may need kinematics and object
association. Simulator state can support a clearly labelled privileged diagnostic, shared
across comparisons. A later sensor-based controller needs its own validation. A guard
cannot supply a missing lid-closing skill, so useful retries must be demonstrated.

## 4. Preserve a useful grasp through premature release requests

> Can an evolved release rule distinguish accidental opening from a necessary release
> during manipulation?

The UNISafe result makes premature release an inexpensive mechanism to inspect. It has
not been established in ManiGuard. Evolution could adjust how persistent an opening
request must be before execution, using finger position and command history, with a
bounded delay and explicit return of control.

**First test:** compare the unchanged policy with a short fixed release delay. Inspect
whether it prevents loss of a useful grasp and whether it obstructs necessary regrasping
or stage transitions. Finger aperture is evidence to investigate, not proof of a grasp.
Do not assume every transport task requires release at the destination: the documented
spatial goal checker requires the target to remain held while intersecting the goal sphere.
Release decisions must follow the actual stage and goal, such as releasing a seated lid.
[Success checking](https://nu-ideas-lab.github.io/ManiGuard/docs/evaluation/#success-checking-evalgoal_checkerpy).

This is a cheap screening candidate. If a fixed rule works across cases, retain it as the
repair and baseline. Evolution earns further attention only if context-dependent release
decisions improve on that rule and matched ordinary learning.

## 5. Stop repeating an unsuccessful approach

> Can a small evolved retry controller recover progress from repeated ineffective
> contact attempts?

The hypothesis is that a failed grasp or ineffective push leaves the policy repeating
similar commands. A bounded retreat and another approach might restore progress before
the interaction becomes dangerous. Safe unsuccessful episodes motivate inspection, but
do not establish that recoverable stalls exist.

Evolution could adjust the no-progress detection window, retreat magnitude, retry limit
and choice among a small set of corrections. Available robot observations and history
must support the detector; simulator object displacement is privileged if used online.

**First test:** identify a recurring failed attempt and demonstrate that one scripted
retry helps. Compare with continuing the original policy for the same episode horizon
and with a simple retry dispatcher. Retries occur within the physical episode and consume
time and interactions; they do not reset or roll back the environment.

This requires more implementation than motion limits because corrections must work from
the states actually encountered. It can be studied without a safety filter. Further
evolutionary search is justified only if useful alternative corrections exist and a single
fixed retry is insufficient. Failed finite retry search does not prove irrecoverability.

## 6. Improve the return to task execution after a safety intervention

> Can an evolved controller prevent repeated unsafe proposals or disruptive switching
> when a safety guard hands control back to the task policy?

After a guard intervenes, the policy might immediately repeat the blocked proposal or
alternate rapidly between task execution and correction. This is a proposed problem to
test after adding a guard, not an observed ManiGuard result. The standard monitor reports
violations after execution and supplies no predictive action filter.
[Monitor documentation](https://nu-ideas-lab.github.io/ManiGuard/docs/foundations/ltl_safety/).

Evolution could search correction duration, fresh-plan requests and conditions for
returning control, using intervention history and available robot observations. Keep the
guard's safety-trigger definition fixed across these comparisons so the study isolates
what happens after intervention.

**First test:** establish that the guard prevents some violations but leaves avoidable
loss of progress. Compare immediate hand-back with a fixed settling period and a fresh
policy query. Develop a richer controller only if these simple alternatives leave useful
headroom. Corrective actions and additional queries count toward the budget.

This most directly connects to the original task-continuation question. Its entry cost is
higher because a functioning guard and useful corrective behaviours are prerequisites.
Unlike Candidate 5, its trigger is a safety intervention rather than generic lack of
progress. Recovery after an actual violation cannot erase that episode's violation label.

## Screening order and shared comparisons

Adaptive replanning remains the first recorded candidate. Motion moderation is the
cheapest additional implementation screen; prerequisite guarding offers a more direct
safety question. Gripper persistence is worth checking if traces suggest premature
release. Retry and hand-back controllers become attractive once the relevant failure and
a working correction have been demonstrated. This is an investigation preference, not a
committed sequence of experiments.

For each candidate, first establish a repeatable failure and a useful simple correction.
Then compare fixed rules, random search, evolution and ordinary learning over the same
behavioural interface and observations. Apply the scene splits, outcome reporting and
full interaction/query accounting described for Candidate 1 to the other candidates too.
Log original policy proposals, adapter outputs and executed actions separately. Test
individual changes before combining them, and retain the fixed physical safety definitions.

The lesson from the [UNISafe study](docs/research/reports/UNISafe%20grasp%20maintenance%20and%20evolutionary%20adaptation.md)
is to check simple repairs early. A useful correction and an evolutionary contribution
are separate findings.
