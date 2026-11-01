# First principles

Established 22 October 2026. Working direction revised 1 November 2026.
[AGENTS.md](AGENTS.md) carries engineering and research controls;
[Principia](principia.pdf) retains the consolidated earlier experiment record.

## Working direction, 1 November

> Can evolutionary search learn compact adapters that recover safe task completion
> around frozen world-model systems, reducing reliance on manually tuned behavioural rules?

The allocated topic is [Safe AI via Evolutionary Algorithms](docs/allocatedTopic.md),
with GECCO 2027 the target venue. LeWM is the opening study across TwoRoom, PushT
and OGBench Cube. Block Plucking is the intended main study; ManiGuard is deferred.
The [two-page Rule42 outline](SafeLeWM/Rule42.pdf) and
[detailed first experiment](SafeLeWM/FirstExperiment.md) are the current design.
The user requested a [SafeLeWM working folder](SafeLeWM/README.md), then explicitly
directed that all first-experiment work take place there. Its documents, new code,
configurations, tests and local outputs live under `SafeLeWM/`. Existing historical
code and evidence retain their original locations.

Evolution may contribute by reaching useful performance with less manual tuning;
beating a specialised expert controller is not a prerequisite. Optimizer superiority,
comparability, reduced tuning and same-weight transfer are distinct claims. Current
evidence does not establish the new study's outcomes. Earlier experiments inform
engineering, but are not assumed fully ablated or correct.

The monitor-only main design, adapter-interface qualification, matched learning
comparisons, three-environment evaluation, focused ablations and later risk-budget
dial remain prospective. Protocols and spending limits still require a fresh freeze.

On 1 November Sachin authorised provisioning one fresh idle RunPod, with nothing
run on it. He explicitly removed the proposed automatic stop and will say when to
turn it off. This does not authorise software installation or experiment computation,
and no historical rental or unused budget is resumed.

## Earlier evidence

The baseline established useful Jar variance: 24 benchmark safe successes and
43 counted violations in 104 episodes, with engagement in 103/104. Repeated scenes
and seeds can yield different trajectories.

The Jar retry, execution-feedback, DSRL and execution-knob studies did not establish
a replicated completion lever. Task 6 failed numerical gates; task 7 ran learning
and failed its completion screen; task 11 stopped at simulator startup. Task 15's
screen was promising, but final recovery-cohort confirmation remains unverified.

Tasks 16 and 18 established useful fixed planning configurations at unchanged
margins in TwoRoom, Reacher, PushT and Cube. Feedback added no established benefit.
Task 17 completed an optimizer comparison without establishing an evolutionary
advantage; eight of ten evolution runs selected shared preset anchors. These
LeWM studies are separate from the original ManiGuard objective.

Task 19's Jar shield had 32/107 strict safe completions in each arm and reduced raw
violation episodes from 37/107 to 21/107. More episodes stalled after closure.
The secondary endpoint, design-time inspection, host differences and absent optimizer
controls limit inference. CMA-ES stayed near its hand-chosen start. The UNISafe
grasp-sensing ablation confirmed useful information in a separate Dreamer system.

## Historical ManiGuard preconditions

These were the prior Jar study's gates. The optimizer-superiority gate below is
not a prerequisite for the current reduced-tuning contribution. The current LeWM
design defines its own prospective qualification and comparison.

1. **Variance:** native execution sometimes completes safely and sometimes does not.
2. **A lever:** a controller improves safe completion, beyond merely changing actions.
3. **Stable evaluation:** sufficient task and learner repeats distinguish gain from noise.
4. **Optimizer advantage:** evolution beats tuned fixed, parameter random search and
   ordinary learning with the same information and matched interaction budgets.

Jar stage 1 is established. A replicated ManiGuard completion lever remains missing.
Fewer violations alone does not complete stage 2. Freeze each new stage before its
eligible episodes, including panels, seeds, information, controller domain, selection,
budget, optimizer repeats, analysis and stopping rules. An unspent allocation or
negative result does not authorize extensions.

## Controls that remain in force

- For ManiGuard, reproduce the paper protocol: 20 Hz, eight executed actions per sixteen-step chunk,
  256-pixel collection-time cameras, Jar horizon 2,000, ten-step goal hold and three
  policy seeds. Record deviations.
- Score benchmark and requested-sequence outcomes, raw and counted violations,
  unfinished episodes and engagement with denominators. Jar closure precedes carry.
  Under independence, zero events in n trials gives an approximate 3/n upper bound.
- The benchmark monitor labels and never vetoes commands. An intervention shield is
  separate. Keep proposed, filtered and executed actions distinguishable.
- Charge simulator interaction for engineering, failed starts, sweeps, discarded
  candidates, tuning and selection. Report actual work, model queries, learner updates
  and elapsed time separately. Equal episode caps do not ensure equal work.
- Keep comparisons by physical host. Use uncertainty over task and optimizer repeats;
  correlated rollouts are not independent evidence. Movement need not be useful progress.
- Keep [panels.json](progress/panels.json) frozen. Held-out tasks have baseline and
  descriptive evidence, but receive no new intervention episodes before the final
  comparison and are unavailable for method selection.
- Preserve original evidence, and use fresh files and output paths for new studies.

## Recorded work through 31 October

This chronological log retains the statements made at each stage. Later closeouts
supersede earlier running statuses; completed protocols do not authorise new work.

On 29 October, Sachin requested a complete iterative ManiGuard adapter and evolution
pipeline, without separate small-study approval gates. The first implementation
evolves a recurrent controller over bounded arm corrections and gripper decisions,
using explicitly selected privileged simulator observations. The
[implementation record](docs/research/maniguard-evolution.md) describes its source
snapshots, development-only panels, accounting and existing rental limits. This changes
the development workflow; it does not establish a completion lever or evolutionary
advantage, change the held-out panel, or authorize another rental.

The first development run completed its 68 episodes on 30 October. The selected
recurrent adapter achieved 0/6 strict safe completions on fresh development cases,
versus 1/6 for native execution. It left 6/6 episodes unfinished and engaged in 4/6;
native left 4/6 unfinished and engaged in 6/6. Raw violation episodes were 2/6 and
3/6 respectively. This does not establish a completion lever. Review fixes are
separately implemented and software tested, with no robot evaluation of that draft.

On 30 October, Sachin requested the [next experiment protocol](docs/research/maniguard-evolution-v3-experiment.md)
and a [future goal prompt](docs/research/maniguard-evolution-v3-goal.md), plus one
fresh Vast GPU rental with a ten-hour limit, explicitly without starting the
experiment. The plan has 108 scheduled development episodes, conservative CMA-ES
perturbations, native/elite/mean selection and fresh validation. The
[preparation receipt](progress/receipts/20261030-vast-maniguard-v3-preparation.json)
records instance 58363639 and its 30 October 18:46:32 SAST cutoff. Sachin then
explicitly invoked the goal. The [execution receipt](progress/receipts/20261030-maniguard-v3-execution.json)
records the completed implementation, 62 passing rented-host software tests, frozen
inputs and the 09:56 SAST (first setup failure; fresh recovery at 10:13 SAST) training launch. Selection and validation remain pending.
The same cutoff and held-out exclusions remain in force; no additional rental or
extension is authorized.

Repository consolidation and manuscript polishing continue alongside implementation. Task 7's
closeout and task 15's missing final confirmation remain unresolved records, not
launch instructions. No current rental status is inferred here.

Completed protocols and all pre-22 October studies are historical. This cleanup
authorizes no new experiment or rental. The prior plan, old task documents and
original status wording are preserved in the [recovery bundle](archive/20261029/README.md).

In the separate LeWM track, Sachin prioritized implementing the learned safety checker
on 29 October, with PushT first, then PushT optimization, then ablations and other
environments. The [implementation record](docs/research/safety-model.md) now includes
a working learned-motion checker using native pusher proprioception and generic
geometry. It completed the island, bar and gate demonstrations with 6 px pusher
margins and no reference-predictor calls. These three runs share a known task scene;
the independent motion test is separate. This does not establish a new performance
advantage, latent-only safety or cross-environment transfer, and authorizes no rental.

The subsequent [small PushT parameter search](docs/research/pusht-tuning.md) completed
on a separately authorized RunPod host. Selection reduced development completion
actions, but both the current and selected settings scored 0/3 safe completions on
the reserved cases. One case began inside its 6 px hazard margin after reset settling;
the other two remained clear but unfinished. The record retains every case, failure
and interaction, including earlier local engineering work. This establishes the
tuning workflow, not a transferred completion improvement. Further experiments need
reset-state admission checks and a fresh freeze.

A fresh, separately authorized author-preset comparison then scored PushT 2/6 safe
reserved hazard completions with the authors' settings and 3/6 after tuning, with
both no-hazard controls 6/6 and no observed violations. A subsequent fixed-setting
budget diagnostic retained the exact first 50 actions and found unchanged completion
at 50, 100 and 150. These small results are recorded in the
[tuning note](docs/research/pusht-tuning.md), without a statistical superiority claim.

The conditional learned-safety extension has completed TwoRoom and Reacher with
150-action limits. TwoRoom improved reserved safe completion from 0/6 to 3/6;
Reacher remained 0/6 in both arms. All unfinished episodes and no-hazard controls
remain explicit in the [navigation receipt](progress/receipts/20261030-lewm-navigation-results.json).
The safety models learn native-proprioception motion and share primitive geometry;
their empirical calibration does not guarantee safety. Cube's separately repaired
comparison is complete and independently audited: both arms scored 0/6 reserved
hazard completions; controls scored authors 3/6 and selected 1/6. Earlier engineering
passes remain excluded and retained. Its original object-position task differs from
older reaching subtask results. This establishes the shared workflow, not universal
completion improvement. Preservation and rental closeout remain separate checks.
Subsequent visual QA found noisy policy-input images in one selected reserved Cube
episode. Its cause remains unconfirmed; the Cube performance result carries that
observation-quality qualification. Original data and outcomes are retained.
The [combined results](progress/receipts/20261030-lewm-results.json) are locally
verified with all models and 48 paired replays. The authorized RunPod was deleted
after verification, with provider absence confirmed and a $3.31 conservative combined
rental bound under the $5 cap. Sachin approved the combined private upload; all 79
files were downloaded from pinned revision `b374ee4c602c538e992ba0e6072e58e18abcb168`
and hash-verified. The
[preservation receipt](progress/receipts/20261030-lewm-private-preservation.json)
records the private destination and exact model identities.
Sachin subsequently authorized an exact hand-set PushT reproduction before further
tuning. Historical and current CPU each retained 3/3 safe completions with bitwise
identical original state traces; current GPU completed 2/3, with bar safe but
unfinished. This is one known scene, not a generalization result. The
[reproduction receipt](progress/receipts/20261030-pusht-hand-preset-reproduction.json)
records all nine trials. Its pod was deleted after local verification; the new
conservative combined rental bound is $3.44 under the same $5 cap. No new tuning ran.
This status authorizes no future rental or archived-job restart.

A separately authorized [matched GPU control](progress/receipts/20261030-pusht-gpu-control-results.json)
then ran six episodes on one fresh pod. Historical and current bundles produced
bitwise equal actions, states and dense traces: both completed 2/3 safely, with bar
unfinished at 50 actions and no observed violations. The conditional score diagnostic
did not trigger. This finds no stack regression on the known scene, while the separate
roles of proposal RNG and numerical scoring remain unisolated. No new tuning ran.
All 144 archive members, raw evidence and three paired replays were verified locally
before deleting this pod. Provider absence was confirmed; the combined conservative
RunPod estimate is $3.50 within the unchanged $5 cap. No other rental was modified.

The three proposed SafeCEM refinements are now implemented, with five focused GPU
software tests passing. A [six-episode control](progress/receipts/20261030-pusht-improvements-results.json)
using the unchanged current models and manual search recipe found 2/3 safe completions
for each arm at 50/100/150 actions, with no observed violations and 3/3 engagement.
The changed planner took 26/150/40 actions versus 25/150/25; bar remained unfinished.
This establishes implementation, not a completion improvement. All 660 changed-arm
iterations had at least 25 feasible candidates, leaving the scarce-elite branch
unstressed in these rollouts. No tuning or per-change ablations followed. The completed
todo was cleared after preserving its original PDF and builder. All 160 archive
members and three paired replays were verified locally before deleting this pod.
The shutdown guard missed its cutoff across the interruption; the closeout includes
the extra idle runtime. Provider absence is confirmed and the combined conservative
RunPod estimate is $4.76 within the unchanged $5 cap. No other rental was modified.

Sachin then authorized a fresh RunPod rental for the proposed LeWM behavioural
adapter's single-episode integration test. The 510-parameter network and resumable
CMA-ES mechanics are implemented, with eleven rented-GPU software checks passing.
The seeded untrained adapter supplied seven proposals, evaluated 105 times within
the existing SafeCEM candidate budget. One known island episode completed safely
in 35 actions: 1/1 safe completions, 0/1 violations, 0/1 unfinished and 1/1 engaged.
No raw adapter proposal was selected directly and no real optimizer updates ran.
This establishes integration, not completion benefit or evolutionary advantage.
The [implementation record](docs/research/lewm-proposal-smoke.md) and
[results receipt](progress/receipts/20261030-lewm-proposal-results.json) retain the
failed pre-simulator launch, source versions and independent audit. All 172 archive
members were downloaded and hash-verified, and the replay was inspected before
deleting pod dcyk2i5ug0e9yv. Provider absence is confirmed. The fresh rental estimate
is $0.115 under its separate $1 limit; no earlier allowance was inherited. Real
training requires a new prospective protocol and budget.

Sachin subsequently authorized a separately frozen
[LeWM proposal-weight pilot](docs/research/lewm-proposal-pilot.md). Four real diagonal
CMA-ES updates evaluated 24 genotypes within a 76-episode schedule; fourteen GPU
software checks passed. Native, initial and evolved elite each completed 1/2
selection episodes safely, with no observed violations and 2/2 engagement. Native
made better progress in the unfinished case and won selection. Reserved validation
therefore repeated native in both arms: each scored 5/6 safe completions, 0/6
violations, 1/6 unfinished and 6/6 engaged. The evolved elite was not validated on
those reserved cases. None of 874 raw adapter proposals was selected directly;
influence through CEM fitting remains possible. This establishes a working real
training pipeline, not a completion benefit or evolutionary advantage. All 895
archive members, trained weights and six paired replays were verified locally
before deleting pod 8ldukx5cq1o6qo. Provider absence is confirmed; the fresh rental
estimate is $0.145 under its separate $1 cap. No further work or rental is authorized
by this completed protocol.

Sachin then authorized implementing influence diagnostics, a bounded warm-start
correction, common-panel selection and a gated development screen. The
[residual-mixture study](docs/research/lewm-residual-study.md) completed nineteen GPU
software checks and 233 valid episodes. Zero corrections exactly recovered native
trajectories, images and model work; nonzero corrections influenced the search.
Four CMA-ES updates and an equal-size parameter random search each evaluated
24 genotypes on one fixed four-case panel. Native completed all four training
cases. On a separate selection panel, native and CMA each scored 2/4 safe
completions, 0/4 violations, 2/4 unfinished and 4/4 engaged; untrained and random
search each scored 3/4 safe completions, 0/4 violations, 1/4 unfinished and 4/4
engaged. Untrained won selection. The evolved-completion gate failed, so fresh-scene
validation, optimizer repeats and ordinary-learning comparisons remained closed.
This establishes a functioning influence mechanism and common-panel experiment,
not a completion benefit from evolution or an evolutionary advantage. All 2,510
archive members and sixteen paired replays were verified locally before deleting
pod qwm6ejjm54ar0e. Provider absence is confirmed; the fresh rental estimate is
$0.296 under its separate $1 cap. No further work or rental is authorized by this
completed protocol.


A separately frozen [LeWM readiness and fixed-adapter screen](docs/research/lewm-screen.md)
completed on 31 October local time, with 26 passing GPU software checks, 54 scored
episodes and four six-action engineering probes. PushT native no-hazard readiness
was 4/6 safe completions. Across eighteen hazard episodes, native scored 3/18 and
the frozen untrained adapter 1/18, with no observed violations, fifteen versus
seventeen unfinished episodes, and full engagement. No-hazard completion was 4/6
native and 5/6 untrained. Cube scored 0/6 original-object-goal no-hazard completions,
with no observed violations and 6/6 engagement; its hazard and adapter stages stayed
closed. Both future training gates failed; no optimizer updates or final-evaluation
rollouts ran. Public source poses were previously used development inputs, not
untouched validation. The first saved-evidence audit's incorrect absolute-pixel
PushT command assumption was corrected to the native relative [-1,1] contract,
without changing or replaying episodes. All 685 archive members, frozen models and
thirty replay videos were verified locally before deleting pod dysjocg4yfj2zd.
Provider absence and retirement of both guards are confirmed. The fresh conservative
rental estimate is $0.2364 under its separate $1 cap. This establishes neither a
completion gain from this fixed guidance nor an evolutionary advantage, and
authorizes no further rental or training.

Sachin subsequently authorized a fresh [seven-hour LeWM campaign](docs/research/lewm-campaign.md)
from 31 October 00:23:14 UTC to 07:23:14 UTC (09:23:14 SAST), with a shared $10
cap, at least two parallel rented pods and up to three pods available. This explicit
instruction authorizes iterative diagnosis, repairs, newly frozen experiments and
real evolutionary training within that campaign; the prior screen remains unchanged.
Two fresh A40 lanes target action-consistent planning and compact persistent
controllers in PushT and TwoRoom. The third allocation is reserved for recovery.
Each stage and its panels are frozen prospectively; optimizer and final-evaluation
controls remain required before claims. No historical job or unused budget resumes.
The seven-hour window has now ended. All three pods received provider STOP requests
at 07:23:19 UTC and subsequently reported desired status EXITED; none was deleted.
The navigation host completed 304 final episodes across three ordinary profiles
and one separate terminal-mean profile. CMA tied or scored below the strong fixed
controller on completion in the available ordinary profiles. The full optimizer
comparison remains incomplete. Navigation and PushT final accounting closed without
unresolved items; Recovery retains one explicit older-transfer ownership gap after
its qualified corrected recollection was interrupted by STOP. Automatic approval
review rejected private artifact archiving and transfer pending explicit user
authorization, so independent full preservation remains unverified. The conservative
rental estimate through stop verification is $9.23; retained storage may continue
to incur charges. This status authorizes no further computation or rental.

Sachin subsequently approved private preservation and explicitly authorized
completing the full comparison with $3 more, then destroying the pods. The new
cumulative cap is $13 and the fresh guarded cutoff is 31 October 09:19:30 UTC.
The original GPU hosts could only resume in zero-GPU data-recovery mode because
their GPUs were occupied. Their existing evidence is being privately archived
and independently verified before deletion. An availability-driven replacement
requires newly frozen host/deadline continuation metadata and a separate V4
analysis binding, preserving all original scientific settings, cases, controls
and completed results. No new training is authorized in this completion phase.
All three original pods were privately preserved with every archive member verified
before their deletion and provider-absence confirmations. One new A40 runs the
remaining frozen profiles. Before any new final outcome, the agent revised its
operational estimate to a 120-minute guarded maximum, 09:49:30 UTC, because GPU
capacity recovery and network-volume I/O took longer than expected. The user-approved
$13 cap, scientific budgets and all original cases, seeds and controllers remain
unchanged. Fresh 69-test qualification and a bitwise cached-encoder check passed
before the first new final episode at 09:07:33 UTC. The campaign record carries
the prospective relocation, actual host, source hashes and completed outcomes.

The authorized completion is now closed. All 704 frozen final episodes completed:
576 main comparisons, 80 separate decoded-native controls and 48 exploratory
terminal-mean episodes. In the primary 50-action comparison, CMA scored 12/16,
13/16 and 10/16 safe completions across the three fixed learner repeats, versus
12/16 in every strong-fixed repeat. At 30 actions, both scored 8/16 in every
repeat. Every terminal mean also scored 8/16. The study establishes a useful
controller architecture but no consistent evolutionary completion advantage.
All final arms had zero observed violations; this does not establish zero risk.
The 16 source cases are shared across repeats, rather than 48 independent cases.

The ten relocated profiles passed independent saved-evidence audits. Four-source
accounting, conditional analysis and scientific figures completed on the rented
host. All original and replacement research archives passed every-member private
local verification before deletion. The replacement received a preservation-only
grace to 09:59:30 UTC after the unchanged science cutoff, with no further experiments
authorized. All four campaign pod IDs were absent from the provider list, and the
replacement returned not found, at 09:52:42 UTC. The conservative cumulative cost
bound is $11.1643 under the authorized $13 cap; the final invoice is not verified.
The [campaign record](docs/research/lewm-campaign.md) indexes the final evidence.
No further rental, training or evaluation is authorized by this completed campaign.
