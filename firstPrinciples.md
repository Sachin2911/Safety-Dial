# First principles

Established 22 October 2026. Working direction revised 1 November 2026.
[AGENTS.md](AGENTS.md) carries engineering and research controls;
[Principia](docs/research/principia.pdf) retains the consolidated earlier experiment record.

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
configurations, tests and local outputs belong under `SafeLeWM/`. On 3 November,
Sachin authorized deletion of the older root archive and all three local output
trees; see the [deletion record](docs/records/progress/receipts/local-deletion-20261103.md).

As of 4 November, Sachin has explicitly requested [LeWM_Experiment1](LeWM_Experiment1/readme.md)
for new experiment preparation and its Vast compute pipeline. New files for that
work belong there, as an exception to the earlier SafeLeWM placement instruction.
Existing study protocols and results remain in SafeLeWM. Infrastructure setup
does not reopen completed protocols or authorize new research runs.
At Sachin's request, the reproduction overview and its LaTeX source live in
[`theOdessy/`](theOdessy/reproduction.pdf) at the repository root.

Evolution may contribute by reaching useful performance with less manual tuning;
beating a specialised expert controller is not a prerequisite. Optimizer superiority,
comparability, reduced tuning and same-weight transfer are distinct claims.
Completed outcomes are indexed below. Earlier experiments inform engineering,
but are not assumed fully ablated or correct.

The original design includes monitor-only evaluation, adapter-interface qualification,
matched learning comparisons, three environments, focused ablations and a risk-budget
dial. The active goal below distinguishes completed work from remaining work. Every
new stage still requires a prospective protocol and finite budget.

## Where to find completed work

[Rule43](SafeLeWM/Rule43.pdf) and the [results index](SafeLeWM/results/README.md)
record the seventeen completed LeWM accounts. [SafeBlock](SafeBlock/README.md)
holds the main study; [Principia](docs/research/principia.pdf) records earlier experiments.
Completed protocols and result records remain local. The sixteen verified SafeLeWM
archives are recoverable using the [cloud notes](SafeLeWM/archive/README.md).
Other local output trees and the root archive, including the earlier navigation
snapshot, were deleted at Sachin's request on 3 November. Historical preservation
statements below describe the original closeouts, not current local availability.
Consult the frozen protocol and its correction before inspecting a historical run.
Completed work does not authorize another experiment or reuse of an unused budget.

## Active goal, 4 November

Sachin explicitly requested renting a GPU and performing the four-environment
frozen LeWM checkpoint reproduction. The prospective protocol is
[`LeWM_Experiment1/reproduction/protocol.md`](LeWM_Experiment1/reproduction/protocol.md).
It covers original LeWM plus CEM on TwoRoom, Reacher, PushT and OGBench single Cube,
with fixed nearby-goal panels and separately reported paper/released-config settings.
The initial session has a six-hour work limit and $8 estimated budget. Rental
shutdown remains manual. This new authorization does not reopen completed studies.
Vast instance 59111700 is rented under this protocol. All four checkpoints and the
TwoRoom/Reacher datasets are verified. TwoRoom's two engineering cases passed the
implementation audit and independent backup verification. Its fixed scientific
job has been submitted. Exact run state is held by the compute controller;
evidence is uploaded to the private Experiment 1
dataset. No scientific reproduction result is yet reported as final.

## Prior active work, 2 November

The matched PPO comparison and PushT AP reproduction are complete, audited and
independently backed up. PPO safe completion was 20/216, CMA 44/216 and random
37/216; both pre-specified contrast intervals include zero. The matched PushT
baseline completed 88/128 using the authors' evaluator and 89/128 using our wrapper.
See Rule43 accounts 14 and 15. These are separate from the earlier studies.

Sachin explicitly authorised the next sequence as an active goal: update Rule43;
freeze the validated PushT baseline and qualify diverse hazard cases; test adapter
authority and, only if it passes, run matched CMA/random/PPO comparisons; revisit
the safety dial and consolidate the paper alongside Block Plucking. Stage-specific
protocols, panels, gates and finite budgets must still be frozen before execution.
No failed prerequisite is waived by this goal and no historical allocation resumes.

He approved using the existing eight-4090 Vast host for PushT, the single RunPod
4090 for TwoRoom safety-dial development, and then explicitly assigned the L4 to a
bounded evolution-improvement study with matched random-search controls. Figures
and audits may run around those jobs. Use development cases for method choices,
keep new final cases untouched, preserve every variant and report failures. Each
optimizer comparison stays within its physical host. No new rental is requested;
shutdown remains manual. The scientific gaps remain cross-environment usefulness,
reduced tuning and reliable gains beyond random search. Block Plucking continues
in its own work area; do not overwrite its concurrently developed protocols.

The current bounded studies are frozen in
`SafeLeWM/PushTValidatedQualification.md`, `SafeLeWM/EvolutionBudgetStudy.md`
and `SafeLeWM/SafetyDialLibrary.md`. The L4 tests population four versus eight
and matched random search in two development repeats, with 240,000 training
and 308,400 total actions allowed. The single 4090 evaluates the 17 already-selected
risk-target actors and four references on 756 development episodes, after 16
engineering checks, with 154,800 total actions allowed. No new optimizer is trained
for this library test. Its 12 layouts were previously used development data.
The preceding zero-rollout diagnostic is `SafeLeWM/SafetyDialDevelopment.md`.
The L4 study remains active. The dial study is now complete, audited and
independently preserved (Rule43 account 17): 17/45 target assignments were
unavailable and 5/28 available settings exceeded the requested empirical risk
limit. Four of six changing-policy curves had nondecreasing risk. These are
development tradeoffs, not evidence of calibrated risk. No new run is authorized
by this completed protocol.

PushT validated qualification is now complete: 329 episodes passed audit. Native
completed 12/16 but violated in all 16; five scenes met baseline/hazard admission.
The sixteen pre-drawn head probes achieved only 0-1/10 safe completions on admitted
scenes, with safe probe completion on one scene. Useful authority failed, so the
conditional optimizer comparison did not run and twelve reserved cases remain
untouched. See Rule43 account 16. A new interface study is not silently substituted
for the failed gate. The L4 study remains active; the safety-dial study is complete.

The current working manuscript is
`docs/research/honours-gecco/world-model-adapters.tex` and its PDF. It consolidates
completed LeWM and Block Plucking evidence while identifying active studies and
unestablished claims. The older October thesis remains unchanged.

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
- Keep [panels.json](docs/records/progress/panels.json) frozen. Held-out tasks have baseline and
  descriptive evidence, but receive no new intervention episodes before the final
  comparison and are unavailable for method selection.
- Preserve original evidence, and use fresh files and output paths for new studies.

## Recorded work through 31 October

The historical chronology, pre-cleanup `firstPrinciples.md`, October ManiGuard and
LeWM notes, and superseded Oracle drafts were bundled during the initial cleanup.
Sachin later authorized deleting the entire root archive, including that bundle
and its manifest. See the [deletion record](docs/records/progress/receipts/local-deletion-20261103.md).
Recover any required historical protocols from an available independent copy before
packaging or review. Recorded outcomes and execution authority are unchanged.
