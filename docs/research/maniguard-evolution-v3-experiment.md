# Conservative evolution of the Jar recovery adapter

Prepared 30 October 2026. Sachin subsequently invoked the explicit execution goal.
Status: the first attempt failed before scene load because the upstream asset key
was missing. The key was installed without a simulator screen. Recovery version
`run-v3-r1` launched at 10:13 SAST, retaining the original failed attempt and its
charges. Selection and validation are pending. Sachin requested remote Codex
handoff because the laptop must go offline. The
[execution receipt](../../progress/receipts/20261030-maniguard-v3-execution.json)
records 62 passing rented-host software tests, frozen inputs and the unchanged
rental deadline. Preparation and execution have separate recorded authorization.
The machine-readable protocol is
[development-v3-protocol.json](../../configs/maniguard_evolution/development-v3-protocol.json).

## Question and basis

Can conservative evolution of a recurrent residual controller improve strict safe
completion while preserving useful native behaviour? The frozen policy supplies
actions; evolution changes only bounded joint corrections and gripper decisions.

The previous development run had 0/48 strict safe completions across sampled
candidate episodes, versus 5/8 native training references. Candidate engagement was
19/48 versus native 8/8. Its final controller achieved 0/6 strict safe completions
versus native 1/6 on fresh development cases. These are different panels and
controllers, not a pooled performance estimate. The
[completed record](maniguard-evolution.md) retains violations, unfinished episodes,
engagement, the startup failure and the fully verified archive.

The working hypothesis is that broad initial corrections disrupted execution and
that a sparse progress signal gave little useful ranking among unsuccessful
controllers. The observations motivate this hypothesis; they do not isolate its
cause. The new version reduces initial arm perturbations, includes a native candidate,
rewards valid target contact below completion, and measures candidates before final
selection. A corrected auxiliary ordering rule requires grasp before departure.

## Controller and fixed execution

Use the existing 205-parameter controller: 35 features, four recurrent units and
nine outputs. Inputs include joint state, native actions, relative jar and
end-effector pose, orientation, original propositions, grasp, contact and causal
history. Simulator-derived quantities are declared privileged inputs. Preserve the
0.08-radian joint bound, 0.5 smoothing and 0.6 gripper threshold.

Start from a freshly seeded controller with exactly zero output weights and biases,
which produces native commands. The failed previous final controller is not an
initialization. Initial parameter standard deviations are 0.025 for arm output
parameters, 0.05 for hidden, recurrent and mixing parameters, and 0.3 for gripper
parameters. These are parameter scales, not action amplitudes.

Keep the pinned public sources and Jar checkpoint listed in the JSON protocol.
Execution remains 20 Hz with eight actions executed from each sixteen-action chunk,
legacy 256-pixel left and wrist observations, raw joint-position control, assisted
grasping, long fingers, a 2,000-control horizon and ten-control goal hold. The original
monitor labels violations and never vetoes commands. Native arms run the same
observer with modifications disabled. Preserve proposed, modified and executed
commands separately. Comparisons use contemporary controls on the new physical host.

## Training

Run diagonal CMA-ES 4.4.4 for six generations with population six, using optimizer
seed 303021 and initialization seed 303022. Train on development tasks 0 and 3.
Generation policy seeds are 303100 through 303105, one per generation.

Each population contains the exact native-output controller and five sampled
controllers. A separate native reference also runs on both tasks. The anchor and
reference remain separate episodes, with all work counted. This produces 84 training
episodes: six generations times seven controllers times two tasks. Candidate order
is seeded, and native-reference placement alternates by generation.

Use the already implemented `completion_progress_contact_v3` objective. For a panel
of n cases, fitness is `(4*n+2)*S + P + 0.25*C - V/(n+1)`, where S is strict safe
completion count, P is the sum of safe ordered progress stages, C counts safe target
contact and V counts raw violation episodes. A strict safe completion outweighs every
possible auxiliary difference. Invalid panels are ineligible.

Ordered progress is closure while supported, then ten consecutive closed/grasped
controls after closure, then a later departure while closed and grasped. A raw
violation or incorrect closure-before-departure sequence removes auxiliary credit.
Contact earns credit only while supported, upright and not dropped, with a valid
sequence prefix and no raw violation. Movement alone earns no credit.

## Selection before validation

Nominate three controllers: native, the best evaluated non-anchor training candidate,
and the final CMA-ES phenotype mean. The training candidate is nominated by recorded
fitness, breaking exact ties by earliest generation then lowest candidate index.
This nomination does not itself select the final controller.

Evaluate all three on tasks 6 and 9 with policy seeds 303200 and 303201, for twelve
episodes. Use task-major case order and rotate native, elite and mean placement by
case index. Keep all three slots even if parameter vectors coincide.

An adapter is eligible only if its strict safe completion count exceeds native on
these four cases. Among eligible adapters, maximize the same v3 fitness over the
selection panel; an exact tie prefers the evaluated elite. If neither qualifies,
retain native. Save the selected parameters, controller identity and selection
receipt before the first validation episode. Validation never selects or updates it.

## Fresh development validation

Use development tasks 15 and 17 with policy seeds 303300, 303301 and 303302. Evaluate
native and the frozen selected controller, alternating arm order across the six
task/seed cases. This produces twelve validation episodes and 108 planned episodes
overall. These are reserved development cases for this run, not globally untouched
tasks. The held-out panel in `progress/panels.json` receives no new episodes.

Report strict safe completion, benchmark safe completion, requested-sequence
completion, raw and counted violation episodes, unfinished episodes and engagement
with denominators of six per validation arm. If selection returns native, label that
arm as a repeated native control; a difference between the two arms cannot be
attributed to evolution. Keep native-anchor training episodes separately identified.

An increase from a selected non-native controller is a development signal requiring
replication. One optimizer seed and six validation cases per arm do not establish
reliable superiority or an evolutionary advantage. The later GECCO comparison still
requires native, tuned fixed settings, parameter random search and ordinary learning
with identical information and matched simulator interaction budgets.

## Failure diagnosis

For each episode, save first contact, closure while supported, ten-control grasp,
later supported-to-departed transition, first raw violation and ten-control goal hold.
Report the farthest valid ordered stage, invalid sequence prefixes, engagement, arm
correction frequency and RMS, and gripper overrides. These diagnostics explain where
execution diverged; they do not alter fitness, selection or validation after launch.

## Time and interaction budget

The rental has an absolute ten-hour deadline measured from its creation request.
Setup, idle time, coding and experiment work all consume that allocation. The rental
receipt supplies the actual instance ID and deadline; neither resume nor a delayed
start resets the clock. The prior 68-episode run took 3 hours 23 minutes, suggesting
about 5 hours 23 minutes for 108 episodes at the same average throughput. This is an
estimate, not a runtime guarantee.

Require at least seven hours remaining before the first scored episode and reserve
the final 45 minutes for preservation. Apply the work cutoff throughout execution.
Allow at most four additional failed engineering starts, for a total ceiling of
112 starts, 225,120 instrumented environment-step attempts and 28,116 model-query
attempts, including at most four server-preparation queries. Warmups and failed
starts count. Report simulator steps and physics counters separately; bootstrap
physics before instrumentation remains unknown.

Run the full schedule without performance-based stopping. A technical or time limit
produces an explicit incomplete result and preserved evidence. Do not replace it
with a smaller favourable panel, change seeds or extend the rental. Software repairs
after launch retain the old version and its costs, and must preserve the scientific
protocol. A scientific change requires a new prospectively written protocol.

## Implementation and preparation status

The conservative mutation profile, native injection, corrected progress rule and
contact objective were initially covered by 33 software tests on the previous host.
The current implementation adds measured selection, phase diagnostics, immutable
generation checkpoints, accounting, audit and explicit incomplete-result preservation.
It passed 62 tests and Ruff on the new rented host before launch. The checks exposed
and repaired exact-anchor roundoff and a pycma list-versus-array archive-key mismatch;
the six-generation regression now rejects injection warnings. A public NVIDIA-wheel
download timeout was repaired without simulator work.

The prepared protocol JSON remains unchanged and is not a runnable configuration.
The [runtime config](../../configs/maniguard_evolution/development-v3.json) binds its
hash, actual rental and all frozen panels. The source snapshot includes every
imported controller, selection, scoring and diagnostic helper. The policy server
used one charged synthetic preparation query and no separate rollout screen. The
first declared native training episode is the first simulator scene.

## Preservation and teardown

Use fresh output paths and the private dataset
`Sachioster/maniguard-jar-replication-20261022`, under
`22_conservative_adapter_development/<actual-launch-utc>/`. Save the source freeze,
configuration, every controller and checkpoint, all episode records and videos,
action boundaries, monitor labels, causal diagnostics, counters, timing, failed
attempts, environment provenance and final audit. Exclude credentials, account
environments, licence keys and unrelated projects. Public model and asset references
remain pinned; new learned artifacts are included in the private evidence archive.

Mirror completed evidence during execution. Before teardown, seal the remote tree,
verify every local file, upload privately at a pinned revision and download every
file to verify its hash. Destroy only the new instance after preservation and confirm
its absence at the provider. The independently armed guards stop GPU computation
at the ten-hour deadline; stopped storage remains billed until the instance is
destroyed. The future goal prompt requires verified preservation and teardown.
