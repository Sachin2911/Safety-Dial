# Phase 3, Part 1: evolving a safety dial for ManiGuard execution

**Date:** 18 October 2026.

**Status:** completed on 18 October 2026. All 48 full episodes passed physical evidence
audits. The frozen controller recorded 4/8 fresh safe completions and 2/8 violating
episodes at full strength, compared with identity's 5/8 and 0/8. See the
[GECCO-template report](docs/research/reports/maniguard-phase3-part1-gecco/named.pdf)
and [frozen protocol](configs/maniguard/phase3/part1.json). The original planning text
below is preserved in planning commit `8df7ac9`; this status update does not revise
the executed protocol or authorize further sampling.

**Resource limit:** one RTX 4090 or RTX 5090, with a hard limit of eight allocated
wall hours, hence at most eight allocated GPU-hours. Setup on the allocated machine,
warm-up, failed attempts, idle time, simulation and reporting all consume this limit.

**Target allocation:** 24 full search episodes and 24 fresh comparison episodes.
The comparison includes dial settings zero, one-half and one. The predefined smaller
version uses 16 search and 12 comparison episodes, for 28 in total.

This document expands the approved recommendation in [phase3.md](phase3.md#21-candidate-maniguard-evolution-pilot-on-one-gpu).
It specifies the first experiment, not the complete GECCO study. Its more precise
Part 1 definitions take precedence over the sketch in that section. Other proposed
phase-three allocations remain separate. Existing phase-two jobs stay paused.

## 1. Purpose and the question we can answer

The aim is to produce a working evolutionary experiment, a saved controller and a
fresh physical comparison within one GPU session. Safe CEM with LeWM is the example
of the desired scale and concreteness. ManiGuard remains the platform.

The primary question is:

> Can a small evolutionary search discover a useful execution controller for a frozen
> ManiGuard Clutter policy, and does a user-controlled dial expose a useful measured
> safety/completion trade-off on episodes excluded from search?

Evolution selects four execution settings. A selected controller then runs for an
entire episode without further optimization. The VLA proposes actions; the controller
moderates arm commands and determines when to ask the VLA for another action chunk.
The original simulator, task goals and physical safety monitor remain the evaluator.
An external scalar dial controls how strongly the evolved adaptation is applied. The
dial is a user input, not a fifth parameter evolved to maximize a score.

The useful first result would be a controller that improves fresh safe-completion
counts without an accompanying rise in violating episodes. A mixed trade-off or no
gain is also a result. A positive outcome cannot be guaranteed by selecting an optimizer
or committing compute.

This experiment can show whether a controller found by this search is useful. It
cannot establish that evolution beats random search, ordinary learning or a simple
fixed setting. Those comparisons belong to the next allocation if the interface merits
further work. It also does not test penalty-free selection, learned safety filtering,
world-model planning, VLA fine-tuning or whole-family transfer.

## 2. Why start with Clutter

Clutter has shown both safe completions and failures in our partial pilot observations.
That provides possible headroom and a less sparse outcome signal than the currently
observed Jar and Stack cases. It does not establish a stable success rate or prove
that execution parameters can repair the failures.

Its native horizon is 1,100 control actions, compared with 2,000 for Jar and 4,200 for
Stack. Earlier serial Clutter processes took approximately 177 to 191 seconds; the later
eight-way-loaded host recorded a 417-second median. The different machines and workloads
prevent treating these as a fresh 4090 or 5090 benchmark. See the
[development log](docs/research/research_notes/ManiGuard%20phase2%20development.md) and
[phase-three evidence](phase3.md#3-evidence-motivating-the-change).

The [six-hypothesis study](docs/research/reports/ManiGuard%20six-hypothesis%20study.md)
did not demonstrate a repeatable adapter repair. It motivates a bounded test rather
than an assumption that this interface will work. Part 1 searches combinations of
existing execution controls; it is not a repetition of those completed batches.

The low-cost opportunity is reuse: the policy server, motion adapter, evolutionary
operators, instrumented evaluator and physical outcome checks already have reference
implementations. The new work is their small, explicitly checked integration.

## 3. Experimental system and fixed contracts

Use one persistent Clutter policy server and one serial simulator worker on the same
GPU. Initialize a fresh simulator process per physical episode using the existing
instrumentation approach. Do not add concurrent clients merely to fit the budget.

Use the pinned metadata in:

- [families.json](configs/maniguard/phase2/families.json): `clutter_pickup`, eight-dimensional
  absolute-joint commands, `joint_position_raw`, assisted grasping and a 1,100-action horizon.
- [sources.json](configs/maniguard/phase2/sources.json): the Clutter policy
  `IDEAS-Lab-Northwestern/pi05-base-datagen-v1-clutter-joint-2cam-lora`, revision
  `f572e33bdb498a8d9992a72312ee849be868a2b6`, step `7100`.
- The same source manifest's ManiGuard revision
  `6bf4c61a1f168cc46e02eb22c41bcb51815e9726` and supporting asset revisions.
- [completion.json](configs/maniguard/phase2/completion.json): raw physical violation
  and requested-completion definitions, including their source hashes.
- [splits.json](configs/maniguard/phase2/splits.json): task-level scene partitions.

Keep the original physics/control rates, grasp mode, action normalization, goal criteria,
safety specification, policy inference settings and native termination behaviour fixed.
Use the checked `legacy` source profile used by the current instrumented runner, and
save its effective camera dimensions and rendering settings. Resolve configuration
differences before search, rather than assuming a profile name proves image equivalence.

ManiGuard's monitor records physics-grounded specification violations; it does not
predictively change the policy's commands. The adapter proposed here is an execution
controller, not a certified safety filter. This distinction follows the
[benchmark formulation](https://arxiv.org/html/2608.17386v1).

The adapter may use proposed commands, measured arm/finger positions and previous
environment-input commands. It may not use target identity, object states, goal flags,
monitor predicates or outcome labels. Task text and images continue to enter the frozen
VLA through its original interface; Part 1 adds no separate encoder.

## 4. The four-parameter controller

### 4.1 Genome and proposed domain

Use a four-coordinate categorical genome, decoded as follows. Values are implementation
defaults to verify in engineering checks, not parameters to tune against validation results.

- `free_step_limit`: `[null, 0.12, 0.08, 0.04]` radians per joint-target displacement.
- `grasp_step_limit`: `[null, 0.08, 0.04, 0.02]` radians per joint-target displacement.
- `smoothing`: `[1.0, 0.75, 0.5]`.
- `horizon`: `[4, 8, 16]` executed control actions before the next policy query.

There are 144 possible parameter vectors. `null` explicitly disables the corresponding
cap; it is not encoded as an arbitrarily large number. The neutral genome is
`[0, 0, 0, 1]`: both caps disabled, smoothing one and horizon eight.

The current [motion adapter](experiments/helpers/maniguard_adaptation.py) requires
positive finite caps. Supporting disabled caps is a small additive Part 1 change,
with its own configuration and identity checks. Do not silently reinterpret the old
adapter's fields or change historical configurations. Other supported controls, including
settling, retries and release delay, stay disabled.

Confirm that the policy actually returns at least sixteen usable actions per chunk.
A requested horizon of sixteen must not silently become eight because the chunk is
shorter. If a domain value is unsupported, remove it and update the draft protocol
before search; keep the reason and revised cardinality. Do not replace it after seeing
comparative outcomes.

### 4.2 Action transformation

Let `p` be the proposed seven arm-joint targets, `q` the measured arm positions and
`a_previous` the previous arm command supplied to the environment. On the first action,
use `q` as the smoothing reference; afterwards use `a_previous`.

```text
reference = measured joints on the first action, otherwise previous environment input
desired   = reference + smoothing * (proposal - reference)
adapted   = desired                                  if the applicable cap is disabled
adapted   = q + clip(desired - q, -cap, +cap)          otherwise
```

The cap operates independently on each joint. It bounds a target displacement relative
to the currently measured position, not angular velocity, contact force or physical
collision risk. The gripper command passes through unchanged. Downstream action-space
and controller clipping retain their original behaviour.

Use the existing finger heuristic to select the grasp-conditioned cap: both finger
positions have absolute values in `[0.003, 0.035]`, their maximum change from the previous
action is below `0.002`, and the proposed gripper command is closing. Otherwise use the
free-motion cap. These fixed thresholds reproduce the existing adapter's interpretation;
they do not establish that the target object is held. Do not add target-held predicates.

For the neutral genome, bypass arm arithmetic and copy the proposal exactly. This avoids
floating-point round-off being mislabelled as an intervention. The neutral controller
must reproduce the original eight-action execution schedule in an engineering check.

All four values remain fixed throughout an episode. State-dependent cap selection comes
only from the causal finger heuristic. Reset history at every episode, and update
`a_previous` from the actual logged environment input after existing action-space clipping.
Do not use a future proposal or an adapted-but-never-applied command as executed history.

### 4.3 Query schedule

Query when the chosen chunk prefix has been consumed, the returned chunk is exhausted,
or the original executor requires it. Do not add tracking-triggered or monitor-triggered
re-querying in this first study. Use the same scheduling code for identity and evolved
conditions, with identity's horizon fixed at eight.

At the full horizon and with sufficiently long chunks, horizons four, eight and sixteen
need at most 275, 138 and 69 normal policy queries respectively. Actual queries can be
lower through native early termination; attempted or failed queries are still counted.
Different query counts are a measured consequence of the controller, not a hidden
information advantage. Report them alongside completion and elapsed time.

### 4.4 The user-controlled safety dial

Let `d` be a scalar in `[0, 1]`, held constant for an episode. Evolution discovers the
four-parameter anchor controller `theta`. The user chooses how strongly to apply it:

- `d = 0`: original policy commands and the original eight-action query horizon.
- `d = 0.5`: half-strength action adaptation and an intermediate query horizon.
- `d = 1`: the full evolved controller and its selected query horizon.

First compute the arm command `adapted` from Section 4.2 using the actual history of
this episode. Then apply the dial:

```text
arm_environment_proposal = original_arm + d * (adapted_arm - original_arm)
effective_horizon = floor(8 + d * (evolved_horizon - 8) + 0.5)
```

At `d = 0`, bypass the transformation exactly; at `d = 1`, return the anchor command
directly rather than introducing blend round-off. The gripper still passes through
unchanged, and the original downstream clipping still applies. Update controller history
with the final environment input after dial application and clipping. At the tested
midpoint, anchor horizons four, eight and sixteen become six, eight and twelve; verify
these schedules too. Record requested and effective horizons in the trace.

For fixed inputs, the dial scales deviation from the original arm proposal. Across
physical episodes, states and VLA observations change, so increasing `d` does not
guarantee smaller physical movements, fewer collisions or a lower violation probability.
An anchor horizon above eight also queries less frequently at higher dial settings.
The endpoints therefore mean intervention strength, not calibrated risk levels.

The term safety dial describes a safety-motivated control whose physical effect we
measure. Do not call `d = 1` certified safe, equate `d` with an allowed failure rate,
or claim monotonic safety by construction. If measured safety worsens or the curve is
nonmonotonic, that is a central finding rather than a reason to rename or reorder settings.

To keep compute small, search only at `d = 1`. Freeze the same anchor genome before
testing all three dial settings. The midpoint is a planned interpolation check, not an
independently trained policy or a setting selected after seeing validation outcomes.
This tests dial sensitivity; it does not optimize the entire trade-off curve. A later
study can compare richer conditioning or Pareto-based controller selection if justified.

## 5. Physical outcomes and search fitness

Reuse [the requested-completion derivation](experiments/helpers/maniguard_phase2_outcomes.py)
without importing the old neural actor into the runtime or its audit requirements.
Preserve both the benchmark verdicts and the supplemental outcomes.

- `C_req`: benchmark success plus the pinned family completion checklist. For Clutter,
  the final native held/intersection conditions must be true and traces must be valid.
- `V`: raw `ltl_violated`, not the engagement-gated `counted_violation` statistic.
- `S_req`: `C_req and not V`.
- Search utility: `U = int(S_req) - 2 * int(V)`.

This is the existing phase-two contract's utility: safe requested completion scores one,
safe unfinished scores zero, and either violating category scores minus two. It differs
from the legacy search utility `int(C_req) - 2 * int(V)`, which can reward unsafe
completion relative to unsafe noncompletion. Use the former consistently and record
its formula in the new protocol. This pilot uses a scalar penalty; it does not establish
an advantage for penalty-free selection.

Fitness is the mean utility over the candidate's two valid, full training episodes.
Both run at `d = 1`; the external dial is never optimized as part of candidate selection.
Each candidate in a population receives the same two scene/seed assignments. Ties use
stable birth order, with the retained elite ordered first. Do not introduce progress,
query count or intervention count as an unrecorded tie-breaker after inspecting results.

Do not convert crashes, missing traces, accounting failures, infrastructure timeouts or
short engineering checks into zero-fitness task failures. Invalid evidence has no fitness.
Stop the affected generation, preserve its costs and diagnose it. An episode that reaches
the native 1,100-action horizon without success is a valid unfinished outcome; a killed
process is not equivalent.

## 6. Evolution: a deliberately small search

Use one search seed, initially `20261018`, and a separate schedule seed, initially
`20261019`. Save the exact generator implementation and state. These seed choices are
identifiers for reproducibility, not tuned hyperparameters.

The full design evaluates populations `g = 0, 1, 2`, each of size four:

1. Start with the neutral genome and three distinct random genomes, sampled uniformly
   from the remaining declared grid. Keep this initialization rule explicit.
2. Evaluate every genome on the two declared training assignments. Audit all eight
   episodes before calling selection or mutation.
3. Rank the current population by mean utility and the fixed tie rule. Retain its best
   two genomes as eligible parents.
4. For the next population, copy the best genome once and create three mutated offspring,
   choosing each parent uniformly from those two. Re-evaluate the copied elite physically.
5. Repeat until the chosen population count is exhausted. The full design has two
   evolutionary transitions; the smaller design has one.

Start from the operators in [AdapterSearch](experiments/helpers/unisafe_search.py).
Its existing mutation kernel chooses one coordinate to mutate, considers each other
coordinate with probability 0.5, and then either resets a considered coordinate uniformly
with probability 0.2 or takes an index step from `{-2, -1, 1, 2}` with clipping.
Use that kernel with an explicit rejection of unchanged children and within-population
duplicates. After a fixed 100 failed draws, sample uniformly from unused grid points
and record that fallback. These resampling draws use CPU randomness, not extra rollouts.

Save the parents, offspring, ancestry, scores, tie decisions and RNG state after each
population. No crossover, neural weight search or additional parameter tuning is needed.
The latest re-evaluated elite can score worse than previously; noise means the curve
need not increase monotonically. Do not reuse cached fitness to force improvement.

Choose exactly one finalist from the final evaluated population using that population's
fitness and the same tie rule. Save its genome, decoded parameters and hash before
opening comparison outcomes. Do not call the old archive-wide `finalists()` unchanged:
selecting the historical maximum is a different rule and can favour a noisy old score.

## 7. Scenes, seeds and exposure

Preserve whole-task partitions from the existing phase-two split. All variants of a
base task stay together. Use only the `base` condition in Part 1.

Proposed metadata-only scene assignments are:

- Short engineering checks: `task_0040/base` from the engineering partition.
- Search: `task_0034/base` and `task_0002/base` from the training partition.
- Both comparison variants: `task_0029/base` and `task_0032/base` from validation.
- `task_0048` and `task_0050` remain outside this Part 1 comparison.

These are concrete starting assignments, not a claim that previous experimenters have
never inspected them. Before freezing, audit scene availability and the exposure ledger
without selecting scenes for their success rate. If a path is absent, use a prospectively
recorded metadata-only replacement rule within the same partition. If validation scenes
already informed controller design, label the check as fresh repetitions on exposed
scenes; do not claim unseen-scene generalization. Do not spend the reserved test split
to repair that claim.

For population `g` and training-scene index `i`, use policy/reset seed `1000 + 10*g + i`.
All four candidates in that population receive those same assignments. Fresh assignments
in later populations prevent retaining fitness on one fixed seed pair, although two
scenes still provide a very small search signal.

For validation-scene index `j` and repetition `r = 0..3`, use seed `2000 + 10*j + r`
for all three dial settings. The smaller version uses `r = 0..1`. Give every physical
episode a distinct episode ID. Verify the
policy server's per-episode random-state reset and pin its inference configuration.
Matching a scene and seed does not make rendered observations or physical trajectories
identical. Our recorded repeatability concerns still apply.

Use the independent scheduling RNG to permute candidate order within each training
scene block. For comparison, schedule all three dial settings adjacently within each
scene/repetition block. Use a seeded sequence of shuffled permutations of the three
settings, cycling through all six permutations before repeating, to balance execution
positions as closely as possible. Save the complete order before comparison starts.

## 8. Allocation and independent comparison

The full allocation is:

- Four candidates x three populations x two training scenes = **24 search episodes**.
- Three dial settings x two validation scenes x four repetitions = **24 comparison episodes**.
- Total = **48 full episodes**, at most **52,800 policy control actions** at the native
  horizon, plus initialization, engineering checks, failures and their physics.

The smaller allocation is:

- Four candidates x two populations x two training scenes = **16 search episodes**.
- Three dial settings x two validation scenes x two repetitions = **12 comparison episodes**.
- Total = **28 full episodes**, at most **30,800 policy control actions**, with the same
  additional-work accounting.

The conditions are the frozen finalist at `d = 0`, `0.5` and `1`. The zero setting is
the identity reference. There are eight comparison episodes per setting in the full
design, or four in the smaller design. No candidate or dial setting is selected using
these comparison outcomes. Once comparison begins, do not tune the controller, replace
the finalist, expand the sample
because it is nearly significant, or reclassify unfavourable episodes as engineering.

If the selected finalist is the neutral genome, report that the search selected identity.
The predefined response is to stop before the redundant three-setting comparison and
retain the search result and unspent allocation. Do not substitute a runner-up to obtain
a different-looking controller. Identity can also emerge as a behavioural near-match;
report measured activation rather than assuming a changed genome changes execution.

Native early success is allowed. Preserve the full configured horizon for every search
and comparison job, and never shorten it just to meet a throughput target. A timeout
alone does not diagnose a safety filter stall or an irreversible physical state.

## 9. How the eight-hour ceiling is enforced

### 9.1 Time estimates and the budget decision

At three to seven minutes per full episode, 48 episodes take approximately 2.4 to 5.6
hours of serial rollout time. Twenty-eight take approximately 1.4 to 3.3 hours. These are
arithmetic scenarios using old measurements, not guarantees. Moderate motion and frequent
re-querying may increase duration relative to the historical policy.

Prepare code and environments before the timed allocation where possible. The current
Vast template is only a preparation base, and the root environment does not own the
ManiGuard simulator or policy environments. A cold asset download and installation must
not be assumed to fit. See [the ManiGuard setup notes](configs/vast/README.md).

Record `T0` when GPU allocation starts, including any on-machine setup. The hard end is
`T0 + 28,800 seconds`. Reserve the last hour for saving and auditing; target completion
of physical work by `T0 + 25,200 seconds`.

Both variants share population zero's eight full training episodes. After its timing
records are complete, a timing-only selector chooses the variant before the optimizer
or a human reads its outcome scores. All eight episodes remain part of the selected
search allocation; this is not an extra profiling batch.

Use an initial conservative per-episode planning duration:

```text
t_plan = max(417 seconds, 1.25 * slowest completed population-zero process duration)
```

The measured duration must include spawn, initialization, simulation and episode audit.
Allow separately for remaining population/finalist bookkeeping. Choose 48 only if its
40 remaining episodes fit before the physical-work deadline; otherwise choose 28 if
its 20 remaining episodes fit. Save the timing inputs, calculation and selected variant.
The selector must not consume fitness or success labels. Timing can correlate with
early completion, so this rule is still part of the procedure and must be reported.

If neither fits, do not start another population. Report the completed exploratory
population and setup limitation. Do not silently substitute an unplanned tiny success
comparison. Once a variant is selected, do not expand it after observing outcomes.

### 9.2 Runtime controls

Check the budget before starting population zero, every later population and each
three-setting comparison block. Use a default 1,200-second process timeout, capped by
remaining time before the physical-work deadline. This is an infrastructure bound,
not a new native episode horizon. If observed behaviour makes it unsuitable, revise it before search
and recost the run; do not extend it after a favourable candidate times out.

Do not launch a job unless enough time remains for its timeout and cleanup margin.
Budget estimates determine whether a whole stage should begin; hard deadlines still
protect against estimates being wrong. Stop new launches when the remaining planned
work no longer fits. Preserve incomplete populations and comparison blocks without
inventing their missing outcomes.

The supervisor must terminate the simulator's process group, wait for child cleanup,
flush available ledgers and stop the persistent policy server by the hard deadline.
Use process-group ownership so this does not kill unrelated work. A graceful attempt
has a bounded grace period inside the eight hours, followed by forced termination if
needed. A fresh attempt after infrastructure failure is not automatic: it needs a new
attempt ID, retained prior costs and space in the already selected allocation policy.

Use one cumulative allocation clock across restarts. Stopping and restarting a process
does not reset the eight hours. If the rental is already running, include its allocation
from the actual start; do not begin the clock only when the first rollout starts.

### 9.3 Memory and hardware checks

A historical passive sample observed 14.3 GB total GPU memory during a serial Clutter
episode. This supports testing co-residency on one GPU; it is not a peak-memory bound.
Check actual peak memory with the policy and simulator resident, together with CUDA,
renderer and driver compatibility on the chosen 4090 or 5090.

Keep CPU thread counts consistent with available cores and container quotas. Save the
CPU model, memory, GPU, driver, environment versions and thread settings. CPU-bound
simulation means a newer GPU alone does not imply proportional speedup. No current
hardware price or rental is selected by this plan.

## 10. Implementation work in the current repository

Use new files and outputs. Reuse existing interfaces while keeping old protocols and
recorded evidence unchanged. Suggested modules are logical boundaries; combine them
when that keeps the implementation smaller without weakening checks.

1. **Pure adapter and configuration:** `experiments/helpers/maniguard_phase3_adapter.py`.
   Implement the four-parameter schema, disabled-cap semantics, dial interpolation,
   effective query horizon, causal history and exact identity path. Reuse the existing
   finger heuristic and motion equations.
2. **Runtime and checked source integration:** phase-three runtime/source helpers next
   to the existing helpers. Reuse `Study` accounting, proposal/adapted/environment-input
   telemetry and checked patch anchors. Add initial monitor recording, complete goal
   telemetry and semantic controller records needed by the new auditor.
3. **One-episode entry point:** `experiments/scripts/maniguard_phase3_episode.py`.
   Use the current family/source registry and scene-access checks, run the pinned
   evaluator and save the new job schema and source snapshot.
4. **Result audit:** a phase-three results helper. Combine physical-record checks with
   the current completion derivation and deterministic replay of this adapter. Accept
   semantic-controller episodes without requiring the phase-two actor, CLIP features
   or PPO likelihoods. Validate failure costs separately from valid outcome evidence.
5. **Pilot coordinator:** `experiments/scripts/maniguard_phase3_pilot.py` and a small
   helper for protocol, search state and scheduling. Implement the two allocations,
   timing-only selector, population audits, finalist freeze and process-group deadline.
6. **Report:** a small report entry point that reads audited immutable results and
   produces the dial/outcome curves, counts, search trace, cost ledger and matched-video
   index.
7. **Protocol:** `configs/maniguard/phase3/part1.json`, generated and frozen after
   engineering checks. It contains the complete domain, exact jobs and hashes, not
   only a prose description or a mutable path to this document.

Important integration details already identified in the existing code:

- The legacy source/policy and split maps do not contain Clutter. Use
  [the current family registry helpers](experiments/helpers/maniguard_phase2_families.py)
  and their pinned manifests rather than changing the legacy maps.
- [The old search validator](experiments/helpers/maniguard_search.py) requires evolution,
  random search and policy-gradient learning together. Add a Part 1 validator; do not
  relax that study's constraints globally.
- [The phase-two runtime](experiments/helpers/maniguard_phase2_runtime.py) loads its
  context encoder even for fixed controls, and its result reader audits a specific
  neural actor. Reuse the relevant lower-level contracts, not that entire route.
- [The base runtime](experiments/helpers/maniguard_runtime.py) constructs a global
  `STUDY` for most job phases when imported. Ensure the new route constructs exactly
  one study and trace, with a regression check for importing its base class. Do not
  accidentally start the legacy study before the new study.
- [The source builder](experiments/helpers/maniguard_source.py) uses exact upstream
  anchors. Preserve source identity checks and save both original and generated code
  hashes; do not edit the installed evaluator in place.

Do not use the frozen phase-two coordinator or old absolute stop times as the Part 1
launcher. Do not resume the 192-episode diagnostic, heritability probe or multi-method
pilot as a dependency. New dependencies are not expected; if genuinely needed, update
the intended environment explicitly and preserve its reproducibility records.

## 11. Verification before spending the rollout budget

Run meaningful CPU checks during implementation, before allocating the GPU where possible:

- Exact identity actions and query schedule, including first-action history, short
  returned chunks, episode reset and downstream action-space clipping.
- A nonidentity case that proves smoothing uses the previous environment input, caps
  use current measured joints, and the gripper is unchanged.
- Finger-heuristic switching and explicit disabled-cap behaviour, including invalid
  configuration rejection and finite output checks.
- Dial bounds, exact identity at zero, full anchor equivalence at one and midpoint
  interpolation for fixed inputs. Verify horizons six and twelve, final-command history,
  and rejection of changing the dial during a nonengineering episode.
- Seeded proposal/selection replay, real elite re-evaluation, no-op mutation handling,
  stable ties, final-population selection and exact 48/28 episode arithmetic.
- Scene/split exclusions, source pins, initial monitor sample, complete trace alignment,
  requested-outcome derivation and rejection of missing or corrupted evidence.
- The timing selector ignores outcomes, includes the shared first population once, and
  refuses allocations that do not fit. A fake subprocess tree verifies deadline cleanup
  and accounting of partial/failed jobs.

On the allocated machine, start with three engineering episodes of at most 32 control
actions each: dial zero, one-half and one for a clearly active bounded anchor controller,
on an engineering scene. This is at most 96 control actions, plus initialization physics.
These are short integration checks, not full completion results. Save their
physics, queries, time and memory in the same allocation ledger. They may verify loading,
action paths, query schedule, controller activation and required traces; they cannot
establish an end-to-end full-episode duration. Population zero supplies that timing.

Require these checks to establish engineering validity, not a controller win. If a check
fails, preserve it and repair the issue within remaining resources. Repeating a failed
attempt has a new ID and cost. Do not count repaired or short attempts as valid search
episodes. No additional unrecorded full-episode screen precedes the search.

## 12. Evidence, storage and accounting

Use a new exclusive output directory such as
`runs/maniguard/phase3-part1/<run-id>/`. This is a proposed ignored run location, not a
directory created by writing this plan. Keep data and checkpoints out of Git.

The run contains a frozen protocol and hashes, allocation clock, environment/service
receipt, engineering attempts, population plans and states, frozen finalist, comparison
schedule, all episode directories and an audit/report directory. Every attempt gets a
unique ID and explicit status: planned, started, valid completed, invalid, failed,
deadline-interrupted or not started.

Each started episode retains its decoded controller and genome hash, source snapshot,
policy metadata, exact scene/seed, dial value, effective query horizon, initial
observations, full monitor/goal traces,
proposed and anchor-adapted commands, dial-blended commands, logged environment inputs,
controller events, raw native
verdicts and derived requested outcomes. Record that an environment input is not a
measurement of low-level actuator output after controller clipping.

Every attempt also retains control actions, environment-step attempts and returns,
physics ticks and simulated seconds, policy query attempts and completed latency,
initialization time, adapter time, process elapsed time and termination status. Preserve
unknown costs and lower bounds explicitly rather than filling them with zero.

Count all four candidate slots in every population, elite re-evaluations, discarded
candidates, failed queries, engineering physics and invalid/partial work. Do not report
only retained candidates or successful rollouts. Sum interaction counts once per raw
attempt; deduplicate report copies and derived artifacts.

Report the continuous allocated GPU-hours separately from process-duration sums. The
persistent server and simulator share one GPU, so overlapping component durations are
not extra allocated GPU-hours. Idle allocation still counts. Keep checkpoint download,
service startup and unavailable-asset time visible.

## 13. Analysis and the tangible result

The primary contrast is the difference in safe requested-completion counts and rates
between `d = 1` and identity (`d = 0`), accompanied by their violating-episode counts.
The planned secondary analysis compares `d = 0.5` with those endpoints and describes
the three-point dial response. Show the four outcome categories per scene and in total.
Report counts and denominators before percentages; retain benchmark and requested-completion
labels separately. Do not select and advertise the best dial point as if it
were a separately validated controller.

Also report task completion, unfinished episodes, engagement, physical actions, queries,
elapsed time, changed-command fraction and the distribution of heuristic/cap activation.
Useful completion must accompany a favourable safety result. Lower intervention or
violation counts alone can reflect inactivity or non-engagement.

The small comparison is descriptive. Eight episodes per setting across two scenes
are not eight independent task families, and same-seed matching does not remove
rendering variability. Do not use a nominal p-value to certify the controller. Even in
an idealized independent homogeneous sample, zero violations in eight episodes only
give an approximate 95% upper rate bound of `3/8`, about 37.5 percent; repeated scenes
further limit population interpretation.

Produce these artifacts:

1. A frozen controller JSON and its genome, source and checkpoint hashes.
2. A lineage and search plot showing every measured candidate score against cumulative
   interactions, with elite re-evaluations distinguishable from new offspring.
3. A per-scene three-setting figure showing safe complete, unsafe complete, safe
   unfinished and unsafe unfinished counts, plus query and time totals. Plot safe
   completion and violations against dial strength with denominators visible. Show
   measured points; do not fit an apparently calibrated continuous risk curve.
4. A complete interaction and allocated-time ledger, including incomplete work.
5. Matched three-setting videos selected by a deterministic rule. Prefer saving every
   comparison video with the existing capture path, consistently across conditions.
   If storage prevents
   that, predeclare scene/repetition blocks before outcomes, rather than choosing only
   wins. Do not replay physics just to produce a nicer video without counting it.
6. A short written conclusion stating the observed gain, trade-off, null or incomplete
   scope, without an evolutionary-superiority or safety-guarantee claim.

For an interrupted comparison, list every planned three-setting block and its completion
state. Show all completed outcomes. A matched descriptive contrast may use fully
completed blocks, but incomplete-block observations and costs remain visible, and outcome-dependent
termination can bias that contrast. Do not present it as the complete frozen sample.

## 14. How the result directs Part 2

**A clear useful lead:** fresh safe completions improve without more observed violating
episodes, with gains not restricted to one scene or inactivity. The next allocation
tests matched random search and strong fixed settings, then isolates which controller
changes mattered. Repeat optimizer seeds before claiming an evolutionary advantage.
Ordinary learning remains necessary for the corresponding broader phase-three claim.

**A useful dial response:** the intermediate setting shows a different measured balance
of completion, violations and unfinished episodes. Replicate that response before making
a monotonicity or user risk-control claim. A flat response, worse safety at the high
setting, or a midpoint outperforming the trained endpoint is informative too. Do not
relabel the dial values or tune interpolation on these same comparison episodes.

**A completion/safety trade-off:** retain it as a trade-off. Investigate the physical
failure and whether the search objective or interface needs revision before enlarging
the search. Do not label more unsafe completions as a safety improvement.

**A training gain that disappears in the fresh check:** investigate selection noise and
scene dependence. The next useful change may be more reliable candidate evaluation,
not more generations on the same two episodes.

**No activation, identity selected, or no observable gain:** inspect whether the adapter
has a behavioural opportunity and whether it addresses the actual failure. A short
null does not prove that execution adaptation or longer search cannot work. It informs
whether to spend more on this interface or test a different, evidence-backed lever.

**Infrastructure or deadline failure:** report the validated work and exact limitation.
Do not turn missing physical evidence into an algorithmic result or compensate with
unbudgeted episodes. A later run is a new recorded allocation.

This first experiment supplies a concrete controller and evidence, or a bounded reason
to redirect effort. The route to GECCO still requires a substantive research finding,
appropriate prior methods, matched budgets and stronger evaluation. Novelty must be
checked against primary literature before making that claim.

## 15. Implementation sequence and completion checklist

Implement the pure adapter and protocol checks first. Then connect the single-episode
runtime and result audit, verify the source integration, add the small coordinator and
deadline handling, and prepare the report. Keep the CPU verification green before the
timed machine run. Validate the machine and short engineering checks, evaluate the
shared initial population, choose the size from timing, finish evolution, freeze one
finalist and run the untouched comparison allocation.

Part 1 is complete when the agreed allocation has a saved terminal report with valid
controller/source provenance, all attempted work accounted for, a frozen finalist or
explicit identity selection, and an honest three-setting physical comparison or a
documented reason it could not be completed. A positive result is not the completion
criterion. Writing this plan does not implement the runner or establish that the machine is ready.
