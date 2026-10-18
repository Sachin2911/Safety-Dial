# Phase 3, Part 2: correction-triggered policy refresh

**Date:** 18 October 2026.

**Status:** proposed 16-episode mechanism experiment. This document saves the agreed
next step; controller details and the execution protocol still need freezing before
physical work. No Part 2 episodes have been launched. The completed Part 1 protocol,
controller and evidence remain unchanged.

This follows the recommendations in `rec.md`, the
[Part 1 investigation](docs/research/reports/ManiGuard%20Part1%20failure%20investigation.md)
and the broader research direction in [phase3.md](phase3.md).

## 1. Question and motivation

> Does requesting fresh policy feedback after correcting an action improve safe
> completion compared with the same motion restraint and fixed query intervals?

The frozen VLA predicts a chunk of future commands. If our execution controller changes
one of those commands, the remaining predictions may no longer fit the resulting state.
The hypothesis is that the amount of correction can help determine when to request a
fresh chunk. This is a mechanism to test, not an established explanation of Part 1.

Part 1's full-strength controller recorded 4/8 safe completions and 2/8 violating
episodes, against identity's 5/8 and 0/8. Its dial changed both motion restraint and the
query interval. The next experiment separates those effects. See the
[original report](docs/research/reports/ManiGuard%20evolutionary%20safety%20dial%20pilot.md).

## 2. Main recommendations

- Coordinate action corrections with fresh policy feedback.
- Include a strong fixed-frequency reference to distinguish useful timing from simply
  querying more often.
- Check future search finalists on more episodes before selecting a deployed controller.
- Find a useful controller endpoint before spending more episodes characterizing the
  safety dial. Stronger intervention does not establish greater physical safety.
- Defer continuous safety-margin instrumentation and further optimizer comparisons
  until the mechanism evidence justifies their cost.

## 3. Proposed allocation: 16 full episodes

| Condition | Execution rule | Purpose | Episodes |
| --- | --- | --- | ---: |
| A: identity | Original commands; query every 8 actions | Baseline | 4 |
| B: fixed restraint, interval 8 | Apply restraint; query every 8 actions | Isolate restraint | 4 |
| C: fixed restraint, interval 4 | Same restraint; query every 4 actions | Control for more frequent feedback | 4 |
| D: correction-triggered refresh | Same restraint; query after sufficient correction, within 4 to 8 actions | Test feedback timing | 4 |
| Total | Four matched blocks, one episode per condition in each block | Bounded mechanism screen | 16 |

Use two preselected Clutter training scenes with two fresh seed repetitions per scene.
Freeze the scene IDs, seeds and interleaved condition order before new outcomes are
inspected. Keep reserved main-study scenes untouched and do not tune against Part 1's
exposed comparison scenes. Matching scene and seed does not guarantee identical camera
images or policy proposals; preserve initial observations and policy seed receipts.

Retain the native **1,100-action horizon**, unchanged completion rules and raw physical
safety monitor. The sixteen full episodes permit at most **17,600 policy control
actions**, plus explicitly accounted initialization, engineering checks and failures.
Do not add episodes based on outcomes or shorten horizons to fit the clock.

## 4. Controller and causal feedback

Conditions B, C and D must use exactly the same motion-restraint rule. Fix its caps,
grasp conditioning and smoothing before the comparison. The known grasp-detection gap
and possible direction-preserving restraint are separate design issues: any change
adopted here must apply equally to all three restrained conditions.

Condition D accumulates differences between proposed arm commands and the actual
environment-input actions recorded after restraint and downstream clipping. Reset the
accumulator on each new policy query. After at least four executed actions, request a
fresh chunk if the correction threshold is reached; otherwise request one after eight.
This bounds its query frequency between the two fixed references. An episode may finish
before either boundary.

Use only causal command history and permitted robot measurements. Never use an action's
future outcome to decide whether to execute it. Command correction measures departure
from the proposal; it does not measure actual motion or predict collision or safety.
Keep object-held labels and monitor outputs outside the execution controller.

Before implementation and physical execution, specify and freeze the correction norm,
normalization, units, threshold, numerical tolerance, reset semantics and scheduling
precedence. Freeze the common restraint settings and failure/timeout accounting too.
Choose these from declared training/development evidence, without tuning on the new
comparison outcomes. Verify exact identity for A and causal scheduling for D with CPU
checks and recorded-input replay before bounded physical engineering checks.

## 5. Evidence and interpretation

Report safe requested completion, raw physical violations and unfinished episodes
together, including all four completion/violation combinations. Keep per-scene and
per-block counts alongside aggregate counts.

The comparisons answer different questions:

- B versus A: does the selected restraint help at the original query interval?
- C versus B: what changes when fixed feedback becomes more frequent?
- D versus B and C: does correction-triggered timing offer a useful outcome/query-cost
  trade-off relative to fixed schedules?

Record every policy query and its latency, simulator interaction, correction and query
trigger. Preserve proposed, adapted and executed actions separately, plus matched videos,
event pictures, outcome figures and a complete interaction/time ledger. Report query
resources explicitly; these conditions do not have equal query counts by construction.

Four episodes per condition provide a descriptive mechanism screen. They cannot establish
a low violation probability, broad generalization, monotonic dial safety or evolutionary
superiority. Preserve adverse, null and incomplete findings. This allocation contains no
new evolutionary search or three-setting dial comparison.

## 6. Resources and subsequent work

Use the existing validated single-GPU ManiGuard stack and frozen Clutter policy, with
one policy server and one serial simulator worker. Preserve the source/checkpoint pins
and explicit uv-managed simulation and policy interpreters from [Part 1](phase3Part1.md).

The original continuous allocation clock starts at **2026-10-18 16:00:37 UTC**. Its
physical-work target is **2026-10-18 23:00:37 UTC** and hard deadline is
**2026-10-19 00:00:37 UTC**. Count setup, implementation, checks, failures, reporting and
idle allocation time; Part 2 does not reset the clock. Check remaining time before
launch and supervise experiment processes independently. Preserve incomplete evidence
if the work cannot finish within the existing limits.

Keep the rented instance running as requested by the user. This is separate from
enforcing experiment-process deadlines and does not expand the episode allocation.

A subsequent, separately specified experiment can evolve a supported controller and
confirm finalists more thoroughly. The broader 48-episode search recommendation is
not included in this sixteen-episode allocation. Matched random search, ordinary
learning and fresh independent evaluation remain necessary for an eventual claim
about evolutionary advantage.
