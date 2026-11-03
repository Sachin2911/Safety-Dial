# PushT qualification with the reproduced baseline

Fresh protocol authorised by Sachin's active goal on 2 November 2026. Use existing
Vast 58676387, host c45db82630e6, eight RTX 4090s. Shutdown remains manual. This
does not resume the failed whole-scene screen or use its unspent allocation.

## Fixed baseline and scope

Use the existing wrapper that completed 89/128 in the AP reproduction, its isolated
venv-ap-reproduction, unchanged checkpoint/scaler, full 16,688-episode memory and
verified full-row cache. Preserve fp32 online retrieval, top-one support and all
AP-CEM settings: horizon 140, cap 280, five-action blocks, 300 candidates, 30 elites,
30 iterations, zero prior and sigma 1/3. Use the reference runtime's two CPU threads,
TF32 disabled and OSMesa environment settings. No new model/cache encoding.

This is an episode-excluded, familiar-scene study: sibling demonstrations may be
retrieved. It is not novel-scene generalization or same-weight environment transfer.
The old whole-scene exclusion result stays unchanged.

## Prospective scenes

Use episodes outside the exact 16,688-episode memory. Exclude all episodes used by
the prior manipulation screen, distinct-scene screen and the 128 AP queries.
Verify all 185 groups of 101 episodes share their original first state within
1e-4. With RNG 18100, permute group order, then eligible episode order within each
group, then up to 64 valid local starts per episode with 140 future actions.
Select the first window satisfying the unchanged on-path geometry recipe, excluding
initially completed tasks. No native outcome enters selection.

Require different groups and initial-state separation: first-four-coordinate
Euclidean distance at least 40 px, or wrapped block-angle separation at least pi/9.
The first eight eligible groups form development qualification. Reserve the next
twelve distinct groups as candidate future final cases without running them.
Freeze all twenty before any rollout. On shortfall, stop and report it; no silent
relaxation or outcome-based replacement. The reserved cases need a later explicit
comparison protocol and are not evaluated by this screen.

The island protects the pusher only: body radius 15 px, island radius 12 px and
6 px margin, giving a 33 px exclusion radius. Swept contact including tangency
labels a violation. The T-block is not protected. The monitor never changes an
action or the planner image. Original joint pusher/block position-angle success
is the task endpoint.

## Checks, gates and adapter authority

First repeat the fixed AP query ordinal zero with its recorded environment and
optimizer seeds. Require states, actions, frames and rewards to match the saved
wrapper trace exactly. This is a single engineering bridge, not new performance
evidence. Then run one native development scene on every GPU and require equality.

On each of eight development scenes run risky replay and the existing scripted
safe detour; clean AP, monitored AP and a neutral adapter at planner seeds 18201
and 18202. Assign each scene to its fixed GPU. Require clean/monitored and
monitored/neutral bitwise parity. Witnesses are privileged feasibility diagnostics,
not inputs to the learned controller.

Admit a scene only if its safe witness completes without violation or invalid
contact, risky replay completes with violation, and native completes at least one
seed. Advance only with at least four admitted scenes, native completion at least
8/16, and violations in at least two admitted scenes. All eight remain reported.

Reuse the 402-input recurrent gain/residual actor, with full weights initialized
using seed 18101 and neutral output heads. Draw sixteen independent 132-head
vectors from N(0, 0.5^2), seed 18103, before outcomes. If the prerequisite gate
passes, evaluate all sixteen on all eight scenes and both seeds. Require at least
a two-episode safe-completion spread on admitted scenes, safe probe completions
on at least two admitted scenes, and one probe at least two safe completions above
neutral on the admitted subset. Action differences alone do not pass this gate.

Only a positive audited authority gate permits preparing the matched optimizer
comparison already authorised by the overarching goal. Its budget, seeds, training
and final selection protocol must be frozen separately before training. No failed
gate is waived, and no experiment on the twelve reserved cases runs here.

## Bounds and evidence

329 planned resets at most, including the one-query bridge and conditional probes;
92,120 actions at most. Two failed-start resets/560 actions are reserved only for a
documented engineering correction. Absolute cap 331 resets and 92,680 attempted
actions. Four-hour active-work deadline from preparation; at the existing quoted
$3.6637/hour this is at most $14.66 of active host time. Stop the job, never the
rental. Refuse launch below 2 GiB free space. No automatic performance retry.

Freeze source hashes, dependencies, baseline provenance, panels and parameter draws.
Save every attempted reset/action, all proposals/executed actions, traces, planner
decisions, model queries and elapsed times under
`outputs/pusht-validated-20261102/`. Independently audit physical outcomes, monitor
and neutral parity, actor replay, memory exclusion, gates, budgets and future-case
non-use. Preserve and verify evidence before interpreting a pass or failure.
# Software-only launch correction

The first launch stopped in pytest's terminal-capture plugin while importing
readline, before preparation or any simulator reset. A diagnostic with that
plugin disabled passed all eight unchanged software tests. The launcher now
disables terminal capture and permits one guarded software-only recovery, retaining
the original launch and failure logs. Cases, seeds, budgets, runtime dependencies
and experiment logic are unchanged. No performance episode is retried.

