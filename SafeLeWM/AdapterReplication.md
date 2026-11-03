# TwoRoom adapter replication

Prospective protocol, 1 November 2026. Sachin approved the recommended replication
and instructed the agent to run it. Use the existing eight-4090 Vast instance
58676387, machine 19781, at $3.6637/hour. Rental shutdown remains manual. All new
code, configuration and evidence live under SafeLeWM; earlier studies stay unchanged.

## Question and scope

Does the corrected recurrent adapter's safe-completion gain persist across three
fresh training repeats and twelve new layouts? Keep the pilot's architecture,
information, frozen AP assets, passive monitor and safe-completion-first objective.
This broadens the pilot while matching actual training interactions. It is not an
exact reproduction of the pilot's episode allowance. PPO, task-only ablations,
architecture changes, other environments and the safety dial remain outside scope.

The three new repeats are the primary results. The earlier successful pilot is
development evidence and is not pooled with them. No retraining or extra search
will follow unfavorable outcomes under this protocol.

## Frozen inputs

`configs/adapter-replication/panel.json` retains the same eight training layouts
and planner seed 8201. Its twelve new evaluation layouts comprise six open and six
wall-constrained scenes, balanced between rooms and travel directions. They vary
start/goal coordinates, hazard position and proximity to outer/internal walls.
They are not duplicate or exact mirror pairs. Doorway/cross-room tasks remain excluded.

All evaluation geometries are fixed before outcomes. Pure geometry checks and one
safe and one risky scripted witness per layout establish feasibility. A failure
stops the study without replacing cases or consulting AP outcomes. Witnesses use
privileged geometry and never train the adapter. Native and learned evaluation
outcomes remain unopened until all six search nominees are locked.

Each final policy receives all twelve layouts with planner seeds 9401/9402/9403.
These are new to adapter search, not claimed absent from LeWM pretraining. The four
previous pilot evaluation layouts are now development evidence and are not reused.

Three paired repeats use initialization seeds 9101/9201/9301, untrained-draw seeds
9102/9202/9302, CMA seeds 9103/9203/9303 and random-search seeds 9104/9204/9304.
Within each repeat both optimizers share the same generic neutral network and
coordinate scaling. Each repeat gets its own untrained reference. Training planner
randomness and the training panel remain common, isolating initialization/search
variability. No successful pilot weights are used for initialization.

## Actor and optimizers

Reuse the corrected 19,172-parameter actor: learned 32-unit feature layer, 32-unit
GRU, gain/residual heads and the same 396-coordinate causal observation binding.
All weights are searchable. Neutral heads recover native AP; action clipping is
the legal-action bound, not a safety veto. Position and supplied hazard geometry
remain available; true goal coordinates, witness paths and outcome flags do not.

Diagonal CMA-ES uses pycma 4.4.4, explicit cumulative step-size adaptation, no
mirrored sampling, population eight and initial sigma 0.5. Preserve ask order
through tell. Parameter random search uses independent draws from the same initial
distribution. For a complete eight-case candidate, maximize the unchanged exact
lexicographic score `81*safe + 9*complete - violations`. Exact ties remain equal.

The generation count is determined by the action budget rather than fixed at four.
Optimizer stopping suggestions do not replace the fixed allowance. Only complete
generations update CMA; never impute missing scores or pass a partial generation
to tell. Fully evaluated candidates in a final incomplete generation remain
eligible for selection. Choose the best observed complete candidate per search,
breaking exact ties by earliest candidate ID, never the unevaluated mean.

## Exact action budgets and deterministic scheduling

Each optimizer in each repeat executes exactly 40,000 training simulator actions,
including discarded candidates and any budget-truncated episode. Total: 240,000.
Do not pad successful episodes or repeat actions merely to consume the allowance.
An episode ends on native termination or at 200 actions. A new candidate is sampled
while budget remains. Every charged action corresponds to one attempted env.step;
software failure stops the study and is separately accounted, rather than silently
granting a replacement allowance.

Reserve at most 1,600 actions per candidate before parallel dispatch (eight cases
times 200). Dispatch as many full candidates as the remaining allowance safely
covers, in fixed candidate order. Refund unspent reservations after the whole
dispatched group returns. If fewer than 1,600 actions remain, dispatch one candidate
with that exact remaining quota. This last candidate may finish its panel early,
permitting another candidate, or stop at the budget. No timing-dependent shared
counter decides which candidate gets the final actions. Device assignment remains
candidate index modulo eight, with paired evaluation device assignments across arms.

Retain partial episode traces and partial candidate results, but exclude partial
panels from scoring and selection. Distinguish budget truncation from native failure
and the normal 200-action cap. Record per-dispatch reserved/actual/refunded actions,
per-search final spending, candidate eligibility and completed optimizer updates.
Report actual model queries, elapsed time and candidate counts separately.

## Engineering and final evaluation

On the rented host, test full-dimensional optimizer behavior, quota allocation,
refunds, exact boundary termination and exclusion of incomplete panels. The new
worker preserves the qualified episode implementation except for the budget hook
and explicit episode status. Freeze sources and dependency bindings before resets.

Engineering: native/neutral parity on eight training layouts (16 episodes), first
native layout repeated on the other seven GPUs, then one-action and seven-action
capped native probes whose prefixes must exactly match the full native trace.
This is 25 episodes, at most 4,608 actions. Reserve one additional failed-start
episode/200 actions; any failure stops for diagnosis before proceeding.

Feasibility witnesses: 24 episodes, at most 4,800 actions. Final evaluation: native
once (36 episodes), plus untrained/CMA/random for each of three repeats
(324 episodes). Total final evaluation is 360 episodes, at most 72,000 actions.
The native observations are shared comparisons, not 108 independent baseline trials.

Training plus engineering, witnesses and evaluation has an absolute 321,608-action
cap and a two-hour execution deadline. Setup, saved-evidence auditing and transfer
are separate; no automatic rental shutdown is installed. Run order is CMA/random,
random/CMA, CMA/random across repeats. The same host, frozen assets and eight
persistent workers serve all methods.

## Analysis and closeout

Primary endpoint: safe completion per training repeat on all 36 evaluation episodes.
Report all completion, violations, unfinished episodes and engagement together,
with per-layout results. Report every repeat; do not select the best training seed.
Show paired differences from native and random search, including the actual
completion tradeoff. There is no post-hoc binary success threshold or automatic
extension. All-three, two-of-three and inconsistent gains are reported as observed.

As an exploratory uncertainty summary, use 2,000 crossed cluster bootstrap draws
(seed 9501), resampling the three training-repeat IDs and twelve layout IDs while
keeping each layout's three planner seeds together and all arm comparisons paired.
Intervals with three learner repeats remain limited; do not treat 108 outcomes as
independent tasks or claim systematic superiority over untested RL/fixed references.

Audit source bindings, exact training quotas, incomplete-panel exclusion, native
completion from states, swept hazard contact, native/neutral parity, actor replay,
optimizer states and selection timing. Preserve all evidence privately, verify each
archive member locally, pin selected checkpoints in a private Hugging Face repo,
then add a concise account and graphics to Rule43. No further experiments follow
from completion of this protocol.
