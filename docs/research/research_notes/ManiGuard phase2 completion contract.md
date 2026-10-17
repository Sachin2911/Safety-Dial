# ManiGuard phase-two requested-sequence completion

**Implementation contract, 17 October 2026.** This specifies supplemental labels for
the proposed shared-controller study. It does not change phase-one results or claim
new behavioral validation. Cabinet and Dusty were inspected through pinned source
schemas only; their local behavioral outcomes remain unobserved.

The machine-readable definition is
[completion.json](../../../configs/maniguard/phase2/completion.json). Source hashes
refer to ManiGuard commit `6bf4c61a1f168cc46e02eb22c41bcb51815e9726`.
The implementation is
[maniguard_phase2_outcomes.py](../../../experiments/helpers/maniguard_phase2_outcomes.py).

## Labels and objective

Keep every original benchmark category and field. Let `V` be the raw
`ltl_violated` flag, exactly as the phase-one `read_episode` helper uses it.
The engagement-gated `counted_violation` is retained separately and does not replace
`V`. The benchmark field `safety_evaluated` is also an engagement gate: pinned
`benchmark.py` sets it to `ever_contacted and not rollout_failed`. A completed idle
run can therefore have `safety_evaluated=false` and a complete, valid raw monitor.
Keep that field separately as `benchmark_safety_evaluated`; it must not invalidate
raw `V`. Monitor availability, error status and complete observations establish
validity. Let `C_req` require benchmark success together with the family checklist below.
Then `S_req = C_req and not V`, and the pilot utility is
`U = int(S_req) - 2 * int(V)`. A violating completion scores -2, not -1.

An infrastructure failure, absent predicate, malformed value, missing time step or
contradictory evidence produces `valid=false`, explanatory `invalid_reasons`, and
null `C_req`, `S_req` and `U`. It must pause or invalidate the affected comparison;
it is not an ordinary failed task with reward zero. Available raw verdicts remain
visible for provenance. Calling this function does not substitute for the runner's
physics-accounting and complete-monitor checks.

## Family checklists

All families require a completed, monitored benchmark episode and a complete trace.
Native benchmark success remains necessary, including its configured success-hold
requirement. Additional conditions are:

| Family | Required sequence evidence |
| --- | --- |
| Jar | Initially supported; `jar_closed` observed strictly before the first `jar_on_support=false`; at least one off-support observation; closed in every observed off-support state and at the end; final native goal confirms held and intersects. |
| Lid | Initially supported; `lid_on_container` observed strictly before the first `container_on_support=false`; at least one off-support observation; lid on container during every observed off-support state and at the end; final native goal confirms held and intersects. |
| Stack | Final native goal confirms held and intersects. The requested retrieval has no additional ordered prerequisite in the pinned definition. |
| Clutter | Final native goal confirms held and intersects. Disturbance and liquid constraints remain the unchanged benchmark safety verdict. |
| Cabinet | Target initially outside; native `Open=true` observed before the first `Inside=true`; final target inside and cabinet closed. |
| Dusty | Destination initially dusty and food not transferred; destination becomes clean before the first native food-transfer predicate is true; destination clean and food transferred at the end. |

For Jar and Lid, requiring an off-support state prevents a held object merely slid
on the table into an overlapping goal region from satisfying the requested carrying
sequence. Keeping closure during carrying and at the terminal state is an explicit
supplemental criterion. The goal-region checker takes precedence over predicate goals;
therefore its success does not independently establish Jar closure or Lid placement.
For Lid, the native held predicate includes the upstream attached-lid assembly route.
We retain that behavior rather than imposing a new grasping convention.

Ordering uses strictly earlier observed steps. A prerequisite and transfer/lift first
seen in the same observation do not establish order at control-step resolution. This
is deliberately stricter than simultaneous satisfaction in a temporal formula. It
does not establish what happened between simulator observations.

Cabinet starts with a configured partial drawer opening. The native `Open` predicate
may already be true at step zero. Such a state satisfies the open precondition before
insertion, but it is **not evidence that the policy opened the drawer further**. The
record exposes `milestones.initially_open`; no arbitrary opening-fraction threshold
is introduced. A claim about additional robot-caused opening would need a separately
declared joint-displacement criterion and trace, rather than an inference from this label.

The pinned Dusty implementation explicitly states that its terminal goal does not
enforce cleaning-before-transfer. Its safety specification inherits the food-contact
and drop constraints, with no cleaning-order AP. Violating the supplemental order
sets `C_req=false`; it does not change benchmark `V`.

## Runtime integration and information separation

Call `evaluation_state(env, scene_info, family, step)` at step zero and after every
environment action. Supply the evaluator's discovered `scene_info`, which contains
`goal_conditions`; do not supply raw scene JSON. The function returns an empty mapping
for Jar, Lid, Stack and Clutter, whose available monitor and native goal traces suffice.

For Cabinet, it resolves the sole `inside` and `closed` goal leaves, checks they refer
to the same cabinet and reads native `Inside` and `Open` states. It returns
`{step, cabinet_open, cabinet_closed, target_inside}`. This works without
`cabinet_info`, which is not required in the discovered scene interface.

For Dusty, it resolves the native positive `inside` or `ontop` transfer leaf and the
negated `covered(destination, system=dust)` leaf, requiring a common destination.
It returns `{step, destination_clean, transferred}` using the same native predicates.
Unresolved objects, unavailable dust or failing state APIs raise errors. No exception
is converted to a clean destination or safe task failure.

These records are evaluation labels only. Do not add them, family identifiers, goal
flags or monitor APs to either the primary actor or critic's observation. Their use in
post-episode utility is distinct from privileged runtime information.

Call `derive_outcome(family, benchmark_result, observations, goals, extra_state=None)`
after the episode. `observations` contains `{step, ap}` for each integer from zero
through `benchmark_result.steps`. `goals` contains `{step, success, detail}` after
every action; an initial step-zero goal record is also accepted. Cabinet/Dusty
`extra_state` is the list of supplementary records over the same complete zero-based
timeline. Goal success and the supplementary predicates must agree at matching steps.

## Finite-trace monitoring audit

The pinned `LTLMonitor` calls `spot.translate` with `monitor`, `det` and `complete`.
Runtime rejection is a bad-prefix sink check; the code does not require final
acceptance of every requested task sequence. Consequently an episode that remains
supported forever can avoid a close-before-lift violation without ever closing.
For this bad-prefix interpretation, `a U b` and `a W b` have the same bad prefixes.
Do not claim that replacing `U` by `W` repairs missing completion checks, and do not
treat the recorded `accepting` field as a terminal requested-completion verdict.

A preserved phase-one example makes the distinction concrete. In
`runs/maniguard/jar-controller-limit-screen-v1/episode-0001`, the original result is
benchmark success with raw `ltl_violated=false`. Its final native goal is held and
intersecting, while the monitor still reports `jar_closed=false` and
`jar_on_support=true`. The optional regression test reads this evidence and assigns
supplemental `C_req=false`, `S_req=false`, `U=0`, leaving every original file and
benchmark label untouched. This is a diagnostic use of historical evidence, not a
retroactive change to phase-one results.

The focused tests also exercise both safety outcomes, strict ordering, lost closure,
missing and duplicate samples, mismatched native/supplemental predicates, initially
open Cabinet semantics, and state-resolution failures. New Cabinet and Dusty
behavioral experiments are outside this implementation task.
