# Stage 1: freeze the system and qualify the adapter slot

Draft protocol, 1 November 2026. Not authorised until Sachin says so. Follows the
probes in [ProbeResults.md](ProbeResults.md) and the plan in [Roadmap.md](Roadmap.md).
Purpose: after Stage 1, nothing about the base system changes again, and we know
(a) which adapter interface to use, (b) how noisy the violation estimate is, and
(c) how fast episodes can run. Stage 1 produces no comparison result and no claim.

## 1. System freeze

Everything below is written to `configs/stage1-freeze.json` with hashes and is never
edited afterwards; later stages reference it.

| element | frozen value | why |
|---|---|---|
| task | `Isaac-Takeoff-Hard-Franka-IK-Rel-v0` with the six regular terminations restored (success stacking tolerance 0.03 m, failure displacement 0.07 m, drop height 0.13 m, boundary x in [0.3, 0.7], y in [-0.3, 0.3], timeout 300) | the variant where the base violates about half the time |
| horizon | 300 controls at 20 Hz, plus 40 continuation controls after first success | sustained success needs the continuation |
| model | UNISafe checkpoint rev 39bd3c84; filter snapshot 200000 in shadow mode (signals logged, never applied) | released system, no retraining |
| base planner | `ap_rank`: posterior mode latent; 64 nearest library steps (steps 0 and 1 of each demo excluded); anchor = feature 5 steps after the nearest step; candidates = the 5 recorded actions following each retrieved step; score = squared distance of the imagined 5th feature to the anchor; execute the best 5 actions; replan | the probe settings, untouched |
| library | `success_only.npz` sha256 66c69bc7...; 1,369 demos, 161,709 steps | authors' public demonstrations |
| violation | at any control, benchmark or continuation: middle or top block below 0.13 m, top block more than 0.07 m from the base in xy, or end-effector outside the boundary | native predicates only |
| task return J | sustained safe success (success held for 40 controls with no violation), 1 or 0 per episode; ties broken by safe success, then by shorter completion time | the thing a user cares about; the probes showed the planner loses it by moving on |
| adapter cadence | one decision per block of 5 controls, same as the planner | the planner's own rhythm |
| common random numbers | all candidates in one generation are evaluated on the same block of starts | halves the comparison noise for free |

## 2. Adapter interface qualification

Two candidates, identical inputs and information, generic initialisation, no expert
fallback, no object geometry as input.

**Shared inputs (26).** End-effector position and quaternion (7), finger positions
(2), the planner's command for this block (7), the previous executed action (7),
the planner's plan cost and retrieval distance (2), normalised time (1).
Hidden layer of 8 tanh units.

**(a) Residual on the planner command.** `a = clip(a_plan + B * tanh(u(x)), -1, 1)`
on the six arm coordinates; the gripper sign is `sign(g_plan + B_g * tanh(u_g(x)))`
with `B_g = 2`, so the adapter can hold or release against the plan. Outputs 7,
parameters 26 x 8 + 8 + 8 x 7 + 7 = 279. Bound `B` qualified at 0.25 and 0.5.
Zero weights give exactly the planner's action.

*Decision record, 1 November 2026.* Sachin accepted the recommendation to include
the gripper override. Evidence from the 20 probe episodes of the bare planner: of its
12 bad outcomes, 3 middle-block drops followed the planner's own open command within
1 to 3 controls (two of them right after success), 4 top-block drops, 1 boundary exit
and 1 closed-gripper middle drop were arm motion, and 3 lost successes were the block
drifting back inside 7 cm with the gripper closed. The gripper is necessary for about
a quarter of the violations and sufficient for none of the rest, and the October
release-delay rule showed the gripper decision is this system's historical weak point.
Gate 2 below tests the choice as a factor so the decision is checked by data.

**(b) Anchor selection.** The planner already holds 64 retrieved segments. The adapter
scores each segment from per-segment features (retrieval distance, imagined cost to
the anchor, mean action magnitude of its 5 actions, gripper sign, position within
its demonstration) with a linear score added to the planner's own cost, and the
best-scoring segment is executed. Zero weights give exactly `ap_rank`. Parameters
about 6 to 10. Cannot leave the demonstrated behaviours, which is both its safety
and its limit: it cannot stand still after success unless a demonstration did.

**Gate 1, parity (6 starts per interface, 12 episodes).** Zero-weight adapter versus
bare `ap_rank` on the same starts: identical executed actions, camera hashes,
physics counters and outcomes at every control. Confirms the wiring changes nothing
when the adapter is silent.

**Gate 2, authority sweep (24 genotypes x 4 starts per interface, 192 episodes).**
Genotypes drawn from the generic initialisation (weights ~ N(0, 0.5)). For the
residual, the 24 genotypes are split two ways in a 2 x 2 design: 12 with gripper
authority (`B_g = 2`) and 12 arm-only (`B_g = 0`); within each half, 6 at `B = 0.25`
and 6 at `B = 0.5`. Pre-registered gripper rule: keep the gripper override if the
gripper half has fewer middle-block drops than the arm-only half and no more violation
episodes overall; otherwise Stage 2 uses the arm-only residual. Requirements for the
interface as a whole, judged on the same 4 starts as bare `ap_rank`:

- at least one genotype completes a start the base does not, and at least one fails
  a start the base completes (outcomes move both ways);
- fewer than 90% of sweep episodes violate (not degenerate);
- both successes and violations appear across the sweep (an optimiser would have
  signal);
- the sustained-success rate across the sweep is not identically zero.

**Decision rule.** If both interfaces pass, take the residual (reusable across tasks,
matches LeWM's Candidate B) and keep anchor selection as a Stage 3 ablation. If only
one passes, take it. If neither passes, widen the bound once (0.75) for the residual
and re-run Gate 2 once; report if that also fails.

## 3. Noise calibration (48 fresh starts, 48 episodes)

Bare `ap_rank` on 48 fresh development starts. Report the violation rate, the sustained
and plain success rates, Wilson intervals, and the per-start pattern (which starts
violate under the planner, how consistently). Then compute, for 8 episodes per
candidate, the chance that a candidate at the base rate looks feasible at each
candidate limit (for p = 0.5 and limit 0.25 that chance is about 14%, which is the
noise the ranking rule must survive). Choose the two Stage 2 requested limits from
this table before any adapter is trained: one about half the base rate, one about a
quarter, provided the quarter is not below what any sweep genotype achieved.

## 4. Throughput lever: in-process resets (12 episodes)

Today every episode is a new simulator process: 16 s start, about 10 s of episode.
Keeping one process and calling `reset(seed)` between episodes would roughly halve
the cost of Stage 2. The October study found that preceding trajectories change
renderer history, so bitwise identity is unlikely.

Test: 6 starts run fresh and run as the 2nd to 7th episode of one process. Compare
per control: executed actions, physical state arrays, camera hashes, outcomes.

- Identical: use in-process resets everywhere.
- Not identical but outcomes agree on all 6 and action differences are at
  floating-point scale: use in-process resets for training evaluations, fresh
  processes for selection and final panels, and say so.
- Outcomes differ: fresh processes everywhere, Stage 2 budget doubles.

Vectorised environments (several arenas in one process, planner batched across
them) are the second lever and are not part of Stage 1 unless the first fails.

## 5. Seeds, budget, outputs

- Seeds: 4,300,000 to 4,300,005 (parity), 4,310,000 to 4,310,003 (sweep starts),
  4,320,000 to 4,320,047 (calibration), 4,330,000 to 4,330,005 (reset test). All
  development seeds, excluded from later panels.
- Episodes: 12 + 192 + 48 + 12 = 264, about 2 lane-hours fresh-process, under $1 of
  host time on instance 58666468. Engineering is the real cost: about a day.
- Outputs: `configs/stage1-freeze.json`, `outputs/stage1-20261102/` with every
  episode's trace, and `Stage1Results.md` with the two gates, the gripper rule's
  verdict, the calibration table, the chosen interface and bound, the chosen limits,
  and the reset decision.

## What Stage 1 must not do

No optimiser runs. No planner setting changes. No look at the main-study seed ranges.
No adapter weight is kept beyond the sweep records.

## Amendment proposed after the sweep (1 November, evening; not yet authorised)

The four sweep starts all turned out easy for the bare planner (4/4 completions, 0
violations), so the "completes a start the base fails" criterion could not be tested
on this panel. The residual at bounds 0.25 and 0.5 with N(0, 0.5) initialisation
completed 1/96 and violated 85/96: destructive authority, with boundary exits and
top-block displacements dominating. The pre-registered contingency (widen the bound
to 0.75) points the wrong way and is not run. Proposed Gate 2b instead:

- Panel: the first four calibration starts, in ascending seed order, where the bare
  planner violated under the frozen definition: 4,320,001, 4,320,002, 4,320,006,
  4,320,009. The bare results on these starts already exist.
- Selection: the same 24 genotypes as Gate 2, to test "completes where the base
  fails" (96 episodes).
- Residual with reduced authority: bound 0.1, initialisation N(0, 0.2), 12 genotypes
  with gripper authority and 12 arm-only (96 episodes). Corrections are then about
  0.03 to 0.06 per coordinate, below typical demonstration actions.
- Gate criteria unchanged. Interface rule unchanged. 192 episodes, about 65 minutes
  on two lanes.

The residual's generic initialisation scale and bound become part of the freeze once
chosen; the Gate 2 result at 0.25 and 0.5 is reported as a negative finding either way.
