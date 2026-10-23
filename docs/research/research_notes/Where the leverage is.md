# Where the leverage is

Sachin Mohan, 23 October 2026. Todo 2, item 8. Evidence: the 104 Jar episodes of record
(dataset `Sachioster/maniguard-jar-replication-20261022` at `9300cebf`) described in
`progress/receipts/02-timelines.md`, `02-task-notes.md`, `02-sensitivity.md` and
`02-pairs.md`, with the frame sheets under `02-frames/` and the pair strips under
`02-pairs/`. Nothing here was run; this note maps what the episodes show onto the
intervention families listed in `progress/02-after-replication.md` and ranks them.
Panels are the frozen ones in `progress/panels.json`.

## What the episodes show

1. **One approach, then a draw.** All 103 engaged episodes approach the same way: over the
   jar, down with the fingers open, lid at the right finger. The two same-seed repeats
   touch the jar a median of 12 steps apart and their main videos are no more different
   during the approach than at step 0 (pre-contact ratio at or below 1.0 in 20 of 26
   tasks). They diverge at or after contact: propositions first differ at a median of 358
   steps, pixels at 543. Eight of 26 tasks agree on the story across the same-seed pair;
   one agrees across all four runs.
2. **The funnel.** 103 touched the jar, 70 closed the lid (median 206 steps after
   contact), 42 lifted it, 31 reached the goal, 24 safely. Only 5 of 42 lifts failed.
   Lift to success is short (median 83 steps, max 179). The losses are in the two phases
   before the lift: the lid push and the regrasp.
3. **Where the 73 failures go.** 18 `stall` (contact, then a hover at the open lid or a
   retreat, no closure, 889 to 1,895 steps left unused), 18 `closed_stall` (lid closed,
   then no lift with 1,242 to 1,757 steps left), 19 `tip_after_close_on_table` (lid
   closed, jar knocked over during the regrasp, median 209 steps after closure), 9
   `tip_before_close`, 3 `lift_open`, 5 `carry_fail`, 1 `no_contact`. Tips are 35 of the
   43 violations; 28 of them happen with the jar on the table.
4. **What differs between a success and a failure of the same task.** In the 16 pair
   strips the difference is visible before contact in 2 (both different-seed pairs on pool
   tasks), at or during the lid push in 6, in the regrasp in 7, nowhere in 1. In the
   same-seed pairs the regrasp dominates (6 of 10): the gripper leaves the closed lid for
   a top-down view of the jar, then either descends onto the body (lift), drifts away or
   never descends (closed stall), or lands on the rim or sweeps low (tip). Where the push
   is the difference, the failing episode pushes at a lid that does not go over for
   hundreds of steps.
5. **Systematic tasks.** Under the coarse outcome, 8 tasks are the same in all four
   runs: 7, 18, 21 tip within 49 to 168 steps of contact in every run in the same posture;
   25 tips within 17 steps in the same-seed pair with frame-identical videos; 24 tips
   after a minute of sweeping among three objects; 4 and 19 close the lid and retreat in
   every run; 12 succeeds in every run. The development panel has none of these; it is
   knife-edge in 5 of 6 tasks (0, 3, 6, 9, 17) and `mixed` in one (15).
6. **Two loopholes.** Six of seven unsafe successes lifted the jar open and carried it
   into the sphere (the goal check needs a grasp and the sphere, not closure). One safe
   success (seed0_run2 task 17) reached the goal with no closure or lift observed. The
   monitor's `jar_closed` is a 5 percent hinge band and flaps in 19 episodes; two
   same-seed pairs (0, 8) show a lid lying on the jar in both repeats but closure
   observed 770 and 135 steps apart.

## Family by family

**Observation-side noise or augmentation.** The approach is already the same across
seeds and repeats, so noise aimed at producing a better approach has nothing to select
between; the first divergence is physical, at the lid. The one place observation noise
could plausibly act is the stall: 36 episodes sit in one posture for 900 to 1,900 steps
(tasks 16, 22, 23 hover at the lid in three runs each; 4 and 19 retreat in every run), and
a perturbed observation might move a policy that is at a fixed point. The same
perturbation applied throughout would also act during the regrasp descent, where 19 tips
already happen, so it would need gating by phase. Leverage: possible on stalls (35
percent of episodes, 7 of 24 development-panel episodes), none visible on tips. Needs:
nothing new to run, but the stall must be detected from the monitor's propositions (no
change for N steps) to gate it.

**The policy's sampling noise or latent.** Repeating a seed, which repeats every
episode seed exactly, is no more reproducible than changing it (13 of 26 same-class
agreements against 11 to 15 for different seeds), and the same-seed pairs are
frame-identical until the lid. So the sampler's draw is not what decides these
episodes; physics at contact is, and it re-randomises the policy's input afterwards. A
noise scale cannot be shown to have leverage from this data, nor shown not to: it would
change the rate of the push and regrasp outcomes in a way only a sweep can measure, at
the 200-episode-per-setting cost of the noise floor. It remains the cheapest lever to
sweep because the hook exists. Leverage: unknown, no supporting evidence in the
episodes. Needs: the sweep itself; the per-step recorder to say afterwards which phase
the scale changed.

**An adapter on the policy.** Would need a training signal; the episodes give 24
safe successes and the failure phases above, and the per-step recorder (item 6) would
give the regrasp descent as joint and jar trajectories. Nothing in the description says
an adapter is more or less likely to help than the native policy, and it is not a
description-stage question. Leverage: not assessable yet. Needs: item 6 data at scale,
then a learning comparison; outside stage 2.

**An execution dial or action filter.** The strongest evidence points here. 28 tips
happen with the jar on the table, 19 of them in the regrasp with the fingers descending
onto the rim or sweeping low beside the jar (pairs 0, 2, 9, 13); the three
always-violating tasks 7, 18 and 21 crash within 49 to 168 steps of contact in the same
posture in all four runs, so a change in how an executed action is carried out acts
identically on every repeat there. The dial candidates the evaluator already exposes are
the executed horizon (8 of 16; shorter means the policy re-sees the jar sooner during
the descent) and a scale on the joint-target delta per step (slower descent, lower
impact). A contact-stop filter (hold the arm when a finger first touches the jar during
the regrasp) needs the per-step contact flag the recorder adds. The cost side is the
horizon: safe successes end at a median of 1,026 steps and 6 of 24 after step 1,300, so a
dial that slows every step by half would push a quarter of the successes past 2,000; it
must be gated to the descent or be small. Leverage: on the 19 regrasp tips and the 9
early tips, which is 28 of the 43 violations, 5 of 8 on the development panel. Needs: the
recorder to see descent speed and contact; nothing else to run.

**An outer loop that scores or selects candidate actions (world model, critic).** The
pairs show the decisive difference is 25 to 50 steps before a tip (pair 2: identical
until the fingers meet the rim; pair 0: the descent over the body starts 175 steps before
the crash). A critic would have to predict rim-versus-body from a 256-pixel image 2 to 6
chunks ahead, under physics that makes same-seed repeats diverge. The server returns one
chunk per query; candidate selection means several queries per step and the budget is
counted in simulator interactions and model queries. Leverage: in principle on the same
28 tips as the dial, at much higher cost and with no evidence yet that the images carry
the signal. Needs: the recorder (to label descents by outcome), then an offline check
that a critic on logged states separates the two before any online loop.

**Monitor-and-retreat or retry.** Two kinds of evidence. Stalls: 36 episodes idle with
a median of 1,632 (closed) or 1,818 (open) steps left, and in 5 of the pairs (5, 10, 14,
15, 16) the failing episode pushes at a lid that will not go over; a retry rule (no
proposition change for N steps, lift the arm and re-approach) has time to act at least
twice. Tips: the monitor only labels, and `jar_upright` trips at 30 degrees, after the
fact; a retreat before the tip needs the jar's tilt or angular velocity, which the
recorder logs and the evaluator today does not. Leverage: retry on the 36 stalls (7 of
24 development-panel episodes, 10 of 36 held-out); retreat on tips only with new per-step
state. Needs: a retreat primitive (a scripted lift to a safe height and a re-approach
from the policy), the recorder, and the stall detector.

## Ranking

1. **Execution dial around the regrasp** (horizon or delta scale, phase-gated). Acts on
   the largest violation class, acts identically on the systematic crashes, needs no new
   model and no training, and the hooks exist. Risk: time-outs.
2. **Retry on stall.** Acts on the largest safe-failure class, has the horizon budget to
   do so, and is cheap. Risk: a re-approach that tips the jar converts a safe failure
   into a violation; engagement must be reported with it.
3. **Sampling-noise scale.** Cheapest to sweep and the plan's stated primary, but the
   episodes give it no mechanism. Keep it as the comparison arm in the same sweep as the
   dial rather than a first experiment of its own.
4. Observation noise (gated to stalls), then the outer-loop critic, then an adapter, each
   needing the recorder's data before it can be sized.

## Three stage 2 experiments, ranked

Sizing: the development panel's safe-success rate is 9 of 24 (0.375) and violations 8 of
24. With n episodes per condition the standard error of a rate is sqrt(p(1-p)/n); at n =
200 and p = 0.3 it is 0.032, so a ten-point change is about two standard errors of the
difference between two conditions. At 4.5 minutes per episode, 200 episodes is 15 hours
on one GPU, about $9 at the pod's $0.59 per hour; a three-condition experiment is 600
episodes, 45 hours, about $27. The panel is the six development tasks, 33 or 34 episodes
each per condition, seeds interleaved across conditions on one host; held-out tasks
untouched.

1. **Does a phase-gated execution dial reduce regrasp tips without raising
   time-outs?** Three conditions, native (horizon 8, scale 1.0), horizon 4, and a 0.5
   delta scale applied from the first closure observation until the first lift; 200
   episodes each; report safe success, violations by story, unfinished, success step
   distribution and engagement. 600 episodes, 45 h, about $27. Needs the recorder on so
   the descent speed and contact are logged for every episode.
2. **Does a retry on stall convert stalls into successes without converting them into
   violations?** Two conditions, native and native plus retry (lift to a fixed height and
   release the policy when no proposition has changed for 300 steps after contact, at
   most three times); 200 episodes each; report the same table plus retries triggered per
   episode. 400 episodes, 30 h, about $18. Needs the retreat primitive and the stall
   detector; the recorder to show what the policy does after the retry.
3. **Does the native sampling-noise scale move the safe-success rate at all?** Three
   settings (0.5, 1.0, 2.0) at 200 episodes each, the stage 2 sweep as written in the
   plan but reduced to the settings that bracket native. 600 episodes, 45 h, about $27.
   Needs nothing new; the recorder would say whether any change is in the push, the
   regrasp or the stall rate.

Ranked, not chosen. Experiments 1 and 3 share the native arm; run together they are
1,000 episodes, 75 h, about $44, and give the dial-versus-noise comparison the plan
needs before any optimizer.
