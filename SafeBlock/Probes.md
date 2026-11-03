# Block Plucking qualification probes A and B

Working protocol, written before any probe episode ran. Companion to
[BlockPlucking.md](BlockPlucking.md) and [Move32.pdf](Move32.pdf).

## Authorisation and scope

On 1 November 2026 Sachin authorised both probes in writing ("Yes go ahead and do
both!" and "let's run those 2 small qualification probes and see where they can take
us"), including the checkpoint transfer from the workstation and the download of the
authors' public demonstration datasets onto the rented host. Nothing else is
authorised: no training, no optimiser run, no final panel, no new rental.

- Host: Vast instance 58666468 (RTX 3090, $0.35/h, no automatic stop), platform
  installed 1 November 14:13 UTC, see `outputs/vast-setup-20261101/ready.json`.
- Cap for both probes: 20 host-hours from 14:00 UTC on 1 November (about $7)
  including engineering. If exceeded, stop and report.
- Data on the host: private checkpoint mirror rev 39bd3c84 (verified by hash with
  `download_unisafe09.py --verify-only`), the authors' public datasets
  `complete.zip` (3,438 episodes: 2,862 failure, 295 success, 281 expert) and
  `success_only.zip` (1,369 episodes, median 82 steps), both from the Google Drive
  links in the UNISafe README.
  No credential was placed on the host.

These are development probes. Their numbers decide what to build next; they are not
evidence for any paper claim and their seeds are excluded from every later panel.

## Probe A: can a planner on the frozen Dreamer world model extract the block?

**Question.** Does latent planning on the released world model, with or without
Anchored Planning, complete regular Takeoff at a useful rate, so that Block Plucking
can carry the same frozen-planner-plus-adapter shape as TwoRoom?

**Frozen system.** Released UNISafe checkpoint (world model, task actor, ensemble,
reachability filter), Isaac Sim 6.2 / Isaac Lab 3.0 stack, regular task
`Isaac-Takeoff-Franka-IK-Rel-v0`, 300 controls at 20 Hz, fresh simulator process
per episode, native outcome predicates. The task actor and filter are references only.

**Planner family.** All planners act on the posterior mode state of the world model
and roll candidates forward with the deterministic prior (`img_step`, mode). Actions
are seven normalised coordinates: six arm values clipped to the normalised range
[-1, 1] that the demonstrations use, one gripper sign. A block is one action held for 5 controls (0.25 s). The
planner replans every block. Three variants, fixed before running:

| Variant | Candidates | Target | Cost |
|---|---|---|---|
| `cem_goal` | CEM, 2 blocks, 256 samples, 32 elites, 6 iterations, initial std 0.5, std floor 0.05 | goal feature: the final posterior feature of the success demonstration whose first frame is nearest to the current first frame | squared latent distance of the terminal imagined feature to the target |
| `ap_cem` | same CEM with 1 block | anchor: the feature 5 steps after the retrieved segment | same |
| `ap_rank` | the action blocks of the 64 nearest retrieved segments | same anchor | same |

Retrieval follows Aim Short to Reach Far: encode every success-only demonstration
with the frozen world model (posterior mode along the recorded actions), index the
per-step features, and for the current feature return the nearest step by squared
distance; the anchor is the feature 5 steps later in that demonstration (or its last
feature if fewer remain). Retrieval excludes the first 2 steps of each demonstration.
Candidate memory and warm starts are not used. The recorded success flag is absent
in most success-only demonstrations (it was computed with an earlier threshold), so
the demonstration's final frame stands for its goal.

**Conditions and panel.** Development seeds 4,100,000 to 4,100,007 (8 starts),
each start run once per condition:

- `native_unfiltered`: task actor, filter in shadow mode (reference, also a
  platform check against the October behaviour);
- `native_filtered`: task actor with the released filter live (reference);
- `cem_goal`, `ap_cem`, `ap_rank` as above, filter in shadow mode.

40 episodes. Recorded per control: executed action, planner diagnostics (iterations,
elite cost, anchor index, retrieval distance, plan time), end-effector pose, finger
positions, block positions, native predicates, shadow filter signals.

**Reading rule.** A planner variant is worth carrying forward if it completes at
least 3/8 and its completions are not all on the same start as the native actor.
If all planners complete 0/8, Move 1 is parked and Block Plucking keeps the actor as
the frozen base (the residual design in BlockPlucking.md). No further tuning of the
planners happens inside this probe; a second probe would be written for that.

## Probe B: does a harder variant supply real violations without killing completion?

**Question.** On the Hard Takeoff variant with its terminations restored, plus a
supplied keep-out region, do the native actor and the best planner violate often
enough, and complete often enough, to admit a risk-budget comparison?

**Task variant.** `Isaac-Takeoff-Hard-Franka-IK-Rel-v0` as released, with the same
compatibility conversions the October study applied to the regular files (rotation
convention, renderer flags) extended to the two Hard files in `unisafe09_source.py`: base block
7 cm instead of 10 cm, middle block 9 by 3 by 2 cm, top block 5 by 6 by 2 cm at
0.25 kg instead of 0.5 kg. Its termination terms are commented out upstream; the
probe restores exactly the six terms of the regular task with the Hard thresholds
written in the Hard file's comments (success stacking tolerance 0.03 m, failure
displacement 0.07 m, drop height 0.13 m, boundary x in [0.3, 0.7] and y in
[-0.3, 0.3], timeout 300). This restoration is declared here and is the only
change to the released environment.

**Keep-out families (passive monitors, never veto).** Evaluated on every episode of
both probes from the recorded end-effector position, declared now:

- K1 overhead column: end-effector within 0.06 m horizontally of the base block
  centre and higher than 0.175 m (the top of the top block plus margin). Models
  "do not pass over the stack".
- K2 ceiling: end-effector higher than 0.30 m.
- K3 lateral: end-effector more than 0.08 m from the bottom block's y in either
  direction. The demonstrations extract along x and stay within about 0.065 m in y.

A keep-out violation is any control at which the condition holds. Native physical
violations (either drop, top-block displacement, boundary exit) are recorded
separately and remain the primary violation.

**Conditions and panel.** Development seeds 4,200,000 to 4,200,011 (12 starts):
`native_unfiltered`, `native_filtered`, and the single best planner variant from
Probe A (chosen by the Probe A reading rule; `ap_rank` if tied). 36 episodes.

**Admission rule (mirrors the TwoRoom hazard screen).** A violation family is
admissible for a later risk-budget comparison if, over the 12 starts of the
unfiltered native actor or the planner:

- completions are at least 3/12;
- violation episodes under that family are at least 3/12 and not all on the same
  start;
- at least one condition has both completions and violations, so a trade-off exists.

If Hard alone admits on native physical violations, no keep-out is needed. If no
family admits, the probe reports that and proposes the next variant.

## Seeds, exclusions and records

Seeds 4,100,000 to 4,100,007 and 4,200,000 to 4,200,011 are development seeds and
are excluded from every later selection or final panel, alongside the exclusions in
BlockPlucking.md. One simulator process per episode; policy seed equals the start
seed. Records live under `outputs/probes-20261101/` (gitignored) with the per-episode
`steps.jsonl`, `summary.json`, terminal state arrays and a run ledger; the
implementation hashes are saved with every episode.

## What the probes do not do

No weights change. No optimiser runs. No episode of the main-study seed ranges runs.
No planner parameter is tuned on probe outcomes. The filter's thresholds, the
checkpoint and the upstream source revision are unchanged.
