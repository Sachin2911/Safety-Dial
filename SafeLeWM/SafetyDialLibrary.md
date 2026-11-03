# Safety dial: prospective development check of a frozen policy library

Authorised by Sachin's current safety-dial goal on the existing single RTX 4090
pwjry7gqqs278s, hostname 075cb160c9b3. Rental shutdown remains manual.

## Question and limits

Does selecting an existing adapter by an empirical violation limit produce useful
risk/completion choices outside its eight selection layouts? This tests a discrete
library selector, not a risk-conditioned network or a safety guarantee. Its actors
were trained with the original weighted utility, not with these risk targets.
The preceding saved-evidence diagnostic found 17 distinct selected actors across
three matched repeats and three optimizers. Freeze all of them before any rollout.
Do not select a favourable optimizer, repeat or target after seeing these results.

Use the diagnostic's five limits, 0, 0.25, 0.5, 0.75 and 1. For each optimizer and
repeat, take exactly its already-selected actor, maximizing training completion
subject to the empirical limit, then safe completion, then fewer violations, then
earliest order. An unavailable target remains unavailable. When several targets
choose the same actor, evaluate it once and reuse that same row, explicitly; these
are not additional independent observations.

## Panel, controls and accounting

Use the 12 former replication layouts and planner seeds 9401, 9402, 9403. These
are previously observed development cases, not new held-out tasks. The later
24-layout mechanism/PPO final panel is not used to select or assess this library.
Each of the 17 distinct actors receives all 36 episodes, alongside shared native
and the original untrained reference from each of the three repeats. This gives
21 policies and 756 development episodes. Shuffle the 21-policy execution order
once with seed 19100 before outcomes. Score every episode for completion, safe
completion, violation, unfinished and engagement; horizon 200, monitor radius 18,
and all frozen backbone/planner settings stay unchanged. PPO actors use their
exported deterministic weights, matching the original nominee evaluation.

Before this, run 16 paired native/neutral engineering episodes on the eight original
training layouts, seed 8201, requiring exact actions, states, frames and outcomes.
Use neutral initialization seed 12101. No optimizer, new retrieval cache or download.
The existing libraries cost 40,000 training/selection actions per optimizer per
repeat, 360,000 total across nine searches; this acquisition cost remains reported
separately. The new test does not make policy acquisition free. R2 was originally
trained on L4 and is now evaluated on the same 4090 as R0 and R1; record this move.

The cap is 154,800 actions and 774 resets: at most 151,200 development actions,
3,200 engineering actions and 400 failed-start reserve. Execution deadline four
hours from preparation; no automatic retry. All failed starts and model work count.

## Analysis and evidence

Verify source checkpoint hashes against the original completed runs. Independently
recompute the diagnostic's choices from training candidate records only. Save the
selected weights, protocol, source inputs, runtime, schedule and panel before the
first reset. Never reconstruct or modify actors from prospective outcomes.

Report per-repeat, per-optimizer curves across all five limits with denominators
36, showing missing targets. Also report each unique actor once. For each curve,
record whether available targets yield nondecreasing observed violation rates;
report empirical target exceedance at every available target and the span of safe
completion and violations. No pass/fail claim of calibrated risk, no significance
claim from these reused cases, and no optimizer superiority claim from these plots.
Monotonicity of the development selection data does not force monotonicity here.
Any formal risk-control or reduced-tuning claim needs a separately frozen study.

Audit trace outcomes, monitor contacts, actor replay, resource counts, exact panel
coverage, selection reconstruction and pre-rollout locks. Archive all evidence and
verify an independent local copy before writing the final result record.
