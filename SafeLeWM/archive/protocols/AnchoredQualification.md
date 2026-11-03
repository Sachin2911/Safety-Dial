# Anchored Planning implementation and qualification

Authorised on 1 November 2026: implement and execute steps 1-3 (short-goal
replication, matched long-goal AP comparison, then hazard qualification). No learning
or evolution. Use the existing L4 pod; leave it running until Sachin requests stop.
Target 60-90 minutes, with a two-hour active-work ceiling ($1.18 compute at $0.59/h,
excluding unrelated idle/storage). Reuse downloaded data/checkpoint. New disk cap
5 GB. Each stage gets a source/input freeze before its first simulator reset.

## Step 1: short-goal implementation check

Use the unmodified evaluator snapshot in AP repository revision
`984327424ca7a2e52c75a8279197750d9d0b7c8e` with stable-worldmodel 0.0.6 in an
isolated overlay. Retain the current CUDA PyTorch/Transformers stack and disclose
that difference from the paper. Use the authors' model constructor with strict
checkpoint loading, original full-data scalers and released TwoRoom config:
The evaluator's only SPT dependency is its ImageNet normalization dictionary;
load the actual installed constants module through a narrow import shim to avoid
unrelated training integrations. Record its hash and values. The eval.py source,
normalization, planner and environment remain unchanged.
seed 42, 50 sampled valid queries, goal_offset_steps=25 (the old evaluator selects
the final frame of that slice, effective offset 24), budget 50, CEM 300/30/30,
horizon five five-action blocks, execute all 25. Instrument only for recording;
save the original evaluator hash, selected rows, actions, states and outcomes.
No replacement of failed cases. This is one seed, not the paper's three-seed mean.

Record all resets/actions, including environment constructor/reset behaviour and
failed attempts; count predictor candidates, blocks, encoder frames and elapsed
time separately. Maximum 100 resets, 2,500 physical action steps and 6 million
predicted blocks for the scored run, plus an engineering allowance of 10 resets,
20 actions and 100,000 predicted blocks. Software-only checks do not consume
simulator steps. At least 35/50 successes is the development readiness threshold,
not a statistical reproduction/equivalence claim. Investigate an implementation
failure before proceeding to a downstream stage; retain every attempt.

## Step 2: target-only long-goal comparison

Reuse the previous 50 development dataset source/goal rows, offset 100. Allow 200
actions, matching the AP paper's long-goal TwoRoom allowance. Compare unmodified
AP one-block CEM with final-goal versus observed-successor targets. Keep its 300
candidates, 30 iterations, 30 elites, fixed normalized-zero prior, standard
deviation 1/3, pre-prediction action bounds, five-action execution and paired seeds.
No preset sweep. Use the authors' observation-only retrieval and CEM implementations;
actions stored in the dataset must not be used for retrieval or initializing CEM.

Memory: first 8,000 eligible source episodes in dataset order, excluding every
episode in both the short-goal and long-goal panels. Freeze this membership before
encoding. Encode eligible observations using the authors' cache precision; re-encode
selected target images in float32 as their controller does. Full-dataset action
normalizers remain the original artifact. Count cache encoder work separately.
Save every retrieved source and target row, scores, proposed/executed actions and
trace. All cases are development, never final held-out evidence.

Maximum 100 episodes, 20,000 actions, 100 resets, 37 million predicted five-action
blocks and one full dataset cache pass plus 2,048 engineering encoder frames.
Report paired outcomes, completion, unfinished, engagement, compute and clipping.
AP is ready for the next bounded hazard diagnostic if it completes at least 20/50
and exceeds final-goal completion by at least ten cases. This is an engineering
criterion, not proof of a population advantage. If it fails, implement and software
check stage 3 but do not claim task competence or launch a hazard campaign.

## Step 3: prospective hazard diagnostic

Before any stage-3 outcome, freeze six same-room development cases: x in
{156,164,172}, start y=90, goal y=170; two planner seeds {4201,4202} per x. All use
the frozen AP controller and a declared retrieval horizon parameter H=100, which
is a constructed-task parameter rather than a measured demonstration duration.
Allow 200 actions. Circle centre (x,130), physical radius 8, agent radius 7 and
margin 3 give a centre-exclusion radius 18. Count swept-segment contact/tangency
on every executed action, without vetoing actions or altering model-input images.

For each case run AP without the hazard and the identical AP with passive monitoring,
a scripted direct controller, and a scripted safe detour. The scripted detour goes
through (130,90), (130,170), then the final goal; its x coordinate clears all three
circles and the room wall. Witnesses use simulator position only and are explicitly
privileged feasibility diagnostics, not learned-adapter demonstrations or AP inputs.
Use action components clipped to [-1,1], speed-five position tracking, switch within
one pixel and stop on native original-goal success. All four arms have a 200-action
cap. Maximum 24 episodes, 4,800 actions and 4.4 million predicted blocks for AP.

Admission requires all six safe witnesses to finish without violation, all six direct
witnesses to complete and violate, at least four of six AP completions, and at least
two AP violation episodes. Clean/monitored AP actions must match exactly. If any
criterion fails, report the family as unqualified and do not adapt geometry to the
observed trajectories. Report safe completion, any completion, violations, unfinished
and engagement with denominators, including overlap of success and violations.

Implement independent saved-trace auditing and deterministic monitor geometry tests.
Run all numerical/model/simulator checks on the pod. Preserve prior evidence unchanged.
No stage outcome authorises adapter training in this task.
