# TwoRoom no-hazard baseline comparison

Authorised by Sachin on 1 November 2026 after reviewing the first qualification.
This is a new diagnostic on the existing L4 pod, not a resumed historical campaign.
No safety monitor influences commands, no intermediate targets and no training.
The pod remains running until Sachin requests shutdown.

## Source and scope

Official LeWM code: `lucas-maes/le-wm` revision
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. Use its JEPA and predictor implementations,
the released checkpoint SHA256
`566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd`, and installed
stable-worldmodel 0.1.1 CEM. Save every relevant source hash. Strictly load weights
and compare original-code inference with the previous implementation on fixed inputs
before episodes. Preserve any discrepancy and halt if outputs are not close.

The paper specifies TwoRoom horizon 5, execution 5 blocks, action block 5, 300
candidates, 30 elites, 10 CEM iterations, spread 1, 150 native actions, dataset goal
offset 100. The released YAML instead defaults to 30 iterations, budget 50 and goal
offset 25. Use the paper planning/evaluation numbers explicitly, not those defaults.
The released checkpoint has three positional slots and its original inference code
uses history 3, whereas the paper states history 1 for TwoRoom. Retain the released
checkpoint's runnable semantics and report the discrepancy. This is a reconstruction
of the released-model baseline, not a claim of an exact original training-seed or
software-stack replication of the reported 87% aggregate.

## Frozen panels and arms

1. **Dataset panel:** 50 valid rows sampled with NumPy seed 42, goal offset 100,
   following official eval.py's valid-row rule and sampling from `len(valid_rows)-1`
   without replacement. Use pinned public dataset revision
   `6903a2de048b13819d812da0b4dd661290bc01e4`. Save source episode IDs, row IDs and
   pixel hashes before running any episode. These become development sources and
   cannot later be called untouched final data. Run the paper-profile arm only,
   planner seed 42. Score native goal termination with a 150-action cap.
2. **Historical diagnostic panel:** the six saved no-hazard inputs `case-06` through
   `case-11` from the earlier TwoRoom evaluation. They are previously inspected
   development evidence, not held-out evaluation. Use seeds 303006 through 303011.
   Run all three arms on identical inputs and seeds with a 150-action cap.
3. **Constructed diagnostic panel:** start (164,90), goal (164,170), no hazard;
   seeds 4201,4202,4203. Run all three arms with the same 150-action cap. The old
   qualification cap was 100; also retain success by action 100.

Arms:
- `paper`: original-code task model and upstream CEM, settings above. No candidate
  projection, no custom standard-deviation floor, and upstream current-mean candidate.
  Execute the complete plan; no residual warm start remains.
- `random05`: fixed historical selected settings, not random parameter search:
  horizon 2, execute 1 block, 150 candidates, 30 elites, 30 iterations, spread 2,
  retention 0.25, no temporal correlation. Reproduce historical mean warm start
  and candidate retention. All candidates are task-ranked; there is no hazard
  filtering or learned safety checker. Explicitly record this port from the old stack.
- `qualification`: preserve the prior custom planner recipe: horizon 3, execute
  1 block, 300 candidates, 30 elites, 8 updates, spread 1, clipped candidate domain,
  population standard deviation with floor 0.05, cold starts, no retention. Use the
  same validated frozen inference model where compatible; quantify original/current
  model equivalence first. Preserve previous qualification files unchanged.

All arms use released ImageNet pixel normalization and saved dataset action mean/scale.
Audit those scalers against the downloaded dataset. Clip only executed commands to
native bounds for paper/random05 and retain both proposed and executed commands.
Report out-of-domain model candidates and executed clipping counts. More task success
with out-of-domain model proposals is an implementation diagnostic, not a safety claim.

## Limits, accounting and decisions

77 planned episodes, at most 11,550 episode actions and 77 episode resets. Engineering
allowance: 8 resets and 20 actions, with every attempt recorded. Maximum 30 million
candidate model-step evaluations and a 60-minute worker deadline, plus up to 60 minutes
for data transfer/setup. At $0.59/hour the two-hour compute allowance is $1.18, excluding
prior/subsequent idle rental and storage. No automatic pod shutdown. New disk allowance
25 GB including the 3.43 GB archive and 12.78 GB HDF5. Abort on accounting, model-parity,
state restoration or malformed-image failures; preserve unsuccessful attempts.

Before evaluation, freeze source and panel hashes. No selecting best settings or
changing cases after outcomes. Report each panel separately, completion/unfinished/
engagement denominators, native reward, goal distance, primitive actions, model calls,
candidate model steps, clipping and elapsed time. Show trajectories. The paper-profile
panel estimates this reconstructed baseline; six historical successes are not a
guarantee of reproduction on a different stack. A settings gain is not an evolutionary
result. Do not launch hazard studies or optimizers from this protocol.

Sources: [paper](https://arxiv.org/html/2603.19312v1),
[official code](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac).
