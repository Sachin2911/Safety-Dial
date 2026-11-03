# Evolution improvement: more updates within the same interaction budget

Fresh development study authorised by Sachin on 2 November 2026 for the existing
L4, RunPod yvrg8ow7i9n9u6, host 8cf3a706cdd1. This is not another final test or a
search that stops when a favourable result appears. All variants and outcomes
are retained. Rental shutdown remains manual.

## Hypothesis and fixed arms

The existing CMA implementation is already diagonal CMA-ES. In the matched PPO
study it made only five complete updates per 40,000 actions. Test one change:
reduce its population from eight to four, allowing more updates at the same
actual interaction allowance. This is a hypothesis, not an expected result or
a new-algorithm novelty claim.

Three arms: existing diagonal CMA with population eight; otherwise identical
diagonal CMA with population four; parameter random search with population eight.
All retain sigma 0.5, explicit CSA, no mirrored samples, neutral initialization,
the same 132 head coordinates, original eight-case safety-aware score and earliest
tie-breaking. No reward shaping, additional actor information or tuning sweep.
Random search keeps drawing independently from the same initial distribution.

Two paired repeats, E0 and E1. Seed bases 17100 and 17200, with initialization +1,
untrained +2, CMA +3, random +4. Both CMA arms use the same CMA seed within repeat.
Orders: CMA8/CMA4/random, then random/CMA4/CMA8. Each arm receives exactly 40,000
actual training and selection actions. Complete eight-case panels can nominate;
quota-cut panels are saved and charged but cannot nominate. Preserve optimizer
ask/tell order and update only complete populations.

## Development cases and gate

Use the existing eight training layouts with seed 8201. Use the twelve former
replication layouts and planner seeds 9401, 9402, 9403 as development validation.
They have been seen before and are explicitly not a new held-out test. Do not use
the later 24-layout mechanism/PPO final panel to choose this variant. Any later
confirmation requires a new frozen panel and fresh learner seeds.

Within each repeat lock all three nominees before validation. Evaluate each on
all 36 development episodes, plus its untrained reference; native is shared and
evaluated once on this host. Report all 324 validation episodes, including safe
completion, completion, violations, unfinished episodes and engagement.

A positive development signal requires CMA4 to exceed both CMA8 and random safe
completion by at least two episodes in each repeat, with no increase in violation
count relative to CMA8 in either repeat. These are advancement criteria, not
statistical significance. A failure does not authorise changing settings or
reusing validation results as a final claim. Report the gate and all outcomes.

## Qualification, bounds and evidence

Use the existing isolated venv-ppo and its qualified frozen TwoRoom backbone,
checkpoint, memory and native environment. Before simulation, verify those hashes
and the installed CMA version. Test optimizer ask/tell ordering for populations
four and eight and exact reservation accounting, including partial populations.
Sixteen native/neutral engineering episodes on training layouts must be bit-exact.

Training allowance 240,000 actions. Validation at most 64,800; engineering at most
3,200; failed-start reserve 400 actions/two resets. Absolute cap 308,400 attempted
actions and 8 hours from preparation. No automatic retry of a performance episode.
Stop work on an error, budget or deadline; never stop the rental automatically.
All failed starts and model queries count. No new download or rental is required.

Freeze code, configuration, panel, source dependencies and seeds before the first
reset. Save every genotype, decoded checkpoint, dispatch reservation, generation,
partial episode, selection lock and trace under
`outputs/evolution-budget-20261102/`. Audit physical success and swept contact,
actor replay, action quotas, population membership, optimizer updates, nominee
selection and validation completeness before interpretation or packaging.
