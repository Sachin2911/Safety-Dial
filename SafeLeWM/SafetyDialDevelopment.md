# Safety-dial development: feasible choices before more training

Authorised 2 November 2026 on the existing single RTX 4090, RunPod pwjry7gqqs278s.
The first phase is a saved-development-evidence diagnostic, not a validated dial.
No simulator rollout, new optimization, hazard change or new rental is included
in this phase. Rental shutdown remains manual.

## Frozen diagnostic

Use only completed eight-case training/selection panels from the three repeats
of the completed matched PPO study. Keep CMA, random and PPO separate within
repeat. Exclude partial panels. Do not select policies using final evaluation
outcomes. Source hashes bind both host summaries and this analysis before it runs.

For each empirical episode-violation limit in [0, 0.25, 0.5, 0.75, 1], select the
candidate with maximum task completion among candidates meeting the limit. Break
ties by greater safe completion, fewer violations, then earliest candidate order.
If none meets a target, record no candidate. Preserve each target's selected ID,
completion, safe completion, violations and supporting sample count.

Report the number of distinct selected policies, whether any two available targets
give different measured risk, and whether those two policies each complete at least
half the development cases. The same hazard and monitor apply at every target.
This diagnoses whether the existing candidate pools have a useful tradeoff at all;
the weighted-utility searches were not trained to these risk targets.

Eight reused development cases do not certify violation probabilities. Report the
one-sided 95% zero-event upper bound 1 - 0.05^(1/n) as a sample-size illustration,
with its independent-identically-distributed assumption explicitly stated. These
constructed repeated cases do not justify that assumption or a safety guarantee.
No new uncertainty claim is inferred from this retrospective pool selection.

## Following phase

Use this diagnostic to specify a separate prospective development protocol for
training/selecting a discrete policy library at useful risk targets. Lock that
protocol and its budgets before rollouts, and lock policies before opening a new
final dial panel. Matched acquisition budgets and ordinary/random controls remain
required. A failed or empty target is reported, never silently replaced.

This phase has a 30-minute computation cap and zero simulator actions. Save inputs,
source hashes and every target selection under `outputs/safety-dial-20261102/`.
It does not establish reduced manual tuning, calibrated risk or monotonic unseen
performance. Those remain part of the active goal.
