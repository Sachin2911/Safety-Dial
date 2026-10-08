# Walker candidate-quality development diagnostic

On 5 October 2026 the user accepted the recommendation to implement a bounded Walker candidate-quality diagnostic before another confirmation study. The immediate question is whether search produces useful controllers that imagination misses, or whether the sampled candidate pool itself lacks useful improvements. This supersedes the next action in the Stage 6 follow-up sequence. Historical experiments, failed gates, and completed results remain unchanged. No supervisor approval is claimed.

## Fixed candidate pool

Freeze all parameters, origins and action caps before querying any new simulator outcomes. Use the two completed Stage 6 development seeds for each of k0-zero and k1-ROSARL-style. Include the baseline, each search's imagined winner, two uniformly sampled population members at generations 0, 2, 5 and 7, and eight new Gaussian directions at 0.25, 0.5, 1 and 2 times the frozen initial residual scale. The same direction is reused across scales. Sampling uses seed 20261401 and never reads simulator outcomes.

Also include winner contractions by 0.25 and 0.5 and winner variants with per-motor correction caps of 0.025, 0.05 and 0.1. The cap compares evolved and baseline actions on the SAME visited visual/history input, then clips their difference. It applies in imagination and real execution at every block. This does not bound divergence between trajectories and does not guarantee safety. Uncapped behavior is unchanged. Record realized same-input action deviation throughout real rollouts. These variants are a bounded mechanism probe, not a newly optimized controller or a change to historical search.

Deduplicate by parameter content and action cap while preserving all origins. The resolved pool contains 89 distinct interventions. The pool is not 89 independent optimization seeds: archived searches have only two paired seeds, contractions and caps share parameters, and random scales share directions.

## Episode roles and selection barriers

Reuse ONLY the same development subsets previously exposed in Stage 6. The first 32 historical Stage 5 selection episodes provide common imagined ranking, the next 32 provide simulator selection, and the first 64 historical Stage 5 evaluation episodes provide a separate development check. Whole episode identities must be disjoint across these roles. No remaining historical evaluation episodes or new final episodes are queried. All results remain exploratory because these episodes were previously used.

On the common imagined ranking bank, score all candidates with k0 unpenalized matched-termination progress and k1 fixed-penalty progress (lambda=0.5, eight samples). These are diagnostic common-pool selectors, NOT a fresh comparison of the original adaptive ROSARL algorithm. Original imagined winners retain their original search provenance. Common noise draws are paired across candidates within each role, and every candidate receives both noise audits.

Evaluate the entire fixed pool on simulator-selection episodes. For the whole pool and each origin family separately, freeze an imagined-selected candidate for each score and a simulator-selected candidate. Every family includes the baseline. The simulator rule requires strictly lower observed failure and at least 90% mean baseline progress; choose lowest failure, then highest progress, then chronological index. Nonpositive baseline progress or no eligible candidate retains the baseline.

Freeze ALL choices before querying any development-check outcomes. Then evaluate the whole pool on the 64 check episodes, enabling both independent-of-current-selection comparisons and explicitly retrospective candidate-quality diagnostics. Do not select a final controller from these check outcomes and label it independently validated.

## Analysis and decision

Report all candidates, origins, action deviations, imagined failures at both noise levels, dense real failures, true block-end failures, real-image probe failures, and real progress. Report the number of candidates satisfying the empirical failure/progress criterion on each bank and both banks. Report each frozen selector's check contrast against the baseline using paired episode bootstrap intervals (10,000 draws, seed 20261403). All intervals are descriptive, unadjusted 95%, conditional on this pool; inspecting many candidates creates multiplicity. Spearman correlations across correlated candidates are descriptive only, and undefined when an axis is constant.

A retrospective best check candidate is an upper bound within this finite pool, not a population optimum. Better candidates missed by imagined selectors suggest a ranking/shortlist problem. Improvements concentrated in smaller or capped residuals motivate a separately declared constrained-search experiment. No improvement in this small pool does not prove that the policy class is incapable of improvement. A baseline fallback is not learning progress. Full episodes and fresh confirmation remain later experiments.

## Budgets and reproducibility

The complete planned diagnostic uses 768,960 new predictor rows and 854,400 new real simulator steps, with zero source collection and zero gradient updates. Every real branch executes all 100 steps, including after failure. Charge renders, encodes and the 64-frame startup fingerprint separately. Reused source collection, original model training, and parent search costs remain historical costs, not zero lifetime costs. The absolute 45-minute cap is an operational ceiling based on the earlier pilot; it is not a statistical power calculation and is preserved on resume.

Use a managed supervisor process. Commit and hash the executable sources and protocol before launch; copy them into the run. Hash archived candidate inputs and parent banks. Atomic query receipts retain complete numeric outputs and reject implicit replay after an interrupted query. Reproduce the analysis and verify query costs, phase barriers, source hashes and action caps before interpreting results. No model checkpoint is created; candidate parameter archives remain ignored run artifacts. No external publication is part of this diagnostic.

Commands:

```bash
uv run python experiments/scripts/evo_candidate_quality.py
uv run python experiments/scripts/evo_candidate_quality.py --execute --run-id walker2d-evo-s6-candidate-quality-20261005-1
```
