# Evaluation and compute after the retry gate

Research notes, 24 October 2026. Budget supplied by Sachin: $50 to $100 for the next phase. These are proposed designs and local calculations, not a frozen protocol or authorization to rent a machine. No simulator jobs were run for these notes.

## What does the existing experiment establish about noise and pairing?

### Takeaway

The retry gate found no detectable average improvement, but did not establish equivalence or safety noninferiority. Renderer nondeterminism breaks identical trajectories, not the usefulness of seed and task blocking: the full paired dataset actually has appreciable positive outcome correlation.

### Cited Findings

- Native safe completion is 17/96, retry 16/96; violations 48/96 and 47/96; safe unfinished episodes 31/96 and 33/96; all episodes engaged and completed on their first attempt. The safe-success paired table is both 8, native-only 9, retry-only 8, neither 71. For violations it is both 31, native-only 17, retry-only 16, neither 32. Both exact McNemar p-values are 1.0. [Frozen results](../../../../progress/receipts/03-results.json), [scoring receipt](../../../../progress/receipts/03-results.md)
- The policy server produces byte-identical chunks for identical requests and a fresh key. Independent simulator launches differ in rendered observations at the first frame. Consecutive episodes with the same seed on the same server do not automatically reseed; the re-seed probe is needed in that situation. [Identity receipt](../../../../progress/receipts/03-identity.md)
- Direct calculations from all 96 pairs give binary Pearson/phi correlation 0.3783 for safe success and 0.3126 for violations. After subtracting each task's arm-specific mean, residual correlations are 0.3009 and 0.3069. These are descriptive estimates from the two treatment arms, not isolated estimates of a pure common-random-number mechanism. [Input rows](../../../../progress/receipts/03-results.json)
- The empirical variance of paired differences is 0.622 of the corresponding independent-Bernoulli variance for safe success and 0.687 for violations. Pairing can therefore help even though trajectories differ. Common-random-number methods exploit shared random scenarios and positive covariance rather than requiring identical outputs. [Local input](../../../../progress/receipts/03-results.json), [Pearce, Poloczek and Branke](https://arxiv.org/abs/1910.09259)
- Rough paired normal intervals for retry minus native are -9.5 to +7.4 percentage points for safe completion and -12.8 to +10.7 points for violations. These use the sample standard deviation of paired differences divided by sqrt(96), with 1.96 standard errors. They are descriptive calculations, not replacements for a preregistered stratified analysis. [Input rows](../../../../progress/receipts/03-results.json)
- The Odyssey currently says the frozen knob "changes nothing" and that renderer nondeterminism "rules out common random numbers for the search". The first wording overstates a null hypothesis test; the second is not supported by the observed covariance. [Current log source](../../../../progress/theOdessy.tex)

### Inferences

- Keep task/seed blocking and alternate arms across GPU lanes and time on one host. Treat the remaining renderer/physics randomness as an additional source of noise. Do not promise the observed correlation will transfer to a different knob.
- Do not use the 75 no-retreat episodes as an unbiased native-replicate experiment. Whether retreat fires is a post-treatment trajectory outcome. Conditioning on it selects a subgroup; the 0/21 success rate among retreat episodes does not estimate the causal effect of retreat on stalled states.
- Initial-task heterogeneity and residual seed association both appear useful, but this experiment cannot separate all causes of covariance. A shared task/seed pair is a block, not an exact counterfactual trajectory.
- Retain the paper-compatible renderer for the main comparison. A deterministic renderer would be a separate benchmark-contract change requiring a new native control.

### Gaps

- The strict closure-before-carry score is absent from the current gate scorer. It should be reconstructed from preserved per-step monitor sidecars before claiming safe requested completion. [Scorer](../../../../scripts/todo3/score_gate.py), [existing timeline implementation](../../../../scripts/todo2/timeline.py)
- The full simulator randomness cannot currently be reseeded or replayed exactly. Four identity launches establish observation differences, not a quantitative decomposition of renderer versus physics variance. [Identity receipt](../../../../progress/receipts/03-identity.md)

## What evaluation is credible at a lean budget?

### Takeaway

Use a cheap, preregistered three-arm mechanism gate before funding optimization. The budget can buy several thousand episodes at the measured host rate, but a broad optimizer or Pareto-front claim remains limited by noisy candidate ranking and only a few independent searches.

### Cited Findings

- The development panel is tasks 0, 3, 6, 9, 15, 17. Held out are 2, 5, 8, 14, 16, 23, 7, 18, 12. They received baseline episodes before the split but must receive no new intervention episode until the final comparison. Pool tasks are diagnostics only, never intervention scoring. [Frozen panels](../../../../progress/panels.json)
- The native development rate in the recent gate is 17.7%, rather than the 23.1% pooled all-task replication rate or the earlier small development estimate of 37.5%. Use the current panel's estimate for planning and show sensitivity to other plausible rates. [Gate results](../../../../progress/receipts/03-results.json), [replication note](../Jar%20replication%20and%20noise%20floor.md)
- Equal-allocation, unpaired, two-sided alpha .05 and 80% power normal calculations give approximately 275 episodes per arm for a 10-point gain from .177, 316 from .23, and 356 from .30. For 15-point gains these are 131, 147 and 163. Formula: n = [z(.975)*sqrt(2*pbar*(1-pbar)) + z(.8)*sqrt(p0*(1-p0)+p1*(1-p1))]^2 / (p1-p0)^2. These are planning calculations under independent episode sampling, not guarantees. [Rates used](../../../../progress/receipts/03-results.json)
- A simple paired normal calculation with correlation .3 gives roughly 192 pairs for a 10-point gain from .177 and 220 from .23; for 15 points, about 91 and 102. Exact binary testing, two treatment comparisons, task structure and a different knob can change these requirements. [Local correlation inputs](../../../../progress/receipts/03-results.json), [pairing rationale](https://arxiv.org/abs/1910.09259)
- At p=.23, the standard error of a candidate's empirical safe-success rate is .172 for six episodes, .121 for 12, .086 for 24, .061 for 48 and .043 for 96. A candidate advantage of ten points is smaller than one standard error at n=12. [Replication base rate](../Jar%20replication%20and%20noise%20floor.md)
- Wilson or Jeffreys intervals have better small-sample behavior than the conventional Wald proportion interval. Use Wilson per-arm descriptive intervals; assess differences with paired/stratified methods rather than overlap of per-arm intervals. [Brown, Cai and DasGupta](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.pdf)
- Few-run evaluation needs explicit uncertainty rather than point-estimate rankings, and optimization benchmarking should specify objectives, methods, evaluation measures and reproducibility. [Agarwal et al.](https://arxiv.org/abs/2108.13264), [Bartz-Beielstein et al.](https://arxiv.org/abs/2007.03488)

### Inferences

1. **Offline preparation first.** Re-score strict requested completion from all preserved sidecars; retain original benchmark scores. Plot failure phase against onset of a proposed gate using existing videos/logs. Calibrate any detector from development/pool evidence only. Do not infer counterfactual robot outcomes by editing logged actions offline. Validate absolute-target transformations and identity behavior with fake policy-server requests before renting compute.
2. **Freeze one next hypothesis.** For example, a bounded arm-target delta reduction around a proprioceptively detectable regrasp phase lowers regrasp tips and raises safe requested completion. Choose at most two non-native settings before examining new outcomes. Mechanism and observability details belong in the separate mechanism notes.
3. **Initial gate: 288 episodes.** Six development tasks x 16 new seeds x three arms, native plus two fixed settings. Share the native control, alternate arms across lanes/time, and record all actions and proposed/filtered/executed distinctions. This is a large-effect screen, not a well-powered equivalence study. It avoids spending $20 on a single knob before any optimizer trial. A stronger version is 576 episodes, 192 per arm, if the first stage is predeclared inconclusive and the continuation/test rule is frozen in advance.
4. **Analysis and stop rule.** Specify the two native comparisons and multiple-comparison handling, a practically meaningful effect target and the safety margin. Inspect mechanism checks separately from outcomes: did the setting actually alter executed descent, did the gate fire before tips, did time-outs increase? A failed mechanism check means the implementation has not tested the intended physical hypothesis. A working mechanism plus wide null interval means inconclusive; a narrow interval excluding the desired gain supports stopping that knob. Do not choose a new sample size after repeatedly checking p-values without a predeclared sequential rule.
5. **Calibration from the gate.** Re-sample balanced task/seed blocks to estimate ranking stability at n=6, 12, 24 and 48. Use this to choose the stage-3 evaluation rule. If ranking is mostly noise, do not run a large population with six episodes per candidate. Increase per-candidate replication, shrink dimensionality, or stop at the fixed-setting result.
6. **Only then optimization.** Keep the genotype at one or two execution parameters initially. Compare the best fixed setting and native against random search, a simple EA and an ordinary learner on the same information and action family. A contextual bandit or policy-gradient-trained tiny gate is a meaningful learning comparison if the deployed gate has decisions; Bayesian optimization is useful as a non-EA search baseline but does not by itself replace a learning comparator. Do not broaden to a recurrent controller solely to make evolution look necessary.
7. **Final held-out test once.** Freeze every selected controller, reporting rule, seed panel and primary claim before any intervention episode on held-out tasks. Evaluate every independent search winner, not just the luckiest one. If budget only supports one deployed configuration per algorithm, predeclare its selection using development validation and describe optimizer variability as a pilot limitation.

Statistical details: the nine held-out tasks are a fixed, deliberately stratified Jar panel. Equal task weights are clear and should be fixed. Report the six middle tasks and the three extreme sentinel tasks separately as secondary descriptions, without quietly removing the sentinels from the primary aggregate. For a claim about these exact tasks, estimate mean paired differences within task, with seed-block bootstrap or a preregistered paired randomization analysis and task-stratified summaries. For a claim about unseen tasks generally, only nine task clusters exist, so uncertainty about task transfer is substantial. Episode count must not be used to pretend there are hundreds of independent tasks. Shared optimizer runs add another hierarchy: thousands of final rollout episodes do not create thousands of independent optimization runs.

Primary safe requested completion penalizes inactivity automatically. Report violations, all unfinished episodes, safe unfinished episodes, engagement, closure/lift rates and time-to-success alongside it. A contact indicator alone is weak engagement evidence because a stalled policy can remain in contact. A safety noninferiority statement requires a predeclared margin and an upper confidence bound, not p>.05 for a rise. At p=.5 and rho=.3, even an observed zero difference needs roughly 379 pairs for a one-sided 95% upper bound below +5 points, versus about 95 pairs for +10 points; these are normal planning approximations, not 80%-power calculations. Zero violations in n episodes still gives a rate bound around 3/n.

### Gaps

- The actual lever effect and its covariance are unknown. A power number computed using the retry arm is a scenario, not a validated design guarantee.
- Ordinary learning is required by AGENTS.md before an evolutionary claim, while firstPrinciples.md defers learning comparisons to a later stage. Resolve that document inconsistency explicitly in the next frozen comparison, favoring the stronger current research control. [AGENTS](../../../../AGENTS.md), [plan](../../../../firstPrinciples.md)
- A one-dimensional fixed sweep may already solve the useful problem. Evolution beating native is then insufficient evidence that evolution adds value over the fixed setting or random search.

## What fits $50 to $100, and what should be counted?

### Takeaway

The actual two-GPU whole-host costs are lower than the old note's per-GPU estimates: a 288-episode gate costs about $5 to $8 raw compute at historical prices. Around $100 can support a narrow complete pilot study if the lever is strong; it does not guarantee enough precision or optimizer diversity for a broad algorithmic claim.

### Cited Findings

- Task 3 completed 192 jobs in 7 h 11 min on two concurrent GPU lanes, approximately 26.7 episodes per whole-host hour. Total simulator interactions were 335,074 steps and 41,907 model queries. Mean episode length was 1,745 steps. [Run closing note](../../../../progress/receipts/03-run.md)
- Arm-specific totals recomputed from the saved rows are native 170,191 steps / 21,286 queries and retry 164,883 steps / 20,621 queries. Thus equal episode counts were not equal actual simulator interaction counts. [Rows](../../../../progress/receipts/03-results.json)
- Instance I cost $0.4993 per hour including disk; earlier validated two-GPU machine 54514 cost $0.6126 per hour. Task 3's run/preserve narrative rounds this to $0.50. These are historical receipts, not a current market quote. [Rental record](https://github.com/Sachin2911/Safety-Dial/blob/1291199e665ed574050de42a7708c19965ecda35/progress/rental-20261022.json), [preservation receipt](../../../../progress/receipts/03-preserve.md)
- The gate upload staged 933 files, approximately 1.50 GB for 192 episodes, or about 7.8 MB per episode; extrapolated video storage is about 23 GB per 3,000 episodes before additional recorder data. [Upload receipt](../../../../progress/receipts/03-preserve.md)
- A 2,000-step horizon with prefix 8 needs at most 250 model queries; prefix 4 can need 500. Prefix changes must report the additional model compute even when the simulator-step contract is identical. [Protocol](../../../../progress/receipts/01-jar-eval.yaml), [existing query checks](../../../../scripts/todo3/score_gate.py)

### Inferences

Historical-cost arithmetic: at the task-3 throughput and $0.4993/h, raw compute is $0.0187 per episode; at five minutes per episode per lane and the historical $0.6126/h host, it is $0.0255. Approximate raw ranges are $5.4 to $7.4 for 288 episodes, $10.8 to $14.7 for 576, $37 to $51 for 2,000, $56 to $77 for 3,000, and $75 to $102 for 4,000. Setup, failed starts, transfer/storage, slower treatments, host availability and idle time are additional. Re-check a real quote before spending; no live pricing was assumed here.

**Suggested spending sequence:** reserve at least $10 to $15 for the mechanism gate and operational overhead. Keep the rest unspent until it passes. For a $50 ceiling, aim for a 1,500 to 2,000-episode development pilot and do not prematurely consume held-out intervention tasks if a credible final comparison cannot fit. For a $100 ceiling, plan a narrow 3,000 to 3,600-episode complete study only if throughput is close to task 3 and the mechanism has a large effect.

One explicit complete-study envelope, illustrative rather than endorsed as adequately powered:

- Gate: 288 development episodes.
- Search: EA, random search and ordinary learning, three independent optimizer seeds each, 192 episodes per run: 1,728 episodes.
- Final test: each of nine learned/search winners gets 108 episodes, nine held-out tasks x 12 fresh seeds: 972 episodes. Native and the fixed setting each get 324 episodes, matched within the three replicate blocks: 648 episodes. Total final episodes: 1,620.
- Grand total: 3,636 episodes, about 136 whole-host hours and $68 raw at task-3 performance; about $93 raw at five minutes/episode and the earlier host rate. Setup and contingency put the latter above $100. Use actual quote/performance to freeze a hard ceiling before launch.

This envelope reveals the constraint: each search run buys only 16 candidates at 12 episodes each, or 32 at six, before spending anything on re-evaluation. That is extremely noisy fitness and very few selection cycles. Three optimizer seeds support showing every result and a descriptive comparison; they do not establish a robust distributional superiority of EA. A scalar gain smaller than about ten points, a large controller, several Pareto-front selections, multiple families, four or five search baselines, or extensive learner tuning will exceed the envelope or dilute inference. A stronger fixed-setting result plus a transparent search pilot is scientifically preferable to a nominally complete but underpowered optimizer contest.

Do not pool the old 96 native episodes into the primary new treatment comparison as if they were contemporaneous controls. They can inform planning and diagnostic plots. New treatments need native episodes interleaved on their host and in their run block. If machine I or machine 54514 is unavailable, report the new host and include native within the new block; do not compare a new treatment directly against an old host's rate.

Freeze actual simulator-step ceilings for search methods, including all candidate, discarded and failed-start interactions, with explicit treatment of the last partially affordable episode. Equal episode count is a simple allocation rule but not exact interaction matching. Report both scheduled maximum and actual consumed steps; do not pad successful episodes with meaningless actions. Final full-horizon comparisons legitimately have different consumed steps because success terminates earlier, and this is a reported efficiency outcome. Model queries and wall time stay separate, including seed probes, multi-candidate policy sampling and shortened prefixes. Any early stop at the first irreversible counted violation is a search-only economy with its censored outputs stated; final episodes run the complete official horizon and preserve the four-class outcome breakdown.

### Gaps

- Current host availability and pricing have not been queried; none of the cost scenarios is a rental quote or purchase authorization.
- Training stability of an ordinary learner over the proposed gate is unknown. It needs a bounded, equal-budget baseline protocol, not an open-ended RL engineering project.
- There is no empirical demonstration yet that the next lever produces candidate rankings stable enough for a 192-episode search run. Passing stage 2 does not automatically pass stage 3.
