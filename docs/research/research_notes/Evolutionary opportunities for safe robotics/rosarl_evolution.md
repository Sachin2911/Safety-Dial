# ROSARL and evolutionary opportunities in safe robotics

Research screening conducted 16 October 2026. These notes distinguish published results, implementation observations and proposed diagnostics. No experiment was run and no existing research record was changed.

## What does ROSARL solve, and which gaps are substantive?

### Takeaway

ROSARL is useful for specifying the relative priority of task reward and avoidance of declared unsafe terminal outcomes. It is not a learned safety predictor, a definition of physical harm, or a mechanism guaranteeing safe exploration. The strongest accessible question is finite-budget learning with rare failures, but evolutionary penalty tuning is a weak standalone bet.

### Cited Findings

- **Theorem scope:** The original paper studies undiscounted stochastic-shortest-path MDPs with bounded rewards, absorbing outcomes and deterministic proper policies. Unsafe absorbing states must already be specified. Its safety definition minimizes unsafe-termination probability, which can remain positive. The stated penalty boundary uses reward bounds, diameter and controllability: `min(Rmin, (Rmin-Rmax)D/C)`, for `C>0`. It concerns optimal policies, rather than intermediate learned policies. Stochastic transitions are explicitly covered. Partial observability, finite-budget confidence and multiple severity constraints are not established by this result. The paper explicitly identifies non-terminal violations and degrees of safety as limitations. Continuous-control evidence is from modified Safety Gym navigation, not contact-rich manipulation. It reports lower failure rates than TRPO, TRPO-Lagrangian and CPO. The practical stochastic estimator is empirical rather than a finite-sample certification. [ROSARL v1, 31 May 2023](https://arxiv.org/html/2306.00035v1)
- **Implementation observation:** The official `MinMaxPenalty` tracks historical extrema of raw rewards and supplied value estimates, returning `min(rmin, vmin-vmax)`. The caller supplies an unsafe indicator and replaces that transition's reward. The class does not infer unsafe labels, estimate controllability explicitly, provide a confidence interval, reset extrema automatically after distribution changes, or perform safety filtering. [Author code](https://github.com/geraudnt/rosarl/blob/main/learning_minmax_penalty.py)
- **Assets and reproduction scope:** The author repository contains lavaworld, modified Safety Gym, theoretical examples and TRPO-Minmax integrations. Its instructions use legacy OpenAI MuJoCo and safety-starter-agents; this is reusable reference code, not a verified Python 3.12 installation. [Official repository](https://github.com/geraudnt/rosarl)
- **Version caution:** arXiv currently lists only v1. A separately indexed OpenReview PDF has the same ROSARL title and an ICLR 2026 submission header. Full PDF and forum fetches returned a browser challenge, so its authors, revision history, acceptance status and changed claims were not verified. Do not merge it with the original theorem or describe ROSARL as an accepted ICLR 2026 paper. [arXiv history](https://arxiv.org/abs/2306.00035); [unverified newer submission](https://openreview.net/pdf?id=qcz3g6mH3L)

### Inferences

- **Specification remains indispensable.** The implementation requires an external unsafe-state decision. In robotics this still leaves the hard work of deciding whether tower tilt, a dropped block, contact force or human proximity constitutes a violation. Searching over those definitions to improve a score could simply redefine away harm. Keep evaluation definitions fixed. This follows directly from the caller-supplied unsafe indicator. [Author code](https://github.com/geraudnt/rosarl/blob/main/learning_minmax_penalty.py)
- **Model error is a separate failure channel.** A perfectly chosen terminal penalty cannot make an image encoder reveal hidden contact or make a world model accurately extrapolate unfamiliar friction. Reward specification and prediction quality should therefore have separate diagnostic measurements. This is a proposed decomposition, not a ROSARL theorem.
- **Finite-budget calibration is a meaningful hypothesis.** With the official extrema-based estimator, one erroneous extreme value can influence subsequent penalties indefinitely; conversely, unvisited reward regimes contribute nothing. This suggests testing critic error, rare failures and reward-scale changes before inventing a new optimizer. These are code-based hypotheses, not demonstrated ROSARL defects in manipulation. [Author code](https://github.com/geraudnt/rosarl/blob/main/learning_minmax_penalty.py)
- **Feasibility must be measured independently.** If every available behavior fails under a disturbance, a larger penalty cannot create a feasible controller. A simple scripted or privileged-state controller is a useful diagnostic of whether apparent learning headroom includes physically solvable cases. Such a controller is a diagnostic upper reference, not a fair deployed baseline.
- **Avoid a theorem extension as an accidental prerequisite.** A robotics claim about partially observed continuous policies, deadlines or graded non-terminal damage would need its own analysis. The scalar transformation can still be tested empirically, with its guarantee scope left explicit.

### Gaps

- No verified numerical manipulation failure floor or finite-budget deficit specific to ROSARL was found. “ROSARL plus evolution” currently names components, not a diagnosed problem.
- Before deriving a new bound, independently audit the original definition of controllability, policy equivalence, positive-gap assumption and converse quantifiers. I did not complete proof verification or verify whether the newer submission revises them.
- I found no primary evidence here that the practical estimator remains calibrated under latent-state aliasing, changing task rewards or learned dynamics. Absence from this bounded search does not establish novelty.

## Which evolutionary and non-evolutionary precedents rule out naive contributions?

### Takeaway

Evolution already searches safety-aware behaviors, adapts return-versus-cost weighting, generates difficult curricula and learns temporal warning rules. The most credible role for evolution is a structured search whose outputs improve useful evidence or a controller portfolio. Merely replacing a standard optimizer is heavily anticipated.

### Cited Findings

- **Safety-informed behavior search:** Marchesini and Amato use visited unsafe states to bias policy mutations, evaluate the offspring, and replace a gradient-trained policy with a similar-return, lower-cost individual. Their GECCO 2022 study covers Safety Gym locomotion and Turtlebot navigation. Thus safety-aware evolutionary mutations and hybrid EA/RL policy improvement are established precedents. [Author paper](https://emarche.github.io/assets/pdf/GECCO2022_safemut.pdf)
- **Statistical safe neuroevolution:** Belzner and Wirsing's Safe Neural Evolutionary Strategies combine policy search with Bayesian verification. Constraint satisfaction probability and confidence influence return-versus-cost weighting. Their examples are Particle Dance and Obstacle Run. This directly anticipates an EA using confidence-aware safety selection or adaptively chosen penalty weights. It does not provide universal physical safety. [2021 primary article](https://doi.org/10.1016/j.scico.2021.102620); [official PSyCo code](https://github.com/lenzbelzner/psyco)
- **Failure curriculum:** Song and Schneider's ICRA 2022 Genetic Curriculum evolves variable-length scenario encodings that reveal current failures. In Table III, BipedalWalkerHardcore failure falls from `10.20%` to `3.96%`, and LunarLander from `5.19%` to `0.64%`. The random-failure curriculum ablation on BipedalWalkerHardcore reaches `24.50%`, versus `3.96%` for their curriculum. These are their benchmarks, not predicted manipulation gains. The main comparison is by training epochs, with simulator-interaction accounting additionally investigated on LunarLander. “Evolve hard training situations” is therefore insufficient novelty. [Primary paper](https://arxiv.org/html/2202.08393v1)
- **Safety-first curriculum without EA:** ICLR 2025 Safe Curriculum Generation prioritizes tasks where the learner can be safe and performant, rather than first maximizing difficulty or reward. It reports improved training safety and sample efficiency against curriculum baselines. Its released tasks include safety maze, goal and passage settings. [Conference paper page](https://proceedings.iclr.cc/paper_files/paper/2025/hash/120ed726cf129dbeb8375b6f8a0686f8-Abstract-Conference.html); [author code](https://github.com/cevahir-koprulu/safe-curriculum)
- **Adaptive data distribution without EA:** Active Domain Randomization learns to sample informative simulation variations using discrepancies between reference and randomized rollouts. Its experiments include simulated and real robotics. Prioritized Level Replay samples previously encountered levels according to learning potential and staleness. These rule out uniform sampling as the only serious comparison. [ADR primary paper](https://proceedings.mlr.press/v100/mehta20a.html); [official ADR code](https://github.com/montrealrobotics/active-domainrand); [official PLR code](https://github.com/facebookresearch/level-replay)
- **Temporal rules already evolved:** Brunello and colleagues' 2025 framework uses grammar-based multi-objective evolution to learn pure-past temporal properties for early failure detection. It reports improvements on Backblaze, C-MAPSS and Tennessee Eastman Process datasets. Its formal result concerns monitor semantics and efficient trace checking, not physical correctness of inferred warnings or successful intervention. [Primary paper and code link](https://arxiv.org/html/2508.17786v1)
- **Penalty dynamics without EA:** PID Lagrangian methods address multiplier overshoot and oscillation and provide reward/cost-scale invariance. Sauté RL augments state with safety-budget information, allowing policies to condition on constraints. Any proposal involving adaptive penalties, accumulated damage or severity budgets should compare against these ideas before claiming an empty space. [PID primary paper](https://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf); [Sauté RL primary paper](https://proceedings.mlr.press/v162/sootla22a.html)

### Inferences

- The genetic-curriculum ablation warns against an attractive but weak objective: finding the largest number of failures. Training on failures is not automatically useful. A candidate should yield learnable, diverse evidence and improve held-out closed-loop outcomes after repair.
- The temporal-monitor direction has a strong, natural EA representation, but the basic algorithmic idea is occupied. Its contribution would need a manipulation-specific failure mechanism, advance warning with deployed signals, and evidence that responding to the warning avoids real failures.
- A useful evolutionary study can still use a standard EA. The burden shifts to the problem definition, search representation, measurement and demonstrable advantage over the closest remedies.

### Gaps

- I did not verify an official Genetic Curriculum implementation or runnable pretrained assets. The paper itself is sufficient to establish prior art, but reproduction would require an additional asset audit.
- The monitoring paper supplies an author repository link, but direct retrieval failed during this audit. Its reproducibility claim is verified in the paper; installation and runtime are not.
- No source in this screening supplies an untouched, contact-rich manipulation benchmark proving large headroom for these exact proposed roles.

## Which narrow questions deserve cheap diagnostics first?

### Takeaway

Three hypotheses merit screening, in descending order below. None yet justifies a full experiment campaign. The first diagnostic should establish a consequential prediction or coverage gap using the migrated assets, with simple remedies and physical ground truth.

### Cited Findings

- The repository's current oracle records that grasp correction substantially improves completion, that simpler methods match or exceed evolution, and that most improvement persists without filtering. This is the local reason to abandon filter-induced continuation as an assumed mechanism. [Current oracle](../../history/20261018/oracle.md)
- Failure-focused curricula, safe curricula and temporal-rule discovery have verified precedents above. They are foundations and mandatory comparison targets, rather than evidence that any of the following questions is novel. [Genetic Curriculum](https://arxiv.org/abs/2202.08393); [SCG](https://proceedings.iclr.cc/paper_files/paper/2025/hash/120ed726cf129dbeb8375b6f8a0686f8-Abstract-Conference.html); [monitoring](https://arxiv.org/abs/2508.17786)

### Inferences

#### Candidate 1: Does failure diversity improve safety-model repair more than failure count?

**Question and hypothesis:** At a fixed simulator and retraining budget, can a diverse archive of physically distinct, confidently missed failures improve a frozen-representation safety predictor more than random or uncertainty-selected trajectories? Search should target a failure-prediction deficiency, not ordinary task-policy weakness. This is a hypothesis, not observed UNISafe headroom.

**EA search space:** Small, physically valid variation sets, such as friction, block mass, grasp offsets, bounded observation delay and occlusion. Use a quality-diversity archive with descriptors tied to failure mechanisms. First audit which variations can be changed without invalidating reset states or pretrained observation semantics. Acquisition fitness combines a measured physical violation with confident predicted safety, while archive coverage prevents collecting duplicates. Filtered action histories must remain distinct from task proposals.

**Closest prior art and baselines:** Genetic Curriculum already generates failures; ADR and PLR already select informative domains. Compare equal-budget random perturbations, stratified boundary sampling, maximum uncertainty, and maximum false-negative score without diversity. A non-EA greedy diverse selection from the same acquired pool is essential. The possible contribution is the difference between uncertainty, confidently wrong predictions and physically diverse errors in contact safety, followed by successful repair.

**Cheap diagnostic and falsifier:** Replay existing physical failures and matched non-failures through the predictor, measuring missed-failure rate at a fixed false-alarm rate and warning horizon. Then a small stratified perturbation sweep should test whether errors concentrate in more than one reproducible, solvable failure mode. Stop if errors vanish with basic calibration, are solely rendering artifacts, lack physical labels, or are fixed equally well by adding a few random examples. A large candidate pool is not free: count its acquisition plus every discarded training run.

**Assets:** The migrated UNISafe controller and evidence are local. ADR and PLR official repositories are verified. No newly trained safety model or necessary raw local dataset has been verified on the run machine in this screening.

#### Candidate 2: Do short temporal warnings recover failure information lost by instantaneous predictions?

**Question and hypothesis:** Can a small temporal rule over deployable histories detect impending slip or tower instability earlier at the same false-alarm rate as a one-step predictor? The EA representation is naturally structured, but only useful if a response remains possible after warning.

**EA search space:** Genetic programming over bounded past-time operators, thresholds and a small input set: measured aperture, proprioception, image-derived motion and existing safety/uncertainty outputs. Simulator-only tower pose may supply labels or a diagnostic upper reference; it must not silently become a deployed signal.

**Closest prior art and baselines:** Reuse the 2025 GP monitor framework as prior art. Compare a threshold, moving average, persistence/hysteresis rule, logistic regression over identical history features, and a small recurrent predictor. The claimed contribution cannot be “GP discovers temporal safety rules.”

**Cheap diagnostic and falsifier:** On trajectory-disjoint splits, plot warning recall against false alarms and lead time. Include failures hidden by intervention censoring. Stop if observable change begins only after unavoidable failure, a persistence rule matches performance, or the original predictor already uses the relevant history effectively. If offline prediction is promising, a scripted abort/regrasp response must improve measured physical outcomes before calling the monitor a safety improvement.

**Assets:** An author monitoring repository is linked by the paper, although not fetched successfully. Existing UNISafe histories are a possible starting point; availability and sufficiency of raw deployable signals still require checking.

#### Candidate 3: Can curriculum repair finite-budget ROSARL learning under rare terminal failures?

**Question and hypothesis:** Does deliberately acquiring near-failure but solvable examples improve time to safe completion when ROSARL's practical value-based penalty estimate is initially inaccurate? Here ROSARL supplies the terminal reward rule; evolution searches the training distribution rather than inventing safety definitions.

**EA search space:** A compact distribution over start configurations and bounded disturbance timings, with task feasibility and failure-mechanism diversity recorded separately. Freeze the reward rule first. Compare uniform training, random-failure replay, PLR-style replay and safety-first curriculum before introducing evolution. If subsequently tuning penalties, include fixed sweeps and PID-style adaptation; SNES already occupies confidence-weighted evolutionary safety selection.

**Cheap diagnostic and falsifier:** Start in an inexpensive small environment, without Isaac integration. Compare the exact or carefully swept penalty reference with the online estimator under bounded reward rescalings, rare terminal failures and stochastic transitions, using identical total interactions. Measure cumulative training violations, completion, final failure rate and penalty error. Stop if the estimator matches the reference at the practical budget, if all methods simply need more policy capacity, or if a simple replay rule closes the gap. Establishing this deficit would be useful; finding one is not assumed.

**Assets:** ROSARL and SCG official source repositories are verified. Their dependencies differ from the current platform and have not been installed. This proposal needs ordinary learning, a value estimator and new training runs, so it is less immediately accessible than diagnosing existing trajectories.

### Gaps and recommendation

First screen **confidently missed physical failures and failure-mode coverage** using existing evidence. It can reveal whether the real bottleneck is a safety predictor, missing information or ordinary task competence before any evolutionary search. If that screen finds no substantial gap, move to another task or system rather than manufacture one by extreme perturbations.

Deprioritize standalone evolutionary penalty tuning, broad neuroevolution of pretrained weights, and a GP monitor without recoverable warning lead time. Keep ROSARL optional until a terminal-reward or finite-budget-learning problem is measured. The current evidence supports a search for a missing capability, not a commitment to any of these three solutions.

### Source register

Primary sources verified above: ROSARL arXiv v1 (2023) and author code; Safety-Informed Mutations (GECCO Companion 2022); SNES/PSyCo (Science of Computer Programming 2021) and author code; Genetic Curriculum (ICRA 2022); SCG (ICLR 2025) and author code; ADR (CoRL 2019, PMLR 2020) and author code; PLR official code; interpretable temporal monitoring (arXiv v1, 25 August 2025); PID Lagrangian (ICML 2020); Sauté RL (ICML 2022). The separate ROSARL OpenReview submission is explicitly unverified beyond its indexed title and submission header.
