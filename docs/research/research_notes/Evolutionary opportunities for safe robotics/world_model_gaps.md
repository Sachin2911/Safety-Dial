# World-model failures and evolutionary opportunities for safe robotics

Research checked live on 16 October 2026. This is a problem-selection memo, not an experiment authorization or a novelty claim. Papers dated after today were excluded. Proposed pilots and stopping criteria below are researcher judgments.

## What did UNISafe solve, and which substantial failures remain?

### Takeaway

UNISafe attacked a concrete weakness of its predecessor: optimistic latent predictions beyond the training distribution. Its contribution combines uncertainty estimation, calibration and control. This was an occupied adjacent area with an unresolved integration problem, not an empty OOD field. Good nominal behavior also does not establish that safety is solved.

### Cited Findings

- UNISafe v2, 23 September 2025, describes world-model hallucinations and missed known or unseen hazards under limited offline coverage. It augments reachability with ensemble epistemic uncertainty calibrated by conformal prediction. Related work explicitly covers existing OOD detectors and approaches restricting systems to in-distribution states. Appendix B.4 says imperfect latent dynamics and RL-derived filters cannot guarantee safety in all cases. Appendix D.3 acknowledges unreliable filtering for entirely different tower configurations. Table 15 reports normal Dreamer safe success/failure/incompletion of 72/20/8%, and hard Dreamer 64/22/14%. These are reported percentages; I did not establish their evaluation denominators. Table 14 varies random-data coverage on Dubins Car, showing UNISafe safety-value balanced accuracy increasing from 0.84 with no random trajectories to 0.97 with 1,000. This is not manipulation evidence. [UNISafe paper](https://arxiv.org/html/2505.00779)
- The current official release includes world-model, uncertainty-ensemble and reachability training entry points, checkpoint links and complete versus success-only data links. It documents ensemble fine-tuning for another 200K iterations. Links and scripts are present; download integrity and training cost were not rerun here. [Official UNISafe repository](https://github.com/CMU-IntentLab/UNISafe)
- Our completed study reports 54/64 successes, seven physical failures and one boundary exit for evolution, versus 2/64 successes and six physical failures for baseline. Fixed/random/PG comparisons match or exceed evolutionary completion. Most benefit survives without filtering. These failures are observed outcomes, not established defects of the filter. [Local oracle](../../../../oracle.md); [completed study](../../reports/UNISafe%20grasp%20maintenance%20and%20evolutionary%20adaptation.md)
- *How Well Do Latent World Models Understand Partially Observable Safety Constraints?* distinguishes failure-state estimation from action-conditioned prediction. On wax hardware, RGB filtering lifts safely in 3/20 trials versus 20/20 with multimodal inputs or privileged multimodal supervision. For rice, 50 filled-bottle trials show RGB predicting low spill likelihood on over half of spill-inducing chunks. After calibration, empty-bottle pouring is allowed in 18% of 50 RGB trials versus 60% of 50 multimodal trials, with both accepting approximately 14% of filled-bottle spill motions. These are controlled tasks, not demonstrations of UNISafe block-plucking failures. [Primary paper](https://arxiv.org/html/2510.06492)
- The MultiSafe project still labels its code link “coming soon.” That link opens a GitHub repository whose default `ghpages` branch contains website files and an academic-template README. I did not verify an executable training release or checkpoints. [Project](https://cmu-intentlab.github.io/multisafe/); [linked repository](https://github.com/CMU-IntentLab/multisafe)
- *When World Models Lie*, submitted 28 September 2026, addresses disagreement signals staying small despite observed prediction error. It estimates latent mismatch with Adaptive Conformal Inference and pessimistically evaluates values over uncertainty sets. Its manipulation results report a safety/completion tradeoff and hypothesize why UNISafe becomes conservative there. These observations concern another task, not our migrated tower. [Primary paper](https://arxiv.org/html/2609.34300); [dated submission](https://arxiv.org/abs/2609.34300)

### Inferences

- The strongest adjacent holes are insufficient predictive information, poor coverage of failure mechanisms, and deployment mismatch. All can leave considerable headroom despite successful nominal filtering. They must be measured separately.
- A model may confidently agree with itself while missing contact physics. Conversely, a perfectly calibrated detector may still lack an effective fallback. A generic “better safety filter” objective hides these different causes.
- Evolution is most defensible when the object is discrete structure, a diverse collection, or a nondifferentiable simulator parameterization. Replacing gradient training of a large differentiable world model with evolution has no established advantage here.

### Gaps

- No live experiment establishes which of these mechanisms causes our residual top-block failures. Neither the 6/64 baseline failures nor the 8/64 evolution physical/boundary outcomes identify prediction, observability, backup-action or timing defects.
- The published hard UNISafe task changes multiple physical/task settings; it cannot by itself isolate friction, mass or perception. Its percentages must not become presumed performance on our stack.

## Which three diagnostic bets give useful headroom and a plausible evolutionary role?

### Takeaway

Prioritize a lightweight offline warning-information audit, then diverse counterexample collection and data repair if the audit exposes coverage failures. Treat physics calibration as a contingent third bet. These are ranked by diagnostic cost and decision value, not by asserted novelty.

For substantive contribution potential, diverse failure-coverage repair is the stronger evolutionary bet here; temporal monitoring is the cheaper feasibility diagnostic with substantial existing methodological overlap. A longer-horizon benchmark may expose more serious temporal/compositional failures, but adds environment and policy integration costs.

### Cited Findings

- Foresight, 22 June 2026, already trains a causal Transformer on action-conditioned world-model features using final success/failure labels, with conformal thresholds. Three-fold evaluation reports balanced accuracy of 0.94 ± 0.06 on LIBERO-Long, 0.80 ± 0.10 on ManiSkill-Long and 0.78 ± 0.02 on BEHAVIOR-1K. The last compares with 0.64 ± 0.05 for SAFE-LSTM. This is rollout classification, not proof of timely preventable physical-hazard warnings. It acknowledges computation and calibration-distribution limits. Its project page says “Code (TBD).” [Primary paper](https://arxiv.org/html/2606.23085); [author project](https://haoranzhangumich.github.io/Forsight_web/)
- Brunello et al., 25 August 2025, already evolve past-time temporal formulas for early failure detection. Their C-MAPSS test has 61 good and 39 faulty traces, obtaining mean F1 0.89 over ten runs and approximately 24.88 seconds average warning in their ablation. Domains are hard drives, chemical plants and engines, not manipulation. The paper provides a code URL, but repeated live opens failed, so accessibility was not verified. [Primary paper](https://arxiv.org/html/2508.17786); [paper-linked code](https://github.com/dslab-uniud/interpretable-failure-detection-gpu)
- Sentinel separates action inconsistency from persistent lack of task progress; its official repository contains the CoRL 2024 monitor implementation. This supplies a non-EA alternative with a different signal model. [Official code](https://github.com/agiachris/sentinel); [primary paper](https://openreview.net/pdf/8a9b36ed31beffc880a260525741f3e2277c447a.pdf)
- MAP-Elites scenario generation is already established in shared-autonomy manipulation. The RSS 2021 primary study searches object positions and human-input parameters, comparing with Monte Carlo and optimization-based testing. Later author synthesis explains why uniform parameter sampling can give poor behavioral coverage, while emphasizing realism constraints and evaluation cost. [RSS primary paper](https://roboticsproceedings.org/rss17/p036.pdf); [author synthesis, 2024](https://arxiv.org/html/2409.04711)
- DataMIL already selects robotics training trajectories using regression/metagradient influence, temporal clustering and target validation loss. It evaluates more than 60 manipulation tasks. This concerns imitation policy data, rather than safety-model coverage, but rules out claiming that robotics data selection is new. Its official release provides MetaWorld/LIBERO scripts and linked datasets, with Python 3.10, older CUDA/JAX and optional four-GPU configurations. [Author explanation](https://robin-lab.cs.utexas.edu/datamodels4imitation/); [official code](https://github.com/UT-Austin-RobIn/datamil)
- RENEW, 15 July 2026, repairs model exploitation with preference-supervised dynamics updates and uncertainty-directed queries. Its experiments are Jumanji/classic control; preference labels use a synthetic ground-truth oracle. This is neither a verified manipulation result nor an evolutionary method. An official code repository is accessible. [Primary paper](https://arxiv.org/html/2607.14180); [official code](https://github.com/FlyingWorkshop/RENEW)
- DROPO already uses CMA-ES to fit physics-parameter distributions from offline trajectories. Its Panda real pushing calibration uses one trajectory, 1,286 transitions, 12.86 seconds; these figures describe target-data acquisition, not optimization or policy-training cost. DROID is another evolutionary system-identification precedent. SimOpt uses relative-entropy sampling updates, while BayesSim supplies a non-EA inference comparison. The current DROPO repository documents old Gym APIs and MuJoCo state restoration, with released Hopper data. It is not an Isaac Lab drop-in. [DROPO paper](https://arxiv.org/html/2201.08434); [official code](https://github.com/gabrieletiboni/dropo); [SimOpt primary paper](https://arxiv.org/html/1810.05687)

### Inferences

#### 1. Offline temporal warning-information audit

**Problem hypothesis:** brief scores miss persistent slip/contact precursors, or disagreement misses consistently wrong predictions. There may be a large gap between detecting a fallen block and predicting a preventable fall. Published temporal and world-model monitors make this an occupied problem with credible tools.

**Pilot:** freeze the world model and extract current failure score, uncertainty, safety values, one-step executed-action prediction residuals and deployment-available proprioception. Diagnose physical hazard onset independently using fixed simulator labels. Compare a single threshold, hysteresis, consecutive-threshold/EMA rules, logistic regression on history, and a small GRU. Plot event recall at a fixed per-episode false-alarm rate against required warning lead time. No simulator or world-model training is necessary for this first audit if sufficient recordings exist.

**EA object:** only after a baseline gap appears, use GP to discover short temporal formulas over those histories, penalizing size and emphasizing early warnings on different failure mechanisms. Discrete temporal operators and variable windows provide a natural structural search role. The contribution must concern transferable early intervention value, latency or compactness against sequence learners; GP monitoring itself already exists.

**Cheap falsifier:** if calibrated single-score thresholds or an EMA match the candidate temporal method, stop. Also stop if warnings only appear after physical violation, or if all useful input comes from privileged object state unavailable to deployment. A larger offline AUC without actionable warning is insufficient.

**Confounds:** failure labels may already include upstream pre-failure relabeling; elapsed time can become a shortcut; a terminal camera can leak outcome. Causal histories must exclude future samples. World-model residual after action execution cannot predict the consequence of that same action retrospectively. A warning has value only if some feasible response at that time changes the physical outcome.

**Long-horizon extension:** a temporal monitor could track persistent hazards or obligations spanning action chunks and subtasks. GP can search that memory structure, but cannot supply missing observations, a valid task decomposition or an effective recovery controller by itself. Measure whether composition creates a predictable hazard before selecting monitor grammar.

#### 2. Diverse counterexample collection followed by data-budgeted safety-model repair

**Problem hypothesis:** the model/filter misses several distinct rare contact failures, while ordinary random collection yields redundant safe behavior. The useful target is not simply maximizing the number of failures, but obtaining informative coverage per interaction and improving a held-out predictor/control outcome.

**Diagnostic:** begin with a small factorial or Latin-hypercube sweep of physically valid mass, friction, reset pose and bounded occlusion families. Keep safety definitions fixed. Identify repeated “confidently safe then physically unsafe” cases, “uncertain but physically safe” cases and fallback failures separately. This diagnoses whether uncertainty is correlated with consequential prediction error before any EA.

**EA object:** a MAP-Elites archive of scenario/short-probe parameters, with separate diversity descriptors for contact geometry, initial support margin and observed failure mechanism. Search fitness should prefer verified predictive blind spots and distinct mechanisms, with validity and feasible-completion constraints. Alternatively evolve sampling proportions over fixed trajectory strata to train a small frozen-encoder hazard head. Selection fitness is held-out false-safe rate at matched false-alarm budget, not training loss alone.

**Comparisons:** grid/Latin-hypercube/random sampling, uncertainty-ranked acquisition, farthest-point feature selection, class-balanced strata and influence selection. Nearest EA precedent is QD scenario generation, not an established demonstration that EA safety-data selection wins. Nearest non-EA precedents include DataMIL and RENEW. A useful contribution could show that uncertainty acquisition overlooks confidently wrong contact modes and a diverse archive repairs them more economically.

**Cheap falsifier:** stop if random/farthest-point/class-balanced data reaches the same held-out failure coverage with the same acquisition and training budget. If the task is impossible from generated starts, or search merely finds renderer artifacts, reject those cases. Test newly drawn parameter combinations rather than only archive neighbors.

**Confounds:** mutation can overfit stochastic trajectories; simulator metadata used for archive construction is an information advantage even if absent from runtime features. Count failed, discarded and validity-check rollouts. Match total training updates and candidate-selection compute, not only final selected-data count. A repaired classifier is not a repaired world model; a new safety model may require retraining reachability before closed-loop claims.

#### 3. Physics-conditioned residual repair or identification

**Problem hypothesis:** changed friction/mass or contact loss produces systematic action-conditioned error that a small context estimator can explain. This requires a clean physical-shift effect first, not the published hard-task aggregate.

**EA object:** estimate a few bounded physics parameters or a distribution using CMA-ES and simulator likelihood/mismatch, then condition a small residual dynamics module or select a controller. The nondifferentiable contact simulator justifies gradient-free search, but DROPO/DROID already occupy this method family. Novelty would need safety-specific identifiability, uncertainty or adaptation evidence.

**Diagnostic/falsifier:** hold initial geometry, policy, cameras and outcome predicates fixed; vary one physical factor. Ask whether a few harmless probe transitions permit a linear residual/context model or simple speed/grip correction to recover prediction and completion. If no systematic error arises, or ordinary least-squares/gradient adaptation matches the benefit, abandon EA identification. Never feed true test mass/friction to one method alone.

**Confounds:** latent mismatch can reflect changed observation statistics rather than physics; several physical parameters can explain the same trajectory. Restoring object pose alone may omit contact-solver state. Calibration rollouts and retraining costs count. I rank this third because simulator inference plus filter retraining can consume substantial work before identifying a remaining contribution.

### Gaps

- No source proves that any candidate supplies a 20%-to-80% opportunity on our assets. That numerical improvement is an aspiration, not a measured premise.
- I found no verified primary demonstration of evolutionary trajectory-subset selection specifically repairing UNISafe or visual latent safety filtering. This is a limited search result, not evidence of novelty or absence.

## What can begin cheaply with current code, and what would justify continuing?

### Takeaway

The first useful deliverable is a mechanism audit with strong simple baselines. Select a method after identifying whether the missing ingredient is warning history, information, data coverage or changed dynamics.

### Cited Findings

- Our inference script exports frozen features/actions and predictions without simulator interaction. It explicitly flags upstream label timing as unknown because of pre-failure relabeling and terminal substitution. [Inference implementation](../../../../experiments/scripts/unisafe_inference.py)
- The controller records proposal, backup, selected and executed actions, uncertainty, safety values and failure scores. Diagnostic prediction errors are optional. The worker records physical states and outcomes, but full cameras are conditional on `record_cameras`; latent histories are not automatically saved in every run. [Controller](../../../../experiments/helpers/unisafe_controller.py); [worker](../../../../experiments/scripts/unisafe_adaptation_worker.py)
- The current local workspace has no `third_party/UNISafe` checkout at the expected path during this audit. Public assets and remote run-machine artifacts must be checked before claiming an immediately available full training pool. [Asset-path assumptions in loader](../../../../experiments/helpers/unisafe_model.py); [setup](../../../../scripts/setup_unisafe.sh)

### Inferences

- Treat all previously inspected reserved traces as development data for a new monitor. Group variants sharing a reset seed/source trajectory across train, selection-validation and test; otherwise nearly identical starts can leak across splits. Freeze a newly reserved physical evaluation later.
- Use ground-truth simulator states for scoring and causal diagnosis only. Runtime features must be available to the robot. New tactile/force information is a separate sensing intervention that every comparison must share.
- Continue only if simple history baselines leave substantial early-warning or coverage headroom, the failure has a feasible corrective response, and the proposed evolutionary representation addresses that mechanism. These criteria prevent another large search study whose main benefit comes from an easy fixed rule.

### Gaps

- Remote recording availability, numbers of independent physical-failure mechanisms and runtime latency for new diagnostics remain unmeasured. The 1,424 prior episodes are not 1,424 independent training conditions, and their label/camera availability must be inventoried before an offline protocol is fixed.
