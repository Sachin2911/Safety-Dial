# Evolutionary policies and explicit manipulation safety

## Which direct manipulation precedents are verified?

### Takeaway
Two close hardware precedents are SafeAPT and Chen et al.'s composable industrial insertion framework. Both combine evolutionary optimization with explicitly supplied safety information and additional mechanisms. They do not demonstrate safety emerging without specifications.

### Cited Findings
- **Rituraj Kaushik, Karol Arndt and Ville Kyrki, SafeAPT: Safe Simulation-to-Real Robot Learning Using Diverse Policies Learned in Simulation. IEEE Robotics and Automation Letters 7(3), 6838-6845, July 2022.** DOI 10.1109/LRA.2022.3177294. The author institution verifies publication metadata. [Aalto publication record](https://research.aalto.fi/en/publications/safeapt-safe-simulation-to-real-robot-learning-using-diverse-poli/).
- SafeAPT evolves a policy repertoire with MAP-Elites. Crucially, offline evolution **does optimize safety**: goal-space cells retain the candidate with the highest minimum safety score across sampled dynamics. Online adaptation then uses Gaussian-process reward/safety models and constrained Bayesian selection. Kuka hockey safety is puck clearance from the table edge, with a supplied 0.1 m limit. The safety-score function, simulated dynamics distribution, repertoire description, confidence parameter and GP settings remain designed. The probability argument assumes accurate GP mean/variance. The paper evaluates three simulated tasks and hardware hockey; hardware reports no observed violations in eight repeats, each with a 15-episode budget. Thus safety is a joint contribution of the evolved repertoire and online safety modelling/selection. [Paper, Sections III-VI](https://arxiv.org/html/2201.13248).
- **Rui Chen, Chenxi Wang, Tianhao Wei and Changliu Liu, A Composable Framework for Policy Design, Learning, and Transfer Toward Safe and Efficient Industrial Insertion. IROS, October 2022.** The author institution verifies title, authors and venue. [CMU publication record](https://publications.ri.cmu.edu/a-composable-framework-for-policy-design-learning-and-transfer-toward-safe-and-efficient-industrial-insertion).
- Chen et al. use CMA-ES to tune parameters of a modular insertion policy and controller, with simulation pretraining of an inference module and hardware optimization. Hardware uses a UR10 inserting **IC sockets**, chosen as more robust than IC chips. Safety means limiting component damage. A hand-designed state machine retracts after force-detected collision; a Cartesian tracker imposes velocity, acceleration and orientation constraints. The demonstration uses a 2 N collision alarm, yet transient forces reach below 20 N; the authors report no collision-induced deformation in that example. The objective retains manually selected coefficients: insertion time squared plus collision, peak-force and failure penalties (Eq. 5). Parameter bounds and modular behaviours remain designed. The paper reports automatic tuning benefits, but holds training configuration fixed and leaves changing-environment learning to future work. Transfer to other manipulation tasks requires redesigning the insertion/inference modules. [Paper, Sections IV-V](https://arxiv.org/pdf/2203.03017).

### Inferences
- SafeAPT is especially close to research on safety-aware evolutionary repertoires and safe adaptation. Calling its online algorithm evolutionary would obscure the constrained Bayesian optimization stage.
- The insertion paper is directly relevant to a reduced-tuning contribution: it explicitly motivates evolutionary search through difficult coupled parameters and manual tuning effort. It is not a precedent for removing hand-crafted safety penalties or behaviour rules.
- Neither source supports unrestricted safe manipulation or establishes that evolutionary optimization is uniquely responsible for the entire safety result.

### Gaps
- The selected publications do not establish the specific combination of frozen learned world models, compact evolved adapters, outcome-only safety feedback and measured total human-design reduction. This bounded search cannot establish that combination is novel.
- We have not completed forward/backward citation screening sufficient for a systematic review or priority claim.

## What is a useful adjacent contact-control precedent?

### Takeaway
Evolution has also automated physical interaction controller tuning, but such evidence should be labelled contact-control research rather than general object-manipulation safety.

### Cited Findings
- **Nicholas A. Nadeau and Ilian A. Bonev, Evolutionary Motion Control Optimization in Physical Human-Robot Interaction. IROS 2018, 1347-1353**, DOI 10.1109/IROS.2018.8593598. Differential evolution tunes eight parameters of parallel force/impedance control on a Kuka LBR iiwa following a human-leg phantom for an ultrasound application. The institution reports contact-force mean absolute error of 0.537 N. This is hardware phantom evidence for automatic tuning, not patient safety validation or a general damage-free manipulation claim. [Author institution record](https://pure.etsmtl.ca/en/publications/evolutionary-motion-control-optimization-in-physical-human-robot-/).
- Another related example is **Lahr et al., A hybrid model-based evolutionary optimization with passive boundaries for physical human-robot interaction, ICRA 2022**. It combines genetic optimization and model-derived passivity bounds for a one-degree-of-freedom wrist rehabilitation device, with three subjects. Its protection mechanism is a restricted passive controller domain. This is adjacent pHRI, not a direct object-manipulation result. [Author manuscript](https://arxiv.org/abs/2203.00458).

### Inferences
- These works show that automatic controller tuning under an explicit safety envelope predates the current project. The distinction between tuning behaviour and defining its admissible envelope is central.

### Gaps
- Detailed analysis of these medical/rehabilitation methods is outside this manipulation-policy search. No clinical effectiveness or injury-prevention claim is made here.

## How broad was this search and what was excluded?

### Takeaway
This is a selective primary-source search conducted on 3 November 2026, not a systematic literature review. Queries included explicit 2023-2026 terms, but the closest fully verified direct policy-learning precedents found here were published in 2022.

### Cited Findings
- Search families combined evolutionary/neuroevolution/CMA-ES/MAP-Elites/quality-diversity with robotic manipulation, safety, safe policy learning and physical interaction. SafeAPT-related searches and its author publication record were followed. Primary venues encountered include RA-L, IROS, ICRA and Robotics and Autonomous Systems. Full text was inspected for SafeAPT and Chen et al.; metadata came from author/institution pages.
- **SafeMimic, RSS 2025** is a strong safe mobile-manipulation neighbour: it adapts actions from a human demonstration and uses safety Q-functions and backtracking. The verified project description does not identify an evolutionary optimizer, so it is not counted as a direct EA precedent. [Author project](https://robin-lab.cs.utexas.edu/SafeMimic/).
- **Winter et al., A context-adaptive policy framework for robust and reactive robotic manipulation via uncertainty-aware imitation learning, Robotics and Autonomous Systems, July 2026** is a recent near-match. Publisher text identifies CMA-ES hyperparameter tuning, while the institutional abstract describes uncertainty-aware imitation and hardware force-conditioned grasping/deformable-food manipulation. Its explicit safety evaluation was not fully verified here, so it is not counted as a direct safety result. [Author institution record](https://elib.dlr.de/225946/); [publisher](https://www.sciencedirect.com/science/article/abs/pii/S0921889026003210).
- SafeEvolve search results concern LLM-agent harness/policy safety, not physical manipulation; general QD grasping papers are treated separately by the adjacent-work researcher. Navigation, locomotion and trajectory-only optimization were excluded from this direct-policy category.

### Inferences
- Lack of another fully verified recent direct match in these searches is a search outcome, not evidence that none exists.
- The literature is distributed across safe adaptation, industrial assembly, physical interaction and constrained control terminology. Searching only the phrase safe evolutionary manipulation would miss important precedents.

### Gaps
- No comprehensive Scopus/Web of Science citation export or exhaustive venue-by-venue screening was performed. This map should guide deeper reading, not support a first-work claim.
