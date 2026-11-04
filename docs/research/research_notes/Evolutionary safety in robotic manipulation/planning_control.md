# Evolutionary manipulation planning and control

## Which direct engineering precedents are relevant?

### Takeaway

Evolutionary collision-avoiding arm motion is established prior work. The strongest directly relevant human-safety example found here combines a genetic trajectory optimizer with a separately supplied runtime safety controller. The papers concern generated motions and their adaptation, rather than a general learned manipulation policy.

### Cited Findings

- **Abo-Hammour, Alsmadi, Bataineh, Al-Omari and Affach, “Continuous Genetic Algorithms for Collision-Free Cartesian Path Planning of Robot Manipulators,” International Journal of Advanced Robotic Systems 8(6), 2011.** A continuous GA evolves joint paths to approximate a supplied Cartesian path, respect joint limits and avoid stationary modeled obstacles. Evidence consists of planar 2R and PUMA 560 simulations. Links use capsule-like geometry and obstacles use spheres. Safety remains explicitly engineered: the fitness includes collision and clearance penalties; the paper sets penalty components to 0.3, 0.5 and 0.2, and a clearance threshold to 10 mm. This is a clear precedent for using evolution to discover a motion, and a counterexample to equating evolution with removal of hand-tuned safety objectives. The abstract's broad “safest” wording should be bounded to its geometry, fitness and simulation evidence. [Publisher full text](https://journals.sagepub.com/doi/10.5772/50902).

- **“Collision free trajectory planning for hybrid manipulators,” Mechatronics 22(6), 836–851, 2012.** The method combines RRT-generated feasible paths with GA optimization of collision status, path length and distance from obstacles. It separates displacement and manipulation phases. The proposed system combines an industrial SCARA with CaPaMan, a parallel manipulator, and includes experimental validation. The accessible publisher account notes that much experimental testing focused on CaPaMan and uses accelerometer instrumentation. Treat this as physical engineering validation of a particular trajectory-generation arrangement, not evidence of general safe policy learning. Detailed penalty weights could not be checked because full-page retrieval failed. [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S095741581200075X); [DOI](https://doi.org/10.1016/j.mechatronics.2012.05.001).

- **Andrea Maria Zanchettin, Costanza Messeri, Davide Cristantielli and Paolo Rocco, “Trajectory optimisation in collaborative robotics based on simulations and genetic algorithms,” International Journal of Intelligent Robotics and Applications 6, 707–723, 2022.** A GA mutates instruction-level trajectory parameters, including waypoints, speed, interpolation type and blending. Candidates are evaluated in a digital twin. Selection prioritizes an occupancy-based collision-risk proxy, then traversal time. A physical ABB YuMi and human perform an assembly task. Crucially, Section 3.2 states that Speed and Separation Monitoring remains active during execution; the optimizer uses a probabilistically relaxed representation for candidate evaluation. The risk quantity is constraint slack under estimated occupancy, not a measured general injury probability. Explicit safety constraints, a designed initial trajectory and tunable mutation variances remain. This is a direct precedent for evolutionary adaptation that reduces disruption from safety interventions while retaining an independently designed safety mechanism. [Full paper, published 30 May 2022](https://link.springer.com/article/10.1007/s41315-022-00240-4).

- **Suping Zhao, Yushuang Du, Chaobo Chen, Xiaohua Song and Xiaoyan Zhang, “Multitask-Based Anti-Collision Trajectory Planning of Redundant Manipulators,” Biomimetics 9(11), 679, 2024.** A special-encoding GA optimizes task-area order, choices of joint configuration and polynomial trajectory coefficients. Collision detection uses an explicit simplified spatial model. The fitness considers end-effector path length and joint-angle variation together with collision handling. Evidence is numerical simulation, with predefined obstacles. This is a recent example of evolutionary search across discrete sequencing and continuous motion parameters. It does not establish a reusable policy that learns the intended meaning of safety. [Original paper archived in PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11591824/); [author abstract and publication metadata](https://pubmed.ncbi.nlm.nih.gov/39590251/); [publisher DOI](https://doi.org/10.3390/biomimetics9110679).

### Inferences

- A claim that evolutionary algorithms can find collision-avoiding manipulator motions is already occupied by these precedents.
- The Zanchettin paper is particularly useful to read alongside SafeAPT: it makes the boundary between evolutionary adaptation and supplied execution-time protection visible.
- Task sequencing in a trajectory chromosome is different from evaluating semantic temporal safety requirements such as contamination or closing a container before transport.

### Gaps

- None of these papers alone establishes general safety across unseen manipulation tasks or removal of all hand-designed constraints.
- Full text was accessible for the 2011 and 2022 papers. For 2012, the publisher-indexed abstract, introduction, conclusion and validation excerpts were available, but direct full-page retrieval failed. The 2024 author abstract and indexed article text were available; direct PMC opening encountered a challenge and publisher opening was rate-limited. Do not invent additional details about their implementations.

## What does this mean for reduced tuning and enforcement?

### Takeaway

These examples show three different uses of evolution: optimize a penalty-defined path, optimize within an explicit geometric collision model, and adapt motion around an existing safety controller. A reduced-tuning contribution must identify what is automated beyond those established uses.

### Cited Findings

- The 2011 source provides numerical penalty choices; the 2022 source explicitly identifies tunable mutation variances and a continuing runtime safety mechanism. Their algorithmic freedom should not be described as constraint-free safety. [2011 paper](https://journals.sagepub.com/doi/10.5772/50902); [2022 paper](https://link.springer.com/article/10.1007/s41315-022-00240-4).

### Inferences

- A generated path satisfies a particular modeled problem. A reusable feedback policy must respond to new states during execution. Evidence for the former does not automatically establish the latter.
- Optimization feasibility, actual execution safety and generalization are separate claims. A runtime controller can be the source of protection even when the planned trajectory was evolved.
- Observing less slowing or stopping need not mean a new safety property was learned; it may mean the trajectory better accommodates a supplied protection mechanism.

### Gaps

- This search does not measure total human tuning effort for any study. Such an assessment requires accounting for modeling, initial trajectories, safety rules, settings and evaluation design.

## What was excluded and how complete is the coverage?

### Takeaway

This is a representative primary-source map, not a systematic census or a claim that no newer work exists.

### Cited Findings

- The 2020 paper “Path planning optimization of six-degree-of-freedom robotic manipulators using evolutionary algorithms” primarily optimizes joint torque in single and cooperating arm transport. It is useful nearby manipulation optimization, but should not be presented as a demonstrated human/object safety study merely because reduced torque is beneficial. [Primary paper](https://journals.sagepub.com/doi/10.1177/1729881420908076).

### Inferences

- The direct intersection includes established engineering planning literature, so a prior-work review should not search only for the phrase “safe reinforcement learning” or for evolved neural policies.

### Gaps

- This bounded search did not establish which approach is best, whether any solves the user's reduced-tuning objective, or whether a proposed new contribution is novel. Coverage was checked up to the requested 3 November 2026 cutoff; the selected sources prioritize clear, directly inspectable evidence over a forced newest-paper list.
