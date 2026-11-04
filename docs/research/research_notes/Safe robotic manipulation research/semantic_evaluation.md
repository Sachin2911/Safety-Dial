# Semantic and task-level safety in robotic manipulation

## How can contextual safety requirements change executed actions?

### Takeaway

Semantic safety filters connect scene interpretation to an action-level protective mechanism. Their evidence differs from benchmarks that only score behaviour.

### Cited Findings

- **Lukas Brunke, Yanni Zhang, Ralf Römer, Jack Naimer, Nikola Staykov, Siqi Zhou and Angela P. Schoellig, _Semantically Safe Robot Manipulation: From Semantic Scene Understanding to Motion Safeguards_, IEEE Robotics and Automation Letters 10(5), 4810-4817, 2025.** The method builds a semantic scene representation, uses language-model contextual reasoning to infer unsafe relationships, and converts these into control-barrier constraints alongside geometric collision constraints. Illustrative hazards include carrying water above electronics and rotating a water-filled cup. It filters proposed control inputs, so this is an intervention method rather than only a labeler. The authors demonstrate teleoperation and learned-policy manipulation on hardware. The barrier formulation and representation of constraint types remain supplied components. [Paper](https://arxiv.org/pdf/2410.15185); [author publication record](https://siqizhou.com/publications/); [lab abstract of published version](https://www.dynsyslab.org/research/publications/).

### Inferences

- The relevant distinction for synthesis is between understanding a contextual hazard and enforcing the resulting encoded restriction. Certification of encoded barrier conditions does not itself certify that perception or language reasoning identified every relevant hazard.

### Gaps

- This targeted review did not establish broad open-world reliability of the semantic inference pipeline. The verified demonstrations should not be generalized into universal physical safety.

## How can safety be learned into a VLA policy?

### Takeaway

Constrained fine-tuning and feasibility-based learning aim to change the policy itself. A method with “shield” in its name need not contain a runtime shield.

### Cited Findings

- **Borong Zhang, Yuhao Zhang, Jiaming Ji and colleagues, _SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning_, NeurIPS 2025.** Its Integrated Safety Approach specifies costs in a constrained decision process, elicits unsafe behaviour through Safety-CHORES, and fine-tunes VLAs with reward/cost updates and adaptive Lagrange multipliers. Tasks include navigation, object pickup and fetching; it is broader than tabletop manipulation. Hazards include collisions and collateral disturbance of objects. The safety rules, binary costs, cost budget and optimisation settings remain supplied. Most evidence is simulated; a physical Safety-PickUp transfer demonstration uses additional perception, control and simulator-alignment engineering. The reported assurance is empirical evaluation, not proof that every rollout avoids harm. [Conference record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e185c7be603426028c32ae1003a59d78-Abstract-Conference.html); [paper, Sections 4 and 5.3 and Appendices C/G](https://arxiv.org/html/2503.03480v3); [author demonstrations](https://safevla.github.io/).

- **Manan Tayal and Akshay Nambi, _ShieldVLA: Feasibility-Aware Safety Alignment for Vision-Language-Action Models_, September 2026 preprint.** A structured VLM rubric supplies a safety margin; a learned reachability-inspired critic separates reward-focused updates in estimated feasible regions from safety-focused updates elsewhere. The trained VLA is deployed **without a runtime shield**, explicitly stated in Figure 1 and Section 4. It evaluates navigation and manipulation simulators, including Safety-CHORES Fetch and Franka-Reach. Its appendix explicitly says finite data, function approximation and rubric error make all safety claims empirical. The safety rubric and gate threshold remain consequential design choices; real-robot evaluation and recalibration are future work. Do not describe its reachability theory as a formal guarantee for the implemented visual policy. [Paper, Sections 4 and Appendix B](https://arxiv.org/html/2609.13231v1).

### Inferences

- These are useful non-evolutionary comparators for reducing fixed reward-penalty tuning. Adaptive multipliers and feasibility-based policy updates still require safety information, acceptance criteria and learning choices.

### Gaps

- SafeVLA's hardware transfer evidence is much narrower than its simulated task collection. ShieldVLA supplies no physical-robot validation. This review does not compare their numerical gains as though task distributions and evidence were interchangeable.

## Which studies measure temporal safety rather than enforce it?

### Takeaway

SafeManip and ManiGuard make task-order and execution-history violations evaluable. Their monitors are measurement mechanisms; ManiGuard additionally supplies safety-annotated demonstrations and fine-tuning experiments.

### Cited Findings

- **Chengyue Huang, Khang Vo Huynh, Mellon M. Zhang, Sebastian Elbaum, Zsolt Kira and Lu Feng, _SafeManip: A Property-Driven Benchmark for Temporal Safety Evaluation in Robotic Manipulation_, 2026 preprint, v3.** It instantiates finite-trace temporal-logic properties over manipulation executions: contact, grasp/release stability, contamination, action onset, recovery, containment and enclosure access. The latest verified version uses RoboCasa and LIBERO simulations. Predicate traces are checked against authored specifications, thresholds and timing windows. Its limitations explicitly identify privileged simulator-state grounding; hardware deployment would require noisy sensor-based predicate estimation. Properties that never trigger are separated from genuine tested satisfaction. This is an evaluation layer, not an action veto or a trained safety policy. [Paper, especially limitations and Appendix grounding discussion](https://arxiv.org/html/2605.12386v3).

- **Yiyan Peng and colleagues, _ManiGuard: A Benchmark and Data Suite for Specification-Grounded Safety Evaluation and Improvement of Robotic Manipulation_, August 2026 preprint.** It separates goal achievement, safety and engagement across contact-rich household task families. Authored temporal-logic specifications over physics-grounded predicates provide labels. Section 5.1 explicitly says the runtime monitor checks every step **without terminating execution**. Demonstration generation discards unsafe/unsuccessful trajectories; subsequent fine-tuning can improve policies, but the benchmark monitor does not protect deployed actions. The paper includes physical Franka evaluation on a subset. Its limitations clarify that hardware verdicts are operator-scored, not produced by the simulator-state automaton. A safe label concerns the encoded properties on an observed rollout. [Paper, Sections 4-6 and Appendix C](https://arxiv.org/html/2608.17386v1); [official evaluation documentation](https://nu-ideas-lab.github.io/ManiGuard/docs/evaluation/).

### Inferences

- The writer should preserve three different contributions: generating safety requirements, changing or restricting behaviour, and evaluating compliance. Treating a benchmark monitor as a protective filter would conflate these.

### Gaps

- This is a selective primary-source map checked through 3 November 2026. Searches covered semantic manipulation safety, SafeVLA, ManiGuard, SafeManip and recent VLA safety/shielding. It is not a systematic novelty review. Additional candidates such as ActFovea and SafeVLA-Bench were located but not fully assessed and are excluded from the supported map.
