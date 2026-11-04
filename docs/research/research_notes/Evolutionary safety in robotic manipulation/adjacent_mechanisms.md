# Adjacent evolutionary mechanisms for manipulation safety

## What does grasp quality-diversity establish?

### Takeaway

Quality-diversity provides a substantial manipulation capability precedent, but grasp diversity and robustness should not be relabelled general safety. In the verified example, conventional planning supplies collision handling around evolved grasps.

### Cited Findings

- Johann Huber, François Hélénon, Mathilde Kappel, Elie Chelly, Mahdi Khoramshahi, Faïz Ben Amar and Stéphane Doncieux, **Speeding up 6-DoF Grasp Sampling with Quality-Diversity**, IROS 2024, evolves diverse grasp poses using QD variants and robotic sampling priors. Simulation covers four grippers; deployment uses a Franka arm with parallel and Allegro grippers. Fitness assesses retention under shaking. Appendix B explicitly supplies RRT-Connect/MoveIt motion planning with object/table collision bodies and discards grasps colliding with the table. Mutation scale is selected by grid search; hand-closing synergies are predefined. This is evidence for grasp generation and transfer, while safe approach trajectories and physical assumptions remain supplied. [Author manuscript](https://arxiv.org/html/2403.06173); [author project](https://qdgrasp.github.io/); [author workshop slides identifying IROS 2024 venue](https://thome.isir.upmc.fr/MLR/workshop-25/Kappel.pdf).
- Aurel X. Appius et al., **Task-Aware Robotic Grasping by evaluating Quality Diversity Solutions through Foundation Models**, revised March 2025, uses a QD archive with semantic segmentation and LLM selection of task-appropriate grasp regions. Evaluation concerns task-fit predictions and human preferences, with physical Franka validation. It supports contextual grasp selection, not a general injury/damage bound. Use revised manuscript values rather than the different earlier abstract if quoting numerical results. [Author manuscript](https://arxiv.org/html/2411.14917).

### Inferences

- A diverse grasp repertoire could supply alternatives to a safety-aware selector, but establishing that selector's safety would be an additional contribution.
- The first paper is especially informative about attribution: evolved grasps coexist with separately supplied collision avoidance.

### Gaps

- These papers do not establish safe exploration or calibrated hazard probabilities for arbitrary manipulation tasks.
- The bounded search did not verify a new manipulation-specific genetic-programming barrier certificate example. This is not evidence of its absence from the literature.

## Can evolution reveal manipulation failures instead of learning a controller?

### Takeaway

Evolutionary search already has a direct manipulation robustness-testing role. This should be distinguished from evolving a safe policy or physically preventing damage.

### Cited Findings

- Naif Wasel Alharthi and Martim Brandão, **Physical and Digital Adversarial Attacks on Grasp Quality Networks**, ICRA 2024, use evolutionary multiobjective optimization to identify perturbations that reduce selected-grasp quality while limiting perturbation size. The two threat models involve a small added scene object and a single depth-image pixel change. The paper evaluates openly available grasp-quality models and reports degraded predictions/selected-grasp quality. Actual real-robot attack evaluation is listed as future work in its conclusion. This is adversarial testing evidence, not demonstrated physical damage prevention or safe learning. [Author-hosted paper](https://www.martimbrandao.com/papers/Alharthi2024-icra.pdf); [publisher record](https://ieeexplore.ieee.org/document/10610886/).

### Inferences

- This offers a separate evolutionary contribution: searching for counterexamples to a manipulation system's apparent robustness. It does not establish that training on discovered counterexamples produces a safe system.

### Gaps

- Do not equate the paper's model quality changes with measured robot injury or object-damage rates.

## Which nearby titles can mislead this literature search?

### Takeaway

The word evolutionary can describe iterative improvement without an evolutionary algorithm. Method verification matters more than title matching.

### Cited Findings

- Yufei Zhu et al., **EvolvingGrasp: Evolutionary Grasp Generation via Efficient Preference Alignment**, ICCV 2025, explicitly builds Handpose-wise Preference Optimization as an extension of direct preference optimization, with a physics-aware consistency model. It also supplies surface-contact and penetration constraints. It should not be counted as population-based evolutionary policy search merely because its title uses evolutionary. [Author paper, introduction and methods](https://arxiv.org/html/2503.14329); [official project identifying ICCV 2025](https://evolvinggrasp.github.io/).

### Inferences

- Robustness, recovery, physical plausibility, safe exploration and harm avoidance should be tagged separately in the eventual literature map.

### Gaps

- Search scope was selective and primarily located 2024-2025 adjacent precedents; it does not support a first/only/no-prior-work novelty claim through November 2026.
