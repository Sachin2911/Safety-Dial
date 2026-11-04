# Safe learning and filtering for robotic manipulation

## How do learned recovery policies protect exploration and execution?

### Takeaway
Recovery methods separate a task policy from a mechanism that predicts risk and substitutes corrective actions. The strongest recent examples also address compliance or uncertainty, while retaining explicit labels, thresholds and controller choices.

### Cited Findings
- **Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones**, Brijen Thananjeyan, Ashwin Balakrishna, Suraj Nair, Michael Luo, Krishnan Srinivasan, Minho Hwang, Joseph E. Gonzalez, Julian Ibarz, Chelsea Finn and Ken Goldberg; RA-L/ICRA 2021. Offline data train a risk critic and recovery behaviour before task learning. A task policy pursues reward; recovery takes over near predicted constraint violations. Evaluations include two simulated contact-rich manipulation tasks and physical robot obstacle avoidance. This is a foundational precedent for learned intervention rather than merging all safety considerations into task reward. Constraints and recovery-switching criteria still have to be supplied. Avoid describing the physical experiment as hardware block manipulation: the abstract distinguishes simulated contact-rich manipulation from physical obstacle avoidance. [Author manuscript and publication metadata](https://arxiv.org/abs/2010.15920); [IEEE publication](https://ieeexplore.ieee.org/document/9392290/).
- **SRL-VIC: A Variable Stiffness-Based Safe Reinforcement Learning for Contact-Rich Robotic Tasks**, Heng Zhang, Gokhan Solak, Gustavo J. G. Lahr and Arash Ajoudani; RA-L 9(6), 5631-5638, 2024. A learned risk critic gates task/recovery policies, while actions specify displacement and variable stiffness. Safety is an explicit contact-force threshold. Scripted simulated contacts provide offline labels; reward penalties and stiffness/action bounds remain designed. The robot explores a narrow maze using contact rather than vision. MuJoCo training transfers to a Franka arm; simulation observation-noise retraining improves transfer after initial failures. The paper records violations during learning, so its abstract's guarantee language should not be paraphrased as zero-violation exploration. It also reports a conservative baseline that gets stuck, and a changed maze where the learned policy avoids violations but cannot find the exit. This is an unusually clear safety-versus-completion example. [Paper](https://arxiv.org/html/2406.13744v1); [publication record](https://ieeexplore.ieee.org/abstract/document/10517611).
- **SEVI: Safe Evidential Real-World Reinforcement Learning**, Shaekh Mohammad Shithil and Lantao Liu; manuscript posted 22 October 2026, marked by its authors as accepted to RA-L in September 2026. An evidential safety critic estimates future violation risk and epistemic uncertainty; either can trigger an offline-trained recovery policy while the task policy learns online. Tests cover simulated block plucking and nut transport plus physical USB insertion. Hardware violations include keep-out-region breaches and excessive force; failure timesteps are manually annotated. Recovery demonstrations, safety definitions and task-specific risk/uncertainty thresholds remain supplied. Thresholds are calibrated using held-out successful and failure rollouts. Results report reduced unsafe-event counts, not zero events or a formal system guarantee. This is a current example addressing safety during hardware learning, rather than only testing a frozen final policy. [Full manuscript](https://arxiv.org/html/2610.31829v1); [author project page](https://sevi-safe.github.io/).

### Inferences
- These papers offer a useful progression: learn when to recover; learn contact compliance too; account for uncertainty in the risk estimate. This is an organizing interpretation, not a historical claim that one paper supersedes all earlier methods.
- A recovery policy itself is learned and may be unreliable. Its existence is not evidence that every unsafe state is recoverable.

### Gaps
- SEVI's linked IEEE destination is a generic journal page and could not be fetched. Use “2026 manuscript reporting RA-L acceptance” if publication status needs precision, rather than inventing an issue/page number.
- The studies do not give a common-budget cross-paper ranking. Their tasks, unsafe-event definitions and data differ.

## How can prediction and uncertainty guide a runtime safety filter?

### Takeaway
World-model filters use predicted consequences to restrict another policy's actions. Their guarantees and failure modes need to be distinguished from guarantees about the real robot.

### Cited Findings
- **Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution Failures (UNISafe)**, Junwon Seo, Kensuke Nakamura and Andrea Bajcsy; CoRL 2025. The method augments latent reachability analysis with epistemic uncertainty, trains a fallback policy, and uses conformal calibration to detect uncertain imagined trajectories. It evaluates simulated block plucking and physical Franka Jenga, as well as a non-manipulation control task. Offline failure labels define the learned failure margin; uncertainty and intervention settings remain specified. The appendix explicitly states that RL training and imperfect dynamics prevent guarantees in all cases. OOD initial observations can defeat the filter; it adds a check that halts and requests human intervention when the selected fallback remains unreliable. The conformal result concerns the detection/calibration component, not unrestricted zero-failure manipulation. [Paper, especially Appendix B.4](https://arxiv.org/html/2505.00779v2); [official project and venue](https://cmu-intentlab.github.io/UNISafe/).

### Inferences
- UNISafe changes the role of a model from task planning to anticipating and intervening in hazardous transitions. It remains a distinct component from the base manipulation policy.
- Safety supervision has not disappeared: outcome labels define the known failure set, while uncertainty broadens caution beyond confidently modeled conditions.

### Gaps
- The source does not support treating its component-level statistical result as a calibrated probability of every real-world manipulation failure.

## How does a learned manipulation policy regulate harmful contact forces?

### Takeaway
Force-aware learning can adapt movements using measured interaction forces. Empirical force reduction is distinct from a certified force bound.

### Cited Findings
- **Ensuring Force Safety in Vision-Guided Robotic Manipulation via Implicit Tactile Calibration (SafeDiff)**, Lai Wei, Jiahua Ma, Yibo Hu and Ruimao Zhang; CoRL 2025, PMLR 305:1049-1062. SafeDiff generates future robot states from vision and current state, then uses force feedback to calibrate the state sequence; actions are derived from that plan. Door-opening experiments distinguish useful tangential force from potentially harmful off-trajectory force. The work supplies SafeDoorManip50k and tests both simulation and hardware. Its success metric uses an assumed 20 N damage threshold, explicitly a benchmark assumption rather than a universal material limit. Other force metrics assess average, maximum and proportions of states below thresholds. The evidence establishes measured force reduction on studied tasks, not a general formal guarantee. [Proceedings](https://proceedings.mlr.press/v305/wei25a.html); [full paper](https://arxiv.org/html/2412.10349v2); [official demos](https://i-am-future.github.io/safediff/).

### Inferences
- SafeDiff and SRL-VIC illustrate two different intervention locations: modifying planned states through sensory feedback versus selecting movement and stiffness through a policy plus recovery gate.
- For research comparisons, count the force detector, its threshold, representation and safety-labelled data as part of the supplied safety information.

### Gaps
- This is a selective five-paper set, verified through primary sources available by 3 November 2026, not an exhaustive survey or novelty assessment. Search focused on recovery-based safe RL, force-safe learned manipulation and uncertainty-aware filters. SafeDiffuser, safe PDE diffusion and cartpole reset papers were not included because this set prioritizes manipulation-specific evidence.
