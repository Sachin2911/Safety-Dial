# Designing a meaningful safe contact experiment

This is a research-design proposal, not a frozen protocol or execution authorization. No simulator preparation, testing, learning, evaluation or rental operation was performed. The proposal puts earlier outcomes aside while retaining the repository's controls on future execution.

## What must the benchmark establish before learning?

### Takeaway

Start with one insertion task where task competence, contact uncertainty and force-limit adherence can be measured separately. Admit it only after showing that the same observation/action interface permits useful safe completion and that the force monitor measures the intended quantity.

### Cited Findings

- FORGE trains force-conditioned assembly policies under uncertain part poses and randomized dynamics. It evaluates peg insertion, gear meshing and nut threading. Some tasks need appreciable contact force, so making every action gentler is not a general solution. Its force threshold is a supplied control objective, with an automatic adjustment procedure for tasks whose required force is initially unknown. [FORGE paper](https://arxiv.org/html/2408.04587v2); [official project](https://noseworm.github.io/forge/).
- Chen et al.'s insertion framework combines designed controller and recovery modules with CMA-ES tuning. Its reward retains weighted terms for insertion time, collisions, peak force and failures. This is a useful competent reference family rather than evidence that search alone supplies safety. [Chen et al., IROS 2022](https://arxiv.org/pdf/2203.03017).
- SRL-VIC learns displacement and stiffness using a risk critic and recovery. Its fixed low-stiffness comparison obtains few violations but can get stuck. The hardware tests also include an unfamiliar maze that the learned controller fails to complete. These motivate joint reporting of completion and safety. [SRL-VIC, RA-L 2024](https://arxiv.org/html/2406.13744v1).

### Inferences

- Use a single peg-in-hole or equivalent insertion task first. Disturb the perceived socket pose within a declared operational range so that the controller must interpret contact and correct alignment. Keep sensor-estimation error distinct from physical initial-pose variation; they produce different inference problems. Add friction variation in a later, separately declared challenge.
- Reproduce the upstream benchmark's ordinary task performance before adding the safety interpretation. Preserve its native success criterion and report it alongside strict safe completion. A safety win against a broken baseline is not evidence for this research question.
- Qualify four references on development cases: the competent native controller, a competent conservative or force-aware reference, a controlled contact/overload diagnostic, and a no-motion or stop-only diagnostic. The last two test the scoring system, not competitive performance.
- A positive safe witness should use the proposed observation/action interface. A witness that sees the true socket location can show physical feasibility, but cannot establish that the learner's information permits recovery. Label those two uses separately.
- Choose the challenge distribution and safety specification before inspecting candidate outcomes. Do not lower the force limit until the native policy fails, add obstacles that only the proposed method can avoid, or retain only scenes that show a preferred algorithm's advantage.
- Require that safe task completion is feasible and useful on a meaningful part of the declared development distribution. If the task has no safety-performance tension, it can test competence but offers a weak safety study. If every qualified controller fails, it is not yet a learning comparison.

### Gaps

- A numerical force boundary cannot be recommended without the task's material/contact assumptions or an externally justified benchmark requirement. A rigid-body simulator may support a force-limit-adherence claim without supporting a claim about real component breakage.
- The benchmark researcher is verifying exact available implementations and sensors. A paper's desired force threshold must not be assumed to equal a reliable physics-level contact measurement in a current implementation.

## What is the smallest informative experimental progression?

### Takeaway

Progress from measurement and control qualification, to a bounded learning comparison, to held-out contact conditions. Change one scientific question at a time, and distinguish a frozen policy facing new conditions from a fixed learning recipe retrained for a new task.

### Cited Findings

- FORGE separately examines force observations, dynamics randomization, force conditioning and success prediction. Its paper explicitly treats force profiles as task-dependent. This supports testing whether contact feedback contributes, while preserving the task requirement independently of optimizer choice. [FORGE](https://arxiv.org/html/2408.04587v2).
- Chen et al. specify component geometry and insertion error and keep designed safety and recovery modules around the optimized parameters. A comparison using such an architecture must distinguish the contribution of the modules from the contribution of optimization. [Chen et al.](https://arxiv.org/pdf/2203.03017).

### Inferences

1. **Measurement and competence qualification.** On a predeclared development panel, reproduce success and verify units, coordinate frames, sampling frequency and the force quantity being scored. Distinguish wrist resultant force, local object contacts and commanded force. Check whether opposing contacts could cancel in a resultant and whether gravity/inertial contributions are present. Define which quantity represents the stated hazard. Record force at the resolution required by the specification; an episode-average force cannot rule out short overloads. Fix any filtering or overload-duration convention before outcomes are compared.
2. **Control authority qualification.** Demonstrate a successful correction or recovery through the same allowed outputs and observations available to learners. The controller must be able to change the relevant physical interaction. For the candidate native FORGE study, retain its pose-target interface and study approach, contact search, retraction and retry. Do not claim learned stiffness adaptation when gains are randomized externally rather than controlled by the policy. Adding stiffness outputs would constitute a separate interface change. The implementation researcher is documenting the exact action masking and gains. [Official FORGE implementation](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/contrib/forge/forge_env.py).
3. **Bounded learning comparison.** Compare evolution with matched parameter random search and a competent fixed controller. An ordinary-learning method with explicit constraint handling is an important comparison before attributing benefit to evolution. Give methods the same deployment information, action limits and safety specification. If architectures or training information differ, describe a method comparison rather than isolating the optimizer. Count rejected candidates, prior demonstrations, critic pretraining, tuning and engineering rollouts in declared accounts.
4. **Mechanism checks.** First inspect whether improvement requires feedback rather than only a good constant setting. A fixed-controller counterpart and a carefully specified force-feedback ablation can answer different parts of this question. Keep any common protective controller identical. Log requested, modified and executed commands and interventions. A policy that completes faster behind a guard may improve useful performance while the guard supplies most protection. Hardware protection should not be removed merely for attribution; diagnostic removal, if scientifically needed, belongs to a separately authorized simulated study.
5. **Frozen-policy evaluation.** Freeze the selected policies before testing new combinations of pose error and friction within a declared distribution. Preserve a separate extrapolation panel for stronger shifts. Neither panel is available for architecture, threshold or seed selection. Report adverse results rather than revising the panel after failure.
6. **Fixed-recipe transfer.** Only after the preceding claim survives, move to a distinct geometry or contact task, such as gear meshing or a different keyed insertion. Retraining with an unchanged procedure and budget tests reduced retuning. Deploying unchanged weights tests policy transfer. Report these separately. A second task from the same simulator strengthens task breadth but does not independently validate the contact physics or real-world safety.

### Gaps

- The selected benchmark and controller family are not yet fixed. Exact observations, policy dimension and allowed changes must follow a source audit and development qualification, not be selected after a preferred optimizer's result.
- The scientific design needs an interaction-budget proposal based on a bounded engineering measurement, not an invented wall-clock estimate. This note provides no rental or run allocation.

## What should count as an encouraging result or a stopping point?

### Takeaway

The first meaningful signal is repeatable additional safe completion, attributable to useful interaction behaviour, on cases not used to choose that behaviour. A single successful insertion, a lower force average or reduced motion is insufficient.

### Cited Findings

- SRL-VIC reports task success and constraint violation separately and includes cautious-but-unsuccessful behaviour. The phenomenon is concrete prior evidence for retaining completion and unfinished episodes in the outcome account. [SRL-VIC](https://arxiv.org/html/2406.13744v1).

### Inferences

- Per episode, report native completion, any declared safety violation, strict safe completion, unfinished/aborted execution, and engagement or task progress. Report peak force, overload duration and intervention frequency as explanatory measurements. Keep training-time and final-policy violation records separate. If an episode is stopped after a violation, it remains a violation and unfinished episode; do not silently discard it.
- An encouraging development signal requires improvements across independent learning repeats, useful completion beyond stop-only behaviour, and evidence that a learned action or feedback response changes the interaction. The separate statistical-design note should supply prospective effect margins and repetition counts. A best-seed demonstration is not the unit of evidence.
- Stop before training if the reference competence or force-measurement check fails. Stop the learning comparison if an independently qualified safe witness cannot act through the chosen interface. Redesigning the interface is a new hypothesis and protocol, not an automatic continuation of a failed gate.
- If safety improves only because tasks are abandoned, do not call the result safe manipulation progress. If a common protective controller accounts for all violation prevention, report the learner's completion or efficiency contribution within that system.
- If random search matches evolution, the result supports a useful search space and potentially a lower-manual-tuning route, but does not establish evolutionary optimizer superiority. If every method fails despite a same-interface safe witness, investigate information, reward resolution, representation and budget before claiming the task is infeasible.
- If every candidate receives the same sparse binary failure label, first diagnose the learning signal. Additional progress or severity feedback can be introduced prospectively, equally for relevant comparisons, with its human design burden recorded. Do not quietly turn an outcome-only hypothesis into a heavily shaped objective.
- A reduced-tuning claim must survive unchanged monitors, controller structure and method settings across the declared task family. Count condition-specific changes and selection evaluations. Replacing one penalty coefficient with several per-task gates does not by itself demonstrate reduced engineering.

### Gaps

- These criteria establish an empirical claim within specified tasks, sensing and simulation assumptions. Hardware transfer, rare-event rates and physical damage prevention require additional evidence and cannot be inferred from simulated benchmark completion.
