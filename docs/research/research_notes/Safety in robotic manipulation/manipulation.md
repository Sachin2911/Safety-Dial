# Operational meanings of manipulation safety

## What is being protected, and against what?

### Takeaway

Manipulation research operationalizes safety through several overlapping targets: human injury prevention, robot integrity, object/environment protection, and context-dependent or temporally ordered requirements. These are not interchangeable definitions of one universal metric.

### Cited Findings

- **Robot integrity and geometric admissibility.** Morton and Pavone's *Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions* (IROS 2025) addresses whole-body obstacle avoidance, self-collision, joint limits, singularities, and workspace containment. Their project demonstrates that multiple trajectories can satisfy constraints while differing in task consistency. Safety here is formulated through explicit constraints on the robot, rather than a general judgement of harmlessness. [Paper](https://arxiv.org/abs/2503.06736); [author project and demonstrations](https://stanfordasl.github.io/oscbf/).
- **Human physical safety.** Haddadin, Albu-Schaffer and Hirzinger's foundational *Safety Evaluation of Physical Human-Robot Interaction via Crash-Testing* (RSS 2007) investigates impact injury potential using crash-test facilities and measures for head, neck and chest. It explicitly excludes sharp-edge injury mechanisms. This exemplifies safety grounded in consequences for a human body, rather than merely detecting a geometric intersection. [Original paper](https://roboticsproceedings.org/rss03/p28.pdf).
- **Required contact versus harmful force.** Wei et al.'s *Ensuring Force Safety in Vision-Guided Robotic Manipulation via Implicit Tactile Calibration* (arXiv first posted December 2024, revised November 2025) studies door opening. It separates force in the useful opening direction from orthogonal forces that can damage the door or robot. Its definition of force safety is keeping harmful forces below a selected threshold. Contact itself is necessary for the task. [Paper, introduction and Figure 1](https://arxiv.org/html/2412.10349v2).
- **Object properties and context.** Brunke et al.'s *Semantically Safe Robot Manipulation: From Semantic Scene Understanding to Motion Safeguards* (RA-L 2025, preprint 2024) includes examples such as carrying water above electronics or rotating a filled cup. It combines geometric constraints with semantic spatial, behavioural and pose constraints inferred using perception and an LLM. The revised paper includes learned diffusion policies in a kitchen environment. [Paper](https://arxiv.org/abs/2410.15185).
- **History and order.** Huang et al.'s *SafeManip* (May 2026 preprint) uses finite-trace temporal logic to check eight safety categories: contact, grasp stability, release stability, contamination, action onset, mechanism recovery, containment and enclosure access. Examples include contacting clean items after contamination and releasing objects before they are contained. Its evaluation separates task completion from satisfaction of the temporal properties. [Paper](https://arxiv.org/html/2605.12386v1).

### Inferences

- A useful conceptual distinction is **what is protected** versus **how a violation is expressed**. Human, robot and object safety identify protected entities; force limits, forbidden regions and temporal rules express different kinds of constraints. This is our synthesis, not a universally adopted taxonomy.
- Manipulation makes the equation "contact = unsafe" particularly unsuitable. Grasping, pushing and door opening require contact; the relevant question is whether that contact is appropriate in location, force, duration and context. The door-opening example above is a concrete case, not evidence that every manipulation benchmark measures all those dimensions.
- A robot can avoid collisions while still doing something hazardous with an object. Semantic safety can therefore add requirements that geometry alone does not express.

### Gaps

- This focused set does not establish how frequently each definition is used across the entire literature.
- Human psychological comfort, privacy and broader social harms are adjacent safety concerns but were not reviewed here. Do not claim the taxonomy is exhaustive.

## What do benchmark metrics and safety claims actually establish?

### Takeaway

The explicit specification, sensor assumptions and evaluation context define the scope of a safety result. Lower observed violation rates, satisfaction of a symbolic rule and a conditional control guarantee are different kinds of evidence.

### Cited Findings

- SafeDiff counts task completion under a 20 N peak-force cap in one success metric, stating that the cap assumes damage above that force. It also reports average/maximum harmful force and threshold-based safety fractions. This illustrates an authored operational threshold, not a universal physical damage boundary. [Section 5.1](https://arxiv.org/html/2412.10349v2).
- SafeManip explicitly limits its claim to the properties specified and uses privileged simulator state to ground predicates. Its authors say real-robot use would require reliable perception, contact estimation, fixture-state tracking and task-event detection. Results are specific to its 50 tasks and fixed policy checkpoints. [Section 6](https://arxiv.org/html/2605.12386v1).
- ManiGuard (August 2026 preprint) similarly evaluates safety independently of task success using finite-trace temporal specifications over physics-grounded predicates. Its six household families include dropping, spilling and ordering hazards. It is an existing familiar example, not the definition of the entire field. [Author project](https://nu-ideas-lab.github.io/ManiGuard/); [paper](https://arxiv.org/abs/2608.17386).
- The semantic-filter paper's formulation includes a softened orientation objective alongside hard inequality constraints. Thus even within one method, requirements can have different enforcement status. [Author paper, formulation](https://utiasdsl.github.io/semantic-manipulation/assets/pdf/semantic-manipulation2024.pdf).

### Inferences

- We should ask of every paper: which hazards are represented, who supplied that representation, how is it observed, and what evidence supports the claimed protection?
- A formal guarantee about a specified safe set does not establish that the set captures all real hazards. Likewise, runtime verification describes a realised execution; monitoring alone does not force future behaviour to comply.
- Lower violation incidence can coexist with inactivity. Safe completion, unsafe completion and unfinished behaviour should be separated rather than treating all non-violating executions as equally desirable.

### Gaps

- No numerical control guarantees were independently re-derived here. The notes do not establish robustness of the cited controllers to unmodelled dynamics, incorrect object labels or arbitrary perception error.
- SafeDiff's text and result-table labels differ on one pair of evaluation thresholds (5/15 N in prose versus 5/10 N in the displayed simulation table). Avoid reproducing those detailed threshold results without checking the released implementation or authors' clarification.

## Which examples are useful to read or watch first?

### Takeaway

Use contrasting papers to learn the different questions called "safety", rather than reading only methods close to our current architecture.

### Cited Findings

- **Watch for physical intuition:** the official OSCBF project offers short demonstrations and a linked paper video showing constraints on a moving manipulator. It is a method demonstration rather than an introductory lecture. [Project](https://stanfordasl.github.io/oscbf/); [official linked video](https://www.youtube.com/watch?v=TbssD__hpCg).
- **Read for contextual hazards:** Brunke et al., RA-L 2025. Start with the introduction and examples, then ask how an inferred semantic concern becomes a controller constraint. [Paper](https://arxiv.org/abs/2410.15185).
- **Read for force-sensitive interaction:** Wei et al., SafeDiff. Start with Figure 1 and Section 5.1 to see how physical intuition becomes measured safety. [Paper](https://arxiv.org/html/2412.10349v2).
- **Read for temporal definitions:** Huang et al., SafeManip, May 2026 preprint. Start with the property definitions and limitations before the model results. [Paper](https://arxiv.org/html/2605.12386v1).

### Inferences

- Our suggested reading exercise: for each paper, write one sentence for the hazard, one for its measurable proxy, and one for what remains outside the claim. That directly supports thinking from primitives without committing to an optimizer or architecture.

### Gaps

- The video link was verified through the authors' official project page; its complete audiovisual content was not watched through the browsing tool.
