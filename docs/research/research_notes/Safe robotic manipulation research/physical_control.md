# Physical safety control for robotic manipulation

## What concrete work constrains physical execution?

### Takeaway
Physical safety research often modifies or filters a nominal controller so that specified geometric, velocity, energy or power requirements hold. These are different protections; their assumptions and remaining engineering should remain visible.

### Cited Findings
- **Daniel Morton and Marco Pavone, “Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions,” IROS 2025.** OSCBF filters operational-space commands while accounting for joint-space motion and task hierarchy. It addresses whole-body collisions, workspace and joint limits, and singularities. Simulation and Franka Panda hardware experiments test constraint satisfaction and tracking. The supplied ingredients include robot dynamics, collision geometry, barrier definitions, limits and controller/objective gains. The paper gives an explicit tuning procedure. Its important qualification is that conflicting constraints can make the quadratic program infeasible; relaxing it may preserve useful behaviour while losing the strict safety guarantee. This is a low-level control framework, not a learned manipulation policy or semantic hazard detector. [Primary manuscript, v3 dated 18 October 2025](https://arxiv.org/html/2503.06736v3); [IROS publication record](https://ieeexplore.ieee.org/document/11246389/); [author demonstrations](https://stanfordasl.github.io/oscbf/).
- **Youssef Michel, Matteo Saveriano and Dongheui Lee, “A Novel Safety-Aware Energy Tank Formulation Based on Control Barrier Functions,” IEEE Robotics and Automation Letters 9(6), 5206-5213, 2024.** The method combines energy storage accounting with CBF constraints on energy and power. Hardware tests address force control, varying stiffness and kinetic-energy limitation. The human supplies tank bounds, power limits and controller settings; the method is model-based control, not learned safety specification. Theoretical passivity conditions depend on the tank design. In the kinetic-energy example, the authors explicitly report occasional violations of the time-varying power constraint because their CBF formulation does not handle time-varying barriers. This is therefore a useful example of both conditional theory and implementation limits. [Primary paper](https://www.inverse-project.org/wp-content/uploads/2024/05/Michel-et-al.-A-Novel-Safety-Aware-Energy-Tank.pdf); [DLR bibliographic record](https://elib.dlr.de/208540/).

### Inferences
- OSCBF represents safety through admissible motion, whereas the tank method monitors how control injects energy and power. A robot can need both forms of protection. Neither source's evidence should be translated into unrestricted human or object safety.

### Gaps
- These studies do not establish learned semantic safety, general object-damage detection, or comparative reductions in total human engineering effort.

## How is physical safety connected to completing a manipulation task?

### Takeaway
Task-specific control research combines compliant contact with a mechanism that limits dangerous escalation when interaction does not proceed as planned.

### Cited Findings
- **Francesco Cufino, Mario Selvaggio, Fabio Amadio and Fabio Ruggiero, “Compliant Non-Prehensile Pushing Manipulation,” arXiv preprint, 25 May 2026.** The method extends a pushing model for impedance-controlled robots, then uses MPC to select motion and contact-point changes. An energy-tank filter modulates velocity when interference could otherwise make the robot continually increase pushing force. Tests include simulation and ABB Mobile YuMi and KUKA LBR iiwa hardware, including human interaction. The model assumes quasi-static pushing, and the authors expect low-speed collaborative operation. Force updates, including those used by the filter, rely on predictions; measured-force integration is future work. Model parameters, controller tuning, tank design and task references remain supplied. This is task-specific model-based manipulation with a passivity claim, not learned general safety. [Primary preprint](https://arxiv.org/html/2605.25672v1).

### Inferences
- The relevant failure mechanism is not simply collision. A compliant robot trying to correct tracking error can still accumulate energy when someone obstructs it. The safety mechanism must address that closed-loop interaction.

### Gaps
- The preprint does not establish safety for arbitrary high-speed pushing, unfamiliar object geometries or all injury mechanisms. Do not treat the hospital logistics illustration as evidence of clinical deployment.

## What foundational distinction prevents overclaiming?

### Takeaway
Passivity is an interaction-stability property, and should not be used as a synonym for absence of physical harm.

### Cited Findings
- **Wenceslao Shaw Cortez, Christos K. Verginis and Dimos V. Dimarogonas, “Safe, Passive Control for Mechanical Systems with Application to Physical Human-Robot Interactions,” ICRA 2021, pp. 3836-3842.** The paper explicitly explains that passivity alone does not ensure velocity, power, force or workspace requirements. It constructs a controller combining a specified operating region, velocity constraints and passivity, with a passive restoring response outside the region. A six-degree-of-freedom robot demonstrates physical human-robot interaction. The safety sets and bounds are supplied, and the proof relies on the modeled mechanical-system assumptions. [Author manuscript](https://people.kth.se/~dimos/pdfs/ZCBF_Mechanical_Systems__conf_Final.pdf); [author publication listing confirming ICRA 2021](https://people.kth.se/~dimos/publication.html).

### Inferences
- For the broader synthesis, distinguish **collision avoidance**, **contact-force or energy limitation**, **stable physical interaction**, and **successful manipulation**. The four papers above protect different subsets of these properties.

### Gaps
- This is a selective primary-source search checked on 3 November 2026, not a systematic inventory of safe manipulation. Queries covered manipulator CBFs, safe force control, energy tanks, passivity and compliant pushing. SafeDiff, FORGE and passivity-aware RL were surfaced but left to the safe-learning researcher to avoid duplication. No experiments, source preparation or archived evidence were used.
