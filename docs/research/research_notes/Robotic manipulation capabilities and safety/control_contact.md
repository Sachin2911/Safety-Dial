# Planning, contact sensing and explicit safety in manipulation

## What remains useful in conventional manipulation, and what has developed?

### Takeaway

Model-based perception, motion planning and feedback control remain active components of manipulation. A useful map separates choosing a task, finding a feasible motion, controlling contact, and enforcing protection constraints; these components can coexist with learned policies.

### Cited Findings

- MoveIt's planning scene supplies robot/world geometry, collision checking and constraint checking. The allowed-collision mechanism distinguishes intended object contact from prohibited contact. This is maintained engineering infrastructure, rather than a general learned manipulation policy. [MoveIt planning-scene tutorial](https://moveit.picknik.ai/main/doc/examples/planning_scene/planning_scene_tutorial.html).
- GPU parallelism improved conventional optimization as well as learning. The 2023 cuRobo work combined inverse kinematics, geometric planning and trajectory optimization to produce collision-free motion efficiently. The versioned library documentation includes world representations based on meshes and depth images and demonstrates UR10 motions. Use this as an example of acceleration in geometric planning, without importing headline speedups into comparisons on different hardware. The page now directs readers to cuRoboV2; do not describe v0.7.6 as the newest release. [Original research page](https://research.nvidia.com/publication/2023-05_curobo-parallelized-collision-free-robot-motion-generation); [versioned documentation](https://nvlabs.github.io/curobo/v0.7.6/index.html).
- Russ Tedrake's MIT *Robotic Manipulation: Perception, Planning, and Control* notes, actively updated for Fall 2026, cover geometric perception, bin picking, constrained motion planning, mobile manipulation, manipulator control, RL, soft robots and tactile sensing. This is an excellent broad educational map beyond the latest model releases. Start with Introduction, Basic Pick and Place, Manipulator Control and Soft Robots and Tactile Sensing. [MIT course notes](https://manipulation.csail.mit.edu/).

### Inferences

- Collision-free planning is a claim relative to the robot/world representation and checked constraints. It does not by itself establish that a selected grasp will not crush an object, that a carried container will not spill, or that all relevant obstacles were perceived.
- Conventional components are not a historical stage that learned policies automatically replace. A learned policy can provide desired end-effector motions while a conventional controller realizes joint motions and enforces limits, as illustrated explicitly in OSCBF below.

### Gaps

- These software and research sources do not establish a worldwide deployment count or an industrial reliability ranking. Avoid equating a maintained library with validated safety in a particular installation.

## What do contact, tactile sensing and compliant hardware add?

### Takeaway

Recent developments improve both the observation of contact and collection of contact-rich demonstrations. The evidence can include better completion and lower interaction forces, which are distinct outcomes.

### Cited Findings

- ViTaMIn, revised September 2025, combines a compliant Fin Ray gripper, visual and tactile sensing, handheld demonstration collection and learned multimodal representations. It evaluates orange placement, test-tube reorientation, scissor hanging, moving-hole peg insertion and dual-arm knife pulling. The paper reports 20 randomized trials per task and better performance than its vision-only baseline. Its failure analysis includes table/rack collisions, weak grasps, poor coordination and release failures. A separate handle-rotation experiment measures lower average axial forces with tactile input over 10 trials per condition. This is real-robot task and force evidence; it is not a guarantee that harmful contact never occurs. [ViTaMIn v2](https://arxiv.org/html/2504.06156v2).
- FreeTacMan is a related scaling development, initially submitted June 2025, revised March 2026 and listed by its authors as ICRA 2026. It provides a wearable gripper with visuo-tactile sensing and optical tracking, plus a dataset of roughly 10,000 demonstration trajectories across 50 contact-rich tasks. The authors open hardware and data to support reproducibility. This illustrates a move from isolated sensing demonstrations toward reusable data-collection infrastructure. The scope is learning/data infrastructure, not a certified safety mechanism. [Paper and revision history](https://arxiv.org/abs/2506.01941); [official project](https://opendrivelab.com/FreeTacMan).

### Inferences

- Better visual task understanding cannot directly replace contact information that the camera does not reveal. Tactile signals can make an otherwise poorly observed control problem easier.
- Compliance and sensing can reduce consequences or improve detection, but safety depends on the resulting behaviour and the hazard being measured. Lower average force does not establish a maximum-force limit or a bound on rare severe events.
- These developments are relevant to safe manipulation independently of whether the final learner is imitation learning, RL or evolution. We should resist treating every failure as an optimizer problem when sensing or control authority may be limiting.

### Gaps

- No general real-world injury rate or rare-damage bound was established by the cited tactile experiments. The task-specific demonstrations should not be labelled universally safe, or universally unsafe because such evidence is absent.

## What explicit safety mechanisms exist, and what do their claims mean?

### Takeaway

Explicit constraint filters and commercial robot safety functions provide concrete protection mechanisms, but their coverage and assumptions must be identified. Safety belongs to the full application and the stated hazards, rather than an architecture name.

### Cited Findings

- OSCBF, IROS 2025, integrates control barrier functions with operational-space control so that protection constraints interfere less with task execution. It handles joint/workspace limits, collision and singularity avoidance, including hundreds of simultaneous conditions at kilohertz rates in reported settings. Its latest arXiv revision, 18 October 2025, includes simulation and hardware validation. Crucial qualification: Section II explicitly states that conflicting barriers and input constraints can make the optimization infeasible. Relaxing constraints with slack gives a practical solution but removes the unconditional safety guarantee. The original March v1 deferred hardware validation, so cite v3 for hardware claims. [OSCBF v3](https://arxiv.org/html/2503.06736v3); [official project and demonstrations](https://stanfordasl.github.io/oscbf/).
- Universal Robots' UR10e documentation lists configurable force/power, momentum, speed, position/orientation and stopping-performance limits. It also states that application safety depends on the tool/end effector, obstacles and other machines; built-in functions may be insufficient for hazardous tools or processes. This is direct manufacturer documentation of available protection mechanisms and their application dependence, rather than evidence that any task commanded on a collaborative robot is safe. [UR application-safety documentation](https://www.universal-robots.com/manuals/EN/HTML/SW5_26/Content/prod-usr-man/complianceUR10e/H_g5_sections/safety_g5/risk_assessment.htm).

### Inferences

- The answer to "what is safe?" should identify a protection mechanism, the hazards it covers, its operating assumptions and the supporting evidence. "Has a safety filter", "is a cobot" and "completed the task" are each insufficient as standalone answers.
- Keep four evidence types distinct: author-controlled research demonstrations; benchmark measurements; mathematical results conditional on assumptions; and manufacturer specifications for deployed hardware. None can silently stand in for all the others.
- Safety is partly an integration problem. A high-level learned action can be acceptable in end-effector coordinates while creating an unsafe elbow motion, and a collision-free motion can still manipulate an object in a damaging way.

### Gaps

- This is a selective research map current to 3 November 2026, not an exhaustive comparison of industrial deployments or a standards/compliance assessment. In particular, sources here do not substantiate a claim that a broadly general-purpose manipulation system is safe across arbitrary household tasks.
