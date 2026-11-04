# Benchmarks for contact and recovery learning

## Which existing task should anchor the first study?

### Takeaway

My recommendation is Factory/FORGE peg insertion as the primary task, followed by Factory/FORGE gear meshing as a second contact task. This is a proposed choice conditional on validating the pinned implementation, assets and force measurements. It isolates the contact-search and insertion phase; it is not an end-to-end pick-and-insert benchmark.

### Cited Findings

- The FORGE paper studies peg insertion, gear meshing and nut threading, with force-conditioned policies, dynamics randomization and physical robot transfer. Its project page still says code is coming soon, but official Isaac Lab now contains a FORGE implementation; the website status should not be mistaken for code absence. The implementation's fidelity to the paper still needs checking. [FORGE project](https://noseworm.github.io/forge/); [official implementation](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/contrib/forge/forge_env.py).
- The official registration currently names `IsaacContrib-Forge-PegInsert-Direct`, `IsaacContrib-Forge-GearMesh-Direct` and `IsaacContrib-Forge-NutThread-Direct`. These differ from older `Isaac-Forge-...-v0` names. A protocol must name a pinned revision rather than infer compatibility from an environment name. [Registration source](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/contrib/forge/__init__.py).
- FORGE's paper initializes the held part inside the gripper, with random hand pose above the fixed part and random held-part position in the grasp. Its physical multistage assembly separately uses known grasp locations and a conventional controller to pick and position parts. Paper success requires the peg or gear to reach within 1 mm of its destination base while laterally centered; nut-thread success is partial threading, not complete fastening. [FORGE supplement](https://noseworm.github.io/forge/static/forge_supplementary.pdf).
- Factory's released reset code positions the robot above the fixed asset using inverse kinematics and constructs the handheld asset pose, corroborating the pre-positioned contact-skill scope. It has task reward and success/engagement machinery. [Factory v2.3.2 source](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py).
- Official current FORGE configuration exposes relative fingertip pose, velocity, force and force threshold to the policy. Actions command pose targets relative to the fixed part; the implementation suppresses roll/pitch actions and uses a success-prediction channel. Controller gains and target-motion limits are randomized by the environment. They are not learned stiffness outputs. [FORGE configuration](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/contrib/forge/forge_env_cfg.py); [action implementation](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/contrib/forge/forge_env.py).

### Inferences

- Peg insertion gives a relatively interpretable first question: can the controller find and enter the socket under pose/contact uncertainty while respecting a declared force requirement? Gear meshing changes the contact geometry while retaining much of the observation/control infrastructure. This is a second task in one simulator, not independent validation across simulators or real hardware.
- Nut threading is a reasonable later extension if the first two justify it, but it should not be the first debugging surface. The paper's partial-thread success criterion must not be described as full assembly.
- Keep native pose-control authority for an initial reproduction. Adding variable stiffness, recovery primitives, new observations or an altered reset would be an explicitly declared study extension. It could be scientifically justified, but it would no longer be the unmodified task/interface.

### Gaps

- No environment was installed or executed. Runtime cost, sensor stability, reproducibility and attainable safe-completion rates remain unknown. All future preparation and experiment computation must follow the repository's rented-host controls.
- No release/commit has been selected for a frozen protocol. Source availability alone does not establish a faithful reproduction.

## What safety measurements exist, and what would we need to add?

### Takeaway

FORGE supplies a relevant force signal and force-exceedance penalty, but this does not substitute for a prospectively specified episode-level safety monitor. The study would establish satisfaction of a simulated force requirement, not directly demonstrate prevention of physical breakage or human injury.

### Cited Findings

- The inspected implementation obtains a six-component incoming-joint wrench, smooths it, transforms its frame, and adds noise to the force observation. Its contact penalty uses the norm of the smoothed force above the episode threshold. This signal is not automatically identical to a pairwise peg/socket contact force or an unsmoothed impact peak. [FORGE implementation](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/contrib/forge/forge_env.py).
- FORGE's paper explicitly supplies a weighted force-exceedance penalty. It notes that forceful insertion can require a minimum force and that nominal insertion forces can come from part datasheets. Thus an arbitrarily tiny force limit can make completion physically infeasible. [FORGE paper](https://noseworm.github.io/forge/static/forge_paper.pdf).
- Factory uses rigid-body simulation. FORGE approximates deformable snap-fit clips using springs, rather than treating material deformation or damage as native general functionality. [FORGE supplement](https://noseworm.github.io/forge/static/forge_supplementary.pdf).
- An unresolved official issue opened 27 October 2026 reports a discrepancy between configured peg-hole diameter and measured USD geometry. This is a reporter's observation, not a maintainer-confirmed defect, but it illustrates why physical clearance must be inspected rather than read from a configuration number. [Isaac Lab issue 8850](https://github.com/isaac-sim/IsaacLab/issues/8850).

### Inferences

- Proposed extension: log benchmark success separately from any force-limit violation, safe completion, unfinished episodes, meaningful engagement and recovery. Define signal frame, filtering, sampling frequency, maximum and sustained exceedance rules before outcomes. Record proposed and executed actions and any independent protective action.
- A measurement qualification stage should check force semantics, contact timing, reset transients, saturation and timestep/solver sensitivity on the pinned host. Changing sampling or smoothing after inspecting algorithm results would change the safety definition.
- Select a requirement using task/part justification, or explicitly call it a simulated constraint study with a predeclared force limit. Do not choose a threshold to manufacture baseline failures. Keep identical monitors and access to force information across compared methods.
- Establish that a useful safe behaviour exists within the chosen action interface before optimizer training. An inactive policy passing the force limit is not a sufficient witness.

### Gaps

- Physical damage tolerances and a validated mapping from this simulated wrench to real component damage are not established by the inspected sources. An empirical simulator result cannot be promoted to a hardware safety guarantee.
- The configuration/asset issue is unresolved, and other implementation revisions can affect observations. A frozen baseline reproduction is necessary before adopting quantitative claims from the paper.

## Where does ManiSkill fit as corroboration or fallback?

### Takeaway

ManiSkill's `PegInsertionSide-v1` is a credible later cross-platform test or fallback, but it is a less clean first isolation of contact recovery because the standard task also includes grasp acquisition. It needs an explicitly added safety specification.

### Cited Findings

- `PegInsertionSide-v1` has dense reward, demonstrations and a 100-step default horizon. The standard task picks a peg up from a table and inserts it sideways into a box. Geometry and initial poses vary; the documented clearance is 3 mm. [Official task card](https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html#peginsertionside-v1).
- Its source evaluates geometric insertion success and provides state observations including peg and hole pose/geometry. The inspected evaluator does not return an episode force-safety outcome. [Task source](https://raw.githubusercontent.com/haosulab/ManiSkill/main/mani_skill/envs/tasks/tabletop/peg_insertion_side.py).
- ManiSkill exposes CPU/GPU pairwise contact-force queries and net contact-force queries. Its end-effector pose controller maps desired motion through inverse kinematics into joint PD control. Force sensing and variable stiffness are not implied by choosing pose control. [Contact API](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/advanced.html); [controller documentation](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html).

### Inferences

- Adding force observations/monitoring is feasible in principle from the documented API, but the engineering and validation are work to be done. A fixed pregrasp reset would isolate insertion, yet must be labelled a modified task rather than stock benchmark performance.
- Use the primary suite's gear task to test whether a learning recipe carries to a different interaction before adding the burden of a second simulator. A later ManiSkill study could assess robustness of the scientific result to different physics and task structure; it would not be a direct numerical leaderboard comparison.

### Gaps

- Source/document review cannot establish runtime fidelity, transfer performance, sensor accuracy or engineering time. Those remain prospective qualification questions.
