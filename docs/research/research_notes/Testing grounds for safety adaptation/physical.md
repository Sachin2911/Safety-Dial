# Physical testing grounds beyond LeWM

## Which existing system offers the strongest near term main study

### Takeaway
Block Plucking remains the lowest integration-risk main candidate because the task, frozen Dreamer components, monitors and simulator recovery work already exist locally. It has an observed safety problem but does not yet have a reliable learned recovery result. Keeping it is a scientific choice contingent on useful adapter authority, not a reason to restart completed searches.

### Cited Findings
- The official UNISafe IsaacLab branch supplies Block Plucking, a downloadable pretrained Dreamer world model and reachability-filter checkpoints, success/failure datasets and evaluation instructions. Its original setup uses Isaac Sim 4.2; the repository warns about migration to later versions. The native upstream filter can override actions, so the project's shadow-mode planner study is a distinct condition. [Official code and assets](https://github.com/CMU-IntentLab/UNISafe/tree/isaaclab).
- The local Stage 1 calibration reports any-control violations in about 52% of 48 fresh starts, with failures including top-block displacement, top/middle drops and boundary exit. Fresh-process parity held; in-process resets did not. [Stage 1 results](../../../..//SafeBlock/Stage1Results.md).
- Search 2 pooled sustained safe completion was native 9/48, untrained selector 16/48, CMA 14/48. CMA also had 30/48 violation episodes versus native 28/48. The pooled verdict was mixed, with no established search benefit over the untrained adapter. [Search 2](../../../../SafeBlock/Search2Pooled.md).
- The large residual qualification was destructive: one safe success and 85 violation episodes in 96 trials. Candidate selection preserved substantially more competence, but neither interface passed that authority gate, whose four native starts all succeeded. [Stage 1 results](../../../../SafeBlock/Stage1Results.md).

### Inferences
- The problem is physically interpretable: extract the target block while preserving stack stability and retaining success after extraction. This supplies a stronger manipulation story than adding arbitrary geometric circles alone.
- A new bounded interface qualification would need to establish recoverable baseline failures and learned headroom on development cases. Current evidence does not prove that larger representations will fix the problem.
- Existing engineering makes this the first main-task candidate for a January submission. It should not absorb an unlimited sequence of interface changes if the necessary authority remains absent.

### Gaps
- The exact runtime and evidence must be recovered on an authorized rental; local output deletion means the presence of result reports is not proof all raw evidence remains local. No experiment, rental or restart is authorized by these notes.

## Which alternative manipulation platforms are practically credible

### Takeaway
The clearest available generic-policy fallback is ManiSkill2 with the released Diffusion Policy or Behavior Transformer checkpoints from Policy Decorator, especially PegInsertionSide. Isaac Lab Factory/FORGE offers a compelling contact-force question but is not yet verified as a ready-to-run frozen-policy package.

### Cited Findings
- The official Policy Decorator repository explicitly provides pretrained base-policy downloads and commands for Diffusion Policy on PegInsertionSide-v2, TurnFaucet-v2 and PushChair-v2. Behavior Transformer examples additionally include StackCube-v0. State and visual configurations are distinguished. The linked Google Drive folder resolves, but files were not downloaded or hash-verified here. [Repository](https://github.com/tongzhoumu/policy_decorator), [checkpoint folder](https://drive.google.com/drive/folders/1W0jwgVP9W1odt_F6vrwtYcnksdbwF1gi).
- Policy Decorator is itself an ICLR 2025 method for online residual refinement of frozen imitation policies, evaluated on ManiSkill and Adroit. It is therefore both an infrastructure source and a close comparator/novelty boundary for a generic frozen-policy adapter claim. [Official implementation](https://github.com/tongzhoumu/policy_decorator).
- The original Diffusion Policy release provides trained checkpoints, configurations and evaluation scripts. Its manipulation tasks include Robomimic Lift, Can, Square, Tool Hang and Transport; action sequences use receding-horizon execution. This is a policy model, not an action-conditioned predictive world model. [Project](https://diffusion-policy.cs.columbia.edu/), [checkpoint/evaluation instructions](https://github.com/real-stanford/diffusion_policy#-evaluate-pre-trained-checkpoints).
- ManiSkill3 is GPU-parallelized and supports PPO, TD-MPC2, Diffusion Policy and VLA baselines; its documented rendering stack uses SAPIEN/Vulkan and Linux/NVIDIA offers the broadest support. This does not establish that a ManiSkill2 checkpoint transfers unchanged to ManiSkill3. [Official ManiSkill repository](https://github.com/mani-skill/ManiSkill).
- ManiSkill3 provides pairwise contact-force queries and net force queries. Those are simulator measurements and still need task-specific interpretation and monitor validation. [Official contact API documentation](https://github.com/mani-skill/ManiSkill/blob/main/docs/source/user_guide/tutorials/custom_tasks/advanced.md).
- Isaac Lab v2.1.1 documents Factory PegInsert, GearMesh and NutThread environments sharing configuration/control options. They are available assembly environments; the page does not establish released checkpoints for all three. [Versioned official environments](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/overview/environments).
- FORGE learns policies conditioned on an allowable force and demonstrates peg insertion, gear meshing, nut threading and snap-fit insertion under pose uncertainty. Its official project page still labels the code link as coming soon. Consequently a directly downloadable author-provided FORGE checkpoint was not verified. [FORGE project](https://noseworm.github.io/forge/).
- Isaac Lab offers a generic pretrained-checkpoint facility only for supported tasks. This mechanism is not evidence that the required Factory task checkpoint exists. [Official quickstart](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/quickstart.html).
- Isaac Lab's contact sensor distinguishes normal, friction and total forces with backend/version-specific semantics, including physics-step sampling. A force experiment must pin these details. [Official sensor documentation](https://isaac-sim.github.io/IsaacLab/main/source/concepts/sensors/contact_sensor.html).

### Inferences
- For ManiSkill PegInsertionSide, a useful prospective question is whether a competent frozen insertion policy can recover from modest misalignment while respecting a declared contact-load limit. It would require a new validated monitor and feasibility witness: the released task is not automatically a safety benchmark. State-based insertion is the lower-complexity qualification; visual transfer is a separate burden.
- Force-aware assembly is scientifically attractive because completion may require contact, while excessive force is undesirable. It forces a real safety/utility tradeoff. Some tasks require a minimum insertion force, so a threshold must not make safe completion impossible. Report compliance with a simulated force requirement, not prevention of hardware damage or injury.
- Recommend Factory/FORGE as a conditional alternative after checkpoint and environment compatibility verification. Do not promise a low-cost pretrained FORGE run based on its paper alone. Starting by training a new task backbone would add a major research dependency.
- For a frozen-WM-specific claim, UNISafe is the direct choice. For a frozen-pretrained-policy claim, ManiSkill/Robomimic diffusion policies provide credible alternatives. Their absence of a world model tests whether the recipe depends on predictions; it should not be obscured by calling every pretrained controller a WM.

### Gaps
- No repeatable runtime or population-throughput measurements were performed for new platforms. GPU simulation throughput does not equal end-to-end evolutionary-evaluation throughput.
- No exact stock Factory author checkpoint was verified. A third-party checkpoint found in search was not used as a recommendation because its provenance and protocol would introduce another qualification problem.
- ManiSkill3 contact APIs cannot be assumed to match the older ManiSkill2 runtime required by Policy Decorator. Pin the released runtime first or treat migration as a separate reproduction exercise.

## How should the project choose between these options

### Takeaway
Choose a problem-backbone-monitor-interface combination, not an environment name. Use Block Plucking as the main candidate, one released generic-policy manipulation task as the fallback, and postpone additional platforms until a clear learning signal exists.

### Cited Findings
- Safety-Gymnasium provides MuJoCo safe-navigation and velocity tasks plus vision/manipulation families. It is expressly designed for safe-RL reward/constraint studies. [Official overview](https://safety-gymnasium.readthedocs.io/en/latest/introduction/about_safety_gymnasium.html).
- The project already has LeWM as its opening controlled study, and Block Plucking as the intended main study; completed protocols do not authorize further execution. [Current authority](../../../../firstPrinciples.md).

### Inferences
- Safety-Gymnasium could serve as a fast objective/optimizer diagnostic, but it adds less to the paper than a physical manipulation task because LeWM already fills the opening diagnostic role. It also does not by itself supply a frozen pretrained manipulation backbone.
- Screen prospective candidates using the same predeclared criteria: reproduce baseline competence; identify recurring avoidable safety violations; demonstrate feasible safe completion through the proposed action interface; verify adequate information at deployment; establish reproducible resets and affordable repeated evaluation. Use development cases and retain failed candidates in the record.
- Useful problem families are collateral object displacement during reaching, unstable release/carry, excessive insertion contact under pose error, and unsafe contact during otherwise successful placement. Select task-native failures first. Avoid choosing thresholds or panels after discovering where evolution happens to win.
- One new main task plus one tightly scoped cross-backbone check is more plausible before January than rebuilding three simulator stacks. Any VLA extension should be judged by the same qualification requirements, not its model size or publicity.

### Gaps
- Baseline competence and safe recoverability in any new platform remain unverified until an explicitly scoped qualification is authorized and run. This report makes no launch or budget decision.
