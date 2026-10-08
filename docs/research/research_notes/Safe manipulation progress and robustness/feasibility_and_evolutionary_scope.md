# Integration feasibility and evolutionary scope

Research date: 8 October 2026. This is an inspection-only comparison of task completion after safety intervention and robustness to object/physics changes. It reuses the [earlier asset audit](../Safe%20manipulation%20research%20opportunities/assets_and_entry_cost.md) and [compute audit](../../reports/UNISafe%20inference%20and%20training%20compute.md). No simulator, checkpoint inference or training was run. UNISafe source links below use revision `ab0fdf87f38edefa48b54a291ec5b7156c17b6e0` where checked. Recommendations are research judgments, not performance predictions.

## What can the released UNISafe system actually support?

### Takeaway

The release has the essential components for both questions: a task policy, visual world model, safety filter and instrumentable simulator. Establishing a faithful baseline is a shared integration project, rather than a checkpoint-download exercise. Neither candidate needs a new world model merely to begin diagnosis. [Official release](https://github.com/CMU-IntentLab/UNISafe)

### Cited Findings

**Policy assets and interface.** The earlier archive audit established that `latest.pt` contains the Dreamer task actor/value networks as well as the world model; filter actor/critic checkpoints are separate. The evaluator loads a second Dreamer instance from `policy_model_path`, separately from the safety model, and constructs the filter with `num_actions - 1` outputs. Thus autonomous evaluation exists, beyond teleoperation. The inspected actor has seven action coordinates; the filter replaces six arm coordinates while retaining the task gripper command. No matching released diffusion-policy checkpoint was verified. [Evaluator, lines 162-228](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/reachability/evaluate_reachability_filter.py#L162), [preserved checkpoint inspection](../UNISafe%20inference%20and%20training%20compute/model_and_checkpoint_audit.md)

The model observation interface contains front/wrist RGB and end-effector position/quaternion, with episode flags. It does not expose object mass or friction. The wrapper clips incoming simulator commands to ±0.15. The task actor uses its own recurrent latent, updated from observations and its preceding action, so frozen weights still allow history-dependent latent updates. [Wrapper, lines 28-97](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamer_wrapper.py#L28), [policy, lines 188-217](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/dreamer.py#L188)

**Action units matter.** `make_env` installs `NormalizeActions`, which maps normalized commands to stored environment bounds before the clipping wrapper. Relative Cartesian-pose IK then uses scale 0.5; the gripper is binary joint-position control. Different numbers at these layers are not evidence of a scaling bug. Preserving the entire chain is necessary when replacing the actor or adding an adapter. [Normalization](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/envs/wrappers.py#L26), [IK configuration](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/takeoff/config/franka/takeoff_ik_rel_env_cfg.py#L25), [gripper configuration](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/takeoff/config/franka/takeoff_joint_pos_env_cfg.py#L88)

**Logging needs extension.** `filter_action` changes the action tensor in place when either uncertainty or safety-value criteria trigger. Evaluation records episode outcome, length, intervention count and uncertainty summaries, rather than a complete proposal/execution training dataset. The collector replaces terminal camera/proprioception observations with the preceding observation and labels several preceding transitions when reward crosses a failure threshold. These conventions require auditing before treating released labels as exact event times. [Filter and evaluation helpers](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/tools.py#L474), [collection, lines 335-358](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/tools.py#L335)

**Outcome semantics are inspectable.** Success requires the upper block to remain stacked over the base, the extracted middle block to be separated, and the middle block not to have dropped. Failure includes either upper/middle block falling below height 0.13, or excessive upper-to-base horizontal displacement. `is_failure` accepts `max_steps` but does not use it. The normal task separately terminates on timeout, end-effector bounds, dropping, success and failure. Its 15-second duration and 20 Hz control imply a nominal 300-control-step limit. These are task-specific geometric tests, not a general safety certificate. [Outcome functions, lines 355-503](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/takeoff/mdp/observations.py#L355), [normal task, lines 247-295](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/takeoff/takeoff_env_cfg.py#L247)

### Inferences

A useful logging boundary would retain the unmodified proposal, filtered normalized command, transformed simulator command, intervention reason, observation history and terminal event. Dynamics-model updates need executed actions expressed in the model's training coordinates. Proposal actions can instead be meaningful inputs to a critic representing the complete filtered controller. This separates two different learning problems; logging only one ambiguous `action` field would obstruct both.

The released actor permits bounded actor-adaptation studies without collecting demonstrations for a new policy family. Keeping the safety model/filter fixed initially would isolate changes in task behaviour. That is an attribution choice, not a requirement to freeze everything forever. A separate model/filter adaptation changes the scientific question and should be identifiable as such.

### Gaps

The existing compute audit found a 5090 port necessary for the original Isaac Sim/PyTorch combination; it did not benchmark this controller. Static source issues also need resolution, including an apparent `_modules.parameters()` call in the evaluator. These are reproduction concerns, not demonstrated runtime failures or research contributions. [Evaluator, line 236](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/reachability/evaluate_reachability_filter.py#L236), [compatibility audit](../UNISafe%20inference%20and%20training%20compute/runtime_compatibility_and_assets.md)

No exact mid-episode snapshot/restore facility was verified. The inspected code resets the scene and writes object poses/velocities, which does not establish restoration of contact-solver state, controller state, cameras and both recurrent latents. Reset-plus-prefix replay is a possible fallback requiring validation, not a demonstrated equivalent of the existing MuJoCo branching. [Reset events](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/takeoff/mdp/events.py#L21)

## How do the candidates differ in additional work and identifiability?

### Takeaway

Continuation has the cleaner behavioural question but first needs evidence that avoidable post-intervention stalls occur in the available task. Robustness has directly configurable physical changes, but the released normal/hard pair is not a controlled physics-only comparison. Neither currently has a verified end-to-end runtime budget.

### Cited Findings

**Continuation evidence is external to the released block-plucking baseline.** *When World Models Lie* attributes some manipulation incompletion to a nominal policy untrained on states reached after safety intervention. Its simulated manipulation uses another task and policy; its hardware latency measurements cannot establish the runtime of the UNISafe release. The project website's Code link remains `href="#"`, so no ready alternative arm package was established here. [Paper, section V-B](https://arxiv.org/html/2609.34300), [project source, lines 47-55](https://github.com/mudhdhoo/WhenWordModelsLie_project_page/blob/main/index.html#L47)

**The physics assets are useful but incomplete.** The normal/hard configurations explicitly change cuboid dimensions and the upper block's mass from 0.5 to 0.25. The hard version also changes outcome thresholds, reward weighting and initial arm posture. Its termination definitions, including timeout, are commented out; the inspected derived configurations do not restore them. Consequently, selecting the hard task unchanged is not a clean robustness comparison or a verified reproduction of the paper's evaluation. [Hard configuration](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/takeoff/hard_takeoff_env_cfg.py#L71), [hard arm configuration](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/takeoff/config/franka/hard_takeoff_joint_pos_env_cfg.py#L31)

The Franka reset configuration varies joints and upper-block positions/yaw. It defines `reset_object_position` twice, so the second class attribute replaces the first. It does not configure a mass/friction randomization distribution. The inspected object definitions also lack an explicit varying friction material. This establishes editable dimensions/masses and pose reset hooks, not a complete friction benchmark. [Reset configuration, lines 31-80](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/takeoff/config/franka/takeoff_joint_pos_env_cfg.py#L31), [object definitions](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/takeoff/takeoff_env_cfg.py#L71)

### Inferences

For **continuation**, the extra asset is correctly logged closed-loop experience containing interventions and subsequent outcomes. Existing successful demonstrations do not establish coverage of those states. A compact adapter to the released actor could be evaluated through the unchanged filter, but interpreting a gain requires distinguishing useful continuation from easier trajectories, weakened filtering, longer time allowances or a generally stronger task policy. A timeout and many interventions alone do not identify why the task stalled.

For **robustness**, the extra work is a declared parameter family with verified instantiated values and coherent outcome definitions. Mass-only changes could avoid changing visible geometry; size changes also affect grasping and the meaning of fixed geometric thresholds. Friction needs an explicit material-setting path and validation. A small parameter family can be informative without claiming arbitrary object generalization. One must retain physically valid, task-relevant configurations rather than count impossible tasks as controller failures.

A frozen recurrent world model can react to new observation histories, but it has no verified explicit physics-conditioning interface. Imagining the same latent with added noise is therefore not established as evaluating another mass or friction coefficient. Behaviour can instead be scored in real simulator variants while the fixed model remains a safety evaluator or representation. Retraining the model across those variants is a legitimate later extension with additional collection and attribution costs.

| Decision factor | Completion after intervention | Object/physics robustness |
| --- | --- | --- |
| Problem clarity | Specific handoff/continuation failure, once reproduced | Clear distribution shift, mechanism initially ambiguous |
| Extra assets | Intervention trajectories and reliable action/event logging | Parameter controls, valid scenario sets and consistent labels |
| Identifiability | Intervention often coincides with already difficult states | Physical shift can alter policy, perception, dynamics and filter simultaneously |
| Closest-prior pressure | Filter-aware policy learning and smoother intervention | Robust control, domain randomization and dynamics identification |
| Likely failure point | No substantial avoidable stalling, or state information insufficient | Invalid scenarios or the fixed model/filter failing across the entire range |
| Bounded entry condition | Recurrent, consequential stalls with useful behaviour still available | Reproducible degradation across a manageable, meaningful range |

This matrix is a judgment from the interfaces and literature, not experimental evidence that either branch will succeed. A bridge is plausible: physical shifts might increase interventions and expose continuation weaknesses. Establishing the intermediate events is necessary before calling that the cause of robustness failures.

### Gaps

No post-intervention coverage audit, block-plucking stall diagnosis, physics-grid throughput or replay-equivalence result exists here. A single 32 GB GPU remains a plausible development target from the earlier audit, not an assurance about total evolutionary evaluation cost. Neither candidate presently warrants an invented population size, training duration or rental estimate.

## What evolutionary scope is scientifically useful, and what are the alternatives?

### Takeaway

Standard evolution can be the adaptation tool in a worthwhile empirical contribution. Its useful role is searching a defensible behaviour space against actual safe completion, while strong alternative adaptation methods remain available. A new optimizer is not required, and replacing an optimizer alone does not explain an improvement.

### Cited Findings

The following reading/asset map separates methodological comparisons from immediately reusable manipulation assets:

| Source | Available foundation and reading purpose |
| --- | --- |
| [UNISafe paper/release](https://github.com/CMU-IntentLab/UNISafe) | Closest ready world-model, task-actor and safety-filter combination; establish the task and original component roles. |
| [When World Models Lie](https://arxiv.org/html/2609.34300) | Direct continuation/mismatch motivation; code placeholder means its arm setup is not currently a verified fallback. |
| [LatentCBF release](https://github.com/CMU-IntentLab/latent_cbf) | Useful smoother-filter comparison; public README documents an end-to-end Dubins pipeline, not verified arm checkpoints. |
| [Provably Optimal RL under Safety Filtering](https://arxiv.org/abs/2510.18082) | Read before defining a filter-aware learning baseline; optimality claims depend on the paper's assumptions. |
| [Reducing Safety Interventions](https://github.com/JakobThumm/safety-intervention-reduction) | Public intervention-reduction implementation and a relevant policy/filter training comparison. |
| [TD-MPC2 models](https://www.tdmpc2.com/models) | Released task-matched MetaWorld/ManiSkill models; a fallback for latent control, not a supplied safety-filter benchmark. |
| [LeWM Cube](https://huggingface.co/quentinll/lewm-cube/tree/main) | Released visual checkpoint if JEPA-style representation is central; task-specific safety readouts and validation remain new work. |

TD-MPC2's MetaWorld wrapper explicitly requires state observations, so it does not preserve the same visual-world-model research claim automatically. LeWM Cube supplies goal-planning assets but no verified matching safety head in this audit. Moving to either platform exchanges the UNISafe port burden for new safety-definition, label and evaluator work. [MetaWorld wrapper](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/envs/metaworld.py), [Cube planning configuration](https://github.com/lucas-maes/le-wm/blob/main/config/eval/cube.yaml)

### Inferences

For continuation, the most bounded EA role is adapting a compact part of task behaviour while scoring the full filtered controller. Possible search spaces include a limited actor adapter or interpretable motion parameters; these are options requiring interface design, not established working parameterizations. Ordinary filter-present policy fine-tuning, simple continuation rules and an unchanged actor are meaningful comparisons. Improving only an already failing frozen baseline would leave the main alternative explanation unresolved.

For robustness, evolution could adapt that behaviour across simulator conditions, with aggregate and difficult-condition safe completion both visible. The comparison is then with ordinary domain-randomized policy adaptation under the same condition and interaction budgets. Robust CEM becomes relevant if the selected intervention is online action-sequence planning. Comparing offline actor evolution against online CEM without separating their training and deployment costs would answer a different question.

With released assets, one can compare unchanged actor, filtered actor and adapted actor while retaining the same world model/filter. Changing policy family, adding a smoother optimization filter, conditioning the model on physics or claiming transfer across model families requires additional integration or training. Keeping component boundaries explicit makes a modest empirical result easier to interpret: did task behaviour improve, did the safety evaluator improve, or did the operating point simply become less conservative?

Both candidates share three prerequisites: a faithful runnable baseline, trustworthy action/outcome accounting, and measured closed-loop throughput. Continuation adds intervention-state experience and a credible diagnosis of avoidable stalling. Robustness adds controlled physical variants and evidence that evaluation semantics remain comparable. Prefer continuation if the diagnosed handoff failure is substantial; prefer robustness if bounded physical changes expose a clearer, repeatable weakness. Current evidence supports these choice conditions, not a universal winner.

### Gaps

The EA search dimension and cost are not yet determined. Evolving the full released actor is not justified merely because its weights exist. Nor has this audit established that compact evolution outperforms gradient adaptation, random search or simple parameter sweeps. Contribution strength would come from a reproducible unresolved failure, a controlled improvement and informative limits, rather than attaching novelty to safety plus evolution plus a latent model.
