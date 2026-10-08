# Assets and entry cost for safe manipulation research

## Which released systems provide a realistic starting point?

### Takeaway

Audit dated 8 October 2026. **UNISafe has the strongest verified combination of a manipulation world model, safety components, labelled experience and simulator evaluation.** TD-MPC2 and LeWM offer credible alternatives with smaller neural components or simpler simulation, but additional safety-definition work. Low entry cost depends on the complete evidence chain, not just parameter count.

### Cited Findings

| System | Verified release | Safety and outcomes | Entry assessment |
| --- | --- | --- | --- |
| **UNISafe block plucking** | IsaacLab task, world-model/filter download, successful and failed trajectory archives, example replays, task-policy evaluation. | Simulator wrapper exposes separate `failure` and `success` fields. Existing failure readout, uncertainty model and filter can be retained. | **Low-to-medium offline; medium closed loop.** Neural assets exist, but the original Isaac Sim 4.2 stack needs migration for a 5090. [Release](https://github.com/CMU-IntentLab/UNISafe), [wrapper](https://github.com/CMU-IntentLab/UNISafe/blob/isaaclab/dreamer_wrapper.py) |
| **TD-MPC2 / Meta-World** | Official catalog lists 150 single-task checkpoints: 50 tasks, three seeds, 5M parameters. The 80-task replay release includes Meta-World. | Task rewards and success information exist; no dedicated safety-cost channel was identified in its wrapper. It explicitly requires **state observations**. | **Low-to-medium for a state-based latent-control study; medium for safety.** A visual-safety study needs other weights or new training. [Models](https://www.tdmpc2.com/models), [wrapper](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/envs/metaworld.py) |
| **LeWM / OGBench Cube** | Public task-matched configuration and 72.3 MB weights; 46.2 GB compressed expert dataset; goal-planning configuration. | Cube poses and simulator state can support declared event labels. The inspected release does not supply a trained safety head or manipulation safety specification. | **Medium.** MuJoCo environment and visual model are available, but readouts, labels and their validation remain work. [Weights](https://huggingface.co/quentinll/lewm-cube/tree/main), [data](https://huggingface.co/datasets/quentinll/lewm-cube/tree/main), [evaluation configuration](https://raw.githubusercontent.com/lucas-maes/le-wm/main/config/eval/cube.yaml) |
| **LatentSafe / successor latent CBF code** | Public RSSM and DINO-WM training code. The current successor repository documents a complete Dubins-car pipeline. | Native obstacle constraints and safety-filter code are present for that car example. A ready arm checkpoint/data package was not verified. | **High relative to UNISafe for arm manipulation.** Valuable methods and references, but paper demonstrations should not be treated as released arm assets. [LatentSafe repository](https://github.com/CMU-IntentLab/latent-safety), [successor repository](https://github.com/CMU-IntentLab/latent_cbf) |

Additional verified boundaries:

- TD-MPC2 also supplies **15 ManiSkill2 checkpoints across five tasks**. The actual files include lift-cube, pick-cube, pick-YCB, stack-cube and turn-faucet, at roughly 31.5 MB per checkpoint. These are useful task-specific alternatives, not evidence that the same weights work on current ManiSkill versions or arbitrary safety tasks. [Actual checkpoint directory](https://huggingface.co/nicklashansen/tdmpc2/tree/main/maniskill2)
- TD-MPC2's public multi-task data cover **DMControl and Meta-World**, not the five ManiSkill2 tasks. The release contains numerical observation vectors and varied replay-buffer behaviour. Thus, public ManiSkill weights do not imply a matching public replay dataset in this release. [Official dataset description](https://www.tdmpc2.com/dataset)
- OGBench provides MuJoCo environments, goal-success evaluation, dataset generation and Gymnasium interfaces. Its native `terminated` and `success` report reaching a goal; these must not be reinterpreted as safety labels. [Official OGBench repository](https://github.com/seohongpark/ogbench)
- LeWM's separate Reacher weights exist, but its evaluation environment is **DMControl Reacher with configuration matching**, not a contact-rich manipulation arm. It is a simpler diagnostic option, not a substitute for block-plucking realism. [Weights](https://huggingface.co/quentinll/lewm-reacher/tree/main), [configuration](https://raw.githubusercontent.com/lucas-maes/le-wm/main/config/eval/reacher.yaml)
- MultiSafe's project page still marks code **coming soon**. Its wax-temperature and opaque-bottle experiments motivate partial-observability questions, but cannot currently be advertised as a verified downloadable benchmark. [Official project](https://cmu-intentlab.github.io/multisafe/)

### Inferences

UNISafe is attractive because it avoids inventing the world model, task, safety definition and baseline simultaneously. TD-MPC2 is an effective way to simplify perception when the research question is about control or robustness. LeWM Cube retains the user's visual-latent interest while moving away from Isaac Sim, but recreating safety supervision can outweigh its smaller model size.

### Gaps

No environment or checkpoint was executed in this audit. The matching library versions, action scaling and task-specific model quality still require validation. No public arm-ready package was verified for SafeDreamer, LatentSafe's hardware demonstrations, or MultiSafe. These are bounded release findings, not claims that such assets cannot exist.

## What can be studied without building a new robotics stack?

### Takeaway

The lowest-cost scientifically useful starting point is a question about an existing system's failure or loss of useful progress. A new task, world-model family and safety mechanism at once is substantially higher effort, even if each neural network fits on one GPU.

### Cited Findings

- The prior UNISafe audit checked source revision `ab0fdf87f38edefa48b54a291ec5b7156c17b6e0`, archive metadata and checkpoint dimensions. It verified a two-camera 128-pixel simulation model, separate filter weights and sample trajectory headers containing actions, proprioception, success and failure fields. The larger archive's filename prefixes do not establish independent episode counts, correct event labels or clean train/test splits. [Preserved audit](../../reports/UNISafe%20inference%20and%20training%20compute.md), [released samples](https://github.com/CMU-IntentLab/UNISafe/tree/isaaclab/log)
- UNISafe exposes model/policy training, ensemble-only fitting and safety-filter evaluation. Its documented evaluation uses the task policy learned during world-model training; that is an available baseline, not a promise of a strong separately trained manipulation policy. [Official README](https://github.com/CMU-IntentLab/UNISafe)
- *When World Models Lie* reports incompletion after safety interventions because its nominal policy was not trained to resume from the resulting states. It also distinguishes safety-filter behaviour in closed-loop control from replayed action sequences. This is direct evidence that a robot can avoid failure yet fail to finish its task. Its manipulation models and datasets are distinct from UNISafe's block-plucking release. [Paper, sections V-B and VI](https://arxiv.org/html/2609.34300)

### Inferences

**Low entry effort: offline diagnosis using frozen released components.** One can ask where predictions, uncertainty or failure readouts disagree with recorded outcomes, and which distinctions matter for choosing actions. This can establish an actual problem before proposing an EA. However, a held-out classifier score or threshold sweep alone is usually a diagnostic result. Recorded actions cannot establish how a new controller would change future outcomes.

**Medium entry effort: useful progress with an existing safety mechanism.** A bounded agent-focused question is whether the system can finish after being redirected, or whether alternative task behaviour avoids repeatedly requiring intervention. UNISafe supplies task and safety outcomes to investigate this without retraining the world model initially. The incompletion mechanism must first be demonstrated in the chosen released task; the recent paper establishes it in another manipulation system, not automatically in block plucking. A substantive result would preserve safety while recovering useful task performance, rather than merely weakening a filter threshold.

**Medium entry effort: a specific robustness weakness in an existing task.** Keeping one task and fixed model, one can ask how a declared physical or observation shift changes safe task completion. It becomes worthwhile when it identifies a reproducible mechanism and a bounded improvement. Merely applying a standard EA to find one adverse setting, or reproducing an existing robustness sweep, would be low coding effort with a weak contribution. The problem-evidence and EA-role notes should establish what remains open before choosing a method.

**High entry effort: new hidden physics or new manipulation assets.** Temperature, liquid contents, deformable bags and force-dependent damage are interesting, but may require new simulation, sensors, labels and model training. They should remain on the research map without being described as the easiest first study. Partial observability also creates an information problem that additional optimization cannot necessarily solve.

### Gaps

No paired reset/replay validation was performed for IsaacLab. Replaying a demonstration file is not proof of exact counterfactual branching. No audit established that the released full dataset contains enough post-intervention trajectories for learning task continuation. Such missing experience could become the dominant acquisition cost.

## What does the compute evidence change about this triage?

### Takeaway

A single-GPU manipulation project is credible. The main remaining entry costs are compatibility, usable data and meaningful outcome evaluation, rather than an assumed need for very large model training.

### Cited Findings

- The existing audit supports a **32 GB 5090 as a profiling target**, not a measured throughput guarantee, and documents the original Isaac Sim/PyTorch compatibility problem. It did not run inference, render the environment or train the model. [Preserved compute and compatibility audit](../../reports/UNISafe%20inference%20and%20training%20compute.md)
- New, separate evidence is available: *When World Models Lie* reports **39.59 ms per step** for its adaptive method and **17.42 ms for its UNISafe baseline** on an RTX 5090 in a 15 Hz hardware setup. This concerns a three-camera dustpan/tomato task. It is not a benchmark of the released block-plucking checkpoint, simulator rendering or training. [Paper, section VI](https://arxiv.org/html/2609.34300)

### Inferences

The practical order is to choose a demonstrated problem, identify which existing assets expose it, then decide whether an EA offers a useful search space. Holding the model fixed is one way to bound cost, but it is not an obligation or an automatic novelty advantage. A modest study can still involve new training if the question requires it.

For the nearest-term investigation, UNISafe offers the shortest path to **native safety plus task progress**. TD-MPC2 offers the shortest path to **many existing latent-control policies with state inputs**. LeWM Cube offers a compact **visual-latent manipulation** starting point with extra safety instrumentation. These are different practical strengths, not a ranking of scientific importance.

### Gaps

No paid runs, installations, full downloads, model training or GPU timing were performed. Public source/metadata were checked on 8 October 2026; several pinned GitHub source pages failed to fetch, so exact task thresholds were not newly verified. The previous metadata audit was reused rather than repeated. No wall-clock research budget is estimated from parameter counts, and no release availability claim implies unchanged checkpoint transfer between simulators or tasks.
