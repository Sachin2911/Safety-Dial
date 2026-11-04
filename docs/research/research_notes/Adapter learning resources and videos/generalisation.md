# Adapter terminology and generalisation resources

## What kinds of adapters should we distinguish?

### Takeaway

Parameter-efficient neural adapters change a network's computation while keeping its original weights fixed. A control adapter instead changes an action, candidate ranking, or conditioning signal around a controller. These are related design ideas, but evidence for one does not validate the other.

### Cited Findings

- **Houlsby et al., Parameter-Efficient Transfer Learning for NLP, ICML 2019.** Primary paper and code links: [PMLR proceedings](https://proceedings.mlr.press/v97/houlsby19a.html). The method adds small trainable modules for downstream tasks while freezing the original network. This is a good foundational source for the phrase "adapter" and for separating shared pretrained parameters from task-specific learned parameters. It studies language tasks, not closed-loop control or safety.
- **Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, 2021.** [Paper](https://arxiv.org/abs/2106.09685) and [official implementation](https://github.com/microsoft/LoRA). LoRA freezes original weights and learns low-rank updates. It is useful for understanding how a large network's adaptation can be restricted to a small parameter space. The source experiments concern language models, so they do not show that low-rank changes preserve robotic competence or improve safety.
- **Hugging Face PEFT conceptual guide: LoRA.** [Official guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora). This is the most accessible implementation-oriented reading in this set. It explains update matrices, rank, target modules, and neutral initialization using one zero matrix. The page is explicitly the main documentation version; use it for concepts rather than assuming the displayed API matches an installed version.

### Inferences

- Suggested priority: residual control and planner adaptation resources from the other researcher should precede Houlsby/LoRA. Generic YouTube searches for "adapters" will disproportionately return LLM material, which answers how to parameterize a change but not where a robot should be given control authority.
- For the current frozen-world-model question, putting LoRA inside the world model would change its effective predictions even though the original parameter tensor is frozen. That differs materially from keeping the model's function fixed and evolving an external adapter. Putting it in a separate controller is another design, not a demonstrated replacement for either existing adapter.
- Read Houlsby for architecture and task-specific parameter sharing; read LoRA for a compact search space; read the PEFT guide only if the matrix mechanism is unclear. Basic neural-network and linear-algebra knowledge is sufficient for the concept, while reproducing the papers requires substantially more infrastructure.

### Gaps

- None of these sources establishes a shared safety adapter across TwoRoom and Block Plucking, or shows that evolutionary search is preferable for these parameterizations.

## What can robotics adaptation teach us about generalisation?

### Takeaway

Rapid Motor Adaptation is a useful example of a reusable adaptation mechanism, but its base controller is explicitly trained to accept an inferred environmental latent. It is not an arbitrary frozen planner retrofitted with a universal module.

### Cited Findings

- **Kumar et al., RMA: Rapid Motor Adaptation for Legged Robots, RSS 2021.** Start with the [authors' project page](https://ashish-kmr.github.io/rma-legged-robots/), then the [paper](https://arxiv.org/abs/2107.04034). The system has a base policy and adaptation module, trained in simulation and deployed on the A1 quadruped without real-world fine-tuning.
- The base policy is first trained with privileged environmental factors encoded into a latent vector. A separate adaptation module learns to estimate that vector from recent states and actions; the base policy consumes the estimate at deployment. See the architecture and Section III of the [full paper](https://arxiv.org/html/2107.04034).
- Our study design already distinguishes method reuse with retrained weights, generalisation to new cases within an environment, and weight transfer without retraining. Equal latent dimensions are explicitly not treated as aligned representations. [FirstExperiment.md](../../../../SafeLeWM/FirstExperiment.md).

### Inferences

- RMA is especially useful for asking what information an adapter needs, why recent history might matter, and whether the base system was prepared to accept its interface. It should not be presented as proof of cross-task safety transfer.
- A reasonable learning progression is external residual control first, then RMA's conditioning interface, then internal neural adapters. Those represent distinct places to intervene, rather than three interchangeable training algorithms.
- "Same architecture, separately learned weights" and "same weights across tasks" should remain separate claims in the reading guide. A shared architecture may still rely on task-specific observations, normalisation, success predicates, and safety predicates.

### Gaps

- These sources do not establish zero-shot transfer between navigation and manipulation. No new experiment or protocol is proposed or authorized by these notes.

## Which verified videos accompany these resources?

### Takeaway

The RMA method video is the most directly relevant video in this subset. The inventor's LoRA explainer is optional background for the broader neural-adapter terminology.

### Cited Findings

- **RMA: Rapid Motor Adaptation for Legged Robots (Method)**, Legged Robots, published 1 July 2021. [YouTube](https://www.youtube.com/watch?v=qKF6dr_S-wQ). This is explicitly linked as the method video on the [authors' project page](https://ashish-kmr.github.io/rma-legged-robots/). Use it to see the base-policy/adaptation-module architecture.
- **RMA: Rapid Motor Adaptation for Legged Robots (Results)**, Legged Robots, published 1 July 2021. [YouTube](https://www.youtube.com/watch?v=nBy1piJrq1A). Demonstration companion, embedded on the [authors' project page](https://ashish-kmr.github.io/rma-legged-robots/); useful for seeing terrain and disturbance variation, not a substitute for learning the mechanism.
- **What is Low-Rank Adaptation (LoRA) | explained by the inventor.** [YouTube](https://www.youtube.com/watch?v=DhRoTONcyZE). The [official Microsoft repository](https://github.com/microsoft/LoRA) points to this exact video as its explainer. Recommended only as the optional internal-network-adapter branch of the list.

### Inferences

- Suggested order within this subset: RMA method video, RMA project page and architecture, then Houlsby and the LoRA explainer if parameter-efficient adaptation is still of interest.

### Gaps

- Links were checked against official project/repository pages and indexed YouTube metadata on 4 November 2026. Direct RMA watch-page fetches failed; the videos were not played. No runtime or timestamps were reliably verified, so none are supplied. The LoRA video title resolved, but its channel and runtime were not verified.
