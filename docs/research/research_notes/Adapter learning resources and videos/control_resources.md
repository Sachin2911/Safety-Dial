# Control adapter learning resources

Checked 4 November 2026. Five core reading resources, ordered for this project. Difficulty and reading priorities are recommendations, not claims from the authors. No experiments or protocol changes are proposed here.

## Which resources explain the useful adapter interfaces?

### Takeaway

Study small evolved controllers, action residuals, proposal generation and safety filtering as distinct interfaces. A component can help planning without being a good standalone policy; a safety filter has stronger modelling requirements than an outcome-ranked behavioural adapter.

### Cited Findings

1. **World Models, David Ha and Jürgen Schmidhuber, 2018.** Start with the [interactive author article](https://worldmodels.github.io/). Its vision and recurrent memory components feed a deliberately small linear controller trained with CMA-ES. The article explains why most complexity can reside in representation learning while evolutionary search operates over a compact decision component. It also discusses exploiting errors in a learned simulation. [Author code](https://github.com/hardmaru/WorldModelsExperiments) supplies reproduction instructions and explicitly records old Gym-version dependencies. **Read for:** where to put the small trainable component and how to separate representation learning from controller search. **Difficulty:** approachable with basic neural-network knowledge. **Boundary:** this is a controller rather than our neutral correction around an existing planner, and task reward/survival are not strict safe completion.

2. **Residual Reinforcement Learning for Robot Control, Johannink et al., 2018 preprint.** The [author project](https://residualrl.github.io/) and [paper](https://arxiv.org/abs/1812.03201) combine conventional feedback control with a learned additive residual, demonstrating real block assembly with contacts and unstable objects. **Read for:** the clean decomposition `executed action = base action + learned correction`, and what the base controller contributes. **Difficulty:** approachable concept, intermediate RL details. **Boundary:** gradient-based RL and a conventional controller, not evolutionary search around frozen visual planning. Our failed residual screen does not establish that all manipulation residuals fail.

3. **POPLIN: Exploring Model-based Planning with Policy Networks, Wang and Ba, 2019 preprint / ICLR 2020.** The [paper](https://arxiv.org/abs/1906.08649) and [author code](https://github.com/WilsonWangTHU/POPLIN) distinguish planning over action sequences initialized by a policy from planning by perturbing policy parameters. The code names these POPLIN-A and POPLIN-P. **Read for:** an adapter can shape what the planner considers, instead of correcting the final action. **Difficulty:** intermediate; understand model-predictive control and CEM first. **Boundary:** proposal generation is different from our suggested candidate reranking; this is not evidence that a shared reranker works across our tasks. The published setup does not establish frozen-world-model safety adaptation. The old report calls this “Learning to Plan with Policy Networks”; the currently verified arXiv title is “Exploring Model-based Planning with Policy Networks.”

4. **LOOP: Learning Off-Policy with Online Planning, Sikchi, Zhou and Held, CoRL 2021 / proceedings 2022.** Start with the [author's CMU explanation](https://blog.ml.cmu.edu/2022/01/07/loop/), then the [paper](https://proceedings.mlr.press/v164/sikchi22a.html) and [code](https://github.com/hari-sikchi/LOOP). It combines learned dynamics, terminal values and actor-guided trajectory optimization. Its safe variant constrains predicted trajectory costs during planning. **Read for:** how a proposal distribution, trajectory score and safety constraint play different roles, and why a learned policy and planner can disagree. **Difficulty:** intermediate to advanced. **Boundary:** it updates learned components and is not our small frozen-system adapter; its cost constraint should not be represented as our episode-level safe-completion criterion.

5. **A predictive safety filter for learning-based control of constrained nonlinear dynamical systems, Wabersich and Zeilinger, Automatica 2021.** The [paper](https://arxiv.org/abs/1812.05506) describes a module that checks a proposed input against predicted future constraint satisfaction, passing it through or replacing it. It explicitly incorporates dynamics uncertainty. **Read for:** how an adapter that improves behaviour differs from a filter intended to enforce constraints, and which state/model/backup assumptions the latter needs. **Difficulty:** advanced; basic dynamics, constraints and MPC help. For a broader tutorial, the authors' [Data-Driven Safety Filters overview](https://www.kimpeter.de/publicationsandsoftware/) describes reachability, control-barrier and predictive-control approaches; an [author-hosted manuscript](https://hybrid-robotics.berkeley.edu/publications/CSM2023_Safety_Filters.pdf) is available. Do not treat these guarantees as automatically applying to a latent world model or our monitor-only experiments.

### Inferences

- Suggested order: World Models, Residual RL, POPLIN, LOOP, predictive safety filters. That moves from “what is the small learned component?” to “where should it act?” and finally “what can be guaranteed?”
- For each resource, extract four items: inputs, outputs, frozen components and source of learning feedback. Comparing those will be more useful for selecting an interface than comparing neural-network sizes alone.
- A common architecture with task-specific learned weights is a more modest target than cross-environment transfer of identical weights. The resources motivate options, but do not settle that choice for TwoRoom and Block Plucking.

### Gaps

- This focused shortlist does not establish that candidate ranking is superior to residual correction, or verify a universal adapter for our two systems.
- Code links were inspected, not installed or executed. Legacy dependencies make them reading references rather than verified runnable packages in our environment.

## Which author-linked videos complement the readings?

### Takeaway

World Models, Residual RL and LOOP have author-linked video material. The separate video researcher should verify metadata and select a small watch list.

### Cited Findings

- World Models author article links to [Recurrent World Models Facilitate Policy Evolution](https://www.youtube.com/watch?v=HzA8LRqhujk). Direct YouTube page title and author-page provenance verified; no transcript was obtained. [Provenance](https://worldmodels.github.io/).
- Residual RL author project embeds [this demonstration](https://www.youtube.com/watch?v=RRllUka-r2A). The embed URL is verified on the author page; direct video fetch failed. Do not invent duration or claim full playback verification. [Provenance](https://residualrl.github.io/).
- LOOP author project embeds [this video](https://www.youtube.com/watch?v=Oj-EMWZB8Mg). The embed URL is verified on the author page; direct fetch failed. [Provenance](https://hari-sikchi.github.io/loop/).

### Inferences

- World Models is the best first conceptual resource; the residual robotics demonstration makes the action-correction interface concrete. LOOP fits after basic planning concepts.

### Gaps

- No reliable duration, transcript or timestamp was retrieved for these three links in this subtask. The other researcher owns broader YouTube verification.
