# Adjacent opportunities: robustness and reliability of filtered manipulation

Research date: 13 October 2026. This is a focused primary-source search and proposal assessment, not an exhaustive novelty review. No simulator or training was run. The recommendations below are inferences, and the current authority remains [oracle.md](../../history/20261018/oracle.md). Engineering effort means marginal effort after a working filtered manipulation baseline, whose reproduction remains a separate unresolved cost.

## Can evolution discover informative intervention failures and turn them into useful adaptation cases?

### Takeaway

This is the most attractive low marginal cost opportunity in this branch. The potential contribution is an intervention-specific diagnosis and repair process, rather than generic evolutionary falsification or curriculum generation, which already exist.

### Cited Findings

- Adaptive Stress Testing searches for likely failure trajectories in stochastic simulations using reinforcement learning. Black-box failure discovery is established, and finding an adversarial trajectory is distinct from estimating deployment failure probability. [AST primary paper](https://arxiv.org/abs/1811.02188)
- BEACON combines Bayesian optimization with CMA-ES to find control-system counterexamples with quantitative specification robustness. It provides a Python implementation, so using evolution for falsification is not itself new. [BEACON paper](https://arxiv.org/html/2403.05925v2), [author implementation](https://github.com/SAILRIT/BO-CMA)
- CaDRE formulates diverse safety-critical scenario generation as quality-diversity optimization, with bounded perturbations of real driving trajectories. This overlaps strongly with any proposal to maintain a diverse archive of difficult scenarios. [CaDRE primary paper](https://arxiv.org/html/2403.13208)
- ACCEL evolves curricula by editing high-regret environments, targeting the frontier of an RL agent's capabilities. Evolving useful training environments is established; a contribution must identify what information specific to safety intervention improves selection or learning. [ICML 2022 paper](https://proceedings.mlr.press/v162/parker-holder22a.html)
- *When World Models Lie* separates physical failure, safe completion and incompletion. It attributes some incompletion in its ManiSkill manipulation task to the nominal policy lacking training on post-intervention states; it also describes persistent UNISafe OOD-triggered intervention. These observations concern that paper's system, not a reproduction of the released UNISafe block task. [Primary paper, Section V-B](https://arxiv.org/html/2609.34300)
- The repository's existing static audit found a released task actor, separate safety filter and configurable simulator, but no verified exact mid-episode snapshot facility or end-to-end runtime budget. Reset-plus-prefix replay requires validation before counterfactual comparisons. [Existing asset audit](../Safe%20manipulation%20progress%20and%20robustness/feasibility_and_evolutionary_scope.md)

### Inferences

- **Candidate question:** Can a search objective based on sustained intervention plus lost task progress discover avoidable continuation failures more efficiently than random scenario sampling, and do those cases improve adaptation on held-out starts?
- **Minimal pilot:** Hold policy, world model and filter fixed. Search a small, physically valid initial-condition space, initially object poses within the original task distribution. Score task progress, intervention duration and episode outcome separately. Compare uniform sampling with CMA-ES at the same total simulator-step budget. Do not start with full coevolution or retrain the world model.
- **Avoidability evidence:** For a discovered stalled case, retain the same filter and outcome definitions. Find a successful safe continuation using a bounded alternative behaviour or compact adapter. A positive witness shows that this particular stall is avoidable within the tested interface and horizon. Failure to find one does not prove impossibility. If exact restoration is unavailable, use matched starts and validated deterministic prefix replay, and describe the weaker causal scope.
- **Attribution controls:** Compare original and alternative continuation after the same intervention history where feasible; additionally replay the original policy with more time to detect horizon problems. Inspect proposal, filtered action and executed action separately. Compare filter variants or thresholds only in separate diagnostic experiments, since those change the system being explained. Distinguish repeated rejection, changed-state policy failure, confidently wrong model predictions and genuinely unavailable safe routes.
- **Curriculum extension:** Feed discovered cases to ordinary filter-present RL and evolutionary adaptation using the same behavioural interface. Compare against uniformly sampled starts and a replay-prioritized curriculum selected from the same initial data. This separates the value of scenario selection from optimizer choice. Novel scientific content could be which intervention patterns predict useful adaptation cases, and whether diversity prevents training on one repetitive failure mode.
- **Budget:** Count scenario discovery, positive-witness search, discarded candidates, curriculum collection and adaptation steps. Provide a second comparison with a shared precomputed curriculum if useful, but label its data cost. Search-selected failures are not independent draws from the deployment distribution; reserve independent held-out evaluation for rates and uncertainty.
- **Effort:** Lower than a new learning architecture once reset, logging and outcome checks work. A frozen-system search can produce useful diagnostic evidence before implementing fine-tuning. A substantial honours result could characterize a reproducible failure family and demonstrate repair; a GECCO claim would need more than attaching CMA-ES to reset parameters.
- **Failure criteria:** Deprioritize if random sampling finds the same cases equally cheaply, discovered scenes are physically invalid or outside the defined scope, no meaningful continuation failures occur, or safe positive witnesses remain absent. If all gains disappear under equal total interactions or held-out starts, the curriculum contribution is unsupported.

### Gaps

- No exact prior match was established for search objectives targeting *avoidable* task-continuation failures under visual learned safety filtering. This is not a novelty assertion. Generic falsification, QD scenario generation, POET-style environment evolution, PAIRED and ACCEL substantially occupy the surrounding space.
- Runtime and restart fidelity on the actual run machine remain unverified. The physically valid search bounds and sufficient repetition budget need empirical calibration.

## Could world-model-assisted candidate screening make evolutionary adaptation more efficient?

### Takeaway

Model-assisted evolution is established and can be tested incrementally, but it is less attractive as the first contribution. The empirical question is whether a pretrained safety world model ranks useful task-policy changes reliably enough to save simulator interactions without suppressing novel safe behaviours.

### Cited Findings

- Dynamics-Aware Quality-Diversity (DA-QD) evolves an imagined skill repertoire with a learned dynamics model, then evaluates selected skills in the environment. It uses model ensembles and reports substantial sample-efficiency improvement in its robotics experiments. Thus imagined evolutionary screening with real evaluation is established. [DA-QD paper](https://arxiv.org/html/2109.08522)
- Reset-Free QD builds on DA-QD with a behaviour-selection policy using imagined repertoire diversity and environmental information for safer autonomous real-world evaluation. Learned-model-based safety selection of evolutionary candidates also has precedent. [RF-QD paper](https://arxiv.org/abs/2204.03655)
- SafeDreamer uses learned reward and cost models with constrained cross-entropy planning over imagined action trajectories. Safe search within a world model therefore already exists, even though trajectory planning differs from persistent policy adaptation. [ICLR 2024 paper](https://openreview.net/pdf?id=tsE5HLYtYg)
- UNISafe learns latent safety filtering with an ensemble uncertainty mechanism. Its task actor and safety components are inspectable in the release; this does not establish a trustworthy task-return surrogate for arbitrary adapters. [UNISafe paper](https://arxiv.org/html/2505.00779), [official implementation](https://github.com/CMU-IntentLab/UNISafe)

### Inferences

- **Candidate question:** Can short-horizon, observation-grounded screening predict the relative *simulator* safe-completion value of bounded policy adaptations well enough to improve budget efficiency?
- **Minimal pilot before search:** Generate a fixed modest pool of adapters. Evaluate every adapter both in imagination and in the simulator on common initial conditions. Measure rank agreement, false rejection of successful safe candidates, and false acceptance of unsafe candidates. Stratify by rollout horizon, intervention frequency and observed prediction error. This pilot can reject the idea without a new optimizer.
- **Method only if pilot supports it:** Screen a larger candidate pool with short imagined rollouts; simulator-evaluate shortlisted candidates, plus a fixed random fraction of rejected candidates to audit missed improvements. Candidate selection can use the model, but final selection and every physical safety claim use independently observed simulator outcomes.
- **Baselines:** Same EA with direct simulator evaluation, random screening, simple model-free return surrogate, and uncertainty-based screening. Hold actor representation, initial pool, seed distribution and information access fixed. Report both matched simulator-interaction results and total elapsed-time results. A high-cost ensemble can save interactions while increasing wall time.
- **Action semantics:** Imagining a proposal as if physically executed misrepresents a filtered controller. Either include the filter in imagined rollouts or explicitly use a proposal-conditioned surrogate of the filtered environment. Dynamics histories must use executed actions.
- **Effort:** Medium to high if faithful closed-loop filter imagination does not already exist. Lower-cost ranking diagnostics are feasible before training any new model. The likely value is a negative or conditional result about ranking reliability, rather than a generic claim that world models make EA sample efficient.
- **Failure criteria:** Stop if rankings do not beat random screening, the model systematically rejects novel safe solutions, saved simulator cost is dominated by model evaluation, or gains require additional uncounted training data. Do not interpret imagined violations or uncertainty as measured physical safety.

### Gaps

- No measured ranking quality for the released UNISafe actor-adaptation interface was found. A safety representation is not automatically a good task-performance representation.
- This branch should not restart the archived Walker imagination-exploitation project. It addresses whether screening is useful for the current manipulation problem and requires new experiment paths and independent simulator checks.

## Is bounded robust adaptation across world-model errors a stronger contribution?

### Takeaway

Robustness is a useful extension of an established continuation result, but broad safe evolutionary robustness and statistical verification already have close precedents. A narrow investigation of how adaptation interacts with learned-filter error is more defensible than promising general safety preservation.

### Cited Findings

- SMC-ES couples evolution strategies with statistical model checking for performance, safety and robustness properties, evaluated on Gymnasium and Safety Gymnasium. Statistical verification around evolutionary control is therefore not open territory. Its certificates concern a specified scenario distribution and confidence parameters. [SMC-ES, July 2026 preprint](https://arxiv.org/abs/2607.15003)
- *Safe and Robust Neural Policy Learning with Statistical Verification for Sim-to-Real Deployment in Robotics* integrates scenario-based ES, verification and curriculum expansion of an operating region, with Cartpole and quadrotor hardware. The latest version is dated 2 October 2026; v1 called the framework LyEvO. These are prior-art claims, not independently verified guarantees. [Current primary paper](https://arxiv.org/html/2608.06481v2), [version history](https://arxiv.org/abs/2608.06481)
- Safe Barrier Bayesian Optimization optimizes configurable CBF-based controller parameters for safe task performance and compares optimization approaches. Tuning safety-related controller settings is already studied; changing safety thresholds can confound claims of improved task behaviour. [Primary preprint](https://arxiv.org/html/2503.19349)
- *When World Models Lie* explicitly studies model mismatch, including observation delay and transition-scale mismatch, and says its adaptive filter does not provide formal physical safety guarantees. It also notes calibration exchangeability can fail if filter recovery actions are absent from calibration data. [Primary paper, Sections V and VII](https://arxiv.org/html/2609.34300)
- Existing source inspection found that UNISafe normal and hard tasks change geometry, mass, thresholds, rewards and initial posture together. They cannot serve as an isolated mass or model-error ablation without new controlled configurations. [Existing asset audit](../Safe%20manipulation%20progress%20and%20robustness/feasibility_and_evolutionary_scope.md)

### Inferences

- **Candidate question:** Does a bounded adapter improve safe completion across a defined range of model mismatch, and does adaptation concentrate the policy in states where the learned safety model is more or less reliable?
- **Minimal pilot:** First establish a nominal adaptation benefit. Then choose one controlled error axis: observation delay or one physical parameter, keeping task definitions fixed. Evaluate unchanged, nominally adapted and mixed-condition-adapted policies on a held-out grid and random held-out starts. Log prediction error and safety scores alongside physical outcomes.
- **Strong comparison:** Compare ordinary RL and EA on the same adapter and training conditions, with nominal training versus matched domain randomization for each. If the method claims value from error feedback, give that feedback to both algorithms and ablate it. Keep simulator state available only for evaluation unless every learner gets it.
- **Boundedness:** Constrain adapter magnitude for interpretability and easier attribution, but never present a parameter norm bound as a safety guarantee. A fixed filter cannot establish unchanged risk after state visitation changes.
- **Potential finding:** Adaptation may avoid filter-hostile trajectories, improving completion, or may gain task success by visiting confidently mispredicted states and increase violations. Establishing which mechanism occurs and under which conditions could be valuable empirical science. Robustness cannot be inferred merely from a nominal success increase.
- **Effort:** Low to medium as an evaluation extension if delay or physics controls are validated. High as a new certification or domain-general robust training method. Do not import the mathematical guarantees of SMC-ES into a visual manipulation system without checking all assumptions and sampling conditions.
- **Failure criteria:** Deprioritize if all gains are nominal-only, changing task parameters also changes the safety definition, the original filter fails catastrophically across the entire shift range, or evidence cannot separate policy adaptation from weakened filtering. Report safe success, violations and unfinished episodes together; zero violations in independent evaluation only supports a finite upper rate bound.

### Gaps

- No verified run-machine budget, minimum independent safety sample size or physically realistic robustness interval is available. Choose these after baseline timing and diagnostic observations, not from unrelated hardware latency numbers.
- Latest robust-evolution preprints are relevant overlaps, but their proofs and reported gains were not independently reproduced in this search. These opportunities remain conditional empirical questions, not established novelty claims.
