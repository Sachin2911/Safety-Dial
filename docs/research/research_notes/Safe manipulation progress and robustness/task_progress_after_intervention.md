# Task progress after safety intervention

Research date: 8 October 2026. Scope: continued task execution in manipulation with learned safety filters. This is a literature assessment, not a reproduced result or an experiment plan. The earlier metaplanner proposal is not assumed.

## 1. What is the problem, and how strong is the manipulation evidence?

### Takeaway

A filter can stop a dangerous action without teaching the task policy what to do next. That is a documented practical limitation, but training the task policy with its filter already exists. The useful question concerns which remaining failures are actually due to poor continuation, and which remedies work with limited adaptation experience.

### Cited findings

**UNISafe establishes headroom, not its cause.** For normal block plucking with a diffusion task policy, Appendix E.2, Table 15 reports safe success/failure/incompletion of 52/44/4% without filtering and 57/15/28% with UNISafe. With a Dreamer task policy, the corresponding figures are 58/41/1% and 72/20/8%. Thus increased incompletion accompanies reduced failure, but depends strongly on the nominal policy. The evaluation denominator was not located in the inspected text; percentages should not be converted into invented counts. ([UNISafe, Table 15](https://arxiv.org/html/2505.00779))

**A successor explicitly identifies continuation.** In simulated Franka globe placement beside a vase, *When World Models Lie* evaluates timing mismatch and observation delay. Section V-B attributes some incompletions after successful protection to a nominal policy that was not trained to resume after intervention. Section VII also identifies calibration-data mismatch when backup actions are poorly represented. These are author-stated explanations and limitations, not controlled proof that every stalled trajectory has a viable continuation. ([Paper, sections V-B and VII](https://arxiv.org/html/2609.34300))

**Improving the filter alone already helps substantially.** LatentCBF evaluates a fixed diffusion policy lifting an open candy bag: 20 trials from each of three initial configurations, 60 trials per method. Table 2 reports aggregate failure/stall/success of 62/0/38% unfiltered, 0/62/38% with least-restrictive filtering, and 0/20/80% with LatentCBF. In the difficult multimodal configuration, its remaining stall rate is 60%, despite no observed spills. Section 7 acknowledges that filtering can place the nominal policy in unfamiliar states. These rounded results show both an existing remedy and residual difficulty. ([LatentCBF, section 6.2 and Table 2](https://arxiv.org/html/2511.18606))

### Inferences

Use **task resumption** to mean restoring useful task progress after protection. This requires neither an irreversibility label nor a guarantee that every state is recoverable. Several distinct explanations must remain separate:

- **Policy/filter mismatch:** the policy repeatedly proposes an action the filter rejects, or cannot act competently from the resulting configuration. Intervention-induced distribution shift is one possible mechanism.
- **Overconservatism or critic error:** the filter rejects feasible progress. A better task policy cannot use actions that remain unavailable to it.
- **Prediction or observation failure:** contacts, hidden state or latent dynamics errors make the filter's evaluation wrong. Apparent policy failure can originate upstream.
- **No feasible continuation:** a task may genuinely be blocked under its current constraints. Failure of a finite search does not establish this.
- **Action or memory inconsistency:** stored actions, action normalization, recurrent histories or filter state do not correspond to what the plant received.
- **Insufficient time:** a safe detour simply exceeds the episode horizon. An incomplete episode does not by itself demonstrate deadlock.

Reserve **chattering** for frequent switching between controllers, and **deadlock** for sustained lack of progress under the controller composition. Neither should be inferred from a large intervention count alone. This diagnostic taxonomy is our synthesis, rather than a claim that the papers experimentally isolate every category.

### Gaps

Published aggregate outcomes do not identify how often each mechanism occurs in the released UNISafe task. We have not reproduced its baseline, established useful post-intervention coverage, or shown that another admissible controller completes the same cases. The strongest author-stated continuation evidence comes from a different manipulation setup; transporting that explanation to block plucking requires evidence.

## 2. Which remedy families already exist, and what do they actually learn?

### Takeaway

The existing remedies act at different levels: improve the filter, train the task policy inside the filtered environment, learn a residual correction, learn from interventions, or compose safety and task control more smoothly. These are strong alternatives to compare, rather than ingredients whose combination automatically creates novelty.

### Cited findings

The ten closest method-level comparisons are below. Dates distinguish verified venues from the preprint/version inspected; an unverified venue is not evidence of rejection or nonpublication.

| Work and status | Learned or fixed components; feedback and assumptions | Relevance and remaining limit |
| --- | --- | --- |
| **UNISafe**, CoRL 2025 | Offline failure-labelled trajectories train a Dreamer-style latent world model and uncertainty estimator. Safety value and backup behaviour are learned through latent rollouts; nominal task policies are separate. | A ready example of a learned visual filter, not task-policy adaptation through that filter. Its world model supplies imagined dynamics and latent state. ([Methods and appendix B](https://arxiv.org/html/2505.00779)) |
| **LatentCBF**, 23 November 2025 preprint | Learns a smooth safety margin with gradient penalties and a safety value from mixed nominal/safety-policy trajectories using DINO-WM. The evaluated diffusion task policy is previously trained. | Mixed-policy **safety-value training is not nominal-policy fine-tuning**. Better action correction already improves completion; latent/model/critic approximation still prevents formal guarantees. ([Sections 4-7](https://arxiv.org/html/2511.18606)) |
| **When World Models Lie**, 28 September 2026 preprint | Uses pretrained latent dynamics, safety value and backup controller. Online observed prediction errors adjust an uncertainty set used for pessimistic safety evaluation. | Adapts filtering rather than task behaviour. Its hardware experiment uses 350 successful and 150 failed teleoperation trajectories for learning; evaluation replays 20 successful and 20 failed trajectories in each of three conditions for four filters, 480 runs. These are not 480 independently learned autonomous tasks. ([Sections IV-VI](https://arxiv.org/html/2609.34300)) |
| **Recovery RL**, RA-L/ICRA 2021 | Offline constraint-labelled transitions initialize a safety critic and backup controller; task SAC learns online with intervention. Safety components also update online. Model-free and learned-model planning variants are considered. | Already learns useful task behaviour under switching. Includes simulated contact-rich object extraction, while its physical experiment is obstacle avoidance. Relevant for controller composition and data semantics, not a requirement to restore a recovery-label programme. ([Sections IV-V](https://arxiv.org/html/2010.15920)) |
| **Provably Optimal RL under Safety Filtering**, IASEAI 2026; September 2026 revision | Treats a fixed filter as part of the environment and trains ordinary task RL through it. Rewards and transitions correspond to executed filtered actions; the agent need not explicitly observe intervention. | Provides a theoretical reason to make ordinary filtered fine-tuning a mandatory baseline. Empirical tasks are Safety Gymnasium, not contact-rich visual manipulation. ([Sections 3-5 and publication record](https://arxiv.org/html/2510.18082); [venue](https://arxiv.org/abs/2510.18082)) |
| **Reducing Safety Interventions in Provably Safe RL**, 2023 | Trains PPO with a reachability-based safety mechanism. Proactive replacement samples a verified alternative; projection seeks a nearby safe action before emergency fallback. Uses system models and bounded uncertainty. | In human-robot collaboration reaching, failsafe intervention falls from about 2.5% of steps to 0.25%; replacement preserves task performance. This is step frequency, not episode failure. It handles intervention quality already, but is not learned visual contact manipulation. ([Methods and experiments](https://arxiv.org/html/2303.03339)) |
| **Residual RL from Demonstrations**, June 2021 preprint inspected | Freezes a visual behaviour-cloned base policy and its representation, then learns an additive residual using proprioception, base action and visual features. Uses sparse success rewards and distributional MPO in simulated arm/dexterous-hand tasks. | Direct precedent for compact policy correction, compared with whole-policy fine-tuning. It improves task behaviour without a dedicated learned safety filter, so does not settle filtering compatibility. ([Method and experiments](https://arxiv.org/html/2106.08050)) |
| **CBF-RL**, October 2025 preprint | Combines filtering during policy training with reward terms reflecting barrier-condition violations and correction magnitude. Navigation and humanoid experiments use structured safety information. | Shows that learning to avoid repeated corrections is established. Its deployment-without-filter experiments do not justify removing a fallible visual filter in manipulation; the information and safety assumptions differ. ([Method and ablations](https://arxiv.org/html/2510.14959)) |
| **AutoSafe**, 30 June 2026 preprint | Trains a performance policy and a small intervention-sharpness head; a structured monitor blends its action smoothly with a safety prior. Equation 5 gives the convex composition and equations 10-11 the learned profile. | Addresses disruptive hard switching and optimization behaviour, using model-based safety structure. Includes physical cart-pole validation, not a demonstrated latent manipulation solution. Feasible action interpolation alone does not prove arbitrary nonlinear physical safety. ([Section 4 and experiments](https://arxiv.org/html/2606.31320)) |
| **HIL-SERL**, October 2024 preprint; March 2025 version inspected | Learns visual task policies/critics online from demonstrations, robot experience and human corrections; a trained success classifier supplies rewards. Human intervention samples also enter the demonstration buffer. | Demonstrates practical real contact-rich policy improvement. The human supplies task competence and judgement unavailable to an automatic filter. Evaluation generally uses 100 autonomous trials; its Jenga and flipping tasks do not provide the same online-intervention evidence because interventions were unsuitable there. ([System, experiments and appendix](https://arxiv.org/html/2410.21845)) |

**What filtered-MDP optimality does and does not establish.** The theorem assumes measurable stationary dynamics/filter, bounded rewards, safe initialization, and a perfect filter on the maximal controlled-invariant safe set. That filter always returns a safe action and leaves every safe action unchanged. Convergence transfers only for an RL algorithm that already converges on the relevant stationary discounted MDP class. Optimality concerns the same filter at deployment. This is not a finite-data convergence theorem for arbitrary deep RL, nor a safety guarantee for an approximate latent filter in a partially observed changing system. ([Definitions 2-3, Assumption 1, Theorem 1](https://arxiv.org/html/2510.18082))

**The action label depends on the learned object.** Recovery RL, section IV-B and Algorithm 1, stores `(state, proposed task action, next state, task reward)` in its task buffer, but `(state, executed action, next state, constraint)` in its safety buffer. The former learns consequences of proposals in the composed environment; the latter learns physical action risk. A plant dynamics model likewise needs the executed action in its trained coordinates. Universally substituting executed actions into a proposal-value learner can remove the very association it needs to learn. ([Original buffer definitions](https://arxiv.org/html/2010.15920))

### Inferences

Simple task-policy fine-tuning with the filter is already an answer to the broad idea. The unresolved practical comparison is how well it works with sparse task reward, unfamiliar post-intervention contacts, imperfect latent safety estimates and a modest interaction budget. Explicit intervention features or correction penalties are options, not prerequisites for calling a learner filter-compatible.

Improving the agent and improving the filter answer different questions. Both deserve comparison when the diagnosis is ambiguous. Smoothness need not solve missing task competence; more task competence need not fix false-safe prediction.

### Gaps

This review did not find a controlled comparison of all these remedy families on one released visual manipulation filter. Their separate successes do not establish which component dominates UNISafe incompletion. Full retraining costs, reliable post-intervention coverage and compatibility with recurrent task actors remain implementation-specific.

## 3. What would make this a bounded, useful research contribution?

### Takeaway

This is a plausible empirical contribution if the released controller has a reproducible continuation problem and compact adaptation improves safe completion against ordinary filtered fine-tuning. It does not currently support a claim that evolving behaviour after intervention is a first-of-kind algorithm.

### Cited findings

The literature already provides concrete contrasting routes: residual adaptation holds a competent visual base fixed, while learning in the filtered MDP optimizes task behaviour with the safety mechanism continuously present. HIL-SERL additionally shows why useful corrective experience can matter, although its human supplies much more information than a binary intervention flag. These are comparisons of information and learning setup, not interchangeable implementations. ([Residual adaptation](https://arxiv.org/html/2106.08050); [filtered learning](https://arxiv.org/html/2510.18082); [human corrections](https://arxiv.org/html/2410.21845))

### Inferences

**A bounded question:** at a fixed adaptation budget, can an existing task controller become better at completing the job under an imperfect learned filter, while preserving the filter's safety benefit? Begin with one task, one documented failure mechanism and an unchanged observation/action interface. This need not retrain a world model or create a new safety specification.

**Necessary comparisons.** Use the frozen task-policy/filter pair as the reference. Compare ordinary task-policy fine-tuning with the same filter, compact residual or task-head adaptation under the same information, and simple random search when the parameter count is small. Include an improved-filter comparison if overly restrictive filtering plausibly explains the stall. A task-specific continuation rule can be informative when it uses explicitly declared privileged information. No method should receive uncharged simulator branches or richer feedback unnoticed.

**Where evolution fits.** A standard EA can optimize a compact controller adjustment from actual outcomes of the complete filtered simulator. It may be convenient when switching, clipping, stochastic contacts or an opaque task actor make gradients awkward. This is a rationale to test, not evidence that evolution outperforms RL. Hold the adaptation interface fixed when comparing optimizers. Demonstrating a useful, reproducible safety/progress trade-off can be worthwhile without inventing an algorithm; replacing RL by CMA-ES alone is insufficient.

**Measure completion and safety together.** Report overall safe completion, physical violations and unfinished episodes with counts and denominators. Add task progress, completion time and censoring at the horizon; intervention step fraction, correction magnitude, switching count and time spent continuously under backup are explanatory measurements. Fewer interventions can mean better behaviour, a robot that stops moving, or a weakened filter. It is not an objective that stands alone.

For post-intervention diagnostics, record progress immediately before intervention and through a prespecified later window, plus return to nominal control and repeated switching. Conditional success among intervened episodes is descriptive: different methods intervene on different populations. Common initial conditions support fair overall comparison. Stronger causal claims require controlled continuation comparisons from validated identical simulator states, including controller memory and filter history where relevant. Failed alternatives do not certify impossibility.

**Preserve action semantics.** Log separate copies of the proposal, filtered command and final applied command, including normalization, timing, observations and event definitions. Distinguish a task learner's filtered transition from a world model's plant transition. A recurrent latent model needs a consistent executed-action history. Timeout, retrospective trajectory labels and a true physical failure event must remain distinct.

**Keep the world model's role explicit.** It may supply the representation, predict futures used to train a safety critic, or be queried online by the filter. The task adapter can still receive fitness from real simulator outcomes. Imagined value improvement is not observed safe completion. Count model queries separately from physical steps and wall time, including common pretraining when making total interaction-efficiency claims.

Training violations, autonomous evaluation violations and runtime protection are separate claims. A learned filter that reduces violations during deployment does not establish safe adaptation, and zero observed evaluation failures does not prove categorical safety. Keep the filter present when evaluating a controller trained to rely on it unless removal is itself the declared experiment.

### Gaps and search limits

The immediate uncertainty is practical rather than conceptual: whether enough resumable failures occur in a reproducible released task, and whether ordinary filtered fine-tuning removes most of them. If that simple baseline succeeds cheaply, the specific EA angle needs a distinct empirical advantage; if it fails, diagnose the failure before increasing architectural complexity.

Targeted searches covered latent-filter task completion, safety interventions and policy learning, action correction/relabelling, residual RL from demonstrations, and smooth policy composition. Backward/forward connections from UNISafe, LatentCBF and the adaptive-filter paper were followed into the ten methods above; method sections, equations, tables and arXiv publication records were checked. This was a decision-focused search through 8 October 2026, not an exhaustive review or proof of absence. No experiment, dependency installation or asset modification was performed. A dedicated EA precedent and artifact audit accompanies these notes.
