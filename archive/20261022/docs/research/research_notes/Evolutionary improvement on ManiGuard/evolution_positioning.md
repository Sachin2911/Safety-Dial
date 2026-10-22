# Evolutionary method positioning for improvement on ManiGuard

Research date: 19 October 2026. Scope: a prospective route to improving the released ManiGuard policies, followed by a separate test of evolution's contribution. These are research recommendations, not results or a frozen protocol. No simulator, rental, installation or new physical evaluation was performed for these notes. Sources below were checked independently of repository citation lists.

## What has already been done, and where could a contribution remain?

### Takeaway

The strongest next candidate is a reusable, compact noise-steering controller around the frozen flow VLA, evaluated directly against physical safe completion. Its advantage over the existing cap is behavioral power: it can change which action the policy generates. Its novelty is conditional. Frozen-policy steering, learned noise policies, evolutionary VLA adaptation, confidence-aware safe evolution and evolved staged programs all have close primary precedents. The defensible initial claim is an empirical method for budgeted safe adaptation on ManiGuard, not a new optimizer or the first evolutionary VLA method.

### Cited Findings

#### Verified primary source ledger and precise distinctions

1. **ManiGuard, arXiv 2608.17386v1, August 2026.** The benchmark defines safe success jointly from completion and monitor satisfaction. Its reported Clutter results have zero violations across evaluated policies despite substantial completion variation; Cabinet and Dusty safe success stays below 2% for the fine-tuned policies. This separates task competence from safety headroom. The monitor evaluates trajectories, rather than certifying proposed actions. Full HTML inspected, especially Sections 4.3, 5 and 6. [Paper](https://arxiv.org/html/2608.17386v1).

2. **VLA-Pilot, arXiv 2511.14178v2, 16 April 2026; first version 18 November 2025.** It uses a multimodal language model to construct a context-dependent reward, then scores and evolves action candidates during deployment through partial noising and denoising. Its published abstract covers six real manipulation tasks and two embodiments. This is direct prior art against claiming that evolution can steer a frozen VLA at inference time. A reusable adapter trained from actual episode outcomes is a different adaptation object and cost placement. Full HTML method inspected, particularly Sections III-A to III-D. The project claims RA-L 2026 acceptance, whereas arXiv comments say submitted; acceptance was not independently verified at the publisher. [Paper](https://arxiv.org/html/2511.14178v2), [project](https://rip4kobe.github.io/vla-pilot/).

3. **DSRL, arXiv 2506.15799v2, 25 June 2025; first version 18 June.** It learns a state-conditioned noise policy with RL while freezing the diffusion/flow policy. The denoiser becomes part of the latent-action environment. Appendix B already trains a single-step noise actor and repeats its output across a chunk to reduce the dimension for pi0. DSRL-SAC and DSRL-NA, the latter using noise aliasing, are substantive ordinary-learning baselines. Its original real-world generalist experiment uses pi0-DROID, not the specific ManiGuard pi0.5 checkpoint. Full HTML including the generalist appendix inspected; official code existence verified, not installed. [Paper](https://arxiv.org/html/2506.15799v2), [official code](https://github.com/ajwagen/dsrl).

4. **Flow Reversal Steering, arXiv 2606.13675v1, June 2026.** This is primary evidence that pi0.5 admits useful noise steering: it evaluates pi0.5-LIBERO and pi0.5-DROID, and Appendix E explicitly describes frozen pi0.5 noise-policy training. Its simulated noise policy predicts seven active dimensions, repeats across the chunk and pads the remaining dimensions with independent Gaussian noise. Section C.3 reports that changing padding noise has little effect. This supports checking active channels rather than spending a small genotype mainly on padding. It does not establish steerability of our joint-action checkpoints. Full HTML inspected, including Sections C.3 and E. [Paper](https://arxiv.org/html/2606.13675v1).

5. **Online-ES, arXiv 2609.38855v1, 30 September 2026.** It samples and perturbs action trajectories, executes them for outcome feedback, and backpropagates a feedback-weighted MSE objective into VLA parameters. Section V-A uses GR00T N1.5 plus depth features and finetunes the action expert while freezing other modules. Section IV uses binary task-completion feedback and failure regularization. Therefore critic-free evolutionary VLA improvement is already covered. Our wholly frozen base and reusable latent controller would differ. Its repeated resets, base trajectories and perturbed executions are adaptation interaction costs; I did not verify a comprehensive wall-time/interaction ledger comparable to ours. A text search found no safety discussion, but that is not proof that every supplementary artifact lacks one. Full HTML introduction, method and experiment setup inspected. [Paper](https://arxiv.org/html/2609.38855v1).

6. **ResiP, arXiv 2407.16677, first published July 2024.** A frozen chunked BC policy is supplemented by a per-step closed-loop residual trained with PPO. The project reports improved precision assembly and explicitly attributes the issue to limited reactivity of open-loop chunks. It also describes distillation and state/vision distinctions. Thus a frozen VLA plus a corrective controller is not novel by itself; direct action-residual RL is essential if claiming a better adaptation representation. Primary abstract and full project methods inspected. [Paper record](https://arxiv.org/abs/2407.16677), [project](https://residual-assembly.github.io/).

7. **ResFiT, arXiv 2509.19301v2, 25 September 2025; first version 23 September.** This is a real paper, titled *Residual Off-Policy RL for Finetuning Behavior Cloning Policies*, with an official Amazon FAR repository. It is distinct from the unrelated shape-fitting paper called ResFit. It provides a stronger off-policy residual-learning reference than a weak episodic categorical update alone. Primary abstract and official repository identity inspected; full experimental equivalence to ManiGuard was not audited. [Paper](https://arxiv.org/abs/2509.19301), [official code](https://github.com/amazon-far/residual-offpolicy-rl).

8. **DLSRL, arXiv 2609.11270v1, 10 September 2026.** Its actor predicts both an initial-noise latent and an action-representation latent injected into frozen Transformer blocks. This limits a novelty claim based on adding a second steering site or calling a latent adapter more expressive. Only the primary abstract was inspected; the claimed empirical advantage and integration cost were not independently reproduced. [Paper](https://arxiv.org/abs/2609.11270).

9. **InterEvolve, arXiv 2610.02196v2, 4 October 2026; first version 1 October.** It evolves staged reward programs around a fixed humanoid controller, with an LLM revising structure and CMA-ES tuning constants. The controller uses an object-aware forward-backward model and a pretrained residual. Full v1 methods describe fixed verification, many scenarios and incumbent/challenger confirmation. This blocks a broad claim of first evolving reusable staged programs around a frozen foundation controller. ManiGuard's VLA does not supply that paper's reward-to-latent interface. Abstract v2 and methods v1 inspected; v2 methodological changes were not exhaustively compared. [Current record](https://arxiv.org/abs/2610.02196), [v1 methods](https://arxiv.org/html/2610.02196v1).

10. **Genetic-programmed behavior trees.** *Learning Behavior Trees with Genetic Programming in Unpredictable Environments*, arXiv 2011.03252, was submitted 6 November 2020 and explicitly learns tree structure for a robotic task. Evolving a guarded sequence is therefore an established representation family. Abstract checked. [Paper](https://arxiv.org/abs/2011.03252).

11. **Extract-QD, arXiv 2502.06585v1, 10 February 2025; GECCO 2025 in the full manuscript.** It unifies uncertain-QD methods and proposes Extract-ME, including reevaluation/archive mechanisms. Adding noisy MAP-Elites to manipulation is not an unoccupied algorithmic contribution. Full HTML introduction and framework inspected. [Paper](https://arxiv.org/html/2502.06585v1).

12. **QD and scenario generation.** Fontaine and Nikolaidis, *A Quality Diversity Approach to Automatically Generating Human-Robot Interaction Scenarios in Shared Autonomy*, arXiv 2012.04283, already uses MAP-Elites to find diverse failure scenarios and compares it with Monte Carlo and optimization. Failure discovery is an evaluation result until a repair learner is trained and independently tested. Primary abstract checked. [Paper](https://arxiv.org/abs/2012.04283).

13. **QDAC, ICML 2024.** *Quality-Diversity Actor-Critic: Learning High-Performing and Diverse Behaviors via Value and Successor Features Critics* is a real primary precedent for learned behavioral repertoires. QD does not imply one deployable controller or a no-critic advantage. Primary PMLR page checked. [Paper](https://proceedings.mlr.press/v235/grillotti24a.html).

14. **Confidence-aware safe evolution exists.** *Synthesizing safe policies under probabilistic constraints with reinforcement learning and Bayesian model checking*, Science of Computer Programming 2021, introduces PSyCo and Safe Neural Evolutionary Strategies, using Bayesian model checking and confidence in constraint satisfaction to adapt a Lagrangian. Publisher abstract and contribution text checked. Therefore confidence-aware evolutionary safety selection alone is not a new idea. [Publisher](https://doi.org/10.1016/j.scico.2021.102620).

15. **Noisy evolution has competing budget trade-offs.** *Large-Scale Noise-Resilient Evolution-Strategies*, GECCO 2019, is an accessible primary paper; racing and reevaluation are established. *Depth over Fidelity in Fixed-Budget Noisy Evolution Strategies*, arXiv 2606.06555v1, studies the cost of stabilization through reduced generations and proposes uncertainty integration and switching. Its assumptions and experiments do not establish the optimum for ManiGuard's correlated binary outcomes. Primary paper and full HTML inspected respectively. [2019 paper](https://www.cmap.polytechnique.fr/~nikolaus.hansen/proceedings/2019/GECCO/proceedings/proceedings_files/pap144s3-file1.pdf), [2026 preprint](https://arxiv.org/html/2606.06555v1).

16. **Curriculum repair also has close prior art.** ACCEL, *Evolving Curricula with Regret-Based Environment Design*, arXiv 2203.01302, mutates useful levels at an agent's learning frontier. PLR selectively revisits levels with estimated learning potential. These are stronger baselines than uniform scenario randomization alone. Primary abstract/project and PMLR page checked. [ACCEL](https://arxiv.org/abs/2203.01302), [official project](https://accelagent.github.io/), [PLR](https://proceedings.mlr.press/v139/jiang21b.html).

17. **Local relevance.** The recovery plan records the Part 1 adverse comparison, inactive Part 2 timing trigger, failed prediction audit and false-alarm-heavy warning model. It separates direct execution search from model/acquisition prerequisites. The Part 1 protocol pins pi0.5 Clutter and its legacy source profile. Existing episode-seeded policy wrappers currently call `infer(observation)` without passing a noise argument. [Recovery plan](../../history/20261019/phase3Recovery.md), [Part 1 protocol](../../../../configs/maniguard/phase3/part1.json), [policy wrapper](../../../../experiments/helpers/maniguard_policy.py), [parallel policy wrapper](../../../../experiments/helpers/maniguard_parallel_policy.py).

### Inferences

- The scientific gap is narrower than “evolution improves a VLA”: do reusable evolutionary adapters improve **physical safe completion**, efficiently and robustly, when interaction and information advantages are matched and the base policy stays entirely frozen?
- A standard optimizer can support a credible empirical systems paper if the representation, mechanism, safety outcomes and budget behavior are convincing across tasks. A new method claim additionally needs a distinctive algorithmic ingredient that survives controlled comparisons with its nearest alternatives.
- A scalar safety penalty, feasibility rules or uncertainty-aware ranking may be useful engineering choices. Their names cannot carry the novelty. An empirical observation that physical binary evaluation noise changes which adaptation method wins could be valuable, but it needs replicated evidence and a useful remedy.
- The strongest design should first expose competent alternative behavior from the released VLA. Current failure evidence does not imply that any search method can fix inadequate task competence, missing perception or unreachable actions.

### Gaps

- No primary evidence establishes that our exact pi0.5 joint-action checkpoints have helpful, locally heritable low-dimensional noise directions. Family-level pi0.5 results only justify a cheap affordance test.
- I found no verified ManiGuard-specific evolutionary improvement paper in the scoped searches. This is not a claim of exhaustive absence or priority.
- Deb's 2000 constraint-handling paper exists at DOI `10.1016/S0045-7825(99)00389-8`, but its primary publisher page failed to open. Do not cite a detailed reading of that paper from these notes; PSyCo is the verified close safe-evolution source.
- VLA-Pilot's code link on the project page says coming soon. This does not prove that no later repository exists. A paper-level implementation comparison requires checking availability again.
- No installation, executable baseline reproduction, published effect replication, GPU memory measurement or fresh single-GPU throughput measurement was done here.

## Which avenues are realistic, and what would falsify each?

### Takeaway

Prioritize a stronger latent behavior interface and physical outcomes. Use direct bounded residual control as the fallback. Keep program composition, repertoires and adversarial curricula as conditional later routes. Repairing the warning model or adding a world model is not required to ask whether evolution improves the benchmark.

### Cited Findings

The comparison below is a prospective assessment. Novelty risk, expected costs and expected nulls are judgments, not empirical results. The cited papers establish neighboring methods; the proposed ManiGuard outcomes remain untested.

#### 1. Reusable low-rank, state-conditioned flow-noise steering: recommended primary

- **Behavioral power:** can select a different action mode and direction rather than merely attenuating the current proposal. Full-policy reach remains limited by the frozen generator and the chosen latent subspace.
- **Evaluator:** direct physical completion and native monitor, with no imagined safety labels. Outcome noise remains substantial.
- **Cost:** one native policy sample per query at deployment; small CPU controller. Requires an audited explicit-noise server path and saved-input checks, not new VLA training.
- **Evolutionary rationale:** reuse and local modification of a behavior-generating controller may exploit interdependent latent coefficients and episodic outcome feedback without fitting a sparse-return critic. This is a hypothesis; smooth latent effects or an easy low-dimensional optimum may favor random search or ordinary learning equally.
- **Novelty risk:** high for the steering concept because DSRL/FRS already learn compact, state-conditioned noise. The potential contribution is specification-grounded physical adaptation and a supported budget/noise mechanism. [DSRL](https://arxiv.org/html/2506.15799v2), [FRS](https://arxiv.org/html/2606.13675v1), [Online-ES](https://arxiv.org/html/2609.38855v1).
- **Essential baselines:** native checkpoint, constant noise bias, direct residual, matched random search, matched parameter learning, and a strong latent-action RL baseline in the paper study.
- **Expected null:** the checkpoint maps most usable noises to the same action, the needed skill is absent, local mutations have no consistent physical effect, or evolution matches random search.

#### 2. Bounded proprioceptive/history-conditioned action residual: recommended fallback

- **Behavioral power:** can change target direction and local contact approach on every control step, beyond a cap. A low-rank residual is not a general task planner.
- **Evaluator:** direct physical labels; joint-target units and final clipping must be audited.
- **Cost:** avoids policy internals and extra VLA queries. The existing execution hook is a relatively simple insertion point, but choosing gains and establishing actual physical effects still costs episodes.
- **Evolutionary rationale:** compact coefficients can be evolved from delayed black-box outcomes. Ordinary PPO/SAC residual learning is a serious alternative and may be better.
- **Novelty risk:** high because ResiP and ResFiT already improve frozen BC using residuals. A paper needs measured advantages in the safety/cost setting, not the residual architecture alone. [ResiP](https://residual-assembly.github.io/), [ResFiT](https://arxiv.org/abs/2509.19301).
- **Essential baselines:** constant offset, tracking-error feedback rule, cap, matched random and PG, plus residual PPO/SAC with the same sensors and physical budget.
- **Expected null:** correction harms a calibrated policy, only one fixed bias helps, learning matches evolution, or task failure requires semantic reasoning the chosen inputs cannot supply.

#### 3. Evolved guarded task programs or subgoal-language schedules

- **Behavioral power:** potentially high for safe ordering, retries and prerequisite management, conditional on the VLA executing the supplied subskills.
- **Evaluator:** physical joint outcomes are reliable labels; observable transition guards are the bottleneck. Privileged simulator predicates would give a separate oracle setting.
- **Cost:** new skill vocabulary, skill-competence checks and grounded guards make this substantially more work than a noise hook.
- **Evolutionary rationale:** discrete structure and temporal composition are natural evolutionary objects, but a scripted automaton, planner or ordinary hierarchy can capture the same prerequisites.
- **Novelty risk:** high for generic GP trees and higher still after InterEvolve. A ManiGuard-specific contribution would need constraint-grounded composition and meaningful transfer. [GP behavior trees](https://arxiv.org/abs/2011.03252), [InterEvolve](https://arxiv.org/abs/2610.02196).
- **Essential baselines:** native instruction, fixed decomposition, hand-authored guarded program, random programs, ordinary learned hierarchy; same skill library and observations.
- **Expected null:** component skills already fail in valid states, guards lack observable information, or a small fixed program explains the whole benefit.

#### 4. QD repertoire of safe alternative steering/residual controllers

- **Behavioral power:** multiple contact styles may be complementary, but a useful deployable repertoire also needs a non-oracle selector.
- **Evaluator:** descriptors and safety outcomes are both noisy. An archive can fill with lucky elites or empty behavioral distinctions.
- **Cost:** several solutions per cell, repeat evaluations and selector development. A tiny sparse archive does not justify a full QD story.
- **Evolutionary rationale:** preservation of specialist alternatives could outperform collapsing all tasks into one average objective.
- **Novelty risk:** substantial because uncertain QD and learned control repertoires exist. The useful new question would be whether audited safety-relevant diversity improves fresh safe completion under a fixed total cost. [Extract-QD](https://arxiv.org/html/2502.06585v1), [QDAC](https://proceedings.mlr.press/v235/grillotti24a.html).
- **Essential baselines:** single best controller, randomly collected repertoire, mixture of independently trained specialists, same selector; oracle selection only as a labeled upper bound.
- **Expected null:** descriptors are cosmetic, all cells contain the same behavior, gains disappear with a realistic selector, or repeat costs dominate.

#### 5. Noise-aware feasibility selection and adaptive reevaluation

- **Behavioral power:** none by itself. This improves evaluation allocation only if the representation already supports useful alternatives.
- **Evaluator:** four physical outcomes per episode give richer data than a scalar, but rare violations and task correlations make short-binomial confidence claims fragile.
- **Cost:** more repeats mean fewer new candidates/generations. Final independent evaluation must be protected from that competition.
- **Evolutionary rationale:** distinguish a lucky parent from a reliably promising lineage and allocate confirmation around consequential uncertain decisions.
- **Novelty risk:** high for generic confidence-aware evolution or racing. A distinct contribution needs a supported multi-outcome allocation rule and comparisons with established uncertainty handling. [PSyCo](https://doi.org/10.1016/j.scico.2021.102620), [noise-resilient ES](https://www.cmap.polytechnique.fr/~nikolaus.hansen/proceedings/2019/GECCO/proceedings/proceedings_files/pap144s3-file1.pdf), [depth/fidelity](https://arxiv.org/html/2606.06555v1).
- **Essential baselines:** one-shot selection, fixed resampling, same-budget adaptive racing, fresh finalist confirmation for every optimizer, scalar penalty versus constrained selection with common objectives.
- **Expected null:** no candidate differences exceed noise at affordable repeat counts, or additional generations beat careful reevaluation.

#### 6. Adversarial or QD scene evolution followed by adapter/policy repair

- **Behavioral power:** scenario generation does not improve behavior until a learner uses the acquired data and improves on untouched tasks.
- **Evaluator:** native physical failures are factual; scenario validity, feasibility and label consistency become new failure modes. Confident-miss acquisition additionally depends on a useful warning predictor, which our current one lacks.
- **Cost:** scene/reset engineering, collection for all acquisition arms, learner updates and a common fresh audit. This is a separate multi-stage study.
- **Evolutionary rationale:** retain and mutate useful failure mechanisms rather than collecting redundant examples or impossible scenes.
- **Novelty risk:** QD scenario generation and evolutionary curricula already exist. The defensible question is safe-completion improvement per acquired physical interaction, not failure yield. [QD scenarios](https://arxiv.org/abs/2012.04283), [ACCEL](https://arxiv.org/abs/2203.01302), [PLR](https://proceedings.mlr.press/v139/jiang21b.html).
- **Essential baselines:** domain randomization, PLR, failure-only selection, random acquisition, fixed repair learner and identical updates/common seed data.
- **Expected null:** discovered scenes are invalid or impossible, repaired learner memorizes acquired scenes, or more failures do not improve fresh behavior.

#### 7. Inference-time evolutionary action search with a model or scoring function

- **Behavioral power:** potentially substantial if candidate actions can be scored accurately before execution.
- **Evaluator:** this is currently the weak link. A learned critic or dynamics model introduces extrapolation and optimizer exploitation; a geometry score can miss task completion and temporal hazards.
- **Cost:** multiple VLA/model queries each decision, and training/data costs for any learned evaluator. Cheap offline ranking does not make the full method cheap.
- **Evolutionary rationale:** online population search can combine and refine task-aligned proposals, as VLA-Pilot already does.
- **Novelty risk:** very high for a generic reranking or CEM wrapper. Our failed readout/dynamics evidence makes this a poor immediate primary route. [VLA-Pilot](https://arxiv.org/html/2511.14178v2), [local recovery record](../../history/20261019/phase3Recovery.md).
- **Essential baselines:** sample-and-rank, random proposals, non-evolutionary optimizer, same learned evaluator and query allowance, direct physical search where feasible.
- **Expected null:** the scorer ranks unsafe or inactive trajectories highly, rankings do not transfer to physical episodes, or gains vanish when total queries and model-training interactions are counted.

### Inferences

- Ranking for the next executable research bet: latent steering first, direct residual second, guarded programs third if a skill-composition diagnosis emerges, then repertoire/curriculum after competence is established. Evaluation-allocation work strengthens the first two routes rather than replacing their behavioral mechanism.
- A benchmark-wide average can hide success on only easy tasks. The paper should report each family and retain families with nulls. A pilot on Clutter alone can test task completion but cannot establish a broad safety contribution if its baseline has no violations.
- Avoid combining all seven routes into one proposal. Representation, optimizer and evaluator changes must remain identifiable, and every failed route keeps its costs and results.

### Gaps

- These rankings are not measured probabilities of publication success. No avenue can honestly be called a winner before a supported physical mechanism and matched optimization comparison exist.
- A four-feature proprioceptive controller may not identify safe semantic phases. More input is not free: image encoders, language grounding or privileged predicates introduce cost and information changes that need explicit controls.

## What exact primary/fallback experiment would decide whether to continue?

### Takeaway

The next useful experiment should first decide whether a compact noise controller exposes consequential behavior, then include actual evolutionary transitions and a matched random search. The one-GPU eight-hour session is a pilot with an independent check, not a final GECCO study. A paper-level ordinary-RL comparison and repeated optimizer runs need a separately costed programme.

### Cited Findings

- The active resource requirement includes machine setup, exploratory runs and evaluation in one GPU allocation of eight hours. Historical serial Clutter estimates in the plan are roughly three to seven minutes per episode, with no guarantee on a new machine. Its 48-episode arithmetic is 2.4 to 5.6 hours of rollout alone. [Active plan, Section 21](../../history/20261019/phase3.md).
- The currently completed pilot used a four-parameter execution adapter and one optimizer seed. Its final eight episodes per condition cannot establish optimizer superiority or low risk. The recovery plan protects independent final evaluation and explicitly requires random and ordinary-learning comparisons for corresponding claims. [Recovery plan](../../history/20261019/phase3Recovery.md).
- DSRL already supports reusable latent policies, while VLA-Pilot places population search at inference and Online-ES updates base-policy parameters. These are the closest comparison dimensions: optimized object, frozen components, training interactions, deployment queries and safety objective. [DSRL](https://arxiv.org/abs/2506.15799), [VLA-Pilot](https://arxiv.org/abs/2511.14178), [Online-ES](https://arxiv.org/abs/2609.38855).

### Inferences

#### Primary route: proposed claim and representation

**Claim to test:** a small reusable controller over a frozen VLA's generation noise can improve safe completion on untouched ManiGuard tasks at native execution settings; retaining and mutating useful controllers then improves adaptation per physical interaction relative to matched non-evolutionary search. Neither half is established.

One concrete starting representation, to be audited rather than treated as optimal:

```text
z_t ~ native policy Gaussian using the native episode/query RNG stream
phi_t = four fixed bounded causal features from permitted robot/history inputs
b_t = B tanh(W phi_t)
z'_t = z_t + alpha * repeat_over_chunk(b_t)
a_t = frozen_policy(obs_t, noise=z'_t)
```

Here `B` is a fixed four-column orthonormal basis in the **audited active noise channels**, padded with zero shifts in inactive channels; `W` has 16 coefficients. Choose the feature definitions, scaling and alpha domain from development-only information. Candidate features are a constant, gripper width, previous executed-target tracking error and previous proposal demand, all causally aligned and scaled with frozen development statistics. These features are a starting engineering hypothesis, not a detector of grasp, progress or safety. Keep horizon, observations, renderer and query cadence native. Do not add the old cap simultaneously in the first test.

At `W = 0`, identity requires the exact same generated native noise, RNG advancement, numerical path, preprocessing and output transforms. “Add zero” is not by itself proof of identity. Preserve padded noise and avoid extra random draws from the native stream. The augmentation may need an explicit wrapper extension; a seed replacement is a different intervention. Native library support for noise does not establish that the current websocket path forwards it correctly.

Reasons this is a better behavioral bet than sparse caps: it intervenes before action generation and can change relative joint movement and grasp approach, while the learned VLA still supplies vision and task context. Reasons it may fail: the selected subspace can be ineffective, different seeds can change modes discontinuously, and proprioception can be insufficient for conditional steering. A positive latent effect on saved inputs only proves proposal influence, not beneficial physical behavior.

#### Predictions and disconfirming observations

1. Zero steering reproduces the native executor on identical inputs and RNG; nonzero steering changes active action outputs beyond numerical tolerance without malformed values. Failure means fix the interface before interpreting any outcomes.
2. Nearby genotypes have more similar action/trajectory effects and often more similar task outcomes than independent random genotypes. No usable local structure weakens the rationale for mutation-based reuse. Do not call action-space smoothness outcome heritability.
3. A finalist selected without audit outcomes improves fresh physical safe completion over native and a constant-noise control. If only the constant helps, the result is a useful bias correction, not a conditional-controller contribution.
4. Evolution's final safe-completion-versus-cost curve improves over matched random search across independent search seeds. A one-seed lucky win is insufficient. If random matches it, retain the adaptation result and reject the claimed evolutionary advantage.
5. A supported confidence-aware selection rule reduces promotion of lucky unsafe candidates without spending so much budget that useful search depth disappears. If fixed resampling or more generations matches it, reject that algorithmic explanation.

#### Minimal pilot with actual evolution and a protected independent comparison

Before a paid allocation, prepare code and pin sources locally. On the run host, include all setup in the eight-hour clock. Audit the currently used renderer/checkpoint/native executor; a historical baseline under a different renderer is not the current control. Perform a bounded saved-observation sensitivity check and at most a few short physical identity/integration checks. These are engineering evidence, not completion episodes.

**First-family decision:** recommend **Jar for the scientific primary**, because the published native policy has both safe-completion headroom and physical safety obligations, whereas published Clutter has no measured violations and strong pi0.5 completion. The local Clutter result and the published rate belong to different recorded execution contexts and cannot be substituted for a contemporary control. Jar's longer native horizon and locally poor outcomes make it a higher-cost, higher-floor-risk experiment. Its policy must still have usable behavior; low safe completion alone does not establish repairable headroom. Prefer Clutter only as an explicitly narrower interface/optimizer feasibility pilot if measured Jar timing cannot protect a complete comparison inside eight hours. Decide that branch from source correctness and conservative timing before comparative search scores, not because one family happens to produce a favorable result. Do not interpret a Clutter-only gain as demonstrated repair of unsafe manipulation. [Benchmark family results](https://arxiv.org/html/2608.17386v1), [native horizon/resource record](../../history/20261019/phase3.md).

The numerical templates below use the existing Clutter throughput solely for costing. They are not promises for Jar. A Jar-first allocation should be frozen around measured native-horizon timing, potentially the separately proposed 44-episode form, with an independent audit protected. Do not squeeze an ordinary-learning arm into the pilot by eliminating the audit; that baseline remains necessary for the paper claim.

A **48-full-episode ceiling** is a concrete pilot option when timing supports it:

- 8 shared initialization episodes: four controllers, two preselected development cases each, with the native controller included as one anchor.
- 16 evolutionary descendant episodes: two further evaluated populations, four controllers, two cases each. Reevaluate retained parents/elites and count them.
- 16 matched random-search episodes: equal new evaluations in the same domain, using the common initialization data. A constant noise controller must be a declared shared reference, rather than retrospectively selecting a convenient offset.
- 8 independent final episodes: frozen native, constant reference, evolutionary finalist and random-search finalist, two common audit blocks each. This is an extremely noisy mechanism check, not powered superiority evidence.

This allocation makes the primary uncertainty explicit and includes real evolutionary updates. It does **not** satisfy the final ordinary-learning comparison. If the native/constant reference must occupy additional initialization slots, count that explicitly rather than silently changing the four candidates. No confirmation episodes are hidden in the arithmetic. Finalists selected from noisy search data will remain noisy; the eight independent episodes describe that limitation honestly.

A better minimum scientific pilot when throughput permits is **64 episodes**: the same 40 search episodes plus 24 audit episodes, six per final condition. It costs 3.2 to 7.5 hours of serial rollout at the historical three-to-seven-minute range, before setup, so it must not be promised inside eight hours without measured throughput. Selection between ceilings must use timing before comparative results, never the result sign. A slower host means smaller scope or an incomplete pilot, not shorter task horizons or an extended clock.

If a complete three-optimizer comparison is required in the first session, use the recovery plan's matched-optimizer design, costed with the new representation, or allocate a smaller explicitly exploratory comparison. Three undertrained arms plus nearly no independent evaluation cannot settle the research question merely because all baseline names appear.

#### Evolution and ordinary learning must be compared fairly

Use a simple established ES or genetic algorithm first. Do not spend the first pilot on inventing mutation rules. Log lineage, parameters, every attempted rollout, the four outcomes, queries and wall time. Common development cases and seeds reduce some variation but do not guarantee deterministic pairing.

For the first pilot, use a frozen transparent physical selection objective and preserve all raw outcomes. A separate feasibility claim needs sufficient repeats. Strict confidence bounds from two rollouts cannot credibly certify a low violation probability; an “uncertain” category is necessary if using bounds. Report safe completion, physical violations, task engagement and unfinished episodes together.

The paper-level baseline should include a state-conditioned latent-action SAC/DSRL-style actor with the same permitted observations and interaction endowment, and a direct-action residual PPO/SAC comparison. A Gaussian parameter-distribution REINFORCE baseline in the same 16-coefficient family is useful but much closer to vanilla ES than the phrase “ordinary RL” suggests. It does not settle the broader residual or latent-action learning alternative. Give all methods equal-quality implementation, comparable tuning allowance and the same confirmation/finalist protection. If a critic uses dense progress or privileged state, either match that information or label the separate advantage.

#### Paper-level replication and separately costed programme

Freeze a main-study protocol only after the affordance pilot. One transparent costing template is five independent optimizer seeds, three methods, two families, and 128 training episodes per method/run: **3,840 adaptation episodes**. Five runs per family are still a modest number of optimizer replicates. Add confirmation, tuning/development, infrastructure failures and every reference interaction separately.

For audit, three methods plus native and constant references on 20 new tasks per family with five repetitions each gives 100 episodes per condition/family. Native and constant references can be acquired once per balanced block; method finalists vary by optimizer run. For three methods times five optimizer runs, that is 1,500 finalist audit episodes per family, plus 200 reference episodes, or **3,400 final audit episodes over two families**. Total in this template is at least **7,240 full episodes**, before tuning, confirmation and engineering. At three to seven minutes each this is roughly **362 to 845 serial GPU-hours**, not an eight-hour experiment. Different native family horizons may raise the cost. This is a scale illustration, not an approved allocation or a power calculation.

Choose the real main-study size using observed variance, task clustering and the smallest scientifically meaningful effect. Independent scenes and optimizer runs are the replication units; frames are not independent safety trials. Report family-stratified and macro-averaged effects, confidence intervals, complete paired outcomes and actual cost curves. Freeze a held-out OOD shift only if claiming robustness. A transfer claim needs excluded tasks/families, not repeated seeds of exposed scenes. A benchmark-wide claim eventually needs all native families, including documented nulls. A strong restricted-family result should say exactly that.

#### Fallback route and claim

If the explicit-noise path is unavailable within its engineering allowance, or the saved-input/physical audit finds no useful latent influence, move to a compact **bounded joint-target residual**, not back to a cap-only optimizer rerun. Use an analogous small causal feature map and low-rank output, with exact zero identity and native queries. The bounded residual can change direction and address local tracking/contact behavior; it cannot manufacture missing semantic skills.

**Fallback claim to test:** a reusable local feedback residual improves physical safe completion around the released VLA, and evolutionary optimization of that residual gives a supported cost advantage over random and ordinary residual learning. The first clause may succeed without the second. The same pilot search/audit structure applies after a separately frozen physical scale audit; changing the interface creates a new study identity and consumes its own counted engineering/search budget. ResiP/ResFiT must be treated as central baselines, not mentioned only after a positive result.

Do not switch routes solely because an early result is negative, pool several attempted interfaces and report only the best, or present the fallback as a guaranteed rescue. A published null is stronger when it distinguishes insufficient behavioral power, absent skills and irreducible evaluation uncertainty.

### Gaps

- No numerical improvement, mutation smoothness, optimum alpha range or safe violation threshold can be responsibly predicted from the available evidence.
- The proposed 16-coefficient design is one concrete testable starting point. It is neither proved optimal nor established as distinct from all prior low-rank latent controllers. A more extensive search is needed before a new-representation priority claim.
- The 48/64-episode options are narrower than the already proposed full matched comparison. They must not silently supersede the active plan or launch authority; the user asked for research direction, not new rental or simulator work.
- A final GECCO contribution remains contingent on measured results and a defensible comparison, even if the representation, implementation and research question are promising.
