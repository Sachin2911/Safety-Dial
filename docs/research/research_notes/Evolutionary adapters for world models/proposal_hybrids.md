# Reusable proposal adapters and population planning

Search cutoff: 30 October 2026. This note covers proposal/planner hybrids; another research strand covers classic evolved world-model controllers. It records literature evidence only, with no experimental launch.

## Which successful policy and planning hybrids most closely match the proposal?

### Takeaway

POPLIN is the closest architectural predecessor found: a policy proposes actions and population search improves them, with an additional parameter-search variant. Safe-LOOP is the strongest relevant safety extension. TD-MPC2 provides a widely evaluated latent policy-prior architecture. Their training arrangements differ from a small evolutionary adapter around frozen LeWM. [POPLIN](https://arxiv.org/html/1906.08649), [LOOP](https://arxiv.org/html/2008.10066), [TD-MPC2](https://arxiv.org/html/2310.16828).

### Cited Findings

- **POPLIN, Wang and Ba: arXiv June 2019, ICLR 2020.** POPLIN-A initializes action sequences from a policy and uses CEM to optimize added action noise. POPLIN-P instead searches policy-parameter perturbations. Its persistent policy uses behaviour cloning, GAN training, or an AVG update that adds the mean optimized parameter perturbation. Models are updated during training, rather than remaining a frozen pretrained foundation. Table 1 reports Cheetah return 12,227.9 +/- 5,652.8 for POPLIN-P versus PETS 4,204.5 +/- 789.0 after 50,000 environment steps: four seeds, a final 3,000-step window. This aggregate POPLIN-P result does not isolate the AVG variant or prove safety. Section 5.2 says AVG works as a planning initialization but sacrifices standalone policy performance. Appendix A.6 finds parameter-space planning can work even with fixed zero base weights, so successful planning does not by itself establish learned-proposal benefit. [Full paper, sections 4-5 and appendix A.6](https://arxiv.org/html/1906.08649); [official ICLR PDF confirming venue](https://openreview.net/pdf/2175c57ed693836c3ce3e6081e69cbd0e8d3df37.pdf).
- **POPLIN implementation.** The author repository exposes action/parameter-space search variants and BC, AVG and GAN training schemes. It recommends small policy networks, including no hidden layer, one 32-unit layer or one 64-unit layer. This is unusually close to our compact-controller intent. [Author code and configuration documentation](https://github.com/WilsonWangTHU/POPLIN).
- **LOOP / Safe-LOOP, Sikchi, Zhou and Held: CoRL 2021, PMLR proceedings 2022.** Actor Regularized Control uses a prior derived from an actor and the previous shifted plan; learned dynamics evaluate candidate trajectories, and an off-policy value function extends the horizon. Actor training is gradient-based, not neuroevolution. Safe-LOOP adds cost constraints, considers worst-case predicted costs over model ensembles and particles, and evaluates two Safety Gym navigation environments plus a simulated RC car. The paper reports improved reward/sample efficiency and competitive safety, averaged across five seeds; it does not report zero violations as a guarantee. Its safe experiments initialize models and policies randomly and learn them, rather than demonstrating a new safety adapter around frozen LeWM. The paper also diagnoses actor/planner divergence. [Full methods, sections 5-6, appendix B.3.1 and C](https://arxiv.org/html/2008.10066); [proceedings and bibliographic date](https://proceedings.mlr.press/v164/sikchi22a.html).
- **LOOP code.** The author repository includes online, offline and safe training entry points, ARC controllers and safety environment definitions. [Author repository](https://github.com/hari-sikchi/LOOP).
- **TD-MPC2, Hansen, Su and Wang: arXiv October 2023, ICLR 2024.** An implicit latent world model, reward/value estimates and policy prior are learned from replay. MPPI planning includes policy-prior action sequences and shifted previous solutions. Default settings allocate 24 of 512 candidate trajectories to the policy prior, with horizon three and six iterations. Evaluation covers 104 tasks across four domains, with three seeds and 95% intervals; a 317M-parameter multitask agent handles 80 tasks. These are reward/success results, not introduced-hazard safety results. The policy prior is optimized through Q-based gradients; the model is updated during learning. [Paper, sections 3-4 and Table 8](https://arxiv.org/html/2310.16828); [official project and models](https://www.tdmpc2.com/).
- **TD-MPC2 implementation verification.** `_plan` rolls policy-generated actions through the latent model and inserts them into the candidate population. `update_pi` uses Q estimates; `_update` trains model parameters and invokes the policy update. [Official implementation](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py).

### Inferences

- POPLIN is the first paper to read for the proposal-adapter mechanism. Safe-LOOP is the next for constraints and actor/planner mismatch. TD-MPC2 is useful engineering precedent for mixing learned proposals with search. This ordering follows architectural relevance, not a claim that any method will improve our pipeline.
- Preserve ordinary/random proposals alongside adapter proposals. This would allow the planner to recover from a poor adapter and would expose whether the learned component actually supplies useful candidates. POPLIN's zero-base-weight result makes a random/untrained-adapter control particularly informative. [POPLIN appendix A.6](https://arxiv.org/html/1906.08649).

### Gaps

- The cited aggregate returns do not establish benefits from an offline ES-trained hazard-conditioned adapter with a fixed world model and independently learned geometry checker.
- The POPLIN numerical example is verified in the linked arXiv HTML Table 1. The official indexed PDF confirms ICLR 2020, but direct OpenReview PDF/forum opens returned browser verification during this search. Do not silently label that number as an independently checked final-conference-version result, or as an AVG-only result.
- Safe-LOOP's plotted safety comparisons provide no directly interchangeable safe-completion count for our task definition. Do not translate reward/cost curves into our strict safe-completion rate.

## Has the exact evolutionary-trained adapter already been demonstrated?

### Takeaway

The components have successful precedents, but this focused search did not verify an exact match to the complete proposed design. That is a search limitation, not evidence of novelty.

### Cited Findings

- POPLIN-P has derivative-free parameter search and a persistent AVG update, making it closer to evolution than a conventional gradient-trained policy prior. However, online parameter search and distillation are different from evolving a reusable controller across whole development episodes and deploying frozen weights with a separate safety checker. [POPLIN sections 4.2-4.4](https://arxiv.org/html/1906.08649).
- **CEM-RL, Pourchot and Sigaud: arXiv 2018, ICLR 2019.** This genuinely combines evolutionary search of actor parameters with TD3/DDPG gradient steps. It is model-free policy search, not a proposal adapter around a learned world model. Table 1 gives CEM-TD3 HalfCheetah-v2 return 10,725 versus multi-actor TD3 9,662; ten runs of one million environment steps. It supports hybrid evolutionary policy optimization as a working mechanism, but does not validate frozen-model planning or hazard safety. [Primary paper](https://arxiv.org/html/1810.01222); [author code](https://github.com/apourchot/CEM-RL).

### Inferences

- A LeWM study can legitimately combine the reusable small-controller idea with proposal-guided planning and fixed safety checking. The contribution would have to be established by its particular safety problem, representation, optimizer comparison and evidence, rather than by claiming that policy-guided planning itself is new.
- Strong comparisons should separate the advantage of having any proposal network from the advantage of training it with evolution. POPLIN supplies the first concern; CEM-RL shows that evolutionary and gradient updates can be meaningfully compared or combined. [POPLIN](https://arxiv.org/html/1906.08649), [CEM-RL](https://arxiv.org/html/1810.01222).

### Gaps

- No verified primary paper in this search simultaneously establishes: frozen pretrained visual world model, small ES-trained hazard-conditioned proposal/residual adapter, separate learned motion-and-geometry checker, and improved safe completion on fresh hazard cases.
- Recent searches included variants of evolutionary policy priors, CMA-ES warm starts, frozen-world-model controllers and evolutionary residual policies. Some surfaced 2027-dated racing work or unrelated language-agent evolution; these were not treated as evidence available by the cutoff.

## What should the LeWM design take from these precedents?

### Takeaway

The literature supports trying an adapter that proposes behaviour, while retaining search and an explicit safety check. It does not support assuming that a successful planner implies successful adapter learning. [POPLIN](https://arxiv.org/html/1906.08649), [LOOP](https://arxiv.org/html/2008.10066).

### Cited Findings

- In Safe-LOOP's safeARC, too few feasible trajectories trigger distribution fitting toward lower predicted cost; otherwise feasible candidates are weighted by predicted reward. This resembles feasibility-first search but is not a proof that every executed prefix is feasible. [Algorithm 4](https://arxiv.org/html/2008.10066).
- POPLIN-A's replanning variant can diverge, while LOOP identifies distribution mismatch between actor and planner. Their results motivate checking proposal quality and realized outcomes independently. [POPLIN section 5.4](https://arxiv.org/html/1906.08649), [LOOP section 5.1 and appendix D.5](https://arxiv.org/html/2008.10066).

### Inferences

- The clearest proposed adapter outputs an initial action chunk, or a bounded residual around a nominal chunk, using current causal observations, goal features and explicitly supplied hazard information. It need not modify LeWM weights.
- Test the planner alone, planner with a random/untrained adapter, planner with the evolved adapter, and the same adapter trained by ordinary learning under matched information and budgets. Keep optimizer repetitions and fresh source cases, not only repeated seeds on familiar layouts.
- Keep the existing execution-time recheck and refusal rule. Report predicted feasibility separately from observed safety. Safety margins and prediction allowances should not be evolvable shortcuts.
- Charge offline adapter development/training and online model predictions separately, then report both and their total. A smaller online search budget alone is not evidence of a lower total acquisition cost.

### Gaps

- These sources do not establish the right adapter size, mutation scale, action basis, horizon or training budget for LeWM. They also do not show that our current unfinished episodes are caused by poor proposal initialization.
