# Quality-Diversity for RL: safety, constraints and learned models (research notes, 3 Oct 2026)

Labels used throughout: **[PR]** = peer-reviewed venue confirmed from the paper page, proceedings or arXiv comment; **[Preprint]** = arXiv only, no confirmed venue; **[Venue unverified]** = a venue is commonly attributed but I did not confirm it in this session. Dates are first arXiv submission or conference date.

Verification caveat: one WebFetch summary was unreliable. It described RSS 2024 paper p104 as "QD with safety constraints". When I read the PDF myself, it turned out to be *POLICEd RL* (Bouvier, Nagpal, Mehr), a hard-constraint RL method with **no QD content**. Treat any tool summary as unconfirmed until it is checked against the primary text. Every claim below comes from an abstract or full text I fetched, unless it is flagged.

---

## Key Question 1: The state of QD-RL (2019–2026), which methods work on locomotion with one GPU, and how many evaluations they need

### Takeaway
QD-RL went from pure evolution (MAP-Elites, then ME-ES) to hybrids that add policy gradients. These hybrids use either off-policy TD3 critics (PGA-ME, QD-PG, DCG-ME/DCRL-ME, ASCII-ME, QDAC) or on-policy PPO inside a DQD/CMA framework (CMA-MEGA in RL, then PPGA). On Walker-style locomotion the community-standard budget is about **1 million episode evaluations per run**, and results are reported with re-evaluated ("corrected") archives. JAX/Brax implementations (QDax) bring this down to minutes or hours on one GPU, but only for Brax or MJX environments, not for CPU MuJoCo Safety-Gymnasium.

### Cited Findings
**Foundations**
- **MAP-Elites**, Mouret & Clune, "Illuminating search spaces by mapping elites" (arXiv 1504.04909, Apr 2015) [Preprint; widely cited foundational work]. It discretises a user-chosen descriptor space into cells and keeps the highest-fitness solution ("elite") per cell. Source: [arXiv](https://arxiv.org/abs/1504.04909)
- **QD as a field**: Pugh, Soros & Stanley, "Quality Diversity: A New Frontier for Evolutionary Computation", *Frontiers in Robotics and AI*, 12 Jul 2016 [PR]. It frames QD as filling a space of behaviours with the best example of each, and covers NSLC and MAP-Elites. Source: [DOAJ/Frontiers](https://doaj.org/article/8292416d6504484db8032e02c4288cd9)
- **ME-ES**, Colas, Madhavan, Huizinga & Clune, "Scaling MAP-Elites to Deep Neuroevolution" (arXiv 2003.01825; GECCO 2020) [PR]. It uses Evolution Strategies as the variation operator so that MAP-Elites scales to large NN controllers. It was designed for post-damage recovery in a high-dimensional control task where plain MAP-Elites fails. Source: [arXiv](https://arxiv.org/abs/2003.01825)
- **PGA-ME**, Nilsson & Cully, "Policy Gradient Assisted MAP-Elites" (GECCO 2021, Neuroevolution best-paper nominee) [PR]. It pairs the GA operator with a TD3-style policy-gradient operator that pushes offspring towards higher fitness. Source: [Imperial news](https://imperial.ac.uk/news/225633/three-papers-accepted-gecco-2021)
- **QD-PG**, Pierrot et al., "Diversity Policy Gradient for Sample Efficient Quality-Diversity Optimization" (arXiv 2006.08505; GECCO 2022) [PR]. It adds a time-step-level *diversity* policy gradient alongside a quality gradient and selects from a MAP-Elites grid. Source: [arXiv](https://arxiv.org/abs/2006.08505)

**Differentiable QD / CMA family (USC, Nikolaidis lab)**
- **DQD / CMA-MEGA**, Fontaine & Nikolaidis, "Differentiable Quality Diversity", NeurIPS 2021 (oral) [PR]. It defines DQD, where the objective and measures are differentiable, and proposes MEGA/CMA-MEGA, which use objective and measure gradients to branch through descriptor space. Source: [NeurIPS](https://proceedings.nips.cc/paper_files/paper/2021/file/532923f11ac97d3e7cb0130315b067dc-Paper.pdf)
- **CMA-MEGA in RL** (supervisor's paper): Tjanaka, Fontaine, Togelius & Nikolaidis, "Approximating Gradients for Differentiable Quality Diversity in Reinforcement Learning", GECCO 2022 [PR]. It approximates the objective gradient with a TD3 critic or OpenAI-ES, and the measure gradients with ES.
  - Tasks are four QDGym locomotion tasks: QD Ant (4 measures, 10,000 cells), Half-Cheetah (2 measures, 1,000 cells), Hopper (1 measure, 100 cells) and **Walker (2 foot-contact measures, 1,000 cells)**.
  - **Each algorithm evaluates 1 million solutions.** Each run used 100 CPUs plus one P100 GPU for TD3 and took **4–20 hours**.
  - CMA-MEGA (TD3, ES) ≈ PGA-ME on all tasks, including QD Walker, and both beat ME-ES and MAP-Elites. CMA-MEGA (ES) underperformed on Hopper and Walker.
  - The authors attribute part of the gap to evaluations spent on estimating measure gradients, and note that TD3's objective gradient was more useful than OpenAI-ES's on Hopper and Walker.
  Source: [arXiv abs](https://arxiv.org/abs/2202.03666); [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2202.03666)
- **CMA-MAE**, Fontaine & Nikolaidis, "Covariance Matrix Adaptation MAP-Annealing", GECCO 2023 (DOI 10.1145/3583131.3590389) [PR]. It fixes three CMA-ME weaknesses: premature abandonment of the objective, flat objectives, and low-resolution archives. It does this by annealing an acceptance threshold with a learning rate α, which interpolates between CMA-ES (α=0) and CMA-ME (α=1). It is invariant to archive resolution. Source: [arXiv](https://arxiv.org/abs/2205.10752v3)
- "Covariance Matrix Adaptation MAP-Annealing: Theory and Experiments" (Zhao, Tjanaka, Fontaine, Nikolaidis, 2025) is listed on the QD papers list [Venue unverified]. Source: [QD papers list](https://quality-diversity.github.io/papers.html)
- **PPGA**, Batra, Tjanaka, Fontaine, Petrenko, Nikolaidis & Sukhatme, "Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning", ICLR 2024 (spotlight) [PR].
  - **Method:** first adaptation of on-policy PPO to the DQD framework. Vectorised PPO (VPPO) computes objective and measure gradients together, Markovian measure proxies are introduced, and CMA-ES inside CMA-MAEGA is replaced with xNES.
  - **Setup:** Brax Ant, Walker2d, Half-Cheetah and Humanoid, with foot-contact measures. Each iteration evaluates 300 policies (about 3M timesteps). Hardware was RTX 2080Ti GPUs. Implemented with pyribs and Brax, with VPPO based on CleanRL.
  - **Results:** reports a 4x best-reward gain on Humanoid. The extracted Walker2d table values are QD-score 1.06×10⁵, coverage 0.39 and best reward 4702 (check against the paper's table before quoting).
  - **Limitations (authors' own):** PPGA is "less sample-efficient than other off-policy QD-RL methods". Archives are re-evaluated 50× per agent to build a "Corrected Archive". Scaling runs went to 1.2M evaluations.
  Source: [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/99b1a49a4e889f6965f221660d32cac9-Abstract-Conference.html); [arXiv html](https://arxiv.org/html/2305.13795v2); [ar5iv](https://ar5iv.labs.arxiv.org/html/2305.13795)

**Descriptor-conditioned / actor-critic family (Imperial AIRL, InstaDeep)**
- **DCG-ME**, Faldor, Chalumeau, Flageat & Cully, "MAP-Elites with Descriptor-Conditioned Gradients and Archive Distillation into a Single Policy", GECCO 2023, pp. 138–146 [PR]. A descriptor-conditioned critic gives gradients that improve fitness while targeting a given descriptor. Training also yields a single descriptor-conditioned policy that distils the archive. **+82% QD-score over PGA-ME on average.** Source: [arXiv](https://arxiv.org/pdf/2303.03832)
- **DCRL-ME** (extended DCG-ME with actor injection), Faldor, Chalumeau, Flageat & Cully, "Synergizing Quality-Diversity with Descriptor-Conditioned Reinforcement Learning", ACM TELO (DOI 10.1145/3696426) [PR]. It covers 7 tasks (Ant/AntTrap/Humanoid Omni, and **Walker Uni**, HalfCheetah Uni, Ant Uni, Humanoid Uni) with **1 million evaluations per run and 20 seeds**. On Walker Uni its median QD-score is higher than PGA-ME's but not significantly. Source: [ACM DL](https://dl.acm.org/doi/10.1145/3696426); [arXiv](https://arxiv.org/html/2401.08632v1)
- **QDAC**, Grillotti, Faldor, León & Cully, "Quality-Diversity Actor-Critic: Learning High-Performing and Diverse Behaviors via Value and Successor Features Critics", ICML 2024 [PR]. It is a single skill-conditioned actor with a value critic and a successor-features critic. The actor objective uses **constrained optimisation (Lagrangian)** to maximise return *subject to* executing the commanded skill. It beats other QD methods on six locomotion tasks and adapts better to perturbed environments. Source: [arXiv](https://arxiv.org/abs/2403.09930)
- **URSA**, Grillotti, Coiffard, Pang, Faldor & Cully, "From Tabula Rasa to Emergent Abilities: Discovering Robot Skills via Real-World Unsupervised Quality-Diversity", CoRL 2025 [PR]. It extends QDAC to real-hardware skill discovery, and the authors cite "safety and data efficiency constraints" as the motivation. It beats baselines in 5 of 9 simulated and 3 of 5 real damage scenarios. Source: [arXiv](https://arxiv.org/abs/2508.19172)
- **ASCII-ME**, Mitsides, Faldor & Cully, "Scaling Policy Gradient Quality-Diversity with Massive Parallelization via Behavioral Variations" (arXiv 2501.18723, 30 Jan 2025) [Preprint]. It is a policy-gradient QD method with no centralised actor-critic. It claims diverse high-performing policies in **under 250 s on a single GPU**, and is **5x faster** than the state of the art at competitive sample efficiency. Source: [arXiv](https://arxiv.org/abs/2501.18723)

**Other 2024–2026 advances**
- **CCQD**, Xue, Wang, Li, Li, Hao & Qian, "Sample-Efficient Quality-Diversity by Cooperative Coevolution", ICLR 2024 [PR]. It splits the policy network into representation and decision layers and coevolves the two subpopulations. It reports about **200% sample-efficiency improvement** on QDax tasks. Source: [ICLR](https://iclr.cc/virtual/2024/poster/18945)
- **Dominated Novelty Search**, Bahlous-Boldi, Faldor, Grillotti, Janmohamed, Coiffard, Spector & Cully, GECCO 2025, pp. 104–112 [PR]. It implements local competition through dynamic fitness transformation and needs no predefined grid bounds. Source: [arXiv](https://arxiv.org/pdf/2502.00593)
- **Soft QD / SQUAD**, Hedayatian & Nikolaidis, "Soft Quality-Diversity Optimization" (arXiv 2512.00810, Nov 2025; accepted at ICLR 2026 per arXiv comment) [PR per comment]. It is a discretisation-free, differentiable QD formulation whose limiting behaviour connects to QD-score. Source: [arXiv](https://arxiv.org/abs/2512.00810)
- **BFM-QD**, Bendib, Perrin-Gilbert & Sigaud, "Behavioral Foundation Models for Quality Diversity" (arXiv 2609.35615, 28 Sep 2026; "accepted at NeurIPS 2026" per arXiv comment) [PR per comment; camera-ready not checked]. It runs QD in the latent z-space of a behavioural foundation model instead of over policy weights. It uses a closed-form, gradient-free improvement operator with no critic training. It outperforms parameter-space baselines on locomotion, sparse navigation and manipulation. Source: [arXiv](https://arxiv.org/abs/2609.35615)
- **SV-QD-RL**, Zuo, Xu, Liu & Luo, "Structure-Conditioned Actor-Critic Branches for Quality-Diversity Reinforcement Learning" (arXiv 2606.08735, 7 Jun 2026) [Preprint]. It represents each candidate as an actor-critic branch with a structural mask, and was tested on MuJoCo. Source: [arXiv](https://arxiv.org/abs/2606.08735)
- **Critique / position paper**: Batra, Tjanaka, Nikolaidis & Sukhatme, "Quality Diversity for Robot Learning: Limitations and Future Directions", GECCO 2024 Companion [PR, short]. They argue that bounded MAP-Elites-style tasks, such as reaching different xy positions, can be solved by **one goal-conditioned policy plus a classical planner**, with O(1) space in the number of policies. Source: [arXiv](https://arxiv.org/abs/2407.17515v1)

**Libraries and single-GPU practicality**
- **QDax**, Chalumeau et al., JMLR 25(108), 2024 [PR]. It is a JAX library, fully JIT-compilable on GPU/TPU, with MAP-Elites, PGA-ME, DCG-ME, MOME, ME-ES and others, plus Brax tasks. Source: [JMLR](https://jmlr.org/papers/v25/23-1027.html)
- **pyribs**, Tjanaka et al., "pyribs: A Bare-Bones Python Library for Quality Diversity Optimization", GECCO 2023, pp. 220–229 [PR]. It has a modular archive/emitter/scheduler API with CMA-ME, CMA-MAE and CMA-MEGA, and is the base for PPGA. Source: [arXiv](https://arxiv.org/pdf/2303.00191)
- **EvoJAX**, Tang et al., "EvoJAX: Hardware-Accelerated Neuroevolution" (arXiv 2202.05008; presented at GECCO 2022) [PR, companion/demo per search snippet]. Source: [arXiv](https://arxiv.org/abs/2202.05008v1)
- **evosax**, Lange, "evosax: JAX-based Evolution Strategies" (arXiv 2212.04180, Dec 2022). It offers about 30 ES/GA algorithms behind an ask-evaluate-tell API. The search results did not clearly confirm a GECCO 2023 companion version [Venue unverified]. Source: [arXiv](https://arxiv.org/pdf/2212.04180v1)
- **CRAX**, Tomilin, Boustani, Beurskens, Nguyen & Simão, "CRAX: Fast Safe Reinforcement Learning Benchmarking" (arXiv 2606.20376, Jun 2026, v4 29 Sep 2026) [Preprint]. It is a safe-RL benchmark on MuJoCo XLA (MJX) that runs entirely on GPU via JAX. The abstract claims about 200x faster training than CPU-based safety benchmarks (one search snippet said "~100x"). It has 8 tasks at 3 difficulty levels, and of the 7 safe-RL methods tested "none dominates". Source: [arXiv](https://arxiv.org/abs/2606.20376)

### Inferences
- **Budget to plan for.** The literature norm for Walker-like QD-RL is about 1M episode evaluations (Tjanaka 2022, DCRL-ME, MOME). With 1,000-step episodes that is up to 10⁹ environment steps, although early termination on falls reduces it. That is routine in Brax/MJX on one GPU but heavy on CPU MuJoCo (Safety-Gymnasium). A realistic Honours plan is either:
  - a much smaller archive (e.g., a 1-D cost axis with 10–20 bins) and a reduced budget (10⁴–10⁵ evaluations), or
  - porting the task to an MJX/JAX environment such as CRAX or a Brax Walker with a velocity-cost wrapper.
- **Which family fits the student's stack.** The student already has PPO and PPO-Lagrangian, so PPGA (PPO inside CMA-MAEGA, built on pyribs) is the closest fit. Its authors admit it is the least sample-efficient family, though. Off-policy hybrids (PGA-ME, DCRL-ME) are the sample-efficient options in QDax.
- **Mapping CMA-MEGA to a safety dial.** CMA-MEGA's "measure gradient" is just ∇ of expected cost. A PPO-Lagrangian cost critic already estimates that quantity, so a "safety-indexed" CMA-MEGA/PPGA needs no new gradient estimator. The cost value head plays the role of the measure-gradient estimator. This is my own inference; I found no paper that does it.

### Gaps
- I did not find a published per-run total of environment steps or wall-clock time for PPGA on Walker2d. Only "300 policies, about 3M timesteps per iteration" and "1.2M evaluations" in the scaling runs were extractable.
- I did not confirm CRAX's exact task list, i.e., whether it includes a Walker2d velocity-cost task equivalent to SafetyWalker2dVelocity-v1.
- I did not fetch the original PGA-ME paper's own budget. It was cited here through the GECCO 2021 news page and later benchmarks.

---

## Key Question 2: QD with constraints or safety, red-teaming and robustness; has anyone built a reward-vs-safety archive for safe RL?

### Takeaway
There is a small, scattered literature on constrained QD:
- feasible/infeasible populations per cell (Constrained MAP-Elites);
- constraint violations used *as* descriptors on numerical benchmarks;
- Bayesian constrained QD for engineering design;
- **one** model-based safe-RL method (GuSS) whose QD planner prefers low-cost elites.

QD is used heavily for *finding* failures (red-teaming, scenario generation) and for damage recovery and robustness. **I found no paper that builds a MAP-Elites/CMA-ME/PPGA archive over a reward-versus-expected-cost (or violation-probability) axis for safe RL policies, nor one on Safety-Gym or Safety-Gymnasium tasks.** The closest non-QD analogue is CCPO, a single policy conditioned on the constraint threshold (NeurIPS 2023).

### Cited Findings
**Constraints inside QD**
- **Constrained MAP-Elites**, Khalifa et al., "Talakat: Bullet Hell Generation through Constrained Map-Elites" (arXiv 1806.04718, 2018) [GECCO 2018 venue unverified this session]. Each MAP-Elites cell holds **two populations, feasible and infeasible**, following FI-2Pop. The feasible population maximises fitness and the infeasible one minimises constraint violation, and individuals migrate between them as they gain or lose feasibility. Source: [arXiv](https://arxiv.org/pdf/1806.04718); description also in [Interactive Constrained MAP-Elites, arXiv 2003.03377](https://arxiv.org/pdf/2003.03377)
- **Constraint violations as descriptors**, Fioravanzo & Iacca, "Evaluating MAP-Elites on Constrained Optimization Problems" (arXiv 1902.00703, Feb/Apr 2019) [Preprint; a later book-chapter version appears in the Univ. Trento repository, venue unverified].
  - **Method:** each constraint becomes a separate descriptor dimension, so the descriptor is a vector of violations. This "illuminates" how violations correlate with the objective.
  - **Findings:** plain MAP-Elites underperforms specialised constrained optimisers, especially on equality constraints. It is good at revealing **constraint-violation-vs-objective trade-offs**.
  Source: [arXiv](https://arxiv.org/abs/1902.00703)
- **Bayesian constrained QD**: Brevault & Balesdent, "Bayesian Quality-Diversity approaches for constrained optimization problems with mixed continuous, discrete and categorical variables", *Engineering Applications of AI* (ScienceDirect, 2024; listed as 2023 on the QD list) [PR]. A follow-on is "Bayesian Quality-Diversity optimization for conditional search-space problems" (Baraton et al., 2025) [Venue unverified]. Source: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197624002768); [QD papers list](https://quality-diversity.github.io/papers.html)
- **QDAC** (ICML 2024, above) uses a Lagrangian. The constraint is skill execution, not safety. It still shows that Lagrangian machinery and QD-RL already coexist in one actor objective. Source: [arXiv](https://arxiv.org/abs/2403.09930)

**QD + safety in RL (the closest prior work)**
- **GuSS**, Paolo, Gonzalez-Billandon, Thomas & Kégl, "Guided Safe Shooting: model based reinforcement learning with safety constraints" (arXiv 2206.09743, v1 20 Jun 2022, v2 12 Sep 2024) [Preprint; no venue on arXiv].
  - **Approach:** model-based RL in which a **safety-aware MAP-Elites runs as the MPC planner inside a learned dynamics model** at each real time step. Cost is an indicator of being in an unsafe state, and the mean cost per trajectory is used as an estimate of P(unsafe). It claims to be "the first MBRL approach that combines QD and safety objectives in a principled way."
  - **Safety-aware STORE rule:** within a descriptor cell, the incumbent is replaced if the newcomer has **lower cost**, or equal cost and higher reward. Note this is lexicographic: cost first, reward second.
  - **SELECT and action choice:** SELECT only mutates zero-cost elites and pads with random policies. The executed action comes from the highest-reward policy among the lowest-cost elites.
  - **Environments:** Safe Pendulum, Safe Acrobot and **SafeCar-Goal (Safety Gym)**.
  - **Baselines and metric:** RS, Safe-RS, CEM, RCEM, Safe-MBPO, CPO and **PPO-Lagrangian**. It reports p(unsafe)%.
  - **Result:** GuSS is the only safe-MBRL method that solves SafeCar-Goal.
  - **Relevance to a safety dial:** the archive here is a throw-away planning structure. It is not a deliverable spectrum of policies indexed by safety level, and cost is a selection criterion, not a descriptor axis.
  Source: [arXiv abs](https://arxiv.org/abs/2206.09743); [PDF](https://arxiv.org/pdf/2206.09743)
- **Reset-free / physical-robot QD with safety filtering**:
  - Lim, Reichenbach & Cully, "Learning to Walk Autonomously via Reset-Free Quality-Diversity" (arXiv 2204.03655, Apr 2022) [Preprint per arXiv page]. It builds on DA-QD to learn under "harsh safety constraints" within training zones with obstacles, choosing behaviours from the imagined repertoire that act as automatic resets. Source: [arXiv](https://arxiv.org/abs/2204.03655)
  - Smith, Lim, Janmohamed & Cully, "Quality-Diversity Optimisation on a Physical Robot Through Dynamics-Aware and Reset-Free Learning", GECCO 2023 Companion [PR]. It has three parts: a learned dynamics model, **behaviour filtering that excludes unsafe or uninteresting predicted policies**, and a recovery mechanism. A quadruped learned diverse skills on hardware in 2 hours. Source: [arXiv](https://arxiv.org/abs/2304.12080)
- **Risk/reliability as a user-chosen trade-off** (closest conceptual analogue to a "dial"): Flageat, Janmohamed, Lim & Cully, "Exploring the Performance-Reproducibility Trade-off in Quality-Diversity" (arXiv 2409.13315, Sep 2024, rev. Mar 2025) [Preprint per arXiv page].
  - **Motivating example:** a solution may reliably walk at 90% of maximum velocity, while faster solutions fall over more often.
  - **Methods:** **four a-priori algorithms** that optimise for a given user preference over the trade-off, and **one a-posteriori algorithm** for when no preference is known.
  Source: [arXiv](https://arxiv.org/abs/2409.13315)
- **ARIA**, Grillotti, Flageat, Lim & Cully, "Don't Bet on Luck Alone: Enhancing Behavioral Reproducibility of QD Solutions in Uncertain Domains" (arXiv 2304.03672, Apr 2023) [Preprint per arXiv page]. It is a plug-in NES module that raises the probability that a solution stays in its niche, and improves archive quality and coverage by ≥50%. Source: [arXiv](https://arxiv.org/abs/2304.03672)

**Non-QD "dial" baselines a reviewer will ask about**
- **CCPO**, Yao, Liu, Cen, Zhu, Yu, Zhang & Zhao, "Constraint-Conditioned Policy Optimization for Versatile Safe Reinforcement Learning", NeurIPS 2023 [PR]. It trains **one policy conditioned on the cost threshold**, using Versatile Value Estimation and Conditioned Variational Inference, for **zero-shot adaptation to unseen thresholds**. Source: [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/29906cbd165b78991da2c4dbabc2a04b-Abstract.html)
- Zahavy et al., "Discovering Diverse Nearly Optimal Policies with Successor Features" (arXiv 2106.00669, 2021) [Venue unverified]. It formulates diversity discovery as a CMDP: maximise diversity subject to a near-optimality constraint. Source: [arXiv](https://arxiv.org/html/2106.00669)

**QD for safety testing and red-teaming (finding failure modes)**
- Fontaine & Nikolaidis, "A Quality Diversity Approach to Automatically Generating Human-Robot Interaction Scenarios in Shared Autonomy", RSS 2021 [PR]. MAP-Elites finds diverse scenarios that minimise the tested algorithm's performance, beating Monte Carlo and CMA-ES. An extended version appeared in ACM THRI ("Evaluating Human–Robot Interaction Algorithms in Shared Autonomy via QD Scenario Generation"). Source: [RSS PDF](https://www.roboticsproceedings.org/rss17/p036.pdf); [ACM THRI](https://dl.acm.org/doi/fullHtml/10.1145/3476412)
- Esquerre-Pourtère, Kim & Park, "Generating Diverse Challenging Terrains for Legged Robots Using Quality-Diversity Algorithm", ICRA 2025 (arXiv 2506.01362) [PR]. It produces an archive of terrains that challenge biped and quadruped controllers in different ways and expose different failure modes, and the terrains can be used to improve RL controllers. Source: [arXiv](https://arxiv.org/abs/2506.01362)
- **MADRID**, Samvelyan, Paglieri, Jiang, Parker-Holder & Rocktäschel, "Multi-Agent Diagnostics for Robustness via Illuminated Diversity", AAMAS 2024 [PR]. It is a MAP-Elites-style search for adversarial settings, scored by the target policy's regret, and it exposed weaknesses in TiZero on 11v11 Google Research Football. Source: [arXiv](https://arxiv.org/pdf/2401.13460)
- **Rainbow Teaming**, Samvelyan et al., NeurIPS 2024 [PR]. It runs MAP-Elites over adversarial LLM prompts with risk-category × attack-style descriptors, reaches >90% attack success, and fine-tuning on the results improves safety. Source: [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8147a43d030b43a01020774ae1d3e3bb-Abstract.html)
- **CaDRE** (arXiv 2403.13208, 2024): controllable, diverse safety-critical driving scenarios using CMA-ME with an "Occupancy-Aware Restart" [Venue unverified]. Source: [arXiv](https://arxiv.org/html/2403.13208v2)

**QD for robustness and damage recovery**
- Cully, Clune, Tarapore & Mouret, "Robots that can adapt like animals" (arXiv 1407.3501; *Nature* 2015) [PR; Nature venue from common knowledge]. A MAP-Elites behaviour repertoire plus trial-and-error lets a damaged hexapod adapt quickly. Source: [arXiv](https://arxiv.org/pdf/1407.3501)
- Allard, Smith, Chatzilygeroudis & Cully, "Hierarchical Quality-Diversity for Online Damage Recovery", GECCO 2022 [PR]. It uses hierarchical repertoires; a hexapod maze task needs 20% fewer actions and has 57% fewer complete failures. Source: [arXiv](https://arxiv.org/abs/2204.05726)
- ME-ES (GECCO 2020), QDAC (ICML 2024) and URSA (CoRL 2025), all above, also report damage or perturbation adaptation.

### Inferences
- **What a safety-indexed archive would add.** The novelty is to treat safety (expected episode cost, or empirical violation probability from repeated or branched rollouts) as a **descriptor axis**, not a constraint. That yields a deliverable "dial": the best-reward policy at each conservatism level, from one run. Prior work does one of three other things:
  - treats cost as a feasibility filter (Constrained MAP-Elites, Reset-free QD),
  - treats cost as a lexicographic replacement criterion (GuSS),
  - or targets numeric benchmarks (Fioravanzo & Iacca).
- **Descriptor versus constraint.** Cost as a descriptor illuminates the whole reward–cost frontier. Cost as a constraint gives one feasible region. The 1-D descriptor version is essentially an ε-constraint sweep done in parallel with shared variation. In each bin the elite approximates max reward subject to cost lying in that bin.
- **The baseline a reviewer will raise.** Given Batra et al. (2024) and CCPO (2023), reviewers will ask why an archive is needed instead of one threshold-conditioned policy. A credible paper should compare against:
  - CCPO-style or Lagrangian-sweep baselines (PPO-Lagrangian at k thresholds, which the student can already run);
  - DCRL-ME/QDAC-style distillation of the archive into one cost-conditioned policy.
- **Variance matters.** Because cost in SafetyWalker2dVelocity is accumulated over stochastic episodes, an elite's cost estimate is noisy. The Uncertain-QD and ARIA literature shows that naive MAP-Elites over-fills cells with "lucky" solutions. A safety descriptor is therefore exactly the regime where reproducibility methods matter. Exact MuJoCo snapshot branching could give low-variance paired re-evaluations; this is my inference, not a published result.

### Gaps
- I found **no** peer-reviewed or preprint QD archive over reward × expected cost (or violation probability) for safe RL policies, and **no** QD-RL work on Safety-Gym or Safety-Gymnasium beyond GuSS's use of SafeCar-Goal. This is a negative result from about eight differently phrased searches and the QD papers list. A paper published very recently or outside the indexed venues could still exist, so do a Semantic Scholar citation-trail check (citations of GuSS, Fioravanzo & Iacca, and CCPO) before submission.
- I could not verify a GECCO 2018 venue for Talakat or a final venue for Fioravanzo & Iacca from primary pages in this session.
- I did not find specific QD work on "chance-constrained" or CVaR-risk-aware archives in RL. Uncertain-QD and performance-reproducibility are the nearest neighbours.

---

## Key Question 3: Model-based and surrogate-assisted QD (SAIL, DA-QD, DSA-ME/DSAGE, world models), and how they handle model error

### Takeaway
Surrogate and model-based QD all follow the same three-phase loop:
1. Run QD cheaply in the model (an "acquisition", "imagined" or "surrogate" archive).
2. Send a selected subset to ground truth.
3. Retrain the model on those results.

Model exploitation is controlled in two ways: (i) uncertainty, via UCB optimism in SAIL or ensemble disagreement and low-uncertainty filtering in DA-QD; (ii) **always confirming archive entries with real evaluations**. Reported savings are about 3x (DSAGE), about 20x (DA-QD) and about 100x (SAIL, on low-dimensional designs). I found no QD method that fills an archive inside a *pixel-based JEPA-style latent world model* like LeWM. The nearest 2026 work searches a behavioural-foundation-model latent space (BFM-QD).

### Cited Findings
- **SAIL**:
  - Gaier, Asteroth & Mouret, "Data-Efficient Design Exploration through Surrogate-Assisted Illumination", *Evolutionary Computation* (2018; arXiv 1806.05865) [PR].
  - Earlier GECCO 2017 version: "Data-Efficient Exploration, Optimization, and Modeling of Diverse Designs through Surrogate-Assisted Illumination" (arXiv 1702.03713) [PR, GECCO 2017 per common attribution].
  - **Method:** a GP surrogate. MAP-Elites builds an **acquisition map** that maximises **UCB = μ + κσ**, and Sobol-sequence-chosen bins from it are evaluated for real (infill). At the end MAP-Elites builds a **prediction map** from the GP mean only. Model error is handled by UCB optimism plus iterative refinement with true evaluations.
  - **Results:** on 2-D airfoils, 1,000 SAIL evaluations versus 100,000 for MAP-Elites. A 3-D velomobile took about 1 day versus a projected 4 months.
  Source: [arXiv 1806.05865](https://ar5iv.labs.arxiv.org/html/1806.05865); [h-brs record](https://pub.h-brs.de/frontdoor/index/index/year/2018/docId/3705)
- **DA-QD**, Lim, Grillotti, Bernasconi & Cully, "Dynamics-Aware Quality-Diversity for Efficient Learning of Skill Repertoires", ICRA 2022 (DOI 10.1109/ICRA46639.2022.9811559; arXiv 2109.08522) [PR].
  - **Model:** an **ensemble of probabilistic NNs** as the forward dynamics model, capturing aleatoric noise (Gaussian output) and epistemic uncertainty (ensemble disagreement).
  - **Search:** QD runs "in imagination" to fill an imagined repertoire. Policies added there are sent for real evaluation, either all of them or only those with **low model disagreement**.
  - **Efficiency:** **about 20x more sample-efficient**, reaching vanilla-QD performance at about 5×10⁴ instead of 10⁶ evaluations on an 18-DoF hexapod.
  - **Failure modes the authors report:** long-horizon model error degrades performance, and some repertoires learnt in simulation relied on the leg that was later damaged.
  Source: [arXiv](https://arxiv.org/abs/2109.08522); [ar5iv](https://ar5iv.labs.arxiv.org/html/2109.08522)
- **Reset-free QD (2022, preprint) and physical-robot QD (GECCO 2023 Companion)** extend DA-QD. The model's predictions are also used to **filter out unsafe predicted behaviours** before executing them on hardware. Source: [arXiv 2204.03655](https://arxiv.org/abs/2204.03655); [arXiv 2304.12080](https://arxiv.org/abs/2304.12080)
- **DSA-ME**, Zhang, Fontaine, Hoover & Nikolaidis, "Deep Surrogate Assisted MAP-Elites for Automated Hearthstone Deckbuilding" (arXiv 2112.03534; GECCO 2022) [PR; also an ICLR 2022 workshop version]. A deep surrogate trained online predicts game outcomes, and MAP-Elites both exploits it and generates diverse training data. It beats offline-trained and linear surrogates. Source: [arXiv](https://arxiv.org/pdf/2112.03534); [ICLR-W](https://mlanthology.org/iclrw/2022/zhang2022iclrw-dsame)
- **DSAGE**, Bhatt, Tjanaka, Fontaine & Nikolaidis, "Deep Surrogate Assisted Generation of Environments", NeurIPS 2022 (arXiv 2206.04199) [PR].
  - **Loop:** "model exploitation, agent simulation, model improvement". QD on the surrogate builds a surrogate archive, a **downsampled** subset gets ground-truth evaluation, and the surrogate is retrained.
  - **Surrogate design:** the surrogate first predicts **ancillary agent-behaviour data** (tile occupancy grids) and then the objective and measures. This self-supervised intermediate target improves prediction.
  - **Efficiency:** about 34k versus about 100k evaluations to reach the same QD-score as MAP-Elites in Maze, roughly 3x.
  Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2206.04199)
- **GuSS** (above) is the only safety-oriented example. Its MAP-Elites runs over *short-horizon* rollouts inside a learned model and is re-planned every step, which limits compounding model error. Source: [arXiv](https://arxiv.org/pdf/2206.09743)
- **BFM-QD** (NeurIPS 2026 per arXiv comment) searches a behavioural foundation model's latent z with a gradient-free improvement operator. This is "QD in a learned latent space", but over *policy* latents, not world-model imagination. Source: [arXiv](https://arxiv.org/abs/2609.35615)
- **BOP-Elites**, Kent, Gaier, Mouret & Branke, "BOP-Elites, a Bayesian Optimisation Approach to Quality Diversity Search with Black-Box descriptor functions" (2023 on the QD list) [Venue unverified]. It is Bayesian-optimisation QD in which the *descriptors* are also expensive and black-box. That matters if "violation probability" itself must be estimated. Source: [QD papers list](https://quality-diversity.github.io/papers.html)
- Model-based *safe RL* outside QD is out of scope here. Note only that a Lagrangian-in-Dreamer approach exists (SafeDreamer, ICLR 2024), seen in search results but not examined. Source: [ICLR 2024 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/ece182f93af26c64187ba3f7dfd4309a-Paper-Conference.pdf)

### Inferences
- **Using LeWM as a pre-screen surrogate.** The standard recipe maps directly onto LeWM:
  - (a) Run CMA-MAE/MAP-Elites with a 1-D cost (or 2-D cost × speed) descriptor, scoring candidates by imagined return and imagined cost in LeWM.
  - (b) Evaluate only the imagined elites, or the low-disagreement ones, in real MuJoCo.
  - (c) Insert into the *real* archive only on ground-truth numbers.
  - (d) Fine-tune LeWM on the new rollouts.
- **The safety-specific failure mode.** Systematic *under-prediction of cost* would place unsafe policies in "safe" bins. Use a pessimistic acquisition for the safety axis, the reverse of SAIL's optimism: UCB on cost, LCB on reward. Also report the model-vs-real calibration of cost per bin.
- **Why branching helps.** LeWM is image-based and JEPA-style. The cited model-based QD methods use state-space ensembles (DA-QD), GPs (SAIL) or task-specific CNNs (DSAGE), so the student would be first to test QD with a latent image world model. Expect long-horizon drift (DA-QD's caveat), which favours short imagined horizons anchored on real MuJoCo snapshots. The student's exact snapshot branching supports this, like GuSS's per-step re-planning.

### Gaps
- No verified 2022–2026 paper fills a QD archive of *RL policies* using a Dreamer/JEPA/latent *world model* trained from pixels. I searched twice with different phrasings; results were world-model surveys and unrelated papers.
- I did not verify the ensemble size or the exact uncertainty-selection threshold used in DA-QD.
- I found no paper that quantifies how surrogate or model error biases *cost* or *safety* descriptors specifically.

---

## Key Question 4: QD versus multi-objective EAs (NSGA-II, MOME) for reward–cost trade-offs: which suits a "dial"?

### Takeaway
If the only thing that matters is the reward–cost trade-off, the target object is a 2-objective Pareto front. NSGA-II/SPEA2, or a 1-D cost-binned MAP-Elites (an ε-constraint grid), both approximate it. On Brax locomotion, MOME's *global* hypervolume is statistically indistinguishable from NSGA-II/SPEA2. QD adds value when you also want diversity in *how* policies achieve each safety level (gait or speed descriptors): MOME keeps a Pareto front per descriptor cell. For a "dial", a 1-D or 2-D QD archive with cost as a descriptor is simpler to explain and evaluate than MOME. NSGA-II should be included as the MOEA baseline.

### Cited Findings
- **MOME**, Pierrot et al. (InstaDeep / Imperial), "Multi-Objective Quality Diversity Optimization", GECCO 2022 (arXiv 2202.03057) [PR].
  - **Archive:** each MAP-Elites cell stores a **Pareto front (max size 50)**.
  - **Metrics:** **MOQD-score** (sum of per-cell hypervolumes) and **global hypervolume**.
  - **Brax tasks:** HalfCheetah, **Walker2d** and Humanoid, with objectives "forward velocity" and "−control (energy) cost" and foot-contact descriptors. 1M evaluations, 20 seeds.
  - **Results:** MOME is significantly better on MOQD-score (p<0.003). On *global* Pareto-front hypervolume, "Wilcoxon ranked-test could not highlight statistically significant difference" from NSGA-II/SPEA2 (p = 0.08–0.38).
  Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.03057); [arXiv](https://arxiv.org/abs/2202.03057v1)
- **MOME-PGX**, Janmohamed, Pierrot & Cully, GECCO 2023 [PR]. It adds policy gradients and crowding to MOME, is **4.3–42x more data-efficient than MOME**, and beats MOME, NSGA-II and SPEA2 on four robot locomotion tasks. Source: [arXiv](https://arxiv.org/abs/2302.12668)
- **MOME-P2C**: Janmohamed et al., "Multi-Objective Quality-Diversity in Unstructured and Unbounded Spaces" (GECCO 2024) [Venue unverified this session; not fetched].
- **QD as MOO**: Lin, Guo, Liu, Zhang & Sun, "Quality-Diversity Optimization as Multi-Objective Optimization" (arXiv 2602.00478, 31 Jan 2026) [Preprint]. It recasts QD as MOO with a huge number of objectives, solved with set-based scalarisation, and is competitive with state-of-the-art QD on robot control. Source: [arXiv](https://arxiv.org/abs/2602.00478)
- **Lagrange-multiplier sensitivity** (why a single PPO-Lagrangian run is not a dial): "An Empirical Study of Lagrangian Methods in Safe Reinforcement Learning" (arXiv 2510.17564, Oct 2025) [Preprint]. It notes that λ governs the return–cost trade-off: too low gives unsafe policies, too high gives overly conservative ones. Source: [arXiv](https://arxiv.org/html/2510.17564v1)
- **Evolutionary MORL for continuous control**: MO-ERL ("Extending Evolution-Guided Policy Gradient Learning into the multi-objective domain", *Neurocomputing* 2025) reports up to 62.7% higher hypervolume than CAPQL/PCN [PR; detail from search snippet only]. **C-MORL** (arXiv 2410.02236; ICLR 2025 per proceedings link) fills Pareto-front gaps with constrained policy optimisation [PR per link; not fetched]. Source: [ScienceDirect](https://sciencedirect.com/science/article/pii/S0925231225006630); [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/adb77ecc8ba1c2d3135c86a46b8f2496-Paper-Conference.pdf)

### Inferences
- **A 1-D archive is a Pareto front.** For a pure "dial" (one safety axis), a 1-D MAP-Elites/CMA-MAE archive over cost bins with reward as fitness is a discretised, ε-constraint view of the reward–cost Pareto front. Its upper envelope *is* the front, and dominated bins show up as non-monotone entries. That makes direct comparison with NSGA-II hypervolume easy and fair.
- **Why QD over an MOEA.** Fixed, user-interpretable safety levels, guaranteed coverage of every level rather than a crowded region of the front, and the option to add a second behaviour descriptor (speed or gait) for MOME-style diversity at each safety level.
- **Why NSGA-II still matters as a baseline.** MOME's own results show MOEAs match QD on global hypervolume. So "QD finds a better frontier" is not a safe claim. "QD gives guaranteed per-level coverage and diversity" is.

### Gaps
- I found no paper comparing QD and NSGA-II specifically on a safe-RL reward–cost front.
- I did not verify MOME-P2C's venue, or the MO-ERL and C-MORL details beyond search snippets.

---

## Key Question 5: Metrics, and how to evaluate a safety-indexed archive in the real environment

### Takeaway
The standard metrics are QD-score, coverage, best fitness and archive CCDF (also called the "archive profile"). For multi-objective archives they are MOQD-score and global hypervolume. The QD-RL community now expects **"corrected" metrics**: every elite is re-evaluated many times (PPGA uses 50) and placed back into its *true* cell, because noisy evaluations inflate archives. A safety-indexed archive needs this even more, because its descriptor (cost or violation rate) is itself a noisy expectation.

### Cited Findings
- **QD-score and coverage**: QD-score (the sum of elite fitnesses, with fitness offset to be non-negative) is commonly attributed to Pugh, Soros & Stanley (Frontiers 2016) [PR; the exact first definition was not checked]. Source: [Frontiers](https://doaj.org/article/8292416d6504484db8032e02c4288cd9)
- **Benchmark metrics with robustness corrections**: Flageat, Lim, Grillotti, Allard, Smith & Cully, "Benchmarking Quality-Diversity Algorithms on Neuroevolution for Reinforcement Learning" (arXiv 2211.02193; GECCO 2022 Workshop on QD Algorithm Benchmarks) [PR, workshop]. It defines tasks and descriptors in QDax and the metrics **coverage, QD-score, max fitness and archive profile**. It also introduces **"corrected" versions** of these metrics to quantify robustness to environment stochasticity. Source: [arXiv](https://arxiv.org/abs/2211.02193)
- **Corrected archive protocol**: PPGA re-evaluates each agent **50 times** and averages performance and measures to build a "Corrected Archive", because "QD algorithms are known to struggle with reproducing performance and behavior in stochastic environments". It also plots **CCDF** curves showing the percentage of archive policies with reward ≥ R, "originally presented in [Vassiliades et al. 2016]". Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2305.13795)
- **Uncertain-QD methodology**: Flageat & Cully, "Uncertain Quality-Diversity: Evaluation methodology and new methods for Quality-Diversity in Uncertain Domains" (arXiv 2302.00463, Feb 2023; "submitted to TEVC" per arXiv) [Preprint/unverified final venue]. Fitness and descriptors are treated as distributions. It proposes a per-generation sampling budget and new metrics, plus Archive-sampling, Parallel-Adaptive-sampling and Deep-Grid-sampling. Source: [arXiv](https://arxiv.org/abs/2302.00463)
- **Uncertain-domain benchmarks**: Flageat, Grillotti & Cully, "Benchmark tasks for Quality-Diversity applied to Uncertain domains" (arXiv 2304.12454, 2023). Source: [arXiv](https://arxiv.org/pdf/2304.12454)
- **Multi-objective metrics**: MOQD-score (sum of per-cell hypervolumes) and global hypervolume with a user reference point (MOME, GECCO 2022). Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.03057)
- **Safety metrics in a QD + safe-RL paper**: GuSS reports p(unsafe)% (the fraction of time steps in unsafe states), its transient version, and Mean Asymptotic Reward. Source: [arXiv PDF](https://arxiv.org/pdf/2206.09743)

### Inferences
Proposed evaluation protocol for a safety-dial archive (my synthesis):
1. **Re-evaluate in real MuJoCo.** Run N ≥ 20–50 episodes per elite, ideally with common random seeds or snapshot branches. Recompute the *safety descriptor* and re-bin, then report corrected QD-score, corrected coverage and per-bin reward.
2. **Report the dial curve.** Plot best corrected reward against realised cost (or violation probability) with confidence intervals. Overlay PPO-Lagrangian trained at k thresholds, a CCPO-style conditioned policy, and NSGA-II. Compute 2-D hypervolume against a fixed reference point.
3. **Report descriptor fidelity.** For each bin, report the fraction of elites whose realised cost stays inside their claimed bin; this is a "safety reproducibility" score. If LeWM is used, also report model-predicted versus real cost, i.e., calibration error per bin.
4. **Report CCDF or archive profile per safety band**, plus monotonicity of the frontier. A safer bin should never beat a less safe bin on reward after correction, or the dial is miscalibrated.
5. **Report cost of production.** Count real environment steps and wall-clock time for the whole archive, compared with training k separate PPO-Lagrangian policies.

### Gaps
- I did not confirm the original CCDF source (Vassiliades et al. 2016); it is cited only through PPGA.
- No established metric exists for the "descriptor reproducibility of a safety axis". The Uncertain-QD metrics are the nearest, and I did not enumerate them because the abstract did not.

---

## Key Question 6: Gaps a 4-month Honours project could credibly fill (targeting GECCO 2027)

### Takeaway
The most defensible novelty is a **safety-indexed QD archive for safe RL**: cost or violation probability as a descriptor axis, with reward as fitness. It should be evaluated with corrected re-evaluation on SafetyWalker2dVelocity, and compared with PPO-Lagrangian sweeps, NSGA-II and a threshold-conditioned policy. An optional second contribution is **world-model-pre-screened** archive filling using LeWM, with pessimistic cost estimates and real-MuJoCo confirmation. None of these pieces is individually unprecedented, as the citations above show. I found no verified instance of the combination.

### Cited Findings
- No prior reward × safety QD archive for safe RL was found (see KQ2). The nearest are GuSS (QD planner with cost-lexicographic replacement, preprint), Constrained MAP-Elites (feasibility populations), Fioravanzo & Iacca (violations as descriptors on numeric benchmarks), and Flageat et al. 2024 (user-preference performance-reproducibility trade-off). Source: [GuSS](https://arxiv.org/abs/2206.09743); [Talakat](https://arxiv.org/pdf/1806.04718); [Fioravanzo & Iacca](https://arxiv.org/abs/1902.00703); [Flageat 2024](https://arxiv.org/abs/2409.13315)
- Model-based QD uses state-space ensembles, GPs or task-specific CNN surrogates (DA-QD, SAIL, DSAGE). No pixel-trained latent world model appears. Source: [DA-QD](https://arxiv.org/abs/2109.08522); [SAIL](https://ar5iv.labs.arxiv.org/html/1806.05865); [DSAGE](https://ar5iv.labs.arxiv.org/html/2206.04199)
- The supervisor's own paper shows CMA-MEGA (TD3, ES) matching PGA-ME on QD Walker with 1M evaluations. Measure gradients are estimated by ES, which costs evaluations. Source: [Tjanaka 2022](https://ar5iv.labs.arxiv.org/html/2202.03666)
- The GPU safe-RL benchmark CRAX (MJX, 2026 preprint) makes JAX-speed safe-RL environments available for QDax-style massive parallelism. Source: [arXiv](https://arxiv.org/abs/2606.20376)
- An anticipated critique: one conditioned policy can replace bounded archives (Batra et al. 2024), and CCPO already provides threshold-conditioned zero-shot safe policies. Source: [Batra 2024](https://arxiv.org/abs/2407.17515v1); [CCPO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/29906cbd165b78991da2c4dbabc2a04b-Abstract.html)

### Inferences
Candidate contributions ranked by feasibility for one RTX 5090 and four months (my judgement):
1. **(Core, feasible) "Safety-dial MAP-Elites/CMA-MAE".**
   - **Setup:** a 1-D cost archive (10–20 bins over the velocity-cost budget range), or 2-D cost × mean-speed. Policies are small MLPs initialised from the student's PPO and PPO-Lagrangian checkpoints, which seed several bins at once.
   - **Variation:** CMA-MAE or a PGA-style operator using a PPO update plus a cost critic for the measure gradient. This last option is the CMA-MEGA/PPGA analogue and connects directly to the supervisor's paper.
   - **Evaluation:** budget about 10⁴–10⁵ real episodes, using the corrected protocol from KQ5.
2. **(Strong add-on) Cost as a descriptor versus cost as a constraint.** Run the same budget with (a) cost as a descriptor, (b) Constrained-MAP-Elites-style feasibility at a fixed threshold, and (c) GuSS-style lexicographic replacement. Show which variant recovers the frontier best. This is a clean, publishable ablation that answers "descriptor vs constraint" directly.
3. **(Ambitious) LeWM-pre-screened illumination.** Run QD inside LeWM, send imagined elites to MuJoCo, and use a pessimistic cost estimate. Report real-evaluation savings and how many unsafe policies the model wrongly placed in safe bins. This is risky because of LeWM's long-horizon fidelity. Shorter horizons branched from real MuJoCo snapshots reduce that risk.
4. **(Cheap, valuable) Safety-reproducibility analysis.** Measure how often an elite's realised violation rate stays in its bin under re-evaluation. Apply ARIA-style or archive-sampling fixes from Uncertain-QD.
5. **(Baselines that make it credible)** PPO-Lagrangian at k thresholds (already available), NSGA-II on (reward, −cost), and optionally distilling the archive into one cost-conditioned policy (DCRL-ME style) to answer the Batra/CCPO critique.

Practical notes:
- **Libraries:** pyribs is CPU/NumPy-friendly and wraps a MuJoCo or Safety-Gymnasium evaluation loop easily, so it suits a CPU-simulated task. QDax suits a JAX/MJX port.
- **GECCO framing:** GECCO tracks favour clear algorithmic or empirical contributions on standard benchmarks with statistical tests across seeds (20 seeds is the norm in Imperial-lab QD papers).

### Gaps
- I could not determine whether any GECCO 2026 paper (July 2026) already addresses safety-indexed QD archives. The GECCO 2026 proceedings were not searched directly, and this should be checked on ACM DL before committing.
- I did not find hard numbers on CPU throughput of Safety-Gymnasium Walker2d. Budget feasibility on CPU MuJoCo should be measured locally before fixing the archive size.
