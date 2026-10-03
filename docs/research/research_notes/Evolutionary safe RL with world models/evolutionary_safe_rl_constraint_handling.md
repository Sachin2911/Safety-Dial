# Evolutionary / population-based safe RL and EC constraint handling as an alternative to hand-tuned penalties

Research date: 3 October 2026. "Peer-reviewed" means a conference or journal version was confirmed. "Preprint" means only an arXiv version was found. Every paper below was checked against a primary page (arXiv abstract, publisher DOI via Crossref, proceedings page, or the PDF itself) unless it is flagged **[unverified]**.

---

## Q1. Which papers apply evolution (ES, CMA-ES, GA, CEM, ERL hybrids, neuroevolution) to constrained or safe RL, and what do they do?

### Takeaway
Few papers do this directly, and they are scattered across venues. The closest to the student's plan are Wen & Topcu's Constrained Cross-Entropy (NeurIPS 2018) and Liu et al.'s Robust CEM (2020 preprint). Both rank feasible samples by objective and infeasible samples by constraint value, which is essentially the student's "Safe CEM" rule, so that ranking idea is not novel in itself. ECRL (IJCNN 2023) is the main constrained-ERL hybrid, but it uses MuJoCo torque constraints, not Safety-Gym/Safety-Gymnasium. Other examples use evolution for a narrower part of the problem: mutation operators (GECCO'22 Companion), cost-function search (AutoCost, AAAI'23), and safe black-box optimisation (Safe CMA-ES, GECCO'24, synthetic functions only). None of them is a pure ES/CMA-ES/GA policy search benchmarked against PPO-Lagrangian on Safety-Gymnasium.

### Cited Findings

#### Constrained Cross-Entropy (CCE), Wen & Topcu, NeurIPS 2018 (peer-reviewed)
- Problem: a CMDP whose constraints are expected costs over finite-length trajectories. The method "explicitly tracks its performance with respect to constraint satisfaction." Source: [NeurIPS 2018 abstract](https://proceedings.neurips.cc/paper/2018/hash/34ffeb359a192eb8174b6854643cc046-Abstract.html)
- Ranking rule, from the PDF. Elites are chosen from a policy-parameter distribution:
  - If the ρ-quantile of the constraint value is still above the threshold d (too few feasible samples), elites are the samples with the best constraint values.
  - Otherwise, elites are the feasible samples with the best objective.
  - Remark 1: maximising the surrogate U "implicitly prioritizes feasibility": every feasible policy ranks above every infeasible one.
  - Source: [Wen & Topcu PDF](https://papers.NeurIPS.cc/paper_files/paper/2018/file/34ffeb359a192eb8174b6854643cc046-Paper.pdf)
- Theory: the asymptotic behaviour is described by an ODE, and the paper gives sufficient conditions for almost-sure convergence. It does not assume that the initial policy is feasible. Source: [NeurIPS abstract](https://proceedings.neurips.cc/paper/2018/hash/34ffeb359a192eb8174b6854643cc046-Abstract.html)
- Experiments (small scale):
  - A mobile-robot navigation task with local sensors, a goal region and a bad region. Policy is an MLP with 2×30 hidden units, horizon N=30, implemented in rllab, 5 repeats.
  - Baselines: TRPO (unconstrained), CPO, and TRPO with a fixed penalty of 100.
  - CCE reached feasible policies in all four objective/constraint combinations, including non-Markovian constraints. CPO "needs significantly more samples to find a single feasible policy, or simply converges to an infeasible policy."
  - With the fixed penalty, TRPO policies "are either infeasible or with very small constraint values."
  - Source: [Wen & Topcu PDF](https://papers.NeurIPS.cc/paper_files/paper/2018/file/34ffeb359a192eb8174b6854643cc046-Paper.pdf)
- Stated limitations: "CCE is expected to be less sample-efficient than gradient-based methods especially for high-dimensional systems". It "does not infer the performances of unseen policies", but "can be easily parallelized". Source: [Wen & Topcu PDF](https://papers.NeurIPS.cc/paper_files/paper/2018/file/34ffeb359a192eb8174b6854643cc046-Paper.pdf)

#### Robust Cross-Entropy (RCE) for safe model-based RL, Liu, Zhou, Chen, Zhong, Hebert, Zhao (arXiv Oct 2020, revised Mar 2021; preprint, no venue on arXiv)
- Setup: MPC with an ensemble neural dynamics model, sparse indicator cost signals, and a "robust cross-entropy method" for optimising control sequences under constraints. Evaluated on Safety Gym. Source: [arXiv 2010.07968](https://arxiv.org/abs/2010.07968)
- Elite rule: if feasible sequences exist, sort them by reward and take the top-k. If none exist, sort by cost in ascending order. This is the same feasibility-first logic as CCE and as the student's Safe CEM, but applied to action sequences rather than policy parameters. Source: [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2010.07968)
- Violations in the first 10,000 training steps:

  | Task | MPC-RCE | MPC-CEM (penalty) |
  |---|---|---|
  | PointGoal1 | 16 | 184 |
  | PointGoal2 | 231 | 746 |
  | CarGoal1 | 7 | 104 |
  | CarGoal2 | 96 | 578 |

  The abstract claims "several orders of magnitude better sample efficiency" than constrained model-free RL (TRPO-Lagrangian, CPO). Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.07968); [arXiv](https://arxiv.org/abs/2010.07968)

#### ECRL, "Evolving Constrained Reinforcement Learning Policy", Hu, Pei, Liu, Yao (IJCNN 2023, peer-reviewed; arXiv Apr 2023)
- Method: an ERL hybrid of a population of actors and a SAC learner. The learner's gradient is injected by replacing the worst individual.
  - Actors are ranked by **stochastic ranking** (Runarsson & Yao 2000) over reward and constraint violation.
  - Each actor has its own **Lagrange multiplier**, initialised uniformly in (0,1).
  - Multipliers are updated from a **constraint buffer** of recent episodic constraint values.
  - Sources: [arXiv 2304.09869](https://arxiv.org/abs/2304.09869); [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2304.09869)
- Benchmarks: MuJoCo Ant, HalfCheetah, Walker2d, Hopper and Swimmer. The constraint is that average torque per motor stays below ε=0.4 (not Safety-Gym hazards). Budget: 1e6 timesteps. Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2304.09869)
- Baselines: RCPO (SAC-based), IPO (PPO-based), vanilla ERL, RCPO-ERL, and ERL with reward shaping over a penalty sweep λ = 1e-5 … 100. **No PPO-Lagrangian or CPO.** Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2304.09869)
- Results:
  - On Hopper and Swimmer, "only our ECRL agent" satisfied the constraint.
  - On HalfCheetah and Walker2d, "no agent satisfies the constraint", but ECRL had the lowest violation.
  - Ablations claim that stochastic ranking, the buffer and the multipliers are "indispensable".
  - Hyperparameters (buffer size 100, sync period 11) were "arbitrarily chosen".
  - Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2304.09869)
- Follow-up: ACERL, "Robust Dynamic Material Handling via Adaptive Constrained Evolutionary RL" (Hu, Wang, Yuan, Liu, Zhang, Yao; arXiv Jun 2025, preprint). It keeps a population of actors and uses constraint violation to restrict policy behaviour, but it is applied to scheduling, not locomotion. Source: [arXiv 2506.16795](https://arxiv.org/abs/2506.16795)

#### Safety-Informed Mutations for Evolutionary DRL, Marchesini & Amato (GECCO '22 Companion, 5-page poster paper, DOI 10.1145/3520304.3533980, peer-reviewed)
- Method: a mutation operator inside an ERL hybrid (a SUPE-RL style implementation on PPO, called "SM-PPO"). It uses a buffer of visited unsafe states and gradient information to bias mutations away from unsafe actions. Source: [PDF](https://emarche.github.io/assets/pdf/GECCO2022_safemut.pdf)
- Benchmarks: Safety Gym **DoggoGoal1** (12-DoF locomotion) and a TurtleBot3 mapless-navigation task. Baselines: PPO, Lagrangian PPO, IPO, CPO. Ten seeds on one RTX 2070. Source: [PDF](https://emarche.github.io/assets/pdf/GECCO2022_safemut.pdf)
- Results, reported only as a "Pareto frontier at convergence" of cost against reward:
  - On DoggoGoal1, L-PPO "maintains the imposed cost limit but fails at learning the locomotion".
  - SM-PPO reached rewards comparable to CPO/IPO and "significantly reduces the cost value".
  - Training-time violations were not reported.
  - Source: [PDF](https://emarche.github.io/assets/pdf/GECCO2022_safemut.pdf)

#### AutoCost: Evolving Intrinsic Cost for Zero-violation RL, He, Zhao, Liu (AAAI 2023, peer-reviewed)
- What is evolved: not the policy, but an intrinsic cost function. It is a 41-parameter MLP with one hidden layer of 4 units.
  - Simple ES: population 50, Gaussian-noise and random-scaling mutations.
  - Fitness: keep the top 10% with the lowest extrinsic violations of the *converged* inner policy, breaking ties by reward.
  - Inner learners: CPO and PPO-Lagrangian.
  - Sources: [ar5iv](https://ar5iv.labs.arxiv.org/html/2301.10339); [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/26734)
- Motivation: "constrained RL methods fail to achieve zero violation even when the cost limit is zero". Source: [arXiv 2301.10339](https://arxiv.org/abs/2301.10339)
- Results: the evolved cost gave converged policies with zero violation across Safety Gym Goal/Push × Hazard/Pillar × Point/Car/Doggo, with more conservative behaviour.
- Cost: one evolution (Goal-Hazard-Point) took **54.1 hours** with 25 parallel candidates on 2× RTX 2080Ti.
- Source for results and cost: [ar5iv](https://ar5iv.labs.arxiv.org/html/2301.10339)

#### Provably convergent constrained ES, Diouane, Lucchi, Patil (AISTATS 2022; arXiv Feb 2022)
- An ES with a sufficient-decrease mechanism, giving global convergence for *stochastic constrained* problems using only function estimates.
- Demonstrated on (i) reward maximisation and (ii) reward maximisation under "a non-relaxable set of constraints". Environment details were not visible on the abstract page.
- Source: [arXiv 2202.10464](https://arxiv.org/abs/2202.10464)

#### Safe CMA-ES, Uchida, Hamano, Nomura, Saito, Shirakawa (GECCO 2024 full paper, peer-reviewed)
- Target: "safe optimization", where evaluating unsafe solutions is itself risky.
  - Estimates Lipschitz constants of the safety functions using the maximum gradient norm of a GP regression.
  - Projects CMA-ES samples to the nearest point of the estimated safe region.
  - Source: [arXiv 2405.10534](https://arxiv.org/abs/2405.10534)
- Assumptions:
  - Requires **safe seeds** (10 in the experiments).
  - Safety thresholds are known.
  - Objective values of unsafe solutions are observable.
  - Source: [arXiv HTML](https://arxiv.org/html/2405.10534v1)
- Benchmarks: Sphere, Ellipsoid, Reversed Ellipsoid and Rosenbrock in d=5 and d=20.
- Baselines:
  - naive CMA-ES
  - **CMA-ES with augmented Lagrangian**
  - CMA-ES with violation avoidance
  - SafeOpt, modified SafeOpt, swarm-based SafeOpt
  - Source: [arXiv HTML](https://arxiv.org/html/2405.10534v1)
- Results: median unsafe evaluations were zero for Safe CMA-ES on all problems. The authors leave realistic problems and discontinuous safety functions to future work. **No RL experiments.** An Optuna sampler exists: [OptunaHub safe_cma](https://hub.optuna.org/samplers/safe_cma/). Source: [arXiv HTML](https://arxiv.org/html/2405.10534v1)

#### Other evolutionary safety work (smaller or adjacent)
- **Meta-Learned Instinctual Networks (MLIN)**, Grbic & Risi (arXiv May 2020; an ALIFE venue is commonly cited but was **not verified**).
  - An *evolved* fixed "instinct" network modulates a plastic RL network to block unsafe actions.
  - Tested on a 2D navigation task with no-go zones; agents reached new targets "without colliding with any of the no-go zones".
  - Source: [arXiv 2005.03233](https://arxiv.org/abs/2005.03233)
- **Guided Safe Shooting (GuSS)**, Paolo, Gonzalez-Billandon, Thomas, Kégl (arXiv Jun 2022, revised Sep 2024; preprint).
  - Model-based RL with three safe planners: one random-shooting planner and two **MAP-Elites (quality-diversity)** planners.
  - Benchmarks and baselines were not visible on the abstract page.
  - Source: [arXiv 2206.09743](https://arxiv.org/abs/2206.09743)
- **SNES, "Synthesizing safe policies under probabilistic constraints with RL and Bayesian model checking"**, Belzner & Wirsing (Science of Computer Programming, June 2021; peer-reviewed, DOI 10.1016/j.scico.2021.102620).
  - Per the publisher abstract snippet, Safe Neural Evolution Strategies combine ES with Bayesian statistical verification. They adjust the Lagrangian weighting of return against cost according to confidence that a *probabilistic* safety specification is satisfied.
  - **Full text not accessed (HTTP 403)**, so details are unverified.
  - Sources: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167642321000137); [Crossref DOI](https://doi.org/10.1016/j.scico.2021.102620)
- **LyEvO**, Curcio, Cao, Caccamo (arXiv 6 Aug 2026; preprint).
  - "Constrained Evolutionary Optimization" plus statistical-model-checking verification plus Lyapunov stability regions.
  - Tested on Cartpole and a 3D quadrotor with real-world experiments. The specific EA is not named in the abstract.
  - Source: [arXiv 2608.06481](https://arxiv.org/abs/2608.06481)

#### Surveys (for positioning)
- Bai, Cheng & Jin, "Evolutionary Reinforcement Learning: A Survey", *Intelligent Computing* (Jan 2023). It covers hyperparameter optimisation, policy search, exploration, reward shaping, meta-RL and multi-objective RL. Safe/constrained RL is not one of its listed categories. Source: [DOI 10.34133/icomputing.0025](https://doi.org/10.34133/icomputing.0025); [arXiv 2303.04150](https://arxiv.org/pdf/2303.04150)
- Sigaud, "Combining Evolution and Deep RL for Policy Search: A Survey", *ACM TELO* (Sep 2023; arXiv 2022). It covers 45 post-2017 algorithms and focuses on combination mechanisms. Source: [DOI 10.1145/3569096](https://doi.org/10.1145/3569096); [arXiv 2203.14009](https://arxiv.org/abs/2203.14009)
- Gu et al., "A Review of Safe RL: Methods, Theories and Applications", *IEEE TPAMI* (Dec 2024). Source: [DOI 10.1109/TPAMI.2024.3457538](https://doi.org/10.1109/TPAMI.2024.3457538); [arXiv 2205.10330](https://arxiv.org/pdf/2205.10330)

### Inferences
- **Novelty risk for "Safe CEM".** The student's feasibility-first CEM, which ranks feasible candidates by goal cost and infeasible ones by violation magnitude, is close to two existing methods:
  - CCE (Wen & Topcu 2018), the same rule over policy parameters;
  - RCE (Liu et al. 2020), the same rule over MPC action sequences on Safety Gym.
  
  A GECCO paper must cite both and cannot claim the ranking rule as new. The contribution has to come from elsewhere: the setting (Safety-Gymnasium locomotion with probability of violation), the evaluation (training-time cost against PPO-Lag at matched simulation steps), the "dial"/Pareto output, or a link to ROSARL (Q5).
- The existing evolutionary safe-RL literature has three weaknesses:
  1. Most of it is either an ERL hybrid, where the gradient learner does most of the work (ECRL, SM-PPO), or evolves something other than the policy (AutoCost, MLIN).
  2. Benchmarks are often non-standard (ECRL's torque constraints, CCE's toy navigation).
  3. Training-time violations are rarely reported. RCE is the exception.
  
  A clean, direct ES/CMA-ES/CEM policy search on SafetyWalker2dVelocity-v1, with a proper safe-RL protocol, appears to be unoccupied ground.
- The student's ~8,000 steps/s × ~24 workers ≈ 1.9×10^5 steps/s gives about 10^8 environment steps in under 10 minutes (my arithmetic). ES-scale budgets are therefore affordable. Salimans et al. showed ES is competitive on MuJoCo when parallelised ([arXiv 1703.03864](https://arxiv.org/abs/1703.03864)), and Mania et al. showed that random search over *linear* policies matches the sample efficiency of state-of-the-art methods on MuJoCo locomotion ([arXiv 1803.07055](https://arxiv.org/abs/1803.07055)).

### Gaps
- No peer-reviewed paper was found that runs pure ES/CMA-ES/GA/CEM *policy* search on Safety-Gym or Safety-Gymnasium and compares it with PPO-Lagrangian/CPO under the Safety-Gym protocol. Absence from searches is not proof of absence.
- Not verified in this session:
  - Benchmark details for SNES (paywall) and GuSS (abstract page only).
  - The MLIN venue.
- Not examined: a "constrained differentiable CEM for safe model-based RL" (Penn State record seen in search results only).
- No 2025–2026 GECCO full paper on evolutionary safe RL turned up in searches of the GECCO 2025 table of contents ([GECCO'25 TOC](https://www.sigevo.org/gecco-2025/toc.html)). The GECCO 2026 proceedings were not checked.

---

## Q2. Which EC constraint-handling techniques have been used in policy search, what does theory/benchmarking say about when they beat penalties, and how does chance-constrained (probability-of-violation) optimisation work in EC?

### Takeaway
The standard EC toolbox is: Deb's feasibility rules, stochastic ranking, the ε-constrained method, (adaptive) penalties, augmented-Lagrangian CMA-ES, active CMA-ES with constraint vectors, and multi-objective "constraints-as-objectives". These techniques exist mainly to avoid choosing penalty coefficients. Their known trade-offs are:
- Feasibility rules have no parameters but are prone to premature convergence.
- Stochastic ranking and ε-constraint trade feasibility against objective in a controlled way.
- Augmented Lagrangian is the CMA-ES community's principled adaptive penalty, with an off-the-shelf pycma implementation.

In policy search, only feasibility-first CEM (CCE, RCE), stochastic ranking (ECRL) and Lagrangian-style penalties (ECRL, SNES) have actually been used. Chance-constrained EC (Neumann group, reliability-based design) estimates violation probabilities by sampling or tail bounds and often turns the problem into a bi- or tri-objective one. This maps directly onto "probability of safety violation".

### Cited Findings

#### Foundational constraint-handling references (all peer-reviewed; metadata confirmed via Crossref)
- **Deb (2000)**, "An efficient constraint handling method for genetic algorithms", *CMAME* 186. Proposes the three tournament feasibility rules: feasible beats infeasible; between feasible solutions, the better objective wins; between infeasible solutions, the lower violation wins. Source: [DOI 10.1016/S0045-7825(99)00389-8](https://doi.org/10.1016/S0045-7825(99)00389-8)
- **Runarsson & Yao (2000)**, "Stochastic ranking for constrained evolutionary optimization", *IEEE TEVC* 4(3). Source: [DOI 10.1109/4235.873238](https://doi.org/10.1109/4235.873238)
  - Stochastic ranking "was designed to deal with the inherent shortcomings of a penalty function (over and under penalization due to unsuitable values for the penalty factors)".
  - Instead of penalty factors, a probability Pf decides whether two infeasible solutions are compared by violation or by objective only, using a bubble-sort-like ranking. It was originally used inside an ES.
  - Source for the description: [Mezura-Montes & Coello 2011](https://doi.org/10.1016/j.swevo.2011.10.001)
- **Runarsson & Yao (2005)**, "Search biases in constrained evolutionary optimization", *IEEE TSMC-C*. Analyses why and when multi-objective constraint handling works or fails and concludes that "the unbiased multi-objective approach to constraint handling may not be as effective as one may have assumed." Source: [DOI 10.1109/TSMCC.2004.841906](https://doi.org/10.1109/TSMCC.2004.841906); [Lingnan record](https://scholars.ln.edu.hk/en/publications/search-biases-in-constrained-evolutionary-optimization/)
- **Runarsson & Yao (2003)**, "Evolutionary search and constraint violations" (CEC 2003). Different penalty functions create different search biases. Treating the problem as multi-objective removes the bias, but "in practice multiobjective methods are not an efficient or effective approach to constrained evolutionary optimization." Source: [Lingnan record](https://scholars.ln.edu.hk/en/publications/evolutionary-search-and-constraint-violations/)
- **Takahama & Sakai**, the ε-constrained method:
  - Relaxes feasibility to violation ≤ ε, then orders lexicographically (violation first, objective second), with ε decreased over time.
  - Strong DE variants add gradient-based mutation and archives.
  - Papers: [CEC 2006, DOI 10.1109/CEC.2006.1688283](https://doi.org/10.1109/CEC.2006.1688283); [CEC 2010, DOI 10.1109/CEC.2010.5586484](https://doi.org/10.1109/CEC.2010.5586484)
  - Description: [Mezura-Montes & Coello 2011](https://doi.org/10.1016/j.swevo.2011.10.001)
- **Mezura-Montes & Coello Coello (2011)**, "Constraint-handling in nature-inspired numerical optimization: Past, present and future", *Swarm & Evol. Comp.* 1:173–194. Key statements (read from the PDF):
  - Feasibility rules are popular because they can "be coupled to a variety of algorithms, without introducing new parameters."
  - "Their main drawback is that they are prone to cause premature convergence … this sort of scheme strongly favors feasible solutions … if no further mechanisms are adopted to preserve diversity (particularly paying attention to the need to keep infeasible solutions in the population), this approach will significantly increase the selection pressure."
  - "empirical evidence has suggested that multi-objective concepts are not well-suited to solve CNOPs", though some competitive MO-based methods exist (e.g., IDEA, which keeps a proportion of infeasible solutions in an NSGA-II-like scheme).
  - Source: [DOI 10.1016/j.swevo.2011.10.001](https://doi.org/10.1016/j.swevo.2011.10.001); [PDF mirror](https://faculty.csu.edu.cn/_tsf/00/65/zAn6Rj7NruE3.pdf)
- **Coello Coello (2002)**, survey of theoretical and numerical constraint-handling techniques, *CMAME* 191. Source: [DOI 10.1016/S0045-7825(01)00323-1](https://doi.org/10.1016/S0045-7825(01)00323-1)
- **Hellwig & Beyer (2019)**, "Benchmarking evolutionary algorithms for single objective real-valued constrained optimization – A critical review", *Swarm & Evol. Comp.* A critique of CEC constrained-benchmark practice; contents not read in this session. Source: [DOI 10.1016/j.swevo.2018.10.002](https://doi.org/10.1016/j.swevo.2018.10.002)

#### CMA-ES-specific constraint handling (peer-reviewed)
- **Arnold & Hansen (2012)**, "A (1+1)-CMA-ES for constrained optimisation", GECCO 2012. Approximates constraint-boundary normals by accumulating steps that violate each constraint, then reduces mutation variance in those directions (active covariance update). The strategy can approach the boundary without losing its ability to move tangentially along it. Source: [DOI 10.1145/2330163.2330207](https://doi.org/10.1145/2330163.2330207); [HAL](https://hal.archives-ouvertes.fr/hal-00696268)
- **Atamna, Auger & Hansen (2016)**, "Augmented Lagrangian Constraint Handling for CMA-ES: Case of a Single Linear Constraint". An adaptive augmented-Lagrangian CMA-ES with linear convergence observed on linearly constrained convex quadratic and ill-conditioned functions. Source: [HAL hal-01390386](https://hal.archives-ouvertes.fr/hal-01390386)
- **Dufossé & Hansen (2021)**, "Augmented Lagrangian, penalty techniques and surrogate modeling for constrained optimization with CMA-ES", GECCO 2021.
  - Studies AL-(μ/μw,λ)-CMA-ES on problems with up to 28 constraints.
  - Shows that surrogate modelling of the constraints overcomes some difficulties.
  - Note: Crossref lists the authors as Dufossé and Hansen. A search snippet attributing it to Atamna is wrong.
  - Sources: [DOI 10.1145/3449639.3459340](https://doi.org/10.1145/3449639.3459340); [HAL](https://hal-udl.archives-ouvertes.fr/INRIA/hal-03196365v1)
- **pycma `ConstrainedFitnessAL`**: an off-the-shelf augmented Lagrangian wrapper.
  - Fitness is f(x) + Σ(λᵢgᵢ + μᵢgᵢ²/2) with adapted λ and μ, and `update()` is called each iteration.
  - It tracks the best feasible solution separately and has a `find_feasible()` helper.
  - Source: [pycma docs](https://cma-es.github.io/apidocs-pycma/cma.constraints_handler.ConstrainedFitnessAL.html)

#### Used in policy search
- Feasibility-first elite selection in CEM: CCE ([Wen & Topcu 2018](https://papers.NeurIPS.cc/paper_files/paper/2018/file/34ffeb359a192eb8174b6854643cc046-Paper.pdf)) and RCE ([Liu et al. 2020](https://ar5iv.labs.arxiv.org/html/2010.07968)).
- Stochastic ranking plus per-individual Lagrange multipliers in ERL: ECRL ([Hu et al. 2023](https://ar5iv.labs.arxiv.org/html/2304.09869)).
- Augmented Lagrangian CMA-ES and "violation avoidance" CMA-ES as safe-optimisation baselines, on synthetic functions only: [Uchida et al. 2024](https://arxiv.org/html/2405.10534v1).
- An ES with sufficient decrease for stochastic constraints in RL: [Diouane et al. 2022](https://arxiv.org/abs/2202.10464).

#### Chance-constrained / probability-of-violation optimisation in EC (peer-reviewed unless noted)
- **Loughlin & Ranjithan**: chance-constrained GA for air-quality management (ASCE, 2001). An early sampling-based chance-constrained GA. Source: [DOI 10.1061/40569(2001)50](https://doi.org/10.1061/40569(2001)50)
- **Deb, Gupta, Daum, Branke (2009)**, "Reliability-Based Optimization Using Evolutionary Algorithms", *IEEE TEVC*. Optimisation subject to a bound on failure probability. Source: [DOI 10.1109/TEVC.2009.2014361](https://doi.org/10.1109/TEVC.2009.2014361)
- **Xie, Harper, Assimi, Neumann (2019)**, "Evolutionary algorithms for the chance-constrained knapsack problem", GECCO 2019. Source: [DOI 10.1145/3321707.3321869](https://doi.org/10.1145/3321707.3321869)
- **Neumann & Witt**, "Runtime Analysis of Single- and Multi-Objective EAs for Chance Constrained Optimization Problems with Normally Distributed Random Variables", IJCAI 2022 (later in *Evolutionary Computation* 2025 per search results). Source: [IJCAI 2022](https://www.ijcai.org/proceedings/2022/665)
- Summary of the field: chance constraints require constraints with stochastic components to be violated only with small probability.
  - Early EC work used simulation/sampling.
  - Recent work uses tail-bound inequalities (Chebyshev, Chernoff) and bi- or tri-objective reformulations, e.g., "3-Objective Pareto Optimization for Problems with Chance Constraints" (GECCO 2023).
  - Applications are mostly combinatorial (knapsack, submodular, mine scheduling).
  - Sources: [search summary of Neumann-group work, RIKEN AIP talk page](https://aip.riken.jp/?p=20295); [arXiv 2404.08219](https://arxiv.org/pdf/2404.08219)
- On the RL side, the RLC 2026 ROSARL successor explicitly targets "maximally safe RL": maximise the probability of satisfying the safety condition, rather than meet an expected-cost budget. Source: [RLJ 2026 Paper96 PDF](https://rlj.cs.umass.edu/2026/papers/Paper96.pdf)
- ConstrainedZero (IJCAI 2024) handles chance-constrained POMDPs by learning a failure-probability head, with adaptive conformal inference on the threshold (non-evolutionary). Source: [IJCAI 2024](https://www.ijcai.org/proceedings/2024/746)

### Inferences
- **The penalty problem in EC terms.**
  - A fixed reward penalty is a static penalty function, and PPO-Lagrangian is a dynamic/adaptive penalty.
  - Feasibility rules, stochastic ranking and ε-constraint are the EC answers to "no penalty coefficient". All three only need to *order* candidates, so they combine naturally with rank-based optimisers (CEM, CMA-ES, NES/OpenAI-ES with fitness shaping).
  - **Defensible baselines** inside the EC family:
    1. fixed-penalty CMA-ES/CEM at several λ;
    2. AL-CMA-ES via pycma `ConstrainedFitnessAL`, the "Lagrangian" analogue;
    3. feasibility-rule CEM/CMA-ES, i.e., CCE;
    4. stochastic-ranking CMA-ES;
    5. ε-constrained CMA-ES with a decreasing ε schedule.
- **When feasibility rules beat penalties.** In a single-objective EC setting they are expected to beat penalties when the scale of the reward/cost trade-off is unknown or shifts during learning, because rules are scale-free. They are expected to lose when the feasible region is disconnected or narrow, through premature convergence (Mezura-Montes & Coello). Walker2d's "health" constraint (staying upright) probably makes infeasible regions large early in training. The ε-constrained or stochastic-ranking variants, which keep some near-feasible solutions, are therefore worth testing against pure Deb rules. This is a hypothesis, not a finding.
- **Noise.** Episodic cost in RL is a random variable, so feasibility is estimated from a finite number of rollouts.
  - Hard feasibility rules can misclassify candidates under noise. Stochastic ranking's randomness, or an ε-tolerance set from a confidence interval (e.g., Wilson/Clopper–Pearson on violation counts), would be natural ways to make the rules noise-aware.
  - I found no EC paper that does this specifically for RL policies. SNES and LyEvO use statistical model checking for the same purpose.
- **Chance constraints.** If the cost is the indicator of first violation and the episode terminates on violation, then E[episodic cost] = P(violation per episode). With that design, "probability of violation ≤ δ" is the CMDP expected-cost constraint and needs no separate machinery. The design matches ROSARL's termination-on-violation setting (my derivation).

### Gaps
- The exact Pf value recommended by Runarsson & Yao (commonly cited as 0.45) could not be confirmed from the primary PDF (the link redirected).
- I did not find quantitative CEC competition rankings (e.g., which methods won CEC 2006/2010/2017) in a primary source during this session. These rankings are often cited but are unverified here.
- No study was found that benchmarks feasibility rules, stochastic ranking, ε-constraint and AL-CMA-ES against each other on RL policy search.

---

## Q3. Multi-objective formulations: return and cost as separate objectives, MORL vs constrained RL, and prior "safety dial" / Pareto-front work

### Takeaway
Treating reward and cost as two objectives and returning a front of policies is **not new in RL**:
- DeepMind's LP3 (CoRL 2021) learns preference distributions to produce sets of constraint-satisfying policies.
- CDT (ICML 2023) and CCPO (NeurIPS 2023) train one policy conditioned on the cost threshold, giving a deploy-time "dial".
- Spoor et al. (preprint 2025/26) build empirical return–cost Pareto frontiers by sweeping fixed Lagrange multipliers.
- The GECCO'22 safe-mutation paper already plots "Pareto frontiers".

What I did **not** find is an *evolutionary* multi-objective method (NSGA-II, MO-CMA-ES, MOEA/D) that produces a reward-vs-violation-probability front on Safety-Gym(nasium) and compares it with λ-swept PPO-Lagrangian. The EC constraint-handling literature also warns that "constraints as objectives" is an inefficient way to *solve* a single constrained problem, even though it is the natural way to *map* the trade-off.

### Cited Findings
- **LP3 / Constrained MORL framework**, Huang, Abdolmaleki et al. (DeepMind), CoRL 2021, PMLR v164 (peer-reviewed).
  - Treats reward and costs as separate objectives and learns which preference over them yields constraint satisfaction.
  - Lagrangian relaxation is the special case "linear scalarization + single learned preference".
  - LP3[MO-MPO-D] learns a *distribution* over preferences, giving "a set of constraint-satisfying policies, useful for when we do not know the exact constraint a priori".
  - Linear scalarization "fundamentally cannot find solutions on concave Pareto fronts"; on humanoid walk the front was concave, and LS-based (Lagrangian) training could not reach those points while MO-MPO could.
  - Tasks: DM Control humanoid walk/run and Safety Gym point goal/button/push.
  - Source: [PMLR PDF](https://proceedings.mlr.press/v164/huang22a/huang22a.pdf)
- **CCPO**, Yao, Liu, Cen, Zhu, Yu, Zhang, Zhao (NeurIPS 2023). Learns a single policy conditioned on the constraint threshold, with "zero-shot adaptation capability to different constraint thresholds". This is effectively a learned safety dial. Source: [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/29906cbd165b78991da2c4dbabc2a04b-Abstract.html); [arXiv 2310.03718](https://arxiv.org/abs/2310.03718)
- **Constrained Decision Transformer (CDT)**, Liu et al. (ICML 2023). Offline safe RL "from a novel multi-objective optimization perspective". It "can dynamically adjust the trade-offs during deployment" with zero-shot adaptation to different cost thresholds. Source: [arXiv 2302.07351](https://arxiv.org/abs/2302.07351)
- **Spoor, Serra-Gómez, Plaat, Moerland**, "Towards a Practical Understanding of Lagrangian Methods in Safe RL" (arXiv Oct 2025, revised Mar 2026; preprint). Shows "the highly sensitive nature of λ" and builds empirical return–cost Pareto frontiers to compare automatic multiplier updates against the best fixed λ. Source: [arXiv 2510.17564](https://arxiv.org/abs/2510.17564)
- **C-MORL**, Liu et al. (ICLR 2025). The reverse direction: uses *constrained* policy optimisation ("maximize one objective while constraining the other objectives") to fill gaps in a MORL Pareto front, scaling to 9 objectives. Source: [arXiv 2410.02236](https://arxiv.org/abs/2410.02236)
- **MO-ERL**, Callaghan, Mason, Mannion, *Neurocomputing* 636 (Jul 2025). The first multi-objective adaptation of ERL. Reports up to 62.71% higher hypervolume on MO-MuJoCo tasks. Not safety-specific. Source: [University of Galway record](https://research.universityofgalway.ie/en/publications/extending-evolution-guided-policy-gradient-learning-into-the-mult/)
- **PGMORL**, Xu et al., ICML 2020 (PMLR v119). A prediction-guided, population-based MORL method for MuJoCo continuous control. Existence confirmed via the PMLR PDF link; algorithm details are from prior knowledge and were not re-read. Source: [PMLR](https://proceedings.mlr.press/v119/xu20h/xu20h.pdf)
- **Safe and Balanced**, Gu et al. (arXiv May 2024; preprint). A primal-based, *non-evolutionary* framework for constrained multi-objective RL. Source: [arXiv 2405.16390](https://arxiv.org/abs/2405.16390)
- **Safety-Informed Mutations** (GECCO'22 Companion) reports results as a "Pareto frontier at convergence" of cost against reward for PPO, SM-PPO, IPO, CPO and L-PPO. Source: [PDF](https://emarche.github.io/assets/pdf/GECCO2022_safemut.pdf)
- **Zero duality gap.** Paternain, Chamon, Calvo-Fullana & Ribeiro (arXiv Oct 2019; NeurIPS 2019 per common citation, venue not re-verified) argue that constrained RL has zero duality gap, which justifies primal-dual (Lagrangian) methods. Source: [arXiv 1910.13393](https://arxiv.org/abs/1910.13393)
- **EC warning.** Runarsson & Yao (2003, 2005) and Mezura-Montes & Coello (2011) report that Pareto-based "constraints as objectives" is generally *not* the most efficient way to solve a constrained problem. Source: [TSMC-C 2005](https://doi.org/10.1109/TSMCC.2004.841906); [SWEVO 2011](https://doi.org/10.1016/j.swevo.2011.10.001)

### Inferences
- **Reconciling zero duality gap with "Lagrangian misses concave regions".** The zero-gap result is over the convex set of (stochastic) policies/occupancy measures. A fixed neural parametrisation trained by gradient methods, and especially a population of deterministic policies, yields an achievable (return, cost) set that need not be convex. Huang et al. demonstrate empirically that linear scalarization misses concave front regions. Order-based EC methods (ε-constraint sweeps, feasibility rules at different thresholds, NSGA-II) do not depend on convexity. This is a defensible argument for an EC "dial" (my synthesis).
- **What a "safety dial" contribution should look like.** A one-run front from an MOEA, or a sweep of feasibility-rule CEM/CMA-ES over thresholds δ, with **probability of violation** on one axis. It would be compared against (i) PPO-Lag swept over cost limits (OmniSafe) and (ii) a fixed-λ penalty sweep, at matched simulation steps, with hypervolume and training-time cost rate reported.
  - This would differ from CCPO/CDT, which give one conditioned policy rather than a front, and from LP3, which is gradient-based MO-MPO.
  - "Avoiding choosing a penalty weight" is then a well-defined claim: the user chooses δ (an interpretable probability) rather than λ.
- Given the Runarsson & Yao search-bias result, a sensible design separates the two jobs: use feasibility/ε ranking to *solve* each threshold, and use MO selection only to *map* the front. Alternatively, report both and show the trade-off.

### Gaps
- No evolutionary multi-objective safe-RL paper (NSGA-II/MO-CMA-ES/MOEA/D on return against cost or violation probability on Safety-Gym(nasium)) was found.
- "NeuroAction" (*Scientific Reports* 2026), a multi-objective neuroevolution method for autonomous driving, appeared in search results, but the article page redirected to a login. Whether safety is one of its objectives is **unverified**: [Nature link](https://www.nature.com/articles/s41598-026-38269-1).

---

## Q4. How do evolutionary safe-RL methods compare empirically with PPO-Lagrangian/CPO on Safety-Gym/Safety-Gymnasium, including training-time violations?

### Takeaway
Direct evidence is thin. The Safety-Gym paper set the reference protocol: Lagrangian methods enforce constraints more reliably than CPO, and *cost rate over all of training* is the safe-exploration metric. Evolutionary or population methods have been compared with Lagrangian/CPO baselines only in the following cases:
- at convergence, as a Pareto plot, on DoggoGoal1 (SM-PPO, an ERL hybrid);
- against RCPO/IPO on MuJoCo torque constraints (ECRL);
- against CPO/TRPO on a toy navigation task (CCE);
- with an evolutionary *outer loop around* PPO-Lag/CPO (AutoCost, which is expensive);
- for training-time violations, only in the MPC planner RCE, against a penalty CEM.

No paper was found that measures a pure population-based policy search's training-time cost rate against PPO-Lag on Safety-Gymnasium.

### Cited Findings
- **Safety Gym** (Ray, Achiam, Amodei, OpenAI, Nov 2019; technical report, not peer-reviewed). Source for all of the following: [OpenAI PDF](https://cdn.openai.com/safexp-short.pdf)
  - Benchmarked PPO, TRPO, PPO-Lagrangian, TRPO-Lagrangian and CPO.
  - "Surprisingly, we find that CPO performs poorly on Safety Gym environments by comparison to Lagrangian methods."
  - "approximation errors in CPO prevent it from fully satisfying constraints on virtually all of these environments … By contrast, Lagrangian methods more-or-less reliably enforce constraints."
  - Proposed **cost rate** (average cost over the entirety of training) as the safe-exploration regret measure. It recommends regret measured over "all of the agent's actual experience (as opposed to, say, only experiences from separate test behavior)".
  - Protocol: (256,256) MLPs, cost limit d=25, episodes of 1000 steps. Training was 10^7 steps (Point/Car) or 10^8 steps (Doggo), with 3 seeds.
- **Safety-Gymnasium** (Ji et al., NeurIPS 2023 Datasets & Benchmarks) adds single- and multi-agent and vision tasks, plus the SafePO library of 16 algorithms. Source: [arXiv 2310.12567](https://arxiv.org/abs/2310.12567)
- **OmniSafe** (Ji et al., arXiv May 2023) is the infrastructure the student already uses. Its JMLR version was not verified this session. Source: [arXiv 2305.09304](https://arxiv.org/abs/2305.09304)
- **PID Lagrangian** (Stooke et al., ICML 2020) targets the oscillation and overshoot of standard Lagrangian updates. Source: [PMLR](http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf)
- **Lagrangian sensitivity** (Spoor et al., preprint): λ is highly sensitive, and automatic updates need evaluation against task-specific constraint geometry. Source: [arXiv 2510.17564](https://arxiv.org/abs/2510.17564)
- **SM-PPO vs L-PPO/CPO/IPO** on DoggoGoal1, at convergence only (see Q1). Source: [PDF](https://emarche.github.io/assets/pdf/GECCO2022_safemut.pdf)
- **AutoCost**: evolution plus PPO-Lag/CPO reaches zero violation at convergence but costs 54.1 GPU-hours per evolution run (see Q1). Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2301.10339)
- **RCE vs penalty-CEM**: about 5–15× fewer violations in the first 10k steps (see Q1). Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.07968)
- **ECRL vs RCPO/IPO** on MuJoCo torque constraints (see Q1). Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2304.09869)
- **CCE vs CPO**: CPO "leaves the feasible region rapidly" even from a feasible start, whereas CCE "regains feasibility much faster" (toy task). Source: [Wen & Topcu PDF](https://papers.NeurIPS.cc/paper_files/paper/2018/file/34ffeb359a192eb8174b6854643cc046-Paper.pdf)
- **ROSARL successor (RLC 2026)**: TRPO-ValueRange against TRPO-Lagrangian, Sauté-TRPO, CPPOPID and P3O (OmniSafe) on PointGoal1, PointPush1 and Humanoid Velocity.
  - PointGoal1 mean cost: 0.08 ± 0.05 for ValueRange against 0.62 ± 0.07 for TRPO-Lag.
  - The trade-off is lower return: 1.87 ± 1.60 against 10.17 ± 0.77.
  - Source: [RLJ 2026 Paper96 PDF](https://rlj.cs.umass.edu/2026/papers/Paper96.pdf)

### Inferences
- **Training-time cost.** Population methods evaluate *every* candidate in the environment. Each generation can therefore incur many violations, and naive ES may have a *worse* training-time cost rate than PPO-Lag even if its final policy is safer.
  - This must be measured, not assumed.
  - A fair metric is Safety-Gym's cost rate computed over all candidate rollouts, at equal environment steps.
- **What "safe exploration" means here.** The student trains in simulation, so training-time violations are a cost *metric*, not real-world harm. The paper should say this explicitly. It could then report two things separately: (a) cost rate over all training rollouts, and (b) final-policy violation probability with confidence intervals.
- **Fair comparison.** OmniSafe PPO-Lag and CPO should run at the same step budget as the ES variant. CPO is a weak but standard baseline (Ray et al.), and PPO-Lag/CPPOPID are the strong ones. The ES side should include a fixed-penalty sweep, to show the "hand-tuned penalty" baseline is not a strawman.

### Gaps
- No published numbers were found for OpenAI-ES/CMA-ES/CEM/GA policy search on any Safety-Gymnasium velocity task (including SafetyWalker2dVelocity-v1), either at convergence or during training.
- No evolutionary safe-RL paper was found that reports Safety-Gym cost rate.

---

## Q5. How could ROSARL's minmax-penalty idea combine with evolution, and has anyone done it?

### Takeaway
I found **no** work combining ROSARL (or its RLC 2026 successor) with evolutionary or black-box policy search.

**Venue discrepancy (important):**
- The brief says ROSARL appeared at RLC 2024. The arXiv version (May 2023) has no venue.
- The RLJ 2024 issue page contains no paper by Nangue Tasse.
- The same authors and algorithm appear as "An Unreasonably Simple Approach to Safe RL", *Reinforcement Learning Journal* vol. 7 (2026), presented at **RLC 2026** (Montréal, 15–17 Aug 2026), with code in a repo named "rosarl".

The student should confirm the correct citation with the supervisor. Conceptually, in a rank-based EA, ROSARL's "penalty larger than the value range" makes every violating episode rank below every safe one, which is almost exactly Deb's feasibility rule. The differences lie in how infeasible candidates are ordered and how multiple stochastic rollouts are aggregated.

### Cited Findings
- **ROSARL** (arXiv 2306.00035, 31 May 2023; Nangue Tasse, Love, Nemecek, James, Rosman; preprint, no venue listed).
  - Seeks "the upper bound of rewards at unsafe states whose optimal policies minimise the probability of reaching those unsafe states, irrespective of task rewards". This is the "Minmax penalty", obtained from the environment's controllability and diameter.
  - Provides a model-free algorithm that learns the penalty alongside the task policy.
  - Source: [arXiv 2306.00035](https://arxiv.org/abs/2306.00035)
- **RLC 2026 version** ("An Unreasonably Simple Approach to Safe RL", RLJ vol. 7, 2026, pre-proceedings).
  - The agent tracks running V_MIN and V_MAX, which are updated from both observed rewards and V(s_t), and sets **R_unsafe = V_MIN − V_MAX**.
  - On a violation (z_t > 0), the reward is replaced by R_unsafe and the episode terminates.
  - It claims "no new hyperparameters", works with any base RL algorithm, and targets "maximally safe" policies, i.e., those maximising the probability of satisfying the safety condition (almost-sure safety when attainable).
  - Evaluated on Lava Gridworld (slip probability), Safety-Gymnasium PointGoal, PointPush, Pillar and Humanoid Velocity, with OmniSafe baselines.
  - Sources: [RLJ page](https://rlj.cs.umass.edu/2026/papers/Paper96.html); [PDF](https://rlj.cs.umass.edu/2026/papers/Paper96.pdf)
- The RLJ 2024 issue page lists no paper with "Tasse" among its authors (checked by text search of the page). Source: [RLJ 2024 issue](https://rlj.cs.umass.edu/2024/2024issue.html)
- Searches for ROSARL/minmax penalty combined with evolution strategies, GA or neuroevolution returned only ROSARL itself and unrelated work. Source: [arXiv 2306.00035](https://arxiv.org/abs/2306.00035)

### Inferences (proposals, not findings)
- **Black-box analogue of the value-range penalty.** ES/CEM/CMA-ES have no value function, but they do observe episodic returns. Two options follow:
  1. Keep running R_min and R_max over all observed (pre-violation) returns, and give any violating episode the fitness R_min − (R_max − R_min). That is a "return-range penalty" with no tuned λ.
  2. Apply ROSARL's per-step R_unsafe to the reward stream inside each rollout, then sum.
- **Equivalence claim, testable and possibly publishable.** Rank-based EAs are invariant to monotone fitness transforms. With single-episode evaluation and termination on violation, any penalty below every safe return makes all violators rank below all safe candidates, which is exactly Deb's first rule. ROSARL-in-ES then differs from Deb/CCE only in its secondary ordering:
  - ROSARL orders violators by pre-violation return. Because of termination, violating later usually means more return, so this roughly encodes "time-to-violation".
  - Deb/CCE orders violators by violation magnitude.
  
  This gives a clean bridge between the supervisor's RL work and the EC constraint-handling literature.
- **Where the equivalence breaks.** With *n* rollouts per candidate and *mean* fitness, ROSARL-style fitness becomes (1−p)·E[R | safe] + p·R_unsafe, a probability-weighted penalty. It is not lexicographic in p unless |R_unsafe| is large relative to return differences. Comparing "mean-fitness ROSARL-ES" against "lexicographic (p̂, return) ranking" against "stochastic ranking" is a natural ablation that targets the brief's "probability of safety violations".
- **Hybrid option.** An ERL hybrid (ECRL-style) with a ROSARL-penalised gradient learner and a feasibility-ranked population would also be novel. It is heavier to engineer, however, and riskier in 4 months than a pure ES study.

### Gaps
- I could not independently confirm whether any ROSARL version appeared at RLC 2024 (dblp lookups failed). The evidence above points to RLC 2026.
- No citing papers of ROSARL that use evolution were found. A citation-graph check on Google Scholar or Semantic Scholar was not performed.

---

## Q6. What gaps remain that a 4-month Honours project could credibly fill?

### Takeaway
The most defensible and feasible contribution is a careful, well-baselined empirical study. Pure rank-based evolutionary policy search (CEM/CMA-ES/OpenAI-ES) on SafetyWalker2dVelocity-v1 would compare EC constraint handling (feasibility rules, which are CCE-style; stochastic ranking; ε-constraint; AL-CMA-ES) and a ROSARL-style return-range penalty against fixed-penalty sweeps and OmniSafe PPO-Lag/CPPOPID/CPO. Reports would cover final violation probability with CIs, training-time cost rate, and a reward vs P(violation) "dial" front.

What is novel:
1. Applying and benchmarking EC constraint handling for policy search on Safety-Gymnasium against PPO-Lag (not found in the literature).
2. The ROSARL ↔ feasibility-rule connection for rank-based optimisers (not found).
3. An evolutionary reward vs violation-probability front compared with λ-swept Lagrangian fronts (not found for EC).

What is not novel: feasibility-first CEM itself (CCE 2018, RCE 2020) and the general idea of a safety dial (CCPO, CDT, LP3).

### Cited Findings (evidence for the gaps)
- **Evolutionary safe-RL papers use non-standard tasks or hybrids.**
  - ECRL: MuJoCo torque constraints, against RCPO/IPO. Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2304.09869)
  - SM-PPO: ERL hybrid, convergence-only Pareto plot. Source: [PDF](https://emarche.github.io/assets/pdf/GECCO2022_safemut.pdf)
  - CCE: toy navigation, 2×30 MLP. Source: [PDF](https://papers.NeurIPS.cc/paper_files/paper/2018/file/34ffeb359a192eb8174b6854643cc046-Paper.pdf)
  - Safe CMA-ES: synthetic functions only, with "realistic problems" left as future work. Source: [arXiv HTML](https://arxiv.org/html/2405.10534v1)
- **Evolutionary outer loops around RL are expensive**: AutoCost took 54.1 h per evolution run. Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2301.10339)
- **Lagrangian methods are λ-sensitive and do not reach zero violation even at a zero cost limit.** Source: [Spoor et al. arXiv 2510.17564](https://arxiv.org/abs/2510.17564); [AutoCost arXiv 2301.10339](https://arxiv.org/abs/2301.10339)
- **Linear scalarization (Lagrangian) cannot reach concave front regions**, whereas MO/ordering methods can. Source: [Huang et al. CoRL 2021](https://proceedings.mlr.press/v164/huang22a/huang22a.pdf)
- **Feasibility rules risk premature convergence**, and ε-constraint and stochastic ranking mitigate this. Source: [Mezura-Montes & Coello 2011](https://doi.org/10.1016/j.swevo.2011.10.001)
- **ES is cheap on MuJoCo with parallel CPUs**, and linear policies are competitive. Source: [Salimans et al. 2017](https://arxiv.org/abs/1703.03864); [Mania et al. 2018](https://arxiv.org/abs/1803.07055)
- **Safe-RL evaluation should count all training experience** (cost rate). Source: [Ray et al. 2019](https://cdn.openai.com/safexp-short.pdf)

### Inferences (candidate project shapes, ranked by risk)
1. **Lowest risk.** "Constraint handling, not penalty tuning, for evolutionary safe policy search."
   - Methods: CEM and/or CMA-ES over small MLP or linear policies on SafetyWalker2dVelocity-v1 (health and speed rules).
   - Arms:
     - fixed-penalty sweep;
     - AL-CMA-ES (pycma);
     - Deb/CCE feasibility-first (the student's Safe CEM, credited to Wen & Topcu);
     - stochastic ranking (Pf sweep);
     - ε-constraint;
     - ROSARL-style return-range penalty.
   - Baselines: OmniSafe PPO-Lag and CPPOPID at matched environment steps.
   - Metrics: final P(violation) with Wilson CIs over many evaluation episodes, return, training-time cost rate, and wall-clock time.
   - This fits 4 months given the student's existing pipeline and throughput.
2. **Medium risk.** Add the "dial": sweep δ (the target violation probability) for the feasibility-ranked ES, or run NSGA-II/MO-CMA-ES on (return, −p̂). Compare the hypervolume of the resulting front with a PPO-Lag cost-limit sweep and a fixed-λ penalty sweep at equal total compute.
3. **Higher risk.** A noise-aware feasibility rule, where feasibility is decided by a confidence bound on p̂ or the ε level is tied to the CI width, or an ERL hybrid with a ROSARL-penalised learner.
- **Novelty wording.** Avoid claiming feasibility-first CEM as new. Frame the contribution as:
  - (i) the first systematic EC constraint-handling benchmark for safe policy search on Safety-Gymnasium against Lagrangian deep RL;
  - (ii) a demonstration that ROSARL's penalty reduces to a feasibility rule under rank-based selection, and where it does not;
  - (iii) an interpretable δ-dial in place of λ.

### Gaps
- I did not check whether anyone has published ES/CEM results specifically on SafetyWalker2dVelocity-v1, and I found none in searches.
- Whether GECCO 2026 (July 2026) contained relevant evolutionary safe-RL papers was not checked; its proceedings were not searched.
- "Hand-tuned penalties" vs EC constraint handling has not been compared on RL in the literature found, so the expected effect sizes are unknown.
