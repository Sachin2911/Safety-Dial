# Conceptual foundations of manipulation safety

Research checked 3 November 2026. This is a selective conceptual map, not a systematic review or an experiment authorization. References are primary papers or surveys written by researchers; source claims and our illustrative inferences are separated below.

## What does safety mean, and what is it distinct from?

### Takeaway

There is no single operational definition across robot learning. A useful separation is between the undesirable physical or semantic outcome, the mathematical requirement imposed on it, and the evidence supporting the requirement.

### Cited Findings

- García and Fernández's foundational survey defines safe RL broadly enough to include performance or constraint requirements during learning, deployment, or both. It explicitly observes that risk in RL does not always mean physical damage: some formulations concern variability or poor-tail outcomes of returns. Thus a paper labelled risk-sensitive is not automatically about physical robot safety. [García and Fernández, JMLR 2015](https://www.jmlr.org/papers/volume16/garcia15a/garcia15a.pdf).
- Brunke et al. connect control and RL through constraints on states and actions under uncertainty. Their review distinguishes constraint satisfaction that is encouraged, probabilistic, or guaranteed. The latter claims depend on assumptions such as dynamics structure and uncertainty bounds. This is a useful vocabulary, not a universal grading system for every safety paper. [Brunke et al., Annual Review 2022, open manuscript](https://arxiv.org/html/2108.06266v2).
- SafeManip's September 2026 revision distinguishes successful final states from safe execution over time. It evaluates low-level rollouts with temporal specifications and includes physical and semantic properties such as grasp stability, containment, contamination and release ordering. Its seven-policy, RoboCasa365/LIBERO study finds successful trajectories may still contain violations. Its per-trigger metrics count safety-relevant opportunities, addressing the fact that policies activate different numbers of obligations. This is a recent preprint, not evidence of a universal consensus. [SafeManip v3, 2026](https://arxiv.org/html/2605.12386v3).

### Inferences

- Proposed working definition, explicitly our synthesis: safe manipulation maintains stated limits on undesirable consequences while acting in a specified environment and uncertainty regime. Useful safe manipulation additionally achieves the intended task.
- Moving a fragile cup illustrates why the property matters: too much grip force can break it, too little can drop it, an unintended collision can injure someone, and premature release can violate a procedural condition despite an eventually correct final position.
- A binary simulator contact flag is an operational safety proxy. It is not automatically evidence of injury, damage or their absence. The inference from the measured predicate to the motivating harm needs explanation.
- Task success, safety and robustness should be separately named. Robust task success means maintaining success under perturbations; robust safety means maintaining safety under them. Neither implies the other without the corresponding definition.
- Motionless behaviour can avoid some violations while failing the task. Conversely, doing nothing can itself be unsafe when a supported object must be stabilised or a hazard requires action. Avoid making inactivity a universal definition of safety.

### Gaps

- No universally accepted manipulation-wide list of hazards or risk threshold emerged. Which objects, people, equipment and processes deserve protection remains application dependent.
- This review does not assess legal conformity or physical injury thresholds.

## What mathematical versions of safety are used?

### Takeaway

Average safety cost, the probability of a bad episode, always remaining safe, and obeying an ordering requirement answer different questions. Choose the intended question before choosing the learning algorithm.

### Cited Findings

- In a constrained Markov decision process, optimize task return while requiring expected cumulative auxiliary cost to remain below a budget: E[sum_t c_t] <= d, with discounting where specified. CPO is a representative algorithm. Its near-constraint-satisfaction statements concern this mathematical cost criterion and an approximate policy-update implementation; they do not mean that no sampled trajectory ever violates a physical limit. [Achiam et al., ICML 2017, Sections 4-5](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf).
- Wachi et al. classify seven constraint formulations, separating expectation-based and probability-based requirements and cumulative from instantaneous constraints. Chance-constrained trajectory safety can require P(all states satisfy their limits over the horizon) >= 1-delta. Their discussion explicitly separates problem formulation from policy optimization and distinguishes safety during training from safety after convergence. Their taxonomy is especially useful for the user's requested return to first principles. [Wachi, Shen and Sui, IJCAI 2024](https://arxiv.org/html/2402.02025v2).
- In control barrier function theory, safety is forward invariance: every trajectory starting in a specified safe set stays within it. The conditions concern a dynamical model and existence of an admissible controller. Constraint satisfaction is therefore conditional on the theorem's model, regularity and feasibility assumptions. [Ames et al., ECC 2019, Definition 1](https://coogan.ece.gatech.edu/papers/pdf/amesecc19.pdf).
- Hamilton-Jacobi reachability asks which states allow avoiding a failure set despite admissible disturbances. Fisac et al. combine a learning controller with a minimally intervening safety controller and refine disturbance estimates from data. Their analysis distinguishes local probabilistic confidence from a global safety guarantee; a convenient local rule is explicitly weaker. [Fisac et al., accepted TAC manuscript, 2018 revision](https://arxiv.org/html/1705.01292v3).
- Temporal logic can express requirements involving a sequence, such as maintaining grasp until placement or waiting until an object is inside before releasing it. SafeManip instantiates these as predicates over finite traces and monitors their satisfaction. Specification and monitoring do not themselves enforce the behaviour. [SafeManip v3](https://arxiv.org/html/2605.12386v3).
- CVaR is a risk measure targeting the high-cost tail of a distribution rather than only its mean. Chow et al. study risk-sensitive MDPs and a connection to robustness under a specified model-error budget. This helps distinguish violation frequency from severity in rare cases. [Chow et al., NeurIPS 2015](https://papers.nips.cc/paper/6014-risk-sensitive-and-robust-decision-making-a-cvar-optimization-approach.pdf).

### Inferences

Use these cup examples as illustrations, not quotations from the papers:

- **Per-step physical limit:** grip force must remain below a specified crushing threshold at every instant.
- **Episode chance constraint:** the probability of dropping or breaking the cup at any point in the whole transfer must be below a chosen bound.
- **Expected cumulative cost:** across transfers, expected accumulated force-limit exceedance must be below a budget. This need not prevent a rare severe incident.
- **Tail severity:** constrain the mean cost among the worst fraction of transfers, using a stated severity model.
- **Temporal obligation:** do not release the cup until it is supported by the destination surface.
- **Reachability/invariance:** remain in states from which available actions can continue avoiding failure. An intact cup is not necessarily recoverable if it is already slipping beyond the robot's ability to catch it.
- A per-step probability bound and a whole-episode probability bound are not interchangeable. Longer episodes provide more opportunities for violations.
- For an undiscounted indicator of the first failure in a terminated episode, expected cumulative cost can equal failure probability. Do not claim these formulations can never coincide; equality requires the appropriate event/cost construction.
- The designer still specifies a cost, predicate, admissible set, preference or label source. Evolution does not remove that requirement merely by changing selection machinery.

### Gaps

- Selecting a specific threshold or proving one formulation best for the current project is premature. These notes establish choices and consequences.
- Severity-sensitive costs are only as meaningful as their mapping to actual harm.

## When does safety apply, and what evidence supports it?

### Takeaway

Learning a safe final policy, conducting learning safely, and proving safety under assumptions are separate claims. Empirical low violation rates belong to a specified evaluation distribution and horizon.

### Cited Findings

- Safe learning includes both methods which encourage safe behaviour and methods which certify learned control. Brunke et al. discuss uncertainty in robot dynamics and distinguish certification from empirical improvements, while highlighting benchmarking limitations. [Brunke et al.](https://arxiv.org/html/2108.06266v2).
- Wachi et al. distinguish algorithms targeting requirements after convergence from algorithms intended to maintain requirements during training. This choice depends on the application. [Wachi et al.](https://arxiv.org/html/2402.02025v2).
- Safety filters can wrap an otherwise arbitrary learning method. Fisac et al.'s reachability-based framework provides an explicit example, with guarantees dependent on modelling confidence. [Fisac et al.](https://arxiv.org/html/1705.01292v3).

### Inferences

- For evolutionary learning, the distinction means evaluating both the selected controller and the candidate evaluations used to discover it when real-world training safety is claimed. Safe final offspring do not retroactively make unsafe candidate trials safe.
- Simulated candidate evaluation avoids physical harm during that simulation, but transferring the chosen controller requires evidence about the real system and modelling error.
- A confidence interval for an observed rate is not a certificate that all admissible future trajectories satisfy a property. Conversely, a theorem under stated assumptions is not unconditional evidence that the physical world satisfies those assumptions.
- Read papers by extracting: protected entity; prohibited event; mathematical risk measure and horizon; information available; training or deployment scope; human specification; empirical or formal evidence and assumptions.
- Suggested reading order for these foundations: Brunke et al. for the bridge between robotics/control/RL; Wachi et al. for formulation; SafeManip for recent manipulation-specific temporal evaluation. Ames or Fisac are later technical branches, not necessary before understanding the distinctions.

### Gaps

- This is deliberately selective rather than an exhaustive 2026 literature census. Recent preprints need separate critical appraisal before adopting their definitions or benchmarks.
