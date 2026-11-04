# Learning safety specifications and synthesizing safety mechanisms

## Can demonstrations replace manually written safety constraints?

### Takeaway

Constraints can be inferred from demonstrations, but the information and assumptions that make them identifiable must come from somewhere. This reduces some explicit programming burdens without automatically removing specification uncertainty.

### Cited Findings

- Chou, Berenson and Ozay's *Uncertainty-Aware Constraint Learning for Adaptive Safe Motion Planning from Demonstrations* (CoRL 2020, proceedings 2021) represents uncertainty over constraints consistent with demonstrations and updates it during execution. It evaluates a 7-DOF arm and a quadrotor. The paper explicitly identifies a central ambiguity: many constraints may explain the same demonstrations. Its planner provides probabilistic safety relative to its constraint belief and assumptions, including a prescribed probability. Thus it does not infer a unique universal safety definition from examples. [Paper](https://proceedings.mlr.press/v155/chou21a/chou21a.pdf).
- Lindner et al.'s *Learning Safety Constraints from Demonstrations with Unknown Rewards* (AISTATS 2024), CoCoRL, learns shared constraints from demonstrations whose reward functions can differ and be unknown. Its convex construction guarantees feasibility under its assumptions. These include safe demonstrator policies and costs linear in the selected feature representation; finite-sample feature estimates require additional treatment. The underlying safety definition is bounded expected discounted cumulative cost in a CMDP, not necessarily zero violation on every trajectory. Evaluations are gridworlds and driving simulation, rather than robotic manipulation. [Paper](https://proceedings.mlr.press/v238/lindner24a/lindner24a.pdf).

### Inferences

- For our cup illustration, demonstrations might communicate acceptable handling without specifying an exact force profile. They cannot reliably reveal an unencountered hazard unless the representation and inference assumptions supply additional information.
- An evolutionary policy search could use inferred constraints as its evaluation signal. This is a possible integration, not an evolutionary result established by these two papers.
- Distinguish the **safety definition** (what should be protected), **detector or model** (how the requirement is represented or recognised), and **policy behaviour** (how the agent acts). Learning the second or third does not mean the first ceased to exist.

### Gaps

- These sources do not establish that evolutionary optimization is better than their original optimizers, or that demonstration-based specification requires less total human work in our manipulation setting.

## Can human preferences replace numerical safety penalties?

### Takeaway

Human feedback can communicate safety through comparisons and reasons instead of an explicitly supplied reward penalty. The feedback channel, human burden and learning hyperparameters remain part of the design.

### Cited Findings

- Kazantzidis et al., *Learning safe behaviour via justified human preferences and hypothetical queries* (29 October 2026), augments preferences with a binary explanation that an alternative is unsafe and asks about hypothetical actions. JPAL-HA learns directly from this feedback and reports results in gridworlds and a mobile-robot task. Its objective is reducing unsafe training events, not proving they impossible. The paper assumes users can correctly recognise unsafe states and the agent can pause for queries. It explicitly retains a hyperparameter trading safety against human burden. This is a useful current example of moving specification into feedback while retaining design choices, not a manipulation or evolutionary demonstration. [Paper](https://link.springer.com/article/10.1007/s10458-026-09770-8).

### Inferences

- Evolution could select candidate behaviours using human rankings or a learned preference model, but this would replace explicit penalty specification with feedback and model assumptions. It would not establish that no tuning remains.
- A preference for one trajectory may mix efficiency, aesthetics and safety. Feedback explicitly identifying the safety reason helps separate those dimensions; the above paper supplies a concrete example rather than a universal guarantee.

### Gaps

- This focused search did not establish that preference-guided evolution reduces the total specification burden for robotic manipulation. Human interventions as a separate data source were not independently surveyed here.

## Can evolution synthesize a mechanism that enforces a safety specification?

### Takeaway

Yes, there is a direct precedent for evolutionary synthesis followed by formal verification. The evolutionary search supplies candidate controllers and certificates; the proof checks their correctness relative to an already supplied system model and specification.

### Cited Findings

- Verdier and Mazo's *Formal Synthesis of Analytic Controllers for Sampled-Data Systems via Genetic Programming* (CDC 2018) uses grammar-guided genetic programming to synthesize control Lyapunov-barrier functions and optionally controller modes, then an SMT solver to check correctness. The user supplies safe, initial and goal sets and system dynamics. The requirements are reaching a goal while staying safe, optionally remaining in the goal thereafter. The reported examples are dynamical systems up to third order. The paper states that search has no guarantee of finding a solution within a fixed number of generations; some auxiliary switching functions remained hand-synthesized. This is a genuine evolutionary safety-synthesis precedent, with modest system scope and explicit assumptions. [Full paper](https://arxiv.org/html/1812.02711v1).

### Inferences

- This offers a precise answer to “enforce”: evolution can help discover the controller or certificate used to ensure a stated property. Selection by a few successful episodes is not equivalent to verification.
- The division of labour matters: **search proposes**, **verification checks the specification**, and **the implemented controller governs execution**. Those roles can coexist but should not be attributed entirely to evolution.
- An evolved barrier does not determine which objects deserve protection. A learned detector does not by itself provide an invariant-set certificate. A selected high-performing policy does not make all candidate evaluations safe.
- For our research framing, “less manual controller or penalty design while preserving an explicit safety objective” is defensible as a question. “Safety without any supplied specification, assumptions or tuning” is not established by these sources.

### Gaps

- This paper does not demonstrate high-dimensional contact-rich robotic manipulation or an evolutionary advantage against modern learning-based controllers. We should not extrapolate its proof to an inaccurate learned world model without separately establishing the needed assumptions.
