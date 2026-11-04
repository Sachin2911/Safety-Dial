# Contribution positioning for GECCO

## What is already established in the closest literature

### Takeaway
Evolving a compact controller on world-model features, combining policy networks with planning, and preferring feasible solutions without a penalty coefficient all have direct precedents. The project needs a specific new result about safe adaptation, its costs, or its failure conditions rather than novelty from the component list.

### Cited Findings
- World Models trains a large visual and recurrent world model, then uses CMA-ES to optimize a small controller. Its CarRacing controller has 867 parameters. This directly precedes the general idea of evolutionary optimization of a compact controller supported by learned world representations. It does not itself establish safety adaptation around an already competent frozen planner. [Author project and paper](https://worldmodels.github.io/).
- POPLIN combines policy networks with learned-model online planning. POPLIN-A initializes action sequences from a policy and optimizes action perturbations; POPLIN-P optimizes policy-parameter perturbations. The model is updated in its training procedure. These are precedents for learned policy/planner interfaces, with a different training and safety setting. The accessible preprint is titled *Exploring Model-based Planning with Policy Networks*. [Primary full text](https://arxiv.org/html/1906.08649).
- Deb's constraint-handling method avoids a penalty parameter by comparing feasibility, objective values among feasible candidates, and violation among infeasible candidates. This is a 2000 publication, not a new rule introduced by this project. The publisher abstract and repository-indexed manuscript were retrieved; direct publisher/PDF fetching returned 403. [Publisher record](https://www.sciencedirect.com/science/article/abs/pii/S0045782599003898), [author manuscript repository](https://repository.ias.ac.in/9407/1/310.pdf).
- Safe Neural Evolutionary Strategies, an implementation of PSyCo, combines evolutionary policy optimization with Bayesian model checking; estimated confidence in satisfaction adjusts the return/cost weighting. It explicitly studies probabilistic constraints. Its own limitations discuss non-stationarity of estimates and absent convergence proofs. This rules out treating evolutionary safety or uncertainty-aware risk balancing by itself as the novelty. [Primary full text](https://arxiv.org/html/2005.03898v2).
- CPO already separates a reward from constraints and supplies an optimization algorithm with near-constraint-satisfaction results under its assumptions. Avoiding a manually chosen reward penalty is therefore not unique to evolution. [ICML primary proceedings](https://proceedings.mlr.press/v70/achiam17a.html).

### Inferences
- A contribution can be an informative empirical study or a useful application of existing algorithms. It need not invent another optimizer. However, a component combination needs a clear scientific question and evidence that teaches something beyond its implementation.
- A strong opening would state the particular adaptation setting: a released frozen world-model system, new explicit safety requirements, no backbone retraining, compact controller intervention, and total acquisition cost including selection. Then identify what the study newly demonstrates in that setting.

### Gaps
- This focused check does not establish that nobody has studied precisely this combination. A final novelty statement requires a focused comparison of the actual final method with the closest papers, including recent frozen-policy adaptation work.

## What claim is feasible from the current evidence

### Takeaway
The most defensible direction is a focused empirical question: when does evolutionary adaptation add reliable safe-completion benefit beyond the frozen planner and an untrained or randomly searched adapter? Reduced manual tuning can be a secondary claim if measured prospectively. Risk calibration is a larger, separate study.

### Cited Findings
- The matched TwoRoom study reports CMA 44/216, random search 37/216, and PPO 20/216 safe completions; both principal uncertainty intervals include zero. The separate mechanism study slightly favors random search. [Matched comparison](../../../../SafeLeWM/results/14-matched-ppo.md), [mechanism study](../../../../SafeLeWM/results/12-adapter-mechanism.md).
- PushT qualification passes baseline/hazard admission but fails useful adapter authority; no optimizer training followed. [Qualification](../../../../SafeLeWM/results/16-pusht-validated.md).
- Block Plucking Search 2 reports 14/48 sustained successes for CMA, 16/48 for an untrained selector, and 9/48 native. This does not isolate the cause of unsuccessful learning. [Pooled results](../../../../SafeBlock/Search2Pooled.md).
- The safety-dial library reports missing targets and empirical risk-limit exceedances rather than calibrated risk. [Dial results](../../../../SafeLeWM/results/17-safety-dial-library.md).

### Inferences
- Recommended candidate claim, contingent on confirmation: a fixed compact-adaptation recipe improves safe completion around frozen systems under a bounded interaction budget, with an identified explanation of when search helps and when interface changes alone account for gains.
- A publishable explanation of failure would require prospective predictions and controlled tests. The existing negative results alone do not prove the interface is the bottleneck, nor that the optimizer is ineffective in general.
- An empirical route can report no CMA superiority honestly, but then the contribution must be the demonstrated mechanism, boundary condition, evaluation insight, or cheaper useful adaptation. Merely failing to reject a difference is insufficient for either equivalence or a general negative conclusion.
- A method route centered on risk-budget ranking would require actual feasibility-based training, an appropriate constrained-learning comparison, a penalty-selection cost comparison, and fresh risk assessment. The existing fixed-utility PPO and post-hoc library cannot stand in for these.

### Gaps
- Reliable added value of learning over untrained adapters, a replicated manipulation benefit, and prospectively measured tuning reduction remain unestablished.

## How to make the contribution convincing without expanding indefinitely

### Takeaway
Choose one principal claim and one primary manipulation task, use TwoRoom for mechanism analysis, and make each comparison answer a reviewer objection. A smaller complete study is more coherent than making every originally planned environment succeed.

### Cited Findings
- The repository authority explicitly distinguishes learned benefit, reduced tuning, optimizer superiority and same-weight transfer; Block Plucking is intended as the main study, and failed qualification gates do not authorize replacement studies. [Current direction](../../../../firstPrinciples.md).
- The current manuscript already states that the implemented LeWM and Block Plucking adapters differ and does not claim identical architecture or shared weights. [Working manuscript](../../honours-gecco/world-model-adapters.tex).

### Inferences
- Essential controls for an adaptation claim: frozen native, neutral/untrained, matched parameter random search, a credible ordinary-learning configuration on the same interface, and an applicable fixed reference if claiming reduced hand design. Prior-work citations are not a requirement to implement every published algorithm.
- For an evolutionary-superiority claim, compare repeat distributions and budget curves rather than only a selected final score. For a practical adaptation claim, demonstrate a meaningful gain and cost tradeoff even if another optimizer ties.
- For a reduced-tuning claim, declare which decisions are eliminated, freeze the recipe before fresh conditions, and count all development/selection evaluations. Do not retroactively estimate person-hours or call the safety definition itself learned.
- Include one causal test of the proposed mechanism. Examples to select prospectively include changing adapter authority while preserving information, or removing a particular safety signal while retraining. Do not interpret a bundle of fixes as a causal ablation.
- Decide an outcome-dependent writing route before final testing: useful learned recovery yields an adaptation paper; a controlled, explanatory negative result yields an empirical limitations paper; inconclusive exploratory counts support a narrower workshop submission. This is strategic advice, not a prediction of acceptance or execution authorization.

### Gaps
- Sample sizes and attainable precision need to be designed from development variation and the smallest meaningful effect. There is no universal number of seeds that makes the present claims ready for publication.
