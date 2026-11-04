# Scientific litmus for evolutionary safe manipulation

## What would establish a useful result?

### Takeaway

Use four progressively stronger gates: valid measurement and attainable safe behaviour; repeated learned benefit; confirmation against matched alternatives; and reduced retuning across new conditions. Passing one gate supports only its corresponding claim. This is a discussion proposal, not a frozen protocol, launch instruction or allocation.

### Cited Findings

- Empirical RL studies should distinguish demonstrations from tests of falsifiable hypotheses, control confounds, and account for hyperparameter selection and experimenter bias. Patterson and colleagues discuss these issues explicitly. [Empirical Design in Reinforcement Learning, JMLR 2024](https://jmlr.org/papers/v25/23-0183.html).
- Few training runs can give misleading rankings; interval estimates and explicit treatment of variability change conclusions drawn from point estimates. Hundreds of evaluation episodes cannot replace independent training runs. [Agarwal et al., NeurIPS 2021](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf).
- Ordinary policy learning can separate reward and safety constraints, so a fixed, poorly chosen penalty PPO implementation would be a weak sole comparison. CPO is a foundational constrained-policy example; its original experiments concern locomotion, not a validated insertion baseline. [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html).

### Inferences

The following are proposed design choices for our research, not results or requirements derived from those papers.

**Gate 1: the problem is measurable and the controller can solve it safely.** Reproduce the benchmark's published competence before applying the proposed safety extension. Freeze a development distribution with interpretable variations such as misalignment and friction. Check that a competent nominal controller encounters a meaningful safety-completion tradeoff under the declared requirement, and that a fixed, task-capable reference can complete safely using the same observable information and available action interface. The safety requirement must have an independent physical or benchmark justification. Do not tighten it until native happens to fail, or select a few favourable scenes after reading outcomes. A hand-designed witness proves attainability, not learnability or sample efficiency. Use an inactive controller as a metric check, not as the main performance baseline.

Also validate the measured quantity. Contact sensor force, end-effector wrench and damage are different variables. Freeze sampling rate, aggregation, transient handling, object-specific limits and monitor semantics before comparative outcomes. A rigid-body force-limit study supports a claim about that proxy, not actual breakage. Apply engineering checks to contact fidelity before an optimiser can exploit it. The benchmark researcher covers exact software options.

Do not handicap an existing force-aware baseline to manufacture a tradeoff. If the benchmark's competent fixed reference already meets the same safety/utility target across the proposed variations without retuning, this setting supplies little headroom for the reduced-tuning claim. That is a substantive negative gate, even if the evolutionary method also succeeds. Any meaningful change of scope must be proposed before a fresh experiment.

**Gate 2: learning produces useful, repeatable behaviour on development cases.** Use a bounded pilot on one insertion family. Compare learned candidates with the untrained/nominal controller, the fixed competent reference and matched parameter random search. A practical early target could be a **10 percentage point absolute gain in safe completion** over nominal, with no increase beyond a declared tolerance in violation probability. This means one more safe completion per ten attempts and is a proposed relevance margin, not a universal standard. It must be chosen before outcomes and revised prospectively if a ceiling makes it inappropriate. Report all training repeats; five development repeats are a reasonable proposed screening starting point, not evidence that five are statistically sufficient. Do not proceed from one fortunate trained policy or one demonstration trajectory.

An encouraging result here is a consistent, substantial signal that warrants confirmation. It is not yet proof of evolutionary advantage. If random search also gains, the representation is usefully searchable; evolution's special contribution remains open. If the fixed controller works but all searches fail, learning signal/search difficulty remains plausible. If no same-interface witness works, optimizer comparison is premature.

**Gate 3: independent confirmation distinguishes learned benefit from optimizer benefit.** Freeze algorithm settings, selection, data access, distributions, analysis and stopping rules before final cases. Use nominal, fixed reference, parameter random search and an ordinary constrained learner with the same policy information, action domain and declared interaction budgets. Choose a compatible learning method and reproduce its baseline during development. An adaptive-penalty constrained learner helps avoid assigning all penalty-tuning burden to the competitor.

For an optimizer attribution, keep candidate representation, evaluation distribution and safety objective/selection criterion shared where possible. If one method uses a different safety objective or additional demonstrations, the comparison is between complete methods; it does not isolate the optimizer. Safety filters and low-level protective limits must be shared, with intervention frequency and proposed versus executed actions logged. Do not remove a physical protection mechanism merely to make learning effects visible; controlled ablations belong in the simulator and would need their own protocol.

Use paired evaluation cases across methods and uncertainty over both learned runs and task instances. Test an interval for the paired difference, not whether two separate confidence intervals overlap. Report safe completion, any-violation probability, unfinished episodes, ordinary completion, progress/engagement, force severity and interventions. State whether the target estimand describes a fixed deployed policy or the whole stochastic training procedure.

A proposed confirmation criterion is a useful safe-completion improvement whose interval supports positive benefit, while also meeting the independently declared risk criterion. A stronger claim that improvement is at least 10 points requires its lower bound to exceed that margin. Be precise about which is claimed. Evolutionary superiority additionally requires a supported advantage against random and the selected ordinary learner at matched budgets. Matching their point estimates is not equivalence.

**Gate 4: test reduced tuning through transfer of the procedure.** On a new contact regime or second insertion geometry, keep the controller structure, safety-handling recipe and optimizer settings fixed. Allow fresh learning under a declared budget, then evaluate on untouched target cases. This tests whether the procedure transfers without per-condition retuning. Separately, test the original policy without learning if same-weight robustness is of interest; do not call fresh learning zero-shot transfer.

For the reduced-tuning claim, compare attainment of the same predeclared safety/utility target against tuning work. Count every candidate setting, reward-weight sweep, controller modification and simulator interaction used for development and selection. Show performance versus task-specific tuning allowance; record shared development effort as well. Developer hours can be secondary because they are difficult to standardize. A proposed 5 percentage point noninferiority margin against a tuned fixed reference could operationalize 'close enough', but its importance must be justified and frozen in advance. Demonstrating noninferiority requires the relevant bound to stay above the negative margin, together with a suitable safety requirement. An inconclusive difference is not evidence of equivalence or reduced tuning.

### Gaps

- Numerical margins, acceptable force/risk, sample size and compute budget cannot be justified from the task description alone. They require a chosen benchmark, a physical/operational interpretation and independent development variance estimates.
- No simulator has been installed, tested or run for this proposal. No existing held-out panels, completed protocols or unused allocations are reopened.

## What would the safety and statistical litmus mean?

### Takeaway

The empirical claim must be tied to a specified policy or training procedure and a declared case distribution. No observed violation is limited evidence, and repeated measurements from the same learned policy do not quantify training instability.

### Cited Findings

- Exact binomial confidence limits are available for one-sided bounds, particularly when event counts are small and normal approximations are unsuitable. Their use presumes a binomial trial model. [NIST exact binomial documentation](https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm).
- Agarwal and colleagues recommend reporting uncertainty from finite independent runs rather than relying on a single score or best run. [Agarwal et al., 2021](https://arxiv.org/abs/2108.13264).

### Inferences

- For a fixed policy evaluated on independent, identically distributed episodes, zero events in n trials gives a one-sided 95% exact upper bound of **1 - 0.05^(1/n)**. This is approximately 3/n. The formula follows by solving P(zero events) = (1-p)^n = 0.05. As illustrations, zero in 59 episodes bounds the event probability just below 5%, while zero in 299 bounds it just below 1%. These are illustrations of evidence requirements, not a prescribed experiment size or acceptable real-world risk.
- Do not apply those counts to correlated time steps, repeated deterministic resets, or a pooled mixture of a few selected policies as if it were a fixed iid policy sample. Condition-specific analysis and hierarchical/clustered inference are needed where the sampling structure demands them. Safety estimates for the training algorithm require independent training repeats as well as episode repeats.
- Final sample size should follow the declared effect, risk target, uncertainty model and an independently estimated variance. Predefine the maximum budget and any sequential stopping rule; repeatedly examining a conventional confidence interval and stopping when it looks good changes its interpretation.
- A successful training algorithm can still have unsafe candidate evaluations. Report cumulative training violations and failed starts separately from held-out execution performance. A claim about safe exploration requires its own budget and criterion.
- Maintain a tuning ledger and a total interaction ledger. Count baseline reproduction, witness construction, sensor checks, random probes, discarded candidates, validation, hyperparameter tuning and failed starts. Report elapsed time/model calls separately. Identical episode caps do not ensure equal interaction cost, especially with early termination.

### Gaps

- We do not yet have a risk target or benchmark-specific variance, so no powered confirmatory run count is proposed.
- An absolute safety rate on the sampled simulated distribution does not establish hardware damage avoidance or safety outside that distribution.

## What is the smallest convincing story?

### Takeaway

First establish a strong result on one well-understood task family, then ask whether the learning procedure survives a meaningful change without retuning. A large benchmark sweep is not a substitute for this causal chain.

### Cited Findings

- Carefully constructed examples and baselines can support explanations of algorithm properties; broad demonstrations and scientific comparisons have different purposes. [Patterson et al., 2024](https://jmlr.org/papers/v25/23-0183.html).

### Inferences

- Minimal convincing progression: (1) valid safety signal and same-interface safe witness; (2) repeated improvement in safe completion within a finite budget; (3) independent comparison with random search and constrained learning; (4) a new contact regime or insertion family with fixed recipe and explicit retuning accounting.
- The first two stages would make the direction promising. The third establishes what evolutionary search contributes. The fourth addresses the lecturer's reduced-manual-tuning motivation. None alone supports all four claims.
- No need start with many environments, huge models or a factorial of every ablation. Reserve mechanisms for the specific ambiguity the pilot exposes: for example, feedback versus constant settings, recovery authority versus action scaling, or learned policy versus a protective filter.

### Gaps

- Which benchmark supplies both useful force information and an appropriate action interface is the parallel benchmark investigation's decision. It must be resolved before numerical stage budgets and acceptance thresholds are frozen.
