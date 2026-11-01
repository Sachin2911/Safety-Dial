# Compact evolutionary controllers for frozen world models

## Which published systems establish a useful precedent?

### Takeaway

Persistent evolutionary controller learning with a learned world representation has worked. It does not imply that a large arbitrary residual network will learn from a few dozen candidate evaluations, and it is distinct from fitting an online CEM distribution at each action.

### Cited Findings

- World Models evolves a linear controller over learned latent and recurrent features, with 867 controller parameters in CarRacing. CMA-ES uses population 64 and averages each candidate over 16 rollout seeds. The appendix reports reaching the 900-reward criterion after 1,800 generations, then evaluating over 1,024 scenarios; its 100-rollout result is 906 +/- 21. CarRacing controller training uses the actual environment; the separate Doom controller is trained inside the learned model. Thus the paper demonstrates evolution over learned representations, with a much larger evaluation budget than our earlier 24-genotype pilot. [Primary article and appendix](https://worldmodels.github.io/), [author code](https://github.com/hardmaru/WorldModelsExperiments).
- POPLIN searches action trajectories initialized by a policy, or searches perturbations of policy parameters. Persistent policy updates use behavior cloning, adversarial learning or averaging. Its appendix finds efficient planning even with fixed zero base weights, so planning success alone does not establish useful persistent learning. Its shared-across-time parameter-noise variant can underperform separate perturbations, a warning that restrictive parameterizations can also remove needed authority. [Primary paper, sections 4 and appendix A.6](https://arxiv.org/html/1906.08649).
- Black-DROPS optimizes policies against uncertain learned dynamics using BIPOP-CMA-ES. It handles noisy rankings by reevaluating a subset of candidates and adapting exploration when rankings are unstable. Models update as real experience arrives. This supports black-box model-based policy learning, but does not establish frozen visual-model safety transfer. [Primary paper, sections IV-C to IV-E](https://arxiv.org/html/1703.07261v2).
- Behavior Policy Learning initializes a high-level policy from solution sketches, with a fixed model-based low-level controller, then uses CMA-ES for small-sigma fine-tuning. Candidates receive multiple randomized evaluations. The paper reports three selected policies changing success from 66/64/60% to 75/82/76%, and one optimized policy succeeding on 8/12 physical configurations. These are task-completion results, not collision-free success rates. [Primary paper, sections 2.4 and 3.3-3.4](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.974537/full).

### Inferences

- Do not interpret 510 weights as inherently too many: a larger controller worked in World Models. The actionable mismatch is the tiny search budget and weakly structured representation relative to that precedent. [World Models](https://worldmodels.github.io/).
- A poor random untrained adapter does not establish that evolution cannot improve the representation. A zero/native anchor and demonstrations of useful nonzero authority are more informative prerequisites than demanding random initialization improve completion. This is an experimental-design inference from the initialized fine-tuning and planning-control distinctions above, not an established LeWM result. [Behavior Policy Learning](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.974537/full), [POPLIN](https://arxiv.org/html/1906.08649).

### Gaps

- None of these sources establishes our full combination of frozen LeWM, learned native-proprioception safety prediction, newly supplied virtual hazards and a reusable evolved proposal adapter. The current search is not an exhaustive novelty review.

## What small representation is worth implementing first?

### Takeaway

Use a compact feedback controller with explicit authority over a few meaningful proposal directions. A structured 8-32 coefficient policy is a practical campaign choice, not a literature-derived optimum.

### Cited Findings

- Augmented Random Search demonstrates static linear policies on continuous-control benchmarks. It normalizes observations, rescales update magnitudes using return variation and selects promising perturbation directions. Its larger-seed evaluation finds substantial performance variation and warns against optimistic few-seed conclusions. These are locomotion results without our explicit safety specification or frozen LeWM. [NeurIPS 2018 primary paper](https://papers.neurips.cc/paper/7451-simple-random-search-of-static-linear-policies-is-competitive-for-reinforcement-learning.pdf).
- Queisser and Steil use CMA-ES over parameterized movement primitives, including a lower-dimensional task manifold and optional unrestricted policy search. They report benefits in some reaching settings, but no significant speed gain in a planar-arm scenario. Poor learned manifolds can guide search away from the eventual solution. This supports testing a small representation while checking that it retains enough authority. [Primary paper, sections 4-6](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2018.00049/full).
- Our earlier residual pilot evaluated 24 CMA genotypes on four training cases where native already completed 4/4. Its separate selection panel did not show evolved completion gain. The later fixed untrained screen scored native 3/18 versus adapter 1/18 on PushT hazards. These are repository observations, not literature findings. [Residual study](../../lewm-residual-study.md), [readiness screen](../../lewm-screen.md).

### Inferences

1. First choice: a zero-anchored bounded linear feedback policy over normalized hazard-relative geometry, causal progress/stall feedback and the native planner's own action or shifted-plan direction. Its output controls radial and tangential proposal bias, activation near hazards and temporal persistence. Use a few shared temporal basis functions instead of independent weights for all action coordinates. This restricts evolution to useful alternatives and avoids asking a random latent projection to reveal spatial goal direction from very few trials. The exact 8-32 coefficient range and basis are our design proposals. [ARS](https://papers.neurips.cc/paper/7451-simple-random-search-of-static-linear-policies-is-competitive-for-reinforcement-learning.pdf), [parameterized skills](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2018.00049/full).
2. Second choice: keep seeded hidden features frozen and evolve only a small final readout or a low-rank factorization of the existing output. This preserves the tested interface while reducing the genotype. Verify that random features encode useful context before spending a full training budget. [World Models](https://worldmodels.github.io/), [ARS](https://papers.neurips.cc/paper/7451-simple-random-search-of-static-linear-policies-is-competitive-for-reinforcement-learning.pdf).
3. Third choice: evolve a small feedback scheduler for search horizon, retention or exploration only if saved failures show a corresponding planning limitation. This could establish an evolutionary meta-control result, but does not by itself establish a behavioral neural adapter. The existing fixed-setting studies are strong controls and must remain explicit. [Current project record](../../../../firstPrinciples.md).

### Gaps

- No cited source determines the right residual bound, population, generation count or feature normalization for this LeWM stack. Freeze these choices prospectively and revise only through a new version with retained earlier evidence.

## How should fitness and evaluation support a real evolutionary claim?

### Takeaway

Train on a common panel with a genuine completion gradient, select on separate common cases, and reserve fresh task and optimizer repeats for the claim. Persistent policy weights must change through real evolutionary updates; online CEM activity is not sufficient.

### Cited Findings

- Black-DROPS emphasizes that noisy candidate ranking, rather than precise value estimation, governs rank-based optimizer behavior. Its reevaluation mechanism tests rank stability instead of assuming one return is reliable. [Primary paper, IV-D](https://arxiv.org/html/1703.07261v2).
- ARS's hundreds-of-seeds evaluation exposes substantial variability obscured by common three-seed comparisons. [NeurIPS paper](https://papers.neurips.cc/paper/7451-simple-random-search-of-static-linear-policies-is-competitive-for-reinforcement-learning.pdf).
- The repository requires native, tuned fixed, parameter random search and ordinary learning with the same information before claiming evolutionary advantage. It also requires counting all simulator interactions and separate model-query accounting. [Research controls](../../../../AGENTS.md).

### Inferences

- Rank candidates with an explicitly frozen constrained rule: observed safety violations first, then safe task completion, then a bounded task-progress or efficiency tie-break. Do not let lower action count, smaller model cost or less motion outweigh completing the original task. This ranking is our proposed implementation of the project specification; it is not a published proof of safety.
- Avoid the earlier 4/4 native training ceiling. A development-only screen can identify a mixed, fixed panel, after which candidate comparisons use the same source cases and seed schedule. Do not pick held-out cases after seeing intervention outcomes. [Recorded ceiling](../../lewm-residual-study.md).
- Use several genuine CMA generations and at least several optimizer repeats for the leading small policy. Compare equal candidate/panel budgets and report actual actions, failed starts, model work and elapsed time. Reevaluate finalists on a fresh common selection panel before opening final evaluation. [Black-DROPS](https://arxiv.org/html/1703.07261v2), [project controls](../../../../AGENTS.md).
- A gradient-trained comparator should use the same feedback and observations. Behavior cloning can be informative if it receives exactly the same retained trajectories, but should not silently get expert demonstrations unavailable to evolution. An RL comparator needs its own recorded interactions. [POPLIN](https://arxiv.org/html/1906.08649), [project controls](../../../../AGENTS.md).

### Gaps

- Seven hours may permit a clear evolved completion benefit while still leaving optimizer superiority unresolved. Report those claims separately. A negative or noisy comparison must remain a negative or inconclusive result rather than being relabelled as evolutionary advantage.
