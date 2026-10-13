# Intervention-aware evolutionary adaptation

Focused primary-source search, 13 October 2026. Candidate opportunities below are hypotheses, not verified novelty gaps. No experiments were run. The current authority remains [oracle.md](../../../../oracle.md).

## Can intervention traces improve evolutionary adaptation?

### Takeaway

Intervention information and compact adaptation already have substantial precedents. The more promising evolutionary question concerns how traces change variation or evaluation, and whether that change improves measured safe completion under matched budgets.

### Cited Findings

- CoFiT, submitted 1 October 2026, already fine-tunes pretrained humanoid trackers with active CBF filtering, constraint-context observations, correction and executed-action feedback, temporal histories and filter-related penalties. It identifies dynamics, objective and information mismatches. Its scenes address mild interference rather than replanning around fully blocking obstacles. These are strong overlaps with the general policy/filter-cooperation motivation. [CoFiT, Sections IV-VI](https://arxiv.org/html/2610.02341)
- Safety Filtering While Training studies learning from filtered transitions and correction penalties, including the effect on chattering. Penalties for corrections are established; intervention count or magnitude alone is consequently a weak method contribution. [Paper, Sections III-IV](https://arxiv.org/html/2410.11671)
- Safety-Informed Mutations, GECCO 2022 Companion, stores visited unsafe states in a cost buffer, computes output sensitivities on that buffer, scales mutations and selects individuals using reward and cost. It evaluates locomotion and navigation. It therefore precedes the broad idea of safety-experience-guided evolutionary variation. [Author-hosted paper](https://emarche.github.io/assets/pdf/GECCO2022_safemut.pdf), [DOI](https://doi.org/10.1145/3520304.3533980)
- Safe Mutations through Output Gradients, GECCO 2018, scales perturbations using output sensitivity to avoid destroying existing behaviour. Here "safe" means behaviour-preserving mutation, not physical safety. [Paper](https://arxiv.org/abs/1712.06563)
- Guided Evolutionary Strategies, ICML 2019, shapes a search distribution using a subspace of surrogate-gradient directions. This precedes using approximate model gradients to guide search. [Paper](https://proceedings.mlr.press/v97/maheswaranathan19a.html)
- SLowRL, submitted 17 March 2026, already combines frozen base policies, LoRA adaptation and learned recovery-policy safety enforcement for locomotion. Low-rank safe fine-tuning is therefore established independently of evolution. [Paper, Sections IV-V](https://arxiv.org/html/2603.17092)
- Residual RL already adds a learned correction to a conventional controller for real-world contact-rich block assembly. Residual RL from Demonstrations extends the concept to policies initialized from demonstrations. [Johannink et al.](https://arxiv.org/abs/1812.03201), [Alakuijala et al.](https://arxiv.org/abs/2106.08050)
- Behavior Policy Learning fine-tunes low-dimensional multi-stage manipulation policies with CMA-ES, starting from solution sketches and model-based controllers. Its authors explicitly state that other optimizers, including PPO or TD3, could be used. [Paper, Section 3](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.974537/full)

### Inferences

- Merely appending intervention history to an evolved adapter would combine established feature design, adaptation and search. It could be an implementation step or a useful empirical study, but needs stronger evidence to support a method contribution.
- The distinctive candidate signal is a *task-stalling intervention*, rather than any intervention or physical violation. It might identify behavioural directions that improve progress while preserving successful nominal behaviour. Whether that distinction is useful must be measured.
- Intervention signals from a learned filter are imperfect labels. A large correction does not prove the proposed action physically unsafe, and minimizing corrections can discourage necessary progress.
- A frozen world model can provide representation and safety assessment without making imagined fitness the evidence for safety. Simulator-evaluated candidates initially offer a cleaner attribution test.

### Gaps

- This search did not establish an exact prior combining intervention-conditioned variation, pretrained visual manipulation and learned world-model filtering. It also does not establish that combination as novel.
- No public CoFiT or SLowRL implementation/checkpoint was verified in this search. HTML papers were checked for project links, but absence of a detected link is not proof code is unavailable.
- No measured prevalence of avoidable stalls, intervention trace quality or candidate-evaluation cost has been established for the proposed project.

## Which adjacent opportunities offer a useful contribution with limited engineering?

### Takeaway

The most promising small method extension is trace-guided variation in a compact adapter. A lower-risk empirical route is to establish when compact evolution improves safe continuation and why. Intervention-focused candidate evaluation is another practical option, conditional on valid simulator restoration.

### Cited Findings

- Guided ES has public TensorFlow and JAX demonstration notebooks, not an integrated manipulation implementation. The repository was archived in April 2026. [Author repository](https://github.com/brain-research/guided-evolutionary-strategies)
- Reverse Curriculum Generation already adapts training start-state distributions to agent competence in simulated manipulation, so restart-based focused learning is established. [CoRL 2017 paper](https://proceedings.mlr.press/v78/florensa17a.html)
- The local UNISafe audit checked source and checkpoint structure but ran no closed-loop experiments. It reports a public simulated block-plucking checkpoint and runtime migration work for a 5090. It does not verify a working benchmark or throughput. [Repository compute audit](../../reports/UNISafe%20inference%20and%20training%20compute.md)

### Inferences

#### Opportunity 1: Intervention-guided evolutionary variation

**Hypothesis.** Variation concentrated on behavioural directions implicated in repeated intervention and low progress can discover useful continuations more efficiently than unguided evolution, while retaining behaviours that already complete safely.

**Nearest work.** Safety-Informed Mutations is the closest EA mechanism. Safe Mutations and Guided ES supply additional strong controls. CoFiT is the closest policy/filter interface precedent. The remaining question is whether distinguishing stalled from productive interventions supplies a better search direction than generic safety or task information.

**Smallest pilot.** Freeze the base policy, representation and filter. Add one small residual or selected-layer adapter. Log latent/proprioceptive features, proposed and executed actions, correction vectors, task progress and physical outcomes. Compare isotropic mutation with one trace-conditioned operator. An initial operator could use proposal/correction sensitivity on stalled traces to bias adapter perturbations, while an independent successful-state buffer constrains behavioural disruption. Retain some isotropic exploration. Select using actual task outcomes, not correction reduction alone.

**Essential controls.** Random search; standard ES or CMA-ES on the identical adapter; output-sensitive mutation on all visited states; Safety-Informed-Mutations-style weighting; a generic Guided-ES variant if surrogate gradients are used; RL on the same adapter with the same observations, traces and reward information. Include stalled versus all-intervention versus randomly sampled trace ablations. Count trace collection, discarded candidates, model queries and derivative-computation time.

**Burden and assets.** Medium marginal implementation once a baseline runs. Backpropagating through the whole visual policy may be expensive; an output residual avoids much of it. Existing theory and Guided ES demos are verified, but no turnkey task implementation or trained asset for this extension is verified. This is the most direct route to a real EA mechanism, although it could remain too close to existing guided mutation without an informative finding.

#### Opportunity 2: Compact adaptation as a controlled empirical study

**Hypothesis.** Filtered policy search has regimes in which small evolutionary updates retain useful pretrained skills while discovering alternatives after intervention more effectively than comparable gradient updates. This is a question about mechanisms and budgets, not a presumption that evolution wins.

**Nearest work.** SLowRL establishes safe LoRA fine-tuning; residual RL establishes additive correction; Behavior Policy Learning establishes compact CMA-ES fine-tuning; CoFiT establishes intervention-aware adaptation. A credible contribution needs to identify conditions under which evolutionary adaptation helps, not introduce the adapter as new.

**Smallest pilot.** Use a bounded action residual with a fixed feature extractor before attempting LoRA across a large policy. Compare no adapter, one compact adapter and one larger adapter across held-out starts and two controlled intervention conditions. Investigate whether gains arise from recovery, anticipating intervention, reduced destructive updates or simply extra information. Full-episode safe completion is the primary outcome.

**Essential controls.** Same adapter, action bounds, intervention features, reward and interaction allowance for EA and RL; frozen filtered baseline; random search; ordinary filtered learning. If reward shaping is varied, apply each variant to both optimizers. Report learning curves versus simulator steps and elapsed time. Provide ablations without intervention features so interface benefits are not attributed to evolution.

**Burden and assets.** Lowest marginal method burden of these candidates, conditional on a working simulator, policy and learned filter. It may be a strong honours empirical project. GECCO strength depends on a reproducible explanatory result with breadth. Neither LoRA nor a residual is inherently low-dimensional: count the actual trainable parameters before selecting an EA.

#### Opportunity 3: Intervention-focused candidate evaluation

**Hypothesis.** Evolution spends many evaluations on easy, already-solved episode prefixes. Evaluating candidate continuations from a mixture of recorded stalled states and ordinary starts could improve budget use without sacrificing full-episode performance.

**Nearest work.** Start-state curricula and reverse curricula already exist. The remaining question is whether intervention/progress traces are a useful allocation signal specifically for evolutionary evaluation, beyond generic difficulty prioritization.

**Smallest pilot.** After collecting a fixed baseline dataset, compare full-episode evaluation with a mixture of full episodes and restored post-intervention continuations. Keep the mutation operator fixed. Use held-out full episodes for final selection and testing. Restore recurrent model state and action/observation history, as well as simulator state; otherwise the comparison changes the information state.

**Essential controls.** Uniform restart sampling, ordinary hard-state sampling, all-intervention sampling and stalled-intervention sampling; RL with the same restart allowance; equal total simulator interactions including state collection and replay. Every candidate should ultimately face original episode starts, because fixing a continuation can break the approach that reaches it.

**Burden and assets.** Low to medium only if exact restoration is already supported; potentially high in a visual contact simulator. No valid UNISafe restoration pipeline was verified. This may be an evaluation-efficiency component for Opportunity 1 rather than a standalone contribution.

### Gaps

- No task or implementation is selected, so engineering rankings are relative marginal estimates, not wall-clock promises.
- No comparison has established a broad evolutionary advantage. Matching simulator steps may still produce different elapsed-time costs; report both.
- The state-restoration requirement may make Opportunity 3 less practical than the apparent simplicity suggests.

## What would decisively falsify these opportunities?

### Takeaway

The project should first seek evidence that avoidable continuation failures exist under fixed safety constraints. Each method then needs a test that can distinguish useful intervention information from extra data, an easier representation or biased evaluation.

### Cited Findings

- CoFiT explicitly evaluates realized constraint satisfaction rather than claiming unconditional safety from its fixed filter. It also notes that different proposals can yield the same executed action and reward. [Paper, Sections III-B and IV-B](https://arxiv.org/html/2610.02341)
- The repository oracle makes the chosen direction conditional on consequential avoidable stalls, and requires reporting physical violations, unfinished episodes and all adaptation interactions. [Current research authority](../../../../oracle.md)

### Inferences

- **Common initial failure criterion:** the accessible task has negligible meaningful stalls, or the examined stalls disappear with adequate horizon or corrected observation/action history. That undermines this specific continuation study even if evolution can optimize generic task return.
- **Opportunity 1 failure criterion:** guided mutations reduce corrections but fail to improve held-out safe completion; or the advantage disappears with randomly permuted intervention labels, generic gradients, extra matched RL data or ordinary output-sensitive mutation. This would show the proposed trace mechanism is unnecessary or misleading.
- **Opportunity 2 failure criterion:** a matched compact RL adapter achieves equal or better results across the declared budget regime and evolution reveals no distinct tradeoff or failure mechanism. That falsifies the intended EA-advantage claim, although a careful negative result could still inform the honours submission.
- **Opportunity 3 failure criterion:** replayed continuations fail to reproduce original dynamics, or restart-sampled gains vanish in original-start full episodes. If uniform or generic hard-state restart sampling performs equally well, intervention-specific allocation lacks support.
- **Safety failure criterion for all three:** safe completion improves only alongside a material increase in observed physical violations, or uncertainty is too large to support the declared safety comparison. A fixed learned filter does not remove this possibility.
- **No-violation interpretation:** retain counts and denominators. With zero violations in n independent episodes, the approximate 95% upper rate bound is 3/n. A study cannot establish unchanged physical safety by observing zero violations in a handful of episodes.
- **World-model contribution check:** if replacing learned safety assessment with a matched simpler filter leaves all important findings unchanged, the method may contribute to safe evolution generally, but should not overclaim a world-model-specific insight.

### Gaps

- No numeric stopping threshold is justified before baseline stall prevalence, outcome variability and runtime are measured.
- A finite unsuccessful recovery search does not prove a stall unavoidable. Positive evidence of safe alternative continuation is stronger than a failed search for one.
- Publication strength and exact novelty require a deeper follow-up search after selecting the mechanism and obtaining a baseline failure trace.
