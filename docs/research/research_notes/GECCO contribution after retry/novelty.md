# Literature and GECCO contribution options

Checked 24 October 2026. Scope: primary-source audit of the closest literature, paper claims and venue fit after the frozen retry gate. The next-phase compute constraint is $50 to $100. These notes propose research; they authorize no run and change no frozen protocol.

## What does the closest literature leave for this project?

### Takeaway

The broad ideas of evolving frozen VLA actions, adding a runtime safety wrapper, adapting chunk horizons and evolving inference-time parameters already exist. The credible opening is a carefully controlled empirical study of compact execution controllers under specification-grounded safe-completion objectives and severe interaction budgets. Whether evolution adds value remains an experimental question.

### Cited Findings

- **ManiGuard** is the exact benchmark at arXiv 2608.17386, submitted 18 August 2026. It separates task success from specification satisfaction, provides six families, 200 base tasks and 1,000 ID/OOD scenarios, and uses physics-grounded LTLf monitors. Its benchmark paper already establishes that successful episodes can violate safety and that fine-tuning leaves a gap. Repeating those observations is context, not our novelty. The official evaluation documentation describes independent success and safety checking, contact-gated safety and engagement reporting. [Paper](https://arxiv.org/abs/2608.17386), [official evaluation overview](https://nu-ideas-lab.github.io/ManiGuard/docs/evaluation/).
- **VLA-Pilot** is directly relevant evolutionary prior art. The author project page identifies arXiv 2511.14178 and now states acceptance at RA-L 2026. It iteratively refines candidate proposals from a frozen diffusion VLA with evolutionary selection and recombination, using an MLLM-reasoned reward. It evaluates DiVLA and RDT-1B and notes runtime overhead and keypoint-grounding limitations. It defeats any claim that evolution around a frozen VLA is itself new. The distinction available to us is an execution controller optimized across episodes and deployed without per-decision populations or an MLLM reward. [Author project page](https://rip4kobe.github.io/vla-pilot/).
- **Online-ES**, arXiv 2609.38855, has the cited title and genuinely uses evolution strategies for flow-matching VLAs. Its evolutionary exploration in action-trajectory space produces a self-supervised objective that changes model parameters. Thus it is relevant ES-on-VLA prior art, while differing from the frozen-base-model constraint here. [Paper](https://arxiv.org/abs/2609.38855).
- **Evolving Cache Schedules**, arXiv 2607.20293, is a verified match. It optimizes cache-refresh schedules by evolutionary search, then deploys the schedule with a pretrained diffusion policy without retraining. Its goal is inference acceleration, not specification-grounded safe completion. It defeats a broad claim that nobody has evolved inference-time knobs of a frozen robot policy. [Paper](https://arxiv.org/abs/2607.20293).
- **APEX**, arXiv 2606.16504, is a verified match. It inserts an adaptive reference-reconstruction layer between a black-box policy and controller and reports improved tracking and task success. The layer is therefore close execution prior art; our claim cannot rest on the existence of an output-side execution layer alone. Its abstract does not establish a specification-grounded violation objective or an evolutionary search comparison. [Paper](https://arxiv.org/abs/2606.16504).
- **DSRL**, arXiv 2506.15799, learns a latent-noise policy with RL while keeping the diffusion/flow base policy fixed. The paper explicitly supports black-box policy access and demonstrates adaptation of a generalist pi0 policy. It defeats claims that frozen-policy adaptation must use evolution or that RL necessarily requires end-to-end base-model gradients. It is a related learning route, though not automatically the fair learning baseline for a different, output-side execution parameterization. [Paper](https://arxiv.org/html/2506.15799), [author project page](https://diffusion-steering.github.io/).
- **A Safe Action Is Not Enough**, arXiv 2610.05166v2, revised 6 October 2026, is especially important. Its alarm-triggered, training-free VICS-G reranks candidates from a frozen VLA toward feasible safe completion. The abstract reports lower cumulative safety cost across six Safety-CHORES settings while staying close to policy sampling in success. It requires neither retraining nor online rollouts. It defeats broad claims that runtime frozen-VLA layers have never been measured for safety, and shows that the literature already distinguishes immediate admissibility from eventual safe completion. Its setting and mechanism differ from a small evolved execution controller for contact-rich ManiGuard manipulation. [Paper](https://arxiv.org/abs/2610.05166).
- **Adaptive execution is already a crowded area.** BCP learns continue-or-replan decisions with RL over a frozen VLA and rewards both completion and efficient query use. CA3C uses imagined futures to select commitment horizons without retraining the base policy. PAV predicts the validity of candidate prefixes and reports success-query trade-offs with pi0 and pi0.5. A scalar horizon sweep is useful mechanism evidence, but adaptive horizon alone is not a novelty claim. [BCP](https://arxiv.org/abs/2608.03483), [CA3C](https://arxiv.org/abs/2610.07949), [PAV](https://arxiv.org/abs/2610.21434).
- Binary-response noisy evolution is not an untouched subject. A 2022 paper introduces SARAGA, a genetic algorithm using dynamic resampling and a surrogate for success-probability optimization from binary-response simulations. This is enough to rule out a broad claim that EC has not addressed Bernoulli simulator fitness. [Publisher paper](https://jkimst.org/upload/pdf/KIMST-2022-25-4-412.pdf), DOI 10.9766/KIMST.2022.25.4.412.

### Inferences

- The useful research question is: **Can a compact controller over policy execution improve safe requested completion of frozen contact-rich VLAs, and does evolutionary optimization discover better controllers than equally informed alternatives under the same simulator budget?** This is specific, falsifiable, and leaves open a negative answer about evolution.
- The distinctive combination may be small deployable execution controllers, temporal specification labels, task-disjoint validation and an honest noisy-selection protocol. This is an empirical contribution hypothesis, not a verified first-in-the-world claim.
- Frozen-policy execution improvements alone justify an application result if sufficiently substantive. Adding a CMA-ES or NSGA-II call does not make an algorithmic contribution. The value must come from demonstrated generalization, a useful controller representation, or evidence about evaluation/search reliability.
- A hand-tuned scalar that matches the evolved result is a useful engineering finding and a negative answer about optimizer benefit. Do not enlarge the controller merely to make random search weaker.

### Gaps

- This targeted audit is not a systematic review of all papers, companion proceedings and unpublished work through 24 October 2026. No strong priority claim is supported.
- No arXiv title/ID mismatch was found for ManiGuard, DSRL, APEX, Online-ES and EVO in this subset. The VLA-Pilot arXiv fetch failed, but its author page itself gives the matching ID and mechanism. Unsupported detailed numbers from the old report should still be checked against full papers before reuse.
- VICS-G does not by itself show that its learned-task setting, action representation or implementation transfers to Jar. It is mandatory related work; whether it is a practical comparator requires a separate compatibility assessment.

## Which contribution path is realistic on a lean budget?

### Takeaway

Aim first for a bounded empirical execution-controller result. Treat noisy evaluation as an essential part of that result. Make a standalone benchmarking contribution only if there is evidence about optimizer reliability and a method that improves it; the current repeated native episodes and retry null are insufficient by themselves.

### Cited Findings

- The repository requires establishing variance, then a lever, then evaluation sizing, then the optimizer comparison. It requires a tuned fixed setting and random search, and its agent controls additionally require ordinary learning with the same information before an evolutionary claim. These are project controls, not optional citation choices. [Current plan](../../../../firstPrinciples.md), [agent controls](../../../../AGENTS.md).
- The frozen retry gate reports 17/96 native versus 16/96 retry safe successes and 48/96 versus 47/96 violations. The implemented stall rule missed visible parked behavior, and only 21 retry episodes received a retreat. This result does not establish that execution changes generally lack leverage. [Gate scoring](../../../../progress/receipts/03-results.md), [smoke diagnosis](../../../../progress/receipts/03-smoke.md).
- The identity receipt shows deterministic policy chunks for identical request bytes, while separate simulator launches differed in rendered observations despite identical initial recorded state. It also discovered consecutive-same-seed server reseeding behavior. These are concrete engineering controls for later studies. [Identity receipt](../../../../progress/receipts/03-identity.md).
- The optimization benchmarking guidelines emphasize explicit goals, well-specified problems, suitable algorithms, appropriate measures and reproducibility, rather than algorithm branding. [Benchmarking in Optimization](https://arxiv.org/abs/2007.03488).

### Inferences

**Path A, preferred: compact execution-controller empirical study.** Establish one useful dial, then freeze a small controller family, perhaps two to four parameters if the mechanisms justify each one. A scalar screen is the prerequisite, not the final optimizer experiment. The minimum credible paper claim is a task-disjoint improvement in safe requested completion at a fixed interaction ceiling, with violations, unfinished episodes and engagement reported together. An evolutionary advantage claim additionally needs independent search replications and a matched-budget win over random search, the fixed setting and an ordinary-learning alternative with the same inputs and controllable outputs. A finite menu of execution choices trained by an explicitly specified small RL or policy-gradient learner is conceptually closer than importing DSRL into a different action space. It must be budgeted and evaluated fairly, not included as a deliberately underpowered token baseline. BO is an additional optimizer comparison, not a substitute for that learning control.

For $50 to $100, do one search algorithm well. Do not simultaneously buy a recurrent controller, Pareto search, RL fine-tuning, cross-family transfer and a world model. Keep architecture, search strategy and evaluation policy as separate questions. If there is only one useful scalar, a fixed sweep or BO may be the sensible method and there may be no strong evolutionary story yet.

**Path B, conditional: reliability of search under noisy binary safety outcomes.** A substantive contribution would compare equal-budget evaluation allocations, such as fixed small panels versus structured repeats and elite reevaluation, then demonstrate improved selected-controller performance on untouched evaluation episodes. Measure winner's curse, rank stability, probability of selecting an actually useful setting and search outcomes. Include genuine search decisions, multiple runs and an existing noise-handling comparator. The question is not merely whether simulator seeds are deterministic. A reusable measured candidate bank can support many inexpensive allocation experiments, but replay conclusions are conditional on that bank and need an independent online check. Binary fitness and resampling themselves are old topics.

**Path C, fallback: diagnostic or negative-result companion paper.** A defensible narrow claim is that this frozen retry design fails to improve safe completion, with mechanism evidence and a reproducible protocol showing why. To make that an EC paper, connect it to the learnability/searchability of compact wrappers, perhaps through a bounded preregistered comparison showing how one-shot selection misleads. Do not present absence of significance as equivalence, do not claim that retry cannot work, and do not claim universal simulator unreliability from one pinned profile. A coherent small companion submission is more defensible than an unsupported full-paper claim.

**Contribution ladder.** First, a useful fixed execution setting is a mechanism result. Second, a compact controller that generalizes beyond that setting is a representation result. Third, replicated optimizer superiority at matched budgets supports an evolutionary result. Fourth, a general and validated evaluation allocation method supports a methodological result. Each step requires new evidence; none follows automatically from the preceding step.

### Gaps

- The final affordable episode count depends on measured cost per simulator interaction, shorter-prefix query overhead, installation and storage overhead. Another researcher is sizing this from the rental receipts. The $50 to $100 ceiling does not itself prove that all mandatory baselines and replications fit.
- No lever has yet passed the project gate. A full optimizer protocol cannot be finalized responsibly before that happens.

## Where does this fit at GECCO and what dates are confirmed?

### Takeaway

EML is the most natural application route for learning a compact controller. EMO becomes appropriate if a genuinely informative completion-versus-violation front is the contribution. BBSR requires an actual EC benchmarking/reproducibility contribution; reproducing a robotics benchmark alone is not enough. Use 2026 submission timing only as planning precedent because the fetched 2027 call still carries 2026 dates.

### Cited Findings

- The official 2027 homepage lists Krakow, 12 to 16 July 2027, in person only. The official 2027 call-for-papers page returned text naming GECCO 2026, 19 January 2026 abstracts and 26 January 2026 full papers. Therefore the source is stale for 2027 deadlines. A third-party listing of 26 January 2027 does not resolve that. [Official homepage](https://gecco-2027.sigevo.org/HomePage), [official call](https://gecco-2027.sigevo.org/Call%2Bfor%2BPapers).
- The 2027 paper instructions list full papers at eight pages excluding references, poster-only papers at four including references, student workshop papers at four including references, and anonymous full/poster review. Because some pages are carried over, recheck the finalized 2027 call before submission. [Official instructions](https://gecco-2027.sigevo.org/Paper%2BSubmission%2BInstructions).
- The EML scope includes reinforcement learning, transfer and application papers, and strongly encourages comparison to relevant non-evolutionary ML methods. [EML scope](https://gecco-2027.sigevo.org/Track?itemId=53).
- The EMO scope includes expensive objectives, uncertainty handling, benchmarking against non-EMO methods and applications that bring broader EMO insights. [EMO scope](https://gecco-2027.sigevo.org/Track?itemId=54).
- BBSR welcomes benchmarking methodology and statistical analysis of EC performance. Its reproducibility category asks for a substantial shift in confidence in a published result and public artifacts at submission. The current ManiGuard replication is a robotics replication consistent with the published row, not that EC result. [BBSR scope](https://gecco-2027.sigevo.org/Track?itemId=2672).
- The currently fetched RWA scope excludes previously existing public real-world benchmarks, directing such work to relevant algorithmic tracks. It is therefore a risky default for ManiGuard. [RWA scope](https://gecco-2027.sigevo.org/Track?itemId=60).
- The 2026 student workshop explicitly admitted Bachelor students as main contributors, first and corresponding authors, and published up-to-four-page papers in the companion proceedings. Its 2026 deadline was 3 April after an extension. This is a legitimate precedent for a smaller route, not a confirmed 2027 deadline or guaranteed acceptance. [2026 student workshop call](https://gecco-2026.sigevo.org/Call%2Bfor%2BStudent%2BWorkshop).

### Inferences

- Choose the track from the final evidence: EML for a learned compact execution controller, EMO for a well-validated trade-off study, BBSR for optimizer evaluation reliability. NE requires a real neural-controller contribution; adding a network solely to match a track is not justified.
- The current null result is worth preserving and explaining but is not yet a complete GECCO main-proceedings contribution. Its best immediate use is to inform the next lever and prevent a poorly founded search experiment.
- Plan internally toward late January while labeling that date assumption provisional. A poster or student workshop route should remain an intentional fallback if the empirical contribution stays narrow.

### Gaps

- Confirmed official GECCO 2027 abstract and paper deadlines were not established by this search. No promise about submission dates or acceptance is supported.
- Track pages themselves show signs of copied historical content, including mismatched chair information. Scope guidance is useful; verify the final 2027 pages before contacting chairs or submitting.
