# Umair Nasir's GECCO research themes

## Which evolutionary algorithms, representations, benchmark designs, statistics, and generalisation claims recur?

### Takeaway

Nasir's recurring technical concern is not a single optimizer. It is how to represent complex artefacts and design variation so evolution can produce useful, diverse, and progressively more capable candidates. His work moves from NEAT-based controller complexification, through quality-diversity archives with LLM code operators, to semantically structured continuous program mutation. The strongest recurring experimental pattern is component isolation through baselines and ablations, but statistical depth varies sharply across papers.

### Cited Findings

- The verified GECCO record as of 20 October 2026 contains five papers: *Augmentative Topology Agents for Open-Ended Learning* in the GECCO Companion 2023, *LLMatic* in the main GECCO 2024 proceedings, *Mortar* and *ProFiT* in the main GECCO 2026 proceedings, and *Continuous Program Search* in the GECCO Companion 2026. The proceedings records give pages 671-674, 1110-1118, 310-318, 727-734, and 477-480 respectively. [ATEP DBLP record](https://dblp.org/rec/conf/gecco/NasirBJC23); [LLMatic ACM/SIGEVO record](https://www.acm.org/binaries/content/assets/sigs/sgb/annual-reports/fy-24-annual-report/appendix-d-fy24-sig-annual-reports.pdf); [GECCO 2026 proceedings index](https://dblp.org/db/conf/gecco/gecco2026.html); [CPS DBLP record](https://dblp.org/rec/conf/gecco/SiperNKSAT26)

- The portfolio describes his core interests as open-ended learning and procedural content generation at the intersection of AI and games. It lists all five GECCO papers and self-reports a best paper nomination for *Mortar*. The proceedings records confirm the publications, but the nomination was only located on the author's portfolio. [Author portfolio](https://umair-nasir14.github.io/); [Mortar proceedings record](https://dblp.org/rec/conf/gecco/NasirLJT26)

- ATEP combines Enhanced POET with NEAT. It co-evolves Bipedal Walker agents and CPPN-encoded terrains, replaces EPOET's fixed-topology evolution strategies with variable-topology NEAT populations, and compares fitness-based, species-based, random, and no-transfer variants. Species-based transfer moves a compatible NEAT species rather than replacing an entire target population. [ATEP paper, methods](https://arxiv.org/pdf/2210.11442)

- ATEP's representations are deliberately complexifying on both sides of the agent-environment pair: NEAT genomes add nodes and connections to agent controllers, while EPOET already uses NEAT-evolved CPPNs for terrain generation and PATA-EC performance vectors for environment novelty. Its primary open-endedness metric is ANNECS, the accumulated number of novel environments that meet a minimal difficulty criterion and are later solved. [ATEP paper, Sections III-IV](https://arxiv.org/pdf/2210.11442)

- ATEP compares variable topology with fixed 20 by 20 and 40 by 40 hidden-layer controllers on a reduced 20-active-environment version of the Bipedal Walker Hardcore benchmark. This reduction is applied to both ATEP and EPOET, but evaluation budgets are not matched: the authors explicitly report more function evaluations for ATEP, especially species-based transfer. [ATEP paper, experimental setup and results](https://arxiv.org/pdf/2210.11442)

- ATEP uses only two optimizer seeds because one run requires about 50,000 to 200,000 CPU hours and 10 to 30 days on 264 Intel Xeon cores. Random-transfer and no-transfer runs are stopped early after poor performance. The paper uses descriptive plots and the word "significantly," but does not report a named hypothesis test or interval in its results section. [ATEP paper, results](https://arxiv.org/pdf/2210.11442)

- ATEP tests cross-environment generalisation by taking the latest 20 environments from each competing method, evaluating selected agents from one method on 60 environments created by the other methods, and running each environment-agent pairing 30 times. Species-based ATEP leaves 10 percent of these environments unsolved. A second analysis tests agents on 80 environments generated and solved within their own algorithm, where later agents outperform early agents on previously unseen landscapes. [ATEP paper, generalisation analysis](https://arxiv.org/pdf/2210.11442)

- LLMatic uses CVT-MAP-Elites with two cooperative archives. The network archive indexes generated PyTorch architectures by width-to-depth ratio and FLOPs, while the prompt archive indexes mutation prompts by prompt identity and LLM temperature. CodeGen-6.1B performs code-level mutation and crossover, and curiosity rewards prompts that place new architectures in the network archive. [LLMatic paper, approach](https://arxiv.org/pdf/2306.01102)

- LLMatic makes diversity and resource cost first-class rather than treating model size only as a penalty. Fitness is reported as network test accuracy after training, while the behavioural descriptors expose FLOPs and architectural shape. The paper says it finds more than 20 competitive CIFAR-10 networks and near-optimal NAS-Bench-201 cells in 2,000 evaluations. [LLMatic paper, approach and conclusion](https://arxiv.org/pdf/2306.01102)

- LLMatic first trains candidates on CIFAR-10 and uses ablations for mutation-only, crossover-only, single-archive, and random-generation variants. Those experiments use 30 seeds and plot means with standard deviations. NAS-Bench-201 evaluation then covers CIFAR-10, CIFAR-100, and ImageNet16-120 over ten runs, with LLMatic reaching mean test accuracies of 94.26, 71.62, and 45.87 percent respectively. [LLMatic paper, Figures 2-5 and Table 1](https://arxiv.org/pdf/2306.01102)

- LLMatic's broader dataset results are not a held-out-controller transfer experiment. NAS-Bench-201 supplies the accuracies of 15,625 cell combinations across three image datasets, and the paper compares search performance within that common cell search space. The conclusion explicitly leaves comparisons on other vision, language, and reinforcement-learning tasks for future work. [LLMatic paper, NAS-Bench-201 setup and conclusion](https://arxiv.org/pdf/2306.01102)

- Mortar returns to quality-diversity and LLM code-level variation. It stores Python game-mechanic functions in a fixed two-dimensional MAP-Elites-style archive indexed by semantic mechanic type and weighted AST code complexity. GPT-4o-mini implements ordinary mutation, diversity mutation, AST-similarity crossover, and compatibility mutation. [Mortar paper, evolution setup](https://arxiv.org/html/2601.00105v1)

- Mortar does not score a mechanic in isolation. MCTS builds complete games around it, then five agents with a declared ability ordering, three MCTS strengths, random, and no-op, play the games. Kendall's tau measures whether observed performance preserves this ordering, and the proposed Constrained Importance Through Search score attributes a game's tau to mechanics using Shapley-style marginal values over combinations actually visited by search. [Mortar paper, evaluation](https://arxiv.org/html/2601.00105v1)

- Mortar compares its MCTS composition against random, LLM-selected, greedy, and Sokoban-initialized alternatives. Five runs are reported as means plus or minus standard deviations for archive elites, QD score, maximum and mean CITS, and game success rate. A ten-person comparative user study then rates six games for interest, novelty, frustration, fun, and understandability; two pairwise comparisons align with the automated metric and one does not. [Mortar paper, experiments and user study](https://arxiv.org/html/2601.00105v1)

- ProFiT represents trading strategies as executable code and uses an LLM-driven evolutionary feedback loop for performance analysis and mutation. Its primary-source abstract reports code-level complexification, self-analysis, five-fold walk-forward validation, seven futures assets, and comparisons with random and buy-and-hold. It reports beating random in every case and buy-and-hold in more than 77 percent of evolved strategy-asset combinations. [ProFiT SSRN record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5889762)

- Continuous Program Search makes representation locality the main research problem. It defines a typed AST trading DSL with four semantic components, trains a transformer VAE with four aligned latent blocks, and measures decode validity, normalized AST edit distance, and action-sequence divergence under controlled perturbations. Perturbations of at most 0.1 define the reported empirical trust region, while divergence and decoding failures rise at 0.5 and above. [CPS full preprint, Sections 3-6](https://arxiv.org/html/2602.07659v1)

- CPS holds a 34-parent, 66-offspring evolution strategy fixed and compares full-space isotropic Gaussian mutation, paired-block Gaussian mutation, and geometry-compiled mutation. The last method restricts changes to semantically paired entry-exit blocks and uses a learned flow model trained on logged mutation outcomes to propose directions. The same optimizer, deterministic seeds, and evaluation budget are used across five assets. [CPS full preprint, Section 7](https://arxiv.org/html/2602.07659v1)

- CPS reports out-of-sample median Sharpe ratios of 1.152 for learned geometry-compiled mutation, 1.005 for isotropic mutation, and 0.890 for paired-block Gaussian mutation. The learned operator uses 13.7 percent of the budget before discovering a strong solution, versus 88.5 and 100 percent, although the Gaussian baselines attain higher maximum Sharpe ratios. The paper reports medians and maxima, not a named formal significance test in the accessible results. [CPS full preprint, Table 1](https://arxiv.org/html/2602.07659v1)

- CPS uses five chronological train-validation-test folds from 2008 to 2025, a ten-day embargo, deterministic backtesting, costs and fees, and five liquid futures contracts. This is genuine temporal out-of-sample evaluation, but the paper makes a framework-level claim across assets rather than demonstrating transfer of one unchanged controller across different policy families. [CPS full preprint, evaluation protocol](https://arxiv.org/html/2602.07659v1)

### Inferences

- The representation trajectory is coherent: variable neural topology in ATEP, executable architecture code and prompt distributions in LLMatic, executable mechanic and strategy code in Mortar and ProFiT, then a learned continuous but semantically factorized program space in CPS. Across these stages, the recurring claim is that useful structure in the genotype and mutation operator improves exploration, diversity, transfer, or evaluation efficiency. [ATEP](https://arxiv.org/pdf/2210.11442); [LLMatic](https://arxiv.org/pdf/2306.01102); [Mortar](https://arxiv.org/html/2601.00105v1); [ProFiT](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5889762); [CPS](https://arxiv.org/html/2602.07659v1)

- QD and archive-based thinking recur, but the descriptors are domain-crafted rather than universal: network FLOPs and width-depth ratio, mechanic category and AST complexity, PATA-EC environment performance, or semantic program blocks. This suggests Nasir's strength is designing interpretable search coordinates around a problem, not applying a fixed black-box optimizer unchanged. [ATEP](https://arxiv.org/pdf/2210.11442); [LLMatic](https://arxiv.org/pdf/2306.01102); [Mortar](https://arxiv.org/html/2601.00105v1); [CPS](https://arxiv.org/html/2602.07659v1)

- "Generalisation" has different meanings across the portfolio. ATEP tests agents on generated unseen terrains, LLMatic reports performance across image datasets in a shared NAS benchmark, ProFiT and CPS use temporal out-of-sample financial data across assets, and Mortar validates generated artefacts with heterogeneous agents and humans. None of these is the stronger ManiGuard claim of one frozen adapter working across separately fine-tuned policy families. [ATEP](https://arxiv.org/pdf/2210.11442); [LLMatic](https://arxiv.org/pdf/2306.01102); [Mortar](https://arxiv.org/html/2601.00105v1); [ProFiT](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5889762); [CPS](https://arxiv.org/html/2602.07659v1)

- Experimental discipline appears to have strengthened over time. ATEP openly accepts unequal function evaluations and only two seeds, while LLMatic uses larger seed counts and ablations, Mortar adds automated and human validation, and CPS deliberately holds the optimizer and evaluation budget fixed while changing only the mutation operator. [ATEP](https://arxiv.org/pdf/2210.11442); [LLMatic](https://arxiv.org/pdf/2306.01102); [Mortar](https://arxiv.org/html/2601.00105v1); [CPS](https://arxiv.org/html/2602.07659v1)

### Gaps

- The portfolio does not link a CV, so this review verifies the publication record rather than claims about awards, reviewing, track leadership, or citation impact. The Mortar best paper nomination remains author-reported. [Author portfolio](https://umair-nasir14.github.io/)

- The detailed ATEP and CPS analyses above use their long arXiv versions. Their archival GECCO items are four-page Companion papers, so the final report should distinguish venue status from the fuller technical evidence. [ATEP DBLP record](https://dblp.org/rec/conf/gecco/NasirBJC23); [CPS DBLP record](https://dblp.org/rec/conf/gecco/SiperNKSAT26)

- The accessible CPS HTML says each run evaluates 1,320 offspring and also says this equals 100 generations of 66 offspring, which is arithmetically inconsistent. The exact charged budget should be checked in code or the final archival artefact before using its efficiency number as a protocol precedent. [CPS full preprint, experimental setup](https://arxiv.org/html/2602.07659v1)

- The accessible ProFiT primary page says the gains are statistically significant but does not expose the exact test, multiplicity correction, or per-comparison sample size. Those details should be verified from the paper before presenting ProFiT as a statistical model for ManiGuard. [ProFiT SSRN record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5889762)

## What makes his GECCO papers venue-appropriate?

### Takeaway

The papers are GECCO-appropriate because the evolutionary mechanism is usually the scientific object, not incidental tuning. Each paper defines a genotype or archive, a variation and selection process, and an empirical question about complexification, diversity, locality, transfer, or evaluation efficiency. Later papers also pair evolutionary novelty with contemporary LLM capabilities without replacing the evolutionary analysis with prompt engineering alone.

### Cited Findings

- ATEP asks a canonical neuroevolution and open-endedness question: whether augmenting network topology and species-aware transfer let co-evolving agents keep pace with increasingly complex environments. The comparisons include two fixed-topology EPOET baselines and transfer ablations. [ATEP paper](https://arxiv.org/pdf/2210.11442)

- LLMatic contributes a dual-archive cooperative QD algorithm rather than merely asking an LLM for a network. The paper specifies behavioural descriptors, curiosity-based prompt selection, mutation-crossover probabilities, a finite 2,000-evaluation budget, random and component ablations, and comparison with random search, GENIUS, DARTS, and Lambda-DARTS. [LLMatic paper](https://arxiv.org/pdf/2306.01102)

- Mortar defines an evolutionary representation, four LLM-implemented variation operators, archive descriptors, an MCTS-based contextual evaluator, a new contribution score, and ablations for the evaluator. It also tests whether the automated fitness corresponds to human preference, instead of relying only on visually appealing examples. [Mortar paper](https://arxiv.org/html/2601.00105v1)

- ProFiT frames LLM analysis and rewriting as mutation inside a closed evolutionary feedback loop over executable programs. Its benchmark spans multiple starting strategies, assets, temporal folds, and non-evolutionary baselines. [ProFiT SSRN record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5889762)

- CPS makes operator design and genotype-phenotype locality explicit. It measures how latent distance changes syntax and action traces, then isolates structured and learned mutation under an otherwise fixed evolution strategy and fixed budgets. This directly addresses sample efficiency and causal attribution within evolutionary search. [CPS full preprint](https://arxiv.org/html/2602.07659v1)

- The publication progression also shows increasing main-track strength: ATEP was a 2023 Companion paper, LLMatic was a nine-page 2024 main-proceedings paper, Mortar and ProFiT were nine-page and eight-page 2026 main-proceedings papers, while CPS remained a 2026 Companion paper. [ATEP DBLP](https://dblp.org/rec/conf/gecco/NasirBJC23); [LLMatic ACM/SIGEVO](https://www.acm.org/binaries/content/assets/sigs/sgb/annual-reports/fy-24-annual-report/appendix-d-fy24-sig-annual-reports.pdf); [GECCO 2026 proceedings](https://dblp.org/db/conf/gecco/gecco2026.html); [CPS DBLP](https://dblp.org/rec/conf/gecco/SiperNKSAT26)

### Inferences

- The characteristic contribution style is a hybrid systems paper with one clear EC mechanism: change the controller representation, add an archive, change the variation operator, or redesign contextual fitness. Venue fit then comes from asking why that evolutionary ingredient helps, supported by component ablations and search-behaviour metrics. [ATEP](https://arxiv.org/pdf/2210.11442); [LLMatic](https://arxiv.org/pdf/2306.01102); [Mortar](https://arxiv.org/html/2601.00105v1); [CPS](https://arxiv.org/html/2602.07659v1)

- Another recurring venue-relevant feature is the quality-versus-diversity or quality-versus-complexity tradeoff. ATEP relates solved environments to controller nodes and function evaluations, LLMatic illuminates architectures by compute and shape, Mortar combines archive coverage with mechanic contribution, and CPS contrasts typical efficiency with rare peak performance. [ATEP](https://arxiv.org/pdf/2210.11442); [LLMatic](https://arxiv.org/pdf/2306.01102); [Mortar](https://arxiv.org/html/2601.00105v1); [CPS](https://arxiv.org/html/2602.07659v1)

- Nasir looks especially relevant as a source of advice on GECCO framing, neuroevolution, QD descriptors, LLM-assisted variation, ablation design, and claims about operator efficiency. The portfolio establishes AI-and-games, open-ended learning, and PCG as his declared specialties, not robot manipulation or safe RL. [Author portfolio](https://umair-nasir14.github.io/)

### Gaps

- Proceedings status alone does not reveal Nasir's individual role on multi-author papers. Author order suggests leadership on ATEP, LLMatic, and Mortar, but contribution statements were not found in the accessible papers, so this review does not assign specific implementation or writing credit. [ATEP](https://arxiv.org/pdf/2210.11442); [LLMatic](https://arxiv.org/pdf/2306.01102); [Mortar](https://arxiv.org/html/2601.00105v1)

- There is no consistent statistical standard across the five papers. Some use large seed counts and standard deviations, some use five or two runs due cost, and CPS reports medians and maxima without a named significance test in the accessible results. A GECCO submission inspired by this work still needs its own prospective power, uncertainty, and interaction-accounting design. [ATEP](https://arxiv.org/pdf/2210.11442); [LLMatic](https://arxiv.org/pdf/2306.01102); [Mortar](https://arxiv.org/html/2601.00105v1); [CPS](https://arxiv.org/html/2602.07659v1)

## Which ideas are technically adjacent to or clearly separate from shared native-noise steering in ManiGuard?

### Takeaway

CPS is the closest technical analogue because it compares isotropic noise with semantically restricted and learned latent mutation under a fixed optimizer and matched budgets. ATEP is the closest conceptual analogue for transfer and generalisation across environments. LLMatic and Mortar are useful mainly for later QD repertoire design. None of the papers studies physical safety, instruction-conditioned manipulation, frozen VLA policies, or one shared adapter across policy families, so they should inform method and evaluation rather than be presented as direct evidence that ManiGuard native-noise steering will work.

### Cited Findings

- CPS directly supports testing mutation geometry rather than assuming all latent directions are equally useful. Its diagnostics measure decode validity, structural change, and action-sequence divergence, and its controlled comparison separates full-space Gaussian mutation from semantic block mutation and learned directions under the same evolution strategy. [CPS full preprint](https://arxiv.org/html/2602.07659v1)

- CPS is also adjacent to response-informed coordinates and bounded trust regions: it empirically locates a perturbation radius where decoded behaviour stays local, then constrains variation to semantically meaningful subspaces. The relevant transferable principle is to measure the latent-to-behaviour map before allocating physical evolution, not to reuse its trading-specific long-entry, short-entry, long-exit, and short-exit factorization. [CPS full preprint](https://arxiv.org/html/2602.07659v1)

- CPS remains technically separate in decisive ways. It trains a new transformer VAE and flow mutation model over a typed program DSL, executes deterministic financial backtests, and optimizes Sharpe. It does not perturb a frozen generative policy's native noise, condition a deployed controller on images and language instructions, or report physical violations and unfinished episodes. [CPS full preprint](https://arxiv.org/html/2602.07659v1)

- ATEP offers a useful warning and a useful design idea. Species-based partial transfer preserves more population diversity than wholesale replacement, and cross-environment testing reveals whether later agents acquire broader competence. However, ATEP evolves entire agent topologies and procedurally generated environments from scratch, uses family-specific populations, and spends more function evaluations than EPOET. It is not evidence for a small shared adapter under matched physical interaction budgets. [ATEP paper](https://arxiv.org/pdf/2210.11442)

- LLMatic is relevant if the project later introduces a QD repertoire of steering controllers. It demonstrates that descriptors should correspond to an intended deployment tradeoff, such as compute and architecture shape, and that archive coverage, elite quality, and operator ablations should all be measured. Its neural-architecture candidates and image-classification fitness have no direct mapping to safe robot execution. [LLMatic paper](https://arxiv.org/pdf/2306.01102)

- Mortar is similarly relevant to contextual repertoire evaluation. It scores a component through the complete systems it helps create, uses both outcome quality and archive coverage, and compares tree-based evaluation with random, greedy, and LLM alternatives. Its mechanic categories, AST complexity, agent skill ordering, and human game-preference study are separate from policy-family transfer and physical safety. [Mortar paper](https://arxiv.org/html/2601.00105v1)

- ProFiT is adjacent at the level of outcome-informed variation and strict temporal separation. It feeds performance analysis back into code mutation and evaluates across walk-forward folds and multiple assets. It evolves executable trading strategies in offline historical data, however, not a bounded observation-conditioned correction to a frozen policy, and financial risk-adjusted return is not a physical safety measure. [ProFiT SSRN record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5889762)

- None of the five papers reports the outcome tuple required for ManiGuard: safe completion, physical violation, and unfinished execution together. ATEP reports reward and environment solving, LLMatic reports classification accuracy and archive diversity, Mortar reports CITS, rank correlation and playability, and the trading papers report return or Sharpe. [ATEP](https://arxiv.org/pdf/2210.11442); [LLMatic](https://arxiv.org/pdf/2306.01102); [Mortar](https://arxiv.org/html/2601.00105v1); [ProFiT](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5889762); [CPS](https://arxiv.org/html/2602.07659v1)

### Inferences

- The strongest borrowing opportunity is an evaluation pattern, not a ready-made algorithm. Freeze the outer EA and budget, compare native isotropic noise with structured but unlearned coordinates and response-informed or learned coordinates, and measure decoded-action locality before physical trials. This mirrors CPS's causal isolation while respecting ManiGuard's frozen policy interface. [CPS full preprint](https://arxiv.org/html/2602.07659v1)

- A Nasir-style GECCO framing would make a distinctive evolutionary mechanism testable. For ManiGuard, that could be whether structured mutation, transfer or archive allocation improves safe completion per total interaction, not simply whether an EA eventually finds a good controller. CPS's fixed-optimizer comparison and ATEP's transfer ablations show the expected level of mechanism-specific evidence. [CPS](https://arxiv.org/html/2602.07659v1); [ATEP](https://arxiv.org/pdf/2210.11442)

- The portfolio reinforces the need to keep representation, optimizer, and generalisation claims separate. A useful steering interface can beat native execution without establishing an evolutionary advantage, and an optimizer can win within one family without establishing cross-family transfer. Nasir's papers use different experiments for representation quality, operator quality, and unseen-case behaviour, which is the right decomposition even though their domains differ. [ATEP](https://arxiv.org/pdf/2210.11442); [LLMatic](https://arxiv.org/pdf/2306.01102); [CPS](https://arxiv.org/html/2602.07659v1)

- His record does not justify a novelty claim for QD, learned mutation, latent trust regions, code-level LLM evolution, open-ended transfer, or variable-topology controllers by themselves. The plausible value of engaging him is methodological guidance on how to turn the specific ManiGuard combination into a controlled EC contribution, while manipulation safety expertise still needs separate grounding. [Author portfolio](https://umair-nasir14.github.io/); [ATEP](https://arxiv.org/pdf/2210.11442); [LLMatic](https://arxiv.org/pdf/2306.01102); [CPS](https://arxiv.org/html/2602.07659v1)

### Gaps

- No reviewed paper tests native-noise steering in a diffusion or flow policy, frozen VLA adaptation, language-conditioned physical control, safety-monitor labels as an evolutionary objective, or transfer of one unchanged controller across separately fine-tuned policy families. [ATEP](https://arxiv.org/pdf/2210.11442); [LLMatic](https://arxiv.org/pdf/2306.01102); [Mortar](https://arxiv.org/html/2601.00105v1); [ProFiT](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5889762); [CPS](https://arxiv.org/html/2602.07659v1)

- The CPS learned mutation operator uses logged evolutionary outcomes, but the accessible preprint does not establish whether one learned operator transfers unchanged across all five assets. This distinction matters before analogizing it to one shared ManiGuard controller. [CPS full preprint](https://arxiv.org/html/2602.07659v1)

- No paper supplies a direct precedent for ManiGuard's full accounting requirement, where every simulator interaction, discarded candidate, adaptation episode, model query, violation, and timeout remains charged. ATEP explicitly uses unequal function evaluations, LLMatic reports candidate evaluations, Mortar reports API costs and five runs, and CPS aims at matched budgets but contains the budget arithmetic inconsistency noted above. [ATEP](https://arxiv.org/pdf/2210.11442); [LLMatic](https://arxiv.org/pdf/2306.01102); [Mortar](https://arxiv.org/html/2601.00105v1); [CPS](https://arxiv.org/html/2602.07659v1)
