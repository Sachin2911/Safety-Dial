# GECCO fit, tooling, compute and evaluation standards for evolutionary safe RL with a learned world model

Notes compiled 3 October 2026. "Verified" means I read the abstract or full text (arXiv or official page). "Title only" means the paper appears on an official GECCO accepted list, but I could not read its content, so the topic shown is a guess from the title. Peer-reviewed status: GECCO full papers appear in the main ACM proceedings; posters appear in the companion proceedings (as 2-4 page extended abstracts); arXiv-only items are preprints. Paper lists come from the official GECCO "Accepted Papers" and "Accepted Posters" pages. dblp and the ACM DL could not be fetched because of bot-protection pages, so these lists have not been cross-checked against the ACM DL tables of contents.

## Q1. Which GECCO 2023-2026 papers resemble this project, and which were best-paper nominees in NE/EML?

### Takeaway
NE and EML each accept roughly 10-18 full papers a year. Only a minority deal with RL or control. These are dominated by quality-diversity RL (QD-RL) from Cully's lab and InstaDeep, by ES/RL hybrids, and by GPU/JAX-accelerated neuroevolution evaluated on Brax. I screened the titles of every 2023-2026 full paper and poster. None covers evolutionary methods on a constrained-MDP safe-RL benchmark (Safety-Gymnasium/CMDP), and none uses a learned world model as the fitness evaluator for neuroevolution. The closest work is: surrogate-assisted evolutionary multi-agent RL (EML 2026, title only); safe CMA-ES on benchmark functions (ENUM 2024); and target-free distributional QD-RL (EML 2026 best-paper nominee). The niche therefore looks open, but no GECCO precedent defines the reviewers' expectations.

### Cited Findings
**Volume and acceptance**
- GECCO accepted 180 full papers in 2023 ([2023 list](https://gecco-2023.sigevo.org/Accepted-Papers)), 178 in 2024 ([2024 list](https://gecco-2024.sigevo.org/Accepted-Papers)), 181 in 2025 ([2025 list](https://gecco-2025.sigevo.org/Accepted-Papers)) and 149 in 2026 ([2026 list](https://gecco-2026.sigevo.org/Accepted-Papers)).
- Counts from those lists by track prefix:

  | Year | NE full papers | EML full papers |
  |---|---|---|
  | 2023 | 9 | 26 |
  | 2024 | 9 | 22 |
  | 2025 | 11 | 18 |
  | 2026 | 8 | 17 |

  These counts come from parsing the track prefixes on the list pages. The 2025 counts match SIGEVO's official per-track chart.
- GECCO 2025 received 501 full-paper submissions and accepted 181 (36.1%). Over the last three years submissions have stayed near 500 and acceptance near 36.5%. SIGEVO set a maximum target acceptance rate of 40% per track Source: [SIGEVOlution vol. 18 issue 4, Dec 2025](https://sigevo.hosting.acm.org/public_html/sigevolution/?p=1241).
- Per-track figures for 2025: NE received 28 submissions and accepted 11 (39%); EML received 48 and accepted 18 (38%) Source: [per-track chart](https://sigevo.hosting.acm.org/public_html/sigevolution/wp-content/uploads/2025/12/gecco25_submissions_acceptance.png).

**GECCO 2023 (Lisbon)**
- Best-paper nominees ([2023 nominations](https://gecco-2023.sigevo.org/Best-Paper-Nominations)):
  - NE: Peng et al., "Fast Evolutionary Neural Architecture Search by Contrastive Predictor with Linear Regions" (NAS).
  - EML: Fontaine & Nikolaidis, "Covariance Matrix Adaptation MAP-Annealing" (CMA-MAE); Shiraishi et al., "Fuzzy-UCS Revisited"; Videau et al., "Interactive Latent Diffusion Model".
- NE, Lim, Flageat & Cully, "Understanding the Synergies between Quality-Diversity and Deep Reinforcement Learning" (verified). Proposes a Generalized Actor-Critic QD-RL framework and introduces PGA-ME (SAC) and PGA-ME (DroQ), which solve Humanoid. Finds that actor-critics in QD-RL are under-trained. Uses 5-10 replications, Wilcoxon rank-sum with Bonferroni correction Source: [arXiv 2303.06164](https://arxiv.org/abs/2303.06164).
- EML, Zheng & Cheng, "Rethinking Population-assisted Off-policy Reinforcement Learning" (verified). Shows that population data in a shared replay buffer can destabilise off-policy RL in evolutionary-RL (ERL) hybrids, and proposes a double replay buffer. Gym locomotion tasks, 10 seeds, 3M steps Source: [arXiv 2305.02949](https://arxiv.org/abs/2305.02949).
- Other 2023 full papers, title only ([2023 list](https://gecco-2023.sigevo.org/Accepted-Papers)):
  - EML: Tran, Doan & Luong, "A Two-Stage Multi-Objective Evolutionary Reinforcement Learning Framework for Continuous Robot Control".
  - EML: Aydeniz, Loftin & Tumer, "Novelty Seeking Multiagent Evolutionary Reinforcement Learning".
  - NE: Nguyen et al., "Stable and Sample-Efficient Policy Search for Continuous Control via Hybridizing Phenotypic Evolutionary Algorithm with the Double Actors Regularized Critics".
  - NE: Macé et al., "The Quality-Diversity Transformer".
  - NE: Pedersen & Risi, "Learning to Act through Evolution of Neural Diversity in Random Neural Networks".
  - CS: Faldor et al., "MAP-Elites with Descriptor-Conditioned Gradients and Archive Distillation into a Single Policy".
  - CS: Grillotti et al., "Don't Bet on Luck Alone…" (behavioural reproducibility of QD in uncertain domains).
  - CS: Tjanaka et al., "pyribs: A Bare-Bones Python Library for Quality Diversity Optimization" (a tooling paper accepted as a full paper).
- 2023 posters, title only: "evosax: JAX-Based Evolution Strategies" (Lange, NE); "Evolution Strategies with Seed Mirroring and End Tournament"; "Overcoming Deceptive Rewards with Quality-Diversity" (EML); "Generative Adversarial Neuroevolution for Control Behaviour Imitation" (NE) Source: [2023 posters](https://gecco-2023.sigevo.org/Accepted-Posters).

**GECCO 2024 (Melbourne)**
- Best-paper nominees ([2024 nominations](https://gecco-2024.sigevo.org/Best-Paper-Nominations)):
  - NE: Wang et al., "Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration".
  - EML: "Survival-LCS"; "NEvoFed"; Dixit & Tumer, "Informed Diversity Search for Learning in Asymmetric Multiagent Systems".
  - CS: Templier, Grillotti, Rachelson, Wilson & Cully, "Quality with Just Enough Diversity in Evolutionary Policy Search".
- NE, TensorNEAT (verified). NEAT tensorised in JAX with Gym, Brax and gymnax support; up to 500x speedup over NEAT-Python on Brax control. On Swimmer with population 10,000, an RTX 4090 took 215.7 s against 42,279.6 s on a CPU. 10 trials with 95% confidence intervals (CIs) Source: [arXiv 2404.01817](https://arxiv.org/abs/2404.01817).
- CS, JEDi (verified). Learns the behaviour-to-fitness relationship so that ES evaluations focus on promising regions. Beats both QD and ES on mazes and on Brax control with large policies. Uses Mann-Whitney U tests, at least 10 seeds on mazes and 5 on Brax, and runs LM-MA-ES from evosax on the Brax tasks Source: [arXiv 2405.04308](https://arxiv.org/abs/2405.04308).
- NE, Flageat, Lim & Cully, "Enhancing MAP-Elites with Multiple Parallel Evolution Strategies" (MEMES, verified). Runs up to about 100 ES emitters on one GPU and beats gradient-based and mutation-based QD on QD-RL tasks. 10 seeds, Wilcoxon rank-sum with Bonferroni correction, wall-clock times measured on the same hardware Source: [arXiv 2303.06137](https://arxiv.org/abs/2303.06137).
- ENUM, Uchida et al., "CMA-ES for Safe Optimization" (verified). Safe CMA-ES projects samples into a safe region built from Lipschitz-constant estimates obtained by Gaussian-process regression, which cuts unsafe evaluations. Tested only on benchmark functions (no RL); medians and IQR over 50 trials Source: [arXiv 2405.10534](https://arxiv.org/abs/2405.10534).
- Other 2024 full papers, title only ([2024 list](https://gecco-2024.sigevo.org/Accepted-Papers)):
  - NE: Triebold & Yaman, "Evolving Generalist Controllers to Handle a Wide Range of Morphological Variations".
  - NE: Nasir, Earle, Togelius & James, "LLMatic: Neural Architecture Search via Large Language Models and Quality Diversity Optimization" (Wits co-authors).
  - EML: Sayar, Iacca & Knoll, "Multi-Objective Evolutionary Hindsight Experience Replay for Robot Manipulation Tasks".
  - ENUM: Porter & Arnold, "Direct Augmented Lagrangian Evolution Strategies".
  - L4EC: "Accelerate Evolution Strategy by Proximal Policy Optimization".
  - GP: Nadizar et al., "Searching for a Diversity of Interpretable Graph Control Policies".
- 2024 posters, title only: "Quality Diversity for Robot Learning: Limitations and Future Directions" (Batra, Tjanaka et al., NE); "Large Language Models As Evolution Strategies" (Lange, Tian & Tang, L4EC); "Generational Information Transfer with Neuroevolution on Control Tasks" (NE) Source: [2024 posters](https://gecco-2024.sigevo.org/Accepted-Posters).

**GECCO 2025 (Málaga)**
- Best-paper nominees ([2025 nominations](https://gecco-2025.sigevo.org/Best-Paper-Nominations)):
  - NE: "Evolving Comprehensive Proxies for Zero-Shot NAS"; "Competition and Attraction Improve Model Fusion" (Sakana AI); "Visualizing the Dynamics of Neuroevolution with Genetic Distance Projections".
  - EML: "Evolutionary Quadtree Pooling for CNNs"; De La Torre, Nadizar et al., "Evolution of Inherently Interpretable Visual Control Policies"; Gonzalez, Dixit & Tumer, "Dynamic Influence For Coevolutionary Agents".
  - BBSR: Nisioti et al., "When Does Neuroevolution Outcompete Reinforcement Learning in Transfer Learning Tasks?"
  - CS: Flageat et al., "Extract-QD Framework".
- BBSR, Nisioti, Pedersen, Plantec, Montero & Risi (verified). Introduces two benchmarks, "stepping gates" and "ecorobot" (Brax plus obstacles and switchable morphologies). Neuroevolution (NE) methods frequently beat RL baselines (PPO, goal-conditioned PPO) at transfer. 10 trials, Kruskal-Wallis then Mann-Whitney U post-hoc; uses evosax and tensorneat Source: [arXiv 2505.22696](https://arxiv.org/abs/2505.22696).
- NE, Mitsides, Faldor & Cully, "Scaling Policy Gradient Quality-Diversity with Massive Parallelization via Behavioral Variations" (ASCII-ME, verified). MAP-Elites with policy-gradient behavioural variations and no central actor-critic. Produces diverse DNN policies in under 250 s on a single GPU and is about 5x faster than state-of-the-art QD-RL. 20 seeds; Wilcoxon-Mann-Whitney U with Holm-Bonferroni correction; 1M-evaluation budget; compared at equal runtime and equal evaluations on an L40S Source: [arXiv 2501.18723](https://arxiv.org/abs/2501.18723).
- Other 2025 full papers, title only ([2025 list](https://gecco-2025.sigevo.org/Accepted-Papers)):
  - NE: Feiden & Garcke, "Diversity in Reinforcement Learning Through the Occupancy Measure".
  - EML: "Dataset Reduction for Offline Reinforcement Learning using Genetic Algorithms…".
  - EML: "PropNEAT".
  - GP: "MAPLE: Multi-Action Programs through Linear Evolution for Continuous Multi-Action Reinforcement Learning".
  - CS: Bahlous-Boldi, Faldor et al., "Dominated Novelty Search".
- 2025 posters ([2025 posters](https://gecco-2025.sigevo.org/Accepted-Posters)):
  - NE: Mustafaoglu, Pingali & Miikkulainen, "Evolutionary Policy Optimization". According to a search snippet of [arXiv 2504.12568](https://arxiv.org/abs/2504.12568), it hybridises neuroevolution with policy gradients, tested on Atari Pong and Breakout. Do not confuse it with a different "Evolutionary Policy Optimization" by Wang et al. ([arXiv 2503.19037](https://arxiv.org/abs/2503.19037), CMU).
  - EML: "Unifying Zeroth-Order Optimization and Genetic Algorithms for Reinforcement" (title only).
  - EMO: "Uncertainty Quantification of the Hypervolume for Evolutionary Multi-Objective Reinforcement Learning" (title only).

**GECCO 2026 (San José, Costa Rica)**
- Best-paper nominees ([2026 nominations](https://gecco-2026.sigevo.org/Best+Paper+Nominations)):
  - NE: Shahabi Sani et al., "Evolving Multi-Channel Confidence-Aware Activation Functions for Missing Data" (not RL).
  - EML: Nasir, Li, James & Togelius, "Mortar: Evolving Mechanics for Automatic Game Design" (Wits); Koohy & Bayne, "Distributional Value Estimation Without Target Networks for Robust Quality-Diversity".
- EML, QDHUAC (Koohy & Bayne, verified). A target-free distributional critic that allows high update-to-data training for Dominated Novelty Search, needing an order of magnitude fewer environment steps on Brax. Its motivation notes that QD "often require[s] tens of millions of environment steps"; it reports N=5 seeds Source: [arXiv 2604.20381](https://arxiv.org/abs/2604.20381) (2026-04-22, "Accepted as Full Paper at GECCO'26").
- Other 2026 full papers, title only ([2026 list](https://gecco-2026.sigevo.org/Accepted-Papers)):
  - EML: Yu, Wang, Li et al. (Tsinghua), "Surrogate-Assisted Evolutionary Multi-Agent Reinforcement Learning with Adaptive Fitness Evaluation". No preprint found.
  - EML: Sygkounas et al., "Evolutionary Discovery of Reinforcement Learning Algorithms via Large Language Models"; "COvolve".
  - NE: Ajani et al., "Measuring Intrinsic Dimension of Multiobjective Landscapes and Dimensionality-Reduced Neuroevolution of Deep Reinforcement Learning". No preprint found.
  - NE: Monsia, Young, Francon & Miikkulainen, "Optimizing Chlorination in Water Distribution Systems via Surrogate-assisted Neuroevolution".
  - ECOM: Uchida et al., "Adaptive Stochastic Natural Gradient Method for Safe Optimization on Binary Space".
  - ENUM: "Evaluation of Element-wise Effectiveness Estimation for Augmented Lagrangian CMA-ES".
  - GECH: "Interactive LLM-assisted curriculum learning for multi-task policy search".
  - SI: Van Der Bank & Nitschke (UCT), "Dynamic Descriptor Mutations Improve Automated Quality-Diversity".
- 2026 posters ([2026 posters](https://gecco-2026.sigevo.org/Accepted-Posters)):
  - CS: Pitzalis, Nisioti et al., "Continual Evolution Strategies in Control Tasks" (verified). ES on sequential MuJoCo tasks (Hopper-v5, Walker2d-v5, Swimmer-v5). Naive ES forgets catastrophically; replay helps Source: [arXiv 2608.13600](https://arxiv.org/abs/2608.13600).
  - NE: Lorenc & Neruda, "Utilizing Evolution Strategies to Train Transformers in Reinforcement Learning" (verified). OpenAI-ES trains a Decision Transformer on MuJoCo Humanoid and Atari Source: [arXiv 2501.13883](https://arxiv.org/abs/2501.13883).
  - Title only: "MO-CEM-RL" (EMO); "Combining Policy Gradients with Quality-Diversity in Cooperative MARL" (CS); "A Multi-Factorial PSO for Balancing Safety and Efficiency for Robot Navigation" (SI); "QDA-VC: A Benchmark for Quality-Diversity Algorithms with Variable Constraints" (BBSR); "Constrained Optimization using an Evolutionary-Augmented Lagrangian approach" (GECH).

**Title screen for this project's topic**
- I keyword-screened every 2023-2026 full-paper and poster title for "safe", "safety", "constrain", "Lagrang", "CMDP", "world model", "latent" and "model-based". The only safety hits are safe *black-box optimisation* (CMA-ES 2024; ASNG 2026), RWA autonomous-vehicle and robot work, and constrained multi-objective optimisation: same official lists as above.

**Older precedents (outside 2023-2026)**
- Stork, Zaefferer, Bartz-Beielstein & Eiben, "Surrogate Models for Enhancing the Efficiency of Neuroevolution in Reinforcement Learning", GECCO 2019. Uses kernel surrogates with phenotypic distance Source: [arXiv 1907.09300](https://arxiv.org/abs/1907.09300).
- Ha & Schmidhuber, "Recurrent World Models Facilitate Policy Evolution", NeurIPS 2018 oral. Evolves compact controllers on world-model features, trains entirely inside the model's "dream", then transfers back Source: [arXiv 1809.01999](https://arxiv.org/abs/1809.01999).
- Wang et al., "A Surrogate-Assisted Controller for Expensive Evolutionary Reinforcement Learning", 2022 preprint. Uses the RL critic as a fitness surrogate and needs strategies to avoid surrogate-induced false minima Source: [arXiv 2201.00129](https://arxiv.org/abs/2201.00129).
- Tang et al., PE-SAERL, a surrogate via policy embedding with up to 7x speedup on Atari (submitted to BIC-TA 2022) Source: [arXiv 2301.13374](https://arxiv.org/abs/2301.13374).

### Inferences
- The recurring winning recipe in NE/EML/CS RL papers is: a clear algorithmic idea, GPU/JAX parallelism, Brax locomotion tasks, QD or ES baselines from QDax/evosax, and 10-20 seeds with rank-based tests. Analysis or benchmark papers also succeed: Nisioti et al. was a best-paper nominee in BBSR, and Zheng & Cheng was an analysis paper.
- A "surrogate-assisted (world-model) neuroevolution for constrained RL" framing matches explicit NE-track topics (surrogate-assisted, multi-objective neuroevolution, RL applications; see Q2), so NE looks like the natural primary track. EML is viable, but its call explicitly asks for comparisons with state-of-the-art non-evolutionary ML, which means PPO-Lagrangian and similar here.
- Wits co-authors have published in GECCO NE (2024) and EML (2026, best-paper nominee). Local supervisors probably know these track cultures, which is worth exploiting for internal pre-review.
- The 2026 EML Tsinghua surrogate-assisted ERL paper is the closest direct competitor in spirit. Obtain it from the ACM DL before writing related work.

### Gaps
- I could not access ACM DL tables of contents or dblp (bot-protection pages). The lists here come from the official GECCO accepted-paper and poster pages and may omit late changes, Hot-off-the-Press items and workshop papers.
- Content of every "title only" paper is unverified, including the two most relevant 2026 papers (Tsinghua surrogate-assisted EMARL; Ajani et al.). Neither had an arXiv preprint I could find.
- Per-track submission and acceptance numbers for 2023, 2024 and 2026 were not found. The 149 full papers in 2026, against 181 in 2025, suggest fewer submissions or a stricter cap, but this is unconfirmed.
- Workshop-paper proceedings (companion volume) were not screened for evolutionary safe RL.

## Q2. What do the NE and EML calls say about scope, what are the submission rules, and which workshops could serve as a fallback?

### Takeaway
The 2026 NE call explicitly lists surrogate-assisted, multi-objective and parallel neuroevolution, plus RL applications, all of which fit this topic. EML asks for comparisons against strong non-evolutionary ML baselines. Rules for 2026:
- 8-page full papers, excluding references, with any appendix submitted as supplementary material.
- Double-blind, per-track review.
- At most 40% acceptance per track.
- Non-extensible deadlines in late January.

As of today the 2027 website is a copy of the 2026 pages, so the 2027 deadlines and workshops are not yet published. The 2026 workshops closest to this topic are LLMs for and with EC, Surrogate-Assisted Evolutionary Optimisation, and EC for Autonomous Cyber-Physical Systems (which covers safety assurance). Their deadlines fell in late March or early April.

### Cited Findings
- NE track (2026, chairs Dennis Wilson and Christian Gagné). Scope includes "Surrogate-assisted approaches", "Multi-objective Neuroevolution", "Applications to reinforcement, supervised, and unsupervised learning", "Parallelized implementations", interpretable models and evolutionary robotics Source: [NE track page](https://gecco-2026.sigevo.org/Track?itemId=59).
- EML track (2026, chairs Ryan Urbanowicz and Will N. Browne). Covers EC for supervised, unsupervised, semi-supervised and reinforcement learning, and parallel EML on GPUs/TPUs. "Authors are strongly encouraged to compare their EML approaches to the corresponding state-of-the-art non-evolutionary ML methods." Work using ML *for* EC is redirected to L4EC Source: [EML track page](https://gecco-2026.sigevo.org/Track?itemId=53).
- BBSR track covers benchmarking methodology, benchmark problems and toolboxes, statistical analysis and visualisation, reproducibility studies and software. Reproducibility studies must share all artefacts publicly at submission; anonymous repositories (Zenodo, anonymous GitHub) are strongly encouraged Source: [BBSR track page](https://gecco-2026.sigevo.org/Track?itemId=2672).
- 2026 call for papers ([2026 CFP](https://gecco-2026.sigevo.org/Call-for-Papers)):
  - Full papers are at most 8 pages excluding references; any content beyond that goes in supplementary material.
  - Review is double-blind and per track. Criteria: "significance of the work, technical soundness, novelty, clarity, writing quality, relevance and, if applicable, sufficiency of information to permit replication."
  - Poster-only papers are at most 4 pages including references.
  - Abstract deadline 19 January 2026; full papers 26 January 2026 (non-extensible); camera-ready 10 April 2026.
  - New regulation: at most CEILING(40% × submissions) full papers per track.
  - Simultaneous arXiv posting is allowed.
  - Supplementary material (≤10 MB, anonymised) may include source code, which is encouraged. It "may or may not be considered" by reviewers.
  - Generative-AI assistance must be disclosed in the Acknowledgments.
  - From 2026, ACM is fully Open Access, so APCs apply to authors from non-ACM-Open institutions (a temporary 2026 subsidy existed).
- Notification for 2026 was 20 March 2026 Source: [2027-site "Important Dates" page (currently showing 2026 dates)](https://gecco-2027.sigevo.org/Important-Dates).
- GECCO 2027 will be held 12-16 July 2027 in Kraków, in person only Source: [GECCO 2027 home](https://gecco-2027.sigevo.org/). Its Important-Dates, Tracks and Workshops pages currently show the 2026 dates, the identical 2026 track chairs and the 2026 workshop titles (for example "IAM 2026", "BENCH@GECCO26") Source: [2027 Tracks](https://gecco-2027.sigevo.org/Tracks), [2027 Workshops](https://gecco-2027.sigevo.org/Workshops). The 2027 call-for-papers URL also renders the 2026 text Source: [2027 CFP URL](https://gecco-2027.sigevo.org/Call-for-Papers).
- GECCO 2026 workshops ([2026 workshops](https://gecco-2026.sigevo.org/Workshops)): IAM; ECADA; Evolutionary Rule-based ML; Analysing algorithmic behaviour; BENCH; Decomposition; EC & Explainable AI; EC for Autonomous Cyber-Physical Systems; Evolving self-organisation; Graph-based GP; LLMs for and with EC; Latin American applications; Multimodal Data-Driven Optimization and Learning; Open Source Software for EC; Program Synthesis; Quantum Optimization; Surrogate-Assisted Evolutionary Optimisation.
- LLMs for and with EC (2026) asks for "LLM-based iterative frameworks (with evolutionary principles)", "LLM-Guided Evolutionary Algorithms", LLM-driven variation operators (for example LMX, ELM, QD through AI feedback) and benchmarking of generated metaheuristics Source: [workshop page](https://gecco-2026.sigevo.org/Workshop?itemId=8250).
- Surrogate-Assisted Evolutionary Optimisation (2026) lists ML techniques for constructing surrogates, model management, multi-fidelity surrogates and "Surrogate-assisted identification of the feasible region". It accepts full papers (8 pages plus references, APC payable) and extended abstracts (up to 4 pages, no APC). Submission deadline was 27 March 2026, extended to 3 April 2026; notification 24-26 April; camera-ready 5 May Source: [workshop page](https://gecco-2026.sigevo.org/Workshop?itemId=8255).
- EC for Autonomous Cyber-Physical Systems (2026) lists "Safety assurance of ACPS", digital twins, uncertainty handling and "Reducing the simulation-reality gap" Source: [workshop page](https://gecco-2026.sigevo.org/Workshop?itemId=8242).
- BENCH@GECCO26 took no paper submissions (its 2026 theme was dynamic optimisation) Source: [workshop page](https://gecco-2026.sigevo.org/Workshop?itemId=8239).
- GECCO 2025 had a "Neuroevolution at work" workshop. Its scope included neuroevolution for deep and reinforcement learning and "surrogate models for fitness estimation in neuroevolution and NAS". It does not appear in the 2026 list Source: [2025 workshop page](https://gecco-2025.sigevo.org/Workshop?itemId=2341).
- Workshop papers in 2026 were limited to 8 pages excluding references (individual workshops may lower this). Accepted full workshop papers are subject to the APC, while extended abstracts are not. SIGEVO said it could no longer offer the subsidy and pointed authors to a Financial Hardship Waiver Source: [2026 workshop-paper call](https://gecco-2026.sigevo.org/Call-for-Workshop-Papers).
- The 2026 call also offers Hot-off-the-Press for work recently published in top venues Source: [2026 CFP](https://gecco-2026.sigevo.org/Call-for-Papers).

### Inferences
- If 2027 follows the 2024-2026 pattern, expect a mandatory abstract around mid-to-late January 2027 and the full paper about a week later, both non-extensible. Workshop and poster fallbacks would then fall around late March or early April 2027, after the thesis is finished.
- A rejected full paper can be downgraded to a poster without re-review. The CFP says there is "no maximum acceptance rate on poster papers submitted originally as full paper". This makes a full-paper submission low-risk.
- Ranked by fit for a workshop fallback: the Surrogate-Assisted EO workshop (LeWM as surrogate) is the most natural; EC4CPS suits a safety framing; the LLM+EC workshop fits only if an LLM component is added. All of these depend on whether the workshops are renewed for 2027, which is unknown.

### Gaps
- The 2027 call for papers, deadlines, track chairs and workshop list were not yet published as of 3 October 2026; the 2027 pages are placeholders.
- I did not verify whether Wits is covered by ACM Open in 2027. A search snippet suggests Wits is a participant through the SANLiC agreement renewed for 2024-2026, but the ACM participants page did not render and the Wits LibGuide link returned 404. Check this before submitting, because it determines whether an APC is due.

## Q3. Which libraries and environments should be used, which are PyTorch or JAX, and what is realistic on one RTX 5090?

### Takeaway
JAX dominates the GECCO ecosystem for this kind of work: evosax, QDax and TensorNEAT, typically on Brax or MJX. Accelerated safe-RL environments now exist: CRAX (a June 2026 preprint built on MJX/Brax) includes Velocity tasks for Walker2D, Hopper and HalfCheetah, eleven PPO/SAC safe-RL baselines, and a PyTorch wrapper. On the PyTorch side, EvoTorch (Ray multi-CPU and GPU), EvoX (now PyTorch-based), pycma (with an augmented-Lagrangian constraint interface) and pyribs (framework-agnostic) fit the existing PyTorch LeWM and OmniSafe stack. Safety-Gymnasium, OmniSafe and SafePO are effectively frozen: their last releases date from 2023.

### Cited Findings
- **evosax** (JAX):
  - "30+ evolution strategies", including SimpleES, OpenAI-ES, CMA-ES, Sep-CMA-ES, xNES, SNES, MA-ES, LM-MA-ES, PGPE, ARS, CR-FM-NES, Guided ES, ASEBO, LES/Discovered ES, GA variants, DE, PSO and Diffusion Evolution.
  - Ask-eval-tell API with jit/vmap/scan; needs Python ≥3.10 and JAX Source: [evosax README](https://github.com/RobertTLange/evosax).
  - Latest release v.0.3.1 on 2026-08-17; about 800 stars; Apache-2.0 (GitHub API, checked 2026-10-03).
- **EvoJAX** (Google, JAX). The repository is archived (last push 2024-06-27). Its README claims Brax locomotion is solved "in tens of minutes" on a GPU Source: [EvoJAX](https://github.com/google/evojax).
- **QDax** (JAX, maintained by Imperial AIRL and InstaDeep):
  - Requires Python ≥3.11.
  - Algorithms: MAP-Elites, Dominated Novelty Search, AURORA, CVT-ME, PGA-ME, DCRL-ME, QDPG, CMA-ME, OMG-MEGA, CMA-MEGA, MOME, MEES, ME-PBT, ME-LS.
  - Baselines: NSGA2, SPEA2, PBT, DIAYN, DADS, SMERL.
  - v0.5.0 released 2025-05-27; last push 2025-10-30 Source: [QDax](https://github.com/adaptive-intelligent-robotics/QDax).
- **pyribs** (NumPy, ask/tell):
  - Implements CMA-ME, CMA-MEGA, CMA-MAE (and scalable variants) and Discount Model Search; Python ≥3.10.
  - v0.12.0 released 2026-07-22 Source: [pyribs](https://github.com/icaros-usc/pyribs).
  - PPGA (ICLR 2024) was implemented in pyribs with a CleanRL-based PPO Source: [PPGA arXiv 2305.13795](https://arxiv.org/abs/2305.13795).
- **EvoTorch** (PyTorch, NNAISENSE):
  - Algorithms: PGPE, XNES, CMA-ES, SNES, CEM, GA (which behaves like NSGA-II with multiple objectives), CoSyNE, MAPElites.
  - Supports RL tasks; Ray splits work across CPUs, GPUs and cluster nodes.
  - v0.6.1 released 2025-05-14; last push 2026-09-28 Source: [EvoTorch](https://github.com/nnaisense/evotorch).
- **EvoX**:
  - A "distributed GPU-accelerated" EC framework "compatible with PyTorch"; the old JAX version lives on the v0.9.0 branch.
  - 140+ algorithms across its ecosystem; integrates with Brax for neuroevolution and RL.
  - v1.4.0 released 2026-09-09; GPL-3.0 licence Source: [EvoX](https://github.com/EMI-Group/evox).
- **TensorNEAT**: JAX, GPL-3.0 Source: [TensorNEAT](https://github.com/EMI-Group/tensorneat).
- **pycma**: handles linear and nonlinear constraints via `fmin_con2` / `ConstrainedFitnessAL` (an augmented-Lagrangian interface), noise via `noise_handler`, and parallel evaluation via `n_jobs`. Release r4.5.0 on 2026-09-13 Source: [pycma](https://github.com/CMA-ES/pycma).
- **Nevergrad**: "Python toolbox for gradient-free optimization", MIT licence; 1.0.12 released 2025-04-23 Source: [Nevergrad](https://github.com/facebookresearch/nevergrad).
- **Brax**: v0.14.2 released 2026-03-15 Source: [Brax](https://github.com/google/brax).
- **MuJoCo Playground**:
  - GPU environments on MJX, now also supporting a MuJoCo Warp backend: DM Control, locomotion and manipulation.
  - Python ≥3.10 with `jax[cuda12]`; v0.2.0 released 2026-03-16 Source: [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground).
  - Paper: RSS 2025, [arXiv 2502.08844](https://arxiv.org/abs/2502.08844).
- **CRAX** (Tomilin et al., preprint):
  - Built on MJX/Brax, with "up to 200x faster training" than CPU safety benchmarks. Eight tasks, three difficulty levels, seven safe-RL methods evaluated Source: [arXiv 2606.20376](https://arxiv.org/abs/2606.20376) (v1 2026-06-18, v4 2026-09-29).
  - The repository lists a **Velocity** task for Ant, Humanoid, HalfCheetah, Hopper, Swimmer and Walker2D, plus Height, Pathway, Lift, Goal, Circle, Button, Push and Reacher.
  - Algorithms: PPO, PPO-Cost, PPO-Lag, PPO-PID, PPO-Saute, FOCOPS, P3O, CRPO, SAC, SAC-Lag, SAC-PID.
  - Install extras include `torch` ("The PyTorch wrapper"), `gym` and `cuda` (`jax[cuda12]`); Apache-2.0; 15 stars; created 2025-12-03; last push 2026-10-01 Source: [CRAX repo](https://github.com/TTomilin/CRAX).
  - Its Navigation and Velocity suites are "close reimplementations" of Safety Gym and Safety-Gymnasium. Velocity thresholds are "set relative to the speed an unconstrained agent of the same morphology reaches" (its Table 7) Source: [CRAX paper HTML](https://arxiv.org/html/2606.20376).
- **mjx-safety-gym** (Yarden As): an MJX port of Safety Gym that implements only Go-To-Goal. Python ≥3.11; 0 stars; last push 2025-08-27 Source: [mjx-safety-gym](https://github.com/yardenas/mjx-safety-gym).
- **Safety-Gymnasium**:
  - Latest release v1.2.0 (2023-08-22).
  - The README says "Python 3.11 is not supported for now, due to the incompatibility of pygame".
  - pyproject pins `mujoco == 2.3.3` and `gymnasium == 0.28.1`, and declares `requires-python >= 3.8` Source: [safety-gymnasium](https://github.com/PKU-Alignment/safety-gymnasium).
- **OmniSafe and SafePO**:
  - OmniSafe's last release was v0.5.0 (2023-05-27), with the last push in March 2025 Source: [OmniSafe](https://github.com/PKU-Alignment/omnisafe).
  - SafePO was last pushed in March 2024, and its instructions use conda Python 3.8 Source: [SafePO](https://github.com/PKU-Alignment/Safe-Policy-Optimization).
- **JAX on recent NVIDIA GPUs**:
  - On CUDA 13, JAX supports NVIDIA GPUs with SM ≥7.5 (driver ≥580 on Linux); on CUDA 12 it supports SM ≥5.2. JAX recommends migrating to the CUDA 13 wheels Source: [JAX installation docs](https://docs.jax.dev/en/latest/installation.html).
  - The RTX 5090 is compute capability sm_120, and PyTorch binaries need CUDA ≥12.8 for Blackwell Source: [PyTorch forum](https://discuss.pytorch.org/t/support-for-nvidia-rtx-5090-cuda-sm-120/223683).
- QDax uses Brax for evaluation; JEDi ran LM-MA-ES (evosax) on Brax tasks, citing memory limits of another ES variant with large genomes Source: [JEDi](https://arxiv.org/abs/2405.04308).

### Inferences
Three stacks are realistic:
- **(A) Keep the current CPU simulator.** Use the vendored Safety-Gymnasium SafetyWalker2dVelocity-v1 with 24 workers, a PyTorch-native optimiser (EvoTorch, pycma's `ConstrainedFitnessAL`, or pyribs for QD), and run LeWM on the 5090. This is the lowest-risk path. It keeps the existing OmniSafe PPO and PPO-Lag baselines valid, because the environment and thresholds are identical, and it avoids JAX/PyTorch interop.
- **(B) Port to CRAX Walker2D Velocity with evosax or QDax on the GPU.** This gives much higher throughput, but the simulator (MJX), the thresholds and the baselines all change. PPO-Lag baselines would have to be re-run in CRAX (CRAX supplies them), and CRAX is an unreviewed preprint with a young codebase (15 stars).
- **(C) Hybrid.** Use B for large-scale sweeps or a QD ablation, and A for the headline Safety-Gymnasium comparison.

For a four-month BSc project with core results frozen on 8 November 2026, stack A plus the LeWM surrogate is the defensible choice. B is a stretch goal.

JAX's stated CUDA 13 support for SM ≥7.5 implies the RTX 5090 (sm_120) should work with current `jax[cuda13]` wheels. This is inferred, not tested here. MuJoCo Playground's and CRAX's documented installs still reference `jax[cuda12]`, so expect some environment wrangling. Running JAX and PyTorch in the same process on one GPU also needs memory-preallocation settings, which is general practice rather than something sourced here.

### Gaps
- I found no published MJX, Brax or CRAX throughput numbers for an RTX 5090; GPU-side throughput is extrapolated from A100, H100 and 4090 figures (see Q4).
- I did not verify whether CRAX's Walker2D Velocity threshold equals Safety-Gymnasium's 2.3415, because its Table 7 values were not extracted.
- No safe-RL JAX environment other than CRAX and mjx-safety-gym was found. Safety-Gymnasium's own GPU path, "Safe Isaac Gym", is limited to manipulation (per CRAX's Table 1 note).

## Q4. How much compute do ES, CMA-ES and QD need on Walker-like tasks, and what fits in a few days on one GPU plus about 24 CPU workers?

### Takeaway
Published single-objective ES or ARS runs reach "good" Walker2d returns with about 8×10^6 to 4×10^7 environment steps. QD-RL papers budget 1-5 million *evaluations*: hundreds of millions to billions of steps, unless episodes are short. On the student's CPU pool, about 24 × 8,000 steps/s, that suggests roughly 1×10^8 steps per run in under 20 minutes. A 10-seed × 4-method grid at 1×10^8 steps would take well under a day, and QD at around 10^9 steps per run would take a few days. All of these are unverified extrapolations.

### Cited Findings
- **OpenAI-ES** (Salimans et al. 2017, arXiv preprint): solved 3D Humanoid in about 10 minutes on 1,440 CPU cores, and matched TRPO's final MuJoCo performance using between 3x and 10x as much data Source: [arXiv 1703.03864](https://arxiv.org/abs/1703.03864). Steps needed to reach TRPO's 5M-step scores (6 seeds), from its Table 3:

  | Task | Target score | ES timesteps | TRPO timesteps | ES/TRPO |
  |---|---|---|---|---|
  | Walker2d | 3830 | 3.79e7 | 4.81e6 | 7.88x |
  | Hopper | 3403 | 3.16e7 | 4.56e6 | 6.94x |
  | HalfCheetah | 2386 | 2.88e6 | 5.00e6 | 0.58x |

- **ARS** (Mania et al. 2018):
  - Linear policies. ARS V2-t averaged 24,000 episodes to reach Walker2d-v1 threshold 4390 (3 seeds), against 89,600 for V2 and 14,250 for TRPO-nn.
  - Steps to reach 3830 on Walker2d: 8.14×10^6 for ARS, against 3.79×10^7 for ES and 4.81×10^6 for TRPO.
  - Over 100 seeds, ARS reached the thresholds about 70% of the time on most tasks, but only 20% of the time on Walker2d.
  - On a 48-CPU machine, Humanoid reached 6000 in at most 13 minutes for 25 of 100 seeds Source: [arXiv 1803.07055](https://arxiv.org/abs/1803.07055).
- **QDax** (Lim et al., "Accelerated Quality-Diversity through Massive Parallelism", arXiv preprint, 2022):
  - Maximum MAP-Elites throughput on Ant Omni (100-timestep episodes), from its Table 1:

    | Implementation | Hardware | Evaluations/s |
    |---|---|---|
    | QDax + Brax | A100 | 30,846 (batch 65,536) |
    | QDax + Brax | RTX 2080 | 11,031 |
    | pyribs + PyBullet | 32 CPU cores | 184 |
    | Sferes (C++) | 32 CPU cores | 1,190 |

  - Budgets of 5M evaluations for QD-RL tasks.
  - Runtimes cut "by two factors of magnitudes, turning days of computation into minutes"; no significant performance loss with large batches (Wilcoxon rank-sum) Source: [arXiv 2202.01258](https://arxiv.org/abs/2202.01258).
- **ASCII-ME** (GECCO 2025): diverse DNN policies in under 250 s on a single GPU; 1M-evaluation budgets; L40S GPU Source: [arXiv 2501.18723](https://arxiv.org/abs/2501.18723).
- **PPGA** (ICLR 2024): QD-RL on Brax Ant, Walker2d, HalfCheetah and Humanoid. Walker2d observation size is 17 and action size 6. Jobs ran on one RTX 2080Ti with 4 CPU cores; 4 seeds; an ablation at 1.2M evaluations Source: [arXiv 2305.13795](https://arxiv.org/abs/2305.13795).
- **QDHUAC** (GECCO 2026): curves run to 2M steps for main results and 5M steps for HalfCheetah and Humanoid archive comparisons. Claims an order of magnitude fewer samples than baselines Source: [arXiv 2604.20381](https://arxiv.org/abs/2604.20381).
- **TensorNEAT** (GECCO 2024): population 10,000 on Swimmer took 215.7 s on an RTX 4090, 292.2 s on an RTX 3090 and 42,279.6 s for NEAT-Python on an EPYC 7543 CPU Source: [arXiv 2404.01817](https://arxiv.org/abs/2404.01817).
- **MuJoCo Playground** (RSS 2025):
  - Brax PPO training throughput on one A100: DM Control WalkerWalk about 139,818 steps/s, HumanoidWalk about 91,563 steps/s, SAC about 6,000 steps/s.
  - DM Control PPO runs were 60M steps with 5 seeds.
  - LeapCubeReorient took about 2,080 s on one RTX 4090 Source: [arXiv HTML 2502.08844](https://arxiv.org/html/2502.08844v1).
- **CRAX** (preprint, 2026):
  - On an H100, peaks at about 340K steps/s for Humanoid Velocity and about 1M steps/s for Ant Velocity and Point Goal. Safety-Gymnasium saturates at 3K-20K steps/s on the same hardware and runs out of CPU memory beyond 256 environments.
  - The full CRAX suite (hundreds of runs, trillions of steps) took 2 weeks on one H100, against close to a year estimated for Safety-Gymnasium.
  - Baselines trained for 500M environment steps per run Source: [CRAX](https://arxiv.org/html/2606.20376).
- **OmniSafe's velocity benchmark** trains on-policy algorithms for 1e7 steps with 5 seeds Source: [OmniSafe on-policy benchmark](https://github.com/PKU-Alignment/omnisafe/blob/main/benchmarks/on-policy/README.md).

### Inferences
All figures here are estimates, not measurements.

**CPU throughput**
- The pool's theoretical peak is 24 × 8,000 ≈ 192k steps/s. Assume 50-75% efficiency after policy inference, inter-process communication and stragglers. Salimans et al. note that variable episode lengths hurt CPU utilisation and cap episodes for that reason. That gives about 100-150k steps/s.

**Run-time estimates on CPU**

| Steps | Time at 100-150k steps/s |
|---|---|
| 1e7 (OmniSafe PPO budget) | ~1-2 min |
| 4e7 (OpenAI-ES Walker2d to 3830) | ~4.5-7 min |
| 1e8 | ~11-17 min |
| 1e9 | ~2-3 h |

**Population arithmetic**
- With 1,000-step episodes, population λ = 256 and one episode per candidate, a generation costs up to 2.56×10^5 steps, about 2 s. So 1×10^8 steps is roughly 390 generations; four episodes per candidate cuts that to about 100 generations.
- Constraint estimates from a binary per-step cost are noisy, so several episodes per candidate, or re-evaluating elites, will likely be needed.

**Feasible experiment grid**
- 4 methods × 10 seeds × 1×10^8 steps = 4×10^9 steps, about 7.5-11 hours on CPU.
- QD at 1M evaluations × 1,000 steps = 1×10^9 steps per run, about 2-3 hours, so 3 QD variants × 10 seeds ≈ 2.5-4 days. Shorter episodes or fewer evaluations would bring this down.

**GPU path**
- If CRAX on an RTX 5090 reached even one third of the H100 Ant-Velocity peak (about 300k steps/s), 1×10^9 steps would take about an hour. This is unmeasured.
- MJX Walker throughput on DM Control (about 140k steps/s *including* PPO training on an A100) suggests pure rollout throughput for evolution would be higher.

**LeWM surrogate**
- The cost of LeWM evaluation is batched GPU forward passes per imagined step. Its value depends on rollout horizon and model error. The literature consistently pairs surrogates with real-environment re-evaluation, because surrogates create false optima (Wang et al. 2022, Ha & Schmidhuber 2018).
- Report the real steps used to train LeWM inside every budget comparison, otherwise reviewers will see the compute accounting as unfair.

**Policy size and optimiser choice**
- A 17-input, 6-output MLP with two 64-unit hidden layers has about 5.7k parameters.
- Full-covariance CMA-ES at that size is borderline. JEDi switched to LM-MA-ES for Brax-size policies, so plan on sep-CMA-ES, LM-MA-ES, OpenAI-ES or PGPE for neural policies, and full CMA-ES only for small or linear policies or low-dimensional "dial" parameters.

### Gaps
- No published RTX 5090 throughput exists for MJX, Brax or CRAX.
- The student's 8,000 steps/s per process was not checked against policy-inference overhead.
- The 192k steps/s aggregate assumes linear scaling, which I did not verify.
- I found no published ES or CMA-ES numbers on SafetyWalker2dVelocity specifically. Unconstrained Walker2d numbers come from older Gym v1 environments (ARS, OpenAI-ES), whose reward scales differ from Gymnasium v4/v5 and Safety-Gymnasium.

## Q5. What evaluation standards do GECCO RL papers follow, and what will reviewers criticise?

### Takeaway
Recent GECCO NE, EML and CS papers on RL use 5-20 seeds (typically 10). They report medians with IQR or quartiles, or means with 95% CIs, and use rank-based tests: Mann-Whitney U / Wilcoxon rank-sum with Bonferroni or Holm-Bonferroni correction, or Kruskal-Wallis followed by post-hoc tests. Budgets are stated in evaluations or environment steps, and the stronger papers also report wall-clock time on the same hardware. Reviewers are told to judge reproducibility, and EML explicitly wants strong non-evolutionary baselines. The well-documented pitfalls are: too few seeds, unequal budgets (iterations versus evaluations versus wall-clock), untuned or weak RL baselines, and ES's 3-10x data inefficiency.

### Cited Findings
**Seeds and statistics in recent GECCO RL papers**

| Paper | Seeds or trials | Statistics reported | Source |
|---|---|---|---|
| ASCII-ME (GECCO 2025) | 20 | Wilcoxon-Mann-Whitney U with Holm-Bonferroni; compared at equal runtime and equal evaluations | [arXiv 2501.18723](https://arxiv.org/abs/2501.18723) |
| Nisioti et al. (GECCO 2025 BBSR) | 10 | Kruskal-Wallis, then pairwise Mann-Whitney U | [arXiv 2505.22696](https://arxiv.org/abs/2505.22696) |
| MEMES (GECCO 2024) | 10 | Median and CI; Wilcoxon rank-sum with Bonferroni; wall-clock on the same hardware | [arXiv 2303.06137](https://arxiv.org/abs/2303.06137) |
| JEDi (GECCO 2024) | ≥10 (mazes), 5 (Brax) | Median final fitness; Mann-Whitney U p-values in a table | [arXiv 2405.04308](https://arxiv.org/abs/2405.04308) |
| Safe CMA-ES (GECCO 2024) | 50 | Medians and IQR | [arXiv 2405.10534](https://arxiv.org/abs/2405.10534) |
| TensorNEAT (GECCO 2024) | 10 | 95% CIs | [arXiv 2404.01817](https://arxiv.org/abs/2404.01817) |
| Lim et al. (GECCO 2023) | 5-10 replications | Wilcoxon rank-sum with Bonferroni | [arXiv 2303.06164](https://arxiv.org/abs/2303.06164) |
| Zheng & Cheng (GECCO 2023) | 10 | 68% CI | [arXiv 2305.02949](https://arxiv.org/abs/2305.02949) |
| QDHUAC (GECCO 2026) | 5 | Standard-deviation bands | [arXiv 2604.20381](https://arxiv.org/abs/2604.20381) |

**Venue rules**
- GECCO review criteria include "sufficiency of information to permit replication". Authors are encouraged to submit source code as anonymised supplementary material and to use Zenodo-style permanent repositories Source: [2026 CFP](https://gecco-2026.sigevo.org/Call-for-Papers).
- EML: "strongly encouraged to compare… to the corresponding state-of-the-art non-evolutionary ML methods" Source: [EML track](https://gecco-2026.sigevo.org/Track?itemId=53).
- BBSR reproducibility studies must provide "all implementation details, input data, parameters and hardware specifications" and public artefacts Source: [BBSR track](https://gecco-2026.sigevo.org/Track?itemId=2672).

**Budget fairness**
- QDax authors ran to a fixed number of evaluations because "Running for a fixed number of iterations would be an unfair comparison" Source: [arXiv 2202.01258](https://arxiv.org/abs/2202.01258).
- ES needed 3-10x as much data as TRPO on MuJoCo. On Walker2d, ES took 7.88x TRPO's steps to reach TRPO's final score Source: [arXiv 1703.03864](https://arxiv.org/abs/1703.03864).

**Seed counts and variance**
- ARS criticised prior RL work for using "fewer than ten random seeds" and ran 100 seeds. Walker2d succeeded on only 20% of seeds, showing that a three-seed headline can hide fragility Source: [arXiv 1803.07055](https://arxiv.org/abs/1803.07055).
- Henderson et al. (AAAI 2018): non-determinism and intrinsic variance make deep-RL results hard to interpret "without significance metrics and tighter standardization of experimental reporting" Source: [arXiv 1709.06560](https://arxiv.org/abs/1709.06560).
- Colas, Sigaud & Oudeyer (2018, preprint): a tutorial on choosing the number of seeds via statistical power for t-tests and bootstrap CI tests Source: [arXiv 1806.08295](https://arxiv.org/abs/1806.08295).
- Agarwal et al. (NeurIPS 2021, Outstanding Paper): in the few-run regime, report interval estimates and performance profiles, and prefer the interquartile mean (IQM) to the mean or median; library "rliable" Source: [arXiv 2108.13264](https://arxiv.org/abs/2108.13264).
- Patterson, Neumann, White & White (JMLR 2024): covers hypothesis testing, comparing multiple agents, baseline construction, hyperparameters and experimenter bias Source: [arXiv 2304.01315](https://arxiv.org/abs/2304.01315).
- CRAX's seed study (20 seeds, Level 1): relative to the mean, the 95% CI half-width shrank from about 90% at 2 seeds to 40% at 3 and 22% at 5, with small gains after that. The hardest environments stayed near 30% even at 20 seeds, and small seed counts sometimes gave misleadingly narrow CIs. CRAX uses 10 seeds Source: [CRAX](https://arxiv.org/html/2606.20376).

### Inferences
**A defensible GECCO protocol**
- At least 10 seeds per configuration (20 if cheap on CPU).
- Report median with IQR, or IQM with bootstrap 95% CIs (rliable).
- Use pairwise Mann-Whitney U (Wilcoxon rank-sum) with Holm correction across all comparisons.
- Report the budget on three axes:
  - real environment steps, including steps used to train LeWM;
  - imagined (LeWM) steps or evaluations;
  - wall-clock time and hardware.
- Compare at equal real-step budgets, with a secondary equal-wall-clock view.
- Release anonymised code (anonymous GitHub or Zenodo) and state hyperparameter-tuning effort per method.

**Likely reviewer criticisms**, inferred from the sources above, since no public GECCO reviews were found:
1. PPO-Lag or OmniSafe baselines under-tuned or run at a different budget.
2. ES advantage disappears once real-environment samples, including world-model training data, are counted.
3. Too few seeds, or no statistics.
4. Surrogate exploitation, where policies score well in LeWM and badly in the simulator. Report the surrogate-versus-real fitness correlation and the re-evaluation rate.
5. A single environment (only Walker). Add Hopper and HalfCheetah velocity tasks, or a CRAX cross-check, if budget allows.
6. Unclear constraint-handling choice (penalty, feasibility rules, augmented Lagrangian, multi-objective).

### Gaps
- GECCO does not publish reviews, and I found no NE/EML reviewer guideline beyond the CFP criteria, so the list of common criticisms is inferred.
- I did not find a GECCO-specific mandatory reproducibility checklist; code release is encouraged, not required, for non-BBSR tracks.

## Q6. What are the safe-RL evaluation norms (cost thresholds and metrics) for Safety-Gymnasium velocity tasks?

### Takeaway
In Safety-Gymnasium velocity tasks the per-step cost is 1 whenever speed exceeds a fixed threshold; for SafetyWalker2dVelocity-v1 the threshold is 2.3415. The standard constraint is an episodic cost limit of 25, used by OmniSafe, Safety Gym and CRAX. Standard reporting is:
- final episodic return;
- final episodic cost against the limit;
- cost rate over the whole of training (the Safety Gym regret measure).

Newer work, including a September 2026 preprint and CRAX, adds training-time violations, distributional and violation-severity reporting, and sweeps over the cost threshold.

### Cited Findings
- Velocity cost is `bool(V_current > V_threshold)`. Thresholds are set at 50% of the agent's maximum velocity reached by PPO after 1e7 steps Source: [Safety-Gymnasium velocity docs](https://safety-gymnasium.readthedocs.io/en/latest/environments/safe_velocity.html). The velocity is the vector sum in the X-Y plane for all agents except Swimmer, which uses the X axis only.

  | Agent | v1 threshold | v0 threshold |
  |---|---|---|
  | Walker2d | 2.3415 | 1.7075 |
  | Hopper | 0.7402 | 0.37315 |
  | HalfCheetah | 3.2096 | 2.8795 |
  | Ant | 2.6222 | 2.5745 |
  | Humanoid | 1.4149 | 2.3475 |
  | Swimmer | 0.2282 | 0.04845 |

- OmniSafe's on-policy velocity benchmark (all six SafetyXVelocity-v1 tasks) uses cost limit 25.0, 1e7 training steps, 5 seeds (0, 5, 10, 15, 20) and 20,000 steps per epoch, and reports episode reward and episode cost. OmniSafe recommends CPU for consistent results Source: [OmniSafe benchmark README](https://github.com/PKU-Alignment/omnisafe/blob/main/benchmarks/on-policy/README.md). SafetyWalker2dVelocity-v1 results:

  | Algorithm | Episode reward | Episode cost |
  |---|---|---|
  | PPO | 6239.52 ± 879.99 | 902.68 ± 100.93 |
  | PPO-Lag | 2982.27 ± 681.55 | 13.49 ± 14.55 |

- Safety Gym (Ray, Achiam & Amodei, OpenAI 2019) defines three metrics Source: [Safety Gym paper PDF](https://cdn.openai.com/safexp-short.pdf):
  - Jr(θ), the average episodic return;
  - Jc(θ), the average episodic sum of costs;
  - ρc, the "average cost over the entirety of training" (sum of all costs ÷ total environment steps), proposed as a safety-regret measure.
- Comparison rules in Safety Gym:
  - Constraint-violating agents are strictly worse than satisfying ones.
  - Among satisfying agents with equal interaction budgets, A1 dominates A2 if it improves return or cost rate without being worse on the other.
  - Aggregation uses normalised return, normalised violation max(0, Jc − d) and normalised cost rate, all relative to unconstrained PPO.
  - The condition ρc·T_ep ≤ d means the average training episode satisfied the constraint.
  - Experiments used d = 25 and 3 seeds.
- Safety-Gymnasium (Ji et al., NeurIPS 2023 Datasets & Benchmarks) provides single- and multi-agent safety tasks with vector and vision inputs, plus SafePO, a library of 16 safe-RL algorithms Source: [arXiv 2310.12567](https://arxiv.org/abs/2310.12567).
- CRAX (preprint, 2026):
  - Reports undiscounted episodic reward and cost of the final policy averaged over 128 evaluation episodes; a policy is "safe" when episodic cost stays below d = 25.
  - Also reports sample efficiency and constraint violations during training (its Appendix C) and the performance-safety trade-off under varying thresholds (Appendix E).
  - Uses 10 seeds and finds no algorithm dominates across tasks Source: [CRAX](https://arxiv.org/html/2606.20376).
- Spoor, Plaat & Moerland, "Evaluation Metrics for Safe Reinforcement Learning" (arXiv preprint, 2026-09-14):
  - Argues that average-case metrics miss violation frequency, severity and consistency, and whether training behaviour matches final-policy behaviour.
  - Recommends aggregate, distributional ("how often and how severely the safety bound is violated") and task-specific reporting, a "safety tier" system, and the open-source SafeRLEval Source: [arXiv 2609.15315](https://arxiv.org/abs/2609.15315).
- SafeDreamer (ICLR 2024) combines Lagrangian methods with Dreamer world-model planning and reports near-zero cost on Safety-Gymnasium tasks. It is the main world-model safe-RL reference point Source: [arXiv 2307.07176](https://arxiv.org/abs/2307.07176).

### Inferences
- OmniSafe's unconstrained PPO incurs a cost of about 903 per Walker episode. With episodes of up to 1,000 steps, the converged unconstrained policy is over the speed limit roughly 90% of the time; the 1,000-step cap is Gymnasium's standard and is assumed rather than verified here. A cost limit of 25 therefore forces the policy to stay under 2.3415 m/s for about 97.5% of steps.
- Report these metrics:
  1. final Jr and Jc, with the fraction of seeds whose final policy satisfies Jc ≤ 25;
  2. training-time cumulative cost, and the cost rate ρc, counting **every real-environment rollout of every population member**;
  3. return at matched real-step budgets against OmniSafe PPO-Lag (2982 ± 682, cost 13.5 ± 14.6);
  4. a cost-limit sweep, for example 10, 25 and 50, if time permits.
- Item 2 is where a world-model-evaluated evolutionary method could stand out, because imagined rollouts incur no real cost. It is also where a naive ES would look worst, because a population explores unsafely in the real environment.
- If CRAX is used as a second benchmark, its thresholds are defined differently (relative to its own unconstrained agent), so its numbers are not directly comparable with Safety-Gymnasium's.

### Gaps
- The exact SafePO metric definitions and default cost limits for velocity tasks were not extracted; I only confirmed that its benchmark covers the Safety-Gymnasium velocity tasks.
- I did not verify whether OmniSafe logs a training-time cost-rate metric by default.
- Safety Gym's normalisation constants were derived for the original Safety Gym tasks, not Safety-Gymnasium velocity tasks.
