# Reading and watching safety in robotic manipulation

Research checked 3 November 2026. This is a focused educational route, not a systematic bibliography or an experiment proposal. Difficulty and reading order below are my pedagogical judgments.

## Which four readings build the necessary foundations?

### Takeaway

Start with one overview and one manipulation example. Then contrast an expected-cost formulation with a control-theoretic invariance formulation. The latter two are foundational publications, not claims about the newest algorithm.

### Cited Findings

1. **Lukas Brunke et al., Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning (2022 journal publication; 2021 preprint).** This survey connects control and RL, distinguishing learning uncertain dynamics, encouraging safety/robustness, and certifying learned controllers. Start with section 2, especially 2.1.3 on hard, probabilistic and soft constraints; then 3.2 and 3.3. Its scope is learning control, not a complete treatment of perception, estimation, planning or multi-agent systems. The formulation assumes direct access to possibly noisy state measurements. [Open paper](https://arxiv.org/html/2108.06266v2); [journal publication](https://www.annualreviews.org/content/journals/10.1146/annurev-control-042920-020211).

2. **Chengyue Huang et al., SafeManip: A Property-Driven Benchmark for Temporal Safety Evaluation in Robotic Manipulation (2026 preprint, v3 28 September).** An application-specific example of safety depending on an execution's history: contamination, premature release and incorrect sequencing can coexist with task completion. It evaluates low-level rollouts using finite-trace temporal logic and reports safety relative to the opportunities that trigger obligations. Start with the introduction, section 3.1 on the specification language, section 3.3 on the framework, and section 4.2 on metrics. This is a definition and evaluation paper, not proof that a learning method produces safe physical behaviour. [Paper](https://arxiv.org/html/2605.12386v3).

3. **Joshua Achiam, David Held, Aviv Tamar and Pieter Abbeel, Constrained Policy Optimization (ICML 2017).** Provides the foundational constrained-RL perspective: maximize task return while bounding expectations of auxiliary costs. Start with the introduction and CMDP formulation before the optimizer derivation. The distinction between an expectation constraint and an episode-wise no-violation requirement is central for our reading. The paper reports near-constraint satisfaction per learning iteration under its theoretical setup, not a universal no-accident guarantee. [Paper and proceedings](https://proceedings.mlr.press/v70/achiam17a.html); [author's accessible explanation](https://bair.berkeley.edu/blog/2017/07/06/cpo/).

4. **Aaron D. Ames et al., Control Barrier Functions: Theory and Applications (2019).** Introduces safety as maintaining membership in a safe set, with optimization-based controllers enforcing barrier conditions. Start with sections I and II, then inspect section III on actuation limits. The starting mathematical setup is a nonlinear control-affine system with regularity assumptions and admissible inputs. Read guarantees together with their model and feasibility conditions. Its examples span robotics and automotive control, rather than focusing on manipulation. [Paper](https://arxiv.org/html/1903.11199v1).

### Inferences

- Reading order: Brunke overview, SafeManip example, CPO, then Ames. The first pair makes the problem concrete; the second pair reveals that different meanings of safety imply different optimization problems.
- Difficulty: Brunke is intermediate but approachable selectively; SafeManip's introduction is accessible, with temporal logic moderately technical; CPO needs RL/probability and optimization; Ames needs differential equations and control basics.
- A useful reading question for each is: what is supplied by the human, what is learned, what is actually measured, and what assumptions make the stated safety claim valid?

### Gaps

- No claim that these four papers exhaust manipulation safety or evolutionary learning. They deliberately establish safety concepts before a separate evolutionary literature review.
- SafeManip is identified as a preprint; the checked arXiv record does not establish peer-reviewed venue acceptance.

## Which videos offer complementary explanations?

### Takeaway

Two established talks provide a broad learning-and-control perspective and a more mathematical safety perspective. Exact titles and speaker associations were verified on official lab/institute pages; no runtime or transcript-specific claims are made.

### Cited Findings

- **Angela Schoellig, Machine Learning for Robotics: Achieving Safety, Performance and Reliability by Combining Models and Data in a Closed-Loop System Architecture (Intersections between Control, Learning and Optimization, 2020).** The lab's official Safe Robot Learning page lists this exact title and links to the recording. [Watch](https://www.youtube.com/watch?v=tMB5tU8buQ4); [official lab listing](https://www.dynsyslab.org/safe-robot-learning/). The recommendation is based on the verified title and lab context; I did not watch or retrieve a transcript. Its value for this user is an introduction to combining learning and control rather than treating them as competing research communities.

- **Aaron Ames, Safety-Critical Control of Autonomous Systems (IPAM, 26 October 2020).** The institutional abstract describes nonlinear control, control barrier functions, safety represented by forward set invariance, robotic demonstrations and automotive applications. [Official talk page with Play Video](https://www.mathinstitutes.org/videos/16688). The official playback link redirects to [IPAM's video player](https://www.ipam.ucla.edu/video.php?video_id=1_zh7dr1fc). The listing and redirect were verified, but the browser tool could not retrieve the player itself. Do not promise that playback was tested. [Verified redirect page](https://www.mathinstitutes.org/videos/16688/view/).

### Inferences

- Schoellig is the better first watch for the broad question. Ames pairs with the CBF reading when the user is ready for a specific mathematical formulation.
- Neither talk is recommended as a snapshot of 2026 manipulation benchmarks. They are conceptual foundations.

### Gaps

- YouTube returned metadata but not substantive transcript content. No runtimes, timestamps or coverage beyond official descriptions were verified.
- IPAM playback could not be tested through the research tool.

## What recent literature updates the foundational route?

### Takeaway

A recent contact-specific survey supports the contemporary relevance of distinguishing exploration safety from execution safety, but it need not expand the user's initial four-paper list.

### Cited Findings

- **Heng Zhang et al., Safe Learning for Contact-Rich Robot Tasks: A Survey from Classical Learning-Based Methods to Safe Foundation Models**, first submitted December 2025, v4 revised 29 October 2026. The current version includes sections titled Definition of Safety in Contact-Rich Tasks, Safety Evaluation Metrics, Safety Abstraction Level, and Safety Enforcement Spaces. It organizes work around safe exploration and safe execution and covers model-based and learning-based approaches, including foundation models. [Current version](https://arxiv.org/html/2512.11908v4); [version record](https://arxiv.org/abs/2512.11908).

### Inferences

- This is a useful subsequent overview after Brunke, especially for contact, sensing and the distinction between training and deployment. It is optional, rather than an additional immediate reading assignment.

### Gaps

- This recommendation uses the accessible preprint. No publication venue or peer-review status is asserted.
