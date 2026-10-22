# Project relevance of Umair Nasir's GECCO experience

## Which parts of Safety-Dial could benefit from his expertise?

### Takeaway

Nasir is a strong prospective methods and venue mentor for the evolutionary core of the
project. The closest fit is representation and mutation design, followed by matched
comparisons, generalization tests, ablations, GECCO framing, and paper presentation. His
record is less direct evidence for ManiGuard implementation or physical safety evaluation.

### Cited Findings

- The current Safety-Dial direction is a shared observation- and instruction-conditioned
  controller that perturbs native flow noise in frozen family-specific pi0.5 policies. It
  requires comparisons against native execution, a tuned fixed correction, matched random
  search, and ordinary learning, with independent evaluation and complete interaction
  accounting. [Current ManiGuard direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)
- Continuous Program Search is unusually close at the conceptual level. It first measures
  whether controlled latent perturbations yield local, predictable behavioural change, then
  compares three mutation operators while holding the evolutionary strategy and evaluation
  budget fixed. It uses held-out behaviour diagnostics and reports both median performance
  and the fraction of the budget needed to discover strong candidates.
  [Continuous Program Search](https://arxiv.org/html/2602.07659v1)
- The same paper explicitly separates a representation question from an optimizer question:
  all methods use the same `(mu + lambda)` evolutionary strategy, latent space, and budget,
  while only the mutation proposal changes. This maps closely to Safety-Dial's need to test
  whether native-noise coordinates are controllable before attributing gains to evolution.
  [Continuous Program Search, controlled comparison](https://arxiv.org/html/2602.07659v1)
- LLMatic gives first-author evidence of designing and evaluating a more elaborate
  evolutionary system. It uses dual CVT-MAP-Elites archives, compares against random search
  and an LLM baseline, reports a 2,000-candidate limit, and runs component ablations over 30
  seeds, with a separate NAS-Bench evaluation averaged over 10 runs.
  [LLMatic paper](https://arxiv.org/pdf/2306.01102)
- Mortar gives first-author evidence of decomposition and ablation in a multi-component
  evolutionary pipeline. It combines MAP-Elites with tree search, evaluates the contribution
  of mechanics to a system-level outcome, and compares the tree-search component with random,
  greedy, and LLM-prompted alternatives over five runs.
  [Mortar paper](https://arxiv.org/html/2601.00105v1)
- ATEP gives first-author evidence on cross-environment transfer and explicit generalization
  tests. It compares two fixed-topology baselines and multiple transfer mechanisms, then tests
  agents on environments produced by other methods. The paper also acknowledges that its
  preferred method uses more function evaluations and that only two training seeds were
  feasible because each run cost about 50,000 to 200,000 CPU hours.
  [ATEP paper](https://arxiv.org/pdf/2210.11442)
- Nasir's current portfolio identifies him as a Wits PhD student in RAIL Lab and the NYU Game
  Innovation Lab, advised by Steven James and Julian Togelius, with research interests in
  open-ended learning and procedural content generation. It also lists current teaching in
  data visualization and exploration.
  [Nasir portfolio](https://umair-nasir14.github.io/)
- GECCO's 2026 EML scope explicitly covers reinforcement learning, transfer, domain
  adaptation, representation learning, and application work, and urges comparison with
  corresponding non-evolutionary methods. The NE scope also includes transformers,
  evolutionary reinforcement learning, transfer, multitask learning, novel representations,
  search mechanisms, fitness functions, and efficiency.
  [GECCO 2026 EML track](https://gecco-2026.sigevo.org/Track?itemId=53)
  [GECCO 2026 NE track](https://gecco-2026.sigevo.org/Track?itemId=59)

### Inferences

- **Highest-value fit: representation and mutation geometry.** Ask him to challenge the
  decoder-authority audit, perturbation bounds, mutation scale, basis construction, and the
  claim that nearby controllers yield partially heritable behavioural changes. Continuous
  Program Search is directly relevant to designing a behavioural-locality or trust-region
  diagnostic, although Safety-Dial must measure decoded robot actions and physical outcomes
  rather than program actions. [Continuous Program Search](https://arxiv.org/html/2602.07659v1)
  [Current ManiGuard direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)
- **High-value fit: comparison methodology.** LLMatic and Mortar show practical experience
  with random baselines, component ablations, repeated runs, QD metrics, and expensive
  evaluation. He could red-team whether the fixed-bias, random-search, EA, and ordinary
  learning arms isolate the intended claim, and whether the protected audit is genuinely
  independent. [LLMatic paper](https://arxiv.org/pdf/2306.01102)
  [Mortar paper](https://arxiv.org/html/2601.00105v1)
- **High-value fit: scope and generalization.** ATEP's transfer experiments make him a useful
  critic of the distinction between per-family refitting, transfer to new tasks within seen
  families, and one frozen controller on an adaptation-held-out family. He should be asked to
  focus on the experimental distinction, not to transplant ATEP's much larger computation
  pattern. [ATEP paper](https://arxiv.org/pdf/2210.11442)
  [Current ManiGuard direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)
- **High-value fit: GECCO positioning and writing.** His accepted work spans a full EML paper,
  a full GECH paper, a full GECCO paper in the benchmarking-oriented portion of the 2024
  programme, and shorter Companion contributions. That gives him practical knowledge of how
  different GECCO audiences distinguish an application result from an EC contribution.
  [GECCO 2026 accepted papers](https://gecco-2026.sigevo.org/Accepted%2BPapers)
  [GECCO 2024 proceedings](https://dblp.org/db/conf/gecco/gecco2024.html)
  [Continuous Program Search record](https://dblp.org/rec/conf/gecco/SiperNKSAT26.html)
- **Moderate-value fit: visualization and narrative.** His teaching role and the clear
  optimizer-controlled comparisons in his papers suggest value in selecting the convergence,
  cost-curve, ablation, and generalization figures that must fit an eight-page paper. This is
  an inference, not proof that he has reviewed safety-robotics papers.
  [Nasir portfolio](https://umair-nasir14.github.io/)
  [Continuous Program Search](https://arxiv.org/html/2602.07659v1)
- **Potential networking value, but do not presume it.** His repeated collaborations across
  Wits, NYU, and Nof1 provide plausible access to evolutionary computation and game-AI
  researchers. Sachin can ask for one or two relevant introductions only after receiving
  technical feedback, but the public sources do not show whom he would be willing to
  introduce or in what capacity. [Nasir portfolio](https://umair-nasir14.github.io/)

### Gaps

- Authorship lists do not disclose Nasir's exact contribution to multi-author Continuous
  Program Search or ProFiT. Ask which parts he personally designed, implemented, analyzed,
  and wrote before treating those methods as his direct expertise.
- None of the reviewed sources records prior work on ManiGuard, pi0.5/OpenPI, flow-policy
  inference hooks, robot-arm control, or physical manipulation safety.
- No public source establishes availability, willingness to mentor, or willingness to join a
  paper. These must be discussed directly without implying authorship in advance.

## What precise questions should Sachin ask him?

### Takeaway

The meeting should be a protocol red-team, not a broad career conversation. Questions should
force concrete decisions about behavioural locality, optimizer fairness, evaluation under
noise, generalization, and what would make the work recognizably evolutionary to GECCO.

### Cited Findings

- Safety-Dial's first pilot is currently framed as 60 planned valid episodes with at most four
  invalid-attempt replacements. It includes shared initialization, matched 16-episode EA and
  random-search descendant phases, and a protected 16-episode audit over native, fixed, EA,
  and random conditions. The document explicitly says this cannot establish superiority and
  leaves ordinary learning outstanding unless a complete arm fits.
  [Current ManiGuard direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)
- The direction proposes a paired decoder audit using fixed observations, common noise draws,
  positive and negative perturbations, complete action chunks, executed prefixes, and charged
  model queries before physical search.
  [Current ManiGuard direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)
- Continuous Program Search treats perturbation locality as an empirical property and derives
  mutation scales from controlled behavioural-divergence tests rather than assuming that a
  continuous latent space is smooth.
  [Continuous Program Search](https://arxiv.org/html/2602.07659v1)
- The EML call asks authors to compare evolutionary methods with corresponding non-evolutionary
  methods where appropriate. [GECCO 2026 EML track](https://gecco-2026.sigevo.org/Track?itemId=53)

### Inferences

Ask these questions in roughly this order:

1. **Contribution and track:** "If the empirical result is that one shared bounded
   native-noise controller improves safe completion per simulator interaction, what is the
   strongest defensible GECCO claim? Would you pitch it to EML, NE, Complex Systems, or
   another track, and what result would make you reject that pitch?"
2. **Personal expertise:** "For Continuous Program Search, ProFiT, Mortar, LLMatic, and ATEP,
   which experimental and writing decisions did you personally own?"
3. **Locality test:** "What behavioural-locality test would convince you that pi0.5 native
   noise is a meaningful evolutionary genotype rather than merely an editable tensor? Are
   decoded chunk divergence, executed-prefix divergence, validity, saturation, and contact
   changes sufficient?"
4. **Response-informed coordinates:** "Would you use random directions, response-informed
   directions, or both in the first study? What comparison would distinguish a useful basis
   from a larger hidden development budget?"
5. **EA design:** "For the frozen controller dimension and a 16-descendant physical budget,
   what simple EA, population size, mutation schedule, and elitism rule can be justified
   without turning optimizer choice into the contribution?"
6. **Noise and reevaluation:** "With binary safe completion, physical violation, unfinished
   episodes, and noisy rollouts, how much budget would you reserve for common cases, elite
   reevaluation, and finalist confirmation? Would you use fixed replication or adaptive
   racing in the pilot?"
7. **Fair baselines:** "Do the proposed native, tuned fixed correction, matched random search,
   and ordinary learner isolate the four claims: interface benefit, conditioning benefit,
   evolutionary advantage, and transfer? What information or tuning advantage is still
   mismatched?"
8. **Pilot interpretation:** "Given four audit episodes per condition and one optimizer seed,
   which outcomes justify scaling, revising the representation, or stopping, and which must be
   reported only as mechanism or effect-size evidence?"
9. **Generalization:** "How would you partition tasks and families so that basis construction,
   controller fitting, finalist selection, and held-out-family evaluation do not leak? What is
   the minimum credible distinction between method reuse and one frozen transferred adapter?"
10. **Paper evidence:** "Which three plots and two tables would a GECCO reviewer expect to see
    for an expensive physical-outcome study, and which ablation is indispensable within eight
    pages?"
11. **Reviewer objection:** "What is the most likely rejection reason: insufficient EC
    novelty, too few optimizer replications, weak ordinary-learning comparison, unclear
    genotype-to-behaviour locality, or too narrow ManiGuard scope?"
12. **Network:** "After the protocol is sharper, is there one person with noisy evolutionary
    optimization or evolutionary robotics expertise whom you recommend I ask to review only
    the evaluation design?"

These questions follow from the current protocol's explicit unresolved decisions and Nasir's
demonstrated work on locality, ablation, transfer, and budgeted evolutionary evaluation.
[Current ManiGuard direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)
[Continuous Program Search](https://arxiv.org/html/2602.07659v1)
[LLMatic paper](https://arxiv.org/pdf/2306.01102)
[ATEP paper](https://arxiv.org/pdf/2210.11442)

### Gaps

- Public papers cannot answer which optimizer and replication plan he would recommend under
  Safety-Dial's actual 64-attempt ceiling.
- GECCO 2027 track chairs and final calls should be rechecked before treating a 2026 track
  recommendation as binding. The repository venue note warns that the 2027 pages previously
  mirrored 2026 content. [GECCO 2027 notes](../../docs/gecco2027.md)

## Where is there little evidence of overlap?

### Takeaway

There is little public evidence that Nasir should be the sole technical authority on the
robotics, safety, statistical, or platform-specific parts. His highest-confidence role is EC
methodology and GECCO strategy, alongside other specialists.

### Cited Findings

- His stated interests are AI and games, especially open-ended learning and procedural content
  generation. The listed publications concentrate on evolutionary search, program or game
  generation, neural architecture search, open-ended learning, and LLM evaluation.
  [Nasir portfolio](https://umair-nasir14.github.io/)
- ATEP uses a simulated bipedal-walker open-ended domain and NEAT/EPOET, not manipulation or a
  frozen vision-language-action policy. It also uses much larger compute and only two seeds,
  making it poor evidence for rare-violation inference under a 64-episode budget.
  [ATEP paper](https://arxiv.org/pdf/2210.11442)
- Mortar evaluates generated games through artificial game-playing agents and a small user
  study. Its five-run experiments and estimated LLM cost concern procedural game generation,
  not physical safety outcomes.
  [Mortar paper](https://arxiv.org/html/2601.00105v1)
- Continuous Program Search uses deterministic backtesting and a learned flow-based mutation
  proposer in a program embedding. This is conceptually helpful, but it is not evidence that
  pi0.5's native flow-noise input has controllable or transferable physical directions.
  [Continuous Program Search](https://arxiv.org/html/2602.07659v1)
- The current ManiGuard plan requires strict separation of proposed and executed actions,
  physical violations and unfinished episodes, native benchmark conditions, simulator
  interactions, model queries, elapsed time, and uncertainty for clustered binary outcomes.
  [Current ManiGuard direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)

### Inferences

- Seek separate input from a ManiGuard or OpenPI maintainer for the explicit-noise hook,
  action-coordinate mapping, renderer and task contracts, and checkpoint-specific behaviour.
  Nasir can assess the experiment once the interface facts are verified, but the portfolio
  does not establish platform expertise. [Current ManiGuard direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)
- Seek safe-RL or manipulation expertise for defining physical violations, requested-sequence
  completion, safety noninferiority, and the implications of an intervention-free monitor.
  None of the reviewed portfolio publications measures robot-manipulation safety.
  [Nasir portfolio](https://umair-nasir14.github.io/)
- Seek statistical-design advice for power, clustered task blocks, paired binary outcomes,
  confidence intervals, and rare violations. His papers demonstrate repeated empirical
  evaluation, but the reviewed sources do not establish specialist expertise in safety
  noninferiority or rare-event inference. [LLMatic paper](https://arxiv.org/pdf/2306.01102)
  [ATEP paper](https://arxiv.org/pdf/2210.11442)
- Do not ask him to validate a world model, safety filter, or ROSARL integration because these
  are not assumed components of the current study. [Current ManiGuard direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)

### Gaps

- Absence from a portfolio is not evidence of inability. A first meeting should explicitly ask
  about unpublished robotics, safety, statistical-design, and simulator experience.
- It is unknown whether he has reviewed for GECCO, served on a GECCO programme committee, or
  attended all conferences in person. Publication history alone should not be described as
  reviewer or organizer experience.

## How strong is the verified GECCO record?

### Takeaway

It is a strong and rapidly improving record for a PhD researcher, especially because it
includes first-author main-proceedings work and a 2026 best-paper nomination. It should be
described accurately as three main-proceedings papers plus two Companion papers across
2023, 2024, and 2026, not as five equivalent full papers or as a best-paper win.

### Cited Findings

- LLMatic is a nine-page main-proceedings GECCO 2024 paper with Nasir as first author.
  [LLMatic paper](https://arxiv.org/pdf/2306.01102)
  [DBLP record](https://dblp.org/rec/conf/gecco/NasirETJC24.html)
- GECCO 2026's official accepted-paper list includes Mortar in EML and ProFiT in GECH. Mortar
  lists Nasir first; ProFiT lists him fourth.
  [GECCO 2026 accepted papers](https://gecco-2026.sigevo.org/Accepted%2BPapers)
- GECCO 2026's official best-paper page lists Mortar as an EML best-paper nominee. This is a
  nomination, not evidence that it won the award.
  [GECCO 2026 best-paper nominations](https://gecco-2026.sigevo.org/Best%2BPaper%2BNominations)
- Continuous Program Search appears in the GECCO 2026 Companion on pages 477 to 480 rather than
  the main proceedings. [DBLP record](https://dblp.org/rec/conf/gecco/SiperNKSAT26.html)
- Augmentative Topology Agents for Open-Ended Learning appears in the GECCO 2023 Companion on
  pages 671 to 674 and was listed as an accepted poster.
  [DBLP GECCO 2023 Companion](https://dblp.org/db/conf/gecco/gecco2023c)
  [GECCO 2023 accepted posters](https://gecco-2023.sigevo.org/Accepted-Posters.html)

### Inferences

- The trajectory matters more than the raw count: a 2023 poster, a 2024 first-author full
  paper, and two 2026 full papers including a first-author best-paper nominee demonstrate
  current familiarity with what clears GECCO review.
- The mix of EML, GECH, and Companion work makes him particularly valuable for deciding
  whether Safety-Dial has enough evolutionary content for a full paper or should initially
  target a narrower Companion route.

### Gaps

- The official sources do not establish acceptance rates for his submissions, rejected
  submissions, reviewer scores, or how much of each paper he personally wrote.
- ProFiT's SSRN page was not accessible through the research tool, so its detailed methodology
  was not used in this assessment. Its acceptance and authorship were verified through the
  official GECCO accepted-paper list.

## What near-term action would extract the most value?

### Takeaway

Before freezing the protocol or spending simulator budget, schedule one 60 to 90 minute
protocol red-team with Nasir. Send a two-page decision packet in advance and ask him to return
three concrete recommendations: mutation geometry, fair comparison design, and the narrowest
GECCO claim supported by the planned evidence.

### Cited Findings

- The current direction explicitly requires a new protocol, comparison design, and budget to
  be frozen before execution. It treats native-noise steering as the primary bet and
  response-informed coordinates as a prospective extension whose physical benefit and
  evolutionary advantage remain untested.
  [Current ManiGuard direction](../../Evolutionary%20improvement%20on%20ManiGuard.md)
- Continuous Program Search's central design pattern is to diagnose latent behavioural
  geometry first and compare mutation operators under an unchanged optimizer and budget.
  [Continuous Program Search](https://arxiv.org/html/2602.07659v1)
- LLMatic and Mortar both use explicit component ablations, providing relevant experience for
  deciding which mechanism test is indispensable and which extension should wait.
  [LLMatic paper](https://arxiv.org/pdf/2306.01102)
  [Mortar paper](https://arxiv.org/html/2601.00105v1)

### Inferences

The packet should contain only material needed to force decisions:

1. One paragraph stating the hypothesis and the nonclaims: no learned safety filter, no new
   world model, no ROSARL integration, and no assumed evolutionary advantage.
2. One diagram of `observation and instruction -> shared controller -> native noise -> frozen
   family policy -> proposed action chunk -> executed prefix -> simulator outcome`.
3. A table freezing controller dimension, features, bounds, candidate genealogy, objective,
   optimizer, random search, fixed correction, and ordinary-learning comparison.
4. The 320-call decoder-authority audit and 64-attempt pilot budget, with every development,
   search, reevaluation, replacement, and audit charge visible.
5. The four separate claims and their evidence: interface benefit, conditioning benefit,
   evolutionary advantage, and transfer.
6. A blank decision log with `keep`, `change before pilot`, and `defer to paper study` columns.

Ask him to annotate the packet before or during the meeting. The desired output is not a new
research direction. It is a one-page decision record that fixes the simplest defensible EA,
the matched baseline set, the locality diagnostic and mutation bounds, the meaning of a
successful or negative pilot, and the likely GECCO track. This uses his demonstrated strengths
at the exact point where changes are cheap and prevents venue advice from arriving after the
interaction budget has already been consumed.

### Gaps

- Nasir's availability before protocol freeze is unknown. If a meeting is not possible, request
  asynchronous comments on the same two-page packet rather than sending the full repository.
- His review should complement, not replace, a platform check by someone who knows ManiGuard
  and OpenPI and a statistical review of the paper-level sample allocation.
