# Evolutionary fit and prior work for safe manipulation

Research date: 8 October 2026. This is a problem-first opportunity map, not a proposal
to invent a planning controller. Evolutionary roles below are hypotheses. Documented
headroom does not establish that evolution is necessary, or that a particular remedy
is original.

## Which practical problems give evolution something useful to improve?

### Takeaway

The strongest starting point is task progress after safety intervention. Bounded physical
robustness is another concrete candidate. Reducing unnecessary warnings and comparing
safe search under computation limits are accessible supporting directions, but can
collapse into routine tuning unless the result explains a reproducible limitation.

### Cited Findings

1. **The filter prevents a failure, but the task then stalls.** *When World Models Lie*
   explicitly attributes remaining incompletions to nominal policies not trained to
   resume from states produced by safety intervention. LatentCBF already improves
   intervention smoothness and trains its safety value on mixed nominal/safety-policy
   data, but still acknowledges nominal-policy dependence and filter-induced OOD states.
   These are different targets: improving the safety critic does not automatically teach
   the task policy to continue. [Adaptive-filter paper, V-B](https://arxiv.org/html/2609.34300);
   [LatentCBF, sections 5 and 7](https://arxiv.org/html/2511.18606)

2. **A safe controller becomes unreliable after physical changes.** UNISafe's hard
   block-plucking setting varies block size, weight and friction. Its diffusion-policy
   combination records 38% safe success, 31% failure and 31% incompletion. The authors
   also say that entirely new tower configurations cannot be reliably filtered. These
   establish bounded generalization weaknesses, not their precise cause.
   [UNISafe, appendices D.2 and E.2](https://arxiv.org/html/2505.00779)

3. **Warnings can suppress useful behaviour for the wrong reason.** A world-model
   bimanual failure detector explicitly reports false alarms from benign background
   changes. UNISafe already rejects unfamiliar actions using ensemble disagreement,
   while the adaptive-filter successor already adjusts to observed prediction error.
   Thus uncertainty-based intervention and online mismatch adaptation are occupied ideas.
   [Bimanual detector, VI](https://arxiv.org/html/2603.06987);
   [UNISafe, section 4](https://arxiv.org/html/2505.00779);
   [adaptive filter, IV](https://arxiv.org/html/2609.34300)

4. **Safety evaluation can consume the action-selection budget.** LatentCBF's particular
   DINO-WM implementation exhausts 48 GB GPU memory at 50 model-based action samples,
   whereas its learned critic evaluates thousands of samples in milliseconds. This
   motivates measuring useful outcomes per available computation, but also supplies an
   existing alternative to expensive rollout search. It is not a universal world-model
   inference limit. [LatentCBF, Table 3](https://arxiv.org/html/2511.18606)

### Inferences

The first two problems concern actual robot outcomes and offer the clearest role for
black-box policy or motion optimization. The third initially suits diagnosis and
calibration better than a large evolutionary system. The fourth warrants research only
if limited search actually damages safety or task progress in the selected setup.

### Gaps

Post-intervention stalling must be established in the released UNISafe block-plucking
task. The newer paper uses different manipulation systems. The observations above do
not prove a common causal failure, an unstudied problem, or superiority of any EA.

## What could a standard EA contribute, and what is already taken?

### Takeaway

CMA-ES can search compact task-policy or motion parameters using discontinuous simulator
outcomes. Multiobjective and quality-diversity methods become useful when there is an
actual trade-off or need for alternative behaviours. Their presence alone is not a
contribution; an informative controlled result can be.

### Cited Findings

- **Constraint handling is established.** Wen and Topcu's constrained cross-entropy
  method addresses safe policy search. SafeDreamer applies constrained planning in
  learned latent dynamics. Feasibility-first ranking must therefore be credited as a
  baseline. [NeurIPS 2018 paper](https://papers.neurips.cc/paper_files/paper/2018/hash/34ffeb359a192eb8174b6854643cc046-Abstract.html);
  [SafeDreamer, ICLR 2024](https://arxiv.org/html/2307.07176v3)
- **Robust shooting is established.** Robust cross-entropy planning already combines
  model uncertainty and constraints. Physics-based manipulation filtering already
  evaluates uncertain mass/friction, identifies critical transitions and uses safe
  physical probing. [Robust planning paper](https://learn-to-race.org/workshop-sl4ad-icml2022/assets/papers/paper_16.pdf);
  [manipulation uncertainty paper](https://arxiv.org/html/2509.12674)
- **Diverse safe search is established.** GuSS uses a learned model and MAP-Elites-based
  action planners to explore safe behaviour; its supplement documents cases where
  simpler shooting methods become stuck. ELVIS already uses mixture-based latent MPC
  to retain different predicted futures. [GuSS abstract](https://arxiv.org/abs/2206.09743);
  [GuSS supplement](https://arxiv.org/html/2206.09743v2);
  [ELVIS preprint](https://arxiv.org/html/2605.04709)
- **Model-assisted repertoire search is established.** M-QD screens candidate robot
  primitives with a learned surrogate before expensive evaluation. DA-QD learns
  dynamics and imagines candidate skills. Therefore, an imagined archive with selective
  simulator validation is not itself new. [M-QD](https://arxiv.org/html/2008.04589);
  [DA-QD](https://arxiv.org/html/2109.08522)

### Inferences

**Opportunity 1: improve task continuation under an existing safety filter.**
Keep the model and filter fixed initially; optimize a compact residual task policy or
small set of motion parameters against actual filtered rollouts. The target is safe
completion, including episodes containing interventions. **Simplest family:** CMA-ES;
NSGA-II only if preserving several explicit operating trade-offs is useful. **Closest
precedent:** LatentCBF's smoother filtering and the adaptive filter's remaining task-policy
dependency, cited above. Compare with ordinary task-policy fine-tuning that experiences
the filter, a small hand-designed continuation rule where reasonable, and improved
filtering alone. **Contribution:** a controlled result about policy/filter compatibility;
a method claim would require a specific additional mechanism. **Novelty risk:** medium
to high because changing the task-policy optimizer alone is weak. **Extra infrastructure:**
medium, requiring closed-loop evaluation and enough post-intervention experience. This
does not require irreversibility labels or certification of recovery.

**Opportunity 2: improve safe completion across bounded physical variation.**
Search compact policies or task primitives over a declared distribution of friction,
mass and geometry, then assess untouched settings. **Simplest family:** scenario-evaluated
CMA-ES with credited constraint handling; NSGA-II if average progress and a declared risk
measure cannot sensibly be reduced to one objective. **Closest precedents:** robust CEM,
physics-based uncertain manipulation and UNISafe's hard setting. Compare with ordinary
domain-randomized policy tuning and an existing robust planner. **Contribution:** a
robustness result or useful benchmark showing which controller changes survive a
specified shift while retaining safety. **Novelty risk:** high for merely adding domain
randomization or a worst-case penalty. **Extra infrastructure:** medium, mainly controlled
environment variation and independent evaluation. Search cannot compensate for missing
safety information or an unusable world model at arbitrarily large shifts.

**Opportunity 3: reduce unnecessary interventions without increasing missed danger.**
First separate benign visual changes from physically dangerous changes using independent
labels. If several interacting filter settings matter, search their operating trade-off.
**Simplest family:** NSGA-II over a small configuration space; grid search or Bayesian
optimization are essential alternatives. **Closest precedents:** UNISafe calibration,
the bimanual detector's anomaly scores and adaptive mismatch correction. **Contribution:**
potentially a useful result on when uncertainty helps or obstructs safe manipulation,
with closed-loop confirmation of task benefit. **Novelty risk:** very high for threshold
tuning alone. **Extra infrastructure:** low to medium offline if suitable labelled
trajectories exist, medium for showing changed robot outcomes. EA fit is conditional:
one or two thresholds do not justify an evolutionary study, and a monitor result is not
automatically a control result.

**Opportunity 4: determine which ordinary search family delivers safe progress under a
small budget.** Compare action-sequence or primitive search using the same frozen model,
candidate information and safety objective. **Simplest families:** constrained CEM,
CMA-ES and, only where different viable modes matter, MAP-Elites. **Closest precedents:**
SafeDreamer, robust CEM and GuSS; mixture MPC and a learned safety critic are important
non-EA comparators. **Contribution:** an empirical comparison or benchmark, potentially
identifying a reproducible failure of unimodal search in contact-rich manipulation.
**Novelty risk:** high if it only replaces CEM with CMA-ES and reports aggregate reward.
**Extra infrastructure:** low to medium once a usable planner interface exists, higher
if behaviour descriptors or new primitives must be built. The result needs a concrete
reason, such as useful safe alternatives being discarded, rather than an assertion that
diversity is inherently better. No evolved search scheduler is presumed.

### Gaps

None of these mappings establishes a novel algorithm. A safe multimodal-search failure
has not been demonstrated in the intended arm task. GuSS's accessible version-two HTML
contains supplementary material; the core-method statement above uses its primary
abstract, without inferring unverified implementation details.

## Which options are worth shortlisting, and what keeps them defensible?

### Takeaway

Shortlist task continuation and bounded physical robustness first. Treat warning
calibration and budget comparisons as cheaper diagnostic or supporting studies until a
specific scientific claim emerges. A modest contribution can use a standard EA if it
establishes something useful beyond an optimizer substitution.

### Cited Findings

UNISafe releases model/filter assets, task-policy evaluation and simulator `success`
and `failure` fields, which reduce the need to build an entire safety stack.
[Official repository](https://github.com/CMU-IntentLab/UNISafe);
[simulator wrapper](https://github.com/CMU-IntentLab/UNISafe/blob/isaaclab/dreamer_wrapper.py)
However, those assets do not by themselves establish sufficient post-intervention data
or modern simulator compatibility. The separate
[asset audit](assets_and_entry_cost.md) records these unresolved costs. The
[problem inventory](documented_open_problems.md) separates observed limitations from
proposed remedies.

### Inferences

The key contribution can be evidence: identify a repeatable mechanism, show a bounded
improvement with a simple method, and establish where it fails. Compare equally capable
baselines using actual safety and task outcomes. Report simulator interactions, model
queries and wall-clock cost separately. Improvements purchased by weaker safety criteria
or privileged information must be visible. Repeated trials matter; variance reduction
alone does not establish lower catastrophic risk.

No requirement here forces a new world model, sensor, hyperheuristic or large policy
architecture. Conversely, frozen weights do not make an idea novel. Hidden-information
problems deserve attention but are poor candidates for solving through extra search
alone. Diverse repertoires and surrogate-assisted validation remain available standard
tools, not unused conceptual territory.

### Gaps

This focused search covered constrained CEM, robust MPC, GuSS, model-based QD, mixture
latent MPC and the safety-filter papers above. It did not exhaust every residual-policy,
multiobjective robotics or shield-aware learning paper. No experiments or GPU runs were
performed. UNISafe is CoRL 2025; *When World Models Lie* is a 28 September 2026 preprint.
LatentCBF and ELVIS are treated as preprints, without implying verified acceptance.
Any shortlisted mechanism still needs a narrower prior-art check before claiming method
novelty. The present evidence supports practical investigation, not a promise of
publication or that the simplest candidate will work.
