# Scientific path to a GECCO paper

## What contribution is defensible

### Takeaway

The most focused route is an empirical contribution about when compact evolutionary adapters produce useful safe-completion gains around frozen world-model planners. The paper needs one consequential, supported conclusion; it need not demonstrate a new optimizer, beat every baseline, solve every planned environment, and calibrate a risk dial simultaneously.

### Cited Findings

- The governing question is safe-completion recovery with compact adapters around frozen models, potentially reducing manual behavioural tuning. Optimizer superiority, comparability, reduced tuning and same-weight transfer are explicitly distinct claims. LeWM is the opening study and Block Plucking the intended main study. [Current authority](../../../../firstPrinciples.md).
- Compact controllers evolved using world-model representations already have a direct precedent: World Models trains a small controller separately from the learned representation and uses CMA-ES. The general combination of a world model and a small evolutionary controller is therefore insufficient novelty by itself. [Primary project](https://worldmodels.github.io/).
- Confidence in satisfaction of probabilistic constraints has also been incorporated into policy learning before this project. [Belzner and Wirsing](https://arxiv.org/abs/2005.03898).
- Actor-guided model-based planning with deployment-time safety constraints has prior empirical and methodological treatment. [LOOP proceedings paper](https://proceedings.mlr.press/v164/sikchi22a.html).
- The current manuscript already recognizes these related-work boundaries and distinguishes its fixed-utility PPO comparison from constrained RL. [Working manuscript](../../honours-gecco/world-model-adapters.tex).

### Inferences

- Recommended core question: under what conditions does evolutionary adaptation improve safe completion beyond the frozen planner and the adapter's untrained structural effect? A specific reusable interface plus evidence explaining its scope could support a useful empirical/application paper without claiming optimizer invention.
- TwoRoom can provide controlled diagnosis and essential ablations. A successful Block Plucking confirmation would substantially strengthen practical relevance. Two environments are a sensible project scope, not a universal publication requirement. PushT's failed qualification can be a documented boundary; Cube and ManiGuard need not be revived.
- A method paper is a different, higher-obligation route: identify a substantive change beyond standard CMA-ES, feasibility ranking or a generic residual network; explain why it addresses the diagnosed problem; compare against the closest existing approach and isolate the change. The current record does not supply that method novelty yet.
- A systematic empirical or negative paper is possible if it establishes a reliable, generalizable finding about adapter authority, search or evaluation. It needs carefully controlled evidence that distinguishes competing explanations. A sequence of failed pilots, an underpowered difference, or one weak PPO configuration does not by itself establish that evolution does not work or that the interface is necessarily the bottleneck.
- Candidate contribution sentence, to be earned rather than asserted now: "We characterize and evaluate compact evolutionary adaptation around frozen world-model planners, showing when it improves safe completion and when interface restrictions prevent useful recovery."

### Gaps

- A verified novelty assessment requires comparing the final proposed interface and finding against the closest papers, including residual policy learning, evolutionary controller adaptation, constrained planning and safe neuroevolution. Broad searches cannot establish an absence claim.
- The practical effect size that would justify the research contribution has not been chosen prospectively for a new final comparison.

## What evidence is missing for each claim

### Takeaway

The main missing result is reliable learned benefit beyond native and untrained controls on fresh cases, with uncertainty narrow enough to support the stated claim. Optimizer superiority, reduced tuning and risk calibration each require additional, claim-specific comparisons.

### Cited Findings

- TwoRoom's initial replication showed CMA safe completion of 8/36, 6/36 and 11/36, versus native 3/36; both exploratory contrast intervals included zero, and one untrained reference beat its CMA run. [Replication](../../../../SafeLeWM/results/09-adapter-replication.md).
- The mechanism study found CMA 45/216 and random 47/216, versus native 7/72 and untrained 32/216. Hazard relocation changed 72/72 safety-aware action sequences and 0/72 blind action sequences, establishing responsiveness but not necessarily safer use of the information. Its ablation contrasts were uncertain and some nominees were identical. [Mechanism study](../../../../SafeLeWM/results/12-adapter-mechanism.md).
- The separate matched comparison found CMA 44/216, random 37/216 and PPO 20/216. Both primary exploratory intervals included zero; only one sparse-terminal-reward PPO configuration was tested. The three repeats reused the same 24 layouts. [Matched PPO study](../../../../SafeLeWM/results/14-matched-ppo.md).
- Block Plucking Search 2 found 14/48 sustained safe completions for CMA, 16/48 untrained and 9/48 native, with violations 30/48, 24/48 and 28/48. Random search was complete in only one repeat. [Pooled Search 2](../../../../SafeBlock/Search2Pooled.md).
- PushT reproduced task competence, but its qualified head-coordinate probes achieved only 0-1/10 safe completions on admitted scenes. This is a failed gate for the tested interface, initialization and sampling distribution, not proof all adapters fail. [Validated qualification](../../../../SafeLeWM/results/16-pusht-validated.md).
- The completed dial library used policies acquired with fixed weighted utility. Seventeen of 45 assignments were unavailable; five of 28 available assignments exceeded requested empirical limits. These are development results and not calibrated population risks. [Dial library](../../../../SafeLeWM/results/17-safety-dial-library.md).

### Inferences

- For useful adaptation: hold backbone, monitor, information and task fixed; compare native, multiple untrained draws, the learned method and parameter random search on the same fresh source cases, with independent training repeats. Retain an ordinary-learning comparator for the existing evolutionary-adaptation question. The principal conclusion may concern usefulness even if random search remains competitive; explain that result rather than hiding it.
- For optimizer superiority: use equal declared information and actual interaction allowances, matched selection work, a defensible bounded configuration-selection allowance for each method, paired evaluation and uncertainty over both learner and task variation. One low-budget PPO configuration only supports a configuration-specific statement. A broader superiority claim requires broader evidence.
- For reduced tuning or matching an expert: add an appropriate tuned fixed reference under the same task/information conditions; predeclare an acceptable performance margin; record prospective development trials, selection evaluations and remaining human choices. A nonsignificant difference is not equivalence. Do not retrospectively invent human-time savings. These obligations are unnecessary if reduced tuning is removed from the claimed contribution.
- For a risk-budget contribution: evaluate a method actually designed for the requested limits; separate candidate acquisition, selection and untouched risk estimation; report unavailable targets; evaluate completion alongside violations; include uncertainty and any simultaneous-selection considerations in the stated guarantee. Feasibility ranking without a penalty is not itself new. Current library results are useful motivation and a limit, not proof of calibration.
- For a mechanism claim: choose the one explanation that makes the central result informative and change that factor alone. Existing hazard-information and untrained controls are useful foundations. Do not demand a full factorial grid or an ablation of every implementation detail. A larger representation helping after bundled changes does not isolate representation capacity as the cause.
- For confirmation: freeze representation, objective, method choices, panels, seeds, nominee rules, primary contrasts and analysis before new outcomes. Size task and training repeats from development variability and a chosen meaningful effect or desired precision, rather than picking an arbitrary number of seeds. Do not expand sample size until significance appears.

### Gaps

- Current primary uncertainty is too broad for strong optimizer conclusions. Its resolution cannot be predicted from episode count alone because layouts, learner repeats and hardware blocks matter.
- The present evidence does not isolate why Block Plucking learning fails to beat untrained selection. Representation, observation content, search budget, noisy selection and distribution shift remain possible contributors.

## What sequence would make the submission ready

### Takeaway

Agree one primary claim and its evidence threshold with the supervisor, resolve the most consequential uncertainty under a bounded development protocol, then freeze and confirm it. A paper can preserve negative results while concentrating its main argument on one result and its explanation.

### Cited Findings

- Existing authority permits no automatic resumption of completed PushT, adapter or dial studies, and requires finite prospective protocols for new stages. The L4 population-size study remains separately active in the governing record. [Current authority](../../../../firstPrinciples.md).
- The working manuscript still describes Search 2 R1 as the first completed repeat with two repeats active, while the pooled report records all three completed. The later pooled report should inform the current synthesis, with audit status and incomplete resource accounting retained. [Manuscript](../../honours-gecco/world-model-adapters.tex), [pooled Search 2](../../../../SafeBlock/Search2Pooled.md).
- Local output deletion was authorized, but not every deleted item has a verified backup; sixteen SafeLeWM archives have a documented verified cloud route. Submission artifacts must be assembled from actually recoverable evidence. [Current authority](../../../../firstPrinciples.md).

### Inferences

- First write a one-page claim-and-evidence outline: proposed contribution sentence, closest prior work, one primary result, the comparison that would falsify it, and the minimum explanatory ablation. Treat this as a supervisor decision, not a new experiment authorization.
- Then bound the development question. For the practical-adaptation route, first establish an interface with useful recovery and a promising learned-versus-untrained signal on development data. Do not launch a large optimizer or risk-dial matrix before that exists. Any Block Plucking or replacement-interface work needs its own new authorization and frozen protocol.
- Go criterion: the selected method shows a practically meaningful development gain over native and untrained controls across relevant cases, and the mechanism/limitations can be explained well enough to justify a prospectively powered or precision-planned confirmation. This is an advancement rule, not final scientific proof.
- No-go criterion: if the bounded development study cannot establish useful learned benefit, stop expanding that positive claim. Assess whether controlled evidence can instead support a specific empirical limitation; otherwise target an appropriately scoped poster, workshop or student submission. A null result alone does not make a main-track contribution.
- Final evidence should support a compact main outcome figure, a budget/performance comparison and one explanatory result. The many historical studies can become context or supplementary evidence without concealing contradictory outcomes.
- Before submission, reconcile manuscript status, audit the exact reported rows, provide an anonymous reproducibility package with pinned code/configuration/model provenance, separate recoverable evidence from historical assertions, and explain information boundaries and dataset familiarity. Do not promise raw artifacts that no longer exist locally or lack a verified remote copy.

### Gaps

- Venue-specific deadlines, page limits and submission alternatives are covered by the separate venue research.
- No new experiments, budgets, rentals or manuscript changes are authorized by this assessment.
