# Behaviour libraries and continuation after intervention

Research date: 13 October 2026. Focused primary-source search, not an exhaustive novelty claim. No experiments run, archive modified, or pretrained manipulation foundation assumed accessible. Proposed questions and pilots below are research judgments.

## What already exists, and where is the closest overlap?

### Takeaway

Quality-diversity (QD), world-model skill discovery, safe behaviour selection, and repertoire-based stall recovery already exist separately and in overlapping combinations. A generic MAP-Elites archive plus a stall detector is therefore weak novelty. The more specific possibility is understanding which diversity survives a learned safety filter and produces complementary physical continuations without restoring an earlier environment state.

### Cited Findings

- **QD-Plan, August 2026:** *Discovering Diverse Planning Policies for Multimodal Embodied Agents with Quality-Diversity Optimization* evolves prompt-policy templates, archives them by interaction intensity and goal-directedness, detects sustained stalls, rolls back to a checkpoint and switches to an untried behaviourally distant entry. Evaluation is the ThreeDWorld transport benchmark. The paper states ACM Multimedia 2026 proceedings metadata; its public repository exists. This is a close conceptual precedent for an archive that rescues stalled embodied execution, but it is not physical manipulation continuation through a learned safety filter. [Paper](https://arxiv.org/html/2608.08523), [repository](https://github.com/EvoNexusX/2026XuQD-Plan)
- **RF-QD, 2022:** *Learning to Walk Autonomously via Reset-Free Quality-Diversity* uses a learned dynamics model and imagined repertoire to select behaviours expected to remain safe, then deploys previously evaluated behaviours to recover toward safety. It studies legged locomotion within constrained regions. Candidate evaluation is capped at 10,000 per run; the reset-free evaluations are sequential. It explicitly demonstrates the value of diverse behaviours for recovery rather than only optimizing one reset objective. [Full paper, Sections 4 and 5](https://arxiv.org/html/2204.03655)
- **DA-QD, 2021/2022:** *Dynamics-Aware Quality-Diversity for Efficient Learning of Skill Repertoires* already performs repertoire search in imagination using a model trained from skill-discovery experience. The authors report 20-fold sample-efficiency improvement in their experiments. This is not evidence that a manipulation adaptation archive will be similarly cheap. Source code is publicly available. [Paper](https://arxiv.org/abs/2109.08522), [code](https://github.com/adaptive-intelligent-robotics/Dynamics-Aware_Quality-Diversity)
- **QDAC, ICML 2024:** the model-based QDAC-MB variant explicitly builds on DreamerV3. Its skill-conditioned actor balances reward and successor-feature objectives, with evaluations on locomotion and perturbed-environment adaptation. Therefore Dreamer plus diverse skills plus adaptation is already established. [Paper, Section 4.2 and Appendix C.2](https://arxiv.org/html/2403.09930), [published paper](https://proceedings.mlr.press/v235/grillotti24a.html)
- **GuSS, 2022, revised 2024:** *Guided Safe Shooting* provides random-shooting and MAP-Elites planners using a learned model to choose actions under safety constraints. This is online action planning, not a persistent repertoire of adapted task policies. Simply putting safe QD inside world-model planning overlaps directly. [Paper abstract](https://arxiv.org/abs/2206.09743)
- **GDA-QD, 2022:** model-based QD already combines gradients for exploitation and imagined perturbations for exploration. A generic hybrid of evolution and RL is insufficient as a contribution. [Paper](https://arxiv.org/abs/2211.12610)
- **Recovery RL, 2021:** separates a task policy and learned safety-recovery policy, including contact-rich manipulation. This establishes learned recovery as an intervention mechanism; its purpose is returning to safety, which differs from selecting a task-continuation alternative after intervention. [Paper](https://arxiv.org/html/2010.15920)
- **SALT, 2024:** *Robots that Suggest Safe Alternatives* evaluates a pretrained goal-conditioned policy through a reach-avoid value network and suggests alternative goals that satisfy safety and liveness. It includes Franka tabletop manipulation. Changing the requested goal is distinct from completing the original goal through different behaviour, but high-level safe alternatives are established territory. [Paper](https://arxiv.org/html/2409.09883)
- **Safety/liveness filtering, 2023/2024:** reachability-based filters can project nominal control into sets of safety- or liveness-preserving inputs and handle time-specific liveness objectives under the paper's assumptions. We cannot claim to introduce task progress into safety control as a broad concept. [Paper](https://arxiv.org/html/2312.15347)
- **OSCBF, 2025:** *Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions* designs a filtering objective that respects operational-space task hierarchy rather than merely minimizing joint-command changes. This gives a substantive controller alternative when poor progress is caused by the intervention rule. [Paper](https://arxiv.org/html/2503.06736)

### Inferences

- The clearest boundary is between (a) evolving reusable task behaviour, (b) modifying the filter or its safety/recovery controller, and (c) selecting a new action sequence through online search. Mixing them makes the cause of improvement difficult to identify.
- QD is attractive only if there is **complementarity**: different behaviours solve different intervention contexts. Multiple entries with the same successes and failures add memory and collection cost without an established benefit.
- A potentially useful empirical mechanism is **diversity collapse through filtering**. Distinct proposals may all execute the same recovery command or visit the same states. This is a hypothesis to test, not a verified literature gap. Describing this as a new finding requires a further targeted novelty search if the pilot supports it.

### Gaps

- No source found in this focused search proves the exact proposed combination novel.
- No verified ready-to-run pretrained manipulation repertoire under a learned filter was identified. QD-Plan's repository is a planner implementation, not an arm policy package.
- No guarantee from a low-dimensional reachability filter should be transferred to latent safety assessment with learned dynamics.

## Which concrete questions are worth a small decisive pilot?

### Takeaway

The most informative low-cost start is a **small complementarity audit**, before training a large repertoire. Among possible contribution directions, filtered/executed behavioural diversity is more specific than generic policy switching, while imagination-based archive construction has greater implementation burden and heavier prior overlap.

### Cited Findings

- QD-Plan already uses stall-triggered repertoire switching with rollback, while RF-QD already uses safe repertoire selection without episodic resetting. These are the principal overlaps any new small-library continuation method must acknowledge. [QD-Plan](https://arxiv.org/html/2608.08523), [RF-QD](https://arxiv.org/html/2204.03655)
- QDAC already supplies an ordinary RL route to diverse, adaptable skills, including a Dreamer variant; evolution must compete with that class of explanation rather than only a single frozen baseline. [QDAC](https://arxiv.org/html/2403.09930)
- Model-based repertoire discovery already appears in DA-QD and safe action search in GuSS. [DA-QD](https://arxiv.org/abs/2109.08522), [GuSS](https://arxiv.org/abs/2206.09743)

### Inferences

**Rank 1: Which diversity survives the filter and helps continuation?**

Research question: Does retaining behaviours by their **executed, filtered trajectories and continuation outcomes** produce more useful task alternatives than retaining them by proposed motion, nominal-policy behaviour, or parameter distance?

Potential contribution: explain when filtering makes a nominally diverse repertoire redundant, then introduce and validate an intervention-conditioned archive criterion. A naive change from proposal actions to executed-state descriptors alone is probably too small; the contribution would need a reproducible mechanism, a relevant descriptor or selection method, and held-out completion improvement.

Smallest pilot: collect a modest bank of bounded actor adapters or motion-parameter variants evaluated through the same filter. On held-out difficult contexts, measure pairwise overlap of failure/success and distances between proposed versus executed behaviour. Test whether an oracle chooser over this fixed bank has worthwhile headroom beyond the best single variant. If it does, compare equal-size banks selected by nominal diversity, executed diversity, random subsampling, and clustering. Oracle selection is a diagnostic upper bound, never a deployable result. Replayed prefixes or branch rollouts all count toward pilot cost.

Essential alternatives: random bank with the same generated candidates; clustering of those candidates without QD; a single best adapter; ordinary RL of the same adapter; simple retreat/reapproach; and, if appropriate, a task-consistent/smoother filter comparison with changes clearly separated. Equal candidate-generation budget is needed in addition to equal deployment archive size. Train any chooser on training contexts only and expose identical observations/intervention histories to alternatives.

Kill conditions: proposed and executed diversity already agree; best single adapter closes nearly all the headroom; different failures have no complementary safe solutions; benefit disappears against clustering; descriptor requires privileged simulator information unavailable to the baselines.

**Rank 2: Can a tiny intervention-conditioned repertoire outperform a single adapted task policy?**

Research question: With the same total simulator budget, can a handful of evolved continuation behaviours selected from recent intervention history improve safe completion more than a single adapted policy or a random/clustered bank?

Potential contribution: demonstrate how small, reusable behaviour changes solve distinct post-intervention contexts without physical rollback. It is more convincing if switching is conditioned on a diagnosed failure type than if triggered only by a timeout. This is narrower than generic QD recovery, but still faces strong RF-QD, QD-Plan and recovery-controller overlap.

Smallest pilot: use the same fixed candidate bank as Rank 1, compare its best individual against cross-context oracle selection, and inspect whether useful alternatives can be chosen from observable history. Only then optimize archive discovery or selector learning. Start with very few alternatives and one task; a large archive is unnecessary for testing the hypothesis.

Essential alternatives: original filtered actor; deterministic simple continuation; same-size random and clustered banks; equal-budget independent restarts of single-policy evolution; ordinary filtered fine-tuning with the same compact parameters and history; and an RL option selector over the same fixed behaviours. Count selector training separately and include it in total interactions.

Kill conditions: no avoidable stalls; restarting/reapproaching with one simple rule works as well; chosen library depends on privileged mass/state labels; switching breaks recurrent state estimation; online selection cannot approach the diagnostic upper bound; adaptation only benefits by receiving more time than the baseline.

**Rank 3: Can model-assisted evaluation cheaply build a reliable continuation repertoire?**

Research question: Does selecting which imagined behaviour candidates deserve simulator evaluation improve the safe-completion versus total-interaction tradeoff for a fixed repertoire representation?

Potential contribution: an intervention-specific model-trust rule, if it demonstrably avoids imagined progress and safety errors. Using a world model to accelerate QD alone is already covered by DA-QD, GDA-QD and QDAC-MB, while safe imagined planning overlaps GuSS. This is a higher-cost extension, not the first honours pilot.

Smallest pilot: rank the same candidate set by imagined continuation outcomes and actual simulator outcomes from matched logged contexts. Quantify ranking errors and unsafe false acceptances, stratified by interventions and model uncertainty. Check whether a simple model-error or disagreement rule separates reliable predictions before proposing an elaborate acquisition method.

Essential alternatives: direct simulator evaluation; random candidate screening; uncertainty-only screening; model ranking without trust control; random shooting/CEM if the object being searched is an action sequence rather than a reusable adapter; ordinary RL with comparable model-query and simulator budgets. World-model training data and retraining interactions belong in total cost.

Kill conditions: frozen model does not rank candidates meaningfully; uncertainty is uninformative; required retraining dominates saved interactions; short-horizon predictions miss the continuation outcome; simulation is cheap enough that imagined screening provides no practical benefit.

### Gaps

- Suggested pilot sizes are intentionally unspecified: controller throughput, prevalence of stalls, viable search dimension and confidence requirements have not been measured.
- A repertoire is not yet justified by the available repository evidence. The cross-context complementarity audit is the decisive prerequisite.

## What assets, costs and controls make this feasible?

### Takeaway

The smallest extra engineering layer is an adapter bank and a chooser around a faithfully running filtered baseline. Building a new model-based QD training stack, latent-conditioned manipulation policy or guaranteed liveness filter substantially increases the project scope.

### Cited Findings

- UNISafe's official release includes task-actor/world-model and safety-filter components, but its README requires Isaac Sim and setup-specific installation. This inspection verifies a release, not a locally working baseline. [Release](https://github.com/CMU-IntentLab/UNISafe)
- The local earlier interface audit identified a seven-coordinate actor, a filter replacing six arm coordinates while preserving the gripper command, action-normalization layers, recurrent latents and no verified exact mid-episode restoration. It also found logging extensions necessary. These are static inspection results, not runtime guarantees. [Local feasibility audit](../Safe%20manipulation%20progress%20and%20robustness/feasibility_and_evolutionary_scope.md)
- QDAC's official repository contains DreamerV3 code, QDAC-MB entry points and adaptation scripts. Its documented platform is Python 3.10; adaptation examples are locomotion. These are useful methodological assets but are not drop-in manipulation checkpoints for this Python 3.11 repository. [Repository](https://github.com/adaptive-intelligent-robotics/QDAC)
- DA-QD has publicly reachable source, while RF-QD's project page and full paper describe its method. No RF-QD dependency compatibility or pretrained controller package was verified here. [DA-QD code](https://github.com/adaptive-intelligent-robotics/Dynamics-Aware_Quality-Diversity), [RF-QD project](https://sites.google.com/view/rf-qd/home)

### Inferences

- Count every simulator transition used to generate candidates, rejected candidates, archive maintenance, selector training, initialization data, failed evaluations and replayed prefixes. Report model queries and elapsed time separately. Equal deployment archive size does not make pretraining budgets equal.
- Keep proposed actions, filter outputs and executed simulator actions separately. Dynamics histories require executed actions in the correct coordinates. A critic of the complete filtered system may deliberately take proposed actions instead; these are different semantics.
- A simulator state reset cannot be assumed to restore contact-solver state, actuator/controller state, camera history or recurrent actor/filter state. Prefer full-episode evaluation until prefix replay or restoration fidelity is established. Do not deploy physical rollback as if it were ordinary continuation.
- Switching behaviour should preserve the correct observation/executed-action history or explicitly reconstruct each policy's latent. Resetting only the task latent at intervention could itself explain a gain and deserves a simple baseline.
- Safe completion, physical violations and unfinished episodes are joint outcomes. A fixed filter does not establish equal safety under changed state occupancy. Zero violations in n independent episodes supports an approximate 95% upper rate bound of 3/n, not zero risk.
- Overall recommendation: first test complementary continuation with a **small bank**, then decide whether evolution, QD, or a single adapter is justified. Rank 1 offers the clearest specific hypothesis; Rank 2 is a feasible method only if complementarity exists; Rank 3 should follow successful model-ranking evidence.

### Gaps

- No simulator runtime, checkpoint load, closed-loop throughput, baseline stall prevalence or GPU budget was verified in this search.
- No evidence yet establishes that safe alternative behaviours exist under the chosen filter, that a small adapter can express them, or that the learned model supports counterfactual ranking from intervention states.
- No proposed contribution here is a publication promise. The strongest result may be an empirical explanation of filtering's effect on behavioural diversity, but that claim requires both positive experiments and a dedicated novelty check.
