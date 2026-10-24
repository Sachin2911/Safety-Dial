# Experimentally grounded next mechanisms

## What does the completed work establish, and which claims need correction?

### Takeaway

Jar offers real outcome variance, but no useful execution lever has yet been demonstrated. The retry experiment rejects neither recovery in general nor the evolutionary-wrapper idea: it found no improvement from one frozen joint-motion detector plus one go-home primitive. The next purchase should establish execution leverage before selecting an optimizer.

### Cited Findings

- The replication produced 24 safe successes, 7 unsafe successes, 37 safe failures and 36 unsafe failures in 104 episodes; 103 engaged. Of the 24 benchmark safe successes, 23 met closure-before-carry. The two seed-0 repeats agreed on the four-class outcome on 13/26 tasks. These data support a baseline compatible with the paper, rather than statistical proof of equivalence. [Replication note](../Jar%20replication%20and%20noise%20floor.md)
- The failure diagnosis identified 19 tips after closure while the jar was still on the table, 9 tips before closure, 18 open-lid stalls and 18 closed-lid stalls. The lid-push and regrasp are therefore mechanisms worth targeting. This is observational evidence, not proof that slowing any particular joint trajectory prevents tipping. [Leverage note](../Where%20the%20leverage%20is.md)
- The frozen retry gate completed 96 episodes per arm: native safe success 17/96 and violations 48/96; retry 16/96 and 47/96. Exact paired tests give p=1 for both labels. Retreats fired in 21/96 retry episodes, none of which reached success. Native safe unfinished was 31/96, retry 33/96; all engaged. [Gate results](../../../../progress/receipts/03-results.md)
- The smoke videos showed visibly parked arms whose maximum joint displacement across the detector window was still 0.02 to 0.1 rad, exceeding the frozen 0.01-rad threshold. The scripted retreat subsequently demonstrated clean release, return and policy resumption. [Smoke receipt](../../../../progress/receipts/03-smoke.md)
- The identity checks found a deterministic policy for identical requests, but different rendered input images from separate simulator launches even at identical step-0 states. They also found that the upstream server reseeds only when the episode seed changes; consecutive identical-seed episodes may therefore continue its RNG unless forced through another seed. The gate runner accounts for this case. [Identity receipt](../../../../progress/receipts/03-identity.md)
- The gate scorer uses success and counted_violation, without separately computing closure-before-carry. The current reporting gap can be addressed from preserved monitor sidecars without another GPU study. [Scorer](../../../../scripts/todo3/score_gate.py); [research controls](../../../../AGENTS.md)

### Inferences

- A p=1 null with 96 pairs is not evidence of exact equality or a bound ruling out a smaller improvement. Describe it as no detected improvement for this implementation.
- The 21 triggered episodes form a post-treatment selected subgroup. Their zero successes and the six successes in native counterparts are useful diagnostic observations, not an unbiased causal estimate of the harm from retreat.
- The 75 no-retreat pairs are also selected by the retry trajectory. Their disagreement is evidence of substantial repeat variation, but should not be used as an unqualified unbiased noise-floor estimate.
- The older narrative that physics alone randomizes an otherwise identical approach is too strong after the identity check. Rendering variation already changes the first policy action; contact may amplify it. Likewise, common seeds remain useful layout/task blocks even though they do not produce identical trajectories or guarantee variance reduction.
- Fixing the RNG reseed behavior is engineering hygiene. Changing rendering to make it deterministic would alter the reproduced benchmark and needs an explicit new experimental condition.

### Gaps

- No intervention has yet shown a confirmed safe-completion improvement.
- Mechanistic measurements during the 192 gate episodes are limited because the per-step recorder was disabled after validation. The recorder can be enabled prospectively; its two validation recordings were lost with the rental, while summaries remain. [Identity receipt](../../../../progress/receipts/03-identity.md); [preservation receipt](../../../../progress/receipts/03-preserve.md)

## Which options deserve the next experiment?

### Takeaway

My definite first recommendation is global arm-only contraction of the policy's joint-target commands at an unchanged eight-action execution prefix, not an unvalidated phase gate. It is the cleanest low-dimensional next lever, avoids an unvalidated phase detector, and does not buy additional model queries. A phase-conditioned version is a later refinement if measured leverage exists. This is a proposed experiment, not a claim that contraction will work.

### Cited Findings

- Jar's config explicitly defines eight-dimensional absolute joint targets reconstructed by the policy server from predicted deltas, a raw joint-position controller, execute_horizon=8, gripper binarization and a 2,000-step limit. [Pinned config](../../../../progress/receipts/01-jar-eval.yaml)
- The existing proxy sees seven arm joints, one gripper state and the action chunk. It does not receive jar pose, jar hinge closure, support status or finger-contact state. It already modifies absolute joint targets during retreats and preserves upstream query advancement. [Proxy implementation](../../../../scripts/todo3/supervisor_proxy.py)
- The validated passive recorder can log current joint positions and velocities, end-effector pose, jar pose and velocity, contacts, grasping, monitor propositions and action boundaries without stepping or altering actions. These variables are diagnostic instrumentation, not automatically legitimate supervisor inputs. [Recorder](../../../../scripts/todo2/record_rollout.py); [validation](../../../../progress/receipts/03-identity.md)
- Closure is a five-percent hinge band and flickered in 19 baseline episodes. A raw first-closure gate may activate too early, chatter or miss apparently closed lids. [Leverage note](../Where%20the%20leverage%20is.md)
- RTC explains the real tradeoff in execution horizon: a long prefix reduces reaction to observations; a short prefix can increase discontinuities and mode switching. Shorter therefore does not imply universally safer. [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/html/2506.07339)
- AutoHorizon is already a published adaptive execution-horizon method for flow-based VLAs. Bernoulli-Continuation Policy learns a lightweight continue-or-replan head with the VLA frozen. Adaptive horizon alone is consequently not a new research claim. [AutoHorizon](https://arxiv.org/abs/2602.21445); [BCP](https://arxiv.org/abs/2608.03483)
- APEX inserts a state-feedback execution layer between black-box policies and controllers, reporting reduced tracking error and manipulation improvement. That supports investigating execution independently of the frozen policy, but does not demonstrate benefit on Jar. [APEX](https://arxiv.org/abs/2606.16504)
- PACS brakes along a policy-derived trajectory and uses reachability to handle safety constraints; it reports preserved task success and formal guarantees under its assumptions. It is relevant precedent for minimally disruptive execution intervention, not evidence that a scalar gain inherits guarantees. [PACS project](https://tum-lsy.github.io/pacs/); [author-hosted paper](https://www.dynsyslab.org/wp-content/papercite-data/pdf/roemer-icra26.pdf)
- DSRL already learns steering over the latent-noise space of a frozen diffusion policy using black-box access. Latent steering is a legitimate option, with an existing learning baseline and novelty burden. [DSRL](https://arxiv.org/abs/2506.15799)
- Latent Safety Filters use a learned world model plus safety classification to prevent spilling and toppling from image observations, so non-geometric manipulation safety itself is not unoccupied. [RSS 2025 paper](https://arxiv.org/abs/2502.00935)

### Inferences

- **First choice, target contraction.** At each query with current arm state q and predicted absolute arm target a_i, return u_i = q + alpha*(a_i-q), with alpha=1 reproducing native targets. Leave the gripper component unchanged. A provisional modest set such as alpha in {1, 0.8, 0.6} is interpretable and leaves prefix and query opportunities unchanged. Do not multiply absolute targets by alpha, which contracts toward the robot's zero configuration rather than toward its current state.
- This query-anchored affine contraction is NOT a hard velocity cap and is NOT exact trajectory time scaling. It may change path shape and closed-loop behavior. Measure actual joint velocity, end-effector descent, contact and success timing to test the intended mechanism. If hard per-step target-distance bounds are the desired lever, implement at the action boundary using live q_t and make the different semantics explicit. Even bounding target error does not prove a bound on actual impact force.
- The global contraction can increase timeouts. It is a clean diagnostic because an untested regrasp classifier does not confound it; a later phase gate can reduce the delay after a genuine tradeoff is measured. Moderate contraction could also improve tracking or worsen the lid push, which the recorder should distinguish.
- **Second choice, prefix 4 versus native 8.** This is easy to implement and interpretable, but the shorter prefix approximately doubles model-query opportunities at fixed steps. Report queries and elapsed time separately; if query budget is part of the claimed matched budget, it cannot silently be treated as free. Freeze which resource is held equal. Changing executed horizon through the evaluator is more direct than padding the proxy chunk with repeats, which changes temporal semantics.
- **Third choice, repair retry offline before buying a new gate.** Threshold sweeps on stored proprioception can establish which detector recognizes annotated stalls. A jar-progress detector needs visual estimates or privileged simulator state; an image-change detector can be confused by renderer noise. A larger arm-motion threshold alone may trigger during legitimate precision manipulation. Separate detector accuracy from recovery effectiveness.
- **Fourth choice, contact-stop/shield.** A first-finger-contact stop could interrupt the very contact needed to close the lid or grasp the jar. It needs phase discrimination and a recovery/resume rule. An actual anticipatory tip filter needs tilt or velocity state before the monitor's 30-degree violation threshold; using ground truth is an oracle experiment, not a deployable proprioception-only result.
- **Latent/noise scale.** Keep as one possible bounded fallback, not the main next spend. Same-seed variability neither proves noise steering powerless nor supplies its mechanism. Use the preserved explicit-noise identity hook if tested; do not resume archived pilots.
- **Tiny recurrent controller.** Appropriate only after scalar leverage and measurable gating features are established. Adding a hundred parameters now combines unknown actuation leverage, unknown phase recognition and noisy fitness in one experiment.
- **Critic/world-model reranking.** Needs either new labelled data or a validated model of contact futures and incurs proposal/query cost. Too much new machinery for the next lean step. A frozen feature risk probe could first be checked offline on genuinely pre-event states, with task and episode separation to avoid leakage.
- **Adapter or residual RL.** Can be a scientific comparison later, with the same observation information and interaction accounting. Full weight adaptation changes the frozen-policy question. A low-dimensional learned execution gate is the more direct ordinary-learning comparator once the same gate can be evolved.
- **Switch family.** Stack remains the predeclared backup, with roughly twice Jar's horizon. No evidence here justifies an immediate pivot away from a working replicated family. Earn a transfer experiment after Jar succeeds, or use Stack only after a prospectively bounded failure of the next lever and the planned fallback.

### Gaps

- Neither global contraction nor a deployable phase classifier has been physically validated on Jar.
- A gripper-event gate is available without simulator privilege but is not synonymous with regrasp phase: the lid push and open-gripper motion may precede any binary gripper transition.
- A paired oracle gate based on debounced jar closure/support could test whether phase targeting has potential, but it adds privileged information and should be labelled diagnostic. Do not quietly promote it to the main deployable policy.

## What is the smallest staged route toward a defensible contribution?

### Takeaway

Use the next rental to decide whether execution has leverage, then freeze a very small search space. The paper case needs reliable held-out gains beyond native, a tuned fixed setting and random search, together with an ordinary-learning comparison using equivalent information. The contribution must survive comparison with existing runtime safety layers.

### Cited Findings

- The project explicitly orders variance, leverage, stable evaluation and optimizer comparison; forbids touching held-out tasks until the final comparison; requires both scoring rules, engagement, unfinished outcomes and every simulator interaction; and requires ordinary learning before an evolutionary claim. [Current plan](../../../../firstPrinciples.md); [AGENTS.md](../../../../AGENTS.md); [frozen panels](../../../../progress/panels.json)
- The last 192-episode gate took 7 h 11 min on a two-GPU host. Setup took 43 minutes. The rental receipt records about $0.50/h for that host and total life about 9 h, excluding any assumption that a future offer has the same price. [Run receipt](../../../../progress/receipts/03-run.md); [host receipt](../../../../progress/receipts/03-host.md); [preservation receipt](../../../../progress/receipts/03-preserve.md)
- A new closely related paper, VICS-G, already reranks actions of frozen VLAs for safety and completion, including an execution-feedback study and comparison to a retry alternative. It uses Safety-CHORES, not ManiGuard Jar, and does not establish evolutionary execution tuning. [A Safe Action Is Not Enough, October 2026](https://arxiv.org/abs/2610.05166); [full text](https://arxiv.org/html/2610.05166v2)

### Inferences

1. Finish reporting hygiene without GPU spend: reconstruct closure-before-carry for gate episodes from preserved sidecars, reconcile episode/step/query totals including identity/smoke, and write a new frozen protocol. Do not overwrite the old retry findings.
2. Before a large rollout gate, validate units, alpha=1 identity for identical requests, unchanged gripper behavior and proposed-versus-filtered-versus-executed action logging. Use a small prospectively fixed development diagnostic panel to verify the lever activates and its intended kinematic effect occurs. Such episodes count toward the budget and are excluded from confirmatory inference if used to modify implementation.
3. A compact initial sweep could use native, alpha=0.8 and alpha=0.6 (provisional choices, to be frozen before any run), balanced over the six frozen development tasks with a predeclared seed set, interleaved on one host. Ninety-six episodes per arm gives 288 episodes, useful mainly for large effects. It is a proposal requiring power calibration, not a sufficient design for proving a ten-point effect or safety noninferiority. Avoid winner selection by noisy best rate alone; inspect uncertainty and failure mechanisms.
4. At the last gate's measured throughput, 288 episodes are about 10.8 host hours and about $5.4 at the same historical host price, plus setup, validation, upload, disk and failure contingency. This is historical arithmetic, not a market quote. A $50-$100 next-phase budget supports a staged leverage study comfortably on comparable offers; it should not be promised to cover every baseline, several evolutionary seeds and a well-powered final comparison.
5. The bounded backup, only if the contraction study fails its prospective leverage criterion, is one native-versus-prefix-4 gate on development tasks. Do not run a combined contraction-plus-prefix treatment or reopen retry tuning in this stage. This implements the plan's one-fallback limit, rather than an open-ended lever hunt. A kinematic change alone passes only an implementation check; the scientific gate is a meaningful change in safe requested completion with the violation and unfinished consequences reported. Continue only if the effect is practically meaningful and survives fresh development-seed confirmation with no unacceptable violation rise. If all measured gains are reduced engagement or increased unfinished episodes, the lever failed the intended objective. Freeze exact gates and stopping rules before looking at results; significance failure is inconclusive for small effects.
6. Only after leverage, add at most a handful of parameters such as contraction strength and a simple phase rule. Compare evolution with random search, the best fixed setting and native under the same simulator interaction budget. Also include an appropriate ordinary-learning method over the same representation, or make the limitation explicit and withhold an evolutionary-superiority claim.
7. A strong eventual claim is conditional and narrow: a compact evolved execution supervisor improves safe requested completion under explicitly matched simulator budgets and disclosed model-query cost, and outperforms credible alternatives on untouched tasks and fresh seeds. A Pareto plot is a presentation, not by itself an evolutionary contribution. One scalar can be swept effectively; do not add dimensions merely to make evolution look necessary.
8. If a well-sized, mechanism-validated bounded set of levers remains flat, a careful empirical study of failure modes and noisy evaluation may be defensible as a smaller contribution. It is not equivalent to showing all frozen-policy safety wrappers fail, and does not automatically satisfy a main-track GECCO evolutionary contribution.

### Gaps

- No novelty priority claim follows from this bounded source check. In particular, the older report's broad runtime-safety and no-existing-example language is stale after VICS-G, PACS and Latent Safety Filters.
- The final budget depends on effect size, actual host quote, search repetitions and final-arm count. The quantitative-design researcher should size it from uncertainty and minimum useful effect rather than the older optimistic episode counts.
- No new simulation, policy inference, rental or changes to existing experiment code were performed for these notes.
