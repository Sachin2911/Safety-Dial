# ManiGuard phase-two related work and baseline choice

Reviewed 17 October 2026. This note compares five primary papers and the author code
located during the review. It supports [phase2.md](../history/20261018/phase2.md); it is not an
exhaustive novelty review, a reproduction, or evidence that our proposed method works.
No models were downloaded and no experiments were run for this review.

The most practical external comparison is **Adaptive Action Chunking (AAC), with its
joint-space adaptation explicitly identified**. Its entropy-based execution schedule
addresses one part of our controller using an accessible chunk interface. A faithful
port needs more work than changing a chunk-length constant. VLA-Pilot is the most direct
warning against claiming novelty for evolutionary steering of a frozen VLA; Mostly
Harmless is an important precedent for learning a reusable steering policy and testing
its transfer.

## What the methods actually adapt

“No VLA fine-tuning” and “no adaptation training” are different claims. Costs below are
additional to the original backbone's training and ordinary deployment.

| Method and inspected paper version | Representation and runtime information | Adaptation interactions and deployment cost |
| --- | --- | --- |
| [VLA-Pilot, 2511.14178v2](https://arxiv.org/html/2511.14178v2) | Evolves a population of action proposals during inference. Image/language reasoning constructs a reward grounded in visual keypoints; elite resampling and partial diffusion refine proposals. Post-execution feedback can revise steering. | No additional task-specific training for the steering procedure. Each steering step adds visual/MLLM processing, population sampling, scoring and repeated denoising. Evolution is performed during deployment, rather than compressed into a previously learned adapter. |
| [AAC, 2604.04161v2](https://arxiv.org/html/2604.04161v2) | Chooses an execution prefix from uncertainty across sampled chunks conditioned on the same observation. Gaussian motion entropy and binary gripper entropy determine an entropy-change point, subject to a minimum-movement floor. | The scheduler needs no training rollouts. The paper uses 20 proposals per query; batching can reduce latency without eliminating the extra computation. Backbone fine-tuning and any parameter selection remain separate costs. |
| [CheckVLA, 2607.26789v1](https://arxiv.org/html/2607.26789v1) | Compares action-conditioned predicted features with arriving observations. A causal risk head triggers a latency-aware rewrite of the executable suffix; keyframe memory preserves earlier context. | Requires world-model/risk-head training, calibration and validation. Deployment adds continuous verification and occasional VLA calls. The reported verifier adds 88.4 million trainable parameters, excluding its frozen encoder. This is a larger training and integration project. |
| [Mostly Harmless, 2606.12299v1](https://arxiv.org/html/2606.12299v1) | Learns a visual language-feedback policy that changes instructions given to a frozen VLA. A calibrated improvement predictor can decline steering. | Narrated videos, interactive language-search rollouts, rejection fine-tuning and calibration train the wrapper. Deployment adds language generation and the improvement head. Freezing the VLA does not make this a training-free method. |
| [SEAM, 2607.04609v1](https://arxiv.org/html/2607.04609v1) | Uses the previous chunk's unexecuted tail to correct the next chunk inside each flow-matching Euler step. Requires sampler access and temporally aligned action history. | No adapter-training rollouts or policy-network backward passes. Adds analytic corrections within denoising. Its near-baseline timing result concerns the cached-prefix denoising loop, not complete robot execution. |

## Safety and transfer are distinct evaluation questions

| Method | What the safety or reliability evidence means | Transfer scope in the inspected paper |
| --- | --- | --- |
| [VLA-Pilot](https://arxiv.org/html/2511.14178v2) | Main outcomes are task success and steering-objective alignment, not a ManiGuard-style physical-violation rate. | Simulation and hardware tasks, multiple policies and embodiments; the stated ID/OOD split is relative to baseline verifiers' training exposure. This is not our proposed adapter-held-out family split. |
| [AAC](https://arxiv.org/html/2604.04161v2) | Includes a qualitative tabletop-collision example. It does not establish a benchmark-level bound on physical violations. | RoboCasa, LIBERO, perturbed LIBERO-Pro and real tasks, including GR00T and π0.5. A shared inference rule is demonstrated, rather than transfer of learned adapter weights. |
| [CheckVLA](https://arxiv.org/html/2607.26789v1) | Its conformal statement controls an unnecessary first intervention under exchangeability. It does not certify collision avoidance or task safety. | RoboCasa365, including unseen compositions and perturbation audits. Distribution shift and calibration assumptions matter for interpreting the intervention guarantee. |
| [Mostly Harmless](https://arxiv.org/html/2606.12299v1) | “Harmful” means steering reduces task performance. Conformal refusal controls harmful-steering false positives under its calibration assumptions, not arbitrary physical hazards. | Visual/semantic shifts, 13 unseen task compositions and two unseen hardware tasks. This is substantive transfer evidence and must be acknowledged in our positioning. |
| [SEAM](https://arxiv.org/html/2607.04609v1) | Evaluates executed-action smoothness alongside task success. Lower jerk is not a physical-safety guarantee. | π0.5 on LIBERO-10. The inspected evaluation does not establish transfer to entirely withheld manipulation families. |

Our inference from these comparisons is that each ingredient already has relevant
precedent. A defensible research question concerns whether one compact execution
controller can jointly improve measured safe completion and transfer under a common
information contract, and whether evolution helps compared with matched optimizers.
The combination alone establishes neither novelty nor superiority.

## Author-code audit and reproducible starting point

The [AAC project page](https://lance-lot.github.io/adaptive-chunking.github.io/) links to
the authors' GitHub organization. Its [LIBERO repository README](https://github.com/Adaptive-Action-Chunking/libero/blob/ff8c8e421f7805b5845a72d07fbd8f1171df74bf/README.md)
identifies this as the official evaluation implementation and links a multi-sample
GR00T server. Pin the inspected client source to
[`ff8c8e421f7805b5845a72d07fbd8f1171df74bf`](https://github.com/Adaptive-Action-Chunking/libero/commit/ff8c8e421f7805b5845a72d07fbd8f1171df74bf),
dated 17 June 2026. This pin establishes source identity, not compatibility or reproduced
results. The README's separate LIBERO checkout pin is not the AAC source revision.

The released [π0.5 client](https://github.com/Adaptive-Action-Chunking/libero/blob/ff8c8e421f7805b5845a72d07fbd8f1171df74bf/pi05_benchmark/inference_client_action_entropy_pi05.py#L207)
expects both sampled actions and normalized actions, each with a sample dimension. It
calls `select_chunk_size` and normally executes sample zero. It does not rank proposals
by simulated success. The [active launch script](https://github.com/Adaptive-Action-Chunking/libero/blob/ff8c8e421f7805b5845a72d07fbd8f1171df74bf/pi05_benchmark/action_entropy_benchmark.sh#L14)
selects Gaussian-plus-Bernoulli entropy and a movement threshold of four.

The [π0.5 entropy module](https://github.com/Adaptive-Action-Chunking/libero/blob/ff8c8e421f7805b5845a72d07fbd8f1171df74bf/action_optimization/action_entropy_pi05.py#L247)
uses the maximum adjacent difference of cumulative mean entropy, then applies a movement
floor. Other functions minimize mean entropy, but the inspected launcher does not call
them. Its movement calculation sums Cartesian translations, composes rotations and
weights gripper changes by 0.2. These are representation-dependent choices. They must
not be applied unchanged to seven absolute joint targets. The code also differs from
the paper's appendix defaults, which describe a movement threshold of three and a unit
gripper-change contribution. Record whether a port follows paper equations or the
released π0.5 configuration; do not silently mix them. [Paper appendix](https://arxiv.org/html/2604.04161v2#A1)

The [VLA-Pilot project](https://rip4kobe.github.io/vla-pilot/) labels its code as coming
soon; the inspected link is a placeholder. The [Mostly Harmless project](https://hyunjoe.xyz/LanguagePolicy/)
exposes no implementation link in its inspected page. No author implementation was
verified for CheckVLA or SEAM through their paper links and focused repository searches.
These are availability observations on the review date, not claims that no code exists.

## Recommended ManiGuard comparison and implementation work

Start with **AAC-joint**, a clearly named adaptation of the pinned method. This is an
engineering recommendation, not a completed baseline. ManiGuard supplies seven absolute
arm-joint targets and a signed gripper command; the proposed adapter already works at
that execution boundary. Its shared sixteen-action prefix is specified in
[phase2.md](../history/20261018/phase2.md). The necessary port has five parts:

1. **Expose multiple independent proposals.** At each query, collect a fixed number of
   stochastic chunks from exactly the same observation and instruction. Begin the
   feasibility audit with the paper's sample count, without asserting it fits available
   memory. Preserve the original executed sample's RNG stream and use independent keys
   for the extra samples. Repeating a cached deterministic chunk is not an entropy
   estimate. Batched inference and sequential repeated queries have different costs.

2. **Define the coordinate adaptation explicitly.** Estimate continuous uncertainty
   over all seven normalized arm coordinates, and binary uncertainty from the actual
   gripper sign. Use fixed, shared normalization and documented covariance
   regularization. The movement floor must measure displacement relative to the
   current joint state, not the sum of absolute joint positions. Freeze its units,
   weighting and threshold using permitted development data. This changes the original
   Cartesian construction and must be described as an adaptation. Forward kinematics
   could instead supply Cartesian reference trajectories, but it introduces another
   mapping whose frame, rotation and normalization conventions require validation.

3. **Preserve the selection mechanism.** Retain entropy-change selection and the
   movement floor; execute the chosen prefix of the original proposal. Do not add our
   cap, smoothing or release-delay heads to this comparator. On synthetic native-format
   fixtures, check the selector against the pinned author function before introducing
   joint coordinates. Include short chunks, equal-entropy samples and gripper changes.
   Do not silently quantize its horizon to our controller's decision grid. If a grid
   restriction is necessary, label and report that extra approximation.

4. **Account for information and computation.** AAC-joint sees a proposal distribution
   that our current single-proposal actor does not. Report that advantage explicitly.
   Keep scene observations and outcome monitors common; expose no goal predicates to
   either controller. Count generated proposals, denoising work, policy requests,
   simulator steps, memory and wall time separately. Equal episodes do not equal
   deployment compute. Include a fixed-horizon control with the same proposal-generation
   cost to isolate the scheduling decision when interpreting timing comparisons.

5. **Freeze evaluation before running.** Choose one setting jointly on permitted
   development families, then freeze it across reserved scenes and families. Report safe
   completion, violations and unfinished episodes per family. Include all parameter
   selection interactions. Adding this arm needs an explicit protocol and budget update;
   this review does not enlarge the currently planned pilot automatically.

A full reproduction would first match the authors' backbone, action representation,
normalization, sampler, settings and benchmark. An AAC-joint result is a new application
of the mechanism; it cannot be presented as a faithful reproduction of their published
numbers. Conversely, a generic variance trigger or fixed `{4, 8, 16}` rule is only an
AAC-inspired heuristic. SEAM would be a reasonable later comparison if sampler editing
is justified, but ordinary output smoothing would not implement SEAM.

## Consequences for phase two

Keep three conclusions separate: a shared adapter may help; its frozen weights may
transfer; evolution may be a better optimizer. The matched random-search and recurrent
PPO arms address the third question. AAC-joint addresses whether the learned wrapper
improves beyond a relevant nonlearned execution mechanism. None replaces the others.

Evolution in our proposal spends simulator interactions before deployment to select a
single recurrent parameter vector. The trained controller then chooses execution
settings from current observations and history. That distinction from online proposal
search is useful, but it is a problem formulation to evaluate, not a novelty claim.
Likewise, shared weights do not imply one generalist backbone: the frozen family
policies remain different. Claims about continuation under an intervening learned
safety filter require the separate filter extension in the plan.
