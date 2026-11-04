# Scope and adapter interfaces

## Should the research extend beyond world models

### Takeaway
The outer recipe can target frozen pretrained control systems, including VLAs. The transferable contribution is a compact adaptation procedure under explicit safety feedback; a planner-specific adapter that requires predicted futures cannot automatically be reused with a VLA.

### Cited Findings
- OpenVLA maps images and language through a visual encoder, projector and language backbone to action tokens decoded into continuous robot commands. It is a policy interface, not an exposed action-conditioned transition predictor. Its checkpoints and training code are public. [OpenVLA](https://openvla.github.io/).
- OpenVLA-OFT uses continuous action chunks, parallel decoding and an action head. Its official inference example supplies two camera views, proprioception and a task description, then returns an action sequence. The code documents roughly 16 GB GPU memory for LIBERO inference. [Official repository](https://github.com/moojink/openvla-oft).
- The authors report 97.1% average success across four LIBERO suites, demonstrating a plausible competent starting policy, not an established safety baseline on a modified task. Their action-generation speed and latency improvements are different metrics, so action throughput should not be described as episode speed. [OpenVLA-OFT](https://openvla-oft.github.io/).
- SafeVLA already applies constrained learning to VLA safety, including long-horizon mobile manipulation and explicit safety-performance evaluation. Its reported cost reductions concern cumulative safety cost, which is not automatically the project's episode-any-violation probability. [SafeVLA](https://arxiv.org/abs/2503.03480).

### Inferences
- Broaden the conceptual scope to frozen pretrained controllers while retaining one qualified main setting and one transfer test. Treat VLA transfer as an empirical question rather than a prerequisite that expands the main experiment indefinitely.
- World model and VLA are not mutually exclusive categories: a system can combine both. Distinguish whether the particular released interface provides predictions of candidate consequences, action proposals, hidden features, or all three.
- A future-scoring adapter needs action-conditioned model predictions or another evaluator. Adding a learned world model to a VLA creates a larger method with extra training and acquisition costs; it must not be silently treated as the same frozen-policy recipe.
- The most reusable outer contract is frozen observations/features plus proposed actions, a compact intervention module, and observed episode-level success/violation feedback. Whether its authority suffices is task-dependent.

### Gaps
- No evidence here establishes that the same adapter architecture works across LeWM and a VLA. That would require prospective qualification and actual tests.
- Public checkpoint availability does not guarantee task competence after simulator, camera, action-normalization or scene changes.

## Which adapter contracts can be compared fairly

### Takeaway
Separate the general learning recipe from architecture-specific connection points. A small action/chunk adapter or compact head adapter could be evaluated across frozen systems; planner-future selection is a more specialized method.

### Cited Findings
- The OpenVLA-OFT implementation separates its pretrained VLA, proprioceptive projector and continuous MLP action head in the inference example. [Official repository](https://github.com/moojink/openvla-oft).
- Original OpenVLA reports that LoRA adapts 1.4% of model parameters and that last-layer-only fine-tuning performed poorly in its tested adaptation setting. A parameter-efficient gradient method can therefore still involve a large number of weights, and a tiny evolved head should not be presumed sufficient. [OpenVLA](https://openvla.github.io/).
- The repository's validated PushT study had native completion 12/16 and violations 16/16, but its 16 random adapter probes achieved only 0-1/10 safe completion on admitted cases. Competence and hazard presence did not imply useful adapter authority. [Local PushT record](../../../../SafeLeWM/results/16-pusht-validated.md).
- Block Plucking Search 2 recorded 16/48 sustained successes for the untrained selector, 14/48 for CMA and 9/48 for native. The interface effect and learned effect remain distinct. [Local Search 2](../../../../SafeBlock/Search2Pooled.md).

### Inferences
- Candidate common interface A: a compact state-conditioned sidecar receives accessible frozen features, proprioception, native action or action chunk and short history, then outputs bounded corrections or action-head modulation. Action scales, chunk execution and gripper semantics must be bound explicitly per robot. Gripper corrections may be highly discontinuous and destructive.
- Candidate interface B: a low-dimensional adapter acts inside an exposed policy/action head, keeping backbone and neutral behaviour fixed. This is less portable than an output-side module but may preserve competence better; that is a hypothesis requiring qualification.
- Do not evolve a full 7B model or assume that evolving all LoRA weights is inexpensive. Specify actual search dimension and construction. If a frozen projection or basis reduces that dimension, disclose how it was obtained and whether it uses demonstrations or development outcomes.
- Gradient-based learning can train the same compact adapter while the backbone remains frozen. Use it as the fair optimizer comparator. Full-VLA fine-tuning is a separate adaptation-system reference and need not be the only RL baseline.
- A shared recipe means shared architecture/ranking/selection choices where technically meaningful, with separately learned weights. It does not mean identical weights, identical raw latent dimensions, or universally interchangeable checkpoints.

### Gaps
- No particular correction bounds, latent basis or action-head dimension is qualified by this review. The earlier failed residual screen is not authorization to substitute a new experiment.

## How should testing grounds be selected

### Takeaway
Select an environment-policy-monitor combination because it contains a reproducible, avoidable safety failure, before observing which optimizer wins. The benchmark should expose the adaptation problem rather than create a backbone-repair project.

### Cited Findings
- SafeVLA's Safety-CHORES examples include task completion accompanied by repeated collisions and collateral object disturbance, as well as impossible tasks where stopping reduces cumulative cost. These are different outcome regimes and should be reported separately. [Project demonstrations](https://pku-safevla.github.io/).
- OpenVLA and OpenVLA-OFT distinguish language-targeting errors from motor execution failures in their experiments. An action-side adapter cannot be assumed to repair missing semantic or perceptual information. [OpenVLA](https://openvla.github.io/); [OpenVLA-OFT](https://openvla-oft.github.io/).

### Inferences
- Admission sequence: reproduce published competence; confirm a passive valid violation monitor; document recurring violation with retained task competence; demonstrate a feasible safe route; qualify useful adapter authority without optimizer training; measure repeatability and actual episode throughput. Freeze thresholds and finite screen budget before seeing outcomes.
- Diagnose each failure as perception/grounding, task competence, safety preference, action authority, or model error. These categories can overlap, but the diagnosis determines whether a compact safety adapter is plausible.
- Privileged simulator-state labels are acceptable for evaluation and shared training feedback if disclosed. They must not silently become deployment observations for the adapter. Safe scripted witnesses establish feasibility, not necessarily learnability from the permitted observations.
- Choose one main problem family with distinct task and safety outcomes, plus one transfer setting testing the same adaptation recipe. Use new final cases after all development selection, including failed benchmark candidates, is complete.
- Existing Block Plucking infrastructure reduces setup work but does not settle which problem should be the main study. A VLA benchmark with a competent released policy, recurring monitored violations and a credible safety baseline could deserve the main role if its bounded admission screen supplies stronger useful-authority evidence. Compare the prospective scientific opportunity and cost rather than preserving the current main task solely because work has already been spent there.
- Avoid admitting only cases an evolved controller can already solve, or tightening hazard thresholds until a favored optimizer wins. Report every development screen and count its cost.
- The useful headline is adapting competent frozen controllers to safety specifications. The method should not claim general VLA semantic safety, real-world safety, or safety without explicit specification from two simulator studies.

### Gaps
- Admission thresholds, spending, candidate task counts and host allocations remain prospective decisions. This review authorizes no simulator preparation, trials, or resumed studies.
