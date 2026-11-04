# VLA testing grounds

## Which released environment and checkpoint pairs should be considered first?

### Takeaway
My first VLA candidate is **SafeLIBERO with the released pi0.5-LIBERO checkpoint**, with AEGIS as an existing safety-filter reference. My second is **LIBERO-Safety with its released pi0.5 safety checkpoint**, initially on the physical obstacle-avoidance suites. These are two different benchmarks. They offer more concrete starting points than a generic request to test OpenVLA on an arbitrary new simulator.

### Cited Findings
- SafeLIBERO is released in the official AEGIS repository. It contains 16 LIBERO-derived tasks, two obstacle-interference scenarios per task, and 50 initial states per scenario. Obstacles either sit near the target or obstruct the travel path. The benchmark and AEGIS code are released; a quick-preview script still needs a real model integration and suitable horizon. [Official repository](https://github.com/THU-RCSCT/vlsa-aegis).
- The AEGIS paper uses pi0.5-LIBERO as its base policy. In full-action evaluations, it reports 57.8% task success and 17.1% collision avoidance for the base, compared with 67.5% success and 68.9% avoidance for AEGIS. These are separate rates, not joint safe completion. Collisions do not terminate episodes. The environment uses a Panda with Robosuite OSC_POSE at 20 Hz. The authors explicitly report that safety corrections can take the policy into unfamiliar states where it fails to recover. [Paper, Table I and Sections V-A/V-C](https://arxiv.org/html/2512.11891).
- OpenPI officially provides the pretrained inference checkpoint `gs://openpi-assets/checkpoints/pi05_libero`, a LIBERO example workflow, and action-chunk inference. The released policy API supports remote inference; a world-model rollout interface is not part of that contract. [OpenPI](https://github.com/Physical-Intelligence/openpi).
- The official LIBERO policy takes a third-person image, wrist image, language instruction and eight-dimensional proprioceptive state (end-effector position, axis-angle orientation and two gripper positions). It returns seven-coordinate actions. The standard example executes the first five actions before requesting another chunk. [Input/output transform](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/policies/libero_policy.py), [evaluation loop](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/examples/libero/main.py).
- The released full-action AEGIS script likewise defaults to five executed actions per query, filters individual commands, and flags collisions by obstacle displacement. It renders at 1024 before resizing policy images to 224, whereas the standard OpenPI LIBERO example renders at 256. Freeze the exact evaluator when reproducing Table I; do not silently substitute ordinary LIBERO settings. [Released AEGIS evaluator](https://raw.githubusercontent.com/THU-RCSCT/vlsa-aegis/main/main/main_aegis.py).
- LIBERO-Safety has five suites: affordance-aware grasping, human interaction, tabletop avoidance, free-space hand/object avoidance, and semantic reasoning; each has five tasks at three levels. Its repository links released code, assets, datasets and pi0.5 weights, while the data-generation pipeline remains announced as forthcoming. [Official repository](https://github.com/LIBERO-SAFETY/LIBERO-Safety).
- The pi0.5-LIBERO-Safety weight repository actually contains assets and parameter files, totaling 12.4 GB. This is a checkpoint already adapted to that safety benchmark, rather than the generic pi0.5 backbone. [Released weights](https://huggingface.co/LIBERO-Safety/pi05_libero_safety/tree/main).
- LIBERO-Safety's physical evaluation separates task performance and collision rate, with collision-induced terminations. Its tabletop suite progresses through static obstacles, moving obstacles and unfamiliar geometries. Its semantic suite instead evaluates refusal, a materially different outcome from completing a permitted task safely. [Paper](https://arxiv.org/html/2606.23686).

### Inferences
- SafeLIBERO is the strongest problem-first candidate: existing task competence, recurring safety failures, an intervention reference, and a published recovery limitation. Start with Spatial/Goal/Object rather than the long-horizon suite. Reproduce all relevant outcomes before assuming the published headroom exists on the chosen host.
- A concrete question is whether a compact learned controller can maintain goal progress through and after an avoidance manoeuvre with fewer task-specific recovery rules. AEGIS is both a useful reference and evidence that frozen-VLA adaptation itself is already established. The proposed evolutionary contribution still needs its own distinction.
- LIBERO-Safety is broader and useful for subsequent recipe-transfer testing, but avoid mixing refusal and physical completion in one primary endpoint. Existing safety training could reduce headroom. Preserve this checkpoint provenance when interpreting additional adaptation.
- Either pair requires only new adapter training in principle. The proposition that the shared evolutionary recipe works is untested; neither benchmark implies a small adapter has adequate authority.

### Gaps
- No simulator, model or policy was run. Actual host throughput, baseline parity, memory needs and adapter authority remain unmeasured.
- SafeLIBERO's README example flags obstacle displacement greater than 0.001 in summed coordinate magnitude; this is a displacement proxy, not a complete contact detector. Check its exact relation to the paper metric before reproduction. Preserve the published monitor and, if needed, add a separately declared contact audit rather than silently redefining it. [Collision-check code example](https://github.com/THU-RCSCT/vlsa-aegis#-automated-collision-check).
- AEGIS's geometry/perception inputs differ from a camera-only VLA. Its implementation uses a VLM and depth-based obstacle geometry. A fair comparison must declare information access and separate monitor-only labels from learner observations. Its end-effector protection does not cover every robot link. [Paper, Sections IV and V-C](https://arxiv.org/html/2512.11891).
- Current release presence was verified, but byte-level checkpoint integrity and source-pinned restoration were not. Baseline reproduction is a prerequisite, not an assumption.
- The project landing page displays 18.1% for full-action base collision avoidance, while the paper's Table I gives 17.1%; use the paper value with that discrepancy recorded. Neither source establishes joint safe completion from the two marginal rates. [Landing page](https://vlsa-aegis.github.io/), [Table I](https://arxiv.org/html/2512.11891).

## What other VLA options are useful, and why are they secondary?

### Takeaway
Ordinary LIBERO with OpenVLA-OFT is a strong competence baseline but needs an independently specified safety problem. SafeManip/RoboCasa offers richer temporal hazards but has a concrete release-completeness concern. Safety-CHORES is a legitimate alternative for navigation/manipulation but expands the engineering and behavioural scope.

### Cited Findings
- OpenVLA-OFT reports 97.1% mean success on standard LIBERO and explicitly releases checkpoints. Its architecture uses continuous action chunks. The official instructions list suite-specific and combined LIBERO checkpoints and evaluation commands; ordinary OpenVLA pretraining should not be assumed to provide equivalent LIBERO competence. [Project](https://openvla-oft.github.io/), [checkpoint and evaluation instructions](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md).
- SafeVLA-Bench is distinct from SafeVLA and Safety-CHORES. Its project evaluates existing checkpoints on LIBERO and RoboCasa with safety properties covering excessive contact, bystander disturbance, handling instability and self-contact. Its reported OpenVLA-OFT LIBERO result is 97.8% success with 18.1% successful-but-unsafe episodes. The displayed safety percentage is not safe-completion percentage. [Project](https://safevla.org/).
- SafeManip provides temporal safety monitors over privileged simulator states, with RoboCasa policy launchers and LTL/DFA logic. Its README explicitly says needed working policy/simulator forks are gitignored and old installation paths need reverification. Thus visible monitor code does not establish a complete reproducible evaluation stack. [Repository](https://github.com/chengyuehuang511/SafeManip).
- Safety-CHORES belongs to the SafeVLA constrained-learning project, with mobile navigation, pickup and fetch tasks. Official code provides environment setup, imitation-checkpoint and aligned-checkpoint download scripts, and evaluation examples. Its asset stack includes ProcTHOR houses and Objaverse objects. [SafeVLA project](https://safevla.github.io/), [code](https://github.com/safevla/safevla).

### Inferences
- Standard LIBERO plus a frozen OFT checkpoint is a good fallback if safety-monitor release validation succeeds. A new custom obstacle extension is a new benchmark specification, not automatic reproduction of standard LIBERO.
- SafeManip could eventually test release stability, containment or contact history, which are closer to meaningful temporal safety than a drawn keep-out region. Its current documented integration gaps make it a risky first expansion under a January submission target.
- Safety-CHORES is better when the scientific question concerns mobile task execution and cumulative collateral damage. It is a larger pivot from the present tabletop-control interface. Do not conflate its cumulative costs with episode-level probability of any violation.
- For this project, prefer one new VLA family after qualification rather than adding all these platforms. ManiGuard remains deferred under current repository authority; this search does not reopen it.

### Gaps
- A release of SafeVLA-Bench's exact official monitoring code was not verified from its project page. Search results include unrelated similarly named repositories, so do not assume they implement the cited paper.
- OpenVLA-OFT's strong native LIBERO performance does not imply strong SafeLIBERO or LIBERO-Safety performance. Indeed, AEGIS reports a much lower SafeLIBERO success rate for OFT. Cross-paper numbers are screening leads, not comparable local measurements.

## Can a common evolutionary recipe extend from world models to VLAs?

### Takeaway
Yes in principle, at the level of small-adapter learning around a frozen action-producing system. A candidate-future scorer specifically requiring a queryable world model is not automatically portable to a standard VLA.

### Cited Findings
- OpenPI exposes observations and prompts to an inference API returning action chunks, supporting a wrapper or action-head intervention. [Official inference API](https://github.com/Physical-Intelligence/openpi).
- AEGIS already modifies nominal VLA actions while preserving the pretrained model, providing a direct precedent for external adaptation of frozen VLA behaviour. [Project](https://vlsa-aegis.github.io/).

### Inferences
- A portable contract could expose observation features, proprioception, nominal actions or chunks, history, and a compact learned correction/selection mechanism. World-model predictions could be optional extra information, but that changes the claim and must be ablated or declared.
- Prefer a limited parameter space over evolving the full VLA or a large unrestricted LoRA update. Exact dimensionality should follow qualification rather than a nominal parameter target.
- VLA token features are not automatically predicted physical futures. Sampling several action chunks supplies candidates but does not supply a model capable of assessing their consequences. Adding a learned predictor introduces another training and information component.
- Ground truth geometry can label simulator outcomes without being given to the adapter. If supplied online, share it with all learning arms and state that the result concerns adaptation with that information.
- Problem-first selection should freeze eligibility rules before optimizer results: competent native behaviour, repeatable avoidable violation, demonstrated feasible safe completion, usable observations and action authority, and affordable reproducibility. Qualify on development cases, preserve rejected cases, reserve new final cases, and do not select benchmarks by where CMA happens to win.

### Gaps
- No evidence yet shows one frozen learning recipe transfers from the current TwoRoom/Block implementation to a VLA. This is a potential extension requiring new prospective work, not a current generality claim.
