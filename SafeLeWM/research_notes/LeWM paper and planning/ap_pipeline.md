# Anchored Planning and the LeWM pipeline

## What does AP add to LeWM?

### Takeaway
LeWM learns an encoder and action-conditioned latent dynamics. AP is a separate, training-free controller that changes the intermediate target used by that frozen model. It does not retrain LeWM or turn the world model itself into a safety model.

### Cited Findings
- Companion paper: Liu, Fang, Wu, Gao and Li, *Aim Short to Reach Far: Your Frozen World Model Can Plan Better Than You Think*, arXiv:2609.30036v4, 30 September 2026. Downloaded unchanged to `docs/papers/AnchoredPlanning-2609.30036v4.pdf`, 25 pages, with adjacent provenance JSON. Figure 2 on PDF page 4 was rendered and visually inspected. This is a different paper from Maes et al.'s original LeWorldModel. [Versioned AP paper](https://arxiv.org/abs/2609.30036v4)
- At a live state, the planner scores a candidate block `u` by `||F(E(o_t),u) - q||²`. Final-goal planning chooses `q=E(o_goal)`. AP instead retrieves a route whose start resembles the live image and whose distant endpoint resembles the goal, then uses an observed image five actions after the retrieved start as `q`. Predictions always begin from the live state, not from the recorded state. [AP sections 2 and 3, pp. 3-5](https://arxiv.org/html/2609.30036v4#S3)
- Retrieval matches concatenated source, distant endpoint and displacement features: query `(z_t,z_goal,z_goal-z_t)` versus record `(z_s,z_(s+h),z_(s+h)-z_s)`, using coordinate-wise standardized squared distance. Eligible source and successor rows must remain inside one episode. `h=max(5,H-t)`, where `H` is the supplied recorded start-to-goal action offset and `t` counts executed actions. This is not a learned time-to-go estimator. The model prediction and anchor offset stay at five actions. [AP equations 4-6, Figure 2, p. 4; appendix A.6](https://arxiv.org/html/2609.30036v4#S3.SS1)
- AP-CEM generates five-action blocks from a fixed Gaussian prior, using 300 candidates, 30 iterations and 30 elites; it does not initialize from recorded actions. AP-rank instead predicts the outcomes of eight retrieved recorded blocks and ranks them against the same anchor. Observation-only refers to AP-CEM's deployment memory, not to LeWM pretraining, which used actions. [AP sections 3.2-3.3; appendix A.3 and A.5](https://arxiv.org/html/2609.30036v4#A1.SS3)

### Inferences
- A useful explanation is: LeWM supplies a local predictive model; AP supplies an experience-based choice of where to aim next; CEM supplies the action search. These are separable components with separable failure modes.
- Better prediction alone cannot solve every planner failure. A short-horizon objective can reject the beginning of a necessary detour even with perfect dynamics. The AP paper proves an existence counterexample; it does not prove arbitrary retrieved anchors are reachable or safe.

### Gaps
- This review did not prove that today's upstream `main` matches our pinned AP revision. All new packaging must retain the recorded commit rather than replacing it silently.

## What evidence matters for interpreting our results?

### Takeaway
AP is strong evidence that the planning objective matters, but its benchmark successes do not establish safety or unfamiliar-scene generalization. Original LeWM short-goal results and AP long-goal results must remain separate.

### Cited Findings
- AP Table 1 evaluates 128 query episodes per task. On standard PushT starts, AP-CEM achieves 69.5% (89/128), versus 2.3% for matched one-block final-goal CEM and 7.0% for the LeWM planner under AP's long-goal protocol. Observed targets are not uniformly best: the learned target performs better on Cube and TwoRoom, while AP has the higher four-task mean. [AP Table 1 and Table 2, pp. 5-7](https://arxiv.org/html/2609.30036v4#S4.SS1)
- AP's PushT goal offset is 140 actions with a 280-action allowance; TwoRoom is 100/200 and Cube 150/150. Its query episodes are excluded from retrieval. PushT queries come from the source training split, so this exclusion does not establish that they were unseen during pretrained world-model learning. [AP appendix A.1, Table 5](https://arxiv.org/html/2609.30036v4#A1.SS1)
- Original LeWM evaluation uses a goal image 24 actions after a dataset start and a 50-action allowance. AP's appendix distinguishes this from its own query construction, rendered start images, restored state information and longer goals. It also flags the released 30 CEM iterations versus ten reported outside PushT. The comparisons are useful but cannot be collapsed into a single success rate. [AP appendix D.2 and Table 10](https://arxiv.org/html/2609.30036v4#A4.SS2)
- Our matched AP reproduction gave authors' evaluator 88/128 and wrapper 89/128, with all 256 episodes audited. That establishes useful baseline agreement, not bitwise equivalence or safety. [Local account 15](../../results/15-pusht-ap-reproduction.md)
- Our TwoRoom target-only comparison was 20/50 AP completions versus 0/50 final-goal completions; this was a development panel. Cube's 26/50 no-hazard success masked weak rest-state grasping: 2/22 successes for early starts and no completion in the 16 hazard episodes. [Local account 5](../../results/05-anchored-planning.md), [Local account 11](../../results/11-cube-screen.md)
- Validated PushT qualification produced 12/16 native completions but violations in all 16. The tested adapter heads failed useful authority; no training followed. That outcome remains closed. [Local account 16](../../results/16-pusht-validated.md)

### Inferences
- The defensible safety question is whether an adapter can preserve useful task progress while reducing declared violations, not whether it can merely alter actions or reduce movement.
- Conceptual insertion points are target selection, candidate scoring and action modification. They require different information and authority checks. Their existence in this architecture does not authorize replacing the failed PushT interface screen with another experiment.

### Gaps
- Neither AP task success nor latent distance provides calibrated hazard probability. The paper does not supply a safety guarantee, arbitrary-goal horizon estimator or evidence that an unseen hazard is represented usefully.

## What must the reusable compute pipeline preserve?

### Takeaway
The model weights are only one dependency. Runtime, renderer, image transforms, action mappings, memory membership and query construction jointly define the baseline.

### Cited Findings
- Upstream AP documents Linux, Python 3.11.14, PyTorch 2.7.1/CUDA 12.6, torchvision 0.22.1 and stable-worldmodel 0.0.6. OSMesa/llvmpipe is the recorded renderer; Reacher needs a different MuJoCo/DM-Control pair. These direct dependency pins are explicitly not a fully tested OS/transitive lock. [AP runtime guide, accessed 4 November 2026](https://github.com/daybraeklaxry/Aim-Short-to-Reach-Far/blob/main/code/ap/README.md)
- Our validated reproduction used AP commit `984327424ca7a2e52c75a8279197750d9d0b7c8e`, Python 3.11.9 and eight RTX 4090s instead of the paper's H20s. Retrieval encoding used bfloat16 with float16 cache storage and full 16,688-episode PushT memory. Its initial missing-OSMesa failure is recorded. [Local account 15](../../results/15-pusht-ap-reproduction.md)
- Each AP-CEM decision entails 31 prediction batches and 9,002 predicted five-action blocks; ranking entails eight predicted blocks. Cache encoding, retrieval and simulator work contribute additional costs. Original dataset row order and episode boundaries are required by frozen protocols. [AP appendix A.3-A.4](https://arxiv.org/html/2609.30036v4#A1.SS3), [AP asset guide](https://github.com/daybraeklaxry/Aim-Short-to-Reach-Far/blob/main/code/ap/README.md)

### Inferences
- Package separately: immutable model/data provenance; pinned runtime and renderer; reproducible cache and eligibility index; controller/query configuration; trace and resource accounting. This gives a reviewable distinction between model, retrieval, search and execution failures.
- The earlier chat's single-4090/24-GB, 32-64-GB RAM and 150-GB disk proposal was an engineering estimate, not a paper-established minimum. The paper's small model does not imply cheap full planning sweeps because CEM repeatedly evaluates thousands of blocks. Measure representative memory and throughput on an authorized rental before treating those estimates as a specification.

### Gaps
- No new host was inspected and no compute, cache construction, simulator preparation or tests were run. Peak memory, fresh-install completeness and single-GPU throughput remain unmeasured in this review.
