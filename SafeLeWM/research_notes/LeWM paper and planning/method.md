# LeWM paper identity and method

## Which paper are we reviewing?

### Takeaway
The canonical work is *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*, by Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun and Randall Balestriero. Review arXiv v3, while preserving the original v1 linked by the official repository.

### Cited Findings
- The official code repository links arXiv `2603.19312v1`; the arXiv record lists v1 on 13 March 2026, v2 on 24 March and latest v3 on 3 June 2026. The record labels the paper a preprint, with no journal/conference acceptance established here. [Official repository](https://github.com/lucas-maes/le-wm), [version history](https://arxiv.org/abs/2603.19312).
- Downloaded, unmodified PDFs: `docs/papers/LeWorldModel-2603.19312v3.pdf` (28 pages, SHA256 `553a01501ac74ec56721cfdb936d3825ac941734133a1fd761f614c01b6887e9`) and `docs/papers/LeWorldModel-2603.19312v1.pdf` (29 pages, SHA256 `36b6dabbf3a37a4229063e76d82854b9e696411abfa0d4bf72325bdd67d64d85`). Adjacent `.provenance.json` files record source, bytes, date, authors and checksum. [v3 PDF](https://arxiv.org/pdf/2603.19312v3), [v1 PDF](https://arxiv.org/pdf/2603.19312v1).
- I extracted both PDFs with Poppler and visually inspected v3 page 1 (Fig. 1, training) and page 4 (Fig. 4, planning). V3 is the page-number reference throughout the remaining notes. [v3](https://arxiv.org/pdf/2603.19312v3).
- Material version differences: v1 Appendix D explicitly specifies a single NVIDIA L40S for training and planning; v3 omits that implementation/hardware paragraph. V3 adds Appendix G's planning-solver comparison, Table 10 on page 26. Treat L40S as an explicit v1 detail rather than silently attributing it to v3. [v1 Appendix D](https://arxiv.org/pdf/2603.19312v1), [v3 Appendix G](https://arxiv.org/pdf/2603.19312v3).

### Inferences
- Keep the versioned paper alongside the exact software/model revisions; the latest manuscript alone is not an executable reproduction specification.

### Gaps
- No exhaustive textual version diff was performed. The review is not a claim that all numerical results remained unchanged between revisions.

## What does LeWM learn, and why does it avoid collapse?

### Takeaway
LeWM learns a compact visual representation and how actions change that representation. Its central contribution is a simple jointly trained prediction objective with a distributional regularizer that prevents the encoder from solving prediction by mapping every image to the same vector.

### Cited Findings
- Training uses offline sequences of images and associated actions, with no reward labels or task specification. The data may come from exploratory or pseudo-expert behavior; useful coverage of dynamics is still required. Section 3.1, p. 4. [Paper](https://arxiv.org/pdf/2603.19312v3).
- The encoder produces `z_t = enc_theta(o_t)`. The action-conditioned predictor produces `z_hat_(t+1) = pred_phi(z_t, a_t)`. The compact notation hides a causal history window. Section 3.1, pp. 4-5. [Paper](https://arxiv.org/pdf/2603.19312v3).
- Default encoder: ViT-Tiny, about 5 million parameters, 12 layers, 3 attention heads, patch size 14, hidden dimension 192. It uses the final CLS token followed by a projector with BatchNorm. The projector avoids the incompatibility between the ViT's final LayerNorm and the desired Gaussian regularization. Section 3.1, p. 4. [Paper](https://arxiv.org/pdf/2603.19312v3).
- Default predictor: about 10 million parameters, six transformer layers, 16 attention heads and dropout 0.1. Actions enter through Adaptive LayerNorm with initially zero conditioning parameters. Causal masking prevents access to future observations. There is also a predictor output projector. Section 3.1, pp. 4-5. [Paper](https://arxiv.org/pdf/2603.19312v3).
- The loss is `L = ||z_hat_(t+1) - z_(t+1)||_2^2 + lambda * SIGReg(Z)`. Prediction alone admits constant embeddings: if every image is encoded as zero and every prediction is zero, loss becomes zero while useful state information disappears. SIGReg discourages this by matching the distribution of encoded observations to an isotropic Gaussian. It projects batch embeddings onto random directions and tests each one-dimensional distribution with an Epps-Pulley statistic. Section 3.1, Eqs. 1-3, p. 5; Fig. 1, p. 1; Appendix A, pp. 13-14. [Paper](https://arxiv.org/pdf/2603.19312v3).
- Defaults are 1,024 random projections and regularizer weight 0.1. The paper describes lambda as the only effective loss hyperparameter because projection-count ablations are insensitive. This is a statement about loss regularization, not a claim that batch size, learning rate, architecture, dropout and planner settings cease to exist. Gradients flow through both encoder branches; no target EMA or stop-gradient is used for training. Section 3.1, p. 5. [Paper](https://arxiv.org/pdf/2603.19312v3).
- The released `JEPA.encode` obtains the CLS token then applies the projector, and `predict` passes action embeddings through the predictor and output projector. This supports reading the projected representation, rather than an arbitrary raw transformer feature, as the dynamics latent. [Official model implementation](https://raw.githubusercontent.com/lucas-maes/le-wm/main/jepa.py).
- There is no pixel reconstruction term in the principal method. The paper's decoded images use a separate diagnostic decoder; they do not mean the deployed planner must generate images. Appendix D, p. 18. [Paper](https://arxiv.org/pdf/2603.19312v3).

### Inferences
- Intuitive explanation: prediction asks the representation to be useful for forecasting, while regularization asks it to remain varied. Neither term alone establishes that the representation preserves every safety-relevant detail.
- The Gaussian constraint concerns the distribution of embeddings across samples. It is not a probabilistic future-state model and does not supply calibrated uncertainty about collision risk.
- Anti-collapse is a representation-learning property. It does not imply correct physical predictions everywhere, successful planning or safe control.
- Reward-free training does not mean behavior-free training: demonstrations encode a particular distribution of states and actions.

### Gaps
- The paper's broad anti-collapse and hyperparameter-search claims should be described as scoped author claims. Finite random projections and minibatches approximate distribution matching; the review does not independently prove the optimization claim or logarithmic search complexity.

## How does it act, and what must our AP pipeline preserve?

### Takeaway
At deployment LeWM freezes its weights and repeatedly searches over candidate actions using predicted latent outcomes. The original planner targets the final goal directly; AP is an additional planning/retrieval mechanism and must have a separate configuration and evidence trail.

### Cited Findings
- Planning encodes the current and goal images, rolls out candidate action sequences in latent space, and minimizes terminal squared distance `C = ||z_hat_H - enc(o_goal)||_2^2`. CEM samples action sequences, keeps elites and refits its sampling distribution. It optimizes actions, not model weights. Section 3.2, pp. 5-6; Fig. 4, p. 4. [Paper](https://arxiv.org/pdf/2603.19312v3).
- Model errors accumulate in autoregressive long rollouts; the paper explicitly identifies a tradeoff between horizon, computational cost and model bias. MPC periodically obtains a new real observation and replans. Section 3.2, p. 6. [Paper](https://arxiv.org/pdf/2603.19312v3).
- Exact original recipe in Appendix D, pp. 17-18: 224-by-224 images; frame skip/action block 5; training batches of 128; sub-trajectories containing 4 frames and 4 five-action blocks. Predictor history is 3 for PushT and Cube, 1 for TwoRoom. Reacher history is not specified in that paragraph. [Paper](https://arxiv.org/pdf/2603.19312v3).
- Original CEM uses 300 candidate sequences, 30 elites, at most 30 optimizer iterations for PushT and 10 for the other environments, and initial sampling variance 1. It plans five action blocks, representing 25 environment actions. Appendix D says it executes the entire optimized sequence before replanning. The official PushT YAML agrees on `horizon: 5`, `receding_horizon: 5`, `action_block: 5`. This differs from the AP reproduction's five-action replanning. [Paper Appendix D](https://arxiv.org/pdf/2603.19312v3), [official PushT config](https://raw.githubusercontent.com/lucas-maes/le-wm/main/config/eval/pusht.yaml), [our AP protocol](../../PushTAPReproduction.md).
- Original code's `criterion` scores only the terminal predicted embedding against the goal embedding; its rollout appends predicted embeddings autoregressively. It does not add a hazard penalty in this implementation. [Official model](https://raw.githubusercontent.com/lucas-maes/le-wm/main/jepa.py).
- Appendix E, p. 18, describes world-model training for 10 epochs per environment: TwoRoom has 10,000 noisy-heuristic episodes averaging 92 steps; PushT is described as 20,000 expert episodes averaging 196 steps; Cube single-cube variant has 10,000 heuristic episodes of 200 steps; Reacher has 10,000 SAC-policy episodes of 200 steps. These are paper dataset descriptions, not a replacement for the actual downloaded HDF5 manifest. [Paper](https://arxiv.org/pdf/2603.19312v3).
- Original official PushT evaluator sets 50 evaluations, goal offset 25 and action budget 50. This is a short dataset-derived goal setup and differs from the AP reproduction's 128 queries, 140-step goal offset and 280-action cap. [Official PushT config](https://raw.githubusercontent.com/lucas-maes/le-wm/main/config/eval/pusht.yaml), [our AP protocol](../../PushTAPReproduction.md).
- V3 Table 10, p. 26, compares original PushT planning solvers: LeWM CEM 96.0 +/- 2.83%, SGD 26 +/- 4.32%, RMSProp 67.33 +/- 2.49%, Adam 84 +/- 7.12%. This is action-sequence optimization for control, not an evolutionary adapter-training comparison. [Paper](https://arxiv.org/pdf/2603.19312v3).
- The official repository provides separate pretrained checkpoint repositories for PushT, Cube, TwoRoom and Reacher, and builds environment management/planning/evaluation on stable-worldmodel. This is not one demonstrated universal checkpoint across the tasks. [Official repository](https://github.com/lucas-maes/le-wm).

### Inferences
- A compact model can still be computationally expensive in a control loop because CEM repeatedly scores hundreds of candidate trajectories. Model parameter count alone does not specify throughput.
- For our study, loading a released model and freezing it avoids all world-model pretraining. Any learned adapter is a separate outer learning process. Distinguish original model training, online CEM action search, AP retrieval, and adapter parameter training in both terminology and accounting.
- The main inherited objective is goal proximity, not safe completion. Our hazards and monitors add research content, and must be independently specified and validated.
- Useful walkthrough order: Fig. 1 and Section 3.1 for learning; Fig. 4 and Section 3.2 for control; Appendix D for units and implementation; then experiments/limitations. Keep LeWM performance numbers separate from AP results and our own adapter studies.

### Gaps
- The AP paper itself is outside this researcher's scope and is being reviewed separately. Its exact anchor selection rule should come from that source, not inference from the original LeWM paper.
- Paper descriptions alone cannot guarantee parity with our pinned AP checkpoint/runtime. Inspect exact revisions and freezes before any new run; this literature review authorizes no experiment or rental change.
