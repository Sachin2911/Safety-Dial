# Inference-time optimisation and steering of frozen diffusion / flow-matching robot policies and VLAs (literature as of October 2026)

Scope: methods that change what a frozen diffusion or flow policy outputs without changing its weights (best-of-N with a verifier, guidance inside the sampler, noise/latent manipulation, test-time adaptation), plus RL fine-tuning only as a headroom reference, plus pi0 / pi0.5 sampler facts. Every number is the paper's own; sim and real-robot results are labelled. Where the literature is silent, that is stated under Gaps. No experiments are proposed.

Note on sourcing: the web-search budget for this session ran out partway through; the second half of the research was done by fetching primary sources directly from known arXiv IDs and GitHub. Items I could not reach are listed as gaps rather than filled in from memory.

## Key question 1: Best-of-N and verifier-guided sampling (V-GPS, RoboMonkey, ITPS, DynaGuide, MG-Select, DiVeR, SDN, test-time scaling for VLAs)

### Takeaway
Best-of-N with a learned verifier is the most-replicated inference-time effect on VLAs: gains of roughly +3 to +13 absolute points in simulation and +10 to +25 points on real robots, with N between 4 and 50, and the gain is almost entirely attributable to the verifier (random selection among N candidates gives ~+1 point). Every published verifier that beat random selection was trained (offline-RL Q-function, VLM reward model, or a small MLP on policy rollouts); the only training-free selectors that worked used either the policy's own token distribution (not available on a flow head) or external perception models plus a jerk rule. No paper in this family measured a safety or constraint-violation metric; one (Temporal Logic Guidance, under KQ2) showed sample-and-rank left an upright-object violation rate unchanged.

### Cited Findings

**V-GPS (Nakamoto, Mees, Kumar, Levine, "Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance", CoRL 2024, arXiv:2410.13816)**
- Mechanism: sample K actions from the frozen policy, score each with a Q-function, pick by softmax over Q/beta or greedy; "requires no fine-tuning or access to the policy's weights" — [arXiv 2410.13816](https://arxiv.org/abs/2410.13816)
- Verifier is a trained critic: Cal-QL (IQL as alternative) on Bridge V2 (~45K demonstrations) plus Fractal (~130K episodes), ResNet-34 image encoder with FiLM language conditioning, ~25.6M parameters, reward = 0 on the last 3 steps of each demo and -1 otherwise, discount 0.98, 1M gradient steps on a single TPU v4-8 — [arXiv 2410.13816 (HTML)](https://arxiv.org/html/2410.13816v2)
- N used: real robot K=50 greedy; simulation swept K in {10, 50} and beta in {0, 0.1, 1.0}, reporting the best per policy — [arXiv 2410.13816 (HTML)](https://arxiv.org/html/2410.13816v2)
- Simulation (SIMPLER, 6 tasks) average success base -> V-GPS: Octo-small 0.27 -> 0.34; Octo-base 0.17 -> 0.18; Octo-small-1.5 0.02 -> 0.18; RT-1-X 0.12 -> 0.14; OpenVLA 0.24 -> 0.27. Several per-task cells declined (e.g. Octo-small spoon-on-towel 0.52 -> 0.46; OpenVLA carrot 0.06 -> 0.04) — [arXiv 2410.13816 (HTML)](https://arxiv.org/html/2410.13816v2)
- Real robot (WidowX, Octo-small-1.5, 6 tasks, 20 trials each): average 0.24 -> 0.44; green pepper in pot 0.15 -> 0.35, mushroom on cloth 0.35 -> 0.70 — [arXiv 2410.13816 (HTML)](https://arxiv.org/html/2410.13816v2)
- Ablation on K (Cal-QL, SIMPLER eggplant): 0.77 at K=10, 0.84 at K=50, 0.63 at K=100 (more candidates hurt); IQL best at K=10 and degrades at larger K — [arXiv 2410.13816 (HTML)](https://arxiv.org/html/2410.13816v2)
- Failure types it fixed: "imprecise grasping of slippery objects, premature release, and late release". Safety is not discussed. Stated limitation: the base policy must sample stochastically — [arXiv 2410.13816 (HTML)](https://arxiv.org/html/2410.13816v2)
- Latency (Octo-small): 0.075 s/step baseline, 0.120 s at K=50, 0.160 s at K=100; OpenVLA had no batched inference so K forward passes were run per step — [arXiv 2410.13816 (HTML)](https://arxiv.org/html/2410.13816v2)

**RoboMonkey (Kwok, Agia, Sinha, Foutter, Li, Stoica, Mirhoseini, Pavone, "Scaling Test-Time Sampling and Verification for Vision-Language-Action Models", CoRL 2025, arXiv:2506.17811)**
- Verifier is a trained VLM reward model: LLaVA-7B with the unembedding layer replaced by a reward head, LoRA rank 512, trained one epoch on 20 million synthetic preference pairs generated from Bridge V2 with a Bradley-Terry loss, on 8 H100s — [arXiv 2506.17811 (HTML)](https://arxiv.org/html/2506.17811v2); [CoRL 2025 proceedings](https://proceedings.mlr.press/v305/kwok25a.html)
- Sampling: 5 OpenVLA samples at temperature 1.0, fit a Gaussian to translation/rotation and majority-vote the gripper, draw 16 candidates from that proposal, score with the verifier; simulation swept initial samples in {5, 9} and candidates in {8, 16, 32} reporting the best per task — [arXiv 2506.17811 (HTML)](https://arxiv.org/html/2506.17811v2)
- Scaling law: action RMSE vs number of samples follows a power law; with an oracle verifier OpenVLA RMSE drops 59.3% at 10,000 samples; at 64 candidates RoboMonkey cuts error 21% vs greedy where V-GPS cuts it 6% — [arXiv 2506.17811 (HTML)](https://arxiv.org/html/2506.17811v2)
- Simulation (SIMPLER, in-distribution): RoboMonkey 47.5% average vs about 38.5% for OpenVLA (+9); eggplant task +19 points — [arXiv 2506.17811 (HTML)](https://arxiv.org/html/2506.17811v2)
- Real robot (WidowX-250S, 4 out-of-distribution tasks, 10 trials each): RoboMonkey 60%, OpenVLA 35%, V-GPS 30% — [arXiv 2506.17811 (HTML)](https://arxiv.org/html/2506.17811v2)
- LIBERO-Long (500 trials): OpenVLA 49.8% -> 56.5% — [arXiv 2506.17811 (HTML)](https://arxiv.org/html/2506.17811v2)
- Latency: about 650 ms per step (about 1.5 Hz) for 16 candidates on one H100, 28 GB GPU memory — [arXiv 2506.17811 (HTML)](https://arxiv.org/html/2506.17811v2)
- No safety metric is reported; collisions with surrounding objects are mentioned qualitatively as a failure mode that verification mitigated; contact-rich tasks are not discussed — [arXiv 2506.17811 (HTML)](https://arxiv.org/html/2506.17811v2)

**Inference-Time Policy Steering (Wang, Wang, Du, Sundaralingam, Yang, Chao, Perez-D'Arpino, Fox, Shah, "Inference-Time Policy Steering through Human Interactions", arXiv:2411.16627, MIT CSAIL / NVIDIA, v2 March 2025)**
- Six strategies on a frozen Diffusion Policy: Random Sampling (RS, baseline), Output Perturbation (OP), Post-hoc Ranking (PR = best-of-N by a hand-designed distance), Biased Initialization (BI = start denoising from a noised version of the target), Guided Diffusion (GD = add objective gradient at each denoising step), Stochastic Sampling (SS = annealed MCMC with multiple reverse steps per noise level). No critic is trained; objectives are hand-designed distances/energies — [arXiv 2411.16627 (HTML)](https://arxiv.org/html/2411.16627v2)
- Maze2D (DP): collision rate RS 0.01, PR 0.01, OP 0.02, BI 0.06, GD 0.06, SS 0.01; alignment (avg L2 to sketch) RS 0.28 -> SS 0.12. For ACT, OP raised collisions from 0.16 to 0.35 — [arXiv 2411.16627 (HTML)](https://arxiv.org/html/2411.16627v2)
- Block stacking (Isaac Sim, DP): PR reached 33% task alignment at 100% constraint satisfaction; GD at beta=100 reached 86% alignment but 15% constraint satisfaction — [arXiv 2411.16627 (HTML)](https://arxiv.org/html/2411.16627v2)
- Real kitchen (DP): RS alignment 38% / constraint satisfaction 90% / aligned success 34%; SS 71% / 73% / 55%; OP 89% / 37% / 30%. The paper states OP's physical corrections "often lead to execution failures" — [arXiv 2411.16627 (HTML)](https://arxiv.org/html/2411.16627v2)
- Headline: "stochastic sampling with diffusion policy achieves the best trade-off between alignment and distribution shift" — [arXiv 2411.16627](https://arxiv.org/abs/2411.16627)

**DynaGuide (Du, Song, "Steering Diffusion Policies with Active Dynamic Guidance", NeurIPS 2025, arXiv:2506.13922)**
- Mechanism: gradient of a learned dynamics model (predicts far-future DinoV2 patch latents from observation + action chunk) added to the noise prediction inside DDIM; guidance strength s = 1.0 to 2.0, sigma 15 to 40 per task, stochastic sampling repeats M=4 — [arXiv 2506.13922 (HTML)](https://arxiv.org/html/2506.13922v2)
- Dynamics model: ~16M-parameter 6-layer transformer encoder, trained 600k gradient steps on the full CALVIN ABCD split (sim) or 500 new demonstrations plus UMI data (real) — [arXiv 2506.13922 (HTML)](https://arxiv.org/html/2506.13922v2)
- Sample-and-rank baseline (GBC) used N=5 samples per inference and was weaker than guidance in MovableObjects, MultiObjectives and low-data settings; exact numbers are in figures only — [arXiv 2506.13922 (HTML)](https://arxiv.org/html/2506.13922v2)
- Results: CALVIN ArticulatedParts 8.7x boost over base to 70% average target success; underspecified objectives 5.4x over goal conditioning; real robot CupPreference 72.5%, HiddenCup 80% (no confidence intervals) — [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/117515); [arXiv 2506.13922 (HTML)](https://arxiv.org/html/2506.13922v2)
- The objective is behaviour selection (which object/behaviour), not success or safety; the paper notes high guidance strength "can produce incoherent trajectories" and does not discuss contact — [arXiv 2506.13922 (HTML)](https://arxiv.org/html/2506.13922v2)

**MG-Select (Jang, Kim, Kim, Kim, Shin, "Verifier-free Test-Time Sampling for Vision-Language-Action Models", arXiv:2510.05681; reported as ICLR 2026 in search metadata, venue not stated on the arXiv page)**
- Mechanism: sample N=4 candidates at temperature 0.5 and pick the one with the largest KL divergence between the VLA's predicted token distribution and a reference distribution from the same VLA with text/state masked, summed over the first 5 FAST action tokens — [arXiv 2510.05681 (HTML)](https://arxiv.org/html/2510.05681)
- Not fully training-free: the VLA is trained with 10% dropout of text, state and both so the masked reference is meaningful — [arXiv 2510.05681 (HTML)](https://arxiv.org/html/2510.05681)
- Applies to token-output VLAs (pi0-FAST, OpenVLA, MiniVLA); it needs a token distribution, which a flow-matching action head does not produce — [arXiv 2510.05681 (HTML)](https://arxiv.org/html/2510.05681)
- Results (base -> MG-Select): RoboCasa pick-and-place 100 demos 17.0 -> 31.0, all 24 tasks 40.2 -> 48.1; SIMPLER-WidowX 46.9 -> 50.3; LIBERO pi0-FAST 92.0 -> 93.1, OpenVLA 69.1 -> 71.7; real world in-distribution 37.5 -> 47.9, out-of-distribution 53.1 -> 71.9 — [arXiv 2510.05681 (HTML)](https://arxiv.org/html/2510.05681)
- N ablation (RoboCasa PnP / all): N=1 27.6 / 43.8; N=2 30.0 / 46.2; N=4 31.0 / 48.1; N=8 30.0 / 46.9; N=64 33.3 / 48.4 — gains saturate at N=4 — [arXiv 2510.05681 (HTML)](https://arxiv.org/html/2510.05681)
- Versus RoboMonkey on SIMPLER at N=4: RoboMonkey 50.7% at 1.94x base latency, MG-Select 50.3% at 1.43x — [arXiv 2510.05681 (HTML)](https://arxiv.org/html/2510.05681)
- Where it helps: "grasping and releasing moments and harder-to-grasp objects"; simple atomic tasks "can see slight drops from added stochasticity" — [arXiv 2510.05681 (HTML)](https://arxiv.org/html/2510.05681)

**DiVeR (Park et al., Microsoft Research Asia, "Decision-Critical Verifier Learning for VLA Test-Time Scaling", arXiv:2610.04933, 4 Oct 2026)**
- Verifier: a two-layer MLP (64 hidden units) on the frozen action expert's internal representation, no image input; trained on offline rollouts of the frozen VLA where every timestep inherits its episode's binary success label; timesteps where K=32 sampled candidates disagree most (high representation variance) are up-weighted — [arXiv 2610.04933 (HTML)](https://arxiv.org/html/2610.04933v1); [Microsoft Research summary](https://www.microsoft.com/en-us/research/articles/diver-decision-critical-verifier-learning-for-vla-test-time-scaling/)
- Rollout budget to train the verifier (success/failure episodes): pi0.5 LIBERO 281/19, RoboCasa 248/292, real robot 94/66; pi0 LIBERO 249/51, RoboCasa 184/356 — i.e. 160 to 540 episodes — [arXiv 2610.04933 (HTML)](https://arxiv.org/html/2610.04933v1)
- Base policies: pi0 and pi0.5 (flow heads); N=16 default, N swept 4 to 32 — [arXiv 2610.04933 (HTML)](https://arxiv.org/html/2610.04933v1)
- LIBERO-Long (N=32): pi0 83.5% -> 89.3%; pi0.5 93.5% -> 97.5% — [arXiv 2610.04933 (HTML)](https://arxiv.org/html/2610.04933v1)
- RoboCasa (N=16): pi0 32.2% -> 38.1%; pi0.5 43.1% -> 54.7%. Ablation (pi0.5): random selection among the 16 candidates 44.1% (+1.0), verifier trained without decision-critical weighting 50.6%, full DiVeR 54.7% — [arXiv 2610.04933 (HTML)](https://arxiv.org/html/2610.04933v1)
- Real Franka Research 3 (pi0.5 LoRA-fine-tuned on 40 demos/task, 4 tasks, N=16): base 58.3% -> uniform verifier 65.6% -> DiVeR 71.9% — [arXiv 2610.04933 (HTML)](https://arxiv.org/html/2610.04933v1)
- Versus a VLM verifier ("RoboMeter", Liang et al. 2026): 50.1% at 0.743 s vs DiVeR 54.7% at 0.001 s per decision — [arXiv 2610.04933 (HTML)](https://arxiv.org/html/2610.04933v1)
- Latency: sampling 16 candidates ~205 ms (pi0.5), ~360 ms (pi0); verifier < 0.5 ms per chunk — [arXiv 2610.04933 (HTML)](https://arxiv.org/html/2610.04933v1)
- Stated limitation: performance "ultimately depends on the quality and diversity of the available action candidates"; safety and contact failures are not discussed, though the qualitative drawer example describes a candidate whose gripper "moves through free space without making effective contact" being rejected — [arXiv 2610.04933 (HTML)](https://arxiv.org/html/2610.04933v1)

**Selected Diffusion Noise / SDN (Nguyen, Dao et al., "Self-Improving VLA Policies: Selected Diffusion Noise for Spurious-Robust Action Smoothing", arXiv:2606.14084, June 2026)**
- Training-free two-stage selection over N=12 noise seeds: stage 1 keeps candidates whose action is most changed when the target object is masked out (Grounding-DINO + SAM2 masks, kNN distances, keep top M in {3,5}); stage 2 executes the candidate with the lowest JerkRMS (RMS third finite difference over the chunk) — [arXiv 2606.14084 (HTML)](https://arxiv.org/html/2606.14084v1)
- Simulation (SimplerEnv, 50 rollouts/task): pi0 63.9% -> 72.6%; GR00T N1.6 49.5% -> 54.5% (Open Drawer 4.0% -> 14.0%) — [arXiv 2606.14084 (HTML)](https://arxiv.org/html/2606.14084v1)
- Real (ALOHA, GR00T N1.5, 30 trials/task): 30.0% -> 48.3% (abstract says +10%, body says +18.3 points); measured jerk fell (12.45 -> 11.80, 7.07 -> 6.54) — [arXiv 2606.14084 (HTML)](https://arxiv.org/html/2606.14084v1)
- ~245 ms per step; no random best-of-N baseline and no jerk-only ablation reported — [arXiv 2606.14084 (HTML)](https://arxiv.org/html/2606.14084v1)

**Genetic denoising (Clemente et al., Huawei, "Two-Steps Diffusion Policy for Robotic Manipulation via Genetic Denoising", arXiv:2510.21991, Oct 2025)**
- Population of partially denoised samples, fitness = Stein-style score from the noise-estimate norm, multinomial selection each step; the decisive change was reducing the per-step noise-injection scale gamma to 0.2 — [arXiv 2510.21991 (HTML)](https://arxiv.org/html/2510.21991)
- Adroit (100 seeds): 100-step DDPM Hammer 0.68 / Relocate 0.69 / Door 0.87 vs 2-step genetic 1.00 / 0.98 / 1.00; plain 2-step DDPM 0.00 / 0.01 / 0.01. Robomimic (500 seeds): gains "within two standard deviations" of the adjusted-schedule baseline; Transport 0.77 and ToolHang 0.49 did not beat 100-step DDPM (0.84 / 0.53) — [arXiv 2510.21991 (HTML)](https://arxiv.org/html/2510.21991)
- Overhead: population 16 is 1.2x a single DDPM step on an RTX 3080 — [arXiv 2510.21991 (HTML)](https://arxiv.org/html/2510.21991)

### Inferences
- The verifier, not the sampling, carries the effect: DiVeR's random-selection control (+1.0 point at N=16) and MG-Select's N=1 vs greedy numbers show that drawing more candidates without a good scorer does almost nothing. This matches the user's observation that the noise draw does not decide outcomes.
- Cheapest verifier with a measured effect on a flow-head VLA: DiVeR's 2-layer MLP on frozen pi0.5 action-expert features, trained on 160 to 540 of the policy's own episodes with binary labels. That is within a 1,000 to 2,000-episode budget and needs read-only access to internals (a forward hook), not a patch to the sampler.
- All best-of-N gains were on success; none reported a safety metric. The failure types verifiers fixed (imprecise grasp of slippery objects, premature/late release, moving through free space without contact) are pre-contact or at-contact choices of the approach, which is the lever a chunk-level selector has over a contact-decided outcome. Once contact has happened, a selector that only sees the next chunk cannot undo the physics.
- Saturation at small N is common (MG-Select N=4; V-GPS degrades past K=50 with Cal-QL), so the N cost is modest.
- MG-Select cannot be transplanted to pi0.5's flow head (no token distribution), and its training-time dropout requirement makes it not strictly frozen-policy anyway.

### Gaps
- No best-of-N paper reports constraint-violation, drop, or tip-over rates; safety-as-verifier-target is unstudied in this family (the only safety-relevant selection result is the Sample & Rank row in Temporal Logic Guidance under KQ2, where ranking did not reduce violations).
- RoboMeter (Liang et al. 2026, the VLM verifier DiVeR compares against) could not be retrieved; only DiVeR's citation of its numbers is available.
- DynaGuide's GBC baseline numbers are in figures only; the N=5 sample-and-rank result cannot be quoted precisely.
- SDN does not ablate the jerk rule alone, so whether a cheap smoothness rule by itself selects better chunks is unknown.

## Key question 2: Guidance inside the sampler (SafeDiffuser, constraint and world-model guidance, Latent Policy Barrier, PPGuide, Feedback World Model, physics-grounded optimisation)

### Takeaway
Gradient guidance inside denoising is the only inference-time family with a measured effect on a physical-safety constraint of the user's type: Temporal Logic Guidance cut "keep the can upright" violations on Robomimic Can Transport from 84% to 4% at unchanged 100% success, while best-of-N ranking left violations at 82%. Every working instance needs a differentiable model of the consequence (a learned world/dynamics model trained on the policy's data, an analytic barrier on the planned state, or a classifier of failure-relevant chunks) and access to the denoising loop; typical costs are 1.2x to 15x per-step latency and 150 to 600 rollouts plus a model training run.

### Cited Findings

**Temporal Logic Guidance for Action-Only Diffusion Policies with World Models (Zoellner, Manganaris, Paleja, ICRA 2026 workshop, arXiv:2606.22729, 3 pages)**
- Constraint: keep the can upright, STL specification G(R_zz > cos 5 degrees); the gradient of STL robustness computed on world-model-predicted future states is added to each denoising update (scale lambda unreported), plus a few gradient-ascent steps on the final action sequence — [arXiv 2606.22729 (HTML)](https://arxiv.org/html/2606.22729)
- World model: separately trained differentiable next-state predictor, trained on the same data as the diffusion policy, applied iteratively over the action horizon; architecture and size unreported — [arXiv 2606.22729 (HTML)](https://arxiv.org/html/2606.22729)
- Robomimic Can Transport, 50 rollouts: base policy 100% success / 84% violation / 8.51 degree average tilt; Sample & Rank 98% / 82% / 8.42 degrees; guidance 100% / 4% / 1.93 degrees — [arXiv 2606.22729 (HTML)](https://arxiv.org/html/2606.22729)
- Compute overhead and denoising-step count are not reported — [arXiv 2606.22729 (HTML)](https://arxiv.org/html/2606.22729)

**SafeDiffuser (Xiao, Wang, Gan, Hasani, Lechner, Rus, "Safe Planning with Diffusion Probabilistic Models", ICLR 2025, arXiv:2306.00148, first posted 2023)**
- Base model is Diffuser, a state-trajectory planner, not an action-space policy; barrier conditions b(x) >= 0 on every planned state are enforced by a minimum-deviation QP at every denoising step on the current noisy trajectory; three variants (robust-safe, relaxed-safe with 50 extra steps, time-varying) — [arXiv 2306.00148 (HTML)](https://arxiv.org/html/2306.00148); [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/28682)
- Maze2D: violations driven to ~1e-7 for all three variants, but planning score fell from 1.016 (Diffuser) to 0.77 to 0.81; time per diffusion step 0.007 s -> 0.107 s (about 15x) with 256 + 50 steps — [arXiv 2306.00148 (HTML)](https://arxiv.org/html/2306.00148)
- Locomotion (Walker2D / Hopper, roof collision): violation metric -9.375 / -2.180 -> 0.0 with score 0.346 -> 0.312 and 0.455 -> 0.430; per-step time 4 to 5x — [arXiv 2306.00148 (HTML)](https://arxiv.org/html/2306.00148)
- PyBullet 7-DoF manipulation with joint-limit and speed-dependent joint-limit specs: reward 0.650 -> 0.925 (robust-safe); per-step time 0.038 -> 0.088 s — [arXiv 2306.00148 (HTML)](https://arxiv.org/html/2306.00148)
- Specifications must be continuously differentiable analytic functions of planned state; the paper acknowledges a "gap between planning and control" and does not address action-only policies or dynamics mismatch — [arXiv 2306.00148 (HTML)](https://arxiv.org/html/2306.00148)

**Latent Policy Barrier (Sun, Song, "Learning Robust Visuomotor Policies by Staying In-Distribution", arXiv:2508.05941, v2 July 2026)**
- Mechanism: OOD score = squared L2 distance of the current latent to the nearest expert latent; when it exceeds a per-task threshold, a learned latent dynamics model predicts the latent after the candidate action and the gradient of that OOD score w.r.t. the action is added to the noise prediction during only the last K_guide denoising steps (10 of 100 DDPM; real robot 16 DDIM steps with guidance in the last 5). Base policy is not fine-tuned — [arXiv 2508.05941 (HTML)](https://arxiv.org/html/2508.05941v2)
- Data for the dynamics model: 150 to 270 rollouts of intermediate checkpoints (kept whether or not they succeeded) plus ~40 expert demos per sim task; real Belt Assembly 200 demos + 400 rollouts — [arXiv 2508.05941 (HTML)](https://arxiv.org/html/2508.05941v2)
- Simulation (20% of demos), base -> LPB: Square 0.56 -> 0.65; Transport 0.68 -> 0.85; Tool-Hang 0.27 -> 0.39; Push-T 0.51 -> 0.65; Libero-10 0.65 -> 0.75 — [arXiv 2508.05941 (HTML)](https://arxiv.org/html/2508.05941v2)
- Real robot Belt Assembly (described as contact-rich and high-precision, 40 rollouts each): 0.55 -> 0.75 — [arXiv 2508.05941 (HTML)](https://arxiv.org/html/2508.05941v2)
- Cost: dynamics model ~24 h on one L40S (sim), ~36 h on six L40S (real); no test-time latency reported. Limitation: "only corrects short-term, local deviations" — [arXiv 2508.05941 (HTML)](https://arxiv.org/html/2508.05941v2)

**PPGuide (Wang, Jha, Romeres, Qureshi, "Steering Diffusion Policies with Performance Predictive Guidance", arXiv:2603.10980, March 2026)**
- Signal: only binary trajectory success/failure. An attention-MIL model assigns per-chunk weights, a z-score threshold labels chunks success-relevant / failure-relevant / irrelevant, and a 3-class classifier is trained on those labels; its log-probability gradients are added to the noise estimate (attract success-relevant, repel failure-relevant), on alternate denoising steps to save cost — [arXiv 2603.10980 (HTML)](https://arxiv.org/html/2603.10980)
- Number of rollouts used is not stated (rollouts from checkpoints at epochs 250 to 450 of base policies trained on 10% of demos) — [arXiv 2603.10980 (HTML)](https://arxiv.org/html/2603.10980)
- Results (DP -> PPGuide, two checkpoints): Square 62/58 -> 72/66; Transport 60/68 -> 68/76; Stack Three D1 28/30 -> 34/28; Coffee D2 54/46 -> 58/58; Mug Cleanup 26/26 -> 30/36. Several cells flat or down — [arXiv 2603.10980 (HTML)](https://arxiv.org/html/2603.10980)
- Compute described only as "lightweight"; contact failures not analysed — [arXiv 2603.10980 (HTML)](https://arxiv.org/html/2603.10980)

**Feedback World Model (An et al., "Feedback World Model Enables Precise Guidance of Diffusion Policy", NeurIPS 2026 per arXiv comments, arXiv:2605.15705)**
- Guidance energy: squared distance between the (feedback-corrected) predicted next latent and the nearest expert latent, weighted by estimated action-controllability of latent dimensions, applied only in the final low-noise denoising steps; the feedback state is updated online from real observations with no parameter updates — [arXiv 2605.15705 (HTML)](https://arxiv.org/html/2605.15705v2)
- Simulation (UNet diffusion policy on 20% of demos), base -> standard world-model guidance -> feedback: Square 0.36 -> 0.40 -> 0.46; Tool-Hang 0.27 -> 0.22 -> 0.36; Transport 0.34 -> 0.46 -> 0.60; LIBERO-Plus Kitchen-S3 0.55 -> 0.61 -> 0.68; average 0.40 -> 0.43 -> 0.52 — [arXiv 2605.15705 (HTML)](https://arxiv.org/html/2605.15705v2)
- Real (20 trials each): Peach pick-and-place 90 -> 95 in-distribution, 40 -> 80 out-of-distribution; Drawer-Open 70 -> 75 and 20 -> 70 — [arXiv 2605.15705 (HTML)](https://arxiv.org/html/2605.15705v2)
- Requires a pretrained world model (Fast-WAM-Joint); drops/slips not addressed; stated limitation is inference latency — [arXiv 2605.15705 (HTML)](https://arxiv.org/html/2605.15705v2)

**Optimisation-guided diffusion (Bodmer, Zurbruegg, Portela, Ma, Didier, Hutter, Jones, Zeilinger, "Grounding Generative Policies in Physics", arXiv:2606.24208, June 2026)**
- Each DDIM step's noise is replaced by an optimisation variable; constraints are reachability (learned MLP surrogate from ~1e5 IK-labelled targets per arm), collision (signed-distance hinge) and controller executability (kinematic rollout of an impedance-style controller); solvers IPOPT (hard), Theseus (soft), L-BFGS (online offset in the last steps). Diffusion prior not retrained — [arXiv 2606.24208 (HTML)](https://arxiv.org/html/2606.24208)
- Grasp success (Dynaarm / Franka): gradient guidance 58.8 / 50.9 -> IPOPT 69.8 / 71.0; visuomotor pick-and-place 29.3 / 41.0 -> L-BFGS 41.8 / 67.0; drawer 79.5 / 57.8 -> 81.8 / 60.0 — [arXiv 2606.24208 (HTML)](https://arxiv.org/html/2606.24208)
- Cost: DDIM ~79 ms vs IPOPT ~5.5 s per grasp trajectory; visuomotor guided 0.28 to 0.32 s vs 0.02 to 0.07 s unguided; grasp slip and object drop are not modelled — [arXiv 2606.24208 (HTML)](https://arxiv.org/html/2606.24208)

**PACT (Wu et al., "Self-Evolving Physical Safety Alignment for Diffusion Policies in Embodied Manipulation", arXiv:2606.08414; search metadata says ICML 2026)**
- Post-training (weights change) that projects the policy onto constraint-feasible regions via a reverse-KL objective on self-rollouts; reports safety violations reduced 31.0% and success improved 30.7% on average, with no absolute numbers on the abstract page — [arXiv 2606.08414](https://arxiv.org/abs/2606.08414)

**Related: DynaGuide and ITPS guided-diffusion rows** (see KQ1): ITPS found plain gradient guidance (GD) raised Maze2D collisions from 0.01 to 0.06 and dropped block-stacking constraint satisfaction to 15% at high beta; DynaGuide used M=4 stochastic repeats per step specifically so guidance strength could be raised "without producing invalid trajectories" — [arXiv 2411.16627 (HTML)](https://arxiv.org/html/2411.16627v2); [arXiv 2506.13922 (HTML)](https://arxiv.org/html/2506.13922v2)

**Value guidance inside flow (VGFM, Koirala, Campbell, "Expressive Robot Policies via Dense Value Guidance in Flow Matching", IROS 2026, arXiv:2609.14261)**
- Offline-RL critic evaluates the action at intermediate flow steps (x-prediction parameterisation) so guidance can be applied at sampled flow times without backprop through time; evaluated on OGBench only; no success numbers on the abstract page — [arXiv 2609.14261](https://arxiv.org/abs/2609.14261)

### Inferences
- The Can Transport upright result is the closest published analogue to "never drop the jar, keep it upright": a tilt constraint on a carried object, enforced through a world model, with ranking of samples (best-of-N) doing nothing and guidance doing almost everything. The cost side is a differentiable next-state predictor trained on the policy's own data and a gradient path into the denoiser, which openpi exposes (the sampler is a plain Euler loop; see KQ6).
- World-model guidance acts before contact by steering the approach; the models used predict object state or visual latents, not contact forces. Whether such a model can be accurate for a contact-decided divergence (the user's post-touch physics) is not tested anywhere; LPB's Belt Assembly (0.55 -> 0.75 real) is the only contact-rich datapoint, and the authors say it only fixes short-horizon deviations.
- Guidance without a dynamics model (ITPS GD, pure objective gradients) tends to push samples off-manifold and raise execution failures; the fixes that worked (DynaGuide M=4 repeats, LPB last-K-steps only, Feedback WM low-noise steps only) all limit guidance to late denoising steps or re-anneal.
- SafeDiffuser-style hard constraints are for state-trajectory planners with analytic specs and cost 4x to 15x per step; they are not demonstrated on action-chunk policies like pi0.5.
- Data budgets for the learned component (LPB 150 to 270 rollouts + 40 demos per task; DiVeR 160 to 540 episodes; PPGuide unspecified) are within a 1,000 to 2,000-episode budget, but each adds a model training run and a sampler patch.

### Gaps
- No paper reports guidance of a VLA-scale flow policy (pi0 / pi0.5) inside the sampler on a safety constraint; all constraint-guidance results are on UNet/Transformer Diffusion Policies or Diffuser.
- Temporal Logic Guidance is a 3-page workshop paper with lambda, step count and compute unreported; no independent replication found.
- No paper models grasp slip, object drop or post-contact dynamics explicitly in the guidance objective.

## Key question 3: Noise and latent manipulation (fixed noise, noise-space RL, noise scale/temperature, denoising-step count, sampling-vs-physics variance)

### Takeaway
Two 2025 to 2026 results show the initial noise of a frozen diffusion/flow policy can carry a large effect: Golden Ticket (a single fixed noise vector chosen by random search) raised SimplerEnv GR00T N1.5 from 63.0% to 93.4% and real Franka tasks from 57.5% to 88.5% average, and DSRL (RL over the noise) took pi0 from 5/20 to 18/20 on a real toaster task in 80 episodes. But the effect is strongly task-dependent (2.5% of random noises beat the base policy on Robomimic Can with a 28-D MLP vs over 50% for pi0 on a LIBERO task; no ticket was found for Square 52.2 -> 42.6 and Threading 62 -> 60), simply adding sampler noise at inference hurts (pi0 LIBERO-Spatial 62.3% -> 9.4% as SDE noise rises), and no paper decomposes per-episode outcome variance into sampling versus physics.

### Cited Findings

**Golden Ticket (Patil, Biza, Weng, Schmeckpeper, Thomason, Zhang, Sivakumar, Walters, Gopalan, Castro, Hart, Rosen, "You've Got a Golden Ticket: Improving Generative Robot Policies With A Single Noise Vector", arXiv:2603.15757, v1 March 2026, v4 3 Oct 2026, no venue stated)**
- Method: replace the per-step random initial noise with one constant noise vector w of shape (H x d_a), used at every action step of every episode; sampling is deterministic DDIM so the initial noise is the only stochasticity; the vector is found by random search (RS) or cross-entropy method (CEM) with sequential halving over held-out environments; zeroth-order gradient search "failed on some seeds" — [arXiv 2603.15757 (HTML)](https://arxiv.org/html/2603.15757)
- Search budgets (tickets x search environments): franka_sim ~250 x 25; robomimic 5000 x 100; LIBERO-Object 1416 x 5, LIBERO-Goal 1338 x 3 to 5, LIBERO-Spatial 1081 x 3 to 5; SimplerEnv 250 x 5 to 10 (CEM); DexMimicGen 500 x 50; real Franka cube pick 6 x 25 (~150 episodes, ~30 min), piston assembly 22 tickets over 61 episodes (~45 min), banana pick 10 x 5, cup push 10 x 5 — [arXiv 2603.15757 (HTML)](https://arxiv.org/html/2603.15757)
- Held-out results, Gaussian noise -> golden ticket: franka_sim (flow-matching MLP, 44 checkpoints) 38.5% -> 96.0%; robomimic (DPPO diffusion) average 46.0% -> 63.1% (Can 42.8% -> 95.0%); SmolVLA LIBERO-Object 92.0 -> 100.0, Goal 81.6 -> 95.2, Spatial 80.4 -> 93.8; GR00T N1.5 SimplerEnv 63.0% -> 93.4% (300 held-out episodes); DexMimicGen UNet 74.2% -> 80.3%; real Franka 4-task average 57.5% -> 88.5% (cube pick 80 -> 98 over 50 episodes, piston assembly 60 -> 88 over 25, banana pick 50 -> 68 over 50, cup push 40 -> 100 over 10) — [arXiv 2603.15757 (HTML)](https://arxiv.org/html/2603.15757); [arXiv abstract](https://arxiv.org/abs/2603.15757)
- Headline: improved 46 of 51 tasks, at least matched on 49; named non-improvements: robomimic Square 52.2% -> 42.6% ("no golden ticket found") and DexMimicGen Threading 62% -> 60%; golden tickets found for 33 of 44 robomimic checkpoints — [arXiv 2603.15757 (HTML)](https://arxiv.org/html/2603.15757)
- Fraction of random tickets beating the base policy is task/policy dependent: 2.5% of 200 on Can with a 28-D MLP vs "over 50%" of 200 for pi0 (1600-D noise) on LIBERO-90 task #57; franka_sim ticket success rates ranged from ~5% to nearly 100% — [arXiv 2603.15757 (HTML)](https://arxiv.org/html/2603.15757)
- pi0 experiment: three LIBERO-90 tasks, 200 sampled tickets; exact pi0 before/after percentages are reported only in Fig. 6, where golden tickets "become comparable" to actor-critic (DSRL) in sample efficiency — [arXiv 2603.15757 (HTML)](https://arxiv.org/html/2603.15757)
- Ticket transfer: a ticket found on one task matched or exceeded the base policy on 7 of 10 LIBERO-Goal, 7 of 10 LIBERO-Object and 5 of 10 LIBERO-Spatial tasks — [arXiv 2603.15757 (HTML)](https://arxiv.org/html/2603.15757)
- A ticket that succeeds only 2/50 times was used to make the real policy "miss reliably", i.e. the noise can also select failure — [arXiv 2603.15757 (HTML)](https://arxiv.org/html/2603.15757)
- The authors explicitly leave a mechanistic explanation to future work and do not decompose outcome variance into noise versus environment; the trade-off between number of tickets N and search environments M per ticket is noted qualitatively — [arXiv 2603.15757 (HTML)](https://arxiv.org/html/2603.15757)
- Comparison with DSRL on DexMimicGen at DDIM-22 vs DDIM-88 with 5k and 10k episode budgets: DSRL "often struggles to match the base policy at DDIM-88 but does better at DDIM-22" — [arXiv 2603.15757 (HTML)](https://arxiv.org/html/2603.15757)

**DSRL (Wagenmaker, Nakamoto, Zhang, Park, Yagoub, Nagabandi, Gupta, Levine, "Steering Your Diffusion Policy with Latent Space Reinforcement Learning", arXiv:2506.15799, June 2025)**
- Keeps the denoiser frozen and trains an actor (SAC) that outputs the initial noise w in a box [-b, b]^d; black-box access only, forward passes only — [arXiv 2506.15799 (HTML)](https://arxiv.org/html/2506.15799)
- pi0 real robot: turn on toaster 5/20 -> 18/20 in 80 online episodes (~11,000 steps); spoon on plate 15/20 -> 19/20 in 65 episodes (~10,000 steps); for pi0 the noise actor outputs a 32-dim single-step noise repeated across the chunk, trained locally on an RTX 3070 (8 GB) — [arXiv 2506.15799 (HTML)](https://arxiv.org/html/2506.15799)
- pi0 LIBERO-90 "pick up the cream cheese and put it in the tray": roughly 20% -> roughly 100% after ~10,000 online samples; on AlohaTransferCube the paper states neither V-GPS nor ResiP significantly improved pi0 — [arXiv 2506.15799 (HTML)](https://arxiv.org/html/2506.15799)
- Franka single-task Diffusion Policy: 2/10 -> 9/10 in ~3,500 steps (~40 episodes); WidowX multi-task DiT: improvements with ~100 to 150 episodes — [arXiv 2506.15799 (HTML)](https://arxiv.org/html/2506.15799)
- Number of denoising steps and training epochs "barely affect" DSRL; the noise-to-action map is argued to alias many noises to one action and the authors warn "a highly concentrated action distribution may offer too few options for steering" — [arXiv 2506.15799 (HTML)](https://arxiv.org/html/2506.15799)
- No dexterous or contact-rich results are reported despite being mentioned as an investigation; no quantification of action diversity vs noise — [arXiv 2506.15799 (HTML)](https://arxiv.org/html/2506.15799)

**Adding sampler noise at inference (pi-RL ablation)**
- Converting pi0's deterministic ODE to an SDE with noise sigma_tau = a * sqrt(tau / (1 - tau)) and running the SFT policy stochastically (no RL): LIBERO-Spatial success 62.3% at a=0.2, 56.0% at a=0.5, 46.6% at a=0.8, 9.4% at a=1.0, while deterministic evaluation of the same policy stays ~65% — [arXiv 2510.25889 (HTML v3)](https://arxiv.org/html/2510.25889v3)
- Genetic denoising likewise found the DDPM per-step noise-injection scale gamma was decisive (gamma=0.2 best; "suffers from using gamma=1") — [arXiv 2510.21991 (HTML)](https://arxiv.org/html/2510.21991)

**Number of denoising steps**
- Diffusion Policy (Chi et al., RSS 2023): "DDIM with 100 training iterations and 10 inference iterations enables 0.1s inference latency"; no success-rate effect of step count is reported — [arXiv 2303.04137 (HTML)](https://arxiv.org/html/2303.04137)
- Dita (Hou et al., arXiv:2503.19757), SimplerEnv Google Robot, trained with 1000 DDPM steps, DDIM at inference (Pick Coke variant / match, Move Near variant / match): 100 steps 76.4 / 79.7 / 52.1 / 49.1; 50 steps 79.1 / 83.3 / 66.0 / 72.0; 20 steps 85.5 / 83.7 / 73.0 / 76.0; 10 steps 85.3 / 82.0 / 69.5 / 74.0; 5 steps 82.7 / - / 63.5 / 72.0; 2 steps 70.4 / 73.3 / 51.6 / 65.0 — far fewer steps than training was better, with a floor around 2 — [arXiv 2503.19757 (HTML)](https://arxiv.org/html/2503.19757)
- DPPO: pretrains with 20 denoising steps and fine-tunes the last 10 (state); with pixels pretrains 100 DDPM steps and fine-tunes 5 DDIM steps; robustness to noise holds "if at least four denoising steps are used" — [arXiv 2409.00588 (HTML)](https://arxiv.org/html/2409.00588v3)
- pi-RL trains pi0 with 4 denoising steps and pi0.5 with 3 to 5 — [arXiv 2510.25889 (HTML v3)](https://arxiv.org/html/2510.25889v3)
- Golden Ticket's DSRL comparison: DSRL does better at DDIM-22 than DDIM-88 — [arXiv 2603.15757 (HTML)](https://arxiv.org/html/2603.15757)

**Where the stochasticity comes from, and what is known about its share of outcome variance**
- Diffusion Policy attributes multimodality to "a random Gaussian initialization and stochastic Langevin-style perturbations during denoising" and shows each rollout commits to one mode; it does not measure how sampler stochasticity affects success rates — [arXiv 2303.04137 (HTML)](https://arxiv.org/html/2303.04137)
- Contractive Diffusion Policies (Abyaneh et al., ICLR 2026, arXiv:2601.01003) bounds how far two sampling flows can separate (Corollary 3.1.1) and adds a training-time contraction loss (gamma=0.1) to reduce "unwanted action variance"; Robomimic average DBC-DiT 0.64 vs CDP-DiT 0.78, DP-UNet 0.88 vs CDP-UNet 0.90; requires retraining; no empirical seed-sensitivity measurement — [arXiv 2601.01003 (HTML)](https://arxiv.org/html/2601.01003v2)
- TRI Large Behavior Model evaluation study (arXiv:2507.05331, 50 real rollouts and 200 sim rollouts per task/policy/condition, sequential statistical test with 5% false-separation bound): "Each simulation is deterministic given a random seed (however, the GPU-based policies do not have the same guarantee)"; the paper does not separate sampler randomness from initial-condition randomness and excludes training-seed variance — [arXiv 2507.05331 (HTML)](https://arxiv.org/html/2507.05331)
- Self-Guided Action Diffusion (Malhotra, Liu, Finn, arXiv:2508.12189) nudges each denoising step toward the previous chunk's actions (gradient scale ~4000, tuned per task) and reports a 71.4% relative single-sample gain on Robomimic and 26.5% on dynamic Push-T, but gives no temperature or diversity measurement — [arXiv 2508.12189 (HTML)](https://arxiv.org/html/2508.12189)

### Inferences
- Golden Ticket and DSRL are evidence that the noise can matter a lot for some policy/task pairs, while the user's local evidence (same-seed repeats diverge only after contact; noise evolution found nothing) matches Golden Ticket's own failures on precision-contact tasks (Square, Threading) where no ticket beat Gaussian sampling. The literature therefore supports both outcomes and offers no rule for predicting which regime a task is in beyond measuring the spread of ticket success rates directly.
- The fraction-of-tickets statistic is the cheap diagnostic the literature uses: if, after evaluating a few hundred fixed-noise candidates on a handful of environments each, the spread of per-ticket success is narrow, the noise is not the lever. Golden Ticket's real-robot budgets (50 to 150 episodes) and SimplerEnv budget (1,250 to 2,500 episodes) bracket a 1,000 to 2,000-episode budget; its LIBERO budgets (~7,000) exceed it.
- Noise-space methods presuppose a deterministic sampler so the noise fully determines the chunk; in openpi that holds (KQ6). They do not presuppose a deterministic environment, which is why they evaluate each candidate on several environments, but a contact-decided outcome with the same chunk yielding different physics shrinks the measurable effect of any noise choice.
- Raising the sampler's temperature is consistently harmful (pi-RL SDE ablation, genetic-denoising gamma); lowering stochasticity (fixed noise, contraction, self-guidance toward the previous chunk) is consistently neutral-to-helpful. For a safety objective, that points toward reducing, not increasing, sampling variance.
- Step count: 10 Euler steps (openpi default) is at or past the plateau in every ablation found (Dita: 10 to 20 best; DPPO: >= 4 is enough; DSRL: no effect). Changing it is not an expected source of success or safety change.

### Gaps
- No paper decomposes per-episode outcome variance into sampling noise versus environment/physics for contact-rich manipulation; the question the user asked is unmeasured in the literature.
- Golden Ticket's pi0 numbers are figure-only; the base vs golden percentages for pi0 could not be extracted.
- No image-diffusion "golden noise" method (learned noise predictors, attention-based noise selection) has been transferred to robot policies other than through the two robot papers above; I found no evidence either way.
- No published noise-scale/temperature knob exists for pi0.5 beyond the pi-RL SDE ablation, which is harmful at every tested level.

## Key question 4: Test-time adaptation and test-time training of VLAs (2025 to 2026)

### Takeaway
Weight-free test-time adaptation of a frozen VLA has one method with concrete numbers on pi0.5: Retrieve-then-Steer stores successful observation-action segments during deployment and injects a retrieved "elite" chunk at an intermediate flow time, giving pi0.5 LIBERO-10 92.4% -> 94.4% and real-robot gains of +8 to +12 points at 1.10x latency, with the memory built from a few hundred online episodes and a progress critic for success labelling. Methods with larger gains (RoboTTT +37 points real, RA-VLA +17.6 points LIBERO held-out) require retraining the policy and are not frozen-weight methods.

### Cited Findings

**Retrieve-then-Steer (Zhao et al., "Online Success Memory for Test-Time Adaptation of Generative VLAs", arXiv:2605.10094, May 2026)**
- Frozen VLA; memory of "progress-calibrated successful observation-action segments"; top-K=10 retrieved chunks filtered by DTW consistency and aggregated into an elite prior a_elite; sampling starts at an intermediate flow state x_t0 = (1 - t0) a_elite + t0 eps, where t0 is set from retrieval confidence (sigmoid, gamma = 2.0); fixed t0 = 0.6 gave 0.68 vs dynamic 0.70 on Moka Pots — [arXiv 2605.10094 (HTML)](https://arxiv.org/html/2605.10094v2)
- Base policies and sim results (50 trials/task): pi0.5 LIBERO-10 92.4% -> 94.4%; pi0 81.6% -> 84.4%; CogACT SimplerEnv Google Robot average 75.8% -> 79.5% — [arXiv 2605.10094 (HTML)](https://arxiv.org/html/2605.10094v2)
- Online budget: ablation runs 300 trajectories on Moka Pots on Stove from an empty memory; cumulative success 61.0% (no memory) -> 71.2% (unlimited) — [arXiv 2605.10094 (HTML)](https://arxiv.org/html/2605.10094v2)
- Success signal: a pretrained VLAC progress critic conditioned on one reference success video, threshold 0.95; precision 0.970, recall 0.678; oracle labels give 94.8% vs 94.4% with the critic — [arXiv 2605.10094 (HTML)](https://arxiv.org/html/2605.10094v2)
- Real pi0.5: OpenArm bowl stacking 72% -> 80%, cube handoff 40% -> 52%; ALOHA-PiPER T-shirt folding average 39% -> 48% — [arXiv 2605.10094 (HTML)](https://arxiv.org/html/2605.10094v2)
- Latency 0.95x base without retrieval, 1.10x with; no parameter updates — [arXiv 2605.10094 (HTML)](https://arxiv.org/html/2605.10094v2)

**Test-Time Perturbation Learning with Delayed Feedback (Zang et al., CVPR 2026, arXiv:2604.18107)**
- A ~9M-parameter perturbation head adds lambda * h_theta(f) to the frozen VLA's action logits; trained online by a REINFORCE-style loss -(r - b) log pi from episode-level reward plus a KL term, with entropy-scaled test-time augmentation and dimension-wise action voting over up to 4 views — [arXiv 2604.18107 (HTML)](https://arxiv.org/html/2604.18107)
- OpenVLA on LIBERO (Table 1): Spatial 0.85 -> 0.90, Object 0.64 -> 0.72, Goal 0.76 -> 0.86, Long 0.53 -> 0.59, average 0.69 -> 0.77; number of adaptation episodes not stated; baseline numbers inconsistent across tables — [arXiv 2604.18107 (HTML)](https://arxiv.org/html/2604.18107)
- Token-logit method; not directly applicable to a flow head — [arXiv 2604.18107 (HTML)](https://arxiv.org/html/2604.18107)

**TT-VLA (Liu et al., "On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning", ACL 2026)**
- Test-time RL with a dense reward from "step-by-step task-progress signals"; abstract gives no base model, numbers or budgets — [ACL Anthology 2026.acl-long.1863](https://aclanthology.org/2026.acl-long.1863/)

**TTT-VLA (Zhang et al., "Test-Time Latent Prompt Optimization for VLA", arXiv:2606.03127, June 2026)**
- Optimises only a latent prompt with a self-supervised proxy loss on data collected in the current environment, policy unchanged; SimplerEnv; abstract claims gains come from "a small number of critical decisions"; no numbers on the abstract page — [arXiv 2606.03127](https://arxiv.org/abs/2606.03127)

**T3VF (Park et al., "Test-Time Training for Visual Foresight VLA Models", ICML 2026 workshop, arXiv:2605.08215)**
- Uses the model's predicted future frame and the actual next observation as a supervision pair with an adaptive update filter; requires a visual-foresight VLA; no numbers on the abstract page — [arXiv 2605.08215](https://arxiv.org/abs/2605.08215)

**RA-VLA (Jang et al., POSTECH, arXiv:2608.25585)**
- Requires fine-tuning GR00T N1.5 with a contextual-adherence loss and a retrieval encoder; at test time retrieves K=1 segment from 3 demos per held-out task; LIBERO held-out average 1.7% (vanilla) -> 38.45%; real UR5e 8.3% -> 56.3% — not a frozen-weight method — [arXiv 2608.25585 (HTML)](https://arxiv.org/html/2608.25585v2)

**RoboTTT (Jiang et al., NVIDIA GEAR, "Context Scaling for Robot Policies", arXiv:2607.15275, 2026)**
- Adds test-time-training layers (fast weights updated per token by a key-value binding loss) to GR00T N1.7 and retrains with up to 8K-timestep context; real bimanual assembly average 42 -> 79; requires retraining, not applicable to a frozen pi0.5 — [NVIDIA project page](https://research.nvidia.com/labs/gear/robottt/)

**Other self-improvement labels that are not inference-time**: SeedPolicy (arXiv:2603.05117, "self-evolving" means a recurrent latent state trained jointly, +7 to +15 points on RoboTwin 2.0 over Diffusion Policy) — [arXiv 2603.05117 (HTML)](https://arxiv.org/html/2603.05117); Time-Unified Diffusion Policy with action discrimination (arXiv:2506.09422) is an architecture/training change, not a candidate selector — [arXiv 2506.09422](https://arxiv.org/abs/2506.09422)

### Inferences
- Retrieve-then-Steer is the only test-time adaptation result on pi0.5 with frozen weights; its gain on LIBERO-10 is small (+2 points) because the base is already at 92%, but real-robot gains were +8 to +12 points. Its mechanism (start the flow integration from a partially-noised retrieved chunk) requires a small change to the Euler loop, and the memory could be built from a few hundred episodes, which fits a 1,000 to 2,000-episode budget.
- The retrieved chunk only biases the approach; it cannot change post-contact physics, and the paper does not test safety.
- Everything else in this family either needs token logits (PDF), retraining (RA-VLA, RoboTTT, T3VF), or has not published numbers (TT-VLA, TTT-VLA).

### Gaps
- TT-VLA (ACL 2026) and TTT-VLA numbers are unavailable from the pages reached.
- No test-time adaptation paper reports a safety or violation metric.

## Key question 5: RL fine-tuning headroom above the frozen policy (DPPO, ReinFlow, pi-RL, SimpleVLA-RL, VLA-RL)

### Takeaway
Online RL on flow/diffusion policies shows the frozen policy is far from the ceiling: pi0.5 went from 77.1% to 98.3% on LIBERO and 40.1% to 90.9% on ManiSkill with pi-RL, and Diffusion Policy Transport went from 0.46 to 0.60 (DPPO) with 400,000 training steps. These budgets are tens of thousands of episodes with 64 to 320 parallel environments and multi-GPU training, and one RL paper documents the policy discovering "pushcut" shortcuts, i.e. behaviours that would count as violations under a safety spec.

### Cited Findings
- DPPO (Ren, Lidard, Ankile, Simeonov, Agrawal, Majumdar, Burchfiel, Dai, Simchowitz, arXiv:2409.00588): fine-tunes only the last denoising steps; pixel Transport reaches ">50%" where the Gaussian baseline stays at 0%; sim-to-real one-leg furniture assembly 80% (16/20) after fine-tuning — [arXiv 2409.00588 (HTML)](https://arxiv.org/html/2409.00588v3)
- DPPO numbers as tabulated by D2PPO (arXiv:2508.02644, Table 2, image-based Robomimic): Diffusion Policy pre-RL Lift 0.97, Can 0.96, Square 0.82, Transport 0.46 -> DPPO 1.00 / 1.00 / 0.83 / 0.60 -> D2PPO 1.00 / 1.00 / 0.89 / 0.87; training budget 90,000 steps (Lift/Can/Square) and 400,000 steps (Transport) — [arXiv 2508.02644 (HTML)](https://arxiv.org/html/2508.02644v1)
- ReinFlow (Zhang, Yu, Su, Wang, NeurIPS 2025, arXiv:2505.22094): learnable noise injected into the flow, discarded after training; Robomimic Can/Square/Transport "improve the success rate of the pre-trained policy by an average of 45.77%"; one denoising step for Can and Square; 62.82% less wall time than DPPO; no real-robot results — [arXiv 2505.22094 (HTML v6)](https://arxiv.org/html/2505.22094v6)
- pi-RL (Chen et al., arXiv:2510.25889 v3, Jan 2026): LIBERO few-shot SFT -> RL: pi0 57.6% -> 97.6% (Flow-Noise) / 96.1% (Flow-SDE); pi0.5 77.1% -> 98.3% / 97.9%; pi0.5 LIBERO-Long 43.9% -> 94.0%; ManiSkill pi0 38.4% -> 78.8%, pi0.5 40.1% -> 90.9% across 4,352 pick-and-place variations; budgets: 64 parallel envs, 500 training epochs, batch 2048 (LIBERO); 320 parallel envs, 150 RL steps, batch 5120 (ManiSkill); Real2Sim2Real Franka: SFT 0% -> RL 40% — [arXiv 2510.25889 (HTML v3)](https://arxiv.org/html/2510.25889v3)
- SimpleVLA-RL (Li et al., ICLR 2026, arXiv:2509.09674): OpenVLA-OFT LIBERO average 91.0% -> 99.1%; one-demo LIBERO-Long 17.1% -> 91.7%; RoboTwin 2.0 38.3% -> 68.8% (pi0 49.2% as a frozen baseline); 8 x A800 80 GB, 500 training scenarios per LIBERO suite, rollout temperature 1.6; sim-to-real 17.5% -> 38.5%; documents a "pushcut" behaviour in which RL replaces grasp-move-place with pushing the object into place — [arXiv 2509.09674 (HTML)](https://arxiv.org/html/2509.09674v1); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10009314)
- VLA-RL (Lu et al., arXiv:2505.18719): OpenVLA-7B with a fine-tuned VLM process reward model; "surpass the strongest finetuned baseline by 4.5%" on 40 LIBERO tasks; budgets not on the abstract page — [arXiv 2505.18719](https://arxiv.org/abs/2505.18719)
- pi-RL's SDE ablation (quoted under KQ3) shows the exploration noise RL relies on is itself harmful at inference without training — [arXiv 2510.25889 (HTML v3)](https://arxiv.org/html/2510.25889v3)

### Inferences
- The headroom is large (tens of points on LIBERO/ManiSkill for pi0.5) but it was bought with budgets two to three orders of magnitude above 1,000 to 2,000 episodes, gradient updates to the action expert, and dense simulator reward; inference-time methods recover a fraction of it (DiVeR +11.6 on RoboCasa pi0.5 vs pi-RL +50 on ManiSkill).
- RL's "pushcut" shortcut is a warning for safety objectives: optimising success alone can create violations; no RL paper on flow VLAs reports a violation rate.

### Gaps
- DPPO's own per-task pretrained-vs-fine-tuned table and episode counts were not extractable from the text (figures only); the D2PPO reproduction is used instead and has internal inconsistencies (Transport pre-training 36.7% in one caption vs 0.46 in the table).
- pi-RL does not state total environment episodes or GPU hours.

## Key question 6: Inference-time sampling of pi0 / pi0.5 specifically

### Takeaway
openpi's sampler is a deterministic 10-step forward-Euler integration from unit-variance Gaussian noise to the action chunk; the initial noise is the only stochasticity, there is no temperature or noise-scale parameter, and the public `Policy.infer()` accepts an explicit `noise` array, so fixed-noise, best-of-N over noises, and partial-noise initialisation are all possible without patching the model. Physical Intelligence has published nothing on sampling variance; third parties report that pi0 and pi0.5 respond to noise choice (DSRL, Golden Ticket), to candidate selection (DiVeR), and that added SDE noise hurts (pi-RL).

### Cited Findings
- `sample_actions(self, rng, observation, *, num_steps=10, noise=None)`: default `num_steps` is 10; noise is `jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))` with unit scale and no temperature parameter; `dt = -1.0 / num_steps`; Euler update `x_t + dt * v_t` from t=1 (noise) to t=0 (actions); the step function uses no RNG, so "given a fixed noise, the output is fixed"; pi0.5 differs only in time conditioning (adaRMS via a time MLP) — [openpi src/openpi/models/pi0.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)
- Policy wrapper: `rng` defaults to `jax.random.key(0)` and is split on every `infer()` call, so the noise sequence is deterministic given the starting key; `infer(self, obs, *, noise=None)` passes a caller-supplied noise straight to `sample_actions`; `num_steps` can only be changed through the constructor's `sample_kwargs` — [openpi src/openpi/policies/policy.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/policy.py)
- pi0 paper: "We use 10 integration steps (corresponding to delta = 0.1) in our experiments"; forward Euler; H=50; "For the 20Hz UR5e and Franka robots, we run inference every 0.8 seconds (after executing 16 actions)"; action chunks executed open-loop without aggregation; flow timestep sampled from Beta((s - tau)/s; 1.5, 1), s = 0.999; evaluation averaged over 10 episodes per task with no variance reported — [arXiv 2410.24164 (HTML)](https://arxiv.org/html/2410.24164)
- pi-RL on pi0/pi0.5: SDE noise injected at inference into the SFT policy reduces LIBERO-Spatial success from ~65% (deterministic) to 62.3 / 56.0 / 46.6 / 9.4% as a rises 0.2 / 0.5 / 0.8 / 1.0; RL training used 4 denoising steps (pi0) and 3 to 5 (pi0.5) — [arXiv 2510.25889 (HTML v3)](https://arxiv.org/html/2510.25889v3)
- DSRL on pi0: 32-dim noise actor, real-robot gains in 65 to 80 episodes; V-GPS and ResiP did not significantly improve pi0 on AlohaTransferCube — [arXiv 2506.15799 (HTML)](https://arxiv.org/html/2506.15799)
- Golden Ticket on pi0 (LIBERO-90 #57): more than half of 200 random fixed noises beat per-step Gaussian sampling — [arXiv 2603.15757 (HTML)](https://arxiv.org/html/2603.15757)
- DiVeR on pi0.5: 16 candidates sampled in ~205 ms; RoboCasa 43.1 -> 54.7, real Franka 58.3 -> 71.9 — [arXiv 2610.04933 (HTML)](https://arxiv.org/html/2610.04933v1)
- Retrieve-then-Steer on pi0.5: starts the flow at intermediate t0 from a blended retrieved chunk; LIBERO-10 92.4 -> 94.4 — [arXiv 2605.10094 (HTML)](https://arxiv.org/html/2605.10094v2)
- Other ports: Hugging Face transformers' pi0 config defaults to 10 inference steps; OpenTau's pi0.6 variant defaults to 5 flow steps (search snippets; not primary-source verified) — [OpenTau docs](https://opentau.readthedocs.io/en/latest/_autosummary/opentau.policies.pi0.modeling_pi0.html)

### Inferences
- Because the ODE is deterministic given `noise`, the user's same-seed frame-identical-until-contact result is exactly what the code predicts; divergence after contact is the simulator, not the sampler. Any noise-side method (fixed noise, noise RL, noise best-of-N) can therefore change only the pre-contact chunk and the approach, never the contact outcome for a given chunk.
- The `noise=` argument on `infer()` is enough to implement Golden Ticket search, DSRL's noise actor, and SDN/DiVeR-style best-of-N over noises with no model patch; guidance methods (KQ2) and Retrieve-then-Steer's intermediate-time start require editing the Euler loop in `sample_actions`, which is about ten lines.
- There is no temperature knob to turn down; the policy already runs at the minimum stochasticity its sampler allows, which removes "lower the temperature" from the list of cheap interventions.

### Gaps
- No Physical Intelligence publication or openpi documentation reports per-episode variance due to sampling, success spread across seeds, or a step-count ablation for pi0/pi0.5; the only step-count statements are the 10-step default and pi-RL's 3 to 5 steps during RL.
- Whether pi0.5 at 16-step chunks / 8 executed (user's setup) behaves differently from the paper's 50-step chunks with 16 executed is not addressed anywhere.

## Simplest things that worked (cheapest demonstrated effects, ranked by cost)

### Takeaway
Ranked by what had to be built, the cheapest measured effects are: (1) a single fixed initial noise chosen by random search (no training, 50 to 2,500 episodes, up to +30 points, but zero on precision-contact tasks); (2) a two-layer MLP verifier on frozen action-expert features trained on 160 to 540 of the policy's own episodes with binary labels, used for best-of-16 (+6 to +14 points on pi0.5); (3) a retrieved-success-chunk prior injected at an intermediate flow time (+2 to +12 points on pi0.5, ~300 episodes, needs a progress critic); (4) a Q-function or VLM verifier (+7 to +25 points, needs a large offline dataset or 20M synthetic pairs); (5) world-model gradient guidance, the only method with a measured safety effect (84% -> 4% upright violations), which needs a dynamics model and a sampler patch. Random best-of-N without a verifier and sampler temperature changes did nothing or hurt.

### Cited Findings
1. Fixed initial noise by random search: SimplerEnv GR00T N1.5 63.0 -> 93.4 with 250 tickets x 5 to 10 episodes; real Franka 57.5 -> 88.5 with 50 to 150 search episodes; no training, no internals; failed on Square (52.2 -> 42.6) and Threading (62 -> 60) — [arXiv 2603.15757 (HTML)](https://arxiv.org/html/2603.15757)
2. Tiny verifier on policy features + best-of-16: pi0.5 RoboCasa 43.1 -> 54.7 (random pick 44.1), real 58.3 -> 71.9; verifier is a 64-unit MLP trained on 160 to 540 self-rollouts with episode labels; < 0.5 ms scoring — [arXiv 2610.04933 (HTML)](https://arxiv.org/html/2610.04933v1)
3. Success-memory prior at intermediate flow time: pi0.5 LIBERO-10 92.4 -> 94.4, real +8 to +12; 300 online episodes; 1.10x latency; no weight updates — [arXiv 2605.10094 (HTML)](https://arxiv.org/html/2605.10094v2)
4. Training-free seed selection with a jerk rule plus object-masking grounding: pi0 SimplerEnv 63.9 -> 72.6, real +18 points; needs Grounding-DINO + SAM2 but no training — [arXiv 2606.14084 (HTML)](https://arxiv.org/html/2606.14084v1)
5. Noise-space RL (DSRL): pi0 real 5/20 -> 18/20 in 80 episodes; black-box, but needs a per-step reward and an online SAC loop — [arXiv 2506.15799 (HTML)](https://arxiv.org/html/2506.15799)
6. Offline-RL Q-function re-ranking (V-GPS): real +20 points average at K=50; needs ~175K-episode offline dataset and 1M critic steps — [arXiv 2410.13816 (HTML)](https://arxiv.org/html/2410.13816v2)
7. VLM verifier (RoboMonkey): real +25 points at 16 candidates; needs 20M synthetic preferences and 8 H100s; 650 ms/step — [arXiv 2506.17811 (HTML)](https://arxiv.org/html/2506.17811v2)
8. World-model gradient guidance for an upright constraint: 84% -> 4% violations at 100% success, where best-of-N ranking gave 82%; needs a differentiable next-state model trained on the policy's data and a gradient path into denoising — [arXiv 2606.22729 (HTML)](https://arxiv.org/html/2606.22729)
9. Latent-barrier guidance from 150 to 270 self-rollouts: Transport 0.68 -> 0.85, real contact-rich Belt Assembly 0.55 -> 0.75; dynamics model costs ~24 h on one L40S — [arXiv 2508.05941 (HTML)](https://arxiv.org/html/2508.05941v2)
- Did not work: random selection among 16 candidates (+1.0 point) — [arXiv 2610.04933 (HTML)](https://arxiv.org/html/2610.04933v1); raising sampler noise (pi0 62.3 -> 9.4%) — [arXiv 2510.25889 (HTML v3)](https://arxiv.org/html/2510.25889v3); sample-and-rank for a tilt constraint (84 -> 82% violations) — [arXiv 2606.22729 (HTML)](https://arxiv.org/html/2606.22729); naive gradient guidance without annealing (collisions 0.01 -> 0.06, constraint satisfaction 100 -> 15%) — [arXiv 2411.16627 (HTML)](https://arxiv.org/html/2411.16627v2); changing denoising-step count beyond ~5 (plateau in all ablations) — [arXiv 2503.19757 (HTML)](https://arxiv.org/html/2503.19757)

### Inferences
- For a contact-decided failure, the ranked list splits into two groups. Items 1, 3, 4 and 5 only change the sampled chunk and therefore the approach; the literature's own precision-contact failures (Square, Threading) and the user's evidence say their effect there is small. Items 2, 6, 7 also act on the chunk but with a learned scorer that has seen the policy's failures, and their documented wins are grasp-quality and release-timing failures, which are pre-contact decisions. Items 8 and 9 are the only ones that explicitly model what the object will do after the chunk is executed, and item 8 is the only one measured on an upright-object constraint.
- Nothing in the literature changes the physics after contact; every inference-time method is a bet that the failure is already decided by the chunk chosen before contact. The user's same-seed divergence evidence does not test that bet directly, because it holds the chunk fixed; it shows the physics is stochastic, not that the chunk choice is irrelevant to the odds.

### Gaps
- No paper combines a safety verifier (violation-aware) with best-of-N on a VLA; the safety-steering literature inside the sampler and the best-of-N literature on VLAs do not overlap.
- No paper reports an inference-time method on the ManiGuard benchmark or any benchmark with drop/tip-over metrics (not researched by instruction, but no hit appeared in any citation chain either).
