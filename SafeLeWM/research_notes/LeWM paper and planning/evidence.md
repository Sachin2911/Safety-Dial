# LeWM empirical evidence and limitations

## What does the original LeWM paper establish quantitatively?

### Takeaway

LeWM establishes a compact and comparatively efficient latent model with strong short-goal control results, particularly on PushT and Reacher. It does not win every task, and its main evaluation is a nearby goal sampled from an offline trajectory, not a full standard-start manipulation task. References below use v3, dated 3 June 2026; v1, dated 13 March 2026, is distinguished explicitly. Both are downloaded under `docs/papers/`. [Version history](https://arxiv.org/abs/2603.19312)

### Cited Findings

- Figure 6, v3 p7, visually checked against the downloaded PDF, gives these rounded percentages. DINO-WM is pixels-only except in the explicitly marked column. [Paper, Fig. 6](https://arxiv.org/pdf/2603.19312v3#page=7)

| Environment | LeWM | PLDM | DINO-WM | DINO-WM + proprioception | GCBC | GCIQL | GCIVL | Random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TwoRoom | 87 | 97 | 100 | 100 | 100 | 100 | 100 | 0 |
| Reacher | 86 | 78 | 79 | not shown | not shown | not shown | not shown | 10 |
| PushT | 96 | 78 | 74 | 92 | 75 | 20 | 33 | 2 |
| OGBench Cube | 74 | 65 | 86 | not shown | 84 | 64 | 56 | 48 |

- The PushT improvement over PLDM is 18 **percentage points** (96 minus 78), rather than an 18% relative increase, despite the text's loose wording. [Paper, Sec. 4.2 and Fig. 6](https://arxiv.org/pdf/2603.19312v3#page=7)
- Appendix F.1, p19, specifies evaluation starts sampled from offline trajectories and a goal 25 timesteps later in the same trajectory. The action budget is 50 for all four environments. Thus the paper's 96% PushT is not directly comparable to the AP full-query 89/128 in our repository. The latter has a different goal horizon and action budget. [Paper, App. F.1](https://arxiv.org/pdf/2603.19312v3#page=19); [our AP result](../../results/15-pusht-ap-reproduction.md)
- Table 5, p24, reports three training seeds evaluated on the **same 50 PushT trajectories**: LeWM 96.0 +/- 2.83, PLDM 78.0 +/- 5.0, DINO-WM 92.0 +/- 1.63. The caption calls the dispersion “variance”; it does not identify these numbers as confidence intervals or clearly as standard deviations. The DINO figure here is 92, whereas the pixels-only Fig. 6 DINO result is 74, and the table does not explicitly state the proprioception variant. Preserve that ambiguity. [Paper, Table 5](https://arxiv.org/pdf/2603.19312v3#page=24)
- Figure 3, p3, reports 0.98 seconds versus 47 seconds for the plotted planning computation, averaged over 50 runs, approximately a 48-fold ratio. Its separate fixed-FLOP comparison is PushT 90% versus 13%, and Cube 74% versus 48%, LeWM versus DINO-WM. Do not substitute those numbers for Fig. 6's unconstrained comparison. [Paper, Fig. 3](https://arxiv.org/pdf/2603.19312v3#page=3)
- Baselines are PLDM, DINO-WM, goal-conditioned behavioral cloning, and goal-conditioned IQL/IVL. PLDM coefficients were selected from 256 PushT configurations and then kept fixed. The DINO encoder has external pretraining, whereas LeWM is trained from scratch on task data. These are meaningful but different information budgets. [Paper, Sec. 4.1, App. C.2, Table 1](https://arxiv.org/pdf/2603.19312v3)

### Inferences

- LeWM is a credible affordable model for our opening study, especially when many candidate sequences must be scored. Its results do not establish a universal controller or a long-horizon safety solution.
- Random control achieves 48% on the Cube short-goal panel, while GCBC reaches 84% and LeWM 74%. This cautions against equating these goals with complete independent pick-and-place episodes from rest.
- Three model seeds tested on a shared 50-query panel measure training variability on that panel; they are not 150 independent new task cases.
- Under-one-second planning is not a guarantee of a 20 Hz controller, nor an end-to-end episode runtime. Benchmark our actual AP configuration before setting throughput or cost promises.

### Gaps

- Figure 6 has error bars but its caption does not define their statistic or give the full per-task evaluation denominators. Do not invent counts such as 87/100 from the displayed percentages.
- The paper does not transparently specify the relationship between the 92% DINO entry in Table 5 and Fig. 6's DINO variants.
- I did not find a sufficiently explicit fixed-FLOP implementation recipe in the prose to independently reconstruct that comparison just from the paper.

## What data, compute and ablations support those results?

### Takeaway

The evidence supports simpler training losses, but the full system still has important architecture, data and planning choices. The pretrained checkpoint route avoids pretraining cost while retaining online model-based planning cost. [Paper, Sec. 3 and App. D/G](https://arxiv.org/pdf/2603.19312v3)

### Cited Findings

- Appendix E, p18: TwoRoom uses 10,000 noisy-heuristic episodes, average 92 steps; PushT describes 20,000 expert episodes, average 196 steps; single-cube OGBench uses 10,000 heuristic episodes of 200 steps; Reacher uses 10,000 SAC-policy episodes of 200 steps. Each world model trains for 10 epochs. These are the **paper descriptions**, not verified sizes of our pinned released data files. [Paper, App. E](https://arxiv.org/pdf/2603.19312v3#page=18)
- Appendix D, pp17-18: 224x224 observations, batch 128, four-frame sub-trajectories, blocks of five actions; history three for PushT/Cube and one for TwoRoom. Native CEM samples 300 sequences, retains 30, and runs up to 30 iterations for PushT and ten for the other environments. It plans five action blocks, or 25 environment steps, and the appendix says it executes that optimized sequence before replanning. This is not our AP five-action replanning configuration. [Paper, App. D](https://arxiv.org/pdf/2603.19312v3#page=17)
- The abstract claims roughly 15M parameters and single-GPU training in a few hours. The **v1** implementation appendix explicitly names an NVIDIA L40S for training and planning. That sentence is absent from the corresponding v3 appendix, so attribute the hardware statement to v1 rather than implying a v3-specific measurement. [v3 abstract](https://arxiv.org/abs/2603.19312v3); [v1 App. D p20](https://arxiv.org/pdf/2603.19312v1#page=20)
- Table 7, p25: adding image-reconstruction loss reduces PushT performance from 96.0 +/- 2.83 to 86.0 +/- 7.54. Table 8: ViT 96.0 +/- 2.83 versus ResNet-18 94.0 +/- 3.27. [Paper, Tables 7-8](https://arxiv.org/pdf/2603.19312v3#page=25)
- Table 9, p26: predictor dropout 0.0/0.1/0.2/0.5 yields 78/96/85.33/66.67% mean PushT success. Table 10 compares LeWM solvers: CEM 96%, SGD 26%, RMSProp 67.33%, Adam 84%. Table 6, p24: tiny/small/base predictors yield 80.67/96/86.7%. [Paper, Tables 9-10](https://arxiv.org/pdf/2603.19312v3#page=26); [Table 6](https://arxiv.org/pdf/2603.19312v3#page=24)
- Figure 15, p24, studies embedding dimensions, SIGReg projection counts and integration knots. Figure 16, p25, reports above-80% PushT performance for regularization weights 0.01 through 0.2, with deterioration at 0.5. [Paper, Figs. 15-16](https://arxiv.org/pdf/2603.19312v3#page=24)

### Inferences

- “One hyperparameter” is about the effective **loss-balancing** choice, not the whole training and planning pipeline. Dropout, predictor size, representation size, optimizer, horizon and CEM settings still matter.
- The paper's suggested logarithmic bisection search is not a general proof of logarithmic black-box hyperparameter optimization. Treat it as the authors' practical tuning claim unless the required shape assumptions are established.
- Our frozen-model study can focus on adapters without incurring model pretraining. It still needs model-query counts as well as simulator-action counts, because online CEM is substantial work.

### Gaps

- No exact per-environment training wall-time or VRAM requirement is established by the “few hours” claim alone.
- Released asset episode counts can differ from paper-level descriptions. Our pinned data manifest and actual HDF5 layout are the execution authority.

## What does this imply for safety, generalization and our proposed adapter?

### Takeaway

The paper demonstrates useful representations and short-horizon control, not calibrated hazard awareness or safe completion. Its limitations closely motivate our work, but none guarantees that an adapter has enough authority to recover a safe route. [Paper, Secs. 5-6](https://arxiv.org/pdf/2603.19312v3#page=8)

### Cited Findings

- Section 6, p9, explicitly limits planning to short horizons and notes dependence on sufficient offline coverage; low-diversity data can weaken Gaussian regularization in simple environments. [Paper, limitations](https://arxiv.org/pdf/2603.19312v3#page=9)
- Table 3, p20: TwoRoom LeWM and PLDM have matching linear position correlation 0.996 and near-perfect nonlinear decoding, despite the planning gap. Its caption says dynamics or planning could explain that gap. This offers an alternative to the main-text speculation about poor latent structure. [Paper, Table 3](https://arxiv.org/pdf/2603.19312v3#page=20)
- Table 4, p21: Cube LeWM nonlinear block-position correlation is 0.999, but end-effector yaw 0.213 and block yaw 0.164. DINO-WM has end-effector-yaw correlation 0.917. Decoded rollouts also lose finer end-effector orientation detail. [Paper, Table 4](https://arxiv.org/pdf/2603.19312v3#page=21); [Fig. 9](https://arxiv.org/pdf/2603.19312v3#page=19)
- Figure 8 and Appendix F.3 test surprise after artificial object teleportation versus color changes, using latent prediction error. The figure caption reports paired-test p<0.01 for teleportation. This is a violation-of-physical-expectation test, not a safety-rule test or a future risk calibration study. [Paper, Fig. 8 and App. F.3](https://arxiv.org/pdf/2603.19312v3#page=9)
- The official release provides separate model checkpoints for PushT, Cube, TwoRoom and Reacher. It does not claim one shared set of weights handles all four environments. [Official repository](https://github.com/lucas-maes/le-wm)

### Inferences

- A physically plausible collision can be predictable and therefore unsurprising. Conversely, changing an object's appearance can cause model error without a task hazard. Surprise cannot be used as an already validated safety monitor.
- A hazard label, forbidden region or requested sequence is extra task information. The model's compact representation might not retain it, and lower latent goal error does not enforce it.
- We should distinguish representation sufficiency, dynamics accuracy, planner reachability, adapter authority and safety monitoring when diagnosing failure. A coordinate probe alone does not validate all five.
- AP's retrieved intermediate goals are a separate planning extension; its benefits, data dependence and long-horizon results must be attributed to that paper rather than added to original LeWM claims.

### Gaps

- No evidence here establishes risk calibration, obstacle avoidance under our added hazards, same-weight cross-environment transfer, evolutionary optimizer superiority, or reliable long-horizon recovery.
- The VoE prose and plots do not supply enough detail here to recover an operational false-positive/false-negative curve or a calibrated deployment threshold.

Source/license note: arXiv lists this work under CC BY 4.0, verified through its license link. These notes paraphrase and tabulate findings with attribution. [arXiv record](https://arxiv.org/abs/2603.19312); [license](https://creativecommons.org/licenses/by/4.0/).
