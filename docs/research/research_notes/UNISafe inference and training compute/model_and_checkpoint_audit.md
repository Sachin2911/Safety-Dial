# UNISafe model and checkpoint compute audit

Recorded 7 October 2026. This is a read-only source and checkpoint-metadata audit, not a numerical benchmark or an experiment. The official `isaaclab` branch was pinned to `ab0fdf87f38edefa48b54a291ec5b7156c17b6e0`. No GPU was rented and no model was trained.

## What model is actually released?

### Takeaway

The advertised public archive is accessible. Its tensor metadata matches the simulated block-plucking configuration: a 39.85 million parameter world model accompanied by a much larger 150.92 million parameter uncertainty ensemble. It is not evidence of a released, general-purpose robot-arm model or of access to the real Jenga checkpoint.

### Cited Findings

- The repository separates Dubins-car and IsaacLab block-plucking implementations. Its pretrained-model link is under the IsaacLab workflow. The README describes a `dreamer.pt` checkpoint and reachability filter. The actual archive directory inspected on 7 October instead contains `latest.pt` and `filter/model/` with actor and two critic checkpoints at 20 saved training steps. [Pinned README](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/README.md), [Public archive](https://drive.google.com/file/d/1RddRw3eVUhufuUdq_BAThjwvO1fsmTeM/view)
- HTTP range metadata gives the archive size as **2,178,778,955 bytes**. Its central directory lists `latest.pt` as **1,984,280,647 bytes uncompressed**, or **1,826,797,925 compressed bytes**. These are measured archive metadata, not VRAM requirements. [Audited public archive](https://drive.google.com/file/d/1RddRw3eVUhufuUdq_BAThjwvO1fsmTeM/view)
- The paper differentiates simulated manipulation, with two RGB cameras at 128 by 128, from real Jenga, with two cameras at 256 by 256. Both use seven proprioceptive and seven action dimensions. [Paper Appendix A.1](https://arxiv.org/html/2505.00779#A1.SS1)
- Source camera definitions are `front_cam` and `wrist_cam`, each 128 by 128. The encoder concatenates them into six channels and combines their encoded representation with a five-layer, width-1024 MLP over end-effector position and quaternion. Camera CNN channels are 32, 64, 128, 256 and 512, with kernel four, stride two and channel LayerNorm. Its flattened visual embedding is 8,192 dimensional; the combined embedding is 9,216 dimensional. [Camera configuration](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/takeoff/config/franka/takeoff_joint_pos_env_cfg.py), [Network implementation](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/networks.py)
- The recurrent state has 512 deterministic dimensions and 32 categorical stochastic variables with 32 classes each. Flattened stochastic plus deterministic features have 1,536 dimensions. The dynamics hidden width is 512, with one GRU update per model step. Reward, continuation and failure heads each use two hidden layers of width 512. Reward has a 255-bin output; the other two heads are binary. [Configuration](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/configs.yaml), [World-model construction](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/models.py)
- Checkpoint tensor shapes independently match that source configuration: first image convolution `[32, 6, 4, 4]`, proprioception input matrix `[1024, 7]`, RSSM posterior matrix `[512, 9728]`, decoder input matrix `[8192, 1536]`, and final camera decoder layer `[32, 6, 4, 4]`. Thus the configuration was checked against the published checkpoint, rather than assumed from repository defaults. [Audited public archive](https://drive.google.com/file/d/1RddRw3eVUhufuUdq_BAThjwvO1fsmTeM/view)

### Inferences

- These dimensions identify the released asset as compatible with the simulation architecture, rather than the higher-resolution real-robot architecture. Its actual dataset provenance, training seed and achieved evaluation score are not established by tensor shapes.
- Deployment in a new task still requires matching camera preprocessing, proprioceptive definitions, action scaling and dynamics. A compatible architecture does not establish reliable predictions for new objects or contact situations.

### Gaps

- The archive does not include a saved configuration or training manifest alongside the checkpoint in the inspected directory. The two root checkpoint keys are `agent_state_dict` and `optims_state_dict`.
- Tensor values, numerical prediction quality and simulator loading were not tested. Only metadata was downloaded: the final 65,557 bytes of the archive and a 524,288-byte range beginning at the checkpoint entry. The outer DEFLATE stream yielded its inner PyTorch ZIP metadata. An explicit `pickletools` opcode interpreter extracted tensor descriptions without importing pickle globals or executing reducers. No `pickle.load`, `torch.load` or checkpoint code execution occurred.
- Audit artifacts are in `/tmp/unisafe-compute-20261007/model-audit/`. The 118,566-byte `data.pkl` metadata has SHA-256 `ba9b37d2ec2909e6b0883f461098a169a887d73138b97625e8e424bd432850b0`. This is a metadata checksum, not an integrity checksum for the un-downloaded full checkpoint.

## How large are its components and what does inference need?

### Takeaway

The world model itself is compact enough for ordinary single-GPU inference. The separate uncertainty estimator dominates parameter storage and dense computation. The complete model-plus-ensemble weights require about 728 MiB in FP32, but actual peak VRAM also includes activations, runtime allocations and any simulator.

### Cited Findings

The following counts were reconstructed from reviewed network classes on PyTorch's meta device, then checked against the public checkpoint tensor shapes. Counts exclude optimizer state, duplicate aliases and the separate reachability filter. Sources are the [network definitions](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/networks.py), [model construction](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/models.py) and [audited checkpoint](https://drive.google.com/file/d/1RddRw3eVUhufuUdq_BAThjwvO1fsmTeM/view).

| Component | Parameters | FP32 weights, MiB |
|---|---:|---:|
| Image and proprioception encoder | 7,002,048 | 26.71 |
| RSSM, including observation posterior | 8,400,896 | 32.05 |
| Image and proprioception decoder | 21,165,005 | 80.74 |
| Reward head | 1,181,439 | 4.51 |
| Continuation head | 1,051,137 | 4.01 |
| Failure head | 1,051,137 | 4.01 |
| **World model total** | **39,851,662** | **152.02** |
| **Five-member uncertainty ensemble** | **150,918,235** | **575.71** |
| **World model plus ensemble** | **190,769,897** | **727.73** |

- The full Dreamer agent adds a 1,057,806-parameter actor, a 1,181,439-parameter value network and another copy for the target value network. Total unique agent parameters are **194,190,581**, plus a two-float reward EMA buffer. The state dictionary repeats world-model and behaviour references; summing every named state entry would incorrectly count aliases several times. Storage identifiers in the checkpoint confirm the unique total. [Dreamer construction](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/dreamer.py), [Audited checkpoint](https://drive.google.com/file/d/1RddRw3eVUhufuUdq_BAThjwvO1fsmTeM/view)
- Each uncertainty network receives 1,536 latent features plus seven actions, giving input width 1,543. Its successive outputs have widths 1,543, 3,086, 4,629, 1,543 and 1,024. The final output is mean and log standard deviation for the next 512-dimensional deterministic state. Five independently parameterized members include LayerNorm after their first four layers. The exact total is `5 * sum(input * output + output + normalization_parameters) = 150,918,235`. [Uncertainty wrapper](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/uncertainty.py), [Ensemble network](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/ensemble/penn.py), [Ensemble linear layers](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/ensemble/ensemble_linear.py)
- The Dreamer action-selection path uses an encoded observation, recurrent update and actor; it does not reconstruct images. However, the supplied teleoperation wrapper reconstructs images in `forward_latent` to record reconstruction losses. Those diagnostics add decoder work to that demonstration. [Dreamer action path](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/dreamer.py), [Teleoperation wrapper](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/teleop_dreamer/filter_with_dreamer_failure.py)
- The authors report latent representation and uncertainty forwarding in under 0.1 seconds on an unspecified standard desktop, and approximately 2 to 4 GB of VRAM for the ensemble. This is limited author evidence, not a reproducible 5090 benchmark, an end-to-end simulator rate or a rollout-planning throughput result. [Paper Appendix A.3](https://arxiv.org/html/2505.00779#A1.SS3)

### Inferences

- An agent using latent predictions need not decode every imagined step. Omitting the decoder from inference leaves 18,686,657 world-model parameters, or 71.28 MiB of FP32 weights, before uncertainty and agent components.
- The ensemble is about 79% of world-model-plus-ensemble parameter storage. Its dense layers require 150,751,100 multiply-accumulates per evaluated state-action pair, excluding normalization, nonlinearities and divergence computation. The RSSM prior's dense layers require about 2,887,168 multiply-accumulates per step. Calling uncertainty on every imagined branch can therefore be a material additional cost. These are source-derived operation counts, not latency measurements.
- Batched imagination should reuse one frozen world model and ensemble across candidate planning procedures. Branch states and activations scale with the batch; weights need not be copied per evolutionary candidate.

### Gaps

- No measured inference batch size, peak VRAM, latency distribution or 5090 throughput is available from this audit. The report's compact parameter totals cannot establish those quantities.
- The separate SAC reachability actor and critics are additional networks. Their released files are approximately 6.3 MB each, and source configurations use four width-512 hidden layers. They are not required merely to query the world model, although reproducing UNISafe's full filter requires them. [Reachability configuration](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/reachability/config.yaml)

## What does extended training require beyond the checkpoint?

### Takeaway

Continued world-model training, uncertainty-only fitting and training the original task policy are distinct workloads. The released script's `model_only` name is misleading for compute budgeting: its current call path still trains the Dreamer behaviour component.

### Cited Findings

- Published visual-manipulation training uses batch 16, sequence length 64, Adam at `1e-4` and 200,000 iterations. The repository defaults use the same batch and sequence lengths, FP32 and imagination horizon 15. [Paper Appendix A.1](https://arxiv.org/html/2505.00779#A1.SS1), [Training configuration](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/configs.yaml)
- `train_model_only` calls `_train`, which trains the world model, optional uncertainty ensemble and task behaviour. A genuinely dynamics-only continuation would need a deliberately separated runner. The entry point constructs IsaacLab even when an offline dataset is supplied. [Dreamer training methods](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/dreamer.py), [Training entry point](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/train_dreamer.py)
- The uncertainty-only method freezes the world model, obtains latents from observations, and trains ensemble members one at a time within each batch. The README suggests an additional 200,000 iterations for this phase. [World-model training methods](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/models.py), [Uncertainty training loop](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/uncertainty.py), [README](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/README.md)
- The checkpoint contains optimizer containers for model, actor, value and ensemble. Only the ensemble optimizer has populated state: 50 parameter entries with 301,836,520 float elements across moments and step scalars. The other three optimizer states are empty. The script's pretrained-load branch restores agent weights but does not restore optimizer dictionaries. [Audited checkpoint](https://drive.google.com/file/d/1RddRw3eVUhufuUdq_BAThjwvO1fsmTeM/view), [Loading implementation](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/train_dreamer.py)

### Inferences

- Loading these weights supports fine-tuning, but does not establish exact continuation of the original world-model optimizer trajectory. The populated ensemble state is consistent with a final uncertainty-fitting phase; no training manifest confirms that history.
- FP32 weights, gradients and two Adam moments alone require approximately `16 * parameter_count` bytes: 0.594 GiB for the world model and 2.249 GiB for the ensemble, or 2.843 GiB combined. Training activations, reconstruction losses, temporary optimizer allocations, behaviour imagination and the simulator come on top. These lower bounds cannot be presented as required VRAM.
- A full source batch contains 1,024 time positions and 2,048 camera frames. Its FP32 image input tensor alone is 384 MiB; convolutional forward and backward activations are substantially larger. Reduced batches, freezing the encoder or using cached latents change the workload and the scientific claim.
- If continued learning changes the latent representation or dynamics, the old ensemble and reachability networks cannot simply be assumed to retain their original calibration and meaning. Those components need a deliberate refitting and evaluation decision.

### Gaps

- No author-reported training wall-clock, training GPU specification or verified peak training memory was found in the audited sources. Duration on a 5090 requires a representative warmed-up training measurement for the chosen continuation scope.
- This audit does not certify compatibility of the original IsaacLab/PyTorch software environment with a 5090. Runtime compatibility is a separate setup concern from the model's parameter and activation requirements.
