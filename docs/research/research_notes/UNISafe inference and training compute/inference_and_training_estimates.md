# UNISafe inference and continued-training compute

Research date: 7 October 2026. This is a source audit and arithmetic estimate, not a
benchmark. No GPU rental, simulator run, model training or full checkpoint download was
performed for these notes. The source inspected is the official `isaaclab` branch at
`ab0fdf87f38edefa48b54a291ec5b7156c17b6e0`. Exact source-derived parameter counts were
provided by the parallel model audit and confirmed against safely inspected tensor
metadata from the public checkpoint. That audit used partial downloads and did not
execute a checkpoint pickle.

## What must run at inference?

### Takeaway

There is encouraging published evidence for desktop inference, but it has limited scope.
The authors report a latent prediction plus uncertainty calculation in under 0.1 seconds,
with approximately 2-4 GB VRAM used by the ensemble. This does not specify RTX 5090
performance, batched planning throughput or end-to-end IsaacLab speed.
[Paper, Appendix A.3](https://arxiv.org/html/2505.00779#A1.SS3)

### Cited Findings

- A real observation passes through preprocessing, the image/proprioception encoder and
  RSSM posterior update. The task actor then receives the resulting latent feature.
  This ordinary policy path does not decode pixels.
  [Dreamer policy code](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/dreamer.py#L202)
- Imagining a proposed action calls the RSSM transition, without generating a new camera
  observation or re-encoding pixels. Reward, continuation and failure heads can score
  latent features. The code's imagined-behaviour training uses precisely this latent
  transition path.
  [Imagination code](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/models.py#L465)
- The supplied teleoperation filter evaluates the uncertainty ensemble, advances the
  proposed action one latent step, evaluates a safety actor and twin critics, and may
  substitute the safety actor's action. It additionally computes a failure diagnostic
  and re-evaluates uncertainty for the executed action.
  [Runtime filter](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/teleop_dreamer/filter_with_dreamer_failure.py#L374)
- Its `forward_latent` helper also invokes the pixel decoder to report reconstruction
  loss. Thus the released demonstration contains work a latent planning API need not
  perform. Removing diagnostics would change the demonstrated execution workload.
  [Observation update and diagnostics](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/teleop_dreamer/filter_with_dreamer_failure.py#L230)
- The supplied uncertainty architecture has five large MLP members; it is not merely a
  tiny scalar head. The source audit counts approximately 151 million parameters for the
  ensemble, compared with 39.85 million for the world model.
  [Uncertainty implementation](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/uncertainty.py)

### Inferences

For the proposed evolved planner, distinguish four timing measurements: observation
encoding, batched latent transitions, uncertainty/readout scoring, and complete simulator
decisions. An agent may encode the current observation once and then investigate many
latent futures. It can share one frozen model across candidates if the implementation
batches their requests. Launching separate GPU processes can duplicate model weights.

A planning budget of N candidate sequences and horizon H requires N × H latent
transitions per iteration, before any extra stochastic samples or search refinement.
For N = 256 and H = 15, that is 3,840 transitions. Scoring every transition with five
ensemble members entails 19,200 member predictions. Neither number is a duration.
Sequences can be batched, while their successive time steps remain dependent.

The paper's sub-0.1-second figure must not be multiplied by 3,840 to forecast this example:
its original batch size, hardware and boundaries are unspecified, and batched operation
has different throughput. Conversely, that figure does not establish that thousands of
predictions fit inside a real-time decision budget.

For world-model-only research, the UNISafe reachability actor/critics are optional unless
used by the experimental method or baseline. Excluding its uncertainty module is also a
scientific design choice, not a free optimization that preserves the complete method.

### Gaps

The inspected paper, project page, repository, public issue list and upstream README did
not expose named inference hardware, latency percentiles, batch scaling, training GPU
hours or peak memory for the complete pipeline. Video playback FPS and the simulator
time step are not throughput benchmarks. The repository contains logging code and three
sample trajectories, not an accessible measured training-rate table.

## What does continued training actually include?

### Takeaway

A modest adaptation budget is plausible, but the selected modules determine the cost.
The released offline `model_only` route trains more than the world model. It also trains
the task actor/critic and, by default, the uncertainty ensemble.
[Training call chain](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/dreamer.py#L143)

### Cited Findings

- The visual-model paper configuration uses two 128 × 128 RGB images in simulation,
  two 256 × 256 images on hardware, batch size 16, sequence length 64, and 200,000
  training iterations. The manipulation latent feature combines 512 deterministic and
  1,024 categorical coordinates.
  [Paper, Appendix A.1](https://arxiv.org/html/2505.00779#A1.SS1)
- The repository defaults use FP32, Adam, and imagination horizon 15.
  `train_model_only` invokes `_train`, whose task-behaviour update starts imagined
  trajectories from the posterior states of the observation batch.
  [Configuration](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/configs.yaml);
  [Actor training](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/models.py#L405)
- The world-model update differentiates reconstruction, reward, continuation, failure
  and KL losses. Its separate ensemble-only routine freezes image encoding and latent
  inference under `no_grad`, then fits the uncertainty model. It currently recomputes
  those latent targets rather than reading a persistent latent cache.
  [World-model and ensemble updates](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/models.py#L130)
- The README suggests another 200,000 iterations for ensemble fine-tuning; the separate
  reachability setup also specifies 200,000 updates. These are distinct workloads with
  different update costs.
  [Training guidance](https://github.com/CMU-IntentLab/UNISafe/tree/isaaclab#full-training-pipeline);
  [Reachability configuration](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/reachability/config.yaml)

### Inferences

There are several legitimate scopes:

1. **Frozen model, new planner:** no world-model gradients; pay for planner search and
   its real evaluations. This most directly isolates the proposed agent contribution.
2. **Full model adaptation:** fine-tune encoder, RSSM, decoder and outcome heads on new
   observations. Reconstruction remains a substantive training workload.
3. **Restricted dynamics adaptation:** freeze selected representation/readout components
   and train an explicitly declared subset. This requires implementation and objective
   design; it is not an existing guaranteed shortcut.
4. **Readout or uncertainty fitting:** fit failure labels or the ensemble on fixed
   representations. Cached latents could avoid repeated encoding, provided temporal
   context, preprocessing and data splits remain fixed.
5. **Policy or safety-filter learning:** train the task actor/critic or reachability
   modules while holding the world model fixed. This has additional imagined rollout
   and replay costs even if no images are decoded.

Changing the encoder or latent dynamics can invalidate previously fitted heads, ensemble
targets, uncertainty thresholds, cached latents and reachability policies. Their validity
must be checked after adaptation. A dynamics-only change can also change what previously
trained safety values mean. Fine-tuning the entire stack is consequently a different
budget from extending a single predictor.

At batch 16 and length 64, one update processes 1,024 observed time steps. The default
actor update adds 15 × 1,024 = 15,360 imagined transitions with differentiable computation.
These counts describe reuse and optimization workload, not newly collected simulator
experience. A smaller batch can reduce peak memory; changing sequence length also changes
the temporal learning problem and is not automatically an equivalent training run.

### Gaps

No verified extended-training schedule establishes how many updates repair a new hazard
or task. Ten thousand or fifty thousand updates are budget scenarios, not convergence
predictions. Existing weights may be unsuitable for changes to camera placement, object
appearance or dynamics despite fitting in memory. Data collection, rendering, transfer,
validation and repeated seeds must be budgeted separately.

## What hardware and time can reasonably be estimated?

### Takeaway

One modern GPU is a reasonable starting point. A 32 GB RTX 5090 is a credible target
for calibration of the simulated-task model and subsequent fine-tuning after software
compatibility work. This is a capacity judgment, not confirmation that the unmodified
training configuration fits or reaches a particular rate.

### Cited Findings

- Source-derived and checkpoint-confirmed counts for the two-camera 128-pixel
  configuration are 39,851,662
  world-model parameters and 150,918,235 uncertainty parameters. Excluding the decoder
  leaves 18,686,657 world-model parameters. These counts concern architecture, not
  measured allocation.
  [Model construction](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/models.py);
  [Networks](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/networks.py)
- UNISafe advertises the legacy Isaac Sim 4.2 stack. Blackwell support arrived in the
  PyTorch 2.7/CUDA 12.8 stack; the separate compatibility audit identifies IsaacSim
  migration as necessary for a 5090 setup. Memory capacity does not resolve this.
  [UNISafe installation](https://github.com/CMU-IntentLab/UNISafe/tree/isaaclab#installation);
  [PyTorch release](https://pytorch.org/blog/pytorch-2-7/);
  [IsaacLab releases](https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html)

### Inferences

For P parameters, FP32 weights require 4P bytes. A conventional FP32 Adam training
budget for weights, gradients and two moment tensors is approximately 16P bytes before
activations and temporary allocations. Thus the world model plus ensemble has:

| Quantity | Arithmetic storage |
| --- | ---: |
| World-model weights | 159.4 MB |
| Ensemble weights | 603.7 MB |
| Combined weights | 763.1 MB, or 0.711 GiB |
| Combined weights, gradients and Adam moments | 3.052 GB, or 2.843 GiB |

These are floors, not GPU requirements. The task actor, critics, target networks,
autograd activations, convolution workspaces, allocator reserves, model copies and
simulator allocations are additional. Mixed precision does not necessarily halve the
optimizer's persistent storage.

The image input alone is:

16 × 64 × 2 × 128 × 128 × 3 × 4 = 402,653,184 bytes = **384 MiB**.

The first 32-channel, 64 × 64 convolution output across these 1,024 times is another
**512 MiB**, before saved intermediates and gradients. At 256-pixel resolution the input
is **1.5 GiB** and that first convolution output is **2 GiB**. The full model also changes
with image resolution, so four times the pixels is not a complete memory estimate.

A **12-16 GB device is a reasonable model-only inference development target**, with
small query batches and no concurrent IsaacSim renderer. This is an estimate supported
by weight sizes and the authors' limited ensemble figure, not a tested minimum.
**24-32 GB is a more comfortable calibration target for substantial planning batches
and training**, particularly when reducing batch size is permitted. Concurrent rendering
or the 256-pixel hardware setup can change this assessment.

Wall-clock training must be measured per selected update type. The reproducible formula is
hours = updates / (updates per second × 3,600). The following rates are hypothetical,
not expected 5090 measurements or bounds:

| Update budget | 0.25 updates/s | 1 update/s | 5 updates/s |
| --- | ---: | ---: | ---: |
| 10,000 | 11.1 h | 2.78 h | 0.56 h |
| 50,000 | 55.6 h | 13.9 h | 2.78 h |
| 200,000 | 222.2 h | 55.6 h | 11.1 h |

Compare these rows only with a measured rate at the declared batch/sequence dimensions
and module set. Ensemble-only, pure world-model and full offline-agent rates are different.
A rate slower than 0.25 requires longer than the leftmost scenario.

A bounded calibration could measure warmed single-observation latency, latent batches
at representative horizons, uncertainty enabled/disabled, and several hundred selected
training updates, followed by a handful of complete simulated episodes. Reserve an hour
of measurement after installation as a planning allowance, not a guaranteed completion
time. Compilation, dependency fixes and dataset transfer are separate.

### Gaps

No GPU benchmark was run here. Exact peak VRAM, useful fine-tuning convergence,
batch-throughput curves and full-system latency remain unknown. A frozen-model
evolutionary study could spend much more on repeated agent evaluations than on model
training. Rental money should be computed from a current quote and measured workload,
not from the paper's model size alone.
