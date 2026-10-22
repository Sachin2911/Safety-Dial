# UNISafe inference and training compute

**UNISafe's released simulated-manipulation model looks suitable for single-GPU research,
including inference and bounded fine-tuning.** A 32 GB RTX 5090 is a sensible machine on
which to establish its actual resource requirements, after updating the software stack.
The released world model has about 40 million parameters, although its uncertainty
ensemble adds about 151 million. The authors provide limited desktop inference evidence;
they do not provide a reproducible 5090 throughput or training-duration benchmark.
Our audit checked source code, public archive metadata and checkpoint tensor dimensions,
without running the model. The main remaining uncertainties are the cost of many imagined
branches, training activations, and concurrent camera simulation.

This assessment is dated 7 October 2026 and uses the official `isaaclab` source revision
`ab0fdf87f38edefa48b54a291ec5b7156c17b6e0`.

## The released model is manageable, while uncertainty adds substantial work

The public checkpoint matches the **simulated block-plucking configuration with two
128 × 128 RGB cameras**, seven proprioceptive inputs and seven action outputs. We
confirmed its tensor shapes against the source, rather than inferring the architecture
solely from default settings. This does not verify access to the paper's physical-Jenga
checkpoint, which uses two 256 × 256 cameras, or establish usefulness in a new task.
([Public checkpoint archive](https://drive.google.com/file/d/1RddRw3eVUhufuUdq_BAThjwvO1fsmTeM/view);
[paper, Appendix A.1](https://arxiv.org/html/2505.00779#A1.SS1))

| Checkpoint component | Parameters |
| --- | ---: |
| World model, including encoder, dynamics, decoder and outcome heads | 39,851,662 |
| Five-member uncertainty ensemble | 150,918,235 |
| Complete Dreamer agent, including task actor and value networks | 194,190,581 |

These are unique parameter counts: repeated references in the saved state dictionary
were not counted twice. The separate reachability actor and critics are additional.
The underlying implementation and checkpoint metadata agree on the totals.
([Model construction](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/models.py);
[uncertainty implementation](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/uncertainty.py);
[checkpoint](https://drive.google.com/file/d/1RddRw3eVUhufuUdq_BAThjwvO1fsmTeM/view))

For inference, the strongest published resource statement is **under 0.1 seconds to
forward the latent representation and associated uncertainty**, using approximately
**2-4 GB of VRAM for the ensemble** on a desktop. The paper does not identify the GPU,
batch size, timing boundaries or latency distribution. This establishes a useful
feasibility reference, but cannot establish robot-loop speed or planning throughput.
([Paper, Appendix A.3](https://arxiv.org/html/2505.00779#A1.SS3))

The agent first encodes an observation and updates its recurrent state. Thereafter,
imagined actions can advance latent dynamics without reconstructing images. The supplied
teleoperation demonstration additionally decodes observations for reconstruction
diagnostics. A latent planner can omit those diagnostics while retaining the model's
predictive state; reproducing the complete UNISafe filter additionally requires its
uncertainty and reachability computations.
([Policy implementation](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/dreamer.py);
[demonstration](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/teleop_dreamer/filter_with_dreamer_failure.py))

For the proposed research, a hypothetical search over 256 sequences of length 15 requires
**3,840 latent transitions per iteration**. Scoring each transition with five uncertainty
members requires 19,200 member predictions. These counts are arithmetic, not runtime
estimates. Batching can reuse one frozen model across planning candidates, while successive
time steps remain dependent. The authors' single-forward figure cannot simply be
multiplied by the transition count to forecast this workload.

## Training scope matters more than the name of the training command

The paper specifies **200,000 visual-model training iterations**, batch size 16 and
sequence length 64. The released configuration uses FP32, Adam and a task-policy
imagination horizon of 15. Crucially, its `model_only: true` route means offline training:
the call chain also updates the uncertainty ensemble and the task actor/critic.
([Paper, Appendix A.1](https://arxiv.org/html/2505.00779#A1.SS1);
[configuration](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/configs.yaml);
[training methods](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/dreamer.py))

Each such batch contains 1,024 observed time positions. The task actor starts imagined
trajectories from those posterior states, adding **15,360 imagined transitions per
update**, with differentiable computation. These are reused training observations and
model queries, not 15,360 newly collected simulator steps.
([Behaviour training](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/models.py#L405))

Consequently, full model adaptation, uncertainty-only fitting and learning a new planner
have different costs. Full adaptation retains image reconstruction and its gradients.
The supplied ensemble-only routine freezes the world model but still computes latent
targets from image sequences. A carefully designed cache could avoid repeated encoding.
Training only selected dynamics modules requires a deliberately separated update path.
([World-model updates](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/models.py#L130))

The checkpoint supports starting from pretrained weights. It does not establish exact
continuation of the original optimizer trajectory: world-model, actor and value optimizer
states are empty, and the loader restores weights without restoring optimizer state.
Only the ensemble optimizer has populated state in this archive.
([Checkpoint](https://drive.google.com/file/d/1RddRw3eVUhufuUdq_BAThjwvO1fsmTeM/view);
[loading code](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/train_dreamer.py))

Changing the representation or dynamics can change the meaning of existing failure
readouts, uncertainty predictions and reachability values. Their validity needs checking
after adaptation, with refitting where necessary. For the agent-focused idea, initially
sharing a frozen model makes both the scientific comparison and the resource accounting
simpler.

No verified training-duration measurement was found. The following are **conditional
budget calculations**, using hours = updates / (updates per second × 3,600).
The rates are hypothetical and are neither 5090 predictions nor performance bounds.

| Update budget | At 0.25 updates/s | At 1 update/s | At 5 updates/s |
| --- | ---: | ---: | ---: |
| 10,000 | 11.1 hours | 2.78 hours | 0.56 hours |
| 50,000 | 55.6 hours | 13.9 hours | 2.78 hours |
| 200,000 | 222.2 hours | 55.6 hours | 11.1 hours |

A measured rate must use the selected modules and declared batch/sequence dimensions.
Ten thousand or fifty thousand updates are possible spending limits, not evidence that
a new task will be learned. Collection, evaluation, repeated seeds and checkpointing add
to these totals.

## Memory arithmetic supports one GPU without establishing peak usage

World-model and ensemble FP32 weights occupy **0.763 GB combined**. Using the conventional
four FP32 tensors for weights, gradients and Adam's two moments gives approximately
**3.05 GB**, before activations and temporary storage. These figures follow directly from
the checkpoint-confirmed parameter count; they are not measured VRAM requirements.
([Network definitions](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/networks.py);
[checkpoint](https://drive.google.com/file/d/1RddRw3eVUhufuUdq_BAThjwvO1fsmTeM/view))

At the published batch shape, the two camera streams alone require
16 × 64 × 2 × 128 × 128 × 3 × 4 bytes, or **384 MiB**, as FP32 inputs. The first convolution
output occupies another **512 MiB**. Saved encoder/decoder activations, recurrent graphs,
policy imagination, optimizer temporaries and rendering allocations come on top.
At 256-pixel resolution those two example tensors grow to 1.5 GiB and 2 GiB respectively,
and the architecture also changes.

Our engineering allowance is **12-16 GB VRAM for model-only inference development** with
small query batches and no concurrent simulator renderer. **24-32 GB is a sensible
training and larger-batch calibration target**, especially if batch size can be adjusted.
Neither is a tested minimum or proof that the untouched full training configuration
fits. Mixed precision also does not automatically halve optimizer storage.

The public model archive is **2.18 GB**, and contains a checkpoint named `latest.pt`,
despite the README's `dreamer.pt` example. The full and success-only data downloads are
**9.23 GB and 3.97 GB** respectively. Keeping the full ZIP and extracted NPZ files consumes
approximately **18.52 GB** before simulator assets, caches and results. Compressed disk
size does not bound decoded replay-buffer RAM.
([Model archive](https://drive.google.com/file/d/1RddRw3eVUhufuUdq_BAThjwvO1fsmTeM/view);
[complete data](https://drive.google.com/file/d/1gaLfQrR53Kiksd-uXRG-WqOSnPsipNya/view);
[success-only data](https://drive.google.com/file/d/14Ofq7gCEnPMZXY9K5lANNzxynyBfBHST/view))

## A 5090 needs a modern runtime before these estimates can be tested

The original environment targets **Isaac Sim 4.2 and PyTorch 2.4.1**, with historical,
overlapping dependency pins. PyTorch 2.7 introduced Blackwell support and CUDA 12.8 wheels.
NVIDIA states that even Isaac Sim 4.5's renderer lacks Blackwell support and recommends
5.1 or later. The released stack should therefore not be recreated unchanged on a 5090.
([UNISafe environment](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/environment.yaml);
[PyTorch release](https://pytorch.org/blog/pytorch-2-7/);
[NVIDIA support](https://forums.developer.nvidia.com/t/isaac-sim-4-5-not-working-on-windows-11-via-wsl2/366463))

Model-only inference can use a modern PyTorch runtime with recorded observations.
Closed-loop simulation additionally needs compatible Isaac Sim/Lab versions and migration
of cameras, controllers and wrappers. Updated rendering or physics could affect checkpoint
predictions even after imports and tensor shapes work. The existing offline training
entry point also constructs IsaacLab, so decoupling it requires implementation.
([Training entry point](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/train_dreamer.py);
[IsaacLab release notes](https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html))

For host provisioning, Isaac Sim 5.1 lists **32 GB RAM, 50 GB SSD and 16 GB GPU memory**
as installation minima, with 64 GB RAM in its higher tier. **64 GB host RAM is a reasonable
initial allowance for combined simulation and learning**, subject to checking the loader;
it is not assurance that an eagerly decoded full dataset fits.
([Official requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html))

## Conclusion

The most useful next measurement would separate observation encoding, batched latent
rollouts and uncertainty scoring, then profile several hundred representative training
updates and a handful of complete episodes. This would reveal whether the research budget
is governed by neural computation, camera simulation or repeated evolutionary evaluation.
It would also establish which optional diagnostics can be removed without changing the
agent being studied.

This audit supports investigating UNISafe on one GPU. It does not require adopting its
complete safety-filter pipeline or expanding the proposed research into model retraining.
No GPU benchmark, installation, simulator execution or training run was performed.
The evidence trail is preserved in the
[model and checkpoint audit](../research_notes/UNISafe%20inference%20and%20training%20compute/model_and_checkpoint_audit.md),
[inference and training calculations](../research_notes/UNISafe%20inference%20and%20training%20compute/inference_and_training_estimates.md)
and [runtime and asset audit](../research_notes/UNISafe%20inference%20and%20training%20compute/runtime_compatibility_and_assets.md).
