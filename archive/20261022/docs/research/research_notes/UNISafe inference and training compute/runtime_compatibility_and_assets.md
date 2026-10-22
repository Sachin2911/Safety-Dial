# UNISafe runtime compatibility and released assets

Audit dated 7 October 2026. Source checkout: `CMU-IntentLab/UNISafe`, `isaaclab`
branch, commit `ab0fdf87f38edefa48b54a291ec5b7156c17b6e0`. This note separates
source evidence, metadata measurements and inference. No simulator, training job,
environment installation or complete external dataset download was performed.

## Will the released stack run directly on an RTX 5090?

### Takeaway

**The original environment is not a supported Blackwell deployment.** Modern model-only
inference is a plausible separate route, but running the original robot simulation
requires a compatible Isaac Sim/Isaac Lab stack and code migration. A runtime or VRAM
estimate must not be mistaken for a successful compatibility test.

### Cited Findings

- **Verified source:** the README explicitly identifies Isaac Sim 4.2.0 and describes
  its stack as stale. It suggests import-path changes for newer Isaac Sim, but provides
  no verified RTX 5090 migration result. Its instructions also retain `latent_safety`
  paths and a notebook name that differs from the checked-out filename.
  [Pinned README](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/README.md)
- **Verified source:** `environment.yaml` pins Python 3.10.16, pip `torch==2.4.1`,
  CUDA 12.1 runtime components and cuDNN 9.1. It also contains conda PyTorch 1.12.1 CPU,
  CUDA 11.6, a corresponding PyTorch3D build, and CPU torchvision. These are overlapping
  historical package specifications, not a clean modern GPU compatibility declaration.
  [Pinned environment](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/environment.yaml)
- **Official compatibility evidence:** PyTorch 2.7 introduced Blackwell support and
  CUDA 12.8 wheels, updating supporting libraries and Triton. Therefore, the released
  PyTorch 2.4.1/CUDA 12.1 environment should not be recreated unchanged for a 5090.
  Use a Blackwell-capable build, not merely a newer system driver.
  [PyTorch release announcement](https://pytorch.org/blog/pytorch-2-7/)
- **Official NVIDIA support evidence:** NVIDIA staff explicitly states that even
  Isaac Sim 4.5's renderer lacks Blackwell support and that a driver change does not
  repair that architecture mismatch; it recommends Isaac Sim 5.1 or later. This
  supports treating the older 4.2 dependency as an obstacle, rather than assuming a
  5090 can run it because it has ample VRAM.
  [NVIDIA support response](https://forums.developer.nvidia.com/t/isaac-sim-4-5-not-working-on-windows-11-via-wsl2/366463)
- **Official migration evidence:** Isaac Lab release notes identify version 1.4.1 as
  the last compatible with Isaac Sim 4.2. Later releases introduce breaking changes.
  The Isaac Sim 5.0 transition includes Python 3.11, PyTorch 2.7/cu128 and Blackwell
  rendering fixes, with changes to simulation settings and some asset paths.
  [Isaac Lab release notes](https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html)

### Inferences

A practical distinction is between two work packages:

1. **Model-only use:** instantiate the neural components in a modern PyTorch runtime
   and feed recorded observations or latent batches. This can avoid Isaac Sim's
   renderer and installation footprint. Preserving architecture, preprocessing,
   recurrent-state handling and checkpoint loading is still required.
2. **Closed-loop simulation:** migrate the environment, cameras, controller, task
   registration and wrapper to a mutually compatible Sim/Lab version. Matching camera
   appearance, physics and action semantics matters for checkpoint usefulness after
   the code runs successfully.

Checkpoint tensors are not intrinsically tied to an old GPU. Reusing them after a
framework update is plausible, but this audit did not execute a forward pass or compare
outputs across versions. Full-object checkpoint serialization, package import paths and
changed defaults can add work even when the layer dimensions match. A successful load
also does not establish that a visually or physically changed simulator stays within
the model's training distribution.

### Gaps

No end-to-end RTX 5090 result was obtained. The README's suggestion that migration needs
only import edits is not sufficient evidence for identical rendering, dynamics or
performance. The exact Lab revision used by the authors was not established here.

## What hardware requirements and assets are actually verified?

### Takeaway

The repository provides a concrete simulated manipulation starting point, with public
weights, data and small examples. The released assets should not be described as a
verified physical-Jenga package. Simulator system requirements also describe an
installation baseline, not the measured memory consumption of UNISafe.

### Cited Findings

**Official hardware requirements.** The following are NVIDIA's published specifications,
not a benchmark of this model:

| Version/document | CPU cores | Host RAM | Storage | GPU VRAM | Linux driver listed |
| --- | --- | --- | --- | --- | --- |
| Isaac Sim 5.1 minimum | 4 | 32 GB | 50 GB SSD | 16 GB | 580.65.06 |
| Isaac Sim 5.1 good | 8 | 64 GB | 500 GB SSD | 16 GB | 580.65.06 |
| Isaac Sim 6.0 minimum | 4 | 32 GB | 50 GB SSD | 16 GB | 580.95.05 |

The pages identify additional RAM/VRAM needs for Isaac Lab training and rendering.
The 5.1 and 6.0 tables name RTX 4080/5080/RTX PRO 6000 tiers. They do not measure
UNISafe on a 5090. [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html);
[Isaac Sim 6.0 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html)

**Original requirements limitation.** The requested 4.2 requirements URL returned HTTP
404. The adjacent official 4.1 archive lists 4 cores, 32 GB RAM, 50 GB SSD and 8 GB
VRAM minimum, but those numbers are not presented here as independently verified 4.2
requirements. [Archived 4.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/installation/requirements.html)

**Verified public dataset metadata.** Tiny HTTP range requests checked archive sizes
and ZIP central directories, without downloading trajectory payloads:

| Release | Download bytes | NPZ files | Extracted NPZ archive bytes |
| --- | ---: | ---: | ---: |
| `data_all.zip` | 9,232,413,771 | 3,438 | 9,290,052,655 |
| `data_success_only.zip` | 3,966,007,341 | 1,369 | 3,998,367,339 |

These are approximately 9.23 GB and 3.97 GB downloads in decimal units. The final column
is disk space occupied by the extracted `.npz` files, not fully decoded image tensors.
Metadata sources: [complete release](https://drive.google.com/file/d/1gaLfQrR53Kiksd-uXRG-WqOSnPsipNya/view);
[success-only release](https://drive.google.com/file/d/14Ofq7gCEnPMZXY9K5lANNzxynyBfBHST/view).

**Metadata observations:** the full archive has 2,862 `failure`, 281 `expert`, and 295
`success` filename prefixes. Success-only has 1,195 `success` and 174 `expert` prefixes.
Only 469 basenames occur in both. These are filename classifications, not verified
outcome labels or independent source-trajectory counts. Do not assume success-only is
a strict content subset. No train/test or licence files appear in either directory;
all non-directory entries are NPZ files. [Complete release](https://drive.google.com/file/d/1gaLfQrR53Kiksd-uXRG-WqOSnPsipNya/view);
[success-only release](https://drive.google.com/file/d/14Ofq7gCEnPMZXY9K5lANNzxynyBfBHST/view)

**Verified local sample headers:** `log/failure_1.npz`, `failure_2.npz` and
`failure_3.npz` contain 79, 83 and 61 steps respectively. Each has two 128x128 RGB
arrays (`front_cam`, `wrist_cam`) stored as float32, 7-dimensional actions,
`eef_pos`, `eef_quat`, and success/failure/episode fields. Their combined compressed
size is about 7.57 MB. Headers were inspected using ZIP/NPY metadata only; pixel
values, label correctness and dynamic replay were not tested.
[Pinned samples](https://github.com/CMU-IntentLab/UNISafe/tree/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/log)

**Task identity:** the `isaaclab` branch is explicitly block plucking. The paper's
physical Jenga experiment uses two 256x256 cameras at 15 Hz and a real Franka Research
3, with 720 collected training trajectories. Therefore, the repo's sample schema
should not silently become a physical-Jenga compute estimate. The parallel checkpoint
audit supplies the exact released model dimensions and ensemble tensors.
[README](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/README.md);
[paper §6.3 and Appendix D.3](https://arxiv.org/html/2505.00779)

**Availability and licensing:** the README links a world-model/reachability-filter
archive and describes optional uncertainty-ensemble fine-tuning. Source code is MIT
licensed, copyright 2026 Junwon Seo. No separate public dataset/weights licence was
identified in the inspected release metadata; the code licence should not be silently
assigned to every external asset.
[Weights release](https://drive.google.com/file/d/1RddRw3eVUhufuUdq_BAThjwvO1fsmTeM/view);
[pinned code licence](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/LICENSE)

### Inferences

Keeping the full dataset ZIP and its extracted NPZ files requires approximately
18.52 GB before model weights, simulator assets, caches, logs or decoded replay buffers.
This arithmetic is useful for disk provisioning but provides no host-RAM bound.
The two RGB streams in the small examples are heavily compressed, so their disk sizes
would be a poor proxy for decoded training memory.

For combined simulation and learning, 64 GB host RAM is a reasonable provisional
provisioning target relative to NVIDIA's published tiers. It is not a guarantee that
an eager loader can hold the full dataset. Host buffering strategy must be checked.

### Gaps

The complete datasets were not decoded, so total frames, image shapes throughout the
release, label frequencies, independent splits and duplication remain unverified.
No separate physical-Jenga checkpoint or hardware dataset was verified. No public
asset checksum or immutable Drive revision was established.

## Which costs and checks must accompany a practical compute estimate?

### Takeaway

Separate neural inference, imagined rollouts, simulator rendering and training/data
handling. The same GPU can be adequate for an isolated model call while a simultaneous
camera simulation and training run has a different memory and latency profile.

### Cited Findings

- The task configuration provides front and wrist RGB cameras and renders on the
  control schedule. The wrapper has code paths that transfer observations to CPU/NumPy.
  Teleoperation includes additional CPU transfers, visualisation and video encoding.
  These operations are outside a latent-transition microbenchmark.
  [Task configuration](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/takeoff/takeoff_env_cfg.py);
  [wrapper](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamer_wrapper.py);
  [teleoperation script](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/teleop_dreamer/filter_with_dreamer_failure.py)
- Isaac Lab's installation guide warns that remotely hosted assets can take minutes to
  load and recommends caching. Startup/asset/shader preparation should therefore be
  kept separate from steady-state inference throughput.
  [Official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

### Inferences

The cost of acquiring two images, preprocessing them, updating the recurrent posterior,
querying many imagined branches, and invoking an uncertainty ensemble should be counted
explicitly. Optional image decoding, plotting and video export should not be included
in a claim about the minimum cost of latent planning unless the intended agent uses
them. Conversely, a model-only timer should not be reported as robot control latency.

Extending training also has distinct meanings: continuing the latent model, fitting
the uncertainty ensemble, learning the task actor/critic, and training the safety filter.
Their costs and compatibility requirements differ. The companion model and compute
audits should determine which the released entry points actually execute.

### Gaps

No measured 5090 inference time, training throughput, rendering throughput or peak
combined VRAM is produced by this note. The remaining concrete check is a small,
authorised compatibility and profiling run after the intended model-only or simulator
scope is chosen. Scratch evidence is under
`/tmp/unisafe-compute-20261007/runtime-assets/`: public preview HTML, HTTP size headers,
ZIP metadata ranges and parsed metadata summaries. No secrets were read or printed.
