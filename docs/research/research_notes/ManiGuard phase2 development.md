# ManiGuard phase-two development log

Implementation began on 17 October 2026 following explicit authorization to implement
phase2.md fully and push regular updates. The full plan remains the scope. Conditional
main-study decisions will follow measured pilot evidence, not optimistic assumptions.

The initial repository is 0decedf8078b1a224a4672eee73738a19ed5683a on
research/maniguard-fruits. The original simulator queues are EXITED; the Jar policy server
is RUNNING and idle. The GPU is an RTX 3090 with 24 GB. Approximately 85 GB disk is free.
No phase-two physical episode has run at this checkpoint.

Implementation is split into a pure recurrent model, continuous-weight search and recurrent
PPO, a six-family registry and asset preparation, and shared feature/runtime integration.
The original split manifest and study outcomes remain unchanged. A structured requirements
ledger tracks the full plan, including all-family evaluation, ablations and publication.

The shared feature encoder will use frozen OpenAI CLIP ViT-B/32 through pinned OpenCLIP
3.2.0. Its checkpoint and projection hashes, implementation dependencies, timing and input
contract will be frozen before physical runs. This is a common representation, not a safety
predictor. Installing its four additive dependencies does not replace the simulator's
Torch, NumPy or Hugging Face packages.

## Integration checkpoint

The shared actor is 692 weights (64 input features, eight recurrent units, four three-way
heads). The actor, adapter, continuous-weight evolution/random search, recurrent PPO,
registry, completion contracts and protocol compiler are implemented. The complete
comparative pilot has not run. A combined integration/legacy regression run passed 169
tests before the new generated-source and baseline-routing checks were added; those
additional runtime checks also pass.

The official CLIP TorchScript archive required OpenCLIP's dedicated OpenAI loader with
QuickGELU and explicit fp32 conversion. An initial loader attempt failed before any
simulator work; its precise wall time was not recorded. The corrected measured encoder
check loaded in 4.749 seconds, performed one text and five two-image encodings in
0.482 seconds including synchronized inference/preprocessing/projection, and allocated
628,513,792 peak GPU bytes. This is a saved-image smoke check, not full simulator
throughput. All encoder parameters are frozen. Full source/checkpoint hashes are logged.

Integration also caught differing actor hash conventions between optimizer and runtime;
all now use the canonical schema-and-weight hash. A saved-NPZ to NumPy-trajectory to
Torch-PPO regression verifies this boundary. The early-query exception now requires
actual exhaustion of a short chunk. Causal feature and runtime tests cover episode reset,
privileged-label rejection, bounded delays, sixteen-proposal access and identity cadence.

The frozen preparation manifest assigns 12 short checks (at most32 actions each) and
24 full training-only reference episodes across Jar, Stack and Clutter. Its action
ceiling is58,784, with initialization separately counted. Integration and full preparation
time ceilings are4 and16 GPU hours respectively. All failed attempts consume allocation;
there are no automatic retries. Preparation measures throughput before setting the
separate pilot GPU-hour ceiling. Cabinet/Dusty outcomes remain reserved.
