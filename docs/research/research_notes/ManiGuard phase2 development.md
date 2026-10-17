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
