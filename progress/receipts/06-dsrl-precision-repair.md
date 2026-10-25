# Task 6 actor inference precision repair

25 October 2026. This source-only engineering repair was selected before any
task-6 VLA query, simulator episode or benchmark outcome. The original stage-A
freeze and preflight reports remain unchanged. A separate immutable stage-A-v2
amendment records the replacement source hashes and the four additional CPU
controller calls.

The pinned author creates Flax Conv and Dense layers without an explicit
precision. Flax 0.10.2 defaults to `precision=None`; JAX 0.5.0 defaults
`jax_default_matmul_precision` to None. Its GPU default may use TF32 internally
despite float32 inputs and outputs. Thus float32 storage alone does not establish
CPU versus GPU agreement at the frozen 1e-5 tolerance. The pinned JAX source and
[JAX precision documentation](https://docs.jax.dev/en/latest/201/precision.html)
identify `highest` as the full float32 GPU precision setting.

`AuthorBackend.sample` now wraps only the call to the original author
`sample_actions_jit` in `jax.default_matmul_precision("highest")`.
`AuthorActorBackend.sample` uses that same method. The context is local to the
sampling request and restores the preceding state on success or exception.
Source networks, parameters, batch statistics, stochastic RNG mapping and noise
bounds are unchanged. Author SAC updates and fixed-batch source parity retain
their original precision defaults. No global JAX or NVIDIA environment override
was added, and `LearnerConfig.resolved()` still matches the original stage-A
learner configuration. The CPU versus GPU criterion remains atol=rtol=1e-5.

JAX 0.5.0 includes this precision context in the JIT cache key. A pure CPU 4 by 4
matrix compiler check traced default and highest contexts separately, producing
two cache entries with DEFAULT and HIGHEST lowering attributes respectively;
returning to the default reused its original trace. This check ran no actor,
VLA, gradient or simulator operation. A focused backend test verifies unchanged
author arguments, the active highest context and restoration after an exception.

Two paired CPU cases then compared the unmodified author sampler at its default
CPU precision with the actor-only service sampler at highest precision, using
the same exported published parameters, observations, keys and query indices.
Maximum sample difference was **0.0**. The source configuration uses seed 6060;
synthetic observation and job RNG seeds are 6062, with job IDs `CPU-HIGHEST-0`
and `CPU-HIGHEST-1` and query indices 0 and 1. Outside sampling, precision remained
None. This is CPU versus CPU evidence; the rental engineering check must still
measure CPU versus GPU agreement under the unchanged tolerance.

The new evidence is isolated under `outputs/dsrl06-preflight/actor-highest/`:

- `cpu-highest-parity.json`: SHA256
  `f7362c0f6b1eebd3d326ec57f21040c7d58e75a885cd3ddde82d5152101b044e`.
- `paired-cases.npz`: SHA256
  `22c4e958367fe553297879cfb6b2e0ef41719a18b3f9ed50ba51962195efb684`.
- `controller-events.jsonl`: SHA256
  `8dda13d823e7f1808d2f6021b20011e39aae9c7a7beded98efcfaf8a468ed34d`.

The repaired helper SHA256 is
`c58e0db8b4cc8c7f95cbbd404abf26b98a4631e76516eeeea705741d92dcb04b`.
There were exactly two reference and two worker sampler calls. Prior controller
calls remain 130, for **134 cumulative controller calls**. New gradient updates,
VLA queries and simulator controls are all zero. The existing two CPU engineering
gradients remain charged once. The analyzer reads the amended count from
`stage_a.amendment.prior_controller_actor_forward_calls`, falling back to 130
for the original freeze.

Focused validation passes: **15 learner tests and 9 analysis tests**, 24 total,
including both localhost HTTP cases. Ruff, formatting and diff checks pass.
