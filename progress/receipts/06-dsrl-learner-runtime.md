# Task 6 learner runtime outcome

25 October 2026. Experimental work stopped at **00:19:36 UTC** after the original
128-query decoder-authority probe returned an invalid gate. No deterministic
learner retry was admitted after that result. This receipt records learner
engineering, not a trained-policy result.

The first GPU profile returned HTTP 409 before timed profiling. Its original
client did not retain the response body. The first retained diagnostic, at
23:58:32 UTC on 24 October, reported a maximum author-versus-bridge state
difference of **3.1837262213230133e-6**. After the recorded shallow active-agent
copy repair removed fresh static optimizer closures, the explicit repaired
profile again returned HTTP 409, with difference **2.1345913410186768e-6** at:

```text
['actor'].opt_state[0].mu['network']['Dense_0']['bias']
```

This is the actor Adam first-moment state for the first network Dense bias.
The **1e-6 parity threshold remained unchanged**. Shared static functions did
not resolve the discrepancy. Default GPU gradient-reduction nondeterminism is
plausible but unproven; neither failure establishes a defect in the author SAC
equations or a learning outcome.

All **eight SAC updates** remain charged: two prior workstation CPU updates and
six GPU updates across three reference/bridge pairs. The requested **128 timed
updates never ran**. There were zero training updates, zero collected learner
episodes, zero replay transitions and no published learned actor. The final
engineering checkpoint records actor version 0 and zero actor-sampling requests.
The 134 workstation controller calls remain accounted in their existing
preflight receipts. Native VLA engineering and decoder-probe queries are recorded
separately from this learner account.

Read-only source inspection identified a possible future learner-only remedy:
start a fresh process with `--xla_gpu_deterministic_ops=true` in `XLA_FLAGS` before
JAX initialization. Pinned JAXlib 0.5.0 uses XLA
`a91d9c30c545eccdbc26075c2b2ff04a014e66e9`; its
[determinism requirement](https://github.com/openxla/xla/blob/a91d9c30c545eccdbc26075c2b2ff04a014e66e9/xla/service/gpu/stream_executor_util.cc#L622)
reaches GEMM and convolution selection, and its
[cuDNN adapter](https://github.com/openxla/xla/blob/a91d9c30c545eccdbc26075c2b2ff04a014e66e9/xla/stream_executor/cuda/cuda_dnn.cc#L6499)
filters nondeterministic engines. This remedy was **not applied or tested**.
Any future test requires a new frozen protocol and fresh accounting; the stopped
task does not authorize another parity attempt, training or evaluation.

The following paths are relative to the synced `outputs/dsrl06/remote/` tree:

- `diagnostics/parity-01/parity.response.json`, SHA256
  `2fd5c9404de8f99b74924148a76570cb758d31c4c37d3cfe2918d07d4ac130cb`.
- `http/engineering-profile-repair-01.error.json`, SHA256
  `9ae8d6737c0fa05509cdc74674d2477a734d5c0b81b0c7261415a2700dee7a5c`.
- `learner/learner-ledger.jsonl`, SHA256
  `ba7d1e49f2bdb4c4e549cd22cd91870eb42f54ad2166ee57a2da89495423cb13`.
- `learner/checkpoints/before-parity-diagnostic-01/manifest.json`, SHA256
  `e630e9e04502cdf9302595e8964a94ea973745487863fdbaf124437e4c28da35`.
- `learner/checkpoints/after-parity-diagnostic-01/manifest.json`, SHA256
  `84517cfec52479d184c5d7f82a77613b70bdbe06d2659f4b62e3743ac0749d76`.
- `learner/checkpoints/engineering-before-authority/manifest.json`, SHA256
  `c62c4242ab152ded479f61d43cca7a7100f17b21018816674ba3aab2e84799b4`.

All three checkpoint manifests were verified against their locally synced
learner, replay and state payloads. This new receipt was prepared offline with
no additional gradient, actor, VLA, simulator or network call. Source files and
earlier receipts were not changed.
