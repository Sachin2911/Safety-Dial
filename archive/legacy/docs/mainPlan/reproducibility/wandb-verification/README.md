# Deferred Walker W&B verification sources

These are inert source snapshots and provenance records, not a completed remote verification. No API call, upload, model run or simulation was performed to prepare this bundle.

`walker_wandb_preparation_20260927.json` preserves the original local logging and timing observation byte-for-byte. Its local prefix evidence does not establish final remote sync. The original v1 verifier is also retained exactly; do not execute v1, which lacks an overall deadline and recipe/output guards. Its independent review is included.

The v2 helper is prepared for root review and execution **only after the recovered S2 stage is completed**. Twenty-two synthetic tests and Ruff passed. It requires the pinned workflow configuration, successful terminal stage and hash-bound final 214,780-step report before using the API. It checks the exact existing entity/project/run, all 2,139 recovered records against the full 10,739-record local history, validation step identities and the final validation value. A single wide step window retains explicit rejection of pre-resume records. A 180-second process deadline bounds SDK history waits.

The exact invocation is:

```bash
uv run python /tmp/verify_walker_final_wandb_v2.py --output runs/continuation-audit/walker_wandb_final_20260927.json
```

If `/tmp` is lost, copy `verify_walker_final_wandb_v2.py.txt` to a fresh reviewed executable path, verify its SHA against this manifest, and run from the project root. It uses `/workspace/Safety-Dial` explicitly. Preserve a copy of the exact source that is actually used and its resulting audit. Existing or alternative output paths are refused; never overwrite an existing audit to retry.

Credentials are read only from existing configuration and are never recorded. The API host is pinned to `https://api.wandb.ai`; absent credentials fail without interactive login. There are no `init`, `sync`, `log`, `finish`, update or delete calls. The SDK can create a local API sidecar/cache; this is not an experiment run. The deadline can terminate the verifier without waiting for SDK cleanup.

A later successful audit proves logging agreement at an observed mutable service. It does not prove model quality, a pinned W&B artifact, earlier interrupted-run sync, all intermediate validation values, or Hugging Face durability. Those remain separate evidence. Failure emits a sanitized reason/type without transport details or credential values.
