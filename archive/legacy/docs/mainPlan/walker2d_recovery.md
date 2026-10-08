# Walker2d training recovery

The live training run keeps its executed sources in `source_snapshot/` and saves an
atomic `optimizer.pt` every 4,000 optimizer updates. That bundle contains the model,
AdamW, scheduler, completed update count, training identity, history, and Torch, CUDA,
NumPy and Python RNG states. Its completed update count controls the sampler; batches
prefetched beyond that update are discarded when resuming.

For an interrupted, unfinished run, stop the original writer first and use a fresh run
name with `walker_s2_train_lewm.py --resume-from <old-run>/optimizer.pt --run-id <new-run>`.
Repeat the original data, split and training arguments. Supply the original
`--push-every` value; the adjacent `<old-run>/config.yaml` is required and checked before
data loading. The current full run uses `--push-every 4000` and 214,780 total updates.
Changing this interval changes the random numbers consumed by validation SIGReg, so it
cannot be used for an exact continuation. Existing bundles remain compatible: the new
guard reads their saved recipe instead of changing their training identity.

A resumed checkpoint needs the original training sources and package/rendering stack.
The current identity checks data, splits, architecture, normalization and optimizer
recipe, but does not reject every possible source-code change. Use the recorded source
hashes and manifest versions when preparing a recovery checkout. The snapshots preserve
relative paths; reconstruct the original repository layout instead of executing the
snapshot entry point in place. Use a new workflow configuration for a new resumed run
name so downstream stages identify the correct model.

If the bundle is already at the final update and only the final upload/report failed,
run this after the original process has exited:

```bash
uv run python experiments/scripts/walker_s2_finalize.py --run-dir runs/<completed-run>
```

The utility loads tensors on CPU, verifies both model exports, configuration,
normalizers, splits, optimizer/scheduler counts, training history, manifest and source
snapshot against the completed bundle, and retries the private upload. It then writes
the final upload receipt and the missing S2 result report. It performs zero optimizer
updates and refuses to overwrite an existing final result report.

Mixed, incomplete or stale exports fail verification before upload. The utility does
not regenerate exports: recover the recorded source/package environment and regenerate
all exports consistently from the atomic bundle before trying again. A model object
that cannot be imported also requires that original environment. Original live bundles
do not retain first-batch render telemetry, elapsed time or steps per epoch, so recovered
reports mark these fields unavailable. They do not claim an unrecorded render check.

CPU regression coverage includes a spawned, prefetched DataLoader crossing an epoch
boundary, exact model/AdamW/scheduler/RNG agreement after recovery, rejection of changed
checkpoint intervals, bounded upload retries, zero-update finalization, and rejection
of stale weights, serialized objects, step records and manifests.
