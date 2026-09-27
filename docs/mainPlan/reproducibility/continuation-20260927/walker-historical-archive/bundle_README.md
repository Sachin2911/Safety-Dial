# Historical Walker diagnostic preservation bundle

This local-only bundle preserves three **unqualified historical diagnostics**. They are superseded by the `walker2d-recovery-20260927-1` continuation. These files are historical provenance, not selected recovery checkpoints or evidence that the adopted S3/S4 gates passed. No scientific result has been recomputed or newly qualified here.

## Exact preserved inputs

| Original repository-relative directory | Files | Role recorded in its original manifest |
| --- | ---: | --- |
| `runs/walker2d-probes-20260926-1` | 6 | Probes referring to the historical smoke model |
| `runs/walker2d-s4-20260926-1-boundary-s0` | 3 | Historical boundary adaptation diagnostic |
| `runs/walker2d-s4-20260926-1-random-s0` | 3 | Historical random adaptation diagnostic |

The `runs/` subtree contains byte-for-byte copies of every file in those three original directories. Original manifests and README files retain their original wording and values; this top-level README states the qualification limits. The original files were neither edited nor linked into this archive. The manifest records SHA-256, byte counts, source paths, and before/after source identity verification.

## Prior remote-coverage evidence

`prior_coverage_excerpt.json` contains the exact selected three run records, the Walker repository observation, and the original limitations from `runs/continuation-audit/historical_model_remote_coverage_20260927.json`, with that full audit's SHA-256 and byte count. The prior audit found no matching run-ID tag or exact payload at its observed pinned repository head. Its precise limit is:

> No exact payload match in the run-ID tag or observed pinned repository head; no claim of absence from all repository history.

This preparation made no remote requests, downloaded no remote bytes, and uploaded nothing. The cited private-repository observation is historical; current privacy, tags, revisions, and remote coverage have not been rechecked. There is no publication receipt for this new bundle.

## Provenance and recoverability gaps

- All three original manifests record dirty repository and upstream LeWM worktrees. Their commit IDs do not capture every executed source change, and this bundle contains no source snapshot.
- Their `upstream_revisions` objects are empty. The probe manifest refers to `runs/walker2d-lewm-a-smoke-20260926-1`; that base-model directory is not included here. The two adapted manifests do not pin a base-model path or revision.
- The original `costs` objects are empty. Their contents do not provide a complete simulator-cost ledger or establish later corrected protocol compliance.
- The preserved inventory has probe weights/metadata and adapted weights, but no optimizer state, training RNG state, original branch banks, or complete resumable-training bundle. Tensor contents were not deserialized, and no model restore was tested.
- Historical metric values remain unchanged inside their original manifests, but this archive adds no interpretation, comparison, selection, or qualified scientific claim.

## Proposed publication, pending separate review

Proposed destination: private Hugging Face **model** repository `Sachioster/safetydial-walker2d`, prefix `historical/walker-historical-diagnostics-archive-20260927-1`.

This is a concrete local preservation proposal only. No remote revision is claimed. Any eventual uploader must separately confirm authorization and privacy, upload this exact reviewed data-only inventory, and verify pinned remote metadata. This bundle includes no logs, Python source, configuration snapshots, added credentials, or Push-T files. In particular, it does not include or resolve the separately blocked Push-T E2 result/media archive.

`manifest.json` hashes every other file in this bundle; it intentionally excludes its own bytes to avoid a self-referential hash. The full inventory including the manifest is reported separately at preparation handoff.
