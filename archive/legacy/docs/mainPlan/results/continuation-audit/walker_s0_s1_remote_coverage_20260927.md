# Walker S0/S1 remote payload coverage — 27 September 2026

Passed a read-only comparison of stable local bytes with private pinned Hugging Face metadata. This covers the exact runtime payload prefixes below, **not** the whole project archive or unknown original execution provenance.

| Stage/run | Revision | Files | Bytes | Scope |
|---|---|---:|---:|---|
| S0: `walker2d-policies-complete-20260926-1` | `b42e3f3e7dd80bc4d95cc2ef43c753cee96b9b55` | 45 | 1,116,490 | Exact prefix inventory; LFS SHA256/size or Git blob SHA1/size |
| S0: `walker2d-policies-continuation-20260926-1` | `59e05132b151c816ec0b267b0d70e15891ea0a25` | 44 | 1,090,007 | Exact prefix inventory; LFS SHA256/size or Git blob SHA1/size |
| S1: `walker2d-data-20260926-3` | `ddd4d51648d2a382725efd20d04cedebedbf8fd2` | 7 | 805,909,289 | Exact prefix inventory; LFS SHA256/size or Git blob SHA1/size |

Both policy runs are in private model repository `Sachioster/safetydial-walker2d`; S1 is in private dataset repository `Sachioster/safetydial-walker2d-data`. Each run-ID tag resolved to its exact receipt revision before and after verification. All local payloads and receipts retained identical bytes and file identities. The 41 earlier actor files are byte-identical to their counterparts in the completed 42-actor ladder; totals count these separate uploaded occurrences.

S1 coverage is exactly `README.md`, `config.yaml`, `manifest.json`, `splits.json`, `setA.h5`, `probe.h5` and `roots.h5`. H5 files were streamed only for hashes. The live `s4/` subtree was neither traversed nor read. The local `hf_upload.json` and `upload_receipt.json` were written after upload and are excluded; external `collection.json` and `policy_ladder.json` are also outside this payload. S0 policy bundles likewise exclude their local upload receipts, raw training jobs, observability artifacts and later evaluation reports.

The original policy packaging command is not preserved by the bounded source search; direct remote inventory/content verification establishes the payload bytes without inventing that invocation. Current collector/uploader source explains S1 ordering but does not retroactively establish the historical dirty-tree source. This audit cannot repair unknown S0/S1 execution provenance. No remote bytes were downloaded, no tensors/H5 were decoded, and no model, simulator, upload or remote change occurred. Final reports, media, source/config/audits and S4 need their own coverage. Tags/privacy can be changed later by administrators.

JSON: `runs/continuation-audit/walker_s0_s1_remote_coverage_20260927.json`. SHA256: `c085c8b718ee7279b9d6c852c89549c6f9bcf255dbf7ef33c5c37a64e54d40b0`.
