# Historical model remote coverage (27 September 2026)

Status: **complete_with_unresolved_coverage**. These seven bundles remain unqualified historical diagnostics.

| Run | Coverage | Pinned revision | Files |
| --- | --- | --- | ---: |
| pusht-acq-20260926-1-boundary-s0 | verified_exact_tagged_payload | 40db66a550fb9d88fd1f1843bb201faad6833d36 | 3 |
| pusht-acq-20260926-1-random-s0 | verified_exact_tagged_payload | 18457ee06a1a56d95fda83e621a35f710200688d | 3 |
| pusht-adapt-e2-20260926-1 | verified_exact_tagged_payload | d7c63d2fa1e5bd8528dfad8b4675377eaa02e715 | 5 |
| walker2d-probes-20260926-1 | unresolved_needs_archive_or_further_provenance | unresolved | - |
| walker2d-s4-20260926-1-boundary-s0 | unresolved_needs_archive_or_further_provenance | unresolved | - |
| walker2d-s4-20260926-1-random-s0 | unresolved_needs_archive_or_further_provenance | unresolved | - |
| pusht-probes-20260926-1 | verified_exact_tagged_payload | 0ce96eaf70d113d2728a290f9e44e08f3ded3ee1 | 10 |

The JSON contains exact local SHA256, canonical Git blob SHA1 and sizes, full per-prefix remote comparisons, repository privacy and run-tag stability observations. No model was loaded and no remote data, tag or receipt was written.

A missing local receipt is not evidence that an upload is absent. Unresolved cases mean the exact run tag and pinned head did not establish full coverage, not that all older repository history was searched. No upload retry is authorized or attempted.

The separate E2 90-file result/media archive approval hold and overlapping GitHub push hold remain unchanged. This read-only model check does not cover those bytes.

Limitations:

- Authenticated pinned metadata and local hashes only; no remote bytes downloaded, tensor decoding or restore test.
- Tags and privacy were observed metadata and can later be changed by repository administrators.
- Coverage verifies local stored payload only, not scientific correctness, qualified selection, or recoverability of missing original base assets.
- The absence of a local upload receipt did not determine remote coverage. No new receipt was written into any original directory.
- The seven runs remain unqualified historical diagnostics and are distinct from current Push-T E2 or Walker S2/S3/S4 results.
- This audit authorizes no upload and does not resolve or bypass the separate blocked 90-file Push-T E2 result/media archive.
