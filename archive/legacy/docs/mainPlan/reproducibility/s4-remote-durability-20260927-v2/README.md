# Walker S4 remote durability verifier v2

This bundle preserves the exact verifier used for the successful terminal S4 remote audit, its focused tests and its prior local-only preflight. Python sources are inert `.py.txt` snapshots. The earlier v1 bundle remains unchanged.

## Narrow correction

The parent recorded that v1 stopped in `payload_plan()` with `Changed acquisition design`, before reaching authentication imports or any API call. Its acquisition-key guard omitted the required `no_update` baseline. The frozen producer already writes this baseline before the random and boundary curves; the scientific data and qualified acquisition design did not change.

Version 2 changes one line to require exactly `{no_update, random, boundary}`. It still verifies twelve adapted model uploads and one bank upload; no additional model upload is expected for the unchanged baseline. Missing baseline or unexpected arms are rejected. `changes.diff` preserves the exact source change.

## Validation and actual execution

- The updated focused test suite passed all 27 cases in 0.44 seconds; Ruff passed for the v2 verifier and tests. New cases cover the frozen complete schema, absent no-update baseline and unexpected arm.
- `local_preflight.json` records the actual terminal preflight: all local statements before the first authentication import passed, producing 13 payload plans with 163 expected remote files. Full local tree, receipt, workflow and source identities remained unchanged. No network call occurred in that preflight.
- The [canonical successful remote audit](../../results/continuation-v2/walker_s4_final_remote_audit_20260927.json) is byte-identical to `runs/continuation-audit/walker_s4_remote_durability_20260927_v2.json`, SHA256 `afe647e62df21dbc2aab035723ff9ce45fb3ce352c3acfcc534b6cf18fef70f9`. The root executed it; this preservation step did not repeat any network call.

The recorded audit verified **163 files, 2,056,232,115 bytes** across twelve private model revisions and one private dataset revision. Model payloads total 96 files/597,546,170 bytes; the bank payload totals 67 files/1,458,685,945 bytes. Each recorded run-ID tag matched its pinned revision.

| Payload | Files | Bytes | Pinned revision |
| --- | ---: | ---: | --- |
| bank | 67 | 1,458,685,945 | `09ac490696cecb951e83644db48e646878905f87` |
| random-s0-b128 | 8 | 49,792,132 | `d13de5f5c07e7ec0808b1a292ece0bcf6a0e11fc` |
| random-s0-b512 | 8 | 49,798,521 | `dcfb64ad75affdd765058c3e3880145c1518b6ce` |
| random-s1-b128 | 8 | 49,792,721 | `bd855acf9c3847900a4f2d5667fc073960f27cc2` |
| random-s1-b512 | 8 | 49,798,118 | `5d550a6570d5f4dda66d35c4216856a5909a1539` |
| random-s2-b128 | 8 | 49,792,634 | `c9e0f5d4068b495090793bb2c74d4fa97dc487fd` |
| random-s2-b512 | 8 | 49,798,204 | `1b3e1a84cbf3e0ddb2007c52861fb3a3dda37baa` |
| boundary-s0-b128 | 8 | 49,792,578 | `e87d2aac5cdf656d81d0ade91ccab7d5d2fc84c1` |
| boundary-s0-b512 | 8 | 49,798,811 | `e32f32e60972d9af47b65fae76c28a3354b24513` |
| boundary-s1-b128 | 8 | 49,792,170 | `7df3da587335fd5e75b1f214ff18842cf774834d` |
| boundary-s1-b512 | 8 | 49,798,899 | `6f4363e451efd9478c3bf1676b5fcc586f7b11e0` |
| boundary-s2-b128 | 8 | 49,792,844 | `6e141a0d741c3a7f696f47849d9da3d0f5c87b9f` |
| boundary-s2-b512 | 8 | 49,798,538 | `f91533415b8b3b5c388b1575fd74abd4c2397ee5` |

Full repository/path identities and per-file SHA256, Git blob identities and sizes remain in the canonical audit. The repositories are `Sachioster/safetydial-walker2d` (model) and `Sachioster/safetydial-walker2d-data` (dataset).

## Scope and limitations

The audit compares local hashes to authenticated metadata at pinned revisions: LFS SHA256 and size, or canonical Git blob SHA1 and size. It is not a remote download/restore test, tensor validation or a fresh scientific evaluation. Local `hf_upload.json` receipts are uploader exclusions. The results report adds its bank revision after uploading; that later report is not claimed as part of the bank commit. Tags and privacy are observations that administrators can later change.

All twelve adapted receipts are justified by the completed acquisition branch. A development-only negative stop would require the bank payload alone; the guard for that branch remains unchanged. No original scientific directory, v1 helper/test, or v1 preservation file was modified. No simulator/model/network operation was performed while creating this bundle.

For reproduction, materialize the inert helper as a `.py` file at an explicit local path and supply the successful local S4 audit plus a fresh allowed output path. It intentionally binds the frozen recovery run, producer/config hashes and expected completed file trees. This is an actual-used version, not a generic tool for other runs.
