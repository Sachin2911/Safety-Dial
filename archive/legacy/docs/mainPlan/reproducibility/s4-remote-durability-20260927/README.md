# Prepared S4 remote durability verifier

This is a prepared, synthetic-tested verifier for the recovery study
`walker2d-s4-recovery-20260927-1`. It has not been executed on actual S4 results
or any remote service. No scientific stage completion or upload durability is
claimed by this source bundle.

The existing `/tmp/check_walker_s4_final_evidence.py` checks local scientific
outputs and recorded receipts. It explicitly leaves remote privacy, payload and
tag checks unverified. The earlier final-S2 upload helper handles a model-only
writer with different receipt timing, so it does not cover the S4 bank upload.
This helper consumes a passed local S4 audit and adds only remote durability
checks. It does not repeat acquisition, evaluation or model calculations.

## Files and execution barrier

| Original file | Inert preserved snapshot |
|---|---|
| `/tmp/audit_walker_s4_remote_durability_20260927.py` | `audit_walker_s4_remote_durability_20260927.py.txt` |
| `/tmp/test_walker_s4_remote_durability_20260927.py` | `test_walker_s4_remote_durability_20260927.py.txt` |

The `.py.txt` suffix avoids automatic module/test/lint discovery. The originals
and snapshots are byte-identical. The synthetic test uses an absolute `/tmp`
helper path; restore that exact layout and verify hashes before rerunning tests.
`validation.json` records 24 passing synthetic checks, clean Ruff and no real
API, model, simulator or S4 payload reads. The tests exercise both study branches,
private/pinned repository metadata, exact file inventories, LFS and Git identity,
receipt ordering, mutation rejection and a brief real signal deadline.

Run only after S4 is terminal and the existing local scientific checker passed:

```sh
uv run python /tmp/audit_walker_s4_remote_durability_20260927.py \
  --local-audit /tmp/walker_s4_recovery_final_evidence.json \
  --output runs/continuation-audit/walker_s4_remote_durability_20260927.json
```

The verifier requires the exact recovery workflow/config identity, a completed
actual S4 subprocess, all three terminal output trees and the development-gate
hash. The passed local scientific audit must bind the same outputs, run and
checker source SHA. All of this is checked before the API client is created.
Output must be fresh and outside the completed run/results/bank directories.
The default hard process deadline is 900 seconds, capped at 1800, and covers
local hashing and metadata pagination. A timeout produces no passed audit.

## Remote upload contract

- Adapted models: private `Sachioster/safetydial-walker2d`, repository type
  `model`, prefix `adapted/<S4-run-id>-<arm>-s<seed>-b<budget>`.
- Bank: private `Sachioster/safetydial-walker2d-data`, repository type `dataset`,
  prefix `banks/<S4-run-id>`.
- Every local `hf_upload.json` must match the expected repository, type, prefix,
  run tag and full pinned commit in the scientific report.
- The pinned remote commit must exist in that exact private repository. The
  declared run tag must resolve to the same commit before and after inspection.
- Every uploaded file under each prefix is enumerated. Missing or extra files
  fail the audit. LFS payloads must match SHA-256 and byte size; other files must
  match canonical Git blob SHA-1 and byte size.
- Full acquisition requires both arms, all three seeds and both budgets, giving
  12 model uploads plus one bank upload. A development diagnostic stop requires
  only the bank upload and forbids adapted checkpoints or final-test claims.

The helper uses existing credentials, a fixed `https://huggingface.co` endpoint
and metadata reads only (`repo_info`, `list_repo_refs`, `list_repo_tree`). It never
creates repositories/tags, uploads, downloads remote bytes, starts stages,
loads tensors/HDF5 arrays or calls models/simulators. Credentials are not printed.
Local payloads, source files, passed-audit evidence and the terminal S4 stage are
checked again after remote inspection. The helper only writes a fresh parent
audit JSON once all checks pass.

## Limits and writer-order details

`hf_upload.json` and `hf_upload.json.tmp` are explicitly excluded by the uploader.
They are bound locally, never claimed to exist at the remote commit. All other
files present in the uploaded model/bank directories must match remote metadata.
S4 writes its bank report before upload, then adds `bank_hf_revision` only to the
local results `study.json`. The remote bank therefore contains the exact
pre-upload report. Later local results and run-level files outside the uploaded
folders are protected by terminal local hashes but are not claimed remotely
archived by this helper.

A negative development gate can still upload generated sealed test/stress banks.
Their presence does not mean final-test evaluation or completed acquisition.
An earlier S3 negative gate with no S4 subprocess has no S4 upload to audit; this
helper rejects that state before network access rather than inventing success.

Remote verification uses metadata, not a downloaded restore test. Git payloads
use SHA-1 because that is the content identity exposed by Git; LFS uses SHA-256.
Tag and privacy checks describe the observed service state and cannot prevent a
later administrator changing it. Scientific claims and model semantics require
the separate local audits. This bundle neither executes nor qualifies S5.

Producer, workflow and local-checker identities are fixed in this reviewed
version. If those inputs or the helper require correction, preserve and review a
new version separately, with actual-use provenance. Do not overwrite this
prepared source snapshot or silently mark it executed.
