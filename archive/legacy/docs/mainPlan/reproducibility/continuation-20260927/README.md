# Preserved Walker recovery audit tools, 27 September 2026

These are byte-identical snapshots of nine reviewed or prepared files that
previously existed only under `/tmp`. This preservation record does not claim
that final S2, S3 or S4 has completed or passed. Scope stops at S4; no S5 work
is authorized by this bundle.

Python sources have the inert suffix `.py.txt` so test, lint and normal Python
module discovery do not execute historical helpers. Original `/tmp` bytes were
not modified. JSON baselines and preparation records retain their exact bytes.
`manifest.json` records every original path, snapshot path, SHA-256, byte count,
status and available actual-use evidence. The source inventory is the nine-item
`reproducibility_tools_only_in_tmp` field of
`runs/continuation-audit/final_publication_coverage_20260927.json`.

## File mapping and status at preservation

| Original `/tmp` filename | Preserved filename | Status and limits |
|---|---|---|
| `audit_walker_recovery_final_upload.py` | `audit_walker_recovery_final_upload.py.txt` | Prepared and reviewed; 20 synthetic checks and Ruff recorded in the preserved proposal. No actual final audit claimed. |
| `check-walker-recovery-final-evidence.py` | `check-walker-recovery-final-evidence.py.txt` | Prepared and statically reviewed for terminal S2/S3. No actual final audit claimed. |
| `check_walker_s4_final_evidence.py` | `check_walker_s4_final_evidence.py.txt` | Prepared and reviewed; responsible reviewer confirmed 27 pure synthetic checks. No actual final audit claimed. |
| `walker_recovery_training_summary.py` | `walker_recovery_training_summary.py.txt` | Prepared; synthetic 10,739-record history rendering and Ruff reported by its author. No final training report generated here. |
| `audit_walker_recovery_upload.py` | `audit_walker_recovery_upload.py.txt` | Actually used for checkpoint audits. Saved audit JSONs bind this exact source and baseline hash; manifest lists those records. |
| `record_walker_recovery_upload_audit.py` | `record_walker_recovery_upload_audit.py.txt` | Parent confirmed operational use at 196,000 and 200,000. These calls did not independently record their source hash. Writes status and performs Git staging/commit at module top level. |
| `walker_recovery_upload_baseline.json` | `walker_recovery_upload_baseline.json` | Frozen baseline actually used by the checkpoint auditor; its hash is bound in saved checkpoint audits. |
| `test_walker_s4_deferred_checker.py` | `test_walker_s4_deferred_checker.py.txt` | Actual synthetic-only test runner, with 27 checks reported by its reviewer. It is not a final research-result audit. |
| `walker_final_upload_index_entry.json` | `walker_final_upload_index_entry.json` | Prepared index-entry proposal. No final remote revision, audit hash or payload inventory is claimed. |

The checkpoint recorder's parent-confirmed invocations produced commits
`a633f65d6d8ba0d010eb7cf64b4f78d7e566f8b6` (196,000) and
`bc68d59811bb116a35e519a3c29ab7ba493b89d1` (200,000). The preservation hash binds
its current reviewed bytes; it does not retroactively add an independent
source-hash receipt to those invocations.

The exact deferred-audits index is included as
`walker_recovery_deferred_audits_20260927.json`. It is a dated preparation
snapshot, not current workflow state. Its `next_checkpoint_command` can be
stale, and it references a watcher outside this nine-file bundle. Do not run
its commands automatically or interpret its presence as a successful audit.

## Execution and provenance limits

The live upload auditor and its frozen baseline bind worker PID 18375, start
ticks 130742, exact recovery configuration and the 172,000-step resume origin.
They require that same worker to exist and advance CPU ticks. The live auditor
cannot be rerun on its terminated process. Do not modify the baseline to make
an old audit pass against a different worker.

Final helpers have terminal-stage barriers. Preparation and synthetic tests
are distinct from invoking a checker against actual completed artifacts. The
final-upload helper uses read-only pinned Hugging Face metadata and private
repository identity checks, not a remote restore test or tensor semantics.
`hf_upload.json` is local-only; the remote `upload_receipt.json` can contain the
previous checkpoint's receipt because the local final receipt is written after
upload. The final S2 result JSON is outside that run upload. A separate final
CPU semantic check remains necessary. No final check was executed while making
this bundle.

These snapshots are not a self-contained experiment environment. They depend
on the saved repository helpers, workflow/source snapshots, data/checkpoint
identities and, where applicable, authenticated read-only HF access described
in their source. Some use exact `/tmp` paths. Reproduction must restore the
intended dependency layout and verify snapshot hashes, respect terminal guards,
and supply the original evidence. Never import the operational recorder just
to inspect it; it has top-level write and commit actions.

Any corrected helper or later actual-used version must be preserved separately
under a new versioned filename or directory, with its own exact source hash and
execution-result provenance. Do not overwrite this reviewed snapshot or silently
relabel prepared tools as executed.

## Preservation checks

All nine current source hashes and byte counts matched the publication inventory.
Copies were checked byte-for-byte and all source hashes were rechecked after
copying. A bounded scan for embedded HF/GitHub tokens, AWS keys, private keys,
literal bearer tokens, credential assignments and credential-bearing URLs found
no candidates. This was an accidental-credential check, not a general security
audit. No secret values were printed. No helper, model, simulator, live final
checker or network operation was executed. Nothing was staged or committed.
