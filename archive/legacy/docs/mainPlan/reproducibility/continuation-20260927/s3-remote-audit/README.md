# Prepared S3 remote probe durability audit

This helper is prepared and locally validated. **No actual network verification has run.** The adjacent source and tests are inert `.py.txt` copies. They must not be mistaken for a completed private-HF archive audit.

The original S2 durability helper supplies the streaming SHA256/Git-blob and pinned-private-metadata pattern. The S4 helper uses the same full-prefix, immutable-stage, deadline and safe-error conventions. Neither directly matches the S3 writer ordering. This separate S3 helper does not import mutable `/tmp` helper code.

## Deferred command after parent review

```bash
uv run python /tmp/audit_walker_s3_remote_durability_20260927.py \
  --local-audit docs/mainPlan/results/continuation-v2/walker_s3_final_local_audit_20260927.json \
  --output runs/continuation-audit/walker_s3_remote_durability_20260927.json \
  --max-seconds 900
```

The output must be fresh under `runs/continuation-audit`, outside all completed hashed directories. A hard SIGALRM deadline bounds the entire operation; the default is 900 seconds and the maximum is 1,800. Errors do not echo HTTP exception text or credentials, and a failed check does not write a success artifact.

## Checks and writer ordering

The helper requires terminal successful S2 and completed or scientifically stopped S3, exact frozen recovery workflow identity, finished subprocesses, complete recorded output hashes and the exact successful local `--mode both` checker report. A failed S3 health gate must leave S4 unattempted. A passed S3 may be checked while S4 advances, provided the completed S2/S3 records and all their output bytes remain unchanged.

It binds the actual final 214,780-step recovered model, weight/config/scaler hashes, the 172,000-step recovery SHA, final local model receipts, and the frozen collection/split identity. Both linear and MLP probe `.pt` and `.json` files, gate rows, config, README and run manifest belong to the uploaded payload. Current reviewed producer hashes are guarded before and after; this is not a replacement for an executed source snapshot that S3 did not retain.

`HFStore.upload_run` excludes `hf_upload.json` and `hf_upload.json.tmp`, uploads the run, then writes the local receipt and creates/checks the run-ID tag. S3 subsequently adds `probe_hf_revision` to the **results** manifest only. The uploaded run manifest therefore stays byte-identical to its pre-upload copy; the later results manifest is checked by removing exactly that one field. The local receipt and the later results manifest are not falsely claimed as files in the remote commit. Gate and row mirrors are byte-checked.

The actual closed-run local preflight found 10 files and plans to verify 9 remote payload files. It passed using the preserved local audit, with both completed output trees and final model tree unchanged. `local_preflight.json` records the exact identities. This remains local evidence only.

After all local checks, the API uses the explicit official endpoint and existing authentication. Only `repo_info`, `list_repo_refs` and `list_repo_tree` are called. The repository must be the pinned private model repository `Sachioster/safetydial-walker2d`, the exact prefix `probes/walker2d-probes-recovery-20260927-1`, and the run-ID tag must point to the recorded immutable revision before and after inspection. Full recursive remote file inventory must equal the upload payload exactly. Every file uses LFS SHA256 plus size or canonical Git blob SHA1 plus size. Privacy/revision are rechecked afterward, and all local sources/output trees are rehashed before a fresh audit is written.

## Validation and limits

Twenty-four synthetic checks passed, covering positive and health-stop branches, pre-payload barriers, wrong local audit/lineage, source and payload mutations, post-upload manifest distinctions, receipt exclusions, complete Git/LFS metadata, missing/extra/duplicate files, wrong privacy/revision/tag, tag races, deadlines and symlinks. Ruff undefined-name checking passed. No API was used in these tests or the authorized local preflight.

Remote metadata verification is not downloading bytes or a restore test. S2 model and S1 data durability remain the responsibility of their separate remote audits. Scientific R2, horizon-error, source-role and gate arithmetic depend on the exact passed local audit; no new scientific measurements are made. Tags/privacy can later be changed by repository administrators. The helper must be reviewed again if its pinned producer/config/checker identities change.
