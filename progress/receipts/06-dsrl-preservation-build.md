# Task 6 preservation implementation

25 October 2026. `experiments/scripts/preserve_dsrl06.py` implements local staging
and private Hugging Face preservation for both a full DSRL run and an engineering-only
stop. This build made no network upload, rental, simulator or real neural call.

After policy servers, simulator workers and the learner have stopped, write a shutdown
receipt containing `instance_id`, `workers_stopped: true`, and `worker_pids` listing all
launched worker processes. The helper checks that each recorded PID is absent or is
verified as a terminated Linux zombie awaiting reaping, matching the coordinator's
process-identity rule. Keep
the independent financial watchdog running. On the rental, run:

```text
python3 experiments/scripts/preserve_dsrl06.py prepare-remote \
  --root outputs/dsrl06 --instance-id INSTANCE \
  --query-ledger outputs/dsrl06/queries.sqlite \
  --workers-stopped-marker outputs/dsrl06/quiescence.json
```

This writes `outputs/dsrl06/preservation/prepared.json`, a complete source inventory,
the shutdown receipt, and `preservation/query-ledger.sqlite`. The ledger uses SQLite's
backup API and an integrity check, preserving committed WAL data consistently.
The original database, WAL and SHM are excluded from upload. All query reservations
remain counted, including failures and unconfirmed responses. Query probe state files
and partial artifacts are retained.

A provisioning failure before query service creation can be preserved without
inventing a database. The helper requires no query artifacts, client query journal
events, positive model-query charges, positive probe counts, SQLite sidecars or
unresolved probe send intentions. It also refuses a different surviving query
database, reserved simulator starts or positive policy/warm-up interaction counts.
It records `query_backup` with `source_missing:
true`, `queries: 0` and `reason: no_query_service_started`, and creates no backup file.
The final marker separately reports this explicit absence and verified accounting;
it does not claim a database backup was made. Surviving query evidence with a missing
database blocks verification.

`network-cost.json`, `watchdog-status.json`, financial watchdog logs and `STOP` are
explicit point-in-time snapshots. The financial guard continues to write the originals
without invalidating the scientific freeze. Preserve their final original versions
after stop in a completion appendix, along with final provider and billing receipts.
Changes or additions to scientific files after preparation invalidate the frozen
inventory. Preparation refuses to overwrite an existing preservation directory.

Sync the final remote outputs after this backup, then stage on the workstation:

```text
.venv/bin/python experiments/scripts/preserve_dsrl06.py stage \
  --remote outputs/dsrl06/remote --local-root outputs/dsrl06 \
  --stage outputs/dsrl06/hf-stage --instance-id INSTANCE

.venv/bin/python experiments/scripts/preserve_dsrl06.py upload \
  --stage outputs/dsrl06/hf-stage \
  --marker outputs/dsrl06/preservation-verification.json --instance-id INSTANCE
```

`--remote` can repeat to include preserved failed hosts. Local receipts, financial and
rental records, task-6 source files, configs, protocol, environment evidence, all raw
attempts, recordings, AP/monitor sidecars, video, query NPZ, learner data, journals,
checkpoints and partial checkpoints are included. The dataset uses only fresh
`jar/dsrl06_*`, `recordings/dsrl06` and `receipts/dsrl06` paths in the existing private
dataset `Sachioster/maniguard-jar-replication-20261022`. Task indexes reference the raw
attempts, including failed starts. Model weights and their full checkpoint/replay state
go only to the new private model repository
`Sachioster/safety-dial-dsrl06-20261025`. An engineering-only stop requires no model
checkpoint or model repository. Nothing here adds artifacts to git.

The process reads its token from the ignored `.env` or environment. It scans staged
files for known token/key values and refuses credential environment files or symbolic
links. Credentials are not passed as CLI arguments, written into manifests or printed.
Top-level network failures report only the exception class, since provider exceptions
can contain credential-bearing URLs.

Before and after upload, the complete staged inventory and all stable source files
are checked. New late source or receipt files require a fresh stage. Both repositories
must be private. Every uploaded file, including the preservation manifest, is downloaded
at its returned pinned commit and checked by size and SHA256. Only after all checks
pass is the verification marker written. A failed attempt removes any stale marker.
The marker includes the manager's required `instance_id`, `local_verified`,
`private_hf_verified`, nonempty `pinned_revisions` dictionary and `manifest_sha256`,
plus `manifest_path`, file counts and byte counts. This is evidence for the manager's
destroy guard; the helper itself never destroys or stops an instance.

Validation: `.venv/bin/python -m pytest -q experiments/tests/test_dsrl06_preserve.py`
reported 17 passed in 0.50 seconds. Ruff check and format passed at line length 100.
Tests use a real SQLite WAL and a local fake Hub. They cover engineering-only stops,
private model separation, all-file pinned downloads, failed attempts and probe states,
worker shutdown checks, continued financial watchdog writes, late evidence, source
tampering, public destinations, wrong downloaded bytes, known secret rejection and
stale marker removal. A dedicated case checks that real Hub-style download-cache
symlinks and previous staging trees are excluded before the source inventory scan.
No actual upload has been exercised by these tests.
