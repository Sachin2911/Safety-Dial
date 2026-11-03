# SafeLeWM archive

Completed evidence consolidated on 1 November 2026. Nothing here authorizes a new
experiment or resumption of a completed protocol.

## Cloud migration, 3 November 2026

Sachin authorized moving archived evidence to his private Hugging Face storage
while retaining local notes and recovery information. The first batch is
[Sachioster/safety-dial-evidence](https://huggingface.co/datasets/Sachioster/safety-dial-evidence):
sixteen original evidence archives (11,624,855,173 bytes), five small closeout
archives and a catalogue containing 107 original manifests, protocols, result
records and recovery files. Existing evidence archives are uploaded unchanged.

**Migration is complete.** All 24 uploaded files passed independent full downloads
at the pinned revision, with matching SHA-256 hashes and sizes. A separate recovery
drill fetched the baseline archive using the documented helper and restored an
original file with matching bytes, permissions and archived modification time.
The sixteen local evidence archives were then removed, releasing 11,624,855,173
bytes (11.62 GB). Temporary verification downloads were also removed. Small
manifests and result records remain available locally. Closeouts and the catalogue
remain in the cloud; their small local copies were subsequently included in the
output-folder deletion described below.
The [cloud index](cloud-index-20261103.json) maps original paths to cloud paths,
sizes and SHA-256 hashes. The [verification receipt](cloud-receipt-20261103.json) records the full pinned
revision, per-file verification, the recovery drill and each local archive removal.
The original per-study manifests below remain unchanged.

The uploaded data is pinned at
[`423c153c28099d7e902f4e49aa078b7feda61ba4`](https://huggingface.co/datasets/Sachioster/safety-dial-evidence/tree/423c153c28099d7e902f4e49aa078b7feda61ba4).

The [final receipt](https://huggingface.co/datasets/Sachioster/safety-dial-evidence/blob/6810311cb908d85e0c0501ce28530b3dcc43017c/SafeLeWM/migration-receipt-20261103.json)
and exact verification script are separately backed up at
`6810311cb908d85e0c0501ce28530b3dcc43017c`; both backups were downloaded and
hash-checked too. The receipt still pins the earlier data revision. Its SHA-256 is
`36ac3a15f77176b5354a46dd2783bc24a61412ee68fbf87469ed7c5e2a5e9f10`.

To fetch a study's original archives after migration, from the repository root:

```bash
.venv/bin/python SafeLeWM/experiments/scripts/fetch_cloud_archive.py --manifest manifest-mechanism-20261101.json
.venv/bin/python SafeLeWM/archive/restore.py --manifest manifest-mechanism-20261101.json --verify
```

Use `--archive baseline-20261101.tar.gz` to fetch one archive, or `--all` for all
sixteen. The fetcher uses the pinned revision and checks SHA-256 before restoring
the original local archive path. It refuses to replace different local bytes.
Authentication uses `HF_TOKEN`, an existing Hub login or the ignored root `.env`.
The ordinary `.venv/bin/python` has the needed Python dependencies; the separately
installed latest CLI remains at `.venv/hf-cli/bin/hf`.

Then use the original `restore.py` commands below to recover study files. Extract
into a scratch directory first; archived status statements are historical.
The cloud catalogue also contains the original manifests and recovery tool for
recovery without this checkout. In that case, extract the verified catalogue into
an empty directory, save the remote index as `SafeLeWM/archive/cloud-index-20261103.json`
and the final receipt as `SafeLeWM/archive/cloud-receipt-20261103.json`, then use
the included fetcher with Python, `huggingface_hub` and `python-dotenv` installed.
Preserve the cloud repository's commit history,
because the recovery records depend on exact revisions. This migration is not an
independent second-provider backup.

This migration batch did not move the root historical outputs, Block Plucking
outputs or the active L4 study. Sachin subsequently authorized deleting the older
root archive and all three local output trees, including `SafeLeWM/outputs/`,
without further migration. See the [deletion record](../../docs/records/progress/receipts/local-deletion-20261103.md).
The cloud index, receipt, original manifests and recovery tools in this directory
remain. Later historical sections describe availability at their original closeouts;
their references to local `outputs/` copies no longer imply those files are present.
No experiments, rental changes or remote deletions were part of this cleanup.

## What is retained

- `protocols/`: the four original protocols with unchanged bytes.
- `evidence/baseline-20261101.tar.gz`: original planning-comparison archive.
- `evidence/fixed-baseline-20261101.tar.gz`: original fixed-baseline archive.
- `evidence/anchored-20261101.tar.gz`: original three-stage AP evidence archive,
  including the retrieval cache.
- `evidence/supplementary-20261101.tar.gz`: all remaining unique original files,
  including initial qualification traces, source snapshots, public reference code
  and license, previous README/design text, and the original protocol paths.
- `manifest.json`: every original file path, SHA-256, byte count, mode and exact
  archive/member needed to restore it. Identical file bytes share one stored copy.
- `cleanup.json`: verification and removal receipt.

The evidence archives are **ignored by Git**. The cloud migration above records
their remote locations; a Git clone alone does not contain these artifacts.
No datasets or checkpoints have been added to Git. Existing evidence exclusions
remain unchanged: the AP archive omitted the regenerable short-evaluator cache
and its videos, while retaining recorded action/state traces. This cleanup does
not recover evidence that was absent before consolidation.

The mapping covers 1,778 original files from the four completed `outputs/` folders
and the prior README, FirstExperiment and protocols. Raw provider administration
remains separately accessible under `outputs/runpod-setup-20261101/`.

## Verify

From the repository root:

```bash
python SafeLeWM/archive/restore.py --verify
```

This checks archive hashes and every original file's bytes against the manifest.
It performs no simulator, model or numerical experiment computation.

## Restore

Restore into an empty scratch directory; existing destination files are never
overwritten. The destination corresponds to the original `SafeLeWM/` directory.

```bash
python SafeLeWM/archive/restore.py --restore /tmp/safelewm-restored --prefix outputs/anchored-20261101/
python SafeLeWM/archive/restore.py --restore /tmp/safelewm-protocol --prefix AnchoredQualification.md
```

Omit `--prefix` to restore the full pre-cleanup snapshot. Prefixes are original
paths from `manifest.json`, not archive member paths. Some original local copies
were under `remote/`; the manifest restores those exact local paths, not the
runtime paths on the rented host. It includes original source/evidence tarballs
for reconstructing the runtime layout where needed.

Historical scripts are retained unchanged and still refer to their original
protocol and runtime paths. Recover the required files before reuse, and use a
fresh prospective protocol and output directory for any new authorized run.
Restoration itself never launches an experiment or alters a rental.

## Broader hazard screen, 1 November

The later 72-episode screen has its own immutable archive and manifest; it does not
alter the earlier cleanup manifest. `evidence/hazard-screen-20261101.tar.gz`
contains 401 evidence files plus its internal manifest. All 25 source bindings
and all archived bytes were verified locally. The AP memory cache was reused;
its original bytes remain in `evidence/anchored-20261101.tar.gz`.

```bash
python SafeLeWM/archive/restore.py --manifest manifest-hazard-screen-20261101.json --verify
python SafeLeWM/archive/restore.py --manifest manifest-hazard-screen-20261101.json --restore /tmp/safelewm-hazard-restored
```

This manifest restores the new runtime output directory, including frozen source
copies under `outputs/hazard-screen-20261101/sources/`. The concise audit, panel
and preservation receipts are also retained in the ignored local output directory.
See [the result](../results/07-hazard-screen.md) and [protocol](../HazardScreen.md).
The screen is complete; no adapter search or training followed.

## Recurrent adapter pilot, 1 November

`evidence/adapter-pilot-20261101.tar.gz` preserves 6,557 files from both the
interrupted first attempt and the completed corrected run. Every file passed
independent local SHA-256 verification. The archive hash is
`d3a3175ccd82c37145437f3e0913be7c754937f95b1e7417d1ca031a28ef7431`.
It includes complete and partial traces, all candidate weights, optimizer states,
frozen sources, software-test failures, audits, summaries and figures. The original
memory cache remains in the earlier AP archive and was not duplicated.

```bash
python SafeLeWM/archive/restore.py --manifest manifest-adapter-pilot-20261101.json --verify
python SafeLeWM/archive/restore.py --manifest manifest-adapter-pilot-20261101.json --restore /tmp/safelewm-adapter-restored
```

The manifest restores both `outputs/adapter-pilot-20261101/run-v1/` and `run-v2/`,
plus supporting setup and postprocessing evidence. The first attempt is excluded
from the optimizer comparison; its 459 resets remain in resource accounting.
See [the result](../results/08-recurrent-adapter-pilot.md) and
[prospective correction](../AdapterPilotCorrection.md). The corrected pilot is
complete. The rental remains on at Sachin's request; no further experiments follow
from this completed protocol.

The selected recurrent-adapter checkpoints have a separate private Hugging Face
backup pinned by commit. The [receipt](../results/08-checkpoints.json) records its
revision and verified hashes for all 11 files. No checkpoint is committed to Git.


## Matched-budget adapter replication, 1 November

`evidence/replication-20261101.tar.gz` preserves 13,944 files: frozen sources,
complete and budget-truncated traces, all sampled candidate vectors, optimizer
states, exact-quota dispatch ledgers, selection, evaluation, audits and figures.
Every member passed independent local SHA-256 verification. Archive SHA-256:
`07864369c79463880b11ab4729bd9a6abeb67e84b34b8999cba95d7eac383de2`.

```bash
python SafeLeWM/archive/restore.py --manifest manifest-replication-20261101.json --verify
python SafeLeWM/archive/restore.py --manifest manifest-replication-20261101.json --restore /tmp/safelewm-replication-restored
```

The closed packaging logs and completion receipt are separately retained in
`outputs/replication-20261101/closeout.tar.gz`, hashed by the
[preservation receipt](../results/09-preservation.json). The checkpoint bundle has
25 redownload-verified files in the approved private Hugging Face repository;
[its receipt](../results/09-checkpoints.json) pins the revision. The study is complete;
no further experiment follows. Rental shutdown remains manual.

## Manipulation screens, 1 November

`evidence/manip-screen-20261101.tar.gz` preserves 1,773 files from the PushT and Cube
hazard screens under [ManipulationScreen.md](../ManipulationScreen.md): every stage
folder (`engineering-v1`, `long-v1`, `long-v2`, `hazard-v1`, `determinism-v1`,
including the failed Cube `long-v1` start), traces, planning records, freezes, audits,
post-hoc retrieval analyses, figures, frozen panels, dataset inspections, data
receipts, launcher scripts and logs. `evidence/manip-memory-20261101.tar.gz` holds the
two memory-only retrieval caches (PushT 982,507 and Cube 1,608,000 encoded frames)
with their shard records; their `latents.npy` hashes match the cache receipts. The
archive hashes are
`14a98f99287b911a22702c3c6e85ea9a4aebf709e854d255059b36dd1820620d` and
`da28ff40a773ad71a106b59e6586f62cc66ea0bb27ce8860937c461d11cc6cfc`. All 1,797 files
passed independent local SHA-256 verification against both the pod's hash list and
the manifest. Pinned datasets and checkpoints are not included; the data receipts
record their revisions and hashes.

```bash
python SafeLeWM/archive/restore.py --manifest manifest-manip-screen-20261101.json --verify
python SafeLeWM/archive/restore.py --manifest manifest-manip-screen-20261101.json --restore /tmp/safelewm-manip-restored
```

The manifest restores `outputs/manip-20261101/` as it stood on the pod, with the
caches under each task's `memory-v1/`. See the [PushT](../results/10-pusht-screen.md)
and [Cube](../results/11-cube-screen.md) results. Both screens are complete; no
adapter work followed. The L4 pod remains on at Sachin's request.

## Representation and mechanisms, 1-2 November

`evidence/mechanism-20261101.tar.gz` preserves 58,675 files: frozen sources,
all training candidates and optimizer states, the 24 selected adapters, initial
weights and projections, complete and quota-truncated episodes, traces, selection
locks, audit evidence and analysis figures. Every original file passed independent
local SHA-256 verification. Archive SHA-256:
`d86338fc7e9eacb3965e59d4113d4ef5c03ed86c3cee77c59f4d15850dec8f7a`.

```bash
python SafeLeWM/archive/restore.py --manifest manifest-mechanism-20261101.json --verify
python SafeLeWM/archive/restore.py --manifest manifest-mechanism-20261101.json --restore /tmp/safelewm-mechanism-restored
```

Restoration preserves archived member timestamps as well as bytes and permissions,
so the saved-evidence ordering checks can be repeated. The original failed audit,
separate v2 auditor, regression tests and amendment receipt are retained. No
experiment was repeated to resolve the filesystem-clock precision issue.

Nine closed packaging/report files are separately retained and verified in
`outputs/mechanism-20261101/closeout.tar.gz`; the
[preservation receipt](../results/12-preservation.json) records both archives.
The [checkpoint index](../results/12-checkpoints.json) hashes all 24 nominees and
records their loading dependencies. No new checkpoint upload was performed.
See [account 12](../results/12-adapter-mechanism.md) and Rule43 page 12.
The eight-GPU host remains running and available; shutdown is manual.

## PushT distinct-scene screen, 2 November

`evidence/pusht-authority-20261102.tar.gz` preserves 933 files: the scene/memory panel,
retrieval cache, all original and corrected sources, the pre-drawn parameter probes,
completed and failed episode evidence, copied-record provenance, audits and logs.
Every member passed independent local SHA-256 verification. Archive SHA-256:
`7f42b9a3312938911d28a681549657cb3c7b14e758a9e0ff255b0dad088fbb6a`.

```bash
python SafeLeWM/archive/restore.py --manifest manifest-pusht-authority-20261102.json --verify
python SafeLeWM/archive/restore.py --manifest manifest-pusht-authority-20261102.json --restore /tmp/safelewm-pusht-authority-restored
```

Twelve closed report/figure/log files are additionally verified in
`outputs/pusht-authority-20261102/closeout.tar.gz`; see the
[preservation receipt](../results/13-preservation.json). The public dataset and
checkpoint can be reacquired from the pinned hashes/revisions in the data receipt.
The 29 copied completed episodes are duplicate preservation, not additional
interactions. Combined unique work was 73 resets and 16,365 actions, including the
one failed replay attempt. No random parameter probe or training followed the failed
prerequisite gate. See [account 13](../results/13-pusht-authority.md).
The host stays running; rental shutdown remains manual.

## PPO and PushT AP reproduction, 2 November

- `manifest-ppo-4090-20261101.json`: 16,994 independently verified files.
- `manifest-ppo-l4-20261101.json`: 8,754 independently verified files.
- `manifest-pusht-ap-reproduction-20261102.json`: 1,180 independently verified files.

The corresponding archives are under `evidence/`. Verify or restore with
`restore.py --manifest MANIFEST --verify` or `--restore DESTINATION`.
The PPO original timestamp audit failure and its v2 correction are retained.
The AP pre-rollout OSMesa import failure and repair are retained. No experiment
was rerun to improve an outcome. The AP retrieval cache remains on the rental;
its hash, row coverage and pinned public inputs are preserved in the archive.
See `results/14-preservation.json` and `results/15-preservation.json` for receipts.

## PushT qualification with the reproduced baseline, 2 November

`manifest-pusht-validated-20261102.json` indexes 2,419 original files in
`evidence/pusht-validated-20261102.tar.gz`. Every member was independently
hash-verified locally. The audit covers all 329 episodes and 81,502 actions.
The baseline/hazard prerequisite passed, but the sixteen-head useful-authority
gate failed. No optimizer training followed and twelve reserved cases remain unrun.

Five closed report/figure/source files are additionally hash-verified in
`outputs/pusht-validated-20261102/closeout.tar.gz`; the manifest lives inside it.
See [account 16](../results/16-pusht-validated.md) and its
[preservation receipt](../results/16-preservation.json). Both the pre-simulation
pytest failure and corrected launch logs remain in the raw archive.

## Safety-dial library, 2 November

[Account 17](../results/17-safety-dial-library.md) records the completed prospective
development check. The independent audit passed all 772 episode records.
`evidence/safety-dial-library-20261102.tar.gz` contains 4,748 verified files,
423,930,947 bytes, SHA256
`7ebdf6baa52a4acaeb74c00e32931919711ac358dd48326043a26bbff1a18582`.
The manifest is `manifest-safety-dial-library-20261102.json`; the independent
[verification receipt](../results/17-preservation.json) also records the six-member
report bundle. All archive members were hash-checked. No simulator work was done
locally. The single 4090 rental remains on; shutdown is manual.
