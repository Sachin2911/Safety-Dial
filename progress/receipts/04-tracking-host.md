# Task 4 native recording check: host

Rental 57534766, machine 33953 (Taiwan), created 24 October 2026 at
16:06:08 UTC with the established template
`62414e8fca6befba085051de7ed36190`. Two RTX 3090 GPUs, 24 GB each,
driver 580.126.09 verified over SSH, 32 effective CPU cores, about 64 GB RAM,
120 GB container disk. Compute plus selected storage: $0.564444/hour.
Listed download: $0.004/GB; upload: $0.0053333/GB. This host is distinct
from the task 1 results host and the task 3 host; results are not pooled.

The frozen design is `progress/04-tracking-check.md`, with jobs in
`04-tracking-plan.json`. Both lanes have six planned episodes. The first episode
on each lane validates its recording and counts in the twelve. No additional
simulator smoke or repeatability episode is planned.

Preparation started at 16:12:31 UTC using
`scripts/stage2/prepare_vast_host.sh`. This replays the pinned setup only, not
the older experiment queues mentioned in historical comments. Workstation
credentials were not copied to the rented host: public downloads are used for
setup, and evidence returns over SSH for private upload from the workstation.

Resource-only installation adjustment: the remote copy of
`scripts/todo1b/install_behavior_replay.sh` adds `export MAX_JOBS=8` before
building curobo to bound compilation memory on this host. The executed source
archive preserves that exact copy. No evaluator or policy source is patched.
The four dependency repairs from task 3 are applied before admission:
dm_control 1.0.49, mujoco 3.16.0, googleapis-common-protos 1.75.5,
and vhacdx 0.1.0. Full environment freezes and source download manifests are
saved under the run's `environment/` directory.

`launch_tracking.sh` checks revisions and the legacy 256 config, starts one
native policy server per GPU on localhost ports 8000 and 8001, and warms both
before starting the two-lane queue. Probes use no simulator interactions and
their model queries are retained in the logs.

`tracking_watchdog.py` is active with deadline **22:06:08 UTC** (six rental
hours), using the instance's existing provider credential without displaying it.
It stops the instance and retains its disk if evidence preservation is delayed.
No new jobs may enter after **20:06:08 UTC**. `sync_tracking.py` copies the
run's evidence to ignored local storage throughout execution.

Preparation, execution and final preservation status are recorded separately in
`04-tracking-run.md` and `04-tracking-preserve.md` when those stages finish.

## Dependency download recovery

The host's public-package connection was unusually slow. Before any simulator
start, curobo's pip download was stopped after its pinned Git checkout had been
preserved intact. All 43 wheel filenames already selected by pip were downloaded
on the workstation from PyPI, checked against PyPI's SHA-256 manifest, and sent
over SSH. The 148,811,186-byte warp-lang wheel took 23.9 seconds locally; the
host had received about 36 MB after several minutes.

`experiments/scripts/resume_behavior_tracking.sh` rechecks all wheel hashes on
the host and builds curobo from the unchanged cached checkout at `cbaf7d32`.
Re-cloning the partial Git object store failed because it omits historical blobs;
the pinned working tree passed `git diff` and is built directly, with the local
source path retained in pip provenance. An installation-only change to
`OmniGibson/setup.py` requires the already built `nvidia-curobo==0.0.0` instead
of fetching its VCS URL again. The exact one-line patch is retained under the
run source evidence; no simulator runtime, evaluator or policy file is modified.
The upstream installer resumes with the same flags. Subsequent pip installs are
constrained to the explicit `==` version pins in task 3's package freeze, excluding
the certificate bundle. Conda packages represented by local build URLs are not
covered by that constraint file. The final environment comparison records remaining
differences. A parallel HTTP
range download completed the large wheel and passed its full SHA-256 check; the
redundant SSH transfer was stopped. The original interrupted log,
resume log and wheel manifest are retained. No simulator start was charged for
this installation retry.

## Final preparation checks

Preparation finished at 17:40:58 UTC, about 89 minutes after setup
began. The simulator check reported numpy 1.26.4, torch 2.6.0+cu124, OmniGibson
3.7.2 and Spot 2.16, with both GPUs available. The OpenPI freeze is byte-identical
to task 3's. After normalizing temporary installer directory names, the behavior
freeze differs only in curobo's local-source provenance described above; the
Isaac Sim wheel filenames and SHA-256 values match. The installation metadata
patch makes ManiGuard's runtime version include a dirty-date suffix, while its
commit and all runtime source remain pinned.

Model-only warm-up was overlapped with the simulator installation. Each server
answered two queries, four in total, with no simulator interactions. The official
two-lane queue started at **17:41:14 UTC** after the revision, config, environment
and download-manifest checks passed.
