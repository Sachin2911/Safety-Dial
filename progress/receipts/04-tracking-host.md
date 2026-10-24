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
the host and installs the same curobo VCS requirement from the unchanged cached
checkout using a process-local Git URL mapping. It then resumes the upstream
BEHAVIOR installer with the same flags. This changes download transport, not the
selected package versions or experiment design. The original interrupted log,
resume log and wheel manifest are retained. No simulator start was charged for
this installation retry.
