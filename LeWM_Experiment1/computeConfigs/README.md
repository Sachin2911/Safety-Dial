# LeWM Experiment1 compute template

`config/vast-template.json` is the source for a private Vast launch template.

```text
computeConfigs/
├── compute.py       # Everyday command
├── README.md
├── config/          # Vast template, session policy and HF destination
├── image/           # Dockerfile, dependencies, build/publish tools and runtime check
└── scripts/         # Controller helpers, remote runner, uploader and template registration
```

Lifecycle tests live separately in `../tests/compute/`. The dependency lock and
image validation script are needed for reproducible releases. Generated Python
caches can be removed at any time.

Saved template: [LeWM Experiment1 - AP runtime v0.1.0](https://cloud.vast.ai/?template_id=1a68bef5f936a0470d984fcafcb10382).

Image: [`ghcr.io/sachin2911/lewm-ap:0.1.0`](https://github.com/users/Sachin2911/packages/container/package/lewm-ap), published privately on 4 November 2026.

The template pins the published digest:

```text
sha256:7a63a5d1d11c429a82b3bb0cfef6d1e48e61ee8c28680ef0e5dd83950428bace
```

## Compute commands

Use `compute.py` for the normal lifecycle. It uses Vast's API, OpenSSH and the
private HF dataset. Local Python needs only the standard library, Python 3.11 or
later, Git and SSH. Worker code is transferred into the existing image over SSH;
changing the controller does not require rebuilding the image.

| Command | Effect |
| --- | --- |
| `search` | Return three offers ranked by estimated session cost, within every hardware filter |
| `up` | Check credentials, rent one container, wait for SSH, deploy helpers and validate the runtime |
| `status` | Read rental, job, progress, backup and estimated compute-cost information |
| `ssh` | Open a shell; `ssh -- nvidia-smi` runs a quoted diagnostic command |
| `run` | Transfer a source snapshot, protocol and JSON config; start a detached, time-limited job |
| `logs` | Read recent stdout/stderr; use `--run-id` when there are multiple runs |
| `stop` | Request job termination, preserving outputs; the rental remains active |
| `sync` | Queue an upload for a running job, or retry final verification for an idle job |
| `recover` | Find an uncertain rental by its unique label and finish setup without renting again |
| `destroy` | Require idle jobs and verified backups, check HF manifests independently, then delete the rental |

The only local runtime state is under ignored `.state/`, including `instance.json`
and SSH host keys. No credentials are stored there. Each operation checks the saved
instance ID and label; the worker additionally checks the remote session identity.
Read-only status does not rewrite session state.

The ignored root `.env` or process environment supplies `HF_TOKEN`. For the private
image, set `GHCR_READ_TOKEN` to a classic GitHub token with `read:packages` access,
or use the existing authenticated GitHub CLI login with package pull access. An
explicit token takes precedence; the CLI fallback is read into memory only during
launch preflight. `VAST_API_KEY` can instead come from the existing Vast CLI key
file. The SSH key configured in `config/compute-policy.json` must already be registered
with Vast. Preflight failures occur before a rental request.

## Environment and credential injection

Vast applies the non-secret environment settings already saved in the template.
The controller loads credentials when needed; adding a key to the local `.env`
does not automatically put it in the instance environment. No manual `export` is
needed for credentials that the controller supports. Process environment values
take precedence over the ignored repository-root `.env`.

| Setting | Source | Where it is used |
| --- | --- | --- |
| `WORKSPACE`, `HF_HOME`, Python output, thread limits, MuJoCo/OSMesa and SDL settings | `config/vast-template.json` | Vast injects these runtime variables into the container |
| `VAST_API_KEY` | Process environment, root `.env`, or the existing Vast CLI key file | Local controller authenticates to Vast; not forwarded to the job |
| `GHCR_READ_TOKEN` | Process environment or root `.env`; otherwise authenticated `gh` login | Controller verifies the private image, then sends the pull credential to Vast in `image_login` |
| `HF_TOKEN` | Process environment or root `.env` | Local HF preflight; `run`/`sync` pass it over SSH stdin to the upload worker, which keeps it in memory |
| `LEWM_RUN_DIR`, `LEWM_CONFIG`, `LEWM_PROTOCOL`, `LEWM_SEEDS`, helper `PYTHONPATH` | Remote runner and submitted run specification | Injected into each experiment process |
| `WANDB_API_KEY`, `GITHUB_TOKEN` and other local `.env` entries | Local credentials, if present | Not automatically forwarded; the controller does not read `GITHUB_TOKEN` directly for registry access, which falls back to `gh auth token` |

The experiment process filters inherited token, secret, password and API-key
variables. It therefore does not receive the HF upload credential. Public asset
downloads can use the configured HF cache, but private checkpoint downloads inside
the experiment and authenticated W&B logging need explicit integration before a
run depends on them. Simply adding these tokens to the template would not bypass
the experiment's environment filter.

Keep the root `.env` local and the template free of secrets. Do not paste token
values into a command, manifest, receipt or committed configuration. When checking
authentication, inspect presence or use the controller's preflight instead of
printing `gh auth token` or dumping the environment. Changes to non-secret template
settings follow the registration procedure below so local and saved versions match.

## Launch and job workflow

Example interface, using explicit session limits:

```bash
cd LeWM_Experiment1/computeConfigs
python3 compute.py search --hours 2 --max-hourly 1 --download-gb 20 --upload-gb 2
python3 compute.py up --hours 2 --max-hourly 1 --download-gb 20 --upload-gb 2 --budget-usd 3
python3 compute.py status
```

`up` is a billable action when executed. The example is not an experiment launch
instruction or a new allocation. Set the intended limits before invoking it.
`--download-gb` means additional inputs; the estimate also includes the image pull
and one full read-back of `--upload-gb`. Hourly cost includes the requested disk.
Reported network speed and the cost estimate are planning inputs, not guarantees.
The budget check does not cap provider billing: idle instances continue charging
until explicit destruction. The job deadline reserves five minutes for final sync
by default; increase `final_sync_reserve_seconds` for large artifacts.

For a prospectively specified experiment, replace these example source/config/
protocol paths with its actual files:

```bash
python3 compute.py run --source ../code --config ../config.json --protocol ../protocol.md \
  --seed 0 --max-seconds 3600 -- python main.py
python3 compute.py logs
python3 compute.py status
python3 compute.py sync
python3 compute.py destroy
```

No scientific experiment entrypoint or protocol is created by these helpers.
`--source` must be inside this repository. It packages non-ignored code/text files,
excluding secrets, caches, outputs and checkpoints. The exact source archive,
configuration, protocol, seeds, Git revision, image digest and host are preserved
with the run. The command runs relative to the source snapshot and receives
`LEWM_RUN_DIR`, `LEWM_CONFIG`, `LEWM_PROTOCOL` and `LEWM_SEEDS`. Bare `python` is
resolved to `/venv/main/bin/python`. Cloud tokens remain with the uploader and are
removed from the experiment subprocess environment.

The detached supervisor survives SSH disconnection. A separate `timeout` wrapper
bounds the job even if the supervisor crashes. Child processes must remain in the
job's process group. A failed or interrupted job is recorded and backed up, never
automatically rerun. The finite runtime covers launch preflight as well as execution.

Use the injected helper for periodic artifact publication:

```python
from run_io import publish_artifact, report_progress

publish_artifact("metrics/chunk-0001.jsonl", data=encoded_rows)
publish_artifact("artifacts/checkpoint-0100.pt", source=closed_checkpoint_path)
report_progress(completed=10, total=100, unit="episodes")
```

Each published name is immutable. The helper copies a completed file, atomically
publishes it, and marks it ready for upload. The uploader sends ready files every
five minutes and on request. Logs are rotated into immutable chunks automatically.
Files written directly under the run's `metrics/`, `logs/` or `artifacts/` directories
are included at finalization after the process exits, but are not periodically
uploaded until marked ready. Data written elsewhere is outside this backup contract.
Use unique filenames for checkpoints and metric chunks; source datasets and caches
belong outside the run directory. Output transfer allowances are shared across the
session's runs, and exceeding the allowance blocks finalization.

On closeout, the uploader snapshots the files, refuses conflicting remote bytes,
commits a hash manifest, and streams fresh downloads at the returned HF revision to
verify every file. `destroy` also downloads the manifest independently from this
workstation. Verified immutable closeouts are reused rather than reuploaded.

If boot fails, use `recover` or explicitly `destroy` the unused rental. If a create
reply is lost, use `recover`; creation is never blindly retried. If a job or backup
fails, inspect `status`/`logs`, use `stop` if needed, then `sync`. Upload errors leave
the rental and local output intact. A disconnected worker, active job or unverified
backup blocks destruction of a rental with submitted jobs.

Controller tests use fake providers and processes, with no simulator imports:

```bash
../../.venv/bin/python -B -m pytest -p no:cacheprovider -q ../tests/compute
```

Validation on 4 November 2026: 51 orchestration tests passed; a live read-only Vast
search returned eligible offers; a 1,231-byte upload of four infrastructure files
passed full HF read-back and idempotent closeout at
[`d4ae2166f22c505627f23ef442809a8c430ae96c`](https://huggingface.co/datasets/Sachioster/lewm-experiment1/tree/d4ae2166f22c505627f23ef442809a8c430ae96c/infrastructure-checks/transfer-check-20261104T121235Z-c795a2).
The controller's independent manifest download check also passed at that revision.
These storage checks started no rental or scientific run.

The first timed `up` check passed on 4 November 2026, instance `59105596`, machine
`10447`, one RTX 4090, using the pinned image and 200 GB disk. Timing starts at
controller invocation and ends after the GPU and rendering checks:

| Stage | Seconds |
| --- | ---: |
| Credential preflight and offer search | 4.62 |
| Rental allocation | 1.33 |
| Image/container startup until successful SSH | 158.00 |
| Worker deployment and prerequisite check | 7.95 |
| Runtime validation | 10.47 |
| **Total to ready** | **182.37** |

The host advertised 1,884.2 Mbps download. Image-layer pulls were observed; the
amount of previously cached data was not measured. This is one observed startup,
not a guaranteed cold-start time. Readiness included CUDA float32/bfloat16,
torchvision NMS, AP imports, one synthetic OSMesa frame and dummy SDL, with zero
physics steps or benchmark episodes. No checkpoints or datasets were downloaded.
The worker's final status confirmed no submitted jobs and readiness for destruction.
The hourly rate was $0.52296 including disk. Estimated cost through readiness was
about $0.062 including a full image pull; provider billing and any subsequent idle
time are separate. Timestamped detail is in ignored
`.state/startup-timing-20261104T124539Z.json`.
The user then authorized destruction through `compute.py destroy`; Vast confirmed
instance `59105596` was absent. This test rental is no longer active.

## Settings

| Setting | Initial value |
| --- | --- |
| Access | Direct SSH, using the account's registered SSH public key |
| GPU selection | One RTX 4090 on a verified amd64 host |
| Driver | At least 560.35.05 and advertised CUDA support of at least 12.6 |
| CPU | At least 8 effective vCPUs |
| RAM search filter | At least 64GB |
| Disk | 200 GB |
| Network search floor | 1,000 Mbps download |
| Historical reliability filter | At least 0.98, not an uptime guarantee |
| Startup | `/opt/instance-tools/bin/entrypoint.sh` |
| Python | Use `/venv/main/bin/python` explicitly |
| Rendering settings | OSMesa/llvmpipe and dummy SDL, matching the AP reference |
| Hugging Face cache | `/workspace/LeWM_Experiment1/cache/huggingface` |

AP and OSMesa are installed in the image. Startup does not fetch models or start
research jobs. Rental shutdown remains manual.

## Custom image

`image/Dockerfile` extends `vastai/pytorch:2.7.1-cuda-12.6.3-py311-22.04` by digest,
adds OSMesa, and installs
AP revision `984327424ca7a2e52c75a8279197750d9d0b7c8e`. Source stays in
`/opt/ap-reference/code/ap`; Python stays in `/venv/main`. Datasets, checkpoints,
retrieval caches and experiment code remain outside the image.

Build on a rented Vast VM with Docker and the NVIDIA Container Toolkit:

```bash
bash image/build_image.sh
```

The saved `image/requirements.lock` pins all 132 Python packages apart from the separately
pinned AP checkout. If the lock is absent, the script first resolves the authors'
direct requirements and exports it. The release build checks that installed packages
match the lock. The lock does not claim bitwise reproducibility of Ubuntu downloads;
the published image digest identifies the actual runtime, and its OS package list
is recorded inside `/opt/lewm-runtime/os-packages.txt`.

With a package-publishing token in the remote `GHCR_TOKEN` environment variable:

```bash
python3 image/publish_image.py
```

Publishing first checks the image's CUDA kernels, torchvision operator and one
synthetic OSMesa frame. These checks execute zero benchmark episodes and no physics
steps. The token is supplied to Docker through an environment-backed credential
helper, outside the image and build context. It then verifies registry pulls by tag
and digest, checks the image identity, and confirms private package visibility.

Release validation passed on Vast instance `59092743`, an RTX 4090 VM with NVIDIA
driver 575.51.03: Python 3.11.14, AP imports, dependency consistency, CUDA float32 and
bfloat16, torchvision CUDA NMS, OSMesa rendering and dummy SDL. One synthetic frame,
zero physics steps and zero benchmark episodes were executed. The registry manifest
was independently checked from the workstation. Image size is 9.28 GB compressed
and 18.06 GB unpacked; cache reuse and host bandwidth determine startup time.

## Private image access

Vast requires registry credentials to pull this private image. The controller sends
the credential through the launch request's `image_login`, loading it from
`GHCR_READ_TOKEN` or the authenticated GitHub CLI login. It stays out of tracked
configuration, local state and command arguments. The template contains no registry password.
See [Vast private image launches](https://docs.vast.ai/api-reference/creating-instances-with-api#using-private-docker-images)
and [GitHub registry authentication](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-to-the-container-registry).

Verify the saved template without creating a receipt file:

```bash
python3 scripts/register_template.py --verify --template-hash 1a68bef5f936a0470d984fcafcb10382
```

After changing `config/vast-template.json`, use `--update` with the current template hash.
Vast creates a new version and hash; keep the returned link here. This does not
rent an instance or alter existing rentals. The custom image passed validation inside
the build VM and the timed Vast container launch described above.
When changing template versions, also update `template_hash` in `config/compute-policy.json`.

## Experiment 1 output storage

The selected private dataset is
[Sachioster/lewm-experiment1](https://huggingface.co/datasets/Sachioster/lewm-experiment1).
`config/storage.json` records its destination and the run directory on the rented instance.
The controller passes this configuration to the remote runner and uploader.

Each run should write under `/workspace/LeWM_Experiment1/runs/<run-id>/` and upload
completed files to `runs/<run-id>/` in the dataset. Use unique run IDs, staged uploads
of closed files, and a final manifest with file hashes and the exact HF commit.
Checkpoints, results and logs belong here; public input datasets and disposable
download caches remain outside the upload directory. A completed remote upload must
be verified before the controller reports the instance ready for destruction.
Destroying a rental remains an explicit user-directed operation.
