# Native-noise controller execution and reconstruction

This is the new 19 October study. The historical phase-two source manifest supplies
code and asset identifiers. Its coordinator and earlier studies are not launched.
`development-v1.json` freezes development before behavior. `platform-v1.json` through
`platform-v3.json` preserve the measured infrastructure repairs and their costs.
A capacity-selected pilot protocol is required before physical search can start.

The authorized instance is 56777435. Physical acquisition ends at 02:30 UTC on
20 October 2026 and the supervised deadline watchdog requests stop, retaining disk,
at 03:00 UTC. These absolute times do not reset when a coordinator restarts.
The user authorized the internal watchdog after provider schedule readback failed.
No provider stop job was removed or postponed. A later rental or restart requires
new authorization.

## Reconstruct the environments and assets

Check out this study branch at the commit recorded in the artifact receipt under
`/workspace/Safety-Dial`. The historical editable installations require that path.
Use uv to create `data/maniguard/envs/tools` with Python 3.10, then install
`tooling-requirements.txt` with `uv pip install --python` for that interpreter.
Do not synchronize the root project environment into either research environment.

Authenticate to the private Hugging Face repositories using the environment token.
Download `runtime.tar.gz` from the dataset
`Sachioster/safety-dial-phase2-handoff-20261018`, revision
`83eedc3181de0849b37f1a0982c397c3fc3b69e8`. Its required SHA-256 is
`cca645c08ea7cb6f67e2f5939cbe8cf37aac8d50d7610e232020edb0200fd135`.
`experiments/scripts/maniguard_latent_restore.py --archive PATH --staging NEW_PATH`
verifies that hash, extracts only the two interpreter environments into staging and
refuses to overwrite existing installations. It restores no historical jobs.

Install uv-managed CPython 3.11.16 under `/.uv/python_install`, keeping the historical
`cpython-3.11-linux-x86_64-gnu` interpreter symlink available. The verified simulator
environment contains CPython 3.10.22, Torch 2.6.0+cu124 and NumPy 1.26.4. The policy
environment contains JAX 0.5.3 and NumPy 1.26.4. Preserve their separate dependencies.

Clone ManiGuard and OpenPI into the canonical `third_party` paths, then check out the
exact revisions in `../phase2/sources.json`; initialize ManiGuard's pinned BEHAVIOR
submodule. Install only the pinned OpenPI client editable package into the tools
environment with `uv pip install --python ... --no-deps -e
third_party/openpi/packages/openpi-client`, plus its declared client dependencies.
Do not install the complete simulator or policy dependency sets into tools.

For each development family, run `experiments/scripts/maniguard_latent_fetch.py
--family jar|clutter --kind both --recovery NEW_PATH`. The downloader verifies every
benchmark and policy file against its immutable upstream identifier and digest.
Acquire BEHAVIOR 3.7.2rc1 and robot assets through their authorized upstream download
flow and install the extended-finger Franka files pinned in the source manifest.
The licensed bundles and encryption key are deliberately absent from study backups.
Restore the pinned CLIP ViT-B-32 asset named by `../phase2/encoder.json`; its SHA-256
is `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.

The matching NVIDIA graphics userspace is in `/opt/nvidia-drivers/lib64`. Keep the
host-injected driver and CUDA library. Supply `__EGL_VENDOR_LIBRARY_FILENAMES` as
the absolute path to `nvidia-egl-vendor.json`. Do not prepend the graphics directory
to `LD_LIBRARY_PATH`: that loaded a second CUDA driver library in this environment.
The runtime also needed `libglu1-mesa` and `libxt6`. The actual bundled Kit Vulkan
enumeration maps renderer index zero to simulation GPU UUID
`GPU-0bfb3ed5-af55-472e-3199-058727ff80a7`; policy CUDA device one is
`GPU-4f452774-6f8b-6fcc-4ce1-2d4d190e6d07`. Recheck this on another host.

## Execute and resume

Serve the verified frozen checkpoint with the policy interpreter and
`experiments/scripts/maniguard_latent_serve.py --family FAMILY --port 18000
--ledger PATH`. Set `CUDA_VISIBLE_DEVICES=1`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MANIGUARD_SKIP_OMNIGIBSON_PATCH=1` and a
persistent JAX compilation cache. Manage the service and coordinator with supervisor.
The ordinary server rejects audit overrides. Use its explicit `--audit` mode only
for the prospective saved-input decoder audit, with a distinct query ledger.

`maniguard_latent_run.py --run RUN --protocol PROTOCOL --jobs JOBS_JSON` executes
immutable serial development jobs. The queue preserves PID plus process start ticks,
worker exit receipts and all costs. A surviving worker is observed on coordinator
resume. Invalid attempts require a specific diagnosed repair before replacement.
Use `maniguard_latent_audit.py --episodes EPISODE1 EPISODE2 --output NEW_DIRECTORY`
against each matching family's audit server. It consumes at most 160 calls per family
and refuses unresolved requests rather than duplicating them.

After measured timing and the full authority audit, freeze a capacity-selected pilot
with exact training panels, independent blocks and family. The pilot entry point
`maniguard_latent_pilot.py --run RUN --protocol SELECTED_PROTOCOL` shares eight initial
episodes, evaluates sixteen descendants per optimizer, seals all four finalists and
then executes sixteen independent comparison episodes. It checkpoints optimizer RNG,
ancestry and candidate weights, with verified private backups after each generation.
No unit test or short engineering episode supplies comparative fitness.

The private output repositories are the dataset
`Sachioster/maniguard-latent-evolution-runs` and model
`Sachioster/maniguard-latent-evolution-controllers`. Each append-only stage has exact
remote revisions and SHA-256 receipts. Download the named archives at those revisions,
verify their hashes and unpack into a new staging directory. Dataset archives retain
raw traces, images, videos, source snapshots and ledgers. Model archives retain the
controller weights, optimizer state, basis and configs. Neither contains base VLA
weights or licensed simulator asset bundles. Receipts distinguish untrained initial
controllers from physically selected finalists.

Checks use explicit interpreters and `PYTHONPATH=experiments`. The tools interpreter
runs the core, queue and pilot tests; the policy interpreter runs
`test_maniguard_latent_policy.py` with `JAX_PLATFORMS=cpu`. Ruff checks the new latent
helper, script and test files. The live physical verifier independently replays causal
features, native RNG chains, decoded chunks and executed actions.
