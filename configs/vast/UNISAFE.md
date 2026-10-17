# Preserved UNISafe platform

These instructions describe the Python 3.12 / CUDA 13 platform used for the UNISafe
study. The current private template now targets ManiGuard preparation; see
[README.md](README.md). The original image and settings are saved in
[unisafe-platform.json](unisafe-platform.json). Use that image for UNISafe reproduction.
Do not run this installer in the ManiGuard Python 3.10 base.

Private Vast.ai template for the platform selected on 15 October 2026. Select it from
[My Templates](https://cloud.vast.ai/templates/) before renting. Updating the template
changes the defaults for future rentals; it does not alter a running instance.

## Lightweight instance startup

The dated base image provides Ubuntu 24.04, Python 3.12, CUDA toolkit 13.0.3 and Vast's
standard tools, including `uv`. The template enables JupyterLab and SSH with direct
connections and requests 200 GB of disk. Startup calls the image's normal entrypoint
only. It does not embed an installer, create a research boot hook, or install PyTorch,
Isaac Lab, Isaac Sim or the research requirements. The base image still needs to be
pulled and its standard access services initialized.

Offer filters request one GPU with at least 24 GB VRAM, 32 GB RAM, eight effective CPU
cores, x86-64, CUDA 13.0 driver capability, driver 580.95.05 or newer, and 99% reliability.
These filters do not establish Isaac Sim renderer compatibility. The selected 48 GB
RTX PRO offer is a suitable capacity target when available. Disk and bandwidth charges
depend on the offer. The template contains no credentials and retains the base image's
authenticated access.

## Manual setup after connecting

Clone the repository, then explicitly start the separate installation step:

```bash
cd /workspace
git clone https://github.com/Sachin2911/Safety-Dial.git
cd Safety-Dial
bash scripts/vast/setup.sh --dry-run
bash scripts/vast/setup.sh
bash scripts/vast/setup.sh --check
```

For an existing clone, enter that directory and run the last command. Run it as root
on the Vast instance. The script installs the required system libraries and uses `uv`
with the saved `requirements-linux-py312.txt` and matching `uv-overrides.txt`. Package
downloads and building Lab happen during this manual step, after instance access is ready.

The pinned installation target is PyTorch 2.11.0, torchvision 0.26.0 and torchaudio 2.11.0
with CUDA 13.0 wheels, Isaac Sim 6.2.0.0 and Isaac Lab 3.0 source revision
`6bfde876089661e116dc5a5d9c6805f18127fe32`. This is the 3.0 development release line.
The public 3.0.0rc1 wheel inspected during initial setup still requests Sim 6.1, so the
manual script builds Lab's aggregate package from the pinned source and uses its tested
dependency overrides. The snapshot was resolved before the original template was created.

The installer prepares the base image's existing `/venv/main` environment. It does not
create another project environment, run the repository's older root setup, download
research datasets or checkpoints, or launch experiments. The root Python pin is now 3.12, and `bash scripts/setup.sh` dispatches here on Vast.
The platform snapshot and overrides are the dependency authority for this installation.
The root `pyproject.toml` and `uv.lock` still describe the earlier research dependencies;
do not run root `uv sync` or an automatically syncing `uv run` against this platform.
Integrating research dependencies is a separate step. Use `/venv/main/bin/python` now.

Inspect installation progress and the resulting versions on the rented instance:

```bash
tail -f /workspace/.evo-safety-filter/setup.log
cat /workspace/.evo-safety-filter/versions.json
```

After successful manual setup, choose **Evo safety filter (Python 3.12)** in Jupyter;
refresh the kernel list if needed. In SSH, use `/venv/main/bin/python` explicitly.
The `ready` marker contains a fingerprint of the installer, checker and dependency files.
Every rerun checks all pinned distributions, Isaac Lab's source revision, core PyTorch
imports and a CUDA matrix multiplication before skipping package installation. Changed
inputs, missing system libraries or package drift trigger reconciliation. Jupyter kernel
registration is refreshed on every successful setup. A lock rejects concurrent installers.
Failures during installation remove the marker; rerun the same command to retry. Cached
package downloads are reused. Installation never retries automatically at instance boot.

`--dry-run` checks Python, OS, architecture, tools and GPU driver, then prints the plan
and free disk space. It does not resolve or download packages. `--check` verifies the
installed snapshot and CUDA without installing or writing setup state. Neither flag
establishes simulator or camera compatibility. The checker stops before installation when all visible GPUs are known to lack
RT cores, including A100. An accepted GPU name is not proof of rendering compatibility;
RTX-capable hardware still needs a simulator and camera smoke test.

Secrets remain in the inherited environment. This installer does not source `.env`, log
credentials, persist logins, or copy environment variables into `/etc/environment`. No
GitHub or Hugging Face token is required for these public platform dependencies.
Ubuntu libraries are installed only when missing; NVIDIA drivers are never installed.
`uv pip install --python /venv/main/bin/python` retains unrelated base-image packages,
including Jupyter. This is a pinned platform installation, not a hermetic container image.
The requirements snapshot is version-pinned but does not contain artifact hashes.

`versions.json` records the validated versions and scope of the checks. It marks simulator
and camera validation as not run. The selected stack remains a development release.
On 15 October 2026 the updated installer passed syntax checks, package-validator tests
and host rejection checks here; the detected GPU was an A100 40 GB. No package installation or
simulator execution was performed as part of that script-development check.

The template retains `OMNI_KIT_ACCEPT_EULA=yes` for unattended Isaac Sim use after manual
setup. Package resolution and template settings were checked, but installation and GPU
execution have not yet been tested on a rented host. PhysX, RTX rendering, cameras and
UNISafe require their own smoke tests; a successful CUDA tensor check does not verify them.

## Maintaining the template

Use `uv run --no-sync python scripts/vast/create_template.py --update --config configs/vast/unisafe-platform.json` to restore the
recorded template. Add `--dry-run` to inspect settings without API calls. Omitting
`--update` creates a template only when no private template with the name exists.
`created-template.json` stores the ID and current hash returned by Vast.

Sources:

- [Isaac Lab 3.0 installation](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/setup/installation/index.html)
- [Pinned source and dependency constraints](https://github.com/isaac-sim/IsaacLab/tree/6bfde876089661e116dc5a5d9c6805f18127fe32)
- [Vast template settings](https://docs.vast.ai/guides/templates/template-settings)
- [Vast base-image startup](https://github.com/vast-ai/base-image)

Installer verification (no platform installation required):

```bash
bash -n scripts/vast/setup.sh scripts/setup.sh
/venv/main/bin/python -m unittest discover -s scripts/vast/tests -v
```

Environment targeting follows [uv's documented Python selection](https://docs.astral.sh/uv/pip/environments/).
See [NVIDIA's GPU requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
for the distinction between CUDA-capable hardware and supported RTX rendering hardware.
