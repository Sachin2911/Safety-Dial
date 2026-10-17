# evo safety filter: ManiGuard base

The private Vast.ai template now targets ManiGuard preparation, selected on 17 October
2026. Select **evo safety filter** from [My Templates](https://cloud.vast.ai/templates/).
Template updates change future rental defaults. Existing instances keep their image.

## Image and startup

The dated `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310-2026-09-07`
provides Ubuntu 22.04, Python 3.10, CUDA toolkit 12.4.1, `uv`, JupyterLab and SSH.
Its Docker Hub digest was verified on 17 October and is recorded in
[evo-safety-filter.json](evo-safety-filter.json). The template uses the dated tag;
the recorded digest is verification metadata, not an enforced digest pull.

Startup only runs `exec /opt/instance-tools/bin/entrypoint.sh`. Research dependencies,
assets, checkpoints and experiments remain separate manual steps after connecting.
`NVIDIA_DRIVER_CAPABILITIES=all` exposes graphics capabilities and
`OMNIGIBSON_HEADLESS=1` selects headless operation. Neither proves camera compatibility.
No credentials or custom research boot hook are embedded in the template.

## GPU and host selection

The template requests one GPU with at least 16 GB VRAM, 32 GB system RAM, eight effective
CPU cores, x86-64, 99% reliability, 200 GB disk and two direct ports. Disk remains at
200 GB to leave room for assets, environments, weights and outputs. ManiGuard documents
80 GB free as a minimum; training datasets can require substantially more.

CUDA driver capability is lowered from 13.0 to 12.4. The driver range is
`550.54.15 <= version < 581.0.0`: the lower limit follows CUDA 12.4 Update 1's
corresponding Linux driver, and the upper limit follows ManiGuard's observed startup
problems on drivers newer than 580. NVIDIA permits some older drivers through CUDA
minor-version compatibility; this template uses a more conservative lower limit.
ManiGuard specifically verifies 580.x. Prefer that branch for the first smoke test;
the wider range is a candidate range, not a claim that every driver was validated.

The compute-capability selection `[750,860,890]` covers Turing, Ampere and Ada candidates
for the default PyTorch cu124 environment. It excludes A100/H100 and Blackwell from this
profile. Check the actual GPU model for RT cores and run a simulator/camera smoke test
before a batch. Compute capability alone does not establish rendering support.

This admits older-driver RTX 3090/4090 offers and 16 GB options such as RTX 4080,
RTX A4000 and RTX 4060 Ti. ManiGuard reports testing 3090, 4080 and 4090. Its authors
report that 16 GB fits their single-card evaluation pipeline, with pi0/pi0.5 peaking
around 13 GB. This is an evaluation sizing claim, not a training-memory guarantee.
Prefer a 24 GB 3090/4090 when prices are similar for additional headroom.

The change broadens GPU models, but does not guarantee more offers at every moment.
A read-only market check on 17 October found 59 offers under the old requirements
restricted to the same three compute capabilities, 80 after lowering CUDA/VRAM without
a driver ceiling, and 51 with the conservative ManiGuard driver ceiling. Inventory
changes continuously. The old unrestricted template also listed incompatible compute
GPUs and Blackwell cards, so its total was not a usable ManiGuard count.

Blackwell RTX 50-series requires a different environment: ManiGuard prescribes PyTorch
2.7.0 with cu128 instead of its default 2.6.0/cu124. Those GPUs can be revisited with a
separate profile after validation. A100/H100 cannot supply the required RTX rendering.

## Manual preparation after connecting

```bash
cd /workspace
git clone https://github.com/Sachin2911/Safety-Dial.git
cd Safety-Dial
/venv/main/bin/python --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

Follow [ManiGuard's installation guide](https://nu-ideas-lab.github.io/ManiGuard/docs/getting-started/installation/)
for the required components. The simulator uses Python 3.10, OmniGibson's pinned Isaac
Sim 4.5 and PyTorch 2.6.0/cu124. The inspected BEHAVIOR submodule revision is
`88454bd04f75dc57c00ab1f1a00bcde1ff505950`. Policy training and serving use model-specific
environments. Prepare Python package dependencies with `uv` targeting each environment
explicitly; upstream's conda/pip commands need adaptation to this repository's uv policy.
A complete ManiGuard installer or dependency lock has not been implemented here.

The BEHAVIOR assets, ManiGuard benchmark and extended-finger Franka asset are separate
required downloads. Check the extended-finger asset path before evaluation: upstream
warns that a missing asset silently loads the stock hand and breaks learned grasps.
After installation, validate headless rendering, camera observations and policy-server
communication on the rented machine before running an experiment.

Do not run `bash scripts/vast/setup.sh` or `bash scripts/setup.sh` to install ManiGuard.
Those scripts still own the completed UNISafe study's Python 3.12 / CUDA 13 stack.
They reject `SAFETY_DIAL_PLATFORM=maniguard` before installing anything. The root
`.python-version`, `pyproject.toml` and `uv.lock` also remain for earlier research;
do not root-sync them into `/venv/main` or a ManiGuard environment.

## Preserved UNISafe setup and template management

[UNISAFE.md](UNISAFE.md) and [unisafe-platform.json](unisafe-platform.json) preserve the
previous platform instructions and image. The existing installer, checker and Python
3.12 dependency files retain that platform's pins.

Inspect and update the current private template from the local development environment:

```bash
uv run --no-sync python scripts/vast/create_template.py --update --dry-run
uv run --no-sync python scripts/vast/create_template.py --update
```

`created-template.json` records the template ID and current hash. To restore the UNISafe
image for future rentals, use `--config configs/vast/unisafe-platform.json` with the same
commands. Neither template operation installs packages or rents an instance.

Local validation covers the template update helper and existing UNISafe checks:

```bash
bash -n scripts/vast/setup.sh scripts/setup.sh
uv run --no-sync python -m unittest discover -s scripts/vast/tests -v
```

The image metadata and offer queries were checked, but this downgraded image and
ManiGuard have not been executed on a rented GPU as part of this change.

Sources:

- [ManiGuard system requirements and driver observations](https://nu-ideas-lab.github.io/ManiGuard/docs/getting-started/installation/)
- [Pinned BEHAVIOR installer](https://github.com/StanfordVL/BEHAVIOR-1K/blob/88454bd04f75dc57c00ab1f1a00bcde1ff505950/setup.sh)
- [CUDA 12.4 Update 1 driver versions](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/index.html)
- [Vast base images and Python versions](https://github.com/vast-ai/base-image)
- [Published image tag](https://hub.docker.com/r/vastai/base-image/tags?name=cuda-12.4.1-cudnn-devel-ubuntu22.04-py310-2026-09-07)
- [Vast template settings](https://docs.vast.ai/guides/templates/template-settings)
