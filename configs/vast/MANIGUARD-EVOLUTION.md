# ManiGuard latent evolution

Private Vast preparation template for the current shared-controller study. Select
**ManiGuard latent evolution** from [My Templates](https://cloud.vast.ai/templates/).
The existing **evo safety filter** template is preserved separately.

## Host driver and hardware

[ManiGuard's installation guide](https://nu-ideas-lab.github.io/ManiGuard/docs/getting-started/installation/)
verifies driver 580.x with CUDA 12.x userland and reports startup segfaults with newer
drivers in its tested stack. The requirement is to select a compatible host. Changing
the container's CUDA toolkit does not change the host driver.

This template selects **580.178.04 exactly**. On 19 October 2026, a read-only Vast CLI
query with `driver_version>=580.0.0 driver_version<581.0.0` returned both 580.x and
610/615 hosts. The CLI's driver allowlist query was rejected by the API. Exact equality
returned nine offers, all reporting 580.178.04: eight RTX 4090 and one RTX 3090. Inventory
changes continuously. Keep the exact equality filter and independently inspect the
selected offer's driver before renting. Another 580.x patch can be selected after its
exact query is checked. The 580 branch is source-supported; this particular patch has
not been smoke-tested by this project on a rented GPU.

The remaining filters request one RTX 3090 or RTX 4090 with at least 24 GB VRAM, 32 GB
system RAM, eight effective CPU cores, x86-64, 99% reliability, 200 GB disk and two
direct ports. The CLI also applies its normal verified/rentable defaults. Prefer the
4090 for the first pilot, subject to the approved price and measured episode timing.
These explicit GPU models avoid non-RTX accelerators and the separate Blackwell
environment requirements.

## Image and startup

The image is
`vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310-2026-09-07`:
Ubuntu 22.04, Python 3.10 in `/venv/main`, CUDA toolkit 12.4.1, uv, JupyterLab and SSH.
Docker Hub reported the tag active on 19 October 2026 with digest
`sha256:29af95aab93e5c7f77559f1dfcabf03f407178c1599e7f952521f719a7e1301b`.
The template pulls the dated tag; the digest is recorded verification metadata.

Startup runs `exec /opt/instance-tools/bin/entrypoint.sh`. It installs no research
packages and starts no experiment. Graphics capabilities and headless mode are exposed
through `NVIDIA_DRIVER_CAPABILITIES=all` and `OMNIGIBSON_HEADLESS=1`. No credentials are
embedded in the profile or startup command.

## Preparation after connecting

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
/venv/main/bin/python --version
cd /workspace
git clone https://github.com/Sachin2911/Safety-Dial.git
cd Safety-Dial
```

Follow the repository's current research plan and the upstream installation guide.
Prepare the Python 3.10 simulation environment and model-specific policy environment
separately using uv with explicit environment paths. The simulator stack uses the
pinned OmniGibson/Isaac Sim 4.5 installation and PyTorch 2.6.0/cu124. Download and verify
the BEHAVIOR assets, ManiGuard benchmark, extended-finger Franka asset and required
policy checkpoints. Confirm the extended-finger asset is actually loaded.

Validate simulator startup, headless camera images and policy communication on the
chosen host before comparative episodes. Check native-noise identity and steering
separately as required by the plan. The template is a preparation base; it does not
contain an installed or validated ManiGuard research environment.

Root `scripts/setup.sh` and `scripts/vast/setup.sh` own the older UNISafe platform and
refuse the ManiGuard platform. Root `pyproject.toml` and `uv.lock` also describe earlier
research. Use the intended environment's explicit Python executable and dependencies.

## Template management

The profile is `configs/vast/maniguard-evolution.json`. Its separate template record is
`configs/vast/maniguard-evolution-template.json`; the legacy record remains
`configs/vast/created-template.json`.

```bash
uv run --no-sync python scripts/vast/create_template.py --config configs/vast/maniguard-evolution.json --update --dry-run
uv run --no-sync python scripts/vast/create_template.py --config configs/vast/maniguard-evolution.json --update
```

For initial creation, omit `--update` while the new record is absent. The helper rejects
creation over an existing record and checks template identity before updates. These
commands manage a private template and do not rent or launch an instance.
