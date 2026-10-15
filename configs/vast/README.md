# evo safety filter

Private Vast.ai template for the platform selected on 15 October 2026. Creating the
template does not rent an instance. Select it from My Templates before renting.

The dated base image uses Ubuntu 24.04, Python 3.12 and CUDA toolkit 13.0.3. First boot
installs PyTorch 2.11.0, torchvision 0.26.0 and torchaudio 2.11.0 from the CUDA 13.0 wheel
index, together with Isaac Sim 6.2.0.0 and Isaac Lab's pinned 3.0 source. This is the 3.0
development release line, not a claim of general availability. The public 3.0.0rc1
wheel still requests Sim 6.1, so the template builds Lab's aggregate package from source
revision `6bfde876089661e116dc5a5d9c6805f18127fe32`, using that revision's dependency
overrides. The resolved package versions are saved in `requirements-linux-py312.txt`
and embedded in the template, alongside the overrides and provisioning script. Package
resolution is checked before the template is created.

The template enables JupyterLab and SSH with direct connections and requests 200 GB of
disk. Offer filters request one GPU with at least 24 GB VRAM, 32 GB RAM, eight effective
CPU cores, x86-64, CUDA 13.0 driver capability, driver 580.95.05 or newer, and 99%
reliability. These filters do not establish Isaac Sim renderer compatibility. The
selected 48 GB RTX PRO offer is a suitable capacity target when available. Disk and
bandwidth charges still depend on the offer.

First boot can take substantial time while packages download and Lab builds. No dataset,
checkpoint or research run is downloaded or launched. The script uses `uv` and prepares
the base image's existing `/venv/main` environment. It does not create a project
environment or run Safety-Dial's older setup and lockfile. Migrate the repository's
Python pin and dependencies before using its setup with this platform. Keep one maintained
project environment when integrating the repository; do not create an extra Isaac Lab
project environment alongside it.

In Jupyter, choose **Evo safety filter (Python 3.12)**. In SSH, use
`/venv/main/bin/python` explicitly. Check provisioning with:

```bash
tail -f /workspace/.evo-safety-filter/provision.log
cat /workspace/.evo-safety-filter/versions.json
```

The `ready` file is written only after package-version assertions and a GPU tensor
operation succeed. This checks CUDA access, not PhysX, RTX rendering, cameras or UNISafe.
An Isaac Sim rendering smoke test on the rented host remains necessary. The template
sets `OMNI_KIT_ACCEPT_EULA=yes` for unattended Isaac Sim use, retains the base image's
authenticated Jupyter/portal access, and contains no credentials.

The on-start command embeds the provisioning script and installs an `80-` boot hook
before calling Vast's normal entrypoint. The hook runs after the platform services start,
so installation logs and SSH remain accessible. A lock prevents concurrent installation;
the success marker skips installation on subsequent boots. Failed provisioning leaves
the marker absent and can be retried by restarting or invoking the script directly.

Sources:

- [Isaac Lab 3.0 installation](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/setup/installation/index.html)
- [Pinned source and dependency constraints](https://github.com/isaac-sim/IsaacLab/tree/6bfde876089661e116dc5a5d9c6805f18127fe32)
- [Vast template settings](https://docs.vast.ai/guides/templates/template-settings)
- [Vast base-image startup](https://github.com/vast-ai/base-image)
