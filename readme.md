# SafetyDial

**Which Experience Makes LeWM Safer to Use?**

Sachin Mohan (2699183), BSc Honours CS, University of the Witwatersrand.
Supervised by Geraud Nangue Tasse.

## Current research

At the same interaction budget, which additional experience improves a pretrained
LeWM's predictions of unsafe outcomes, and does that improvement transfer to new
hazards, starting states and goals?

Start with the released Push-T checkpoint. Diagnose information available in actual
observations versus errors in imagined futures. If the predictor is the bottleneck,
freeze the visual encoder and physical readout, then compare ordinary, boundary-focused
and eventually learned error-risk acquisition using the same predictor-side adaptation.
The safety rule is supplied explicitly; this is not label-free safety discovery.

**Adopted 21 September 2026.** The new study has not run. The existing Push-T and Phase 0
experiments are preserved as evidence and reusable infrastructure.

- [Adopted research direction](docs/researchDirection.md): question, gates and full protocol.
- [First-pilot checklist](docs/research/pilot.md): next work and completion criteria.
- [Checkpoint reference](docs/research/checkpoints.md): available models and asset dependencies.
- [Related work](docs/research/relatedWork.md): closest overlaps and limits on novelty claims.
- [Experiment index](experiments/README.md): existing runnable work and planned additions.
- [Experimental notes](notes/README.md) and [animations](animations/README.md).
- [Agent and engineering guidance](AGENTS.md).

## Setup and asset recovery

```bash
git clone https://github.com/Sachin2911/Safety-Dial.git
cd Safety-Dial
bash scripts/setup.sh
```

Recover the earlier checkpoint, fitted normalisers and a small replay subset if available.
The local audit found the expected assets absent. Weights alone do not supply the fitted
normalisers used by the existing planner.

If downloads are needed, Push-T is now the default:

```bash
# Source and Push-T checkpoint only; still needs normalisers/data for the full pilot.
uv run python scripts/download_data.py weights_only=true

# Push-T weights and expert data.
uv run python scripts/download_data.py

# Optional other supported datasets, not required for the first pilot.
uv run python scripts/download_data.py --config-name cube
uv run python scripts/download_data.py --config-name all
```

Push-T expert data is about 13 GB compressed; Cube is about 46 GB compressed and is
optional. Check storage before downloading and decompressing. Preview configuration
without downloading with `uv run python scripts/download_data.py --cfg job`.

Run JupyterLab with `experiments/` as the working directory and use the
**Python (safetydial .venv)** kernel. New reusable code belongs in
`experiments/helpers/`, with small entry points in `experiments/scripts/`.

## Preserved work

- [Push-T penalty-CEM and Safe-CEM findings](notes/safeCEM.md),
  [report and recorded results](docs/safeDial/), and
  [notebooks](experiments/README.md).
- [Phase 0 recovery findings](notes/phase0Report.md),
  [report and data](docs/phase0/), and locomotion helpers/scripts.
- Submitted academic deliverables and reference papers remain under `docs/`.
  They record earlier work and do not define the current research requirements.
