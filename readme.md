# SafetyDial

**Evolutionary adaptation for task continuation under learned safety filtering.**

Sachin Mohan, BSc Honours Computer Science, University of the Witwatersrand.
Supervised by Geraud Nangue Tasse.

Start with [oracle.md](oracle.md), the central reference for the current research direction:
can evolutionary adaptation help a pretrained manipulation policy work more effectively
with its safety filter, improving safe task completion? The choice remains conditional on
finding meaningful, avoidable stalls. The method and experiment plan are still open.

## Current workspace

- [Research and documentation](docs/README.md): the manipulation literature, UNISafe asset
  and compute audits, proposal template and academic context.
- [Research oracle](oracle.md): decisions, scope and unresolved questions.
- [Experiment workspace](experiments/README.md): where new implementation will go.
- [Agent and engineering guidance](AGENTS.md).

## Preserved research

The [archive](archive/README.md) keeps earlier work together with its code, configurations,
tests, reports, notebook outputs and videos. Nothing was selected for deletion.

- **Push-T Safe CEM and safety dial:** [findings](archive/legacy/notes/safeCEM.md),
  [report](archive/legacy/docs/safeDial/SafeDialReport.pdf) and
  [demonstrations](archive/legacy/animations/README.md).
- **Hopper and Walker locomotion:** [Phase 0 findings](archive/legacy/notes/phase0Report.md),
  [report](archive/legacy/docs/phase0/Phase0Report.pdf) and
  [videos](archive/README.md#hopper-and-walker-locomotion).
- **LeWM experience study:** [through-S4 report](archive/legacy/docs/mainPlan/reports/through-s4-20260927/README.md).
- **Walker evolutionary studies:** [recorded results](archive/legacy/docs/evoPlan/results/README.md)
  and [paper draft](archive/legacy/docs/geccoPaper/main.pdf).

The archive index explains the layout, provenance and reproduction entry points. Earlier
plans and scheduled-job configurations describe their original studies.

## Setup

The current [Vast template](configs/vast/README.md) is a lightweight ManiGuard preparation
base with Python 3.10 and CUDA 12.4.1. Clone the repository after connecting:

```bash
git clone https://github.com/Sachin2911/Safety-Dial.git
cd Safety-Dial
```

ManiGuard simulation, assets and model-serving environments are separate manual preparation
steps described in the template notes. The template does not install or validate them.
For the completed UNISafe study, use the preserved
[Python 3.12 / CUDA 13 platform](configs/vast/UNISAFE.md) and its installer instead.

## Pretrained UNISafe checkpoints

The original Google Drive checkpoints are mirrored in the private
[Sachioster/unisafe-pretrained](https://huggingface.co/Sachioster/unisafe-pretrained)
model repository. With `HF_TOKEN` available, install the additive dependencies and
download the pinned revision with all original file hashes verified:

```bash
bash scripts/setup_unisafe.sh
```

The source revision and manifest hash are recorded in
[configs/download/unisafe.json](configs/download/unisafe.json). Files are saved under
ignored `data/unisafe/pretrained/`: `latest.pt` is the Dreamer checkpoint and
`filter/model/` contains actor/critic snapshots from steps 10000 through 200000.
The mirror includes original configuration, provenance and tensor inspection metadata.
Frozen inference and the migrated environment have executable checks in
[experiments/README.md](experiments/README.md#unisafe-engineering-checks). Run both with
`bash scripts/smoke_unisafe.sh` after `bash scripts/setup_unisafe.sh`. Closed-loop
policy performance remains unvalidated.
