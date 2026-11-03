# Safety Dial

Compact controllers around frozen world models, evaluated on safe task completion.
Sachin Mohan, BSc Honours CS, University of the Witwatersrand.
Supervisor: Geraud Nangue Tasse.

## Everyday work

| Work on | Start here |
| --- | --- |
| Direction and authorized next steps | [First principles](firstPrinciples.md) |
| LeWM opening study | [SafeLeWM](SafeLeWM/README.md) |
| Block Plucking main study | [SafeBlock](SafeBlock/README.md) |
| Current research paper | [PDF](docs/research/honours-gecco/world-model-adapters.pdf), [source and build notes](docs/research/honours-gecco/README.md) |
| Oracle manuscript | [PDF](docs/research/Oracle/paper.pdf), [source](docs/research/Oracle/paper.tex), [Overleaf bundle](docs/research/Oracle/evolve-dont-retrain-overleaf.zip) |

For a quieter file tree, open [Safety-Dial.code-workspace](Safety-Dial.code-workspace)
in VS Code or a compatible editor using **File > Open Workspace from File**.
It hides local assets, historical records and tool metadata from the Explorer.
Open this folder normally to see all remaining files,
or edit `files.exclude` in the workspace to reveal individual folders.

## Look things up

- LeWM design: [Rule42](SafeLeWM/Rule42.pdf) and [FirstExperiment](SafeLeWM/FirstExperiment.md).
- Completed LeWM experiments: [Rule43](SafeLeWM/Rule43.pdf) and [results index](SafeLeWM/results/README.md).
- Earlier experiments: [Principia](docs/research/principia.pdf) and [retained records](docs/records/README.md).
- References, academic material and document builds: [documentation](docs/README.md).
- Local archive and output deletion: [3 November record](docs/records/progress/receipts/local-deletion-20261103.md).
- Folder consolidation: [path map and verification](docs/records/README.md).
- Cloud evidence and download instructions: [SafeLeWM archive](SafeLeWM/archive/README.md#cloud-migration-3-november-2026).

## Working conventions

Read [AGENTS.md](AGENTS.md) before research work. Put new LeWM work in `SafeLeWM/`
and Block Plucking work in `SafeBlock/`; keep paper sources under their existing paths.
SafeBlock owns its required UNISafe helpers, configuration and ignored local assets.
Manuscripts and historical receipts live under `docs/`. The unused root campaign
code, launch scripts, upstream checkouts and asset caches were removed after
relocating the dependencies needed by the retained work.

The old root archive and three local output trees were deleted at Sachin's request
on 3 November; see the deletion record above. The verified SafeLeWM cloud evidence
retains its separate recovery instructions. Future local outputs and study data may
contain research records that Git does not back up. The root Python
environment is legacy. Experiment computation and tests run only on rented hosts;
completed protocols do not authorize new experiments. Rental shutdown stays manual.
