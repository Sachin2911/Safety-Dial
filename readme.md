# SafetyDial

Small controllers around frozen learned models, evaluated on safe task completion.

Sachin Mohan, BSc Honours Computer Science, University of the Witwatersrand.
Supervised by Geraud Nangue Tasse.

## Start here

- [Principia](principia.pdf): the consolidated experiment record. Thirty-seven
  experiment accounts, one page each, plus an overview and linked index.
- [Research manuscript](docs/research/honours-gecco/honoursThesis.pdf): the current
  argument, methods, figures and limitations in ACM format.
- [Current direction and research controls](firstPrinciples.md).
- [Documentation](docs/README.md) and [evidence locations](progress/README.md).

Tuned fixed planning improved safe completion at unchanged margins in four tasks.
Feedback and evolution have not established an advantage over strong fixed settings
and simpler search. A ManiGuard Jar tip shield reduced observed violations with
unchanged safe completion. The original evolutionary completion goal remains open.

## Repository layout

- `docs/research/`: Principia source and the current manuscript.
- `progress/receipts/`: frozen machine-readable evidence, audits and source snapshots.
- `experiments/{helpers,scripts,tests}/`: reusable code, entry points and tests.
- `configs/`: task-specific configurations. `scripts/`: baseline and study tooling.
- [archive/](archive/README.md): historical studies and the compressed document bundle.
- `outputs/`, `data/`, `third_party/`: ignored local evidence, data and upstream clones.

Read [AGENTS.md](AGENTS.md) before research work. The root environment is legacy;
use the intended interpreter explicitly and do not sync it into ManiGuard's runtime.
Local tests use `.venv/bin/python`, without `uv run`. No new experiment is authorized
by a completed protocol or unused historical allocation.
