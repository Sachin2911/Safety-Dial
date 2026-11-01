# SafetyDial

Small controllers around frozen learned models, evaluated on safe task completion.

Sachin Mohan, BSc Honours Computer Science, University of the Witwatersrand.
Supervised by Geraud Nangue Tasse.

## Start here

- [Rule42](SafeLeWM/Rule42.pdf): the current two-page experiment outline,
  with [editable LaTeX](SafeLeWM/Rule42.tex).
- [First experiment](SafeLeWM/FirstExperiment.md): the detailed LeWM design,
  adapter alternatives, comparisons, ablations and safety dial.
- [SafeLeWM working area](SafeLeWM/README.md).
- [Principia](principia.pdf): the consolidated earlier experiment record,
  with one-page accounts, an overview and linked index.
- [Research manuscript](docs/research/honours-gecco/honoursThesis.pdf): the current
  argument, methods, figures and limitations in ACM format.
- [Current direction and research controls](firstPrinciples.md).
- [Documentation](docs/README.md) and [evidence locations](progress/README.md).

The current question is whether evolutionary search can learn compact adapters
around frozen world-model systems with less reliance on manually tuned behavioural
rules. LeWM across TwoRoom, PushT and OGBench Cube is the opening study; Block
Plucking is the intended main study. ManiGuard is deferred. The allocated topic is
[Safe AI via Evolutionary Algorithms](docs/allocatedTopic.md), targeting GECCO 2027.

Earlier studies found useful fixed planning configurations, but did not establish
a consistent evolutionary advantage. They remain exploratory context for the new
design. Protocols and budgets are not frozen, and provisioning a pod does not
authorise experiments.

## Repository layout

- `docs/research/`: Principia source and the current manuscript.
- `SafeLeWM/`: all documents, new code, configs, tests and local outputs for the first experiment.
- `progress/receipts/`: frozen machine-readable evidence, audits and source snapshots.
- `experiments/{helpers,scripts,tests}/`: reusable code, entry points and tests.
- `configs/`: task-specific configurations. `scripts/`: baseline and study tooling.
- [archive/](archive/README.md): historical studies and the compressed document bundle.
- `outputs/`, `data/`, `third_party/`: ignored local evidence, data and upstream clones.

Read [AGENTS.md](AGENTS.md) before research work. The root environment is legacy;
use the intended interpreter explicitly and do not sync it into ManiGuard's runtime.
Experiment computation and experiment tests run only on rented RunPod or Vast
hosts. Local work is limited to editing, orchestration, transfers and saved-evidence
inspection. No new experiment is authorized by a completed protocol or unused
historical allocation.
