# SafeLeWM

The LeWM opening study of compact learned adapters around frozen world models.
TwoRoom, PushT and OGBench Cube provide the development environments; the main
Block Plucking study lives in [SafeBlock](../SafeBlock/README.md).

## Start here

- [Rule42](Rule42.pdf) and [FirstExperiment.md](FirstExperiment.md): research design.
- [Rule43](Rule43.pdf): seventeen completed experiment accounts.
- [Results index](results/README.md): outcomes, limitations, graphs and evidence.
- [Current direction and authority](../firstPrinciples.md): active work and controls.
- [Working manuscript](../docs/research/honours-gecco/world-model-adapters.pdf).
- [Evidence archive](archive/README.md): manifests, verification and restoration.

The recorded studies establish useful fixed planning and some adapter gains on
specific panels. They do not establish a general evolutionary advantage or
calibrated safety-dial risk. Completed studies and failed gates authorize no new
experiments. Rental shutdown requires Sachin's explicit instruction.

## Protocols and implementation

| Location | Purpose |
| --- | --- |
| [EvolutionBudgetStudy.md](EvolutionBudgetStudy.md) | L4 population-size study; consult the current authority record for status |
| [HazardScreen.md](HazardScreen.md) | Completed TwoRoom hazard qualification |
| [AdapterPilot.md](AdapterPilot.md), [correction](AdapterPilotCorrection.md) | Original and corrected recurrent-adapter pilot |
| [AdapterReplication.md](AdapterReplication.md) | Three-repeat matched-budget replication |
| [ManipulationScreen.md](ManipulationScreen.md) | PushT and Cube hazard screens |
| [AdapterMechanism.md](AdapterMechanism.md) | Representation selection and mechanism study |
| [MatchedPPO.md](MatchedPPO.md) | Matched CMA, random-search and PPO comparison |
| [PushTAuthority.md](PushTAuthority.md), [correction](PushTAuthorityCorrection.md) | Distinct-scene qualification and replay correction |
| [PushTAPReproduction.md](PushTAPReproduction.md) | Matched authors' baseline reproduction |
| [PushTValidatedQualification.md](PushTValidatedQualification.md) | Validated baseline and adapter-authority screen |
| [SafetyDialDevelopment.md](SafetyDialDevelopment.md), [SafetyDialLibrary.md](SafetyDialLibrary.md) | Saved-evidence diagnostic and library evaluation |
| `experiments/{helpers,scripts,tests}/` | Implementation and experiment tests |
| `configs/` | Frozen configurations and panels |
| `results/` | Completed accounts, structured records and graphs |
| `archive/protocols/` | Earlier completed protocols |
| `archive/evidence/` | Ignored local cache of the [pinned private cloud evidence](archive/README.md#cloud-migration-3-november-2026) |
| `outputs/` | Local output path; its existing contents were deleted on 3 November |

Protocols, code and result records retain their original paths. The
[3 November deletion record](../docs/records/progress/receipts/local-deletion-20261103.md)
documents the subsequent removal of all local output trees and the older root
archive at Sachin's request. The sixteen verified cloud archives remain recoverable.
Recover required inputs before reusing frozen packaging or source-freeze tools.
The cloud archive notes include commands to fetch a single archive or a whole study
back to its original path, with hash verification before use.

## Rebuild the documents

From the repository root:

```bash
mkdir -p /tmp/safelewm-docs
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/safelewm-docs SafeLeWM/Rule42.tex
cp /tmp/safelewm-docs/Rule42.pdf SafeLeWM/Rule42.pdf
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/safelewm-docs SafeLeWM/Rule43.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/safelewm-docs SafeLeWM/Rule43.tex
cp /tmp/safelewm-docs/Rule43.pdf SafeLeWM/Rule43.pdf
```

Rule42 has two pages. Rule43 has seventeen experiment accounts. These builds use
saved results and do not run experiments.
