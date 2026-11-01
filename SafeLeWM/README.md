# SafeLeWM

Working area for the LeWM opening study of **Safe AI via Evolutionary Algorithms**.

- [Rule42](Rule42.pdf): the simple two-page outline, with [LaTeX source](Rule42.tex).
- [Detailed experiment design](FirstExperiment.md).
- [Current direction and controls](../firstPrinciples.md).

The study covers TwoRoom, PushT and OGBench Cube before the intended main Block
Plucking study. As requested on 1 November, all first-experiment work belongs here.
This is part of the parent repository, not a separate Git repository.

## Layout

- `FirstExperiment.md` and `Rule42.{tex,pdf}`: the active design.
- `experiments/{helpers,scripts,tests}/`: new implementation and tests.
- `configs/`: prospective first-experiment configurations.
- `outputs/`, `data/`, `checkpoints/` and `runs/`: ignored local artifacts and evidence.

Create implementation directories when work begins. Existing studies and shared
code elsewhere in the parent repository remain at their original paths. Datasets,
checkpoints, credentials and generated runs stay outside Git. Provisioning records
are local under `outputs/runpod-setup-20261101/`; automatic stopping is disabled
at the user's request. No workload has been launched on the pod.

## Rebuild Rule42

From the parent repository root:

```bash
mkdir -p SafeLeWM/outputs/rule42-build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=SafeLeWM/outputs/rule42-build SafeLeWM/Rule42.tex
cp SafeLeWM/outputs/rule42-build/Rule42.pdf SafeLeWM/Rule42.pdf
```

The design is prospective. No experiment implementation or run is initiated by
creating this folder.
