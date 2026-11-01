# Documentation

- [Rule42](../SafeLeWM/Rule42.pdf): the current two-page LeWM experiment outline and
  shared research question, with [editable LaTeX](../SafeLeWM/Rule42.tex).
- [FirstExperiment.md](../SafeLeWM/FirstExperiment.md): the detailed three-environment
  study, adapter qualification, comparisons, ablations and safety dial.
- [SafeLeWM working area](../SafeLeWM/README.md).
- [Principia](../principia.pdf): one-page experiment accounts across the research history.
  Edit [principia.tex](research/principia.tex); source identities are in
  [principia-evidence.json](research/principia-evidence.json).
- [Current manuscript](research/honours-gecco/honoursThesis.pdf),
  [LaTeX source](research/honours-gecco/honoursThesis.tex),
  [build and review notes](research/honours-gecco/draft-notes.txt).
- [Current research direction](../firstPrinciples.md) and
  [evidence locations](../progress/README.md).
- [SafeLeWM](research/SafeLeWM.pdf): a three-page intuitive explanation of hazards,
  SafeCEM, planner tuning and a proposed safety adapter, with an
  [editable builder](research/build_SafeLeWM.py).
- [Implemented SafeCEM refinements](research/pusht-tuning.md#implemented-safecem-refinements-30-october):
  the three planner changes, GPU software checks and completed short control.
- [LeWM proposal-adapter integration](research/lewm-proposal-smoke.md): the 510-parameter
  behavioural proposal network, eleven passing GPU checks and one audited untrained
  engineering episode.
- [LeWM proposal-adapter training pilot](research/lewm-proposal-pilot.md): four real
  CMA-ES updates and 76 audited episodes. Native won selection; no completion gain
  was established. Trained weights, raw evidence and replays are preserved locally.
- [LeWM residual-mixture study](research/lewm-residual-study.md): proposal influence,
  exact zero/native equivalence, nineteen passing GPU checks and 233 audited
  common-panel episodes. CMA tied native at 2/4 selection safe completions;
  untrained and random search scored 3/4. The expansion gate stayed closed.
- [LeWM readiness and fixed-adapter screen](research/lewm-screen.md): 26 passing GPU
  checks, 54 scored episodes and four engineering probes. PushT hazard completion was
  native 3/18 versus untrained 1/18; Cube closed at 0/6 no-hazard completions.
- [PushT safety-model implementation](research/safety-model.md): the working learned
  motion checker, three learned-controlled demonstrations, evidence and limitations.
- [PushT automatic planner tuning](research/pusht-tuning.md): the frozen small search,
  remote execution record and completion, safety and interaction accounting.
- [ManiGuard evolution pipeline](research/maniguard-evolution.md): recurrent residual
  control, CMA-ES training, native comparisons, checkpoints and development evaluation.
- [Next ManiGuard experiment](research/maniguard-evolution-v3-experiment.md) and
  [future goal prompt](research/maniguard-evolution-v3-goal.md): the prepared conservative
  adapter protocol, measured controller selection and ten-hour rental constraints.

## References and academic material

- [ManiGuard base rates](research/research_notes/ManiGuard%20base%20rates.md) and
  [baseline replication](research/research_notes/Jar%20replication%20and%20noise%20floor.md).
- [ManiGuard paper](papers/Maniguard.pdf).
- [Allocated topic](allocatedTopic.md) and [GECCO notes](gecco2027.md).
- [Research concept note](researchConceptNote.pdf),
  [its source](researchConceptNote/README.md) and [proposal template](templates/research-proposal/).

## Rebuild Principia

From the repository root, using the installed TeX Live toolchain:

```bash
mkdir -p outputs/principia/build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=outputs/principia/build docs/research/principia.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=outputs/principia/build docs/research/principia.tex
cp outputs/principia/build/principia.pdf principia.pdf
```

The second pass resolves the experiment index. There should be 40 pages: an overview,
an index and 38 single-page accounts. Old task prose is in the
[recovery bundle](../archive/20261029/README.md); the existing frozen archives retain
their original source paths.
