# Documentation

- [Rule42](../SafeLeWM/Rule42.pdf): the current two-page LeWM experiment outline and
  shared research question, with [editable LaTeX](../SafeLeWM/Rule42.tex).
- [Rule43](../SafeLeWM/Rule43.pdf): seventeen completed SafeLeWM experiment accounts,
  with [editable LaTeX](../SafeLeWM/Rule43.tex), [results and graphs](../SafeLeWM/results/README.md)
  and [verified evidence archives](../SafeLeWM/archive/README.md).
- [FirstExperiment.md](../SafeLeWM/FirstExperiment.md): the detailed three-environment
  study, adapter qualification, comparisons, ablations and safety dial.
- [SafeLeWM working area](../SafeLeWM/README.md).
- [Principia](research/principia.pdf): one-page experiment accounts across the research history.
  Edit [principia.tex](research/principia.tex); source identities are in
  [principia-evidence.json](research/principia-evidence.json).
- [Current working manuscript](research/honours-gecco/world-model-adapters.pdf),
  [LaTeX source](research/honours-gecco/world-model-adapters.tex), and
  [scope and build notes](research/honours-gecco/README.md).
  The [October thesis draft](research/honours-gecco/honoursThesis.pdf) retains the
  earlier evidence and framing.
- [Current research direction](../firstPrinciples.md) and
  [retained records and path map](records/README.md).
- [SafeLeWM](research/SafeLeWM.pdf): a three-page intuitive explanation of hazards,
  SafeCEM, planner tuning and a proposed safety adapter, with an
  [editable builder](research/build_SafeLeWM.py).
- [Block Plucking](../SafeBlock/README.md): main-study design and results.
- [Oracle manuscript](research/Oracle/paper.pdf), [source](research/Oracle/paper.tex), and
  [Overleaf bundle](research/Oracle/evolve-dont-retrain-overleaf.zip).
- [Local archive and output deletion](records/progress/receipts/local-deletion-20261103.md):
  records the removal of the older document bundles and local output trees.

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
mkdir -p docs/research/build/principia
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=docs/research/build/principia docs/research/principia.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=docs/research/build/principia docs/research/principia.tex
cp docs/research/build/principia/principia.pdf docs/research/principia.pdf
```

The second pass resolves the experiment index. There should be 40 pages: an overview,
an index and 38 single-page accounts. The older root document-recovery bundle was
deleted at Sachin's request on 3 November; see the deletion record above.
