# Documentation

- [Principia](../principia.pdf): one-page experiment accounts across the research history.
  Edit [principia.tex](research/principia.tex); source identities are in
  [principia-evidence.json](research/principia-evidence.json).
- [Current manuscript](research/honours-gecco/honoursThesis.pdf),
  [LaTeX source](research/honours-gecco/honoursThesis.tex),
  [build and review notes](research/honours-gecco/draft-notes.txt).
- [Current research direction](../firstPrinciples.md) and
  [evidence locations](../progress/README.md).

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

The second pass resolves the experiment index. There should be 39 pages: an overview,
an index and 37 single-page accounts. Old task prose is in the
[recovery bundle](../archive/20261029/README.md); the existing frozen archives retain
their original source paths.
