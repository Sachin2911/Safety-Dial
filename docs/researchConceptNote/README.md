# Research Concept Note

An editable initial concept note for Sachin's honours research, adapted from
the [research proposal template](../templates/research-proposal/README.md).
It retains the reference's Charter type, A4 margins, Wits cover and author-year
citations, using sections suited to a short concept note. It is a discussion
draft, not an approved proposal or an official university template.

## Editing

- `main.tex`: document entry point and formatting.
- `metadata.tex`: title, author, supervisor and document date.
- `concept-note.tex`: the user's wording for the idea's origins, research question,
  problem and motivation, potential contribution and eight open questions.
  Preserve the wording unless explicitly asked to edit it.
- `references.bib`: retained reading-list entries, currently omitted from the PDF.
  Add citations with `\parencite{key}` or `\textcite{key}` and restore
  `\printbibliography` in `main.tex` if references are needed.

The draft records the earlier direction in the
[archived oracle](../../archive/20261022/docs/research/history/20261018/oracle.md). The current direction is
[first principles](../../firstPrinciples.md);
the concept note has not been revised to match it. The crest is
copied from the existing template, which extracted it from the supplied proposal.

## Building

Install TeX Live (or an equivalent distribution) with pdfLaTeX, latexmk, Biber
and the packages in `main.tex`. From the repository root:

```bash
bash docs/researchConceptNote/build.sh
```

The PDF is written to **`docs/researchConceptNote.pdf`**, outside this source
folder. Auxiliary files stay in the ignored `researchConceptNote/build/`
directory. The script also works when called using an absolute path from another
working directory.

Alternatively, from this source folder, run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The local `.latexmkrc` sets the output name and directories for either command.
Run latexmk from this folder so it loads that configuration. Compiling with
pdfLaTeX directly does not read `.latexmkrc` and will not use these output rules.

For Overleaf, upload this source folder (including `assets/`), select `main.tex`
and use pdfLaTeX. Download the resulting PDF to `docs/researchConceptNote.pdf`
when bringing it back into the repository; the local build script handles that
placement automatically.
