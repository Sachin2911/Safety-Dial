# Research proposal LaTeX template

A reusable honours proposal skeleton based on the supplied
`Tristan Bester 2090454 Research Proposal (1)-1.pdf`.
It follows the reference's A4 layout, Charter type, centred Wits title page,
Roman-numbered front matter, numbered chapters, blue internal/citation links
and author-year references. It is a visual adaptation, not the original LaTeX
source or a verified official university template.

The six main chapters retain the reference's structure:

1. Introduction
2. Background and Related Work
3. Research Objectives
4. Research Methodology
5. Research Plan
6. Conclusion

The abstract, declaration, contents and bibliography are included. Topic-specific
subsections use editable prompts for Sachin's current research interests. No
environment, algorithm, hypothesis, result or date range has been selected by
this template. `oracle.md` remains the authority for research decisions.

## Editing

- Open `main.tex` as the entry point.
- Edit `metadata.tex` for the title, author, supervisor, degree and date. Sachin's
  details are prefilled; the degree wording should be checked before submission.
- Write each chapter in its corresponding file under `chapters/`.
- Replace grey bracketed prompts with your prose. The declaration and signature
  are placeholders, with no signature or assertion copied from the reference.
- Add verified sources to `references.bib`. Use `\parencite{key}` for square-bracket
  author-year citations and `\textcite{key}` for citations within a sentence.
- Remove `\nocite{*}` in `main.tex` when drafting so only cited sources appear.
  The preview intentionally lists two real sample sources to show the format.
- The planning table contains editable milestones with no assigned dates.

The crest in `assets/wits-crest.jpg` was extracted unchanged from the title page
of the supplied reference. The reference's research prose, signature and figures
have not been copied into the template. Keep the crest's original proportions.

## Building

Install a LaTeX distribution containing pdfLaTeX, latexmk, Biber and the packages
listed in `main.tex` (a typical full TeX Live installation or Overleaf works).
From the repository root:

```bash
cd docs/templates/research-proposal
latexmk -pdf -outdir=build main.tex
```

The generated document is `build/main.pdf`. Build artifacts are ignored.
`preview.pdf` is the compiled template preview and will not update automatically
when you edit the source. To refresh it after a successful build:

```bash
cp build/main.pdf preview.pdf
```

For Overleaf, upload the contents of this folder, set `main.tex` as the main
document and use pdfLaTeX. Preserve the `chapters/` and `assets/` directories.
