# Synthetic ManiGuard demonstration report

This folder is a self-contained demo report bundle, styled as a two-column supervisor
paper. **All favourable outcome counts are invented. They are not experimental evidence.**
The PDF identifies its synthetic status on every page, in its chart and table, and in
its metadata. It also states the actual adverse Part 1 results.

- `synthetic-supervisor-demo.pdf`: presentation copy.
- `synthetic-supervisor-demo.tex`: editable report source.
- `demo-data.json`: hand-authored synthetic counts and explicit provenance.
- `build.py`: validates category totals, generates the table/chart inputs, and compiles.
- `synthetic-table.tex` and `synthetic-chart.tex`: generated solely from the demo JSON.

From the repository root, rebuild with:

```bash
uv run --no-project python output/pdf/maniguard-synthetic-demo/build.py
```

The builder uses only the Python standard library plus system `pdflatex`. TeX requires
the packages listed in the source, including PGFPlots and draftwatermark. It makes no
network request, simulator call, policy query or training update.

The original supervisor report remains at
`docs/research/reports/maniguard-phase3-part1-gecco/draft.pdf`.
This demo does not overwrite it or supply data to experimental result readers.
Preserve the synthetic labels when reusing any part of the example.
