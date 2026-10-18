# ManiGuard shared GECCO-format report

Open [named.pdf](named.pdf). This shared paper uses the repository's GECCO
`acmart`/`sigconf` template and now covers the completed Part 1 and Part 2 pilots,
with the frozen Part 3 methods. Part 3's recording/forecast-audit stage is in progress
in this version. The archived GECCO paper and its experiments remain unchanged.

Part 1's adverse result is retained: full strength has 4/8 safe completions and 2/8
violating episodes, versus identity's 5/8 and 0/8. Part 2 has 3/4 safe completions for
identity, fixed-eight and correction-triggered execution, versus 2/4 for fixed-four.
Each condition has one violation. D never triggers early; its sole threshold crossing
is at action eight, when a query is already due. These small pilots establish neither
superiority nor calibrated risk. Part 3's physical search depends on its frozen
forecast audit, not on a requirement to produce a positive evolutionary result.

[Part 1 videos and pictures](videos.md) covers every three-dial comparison block.
[Parts 2 and 3 videos and pictures](followup-videos.md) links all available matched
videos and complete event/episode strips. Videos are in the approved private dataset,
revision-pinned and download/hash verified; access requires the owner's account.
Ended panels are black and labelled ENDED. Reporting never replays simulator physics.

The companion reports are [Part 1](../ManiGuard%20evolutionary%20safety%20dial%20pilot.md),
[Part 2](../ManiGuard%20Part2%20correction%20triggered%20feedback.md) and
[Part 3](../ManiGuard%20Part3%20model%20assisted%20evolution.md).
Part 2's [public audit](../../research_notes/ManiGuard%20phase3%20Part2%20audit.json),
[mechanism diagnostics](../../research_notes/ManiGuard%20phase3%20Part2%20diagnostics.json),
[controller export](../../../../configs/maniguard/followup/part2-frozen-controller-export.json)
and [figure provenance](../../figures/ManiGuard/phase3-followup/figure-provenance.json)
retain counts, denominators and hashes. Generated run directories, raw datasets,
licensed assets, credentials and model weights are excluded from Git.

Build from this directory with `make`, using Tectonic 0.17.0, or
`tectonic -k --keep-logs named.tex`. The compiler is available from its
[official releases](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0).
A TeX installation with `acmart` can run `latexmk -pdf named.tex`. `main.tex` retains
an anonymous review toggle; `named.tex` is the named internal sharing build.

The paper builds from committed derived figures without private run data. Rebuild
followup charts, matched videos, pictures and public gallery from the saved evidence:

```bash
uv run --no-project --python data/maniguard/envs/cpu/bin/python python \
  scripts/maniguard/followup_diagnostics.py
uv run --no-project --python data/maniguard/envs/cpu/bin/python python \
  scripts/maniguard/followup_evidence.py all
uv run --no-project --python data/maniguard/envs/cpu/bin/python python \
  scripts/maniguard/followup_report_assets.py
```

Part 1's figures retain their original provenance and can be rebuilt independently
with `scripts/maniguard/part1_paper_assets.py`. Its sixteen fixed-time video strips
remain unchanged. New pictures use losslessly saved native RGB, with raw source hashes;
violations have before/at/after strips, and nonviolations start/middle/final strips.
No policy query or physics step is made for reporting. The shared paper retains the
AI-assistance disclosure and distinguishes unknown initialization physics from zero.
