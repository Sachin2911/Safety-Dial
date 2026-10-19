# ManiGuard shared GECCO-format report

Open [named.pdf](named.pdf). This shared paper uses the repository's GECCO
`acmart`/`sigconf` template and now covers the completed Part 1 and Part 2 pilots,
the completed Part 3 forecast audit, and Part 4's completed offline counterexample
premise diagnostic. Part 3 failed accuracy despite adequate event support, so its
64-episode physical optimizer comparison was scientifically gated. Part 4's frozen
primary warning predictor alarms on 2,045/2,229 negative development windows and has
zero confident misses among one first-event episode. Its physical scene engineering,
80-episode acquisition/audit design and matched repair remain unstarted. The archived
GECCO paper and its experiments remain unchanged.

This is a cumulative internal evidence report in the GECCO template, not an eight-page
submission prepared for review. Parts 1, 2 and 3 retain their original outcomes and costs.
The [current build receipt](current-build.json) records the reviewed eleven-page PDF,
source hashes and Part 4 checks; the older `provenance.json` retains the original Part 1
export record. The cumulative time plot covers the prior Parts 1 through 3 rental.
Part 4 has its own recorded provider start and complete allocation ledger.

Part 1's adverse result is retained: full strength has 4/8 safe completions and 2/8
violating episodes, versus identity's 5/8 and 0/8. Part 2 has 3/4 safe completions for
identity, fixed-eight and correction-triggered execution, versus 2/4 for fixed-four.
Each condition has one violation. D never triggers early; its sole threshold crossing
is at action eight, when a query is already due. These small pilots establish neither
superiority nor calibrated risk. Part 3's model was 45.1% worse than persistence at four actions; all calibration
corrections occupied one bin, making its two trust masks identical. No Part 3
evolutionary finalist or optimizer outcome is claimed.

Part 4 adds zero simulator attempts, controls, physics, VLA queries or encoder calls.
Its 24 historical donors are reused once, with the original costs retained. The
[Part 4 report](../ManiGuard%20Part4%20counterexample%20premise.md),
[frozen protocol and export](../../../../configs/maniguard/part4/README.md),
[public audit](../../research_notes/ManiGuard%20phase3%20Part4%20offline%20audit.json)
and [video gallery](part4-videos.md) preserve task/episode denominators, failed criteria,
all eight first-event camera strips, two annotated historical videos and private
revision-pinned storage. Neither the calibration low-risk miss nor the visual-only
reference's better ranking is used to bypass the failed development operating point.

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

The [cumulative attempt ledger](../../research_notes/ManiGuard%20phase3%20cumulative%20attempts.csv)
and [allocation/model accounting](../../research_notes/ManiGuard%20phase3%20cumulative%20accounting.json)
count Part 1 and reused data once: 80 physical attempts, 53,911 known controls and
6,582 known VLA queries. The [Part 3 audit](../../research_notes/ManiGuard%20phase3%20Part3%20forecast%20audit.json)
and [frozen artifact export](../../../../configs/maniguard/followup/part3-frozen-model-export.json)
preserve the failed checks and private, revision-pinned model. The final closeout
receipt distinguishes verified GPU-worker exit from the later provider rental state.
