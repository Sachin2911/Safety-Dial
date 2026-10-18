# ManiGuard Part 1, GECCO-format report

Open [named.pdf](named.pdf). This four-page named paper uses the repository's archived
GECCO `acmart`/`sigconf` template. The old paper and its experiment remain unchanged.
The report leads with the adverse fresh result: full strength has 4/8 safe completions
and 2/8 violating episodes, compared with identity's 5/8 and 0/8.

[Videos and pictures](videos.md) covers every comparison block. All 16 matched videos
are individually saved in the approved private dataset. Every block has overview and
wrist video, with dial 0, 0.5 and 1 from left to right. Sixteen PNG strips show the same
18-second timestamp across every camera and block. The paper's three illustrative
strips were selected after outcomes and are explicitly labelled as illustrations.
The public gallery and numerical comparison retain the entire frozen sample.

Build from this directory with `make`, using Tectonic 0.17.0, or
`tectonic -k --keep-logs named.tex`. The compiler is available from its
[official releases](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0).
A normal TeX installation with `acmart` can instead run `latexmk -pdf named.tex`.
The source supports the template's anonymous review toggle through `main.tex`, while
`named.tex` is the named sharing build. Rights metadata identifies an internal pilot.

The paper builds from committed figures without private run data. Rebuilding the
vector graphs from the saved audit uses the explicit CPU interpreter:

```bash
uv run --no-project --python data/maniguard/envs/cpu/bin/python \
  scripts/maniguard/part1_paper_assets.py RUN/audit-final \
  docs/research/figures/ManiGuard/phase3-part1
```

A recorded picture can be reproduced directly from its matched video:

```bash
ffmpeg -nostdin -v error -ss 18 -i MATCHED_VIDEO.mp4 -frames:v 1 FRAME.png
```

No simulator state is restored or advanced, and no policy query is made for reporting.
Figure/frame provenance records source hashes. The companion Markdown report and public
52-row ledger carry detailed accounting, including unknown pre-subscription physics.
The paper includes an AI-assistance disclosure. Build validation and the final private
backup receipts are recorded with the public audit.
