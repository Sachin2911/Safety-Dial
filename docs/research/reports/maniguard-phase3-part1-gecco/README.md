# ManiGuard shared GECCO-format report

Open **[draft.pdf](draft.pdf)** for the current eight-page native-noise controller paper,
updated on 20 October 2026. Its article source is
[native-noise-paper.tex](native-noise-paper.tex), included by `draft.tex`.
It describes the 176-parameter observation/instruction-conditioned adapter,
native-sampling identity, the completed 320-call decoder audit and four full
development episodes, and the frozen native/fixed/EA/random-search Jar comparison.
All eight shared initialization episodes are now included: zero requested
completions, with the nonviolating negative constant never grasping. Independent
rates and finalist results remain pending. The original cumulative
report, its historical figures and numerical results remain in `main.tex`/`named.pdf`.

Section 7, sourced from [model-followup.tex](model-followup.tex), adds the separate
world-model, action-changing filter and ROSARL preparation. It reports the
completed CPU audit, including its failed prediction/calibration gate, and the
proposed second physical experiment. That experiment starts with four reference
episodes and expands only after a predefined progress criterion, up to a proposed
12 valid episodes plus two infrastructure replacements. It has not been physically
launched. This smaller staged proposal replaces no part of the running comparison
and does not add an allowance to the earlier deferred 16-plus-two proposal.

The physical numerical evidence in Sections 5 and 6 retains the earlier
07:11:14 UTC diagnostic cutoff, including its statement that offspring were still
pending then. The new Section 7 uses the separately preserved CPU audit and later
CPU source checks. Generation one has since completed; its dated summary is in
the current study records, pending the execution session's next physical-results
paper update. Independent checking is still pending. The CPU evidence is not
reported as a physical result or an evolutionary advantage.

The new [protocol](../../../../configs/maniguard/latent-evolution/capacity-20261020-v2.json),
[development summary](../../../../configs/maniguard/latent-evolution/development-summary-20261019.json),
[study report](../ManiGuard%20latent%20evolution%20pilot.md) and
[study README](../../../../configs/maniguard/latent-evolution/README.md) supply exact
task/seed lists, costs, source pins and restoration instructions. The current build
receipt is `latent-build-20261020-model-followup.json`;
`latent-build-20261020-diagnostic.json` retains the earlier six-page diagnostic
build, and `latent-build-20261020-initial.json`
retains the earlier build after initialization, and `latent-build-20261020.json` retains
the earlier five-page build at commit `82c885a`. The paper distinguishes implemented paths,
decoder responses, physical development evidence and pending comparisons. Ordinary
learning, optimizer replication and cross-family adapter transfer remain untested.

Build the current paper with `make paper TECTONIC=/path/to/tectonic` (`make` also
selects it). Tectonic 0.17.0 is pinned by the build receipt. The added figure uses
only the committed numerical development summary, with zero new physics/model calls:

```bash
uv run --isolated --no-project --python /usr/bin/python3 \
  --with matplotlib==3.10.7 --with pymupdf==1.26.7 \
  python experiments/scripts/maniguard_latent_paper_assets.py
```

Run the figure command at the repository root. The plotting dependencies are isolated
from the simulation/policy environments. Its new figure and source/hash receipt live
under `docs/research/figures/ManiGuard/latent-evolution/`. Closest primary sources are
recorded separately in `latent-references.bib`; the historical bibliography is retained.

The [complete initial summary](../../../../configs/maniguard/latent-evolution/shared-initial-summary-20261020.json)
includes all eight outcomes, contact/grasp counts, robot/camera reset comparisons
and 1,500 stored native-noise/RNG comparisons. It reuses actual records, with no
new inference. After refreshing the study's independently verified analysis,
`maniguard_latent_initial_summary.py --run RUN --output NEW_JSON` reproduces this
summary. The command refuses to overwrite a recorded result. This is shared training
evidence on two cases, with identical native streams but different initial camera
pixels; it is not the independent four-block comparison.

The current draft additionally reports the four incumbent rechecks and the
[48-call gripper diagnostic](../../../../configs/maniguard/latent-evolution/gripper-authority-summary-20261020.json).
Its eight middle/last development frames show stronger-offset gripper influence
with accompanying arm changes. No physical method change or benefit is inferred.
The diagnostic archives and all 48 chunks have passed a remote round trip, including
their input hashes checked against the earlier restored development records.
An [independent offline review](../../../../configs/maniguard/latent-evolution/review-sensitivity-summary-20261020.json)
finds active, mostly unsaturated EA mutations on factual feature pools. The original
comparison resumed after the user resolved the hold; descendant and final checking
results remain pending at this paper's diagnostic evidence cutoff.

The following notes describe the **19 October supervisor check-in**, including its
[paper PDF at the preserved main revision](https://github.com/Sachin2911/Safety-Dial/blob/4783369c5b599580e26f9c3a2723ee55518c46a3/docs/research/reports/maniguard-phase3-part1-gecco/draft.pdf)
and [one-page meeting brief](meeting-brief.pdf). That paper draft uses the same ACM
format with a focused research narrative, formal controller equations, three
algorithms, an information-flow diagram and the completed saved-trace recovery
diagnostic. The original cumulative report below remains unchanged.

The paper was renamed from `supervisor-draft` to `draft`. The revision adds parameter
descriptions and layout support; the reported empirical outcomes are unchanged.
Historical build receipts retain the filenames used when those builds were recorded.

The new diagnostic reuses 18,657 commands from 24 historical episodes. At a uniform
.04 rad cap, componentwise clipping distorts joint-error direction by more than five
degrees on 9/55 modified commands; vector scaling has 0/55, with a maximum of
.00019 degrees. The median threshold gives three early-query opportunities in
482 factual prefixes, all on one diagnostic task, from only five positive eligible
calibration values. This is a geometry and activation diagnostic. It adds no new
physics, VLA queries, encoder calls, model fits or physical safety result.
Its [protocol](../../../../configs/maniguard/recovery/20261019-checkin/protocol.json),
[public result](../../research_notes/ManiGuard%20recovery%20checkin%20diagnostic.json)
and [figure provenance](../../figures/ManiGuard/phase3-recovery/provenance.json)
retain the scope, task roles, thresholds, counts and source hashes.

The three presentation videos are saved locally under
`runs/maniguard/recovery/20261019-checkin/showcase/` and backed up to the explicitly
approved private Hugging Face destination, pinned at revision
`c20b8b5f53bbd579935e93b90fb06bebbc107769`:

- [Selected full-dial success](https://huggingface.co/datasets/Sachioster/safety-dial-phase3-part1-20261018/blob/c20b8b5f53bbd579935e93b90fb06bebbc107769/recovery-checkin/20261019/videos/01-selected-success.mp4).
- [All settings complete safely](https://huggingface.co/datasets/Sachioster/safety-dial-phase3-part1-20261018/blob/c20b8b5f53bbd579935e93b90fb06bebbc107769/recovery-checkin/20261019/videos/02-all-settings-safe.mp4).
- [Adverse comparison for context](https://huggingface.co/datasets/Sachioster/safety-dial-phase3-part1-20261018/blob/c20b8b5f53bbd579935e93b90fb06bebbc107769/recovery-checkin/20261019/videos/03-adverse-context.mp4).

Access requires the owner's Hugging Face account. The six source clips and local
hash/timing receipts remain saved. These are historical, post hoc illustrative
cases, including the adverse case. Both native cameras, original frame count/rate
and ended panels are retained. Generated videos remain outside public Git; the
complete historical gallery remains below. The presentation does not turn the
native monitor into an action filter. The [private preservation receipt](../../research_notes/ManiGuard%20recovery%20checkin%20videos.json)
verifies all five uploaded files by download and SHA-256.

The meeting brief remains the historical 19 October document. Its paper source
is recoverable at the preserved revision above; `make paper` builds the current
native-noise article. Build the historical brief with
`make meeting-brief.pdf TECTONIC=/path/to/tectonic`.
Reproduce the diagnostic in a dedicated uv-managed CPU environment, without syncing
the root project into ManiGuard's simulation/policy environments:

```bash
uv venv --python 3.12 data/maniguard/envs/recovery-cpu
uv pip install --python data/maniguard/envs/recovery-cpu/bin/python \
  -r configs/maniguard/recovery/20261019-checkin/cpu-requirements.txt
uv run --no-project --python data/maniguard/envs/recovery-cpu/bin/python python \
  scripts/maniguard/part4_restore.py --run runs/maniguard/recovery/fresh-checkin
uv run --no-project --python data/maniguard/envs/recovery-cpu/bin/python python \
  experiments/scripts/maniguard_recovery_trace_screen.py \
  --run runs/maniguard/recovery/fresh-checkin \
  --protocol configs/maniguard/recovery/20261019-checkin/protocol.json \
  --output runs/maniguard/recovery/fresh-checkin/diagnostic
```

Restoration requires the owner's private Hugging Face access. A new output directory
is required; the runner will not overwrite scientific records. The recorded v1/v2
analyses have identical geometry/timing tables. The second adds automatic source
snapshot preservation and lint cleanup, with the first source/results retained.
The public result identifies its exact scored source hashes. Twenty-one targeted
CPU tests cover controller semantics and replay eligibility; Ruff passes.

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
