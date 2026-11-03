# Current paper draft

[world-model-adapters.pdf](world-model-adapters.pdf), with editable source
[world-model-adapters.tex](world-model-adapters.tex), is the 2 November working
consolidation for the current LeWM opening study and Block Plucking main study.
It distinguishes completed evidence from active experiments. It is not submission-ready.

The manuscript uses Rule43 accounts 9 and 12-17 and the completed Block Plucking
pilot. PushT authority failed its frozen gate, so no optimizer training followed.
The completed dial test found missing settings and empirical risk-limit failures.
The first completed Block Plucking Search 2 repeat (R1) passed its saved-record audit
and received the frozen mixed verdict. Two repeats and the L4 population-size study
remain pending. Their results must be integrated after their respective audits. It does not
claim shared actor weights, calibrated risk, reduced tuning or established
optimizer superiority. The fourteen core references have been checked against primary publication pages or original papers, including feasibility rules, stochastic ranking, direct evolutionary safe-RL methods, multi-objective learning and probabilistic safety. The working draft has four pages including references.
Broader literature coverage remains.

The [R1 summary](../../../SafeLeWM/results/block-search2-r1-paper-summary.json),
[audit](../../../SafeLeWM/results/block-search2-r1-audit.json) and
[preservation receipt](../../../SafeLeWM/results/block-search2-r1-preservation.json)
bind the new Block paragraph to 1,156 episode records. All 4,836 files in its
paper-evidence bundle passed independent hash verification. This is a record backup,
not a full camera/model backup. Separate audits retain the discarded five-lane
attempts and count the three original setup episodes once despite duplicate local
copies. The [setup reconciliation](../../../SafeLeWM/results/block-search2-setup-reconciliation.json)
records the remaining gap: three withdrawn-repeat smoke episodes are documented,
but their raw resource records have not been located. Incomplete attempts give
recorded lower bounds; complete study-wide resource use is not verified.

`honoursThesis.tex` and its PDF retain the older October evidence and framing.
They have not been overwritten or relabelled as the new studies. `build.sh`
continues to build that historical document.

The October figure generators retain their dependencies in `figure_tools/` and
read retained receipts from `docs/records/progress/`. Some also require historical
`outputs/` files deleted in the earlier cleanup; recover those inputs before
regenerating figures. Existing manuscript PDFs and figures remain unchanged.

Build the current draft from the repository root:

```sh
mkdir -p /tmp/safelewm-current-paper
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/safelewm-current-paper docs/research/honours-gecco/world-model-adapters.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/safelewm-current-paper docs/research/honours-gecco/world-model-adapters.tex
cp /tmp/safelewm-current-paper/world-model-adapters.pdf docs/research/honours-gecco/world-model-adapters.pdf
```
