# Hugging Face dataset presentation

The dataset was renamed on 5 November 2026 from `Sachioster/lewm-experiment1`
to `Sachioster/LeWMExp1_Baseline_Reproduction` at Sachin's request. The
[rename receipt](rename-20261105.json) records the preserved revision, file
inventory and updated navigation. Historical protocols and receipts retain the
repository name used at the time; their pinned revisions belong to this same
renamed dataset.

The [private dataset home](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction)
now contains the experiment index and eight inline video players, grouped into
TwoRoom, Reacher, PushT and OGBench Cube. Each environment page combines its success
and failure examples, trajectory images, planner settings and results.

Sachin requested the reorganization on 5 November 2026 and selected organizing the
existing clips by environment. No new videos, simulation or model queries were
needed. The original scientific results and all original run files remain intact.

`content/` mirrors the text and JSON files published to the dataset. Its root
`README.md` is the dataset card. Videos and images are kept in Hugging Face rather
than committed here. The published `media/reproduction-20261104/index.json` records
the original pinned source, exact case metadata, size and SHA-256 for each copy.

The [publication receipt](publication-20261105.json) records the old and new
revisions, every published file hash, fresh download verification and preservation
of the original evidence. The existing report README was updated; its previous
version remains at the receipt's previous revision. All 1,067 other pre-existing
files were checked for unchanged Git object IDs and sizes.

The shorter Odyssey overview and its LaTeX source remain under
[`theOdessy/`](../../theOdessy/reproduction.tex). The separate
[PDF publication receipt](pdf-publication-20261105.json) records their later
5 November refresh on Hugging Face, including fresh download verification.

The card metadata passed Hugging Face validation. All 122 internal link targets
were checked against the repository inventory, and a local Markdown/HTML preview
was visually inspected in isolated headless Firefox. Signed-in Hugging Face page
rendering was not available to this session; authenticated downloads verified all
published bytes, including the eight MP4s and eight trajectory PNGs.

Future experiment reports should add a row to `reports/README.md`, a separate
report folder and links to their final media. Keep exact source revisions and
scientific evidence separate from editable navigation. Do not move or overwrite
original run artifacts to improve presentation.
