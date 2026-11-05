# Hugging Face dataset presentation

The completed [AP reproduction](../ap_reproduction/README.md) has its own
[private dataset](https://huggingface.co/datasets/Sachioster/LeWMExp1_AP_Reproduction).
AP-CEM achieved **306/512 successes (59.8%)**, versus **47/512 (9.2%)** for matched
final-goal CEM. With 128 cases per arm per environment, AP/control counts were
TwoRoom **58/2**, PushT **88/4**, Reacher **100/32** and Cube **60/9**.

The [AP report and environment galleries](https://huggingface.co/datasets/Sachioster/LeWMExp1_AP_Reproduction/blob/main/reports/ap-reproduction-20261105/README.md)
bring together original clips, trajectories, settings and eight derived ghost
comparisons. Each environment shows the first AP success and failure by original
query order, paired with the control from the same start, goal and optimizer seed.
The revised comparisons use a larger widescreen layout at 30 fps, without action
bars. AP ghosts are teal and CEM ghosts are orange, matching the paths. Soft
pixel-color masks tint the main view while individual previews keep original colors.
Each original clip is motion-smoothed before the 50/50 ghost blend. These
in-between images are visual estimates, not recorded observations. Both planners
share the same action clock, and the plots connect saved physical states. Held
images are labeled and can precede the final scored state. Original clips remain
linked. Creating this media used no new simulator or model work.

The [scientific audit](../ap_reproduction/evidence/final-audit.json),
[complete accounting](../ap_reproduction/evidence/final-accounting.json),
[source and gate proof](../ap_reproduction/evidence/source-gate-verification.json)
and [512-pair provenance](../ap_reproduction/evidence/paired-media-provenance.json)
retain the evidence behind the presentation. The local
[Odyssey PDF](../../theOdessy/reproduction.pdf) summarizes the results. AP evaluates
supplied fixed goals 100-150 recorded actions ahead; full canonical-task completion
and safety require separate tests.

The [initial AP publication receipt](records/ap_reproduction/publication-receipt.json)
pins revision `995d62a638359a0ccf266a5b50c57e705ff10edb`. All 61 presentation
files passed fresh HTTP size and SHA-256 checks, and the original run tree remained
unchanged. The [local presentation checks](records/ap_reproduction/local-presentation-verification.json)
cover all 294 Markdown/HTML destinations and card parsing. The eight comparison
videos passed full decoding and source-hash checks. Signed-in Hugging Face browser
playback was not checked.

The [video refresh receipt](records/ap_reproduction/media-refresh-20261105/publication-receipt.json)
pins the improved gallery at `d85622b7bfd8654fd5888f5ce8d7e239c23c221d`.
All eight videos passed 30 fps, frame-count, timing and streaming-layout checks;
[render and link verification](records/ap_reproduction/media-refresh-20261105/render-and-link-verification.json)
also checks all 294 gallery destinations. Fresh downloads verified all 61
presentation files after replacing 24 video, poster and gallery files. Original
run bytes, scientific results, original clips and the PDF remain unchanged.
Versioned media links refresh browser caches after this update.

The AP pipeline is `ap_media.py` for fixed-order original clip selection and
saved-state plots, `ap_ghost_videos.py` for paired comparisons, `ap_catalog.py`
for navigation and results, and `publish_ap_report.py` for exact-manifest staging,
publication and fresh download verification. Generated media stays in ignored
`../runs/AP/publication-content-smooth-tinted/` and the private dataset, outside Git.
The earlier presentation folders retain the prior renderings for comparison.

The following record describes the baseline dataset's earlier reorganization.

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
