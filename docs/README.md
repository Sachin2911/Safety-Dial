# Documentation

## Active research

- [mainPlan/](mainPlan/README.md): the executable main plan adopted 26 September 2026:
  shared protocol, the Push-T study, a LeWM trained on Safety-Gymnasium Walker2d, the
  timeline and the infrastructure (fresh 5090 setup, Hugging Face checkpoints). It controls
  execution; committed results of the new study will live under `mainPlan/results/`.
- [researchDirection.md](researchDirection.md): the adopted research question and
  interpretation rules, **Which Experience Makes LeWM Safer to Use?**, adopted
  21 September 2026.
- [research/pilot.md](research/pilot.md): the 21 September E0-E5 checklist, superseded for
  execution by `mainPlan/`.
- [research/checkpoints.md](research/checkpoints.md): checkpoint and interface audit.
- [research/relatedWork.md](research/relatedWork.md): closest literature and claim boundaries.

## Preserved experiment records

- [safeDial/](safeDial/): Push-T report, LaTeX, figures and JSON/NPZ results.
- [phase0/](phase0/): recovery-triage report, LaTeX, figures and data.
- [Experimental notes](../notes/README.md), [code/notebooks](../experiments/README.md)
  and [animations](../animations/README.md).

The report build scripts remain in their original directories. Building PDFs needs the
LaTeX packages recorded in the setup history, including latexmk, biber and recommended
LaTeX/font packages. Existing compiled reports are preserved; cleanup did not rerun them.

## Academic records and references

Submitted ideation, bibliography, literature-review and proposal PDFs remain in their
original `submitted/` directories, with assignment guidance under `guides/`.
They document earlier work and do not override the active plan. [papers/](papers/) is the
reference library. `projectPresentation/` and `projectReport/` remain deliverable locations.

Superseded proposal drafts, brainstorming whiteboards and duplicate generated research
notes were removed on 21 September 2026. Tracked versions remain in git history.
