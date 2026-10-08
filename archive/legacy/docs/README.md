# Documentation

## Current direction

- [paperIdea.md](paperIdea.md): **evolution cheats imagination**, the current direction
  and its experiment plan. Chosen 3 October 2026, to be confirmed with the supervisor.
- [evoPlan/](evoPlan/README.md): the implementation plan (stages, code, tests, gates and
  dates); committed results will live under `evoPlan/results/`.
- [allocatedTopic.md](allocatedTopic.md): the supervisor's allocated topic, **Safe AI via
  Evolutionary Algorithms**, verbatim.
- [gecco2027.md](gecco2027.md): the target venue, with expected deadlines, submission rules
  and costs, as checked on 1 October 2026.
- [Deep-research report](research/reports/Evolutionary%20safe%20RL%20with%20world%20models.md)
  ([PDF](research/reports/Evolutionary%20safe%20RL%20with%20world%20models.pdf)) and its
  [notes](research/research_notes/Evolutionary%20safe%20RL%20with%20world%20models/):
  prior work, ranked directions, experiment designs and risks, 3 October 2026.

## Completed LeWM experience study (through S4)

The study asked which additional experience makes a LeWM safer to use. It ran on
26 and 27 September 2026 and stopped after S4. Its Walker LeWM, probes, policies and
snapshot branching are the starting assets for the current direction.

- [mainPlan/](mainPlan/README.md): the executable plan adopted 26 September 2026 and its
  committed results under `mainPlan/results/`. The
  [through-S4 report](mainPlan/reports/through-s4-20260927/README.md) summarises them.
- [researchDirection.md](researchDirection.md): that study's question and interpretation
  rules, **Which Experience Makes LeWM Safer to Use?**, adopted 21 September 2026.
- [research/pilot.md](research/pilot.md): the 21 September E0-E5 checklist, superseded for
  execution by `mainPlan/`.
- [research/checkpoints.md](research/checkpoints.md): checkpoint and interface audit.
- [research/relatedWork.md](research/relatedWork.md): closest literature and claim
  boundaries for that study.

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
They document earlier work and do not override the current direction. [papers/](papers/)
is the reference library. `projectPresentation/` and `projectReport/` remain deliverable
locations.

Superseded proposal drafts, brainstorming whiteboards and duplicate generated research
notes were removed on 21 September 2026. Tracked versions remain in git history.
