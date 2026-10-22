# SafetyDial

**Evolutionary execution adaptation for safe manipulation.**

Sachin Mohan, BSc Honours Computer Science, University of the Witwatersrand.
Supervised by Geraud Nangue Tasse.

Start with [firstPrinciples.md](firstPrinciples.md). It is the current plan: establish
the preconditions for an evolutionary safety dial on ManiGuard one measurement at a time,
beginning with a reproduction of the published Jar base rate, before any optimizer runs.

## Current workspace

- [First-principles plan](firstPrinciples.md): the four staged preconditions and gates.
- [ManiGuard base rates](docs/research/research_notes/ManiGuard%20base%20rates.md): the
  published four-outcome counts for every family and why Jar is the testbed.
- [ManiGuard paper](docs/papers/Maniguard.pdf): the benchmark, protocol and full results.
- [Allocated topic](docs/allocatedTopic.md) and [GECCO notes](docs/gecco2027.md).
- [Research concept note](docs/researchConceptNote.pdf) with
  [LaTeX source](docs/researchConceptNote/README.md).
- [Proposal template](docs/templates/research-proposal/).
- [Agent and engineering guidance](AGENTS.md).

## Archived research

Everything completed before 22 October 2026 is preserved unchanged under
[archive/](archive/README.md), with hash manifests checked by `archive/verify.py`.

- [22 October 2026 archive](archive/20261022/): ManiGuard Phase 3 Parts 1 to 4, the
  six-hypothesis study, the latent-evolution pilot and continuation screen, the GECCO
  negative-result draft, UNISafe studies, the previous research direction, plan history
  and all experiment code, configs and tests. Entry points:
  [previous direction](archive/20261022/Evolutionary%20improvement%20on%20ManiGuard.md),
  [progress audit](archive/20261022/reports/Recent%20ManiGuard%20progress%20audit.md),
  [negative-pilot paper](archive/20261022/docs/research/reports/maniguard-phase3-part1-gecco/named.pdf).
- [Legacy archive](archive/legacy/): Push-T Safe CEM, Hopper and Walker locomotion,
  LeWM experience study and Walker evolutionary studies.

## Setup

The [Vast template](configs/vast/README.md) is a ManiGuard preparation base with
Python 3.10 and CUDA 12.4.1. ManiGuard simulation, assets and policy serving are separate
manual steps described there. Clone after connecting:

```bash
git clone https://github.com/Sachin2911/Safety-Dial.git
cd Safety-Dial
```

Archived studies document their own environments; see the archive manifests and the
preserved `configs/` and `scripts/` under each archive root.
