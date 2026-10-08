# SafetyDial

**Evolutionary adaptation for task continuation under learned safety filtering.**

Sachin Mohan, BSc Honours Computer Science, University of the Witwatersrand.
Supervised by Geraud Nangue Tasse.

Start with [oracle.md](oracle.md), the central reference for the current research direction:
can evolutionary adaptation help a pretrained manipulation policy work more effectively
with its safety filter, improving safe task completion? The choice remains conditional on
finding meaningful, avoidable stalls. The method and experiment plan are still open.

## Current workspace

- [Research and documentation](docs/README.md): the manipulation literature, UNISafe asset
  and compute audits, reference papers and academic context.
- [Research oracle](oracle.md): decisions, scope and unresolved questions.
- [Experiment workspace](experiments/README.md): where new implementation will go.
- [Agent and engineering guidance](AGENTS.md).

## Preserved research

The [archive](archive/README.md) keeps earlier work together with its code, configurations,
tests, reports, notebook outputs and videos. Nothing was selected for deletion.

- **Push-T Safe CEM and safety dial:** [findings](archive/legacy/notes/safeCEM.md),
  [report](archive/legacy/docs/safeDial/SafeDialReport.pdf) and
  [demonstrations](archive/legacy/animations/README.md).
- **Hopper and Walker locomotion:** [Phase 0 findings](archive/legacy/notes/phase0Report.md),
  [report](archive/legacy/docs/phase0/Phase0Report.pdf) and
  [videos](archive/README.md#hopper-and-walker-locomotion).
- **LeWM experience study:** [through-S4 report](archive/legacy/docs/mainPlan/reports/through-s4-20260927/README.md).
- **Walker evolutionary studies:** [recorded results](archive/legacy/docs/evoPlan/results/README.md)
  and [paper draft](archive/legacy/docs/geccoPaper/main.pdf).

The archive index explains the layout, provenance and reproduction entry points. Earlier
plans and scheduled-job configurations describe their original studies.

## Environment

Python 3.11 and `uv` remain the shared environment for the repository.

```bash
bash scripts/setup.sh
uv run python scripts/download_data.py --cfg job
```

The downloader retains the existing Push-T/Cube interfaces. Downloading those assets is
optional for the new research direction. Local data, credentials and upstream checkouts
remain at their existing root locations.

Run preserved code through the archive launcher, which uses the root environment and
sets the historical working directory:

```bash
bash archive/run.sh python experiments/scripts/safe_dial_pusht.py --help
bash archive/run.sh python -m pytest -q
uv run python archive/verify.py
```

See [archive verification](archive/verification.md) for the checks performed during the
reorganisation and the local multiprocessing-test limitation.
