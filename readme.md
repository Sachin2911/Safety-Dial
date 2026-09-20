# SafetyDial

**When Are JEPA Predictions Reliable Enough for Safe Planning?**

Sachin Mohan (2699183), BSc Honours CS, University of the Witwatersrand.
Supervised by Geraud Nangue Tasse.

## The idea

SafetyDial studies whether a predictive latent world model makes reliable safety decisions
when a planner chooses the actions. A model may predict ordinary trajectories accurately
while CEM selects plans whose predicted constraint satisfaction is overly optimistic.

**Research question.** Does planner optimisation amplify optimistic safety errors in
JEPA-style world models, and can a simple correction reduce those errors while preserving
useful task performance?

The adopted plan accepts Safety-Gymnasium's supplied cost constraints and uses programmatic
labels. It audits actual versus imagined futures, compares ordinary and planner-selected
actions, and tests one small correction such as a horizon-dependent error margin. The
**Safety Dial** controls deployment conservatism on those predictions. Closed-loop Safe-CEM
is an extension after the offline evidence is established.

Full adopted plan, hypotheses, metrics, baselines, gates and first pilot:
[`docs/researchDirection.md`](docs/researchDirection.md). Repo conventions and current
status: [`AGENTS.md`](AGENTS.md).

The existing [Phase 0 findings](notes/phase0Report.md) explain why termination should not
be treated as irreversibility. The [Safe-CEM pilot](notes/safeCEM.md) motivates auditing
imagined safety decisions and records corrections to the original penalty comparison.

> **Direction adopted 20 September 2026.** The earlier label-free irreversibility proposal
> in `docs/revisedProp/` is superseded by the Markdown plan above. The older NSGA-II and
> LLM-alignment documents are historical. This project now studies supervised constraint
> prediction; it does not require an irreversible-failure environment or a new safety theorem.

## Setup

```bash
git clone https://github.com/Sachin2911/Safety-Dial.git
cd Safety-Dial
bash scripts/setup.sh
```

Then download the LeWM source, checkpoints and expert data:

```bash
uv run python scripts/download_data.py                 # all: source + Push-T + Cube
uv run python scripts/download_data.py --config-name pusht
uv run python scripts/download_data.py --config-name cube
uv run python scripts/download_data.py weights_only=true   # skip ~60 GB datasets
uv run python scripts/download_data.py clone_source=false  # skip third_party/le-wm
```

> **Disk warning.** The default (`all`) pulls both datasets. Push-T is ~13 GB compressed and
> decompresses to ~46 GB; Cube is ~46 GB compressed and decompresses far larger. Check free
> space before running the default, or use `--config-name pusht` / `weights_only=true`.

Work happens in the notebooks under [`experiments/`](experiments/); run JupyterLab with
`experiments/` as the working directory so `import helpers.linProbeHelpers` resolves.

## Documents produced

All under [`docs/`](docs/).

- [x] Ideation document
- [x] Annotated bibliography
- [x] Literature review
- [x] Research proposals (historical, including `docs/revisedProp/`)
- [x] Adopted research direction, [`docs/researchDirection.md`](docs/researchDirection.md)
- [ ] Project presentation
- [ ] Project report
