# SafetyDial

**Label-Free Reachability in Joint-Embedding Predictive Architectures for Safe Reinforcement
Learning.** Subtitle: The Safety Dial, irreversibility as a deployment-time control.

Sachin Mohan (2699183), BSc Honours CS, University of the Witwatersrand.
Supervised by Geraud Nangue Tasse.

## The idea

Latent safety filters removed the hand-tuned penalty weight from safe RL, which was real
progress. They did not remove supervision: the failure set that seeds the reachability
computation is a classifier trained on human-annotated failure observations, so the filter is
blind to any failure mode nobody thought to label.

This project asks whether that failure set can be derived from the dynamics instead. The
criterion is **irreversibility**: a transition is unsafe when no available action sequence
returns the system to where it was. That is a property of the transition structure, needs no
annotation, and is only computable with a world model, since it depends on rollouts of actions
that were never taken.

The work builds return-reachability estimators in the latent space of a JEPA world model
(LeWM / SIGReg), a substrate where reachability analysis has not been attempted, and then
exposes the resulting conservatism as the **Safety Dial**: a threshold an operator sets at
deployment rather than an engineer fixing during training.

**Research question.** Can the failure set required by a latent safety filter be derived from
irreversibility in the dynamics, without failure labels, and does such a filter, computed in a
JEPA latent, detect failure modes that a supervised latent safety filter trained on different
labelled failures cannot?

Full plan, method, evidence and timeline: [`docs/revisedProp/RevisedProposal.pdf`](docs/revisedProp/RevisedProposal.pdf)
(LaTeX source in [`docs/revisedProp/latex/`](docs/revisedProp/latex/), rebuild with
`./docs/revisedProp/compile.sh`). Repo conventions and current status:
[`AGENTS.md`](AGENTS.md).

> **Note on older documents.** The project pivoted in August 2026 from evolutionary
> multi-objective planning (NSGA-II Pareto fronts) to label-free irreversibility, and
> revised again in September 2026 onto the Safety-Gymnasium locomotion suite. The
> deliverables under `docs/*/submitted/` were written under the older framing and are kept
> as the record of graded work; their LaTeX sources were removed on 20 September 2026.

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
- [x] Research proposal (superseded by `docs/revisedProp/`)
- [ ] Project presentation
- [ ] Project report
