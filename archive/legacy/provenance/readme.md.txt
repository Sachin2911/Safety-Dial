# SafetyDial

**Evolution cheats imagination: keeping evolved policies safe in learned world models.**

Sachin Mohan (2699183), BSc Honours CS, University of the Witwatersrand.
Supervised by Geraud Nangue Tasse. Allocated topic:
[Safe AI via Evolutionary Algorithms](docs/allocatedTopic.md). Target venue:
[GECCO 2027](docs/gecco2027.md).

## Current direction

If you train a robot to be safe inside its own imagination, it learns to cheat the
imagination. This project measures how much, and tests a fix.

1. **Practise in your head.** Evolution needs many trials. Running them inside a learned
   world model (LeWM) is cheap, and nobody gets hurt.
2. **But the imagination is sometimes wrong about danger.** Evolution keeps whatever
   scores best, which is often a policy that found a spot where the model wrongly says
   "safe".
3. **The fix.** Add noise to the imagination where it is unsure, and use ROSARL's penalty
   so that falling costs enough, with no hand-tuned number.
4. **The check.** Replay every policy evolution selects from the same saved MuJoCo state,
   and compare imagined safety with real safety.

The first experiment uses the existing Walker2d LeWM and needs no new model training.
Chosen 3 October 2026, to be confirmed with the supervisor.

- [Paper idea and experiment plan](docs/paperIdea.md): the direction, the steps, the
  go/no-go gate and the timeline.
- [Implementation plan](docs/evoPlan/README.md): stages, code to write, tests, gates,
  dates and run sizes.
- [Deep-research report](docs/research/reports/Evolutionary%20safe%20RL%20with%20world%20models.md)
  ([PDF](docs/research/reports/Evolutionary%20safe%20RL%20with%20world%20models.pdf)):
  prior work, ranked directions and risks.
- [GECCO 2027 notes](docs/gecco2027.md) and the [allocated topic](docs/allocatedTopic.md).
- [Agent and engineering guidance](AGENTS.md).

## Completed work

- **LeWM experience study, through S4 (26 and 27 September 2026).**
  [Report](docs/mainPlan/reports/through-s4-20260927/README.md),
  [plan](docs/mainPlan/README.md) and [question](docs/researchDirection.md). On Push-T,
  the diagnosis passed (21 of 28 attributable false-safe decisions came from imagined
  dynamics) but the repair gate was not met. On Walker2d, a LeWM trained from scratch
  (214,780 updates) qualified for the health rule, and random versus boundary-focused
  experience showed no consistent advantage. Its Walker LeWM, probes, policies and
  snapshot branching are the starting assets for the current direction.
- **Safe CEM and the safety dial on Push-T.** [Findings](notes/safeCEM.md),
  [report and results](docs/safeDial/).
- **Phase 0 locomotion triage.** [Findings](notes/phase0Report.md),
  [report and data](docs/phase0/): benchmark termination is not irreversibility.
- [Experiment index](experiments/README.md), [experimental notes](notes/README.md) and
  [animations](animations/README.md).
- Submitted academic deliverables and reference papers remain under `docs/`.
  They record earlier work and do not define the current research requirements.

## Setup and asset recovery

```bash
git clone https://github.com/Sachin2911/Safety-Dial.git
cd Safety-Dial
bash scripts/setup.sh
```

The historical Vast assets (checkpoint copies, fitted normalisers, probe cache) no longer
exist. Rebuild them on a fresh RTX 5090 instance following
[docs/mainPlan/infrastructure.md](docs/mainPlan/infrastructure.md). Weights alone do not
supply the fitted normalisers used by the existing planner. Every model checkpoint is
stored in private Hugging Face repositories, never only on the instance and never in git.

Downloads (Push-T is the default):

```bash
# Source and Push-T checkpoint only; still needs normalisers/data for the full pilot.
uv run python scripts/download_data.py weights_only=true

# Push-T weights and expert data.
uv run python scripts/download_data.py

# Optional other supported datasets, not required for the first pilot.
uv run python scripts/download_data.py --config-name cube
uv run python scripts/download_data.py --config-name all
```

Push-T expert data is about 13 GB compressed; Cube is about 46 GB compressed and is
optional. Check storage before downloading and decompressing. Preview configuration
without downloading with `uv run python scripts/download_data.py --cfg job`.

Run JupyterLab with `experiments/` as the working directory and use the
**Python (safetydial .venv)** kernel. New reusable code belongs in
`experiments/helpers/`, with small entry points in `experiments/scripts/`. Existing
experiment files are read-only: new work adds new files and writes to new paths.
