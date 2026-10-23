# SafetyDial

**Evolutionary execution adaptation for safe manipulation.**

Sachin Mohan, BSc Honours Computer Science, University of the Witwatersrand.
Supervised by Geraud Nangue Tasse.

Start with [firstPrinciples.md](firstPrinciples.md). It is the current plan: establish
the preconditions for an evolutionary safety dial on ManiGuard one measurement at a time,
beginning with a reproduction of the published Jar base rate, before any optimizer runs.

## Current workspace

- [First-principles plan](firstPrinciples.md): the four staged preconditions and gates.
- [Jar replication and noise floor](docs/research/research_notes/Jar%20replication%20and%20noise%20floor.md):
  stage 1 done on 23 October 2026; the measured Jar base rate, the same-seed noise
  floor and the per-task structure, with the single-host decision.
- [ManiGuard base rates](docs/research/research_notes/ManiGuard%20base%20rates.md): the
  published four-outcome counts for every family and why Jar is the testbed.
- [Progress](progress/README.md): checklists, receipts and the rental record; the
  rollout dataset is `Sachioster/maniguard-jar-replication-20261022` (private).
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

## Setup on Vast

No instance is running. To run the next checklist, rent an RTX 3090 or 4090 host on
NVIDIA driver 580.x with the private **ManiGuard latent evolution** template (hash
`62414e8fca6befba085051de7ed36190`); the results of record came from machine 54514
(two RTX 3090), and comparisons must stay on one host. The template provides Ubuntu
22.04, Python 3.10 in `/venv/main`, CUDA 12.4.1, tmux, git and uv, and installs no
research packages. `HF_TOKEN` and `GITHUB_TOKEN` are set in the instance environment;
never print them. Record the rental in `progress/rental-<date>.json`.

Connect, install Claude Code, clone and start the agent inside tmux so it survives
disconnects (`Ctrl-b d` detaches, `tmux attach -t jar` reconnects):

```bash
ssh -p <port> root@<host>.vast.ai
source /opt/nvm/nvm.sh && npm install -g @anthropic-ai/claude-code
cd /workspace && git clone https://github.com/Sachin2911/Safety-Dial.git && cd Safety-Dial
tmux new -s jar
claude
```

Sign in when prompted, then paste the goal prompt for the checklist at hand. The todo 1
prompts ([goal-prompt-01.md](progress/goal-prompt-01.md),
[goal-prompt-01b.md](progress/goal-prompt-01b.md),
[goal-prompt-01c.md](progress/goal-prompt-01c.md)) are kept as the pattern: the agent
replays the pinned install from `progress/receipts/01-install.md`, works through its
checklist, commits evidence under `progress/`, uploads rollouts to the dataset and
stops its own instance with `scripts/todo1c/stop_instance.py`.

Earlier Vast templates, rental records and setup scripts are preserved under
[archive/20261022/configs/vast](archive/20261022/configs/vast/) and
[archive/20261022/scripts](archive/20261022/scripts/). Root `pyproject.toml` and
`uv.lock` describe the local legacy environment used by `archive/run.sh`.
