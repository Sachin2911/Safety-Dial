# SafetyDial

**Evolutionary execution adaptation for safe manipulation.**

Sachin Mohan, BSc Honours Computer Science, University of the Witwatersrand.
Supervised by Geraud Nangue Tasse.

Start with [Evolutionary improvement on ManiGuard](Evolutionary%20improvement%20on%20ManiGuard.md),
the current research direction: evolve a shared controller around frozen manipulation
policies, with safe completion and transfer across tasks as the targets. It compares
latent steering options, prior evidence, matched baselines and staged evaluation.
The proposed method and experiment budgets still require a prospective protocol freeze.

## Current workspace

- [Research and documentation](docs/README.md): the manipulation literature, UNISafe asset
  and compute audits, proposal template and academic context.
- [Current research direction](Evolutionary%20improvement%20on%20ManiGuard.md): shared
  latent controllers, low-cost extensions and cross-task comparisons.
- [Agent execution prompt](docs/research/research_notes/ManiGuard%20latent%20evolution%20goal%20prompt.md):
  implementation, recovery, the current rental limits and GitHub/Hugging Face preservation.
- [Archived Phase 3 plans](docs/research/history/20261019/README.md): the earlier
  direction, Parts 1 to 4 and recovery proposal, with original contents preserved.
- [Part 4 premise result](docs/research/reports/ManiGuard%20Part4%20counterexample%20premise.md):
  frozen direct prediction check, scientifically gated before physical acquisition.
- [Shared GECCO-format report](docs/research/reports/maniguard-phase3-part1-gecco/named.pdf):
  Parts 1 through 4, including adverse and null results.
- [Earlier research plans](docs/research/history/README.md): preserved decisions and studies.
- [Paused phase-two pilot](docs/research/history/20261018/phase2.md): preserved protocol and handoff.
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

## Setup

The current [Vast template](configs/vast/README.md) is a lightweight ManiGuard preparation
base with Python 3.10 and CUDA 12.4.1. Clone the repository after connecting:

```bash
git clone https://github.com/Sachin2911/Safety-Dial.git
cd Safety-Dial
```

ManiGuard simulation, assets and model-serving environments are separate manual preparation
steps described in the template notes. The template does not install or validate them.
For the completed UNISafe study, use the preserved
[Python 3.12 / CUDA 13 platform](configs/vast/UNISAFE.md) and its installer instead.

## Install Codex and start the agent on Vast

On the rented machine, load the image's Node installation and install the CLI. The
current base has Node 24, npm and tmux. These commands follow the
[official Codex installation guidance](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex)
and [headless authentication instructions](https://learn.chatgpt.com/docs/auth#login-on-headless-devices).

```bash
source /opt/nvm/nvm.sh
npm install -g @openai/codex@latest
codex --version
codex login --device-auth
```

Complete device-code sign-in in your browser. Enable device-code login in your ChatGPT
security/workspace settings if required. Keep the authentication cache outside Git.

Clone `main` into a fresh working directory if you have not already cloned it:

```bash
cd /workspace
git clone --branch main https://github.com/Sachin2911/Safety-Dial.git
cd Safety-Dial
tmux new -s maniguard
codex
```

For an existing clone, first inspect `git status` and update it without discarding local
work. In Codex, paste:

```text
/goal Execute docs/research/research_notes/ManiGuard latent evolution goal prompt.md, following AGENTS.md and Evolutionary improvement on ManiGuard.md. Implement and physically evaluate the authorized pilot, diagnose and repair failures, preserve fair comparisons, and verify GitHub/Hugging Face backups within the current rental deadline.
```

The [full prompt](docs/research/research_notes/ManiGuard%20latent%20evolution%20goal%20prompt.md)
contains the current instance/deadline, private HF input repositories and new output
destinations. [The dated rental record](configs/vast/maniguard-rental-20261019.json)
records the two-GPU allocation; verify it is still active before using it. Starting a
goal does not extend the rental. Detach tmux with `Ctrl-b`, then `d`; reconnect using
`tmux attach -t maniguard`. This preserves the terminal across SSH disconnections.

## Pretrained UNISafe checkpoints

The original Google Drive checkpoints are mirrored in the private
[Sachioster/unisafe-pretrained](https://huggingface.co/Sachioster/unisafe-pretrained)
model repository. With `HF_TOKEN` available, install the additive dependencies and
download the pinned revision with all original file hashes verified:

```bash
bash scripts/setup_unisafe.sh
```

The source revision and manifest hash are recorded in
[configs/download/unisafe.json](configs/download/unisafe.json). Files are saved under
ignored `data/unisafe/pretrained/`: `latest.pt` is the Dreamer checkpoint and
`filter/model/` contains actor/critic snapshots from steps 10000 through 200000.
The mirror includes original configuration, provenance and tensor inspection metadata.
Frozen inference and the migrated environment have executable checks in
[experiments/README.md](experiments/README.md#unisafe-engineering-checks). Run both with
`bash scripts/smoke_unisafe.sh` after `bash scripts/setup_unisafe.sh`. Closed-loop
policy performance remains unvalidated.
