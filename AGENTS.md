# Agent notes

## Current direction and authority

Read [firstPrinciples.md](firstPrinciples.md). As agreed on 1 November 2026, the
shared question is whether evolutionary search can learn compact adapters that
recover safe task completion around frozen world-model systems, reducing reliance
on manually tuned behavioural rules. LeWM is the opening study across TwoRoom,
PushT and OGBench Cube; Block Plucking is the intended main study. ManiGuard is
deferred. Its former Jar/Stack choices and gates are historical context.

[Rule42](SafeLeWM/Rule42.pdf), its [LaTeX source](SafeLeWM/Rule42.tex) and
[FirstExperiment.md](SafeLeWM/FirstExperiment.md) hold the working study design.
Matching an expert reference with demonstrably less tuning may be useful; optimizer
superiority is a separate claim. The adapter interface, protocols and budgets are
not frozen. Earlier experiments are exploratory context, not assumed fully ablated
or correct. Preserve their results as recorded.

Sachin authorised one fresh idle RunPod on 1 November, explicitly without running
anything on it. He then removed the proposed automatic stop and said he will tell
the agent when to turn it off. This is provisioning authority only, not authority
for installation, tests or experiments, and inherits no historical budget.

[principia.pdf](principia.pdf) is the consolidated experiment record, with one page
per experiment. The current manuscript is under `docs/research/honours-gecco/`.
As of 29 October 2026, the planning studies establish a useful fixed-configuration
lever; an evolutionary advantage and a replicated ManiGuard completion lever remain
unestablished. Task 19 reduced observed violations without improving completion.

Everything completed before 22 October 2026 lives under `archive/` and is historical
evidence, not a launch instruction. Do not resume archived pilots, screens or managed
jobs. Do not inherit their protocols, gates or budgets.

- **Student:** Sachin Mohan (2699183), BSc Honours CS, University of the Witwatersrand.
- **Supervisor:** Geraud Nangue Tasse.
- **Target venue:** [GECCO notes](docs/gecco2027.md) and [allocated topic](docs/allocatedTopic.md).

## Repository map

1. `firstPrinciples.md`: the current plan.
2. `docs/research/research_notes/`: current notes. `docs/papers/`: reference papers,
   compressed with Ghostscript `/ebook` before committing.
3. `docs/allocatedTopic.md`, `docs/gecco2027.md`, `docs/researchConceptNote/`,
   `docs/templates/`: topic, venue, concept note and document templates.
4. `progress/`: start at `README.md`. It holds machine-readable evidence, not duplicate
   task write-ups. On 29 October, Sachin requested consolidation into `principia.pdf`
   followed by removal of old documents while retaining code and evidence. The old
   task plans, prompts, reports, Odyssey and framing guide are preserved with original
   bytes in `archive/20261029/documents.tar.gz`; its manifest hashes every member.
   Restore required document paths from that bundle before running historical
   packaging, source-freeze scripts or tests that read them. Completed protocols do
   not authorize more work. Task 7's experiment is complete with closeout records
   pending; task 15's final recovery confirmation remains unverified. Neither status
   asserts a live rental. `receipts/README.md` indexes retained evidence; task 5 executed
   `05-adapter-plan-v4.json`, not its earlier versions. Todo 1 has one instance per
   prefix (`01` B, `01b` C, `01c` D), followed by `02-*`, `03-*`, `04-tracking-*` and
   `05-adapter-*` receipts. The finished
   todo 1 checklists and prompts are in git history up to `c248194`, todo 2's up to
   `b98b3cf`, todo 3's up to `af888ab`. Instance
   B's seed 0 is excluded from results and kept under `progress/receipts/excluded-instance-B/`.
   `scripts/todo1*/`, `scripts/todo2/`, `scripts/todo3/`: the pinned download, serving, lane,
   comparison, upload, rename, self-stop, recorder and supervisor-proxy scripts those
   checklists used; the paper-protocol eval config is
   `progress/receipts/01-jar-eval.yaml`. `configs/dsrl06/` pins the isolated learner
   dependencies; new Hydra configs go in `configs/<group>/`. The previous Vast templates, rental records and
   setup scripts were under `archive/20261022/configs/vast/` and `archive/20261022/scripts/`.
   Sachin requested deletion of all rental files on 29 October; the removal inventory
   is `archive/20261029/removals.json`. The older archive manifest records this explicit
   exception. Historical scripts may require separately recovered rental records.
5. `experiments/` contains the tracking diagnostic and execution adapter. Reusable
   code goes in `experiments/helpers/`,
   entry points in `experiments/scripts/`, tests in `experiments/tests/`. No `src/`.
6. `archive/20261022/`: the ManiGuard and UNISafe studies, previous direction, plan
   history, reports, code, configs and tests, moved unchanged on 22 October 2026 with
   `archive/20261022/manifest.json`. Check with
   `python archive/verify.py --manifest 20261022/manifest.json`.
7. `archive/legacy/`: Push-T, Hopper, Walker and LeWM studies from 8 October 2026,
   checked by `python archive/verify.py`.
8. `archive/20261029/`: the compressed superseded-document bundle and its inventory.
   Check with `python archive/verify.py --manifest 20261029/manifest.json`.
   Edit `docs/research/principia.tex` to update the consolidated record, then rebuild
   `principia.pdf` using the instructions in `docs/README.md`.
9. `SafeLeWM/`: all new work for the first experiment belongs here, as explicitly
   requested on 1 November. This includes documents, implementation, configurations,
   tests and ignored local outputs. Use `SafeLeWM/experiments/{helpers,scripts,tests}/`
   and `SafeLeWM/configs/` for new first-experiment code and configuration. Earlier
   shared code, evidence and archives retain their existing paths.

## Research controls

- Reproduce the published baseline before building on it. The ManiGuard paper's
  protocol is 20 Hz control, 8 executed actions per 16-step chunk, 256 by 256
  observations from collection-time cameras, family horizons (Jar 2,000), goal held for
  10 steps, three policy seeds. Match it and record any deviation.
- Score every episode under both the benchmark monitor and the requested-sequence rule
  (Jar closure before carry). Report safe completion, violations and unfinished episodes
  together, with denominators. Zero violations in n episodes bounds the rate at about
  3/n, not zero.
- Fewer interventions can mean inactivity. Report engagement alongside safety.
- Count every simulator interaction, including sweeps, discarded candidates and failed
  starts. Report model queries and elapsed time separately.
- The ManiGuard monitor labels violations and never vetoes actions. Keep proposed,
  filtered and executed actions distinguishable.
- Evaluate evolutionary learning against native, a tuned fixed reference, parameter
  random search and ordinary learning with the same information and declared budgets.
  Distinguish learned benefit, reduced tuning and optimizer superiority. Matching
  performance is not proof of equivalence; preset fallback is not learned discovery.
  A development-only ordinary-learning pilot may precede the full comparison.
- Freeze each stage's protocol before running it. Choose panels and seeds before reading
  outcomes. Keep held-out tasks untouched until the final comparison.
- Archived results stay as recorded. New experiments use new files and output paths.

## Environment and engineering

- Run experiment computation only on rented RunPod or Vast hosts. Local work is
  limited to editing, orchestration, transfers and inspecting saved evidence. This
  includes simulator preparation, training, tuning, evaluation and experiment tests.
- Public data downloads are authorized for future research work. Keep them within
  the active task's spending and storage limits; no repeated download approval is needed.
- Historical host: the Vast template "ManiGuard latent evolution" (hash
  `62414e8fca6befba085051de7ed36190`) on an RTX 3090 host with driver 580.x, installed
  from `progress/receipts/01-install.md` and `01-sources.json` at the pinned revisions
  (ManiGuard `6bf4c61` with the legacy 256 render profile). Baseline replication results
  of record come from machine 54514; later studies have their own host records and
  contemporary controls. Comparisons remain blocked by host. The selected task 6
  configuration used four RTX 4090 GPUs; its plan and $15 cap are preserved as
  `progress/06-dsrl-plan.md` inside the document bundle. It stopped at engineering gates. Do not rent or run
  from historical host records or resume its unused development allocation.
- Rollouts, sidecars and results rows live in the private dataset
  `Sachioster/maniguard-jar-replication-20261022`, task folders `<n>_<outcome>` with an
  `index.json` per run.
- Root `pyproject.toml` and `uv.lock` describe the legacy local environment, not a
  ManiGuard environment. Do not sync them into `/venv/main`. Use the intended
  interpreter explicitly. Locally, run tests with the venv's own Python, not `uv run`.
- Ruff line length is 100. Fix new lint errors instead of broadening ignores.
- Do not commit datasets, checkpoints, credentials, `wandb/`, `runs/` or generated run
  directories. Checkpoints belong in private Hugging Face repositories pinned by revision.
- Read archived environment notes before reusing MuJoCo or ManiGuard code, especially
  EGL ownership, renderer profile, chunk execution and action normalization.

## Secrets

Secrets stay in the ignored root `.env` or the remote account environment. Never print,
commit or pass them in configs, manifests or command arguments. Recognized keys:
`WANDB_API_KEY`, `GITHUB_TOKEN`, `HF_TOKEN`, `HF_NAMESPACE`, `GIT_AUTHOR_NAME`,
`GIT_AUTHOR_EMAIL`.

## Writing

Plain prose, no em or en dashes as clause breaks, plain hyphens in compound modifiers.
First-experiment notes go under `SafeLeWM/`; other current research notes go under
`docs/research/`. Do not create competing plan folders.
