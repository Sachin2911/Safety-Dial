# Agent notes

## Current direction and authority

Read [firstPrinciples.md](firstPrinciples.md). The goal is unchanged: a compact
evolved controller that raises safe task completion of frozen ManiGuard policies under
matched interaction budgets. The plan now establishes four preconditions in order,
variance, a lever, stable evaluation, then an optimizer comparison, and each stage is
frozen before it runs. Jar is the chosen family; Stack is the backup. The published
base rates and the reasons are in
[ManiGuard base rates](docs/research/research_notes/ManiGuard%20base%20rates.md).

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
4. `progress/`: start at `README.md`. `theOdessy.pdf` is the running log with one short
   entry per finished task (LaTeX source `theOdessy.tex` beside it; add a section per task
   and recompile with `pdflatex`). Tasks 1 through 6 are complete. Task 6 stopped
   at engineering gates; no DSRL training or evaluation ran. Its result is
   `receipts/06-dsrl-results.md`. Its plan and source freezes are historical launch
   evidence, not permission to resume unused work.
   Task 7 is now authorized under `progress/07-dsrl-repair.md`: repair numerical
   integration, verify live execution and pursue a development learning result,
   retaining the cumulative $15 DSRL cap and all task-6 historical charges.
   Task 7 uses fresh `dsrl07` paths and freezes. The original screen
   and launch prompt are under `superseded/`, with inactive notices at their old paths.
   Rental records and completed protocols retain their existing paths for source and
   script compatibility. `receipts/README.md` indexes final evidence; task 5 executed
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
   setup scripts are under `archive/20261022/configs/vast/` and `archive/20261022/scripts/`.
5. `experiments/` contains the tracking diagnostic and execution adapter. Reusable
   code goes in `experiments/helpers/`,
   entry points in `experiments/scripts/`, tests in `experiments/tests/`. No `src/`.
6. `archive/20261022/`: the ManiGuard and UNISafe studies, previous direction, plan
   history, reports, code, configs and tests, moved unchanged on 22 October 2026 with
   `archive/20261022/manifest.json`. Check with
   `python archive/verify.py --manifest 20261022/manifest.json`.
7. `archive/legacy/`: Push-T, Hopper, Walker and LeWM studies from 8 October 2026,
   checked by `python archive/verify.py`.

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
- Before any evolutionary claim, compare against native, a tuned fixed setting,
  parameter random search and ordinary learning with the same information. A
  development-only ordinary-learning pilot may precede that optimizer comparison.
- Freeze each stage's protocol before running it. Choose panels and seeds before reading
  outcomes. Keep held-out tasks untouched until the final comparison.
- Archived results stay as recorded. New experiments use new files and output paths.

## Environment and engineering

- Historical host: the Vast template "ManiGuard latent evolution" (hash
  `62414e8fca6befba085051de7ed36190`) on an RTX 3090 host with driver 580.x, installed
  from `progress/receipts/01-install.md` and `01-sources.json` at the pinned revisions
  (ManiGuard `6bf4c61` with the legacy 256 render profile). Baseline replication results
  of record come from machine 54514; later studies have their own host records and
  contemporary controls. Comparisons remain blocked by host. The selected task 6
  configuration used four RTX 4090 GPUs; its plan and $15 cap are in
  `progress/06-dsrl-plan.md`. It stopped at engineering gates. Do not rent or run
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
Current research notes go under `docs/research/`. Do not create competing top-level plan
folders.
