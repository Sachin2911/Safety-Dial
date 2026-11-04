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

## Current execution status

Use the "Active goal" section of [firstPrinciples.md](firstPrinciples.md) for the
latest authority. Keep the status there rather than duplicating the full chronology
in this file. The prior navigation snapshot was removed with the root archive in
the [3 November deletion](docs/records/progress/receipts/local-deletion-20261103.md).

As recorded on 2 November:

- The L4 population-size study remains active only under
  `SafeLeWM/EvolutionBudgetStudy.md`: population four versus eight and matched random
  search, two development repeats, 240,000 training and 308,400 total actions.
- The adapter pilot, replication, representation/mechanism study, matched PPO and
  PushT AP reproduction are complete. Their protocols authorize no additional runs.
- All PushT screens are complete. The validated screen failed useful adapter
  authority; no optimizer training followed and twelve reserved cases are untouched.
  Do not resume a completed screen or substitute a new interface study for a failed gate.
- The safety-dial library is complete and preserved. It does not establish calibrated
  risk and authorizes no new run. See Rule43 accounts 14-17 and the results index.
- Block Plucking has its own work area and protocols. Preserve concurrent work there.
- All rental shutdowns remain manual. This cleanup changes no rental or execution
  authority. No historical allocation or completed protocol resumes automatically.
- The current manuscript is `docs/research/honours-gecco/world-model-adapters.tex`;
  retain the October thesis and earlier recorded results unchanged.

[principia.pdf](docs/research/principia.pdf) is the consolidated experiment record, with one page
per experiment. The current manuscript is under `docs/research/honours-gecco/`.
As of 29 October 2026, the planning studies establish a useful fixed-configuration
lever; an evolutionary advantage and a replicated ManiGuard completion lever remain
unestablished. Task 19 reduced observed violations without improving completion.

The former `archive/` held work completed before 22 October 2026; Sachin authorized
its deletion on 3 November. Historical evidence is not a launch instruction.
Do not resume archived pilots, screens or managed
jobs. Do not inherit their protocols, gates or budgets.

- **Student:** Sachin Mohan (2699183), BSc Honours CS, University of the Witwatersrand.
- **Supervisor:** Geraud Nangue Tasse.
- **Target venue:** [GECCO notes](docs/gecco2027.md) and [allocated topic](docs/allocatedTopic.md).

## Repository map

1. `firstPrinciples.md`: current direction and execution authority.
2. `SafeLeWM/`: LeWM design, protocols, results, code and cloud recovery records.
   New code goes in `SafeLeWM/experiments/{helpers,scripts,tests}/` and configuration
   in `SafeLeWM/configs/`. Its relative runtime paths stay within SafeLeWM.
3. `SafeBlock/`: Block Plucking design, protocols and code. Required UNISafe helpers
   live in `SafeBlock/experiments/helpers/`, with settings in
   `SafeBlock/configs/unisafe09/`. Ignored weights/runtime data and the pinned upstream
   checkout live in `SafeBlock/data/` and `SafeBlock/third_party/`. Use
   `SafeBlock/experiments/project_paths.py` for historical asset/output prefixes;
   frozen configuration bytes remain unchanged.
4. `docs/`: manuscripts, academic material, papers and retained records. Oracle is
   under `Paper/`; the consolidated experiment record is
   `docs/research/principia.pdf`. Edit its adjacent `.tex` source and use the build
   instructions in `docs/README.md`.
5. `docs/records/`: original progress receipts, frozen historical configs, figures
   and environment records moved from the root. The consolidation receipt maps
   original paths and hashes to their new locations. Historical record contents
   retain old paths; resolve them through `docs/records/README.md`.
6. `LeWM_Experiment1/`: Sachin's explicitly requested new experiment preparation
   area. Its compute pipeline, tests and future experiment
   files belong here. This is a scoped exception to the SafeLeWM placement rule;
   keep the existing SafeLeWM protocols and recorded results in place.
7. `theOdessy/`: the LeWM checkpoint-reproduction overview, with LaTeX source and
   PDF. Sachin explicitly requested this folder at the repository root.

The old root `experiments/`, `scripts/`, `configs/`, `data/`, `third_party/`,
`progress/`, `results/` and `environment/` are removed after relocation of needed
code/assets and retained records. Do not recreate root-level campaign folders.
Current source preparation, tests and simulations remain restricted to rented hosts
where applicable; cleanup authorizes no experiments or rental changes.

## LeWM Experiment 1 compute

Start with [the experiment README](LeWM_Experiment1/readme.md) and
[the compute guide](LeWM_Experiment1/computeConfigs/README.md). Use
`python3 LeWM_Experiment1/computeConfigs/compute.py` from the repository root for
routine Vast operations. Use the wrapper before consulting provider docs or
writing ad hoc CLI/API commands; it owns selection, session identity, SSH, uploads
and destruction checks. Local setup needs Python 3.11+, Git and OpenSSH, plus `gh`
when using the existing GitHub login for private image access.

- `search` is read-only. `up` rents one instance with explicit hours, maximum
  hourly price, input/output transfer allowances and estimated budget. Use the
  session's current authorization and limits; examples and historical receipts
  grant no new rental or research authority. Billing continues until destruction.
- Use `status`, `ssh`, `run`, `logs`, `stop` and `sync` for the active session.
  Keep ignored `computeConfigs/.state/`; it records the current instance identity.
  If allocation is uncertain or boot fails, use `recover` before another `up`.
- Runtime environment settings are already in `config/vast-template.json`.
  Credentials are loaded automatically from the ignored root `.env`, process
  environment or existing CLI logins. `GHCR_READ_TOKEN` is optional when `gh`
  already has package pull access. Never print `gh auth token` or upload `.env`.
  The HF token travels over SSH stdin to the uploader, not the experiment child.
  W&B credentials and arbitrary environment forwarding are not implemented.
- `run` needs an actual source directory, JSON config, prospective protocol,
  seeds and finite runtime. No scientific entrypoint is supplied by the pipeline.
  Outputs go under `LEWM_RUN_DIR` and then to private dataset
  `Sachioster/lewm-experiment1`, prefix `runs/<run-id>/`. Use `run_io` for periodic
  publication; the compute guide defines the artifact and verification contract.
- Destroy only when the user directs shutdown. `compute.py destroy` checks idle
  jobs, verified backups, and the exact instance identity. `stop` ends the job
  only. Do not bypass a failed backup check or assume a deadline stops billing.
- Runtime/build checks execute on rented hosts. The fake-provider orchestration
  tests in `LeWM_Experiment1/tests/compute/` run locally with `.venv/bin/python`;
  they import no simulator. Preserve `image/requirements.lock` and runtime checks.

## Repository cleanup, 3 November 2026

The initial cleanup bundled older notes and navigation snapshots in the root archive.
Sixteen SafeLeWM evidence archives were then moved to private Hugging Face storage
with independent full-download verification and a successful restoration drill.

Sachin subsequently explicitly confirmed deletion of all three remaining groups,
after clarification that they included active work and evidence not yet verified
in cloud storage. The exact local scope is `archive/`, `outputs/`,
`SafeLeWM/outputs/` and `SafeBlock/outputs/`. This supersedes earlier preservation
requirements for those paths only. The [deletion record](docs/records/progress/receipts/local-deletion-20261103.md)
and adjacent JSON receipt document completion and a metadata-only file inventory.
The older root document bundles and navigation snapshots were included. Do not
claim that all deleted files have verified backups. Historical preservation records
describe their original verification dates, not current local availability.

Current code outside the root archive, protocols, result reports and the verified
SafeLeWM cloud archive remain. Local deletion changes no rental or execution
authority and authorizes no new experiment or remote deletion.

Sachin then approved consolidating the repository into SafeLeWM, SafeBlock and
`docs/`: required UNISafe runtime dependencies moved into SafeBlock, manuscripts and
records moved into docs, and unused legacy campaign code/assets were deleted.
See `docs/records/consolidation-20261103.json` for the path map and hashes. This
supersedes earlier instructions to retain the old root code/configuration paths.

## Research controls

On 3 November Sachin authorized moving archived evidence to private Hugging Face
storage and keeping recovery notes. The initial sixteen-archive SafeLeWM batch is
documented in `SafeLeWM/archive/README.md`, `cloud-index-20261103.json` and
`cloud-receipt-20261103.json`. Consult the receipt for actual transfer/removal status.
Fetch any absent archives using `SafeLeWM/experiments/scripts/fetch_cloud_archive.py`
before running the original archive verifier. Preserve pinned cloud revisions and
original manifests. This migration changes no experiment or rental authority.

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
  from `docs/records/progress/receipts/01-install.md` and `01-sources.json` at the pinned revisions
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
`GIT_AUTHOR_EMAIL`, `VAST_API_KEY`, `GHCR_READ_TOKEN`. A recognized key is not
automatically forwarded to a rented instance; see the compute guide's environment
and credential table.

## Writing

Plain prose, no em or en dashes as clause breaks, plain hyphens in compound modifiers.
Existing SafeLeWM study notes stay under `SafeLeWM/`; notes for the explicitly
requested new experiment go under `LeWM_Experiment1/`. Other current research
notes go under `docs/research/`. The reproduction overview is the explicit
root-level `theOdessy/` exception. Do not create competing plan folders.
