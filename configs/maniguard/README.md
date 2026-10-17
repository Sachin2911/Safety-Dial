# ManiGuard fruit experiments

This study implements the six hypotheses in [fruits.md](../../fruits.md). Physical
mechanism screens are running; optimizer comparisons and reserved evaluation remain pending. Passing controller tests does not
establish improved task completion or safety.

`sources.json` pins upstream code, benchmark assets and family-specific policy checkpoints.
The simulator uses Python 3.10 with native conda-forge Spot; the policy uses the upstream
OpenPI lockfile in a separate environment. Do not sync the root Python environment into
these environments. `scripts/maniguard/setup_sim.sh` installs simulator dependencies.
Clone the pinned upstream repositories first, including ManiGuard's BEHAVIOR submodule.
Isaac rendering also requires the system packages `libglu1-mesa` and `libxt6`;
GPU driver availability alone does not establish that these userspace libraries exist.

Download assets with `scripts/maniguard/download.py` using an interpreter with
`huggingface_hub` installed. Kinds are `benchmark`, `assets`, `robot`, `jar`, `lid`, `stack`.
Download `robot` after `assets` finishes. Install the BEHAVIOR encryption key through
OmniGibson's official asset installer, subject to its academic research terms. Never
commit or redistribute the key, simulator data or policy weights.

The frozen split in `splits.json` assigns whole task directories, including every variant,
to engineering, training, validation or test. Engineering scenes diagnose integration and
failure mechanisms. Candidate search uses training scenes, selection uses validation,
and final evaluation uses test scenes only after selection freezes. These splits do not
establish that scenes were excluded from the released VLA's training set.

Run the policy locally with `scripts/maniguard/serve.sh --family jar`. Manage persistent
serving through supervisor. It listens on 127.0.0.1:18000. Every episode has a distinct
identifier; the server resets sampling from the benchmark seed even when successive
methods use the same scene and seed.

Run a single episode with the simulator interpreter:

```bash
data/maniguard/envs/sim/bin/python experiments/scripts/maniguard_episode.py \
  --family jar --scene task_0013/base --seed 0 \
  --adapter configs/maniguard/identity.json \
  --output runs/maniguard/engineering/jar-baseline-0013-seed0
```

The launcher selects the legacy visual profile and the checkpoint's joint-position
controller, camera and grasping settings. It preserves the upstream task horizon and
physical monitor. `--max-steps` and `--random-policy` are engineering diagnostics only.
Each output path must be new. Source patch anchors and the original evaluator revision
are checked; the generated runtime and hashes are saved with each episode.

Initial adapter configurations are examples to screen, not selected winners. The
prerequisite and hand-back prototypes require explicitly labelled privileged object
predicates and contact association. Other adapters use robot observations and history.
The monitor still measures physical violations without changing their definitions.

Traces retain model chunks, controller proposals, adapted actions, clipped executed
commands and observations. The ledger counts physical ticks, environment steps, policy
queries and time, including initialization. Upstream result files preserve completion,
violations and errors. A failed or incomplete monitor must not be scored as safe.

Controller and sampling-isolation checks:

```bash
data/maniguard/envs/sim/bin/python -m pytest -q \
  experiments/tests/test_maniguard_adaptation.py \
  experiments/tests/test_maniguard_policy.py
```

A predeclared sequential batch uses the same simulator interpreter:

```bash
data/maniguard/envs/sim/bin/python experiments/scripts/maniguard_batch.py \
  --plan configs/maniguard/jar-baseline-engineering.json \
  --output runs/maniguard/jar-baseline-engineering-v1
```

Run persistent batches through supervisor. The batch creates a fresh process per episode,
records exit codes and elapsed time, and stops on infrastructure or monitoring failures.
It checks each task against the declared scene split. Complete evidence retains all four
safe/unsafe and completed/unfinished categories; it does not silently drop inactive or
failed episodes. The initial Jar plan uses full benchmark horizons on four engineering
scenes and seed 0. It is a mechanism/throughput diagnostic, not a final performance estimate.

`experiments/scripts/maniguard_search.py` runs matched evolution, uniform random search
and episodic categorical REINFORCE from an explicitly supplied protocol. It reuses the
repository's tested `AdapterSearch` implementation without inheriting UNISafe budgets.
The protocol declares a finite domain, fixed adapter family and information access,
training scenes/seeds, population, generations and search seeds. All methods receive the
same initial population and candidate episode allocations. Search only accepts the frozen
training split. Hand-back searches cannot change the guard's trigger or policy horizon.

The shared terminal training objective is task completion minus twice the physical
violation indicator. All four physical outcome categories remain separately recorded.
This is a training preference, not a safety guarantee. REINFORCE learns a distribution
over the same adapter parameters; pretrained VLA weights remain frozen. Finalists require
separate validation before reserved testing. Optimizer states, source hashes, proposed
candidates, all episode costs and failures are saved. A failed stage stops the search;
its partial interactions must still enter the study accounting. Before any optimizer update,
the runner audits the saved generation plan against its proposed candidates, validates each
physical result and safety monitor, requires complete cost accounting, and checks that every
candidate received the same distinct scene/seed pairs. Fitness is derived from those physical
records, not the cached `evidence.jsonl` summaries. Every generation saves its full audit and
four outcome counts per candidate. No search budget has yet been selected or executed for
ManiGuard.

The optional `--parallel-clients` policy-server flag keeps a separate sampling stream
for each episode identifier. Batch `--workers 2` requires that server metadata explicitly
advertise this isolation; the default remains one worker. A failed job stops new launches
while already running jobs finish and retain their process records and evidence. Batch
source hashes and concurrency are saved. CPU tests cover interleaved sampling and failure
draining. Both engineering smoke episodes have passed, and two full Stack episodes were observed
advancing concurrently. Initialization overhead can be substantial; no matched throughput
improvement has yet been established. See the study log and concurrency evidence.

The idle-server replay checker accepts `--interleaved` to compare two interleaved clients
with a sequential reference on an identical saved observation. Every extra model query
is recorded. This verifies sampling isolation, not deterministic physical rendering.

Search protocols may declare `workers: 2` after concurrency validation. The same worker
count applies to every optimizer, and each batch verifies server isolation before starting
physics. Omitting the field retains serial evaluation. Realized model latency may include
queueing behind another client and must be reported as measured.

`experiments/scripts/maniguard_batch_report.py` audits explicitly supplied batch directories
against their saved plans. It retains not-started jobs, missing process records, failures
and invalid results; only validated completed episodes enter the four outcome counts.
Recorded costs include incomplete and failed jobs, with missing fields and unverified
physics attachment identified. Episode elapsed-time sums are not batch wall time. These
batch reports exclude standalone reference episodes and inference-only diagnostics unless
those costs are added separately. A missing process record is not evidence that a job is live.
Output files are created exclusively to preserve earlier snapshots:

```bash
data/maniguard/envs/sim/bin/python experiments/scripts/maniguard_batch_report.py \
  --batch runs/maniguard/stack-fixed-screens-v1 \
  --output runs/maniguard/reports/stack-progress.json
```

For costs across batches, standalone physical references and inference diagnostics, use:

```bash
data/maniguard/envs/sim/bin/python experiments/scripts/maniguard_accounting.py \
  --runs runs/maniguard \
  --output runs/maniguard/reports/accounting-snapshot.json
```

This inventories raw episode ledgers and inference check records, including failed and
partial work. It hashes the records read and excludes derived reports and queue durations
to avoid counting the same interaction twice. Missing ledgers, missing fields and
unverified physics attachment remain visible. A running snapshot is not atomic and does
not verify live handles or physical outcomes. Recorded costs can still omit historical
work: see `ManiGuard fruits accounting.json` in the research notes for the first native
startup failure and the earlier failed query whose latency was not recorded. Elapsed
time sums are not total study wall time.

A completed batch can be exported as an audited four-outcome/query-cost figure:

```bash
data/maniguard/envs/sim/bin/python experiments/scripts/maniguard_outcome_plot.py \
  --batch runs/maniguard/stack-fixed-screens-v1 \
  --figure runs/maniguard/reports/stack-outcomes.png \
  --report runs/maniguard/reports/stack-complete.json
```

The plot requires every planned episode to pass validation. It reports counts and
realized query totals, including early completions; it does not establish equal query
budgets or an adapter effect. Choose new output paths to preserve previous artifacts.

For completed Jar identity/guard/hand-back episodes from one scene and root seed, export
prerequisite, held-target and intervention timelines:

```bash
data/maniguard/envs/sim/bin/python experiments/scripts/maniguard_guard_timeline.py \
  --episode runs/maniguard/jar-fixed-screens-v1/episode-0014 \
  --episode runs/maniguard/jar-fixed-screens-v1/episode-0008 \
  --episode runs/maniguard/jar-fixed-screens-v1/episode-0029 \
  --figure runs/maniguard/reports/jar-guard-timeline.png \
  --report runs/maniguard/reports/jar-guard-timeline.json
```

The exporter validates full-episode outcomes, checks block/hold counts against the
recorded ledger and diagnosis, and preserves trace/script hashes. Same-scene starts are
not deterministic counterfactuals. Timeline associations alone do not prove that a guard
caused or prevented a later failure. Choose new output paths for each export.

The follow-up `lid-guard-screen.json` compares unchanged control with the fixed privileged
prerequisite guard across all four engineering Lid scenes and root seeds one and two.
It was declared after an extra baseline first grasped the container and lifted it before
lid placement. The sixteen full-horizon trials retain all scenes, including those without
an observed premature transition. It is a fixed-rule screen, not an optimizer comparison.
The live queue waits for the existing Jar repeats; process handles and immutable snapshots
are recorded in `ManiGuard fruits screen progress.json` under the research notes.

After the pinned Lid server and its fresh reference/interleaved replay are validated, the
batch command is:

```bash
data/maniguard/envs/sim/bin/python experiments/scripts/maniguard_batch.py \
  --plan configs/maniguard/lid-guard-screen.json \
  --output runs/maniguard/lid-guard-screen-v1 --workers 2
```

The active managed queue owns this output path. A separate reproduction needs a new path
and an idle matching policy server; do not launch a duplicate alongside the live queues.

`maniguard_compare.py` compares every condition with a declared control within the same
scene and root seed. It preserves unplanned, unfinished and failed counterparts, and uses
only validated complete pairs for outcome transitions. Equal safe-completion totals can
contain both gains and losses across scenes; all four outcome categories remain explicit.
The default control label is `identity`; the initial Jar screen uses `baseline_repeat`:

```bash
data/maniguard/envs/sim/bin/python experiments/scripts/maniguard_compare.py \
  --batch runs/maniguard/jar-fixed-screens-v1 --control baseline_repeat \
  --output runs/maniguard/reports/jar-condition-comparison.json
```

Use a new output path for each snapshot. The report records source/plan hashes. Shared
scene and seed do not guarantee identical physical starts, and incomplete comparisons
cannot rank conditions fairly. Whole-batch costs count every recorded episode once;
comparison totals reuse controls and must not be summed as study costs. Missing pair
costs remain null rather than becoming zero.


`maniguard_motion_tracking.py` checks all validated completed identity/motion episodes in
a batch. It verifies action/observation ordering, adjacent measured states, exact agreement
between adapted and executed arm targets, the configured active cap, and commanded target
displacements. Measured joint motion is reported separately and may exceed a command cap.
The output retains incomplete batch coverage, source/plan/trace hashes and the timing of
measured excursions relative to the first violation. It does not infer a failure cause.

```bash
data/maniguard/envs/sim/bin/python experiments/scripts/maniguard_motion_tracking.py \
  --batch runs/maniguard/stack-motion-repeat-v1 \
  --output runs/maniguard/reports/stack-motion-tracking.json
```

Use a new output path for each audit. This command reads completed traces and makes no
simulator actions or policy queries.


For a completed release trial, the diagnosis command also audits the executed delay:

```bash
data/maniguard/envs/sim/bin/python experiments/scripts/maniguard_diagnose.py \
  runs/maniguard/stack-release-trigger-v1/episode-0000
```

The release audit checks exact adapted/executed action agreement, unchanged arm targets,
and at most the declared number of delayed actions at the start of each uninterrupted
run of opening requests. It counts target holding from the goal observation **before**
each delayed action; a missing observation remains unknown. This measures coverage on
the goal-checker target only. Holds on other objects and causal grasp preservation are
not established by those counts. Diagnosis makes no simulator actions or policy queries.
