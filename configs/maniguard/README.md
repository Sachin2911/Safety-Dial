# ManiGuard fruit experiments

This study implements the six hypotheses in [fruits.md](../../fruits.md). Physical
validation and optimizer comparisons are pending. Passing controller tests does not
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
its partial interactions must still enter the study accounting. No search budget has yet
been selected or executed for ManiGuard.
