# Research starting point

**Discussion notes, 15 October 2026.**

This records a practical starting recommendation for the direction in [oracle.md](oracle.md),
which remains the authority for research decisions. Task continuation under learned safety
filtering is the chosen conditional focus. Active perception, safety-model training-data
selection and temporal safety monitors remain adjacent possibilities. The recommendations
below do not fix an environment, adaptation method or experiment budget.

Sachin plans to rent a Vast.ai GPU, with a 48 GB RTX PRO listing selected as the capacity
target. Confirm the full GPU model and host driver on the rented machine. Actual runtime,
memory use and simulator compatibility still need measurement. Template creation and package
resolution have been checked, but no new simulator execution, model inference or training
was performed to establish these recommendations.

## Selected platform and Vast template

On 15 October 2026, Sachin chose to base the engineering setup on Python 3.12 and the
current Isaac Lab 3.0 development line, then adapt the other components around that stack.
This platform choice does not establish UNISafe compatibility or fix the research task.

A private template named **evo safety filter** was created and read back successfully in
the Vast account. Sachin subsequently chose to defer research package installation until
after connecting, so startup now runs only the base image's normal entrypoint. Select the
template under [My Templates](https://cloud.vast.ai/templates/).

- Template ID: `766408`.
- Template hash after removing boot-time installation: `73fcff5e93110dbb229ec73b7e43421c`.
- Base image: `vastai/base-image`, tag
  `cuda-13.0.3-cudnn-devel-ubuntu24.04-py312-2026-09-07`.
- Operating system and Python: **Ubuntu 24.04, Python 3.12**.
- CUDA toolkit in the base image: **13.0.3**.
- Access and storage: **JupyterLab and SSH with direct connections; 200 GB disk default**.

The separate manual setup script installs the following pinned software stack using `uv`:

- PyTorch: **2.11.0+cu130**; torchvision: **0.26.0+cu130**;
  torchaudio: **2.11.0+cu130**.
- PyTorch wheel runtime: **CUDA 13.0**, selected with `cu130`.
- Isaac Sim: **6.2.0.0**.
- Isaac Lab: pinned 3.0 source revision
  `6bfde876089661e116dc5a5d9c6805f18127fe32`.

The published Isaac Lab `3.0.0rc1` wheel inspected during setup still requests Isaac Sim
6.1, while the current 3.0 branch targets Sim 6.2. The manual setup therefore builds Lab's
aggregate package from the pinned source revision and uses its matching dependency
overrides. This is a development release line, not a general-availability release. PyTorch
2.11.0 is selected to match this Lab stack. Neither package installation nor upgrades run
automatically at instance boot.

Offer filters request one GPU with at least 24 GB VRAM, 32 GB RAM, eight effective CPU
cores, x86-64, CUDA 13.0 driver capability, driver 580.95.05 or newer, at least 99%
reliability, 200 GB available disk and two direct ports. These are selection filters,
not proof that Isaac Sim's renderer works on a host. The planned 48 GB GPU provides
headroom beyond the VRAM filter. Rental, storage and bandwidth costs depend on the offer.
Creating this template did not rent an instance.

The configuration, package snapshot, source revision and creation record are saved under
[configs/vast/](configs/vast/README.md). The [manual setup script](scripts/vast/setup.sh)
reads the resolved requirements and matching overrides from the repository, rather than
embedding them in the template. [create_template.py](scripts/vast/create_template.py)
supports creation and an explicit `--update` mode for the existing template.

### Connect first, then install manually

First boot initializes Vast's normal access services. There is no research installer or
research boot hook. The base image still needs to be pulled, but downloading and installing
PyTorch, Lab, Sim and their dependencies happens only when the manual script is invoked.
Changing the template affects future rentals; it does not change an existing instance.

After connecting, clone the repository and run the setup as root on the Vast instance:

```bash
cd /workspace
git clone https://github.com/Sachin2911/Safety-Dial.git
cd Safety-Dial
bash scripts/vast/setup.sh
```

If the clone already exists, enter its directory and run the last command. The script
installs system libraries and uses `uv` with the pinned package snapshot in the base
image's existing `/venv/main` environment. Downloads and building Lab can take substantial
time during this explicit setup step. No research datasets, checkpoints or experiments
are downloaded or launched by it.

Use these commands on the rented instance to inspect the manual installation:

```bash
tail -f /workspace/.evo-safety-filter/setup.log
cat /workspace/.evo-safety-filter/versions.json
```

After setup succeeds, choose **Evo safety filter (Python 3.12)** in Jupyter. For SSH commands, use
`/venv/main/bin/python` explicitly. A `ready` marker is written under
`/workspace/.evo-safety-filter/` only after the version checks and a CUDA tensor operation
succeed. Reruns validate all package pins, source provenance and CUDA before skipping installation. Failures leave the
marker absent; inspect the log and rerun the setup command to retry. No retry runs at boot.
The template retains `OMNI_KIT_ACCEPT_EULA=yes` for unattended Isaac Sim use after setup
and includes no credentials.

Package resolution, template privacy, access settings and disk allocation were verified
during creation. Installation, GPU execution, PhysX, RTX rendering,
cameras and UNISafe have not yet been tested on a rented host. The immediate engineering
task is to connect, run the manual setup and then verify a small simulator and camera smoke test.

Safety-Dial's root Python pin is now 3.12, and root setup dispatches to the platform
installer on Vast. The root project lockfile still describes earlier research dependencies;
do not sync it into this platform before research dependency integration.
Maintain one project environment during integration; the base platform installation is
not evidence that the existing project lock or original UNISafe requirements are compatible.
After platform verification, continue with the recorded-data and manipulation pilot below.

## Engineering progress on the rented host

The initial platform smoke tests now run on an RTX 3090 with 24 GB VRAM. The pretrained
UNISafe files have been verified and mirrored privately on Hugging Face at a pinned
revision. Frozen-model diagnostics and the migrated block-plucking environment acceptance
tests are implemented; commands, outputs and migration details are in
[the experiment guide](experiments/README.md#unisafe-engineering-checks).

These checks cover inference on the three bundled failure recordings and simulator
reset/camera/control/outcome wiring. Injected simulator outcomes test the task logic;
they are not autonomous policy results. The earlier platform-planning text above records
the setup decision before execution.

A migrated autonomous baseline now covers 20 seeds with and without the frozen safety
filter. Each arm completed 2/20 tasks, failed physically on 2/20, and timed out on 16/20.
The [baseline pilot report](docs/research/reports/UNISafe%20baseline%20pilot.md) records the
controller contract, continuation diagnostics, interaction costs and limitations.
Matching initial physical states did not produce identical reset images between paired
arms, and task proposals diverged before filtering. The many unfinished episodes warrant
investigation, but this pilot does not yet isolate avoidable filter-induced stalls.
No evolutionary target or training study has been selected.

## Where to begin

Begin with a small UNISafe block-plucking pilot, using its pretrained components to
understand one concrete failure. The immediate low-hanging fruit is reliable episode-level
evidence that helps decide which research question deserves an evolutionary experiment.

There are two useful entry points: an offline investigation of recorded trajectories, and
closed-loop diagnosis of task continuation. They can share observation processing, model
inference and analysis tools. The first can begin before the simulator migration is complete;
the second is necessary to establish the central continuation problem.

The first useful milestone is:

> We can run and explain this filtered manipulation system, and we have identified one
> measurable weakness with room for improvement.

## Environment shortlist

### UNISafe block plucking: first candidate

UNISafe supplies a manipulation task, Dreamer task policy, world model, learned safety filter
and successful/failed trajectory data. The release includes autonomous policy evaluation.
This reduces the number of research components that would need to be built before studying
their interaction. See the [official release](https://github.com/CMU-IntentLab/UNISafe/tree/isaaclab)
and the repository's [asset and integration audit](docs/research/research_notes/Safe%20manipulation%20progress%20and%20robustness/feasibility_and_evolutionary_scope.md).

The main uncertainty is reproduction and compatibility. The release describes its Isaac
Sim 4.2 stack as stale, and the existing [compute and compatibility audit](docs/research/reports/UNISafe%20inference%20and%20training%20compute.md)
identifies migration work for Blackwell GPUs. First test recorded-data inference, then establish
whether faithful closed-loop execution is manageable. Loading weights alone does not verify
the policy, action transformations, camera appearance or physical behaviour.

Use UNISafe as the initial candidate. If faithful reproduction becomes a substantial
simulator-migration project, reassess its cost before investing further.

### robosuite with a Panda arm: fallback

Investigate **Wipe** for contact-history monitoring, or **Lift/Stack** for grasping and object
stability. robosuite exposes configurable cameras and force-torque sensing, which could
support temporal prediction and controlled observation experiments. See the official
[task descriptions](https://robosuite.ai/docs/modules/environments.html) and
[sensor documentation](https://robosuite.ai/docs/modules/sensors.html).

This trades some simulator integration work for research-component work. Appropriate safety
definitions, a learned representation or world model, and a safety filter would still need
to be established. Existing [robomimic checkpoints](https://robomimic.github.io/docs/model_zoo/robomimic_v0.1.html)
have explicit version requirements and are not automatically compatible with the latest
environment. A basic lifting task is useful for checking a pipeline, but its relevance to
safety filtering must be demonstrated.

Keep robosuite as the fallback while assessing UNISafe's actual integration cost.

### ManiSkill: option if repeated simulation becomes the bottleneck

ManiSkill offers tabletop tasks and demonstrations, including **PickCube**, **StackCube**
and **PegInsertionSide**. PickCube is useful for checking a manipulation pipeline; insertion
could support a later contact-rich study. See the
[official task catalogue](https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html).

Consider ManiSkill if its simulation and evaluation capabilities fit the selected experiment.
The task catalogue does not provide the matching learned world model and safety filter needed
for this research. Their availability and integration cost would require a separate check.
No throughput advantage on the available machine has been measured here.

Start with one platform. The shortlist is a set of alternatives, not a recommendation to
build three experiment stacks.

## First investigation

### 1. Inspect recorded trajectories and run the frozen model

Establish:

- What successful and failed episodes actually look like.
- Which observations and action histories are available.
- What the failure labels mean and when they become positive.
- Whether predictions provide useful warning before a physical failure.
- What inference and feature extraction cost on the available machine.

An important annotation issue is already visible in the source: UNISafe's collector marks
several preceding transitions as failures and replaces terminal observations with previous
observations. Distinguish annotation timing from the physical event before evaluating early
warning. See the audited [collection source](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamerv3_torch/tools.py#L335).

The first useful output is a few understandable episode traces showing observations,
recorded actions, predictions and outcome labels. Inspect actual fields before assuming
that force, object motion or intervention histories are available. Recorded demonstrations
alone cannot establish how an autonomous policy behaves after safety intervention.

### 2. Establish the filtered controller's behaviour in simulation

Run the unchanged policy with and without filtering from comparable starts. Initially
retaining the pretrained model and filter would help attribute behaviour to their existing
interaction; this is a pilot recommendation rather than a permanent restriction.

Record proposed, filtered and executed actions, intervention reasons, task progress and
physical outcomes. Preserve the coordinates and transformations associated with each action.
Physical dynamics histories need executed actions; a task learner can intentionally treat
its action as a proposal to the filtered environment.

Inspect unfinished episodes:

- Does the policy repeatedly propose an action that the filter rejects?
- Does it resume progress, but slowly?
- Is the safety prediction inaccurate or unnecessarily restrictive?
- Is relevant information missing from the observations or controller history?
- Does a longer allowance permit completion?
- Can an alternative behaviour complete the task under the same safety constraints?

An unfinished episode alone does not diagnose poor continuation. A successful alternative
provides evidence of avoidability; unsuccessful finite search does not prove impossibility.
Matched initial conditions help comparison, but intervention can change subsequent states.
Do not assume exact mid-episode branching until simulator and controller restoration have
been validated.

### 3. Select the smallest evolutionary target supported by the evidence

The adjacent directions have a provisional ordering by initial experiment burden. The
diagnosed weakness should determine which one to pursue. If the baseline reveals substantial
avoidable stalls, compact task adaptation may remain the strongest first study.

**Safety-model training-data selection** is the easiest evolutionary pilot once labels and
representations are trustworthy. Freeze the representation and train a small safety
predictor. Start by evolving a few sampling proportions over meaningful training groups,
with random and balanced sampling as comparisons. Match labelled-data and training budgets;
uncertainty-based selection is another relevant comparison. Count candidate training costs.

Keep trajectory-level training, selection-validation and final test splits separate. Audit
overlap with the pretrained model's training data. Independently collected holdouts may be
needed for credible generalization claims. A held-out predictor result is preliminary
evidence; closed-loop safety and completion still require evaluation.

**Temporal safety monitors** become attractive if observation history adds useful predictive
information. First compare a single-step predictor with a simple window-based predictor.
If history helps, genetic programming has a concrete target: compact rules providing useful
warning with tolerable false alarms. Use causal histories and assess missed failures, false
alarms and warning time on held-out episodes. Predictive warnings do not by themselves
establish a safe intervention strategy.

**Active perception** follows if the diagnosis identifies an information problem. Before
evolving inspection movements, show that a different view improves prediction or task
performance. A lower uncertainty score alone is insufficient. Check whether changed views
remain meaningful to the pretrained representation, and include sensing movements and time
in the evaluation budget.

## Evaluation principles

Report safe completion, physical violations and unfinished episodes together, with progress
and completion time where useful. Fewer interventions can reflect inactivity. A fixed
learned filter does not guarantee that adaptation leaves the violation rate unchanged.
Zero observed violations does not establish zero risk; retain counts and denominators.

Separate simulator interactions, model queries, training computation and elapsed time.
Include data acquisition, discarded candidates and adaptation experience. Match available
information and budgets across comparisons. Ordinary learning through the same filter is
essential when evaluating evolutionary task adaptation.

The initial aim is one explained weakness and one well-evaluated response. The environment,
search representation, evolutionary algorithm, ROSARL role and experiment budgets remain
open until the pilot gives a concrete reason to choose them.
