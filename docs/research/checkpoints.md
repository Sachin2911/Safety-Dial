# Existing LeWM assets for targeted safety experience

**Audit snapshot: 21 September 2026.** No weights were loaded or downloaded, no model was trained, and no environment was changed. “Available upstream”, “implemented locally”, and “previously tested” are separate statuses below.

The [adopted plan](../researchDirection.md) controls scope; use the [pilot checklist](pilot.md) for execution. Recheck local availability on the machine used for a run.

## 1. What can we reuse immediately, and what is actually present?

### Takeaway

We can reuse the released Push-T encoder, predictor and goal-planning interface, together with the project's probe and constrained-CEM code. We do not need to train a world model from scratch. However, the current checkout has neither the checkpoint nor the expert data, and targeted collection/fine-tuning has not been implemented.

### Cited Findings

| Asset | Verified status and evidence |
|---|---|
| LeWM checkpoints | Official README lists Push-T, Cube, TwoRooms and Reacher. No compatible locomotion checkpoint is listed in that release. [Official repository](https://github.com/lucas-maes/le-wm) |
| Push-T weights | Public `quentinll/lewm-pusht`, `weights.pt` plus `config.json`; weight file is 72.3 MB. [Weight metadata](https://huggingface.co/quentinll/lewm-pusht/blob/main/weights.pt) |
| Cube weights | Separate model with 192-dimensional latents and a 25-dimensional action-block input. It is not the Push-T weights transferred to Cube. [Cube configuration](https://huggingface.co/quentinll/lewm-cube/blob/main/config.json) |
| Other published models | Reacher and TwoRooms each have public weights/config files, approximately 72.3 MB per repository. Neither is integrated into this project's downloader. [Reacher](https://huggingface.co/quentinll/lewm-reacher/tree/main), [TwoRooms](https://huggingface.co/quentinll/lewm-tworooms/tree/main), [local download config](../../configs/download/all.yaml) |
| Local binaries/data | Filesystem inspection found only `.gitkeep` in `data/` and `third_party/`. Expected `data/stablewm`, `data/processed`, `data/probes` and `third_party/le-wm` are absent. Expected paths are recorded in [safe_dial_pusht.py](../../experiments/scripts/safe_dial_pusht.py#L63). This does not establish whether assets still exist on Vast or elsewhere. |
| Installed implementation | `.venv` contains stable-worldmodel 0.1.1, stable-pretraining 0.1.8 and transformers 5.14.1, matching package metadata and [uv.lock](../../uv.lock#L4588). Package presence is not a fresh runtime validation. |
| Local download/conversion | Implemented for Push-T/Cube, including transformers-v5 key remapping and strict weight loading. No conversion was executed in this audit. [download_data.py](../../scripts/download_data.py#L97) |
| Existing diagnostic readout | A 192-256-128-2 MLP trained to predict **pusher x/y**, with caching support. No trained block-pose readout was found. [probes.py](../../experiments/helpers/probes.py#L26), [target selection](../../experiments/helpers/probes.py#L135) |
| Existing planners | Penalty and feasibility-first variants already wrap a base LeWM model, with common episode code and diagnostics. [safeCEM.py](../../experiments/helpers/safeCEM.py#L455) |

The public Push-T expert dataset is 13.1 GB compressed. The inspected model repository contains no normalisation-statistics file, and the dataset repository contains the compressed HDF5 rather than a small statistics artifact. [Dataset files](https://huggingface.co/datasets/quentinll/lewm-pusht/tree/main), [model files](https://huggingface.co/quentinll/lewm-pusht/tree/main)

### Inferences

The immediate setup is asset recovery plus a smoke test, not pretraining. Prefer recovering the previously used checkpoint, source revision, action scalers and a small replay subset from the existing experiment machine. Public weight restoration is small, but downloading the full expert archive merely to reconstruct scalers is avoidable if those artifacts can be recovered. The public Cube dataset is not a prerequisite for a Push-T pilot.

### Gaps

- No fresh checkpoint load or numerical comparison was performed. The current source tree and public release can differ from the historical run, so pin both before comparing results.
- No compatible released locomotion model was verified. “No model is listed in the checked official release” is narrower than asserting none exists anywhere.

## 2. What exactly does the checkpoint supply, and how can it be adapted?

### Takeaway

The checkpoint has separable visual and dynamics components. A bounded experiment can keep the visual representation and diagnostic readout fixed, then ask whether targeted new transitions improve the dynamics predictor more than the same amount of ordinary data.

### Cited Findings

- The Push-T configuration uses a tiny ViT, 224-pixel images and patch size 14; a 192-dimensional predictor with three-frame capacity, six layers and 16 heads; a 10-dimensional action input; and separate observation/projector and prediction/projector modules. [Published configuration](https://huggingface.co/quentinll/lewm-pusht/blob/main/config.json)
- The installed model exposes `encoder`, `projector`, `action_encoder`, `predictor` and `pred_proj`. `encode` takes the visual CLS token through the observation projector. `predict` applies predictor plus prediction projector. [Installed LeWM](../../.venv/lib/python3.11/site-packages/stable_worldmodel/wm/lewm/lewm.py#L19)
- Upstream training uses frameskip five. The local policy unpacks a planned action block into successive environment actions, so this is **five two-dimensional commands**, not necessarily one repeated command. [Upstream training config](https://raw.githubusercontent.com/lucas-maes/le-wm/main/config/train/data/pusht.yaml), [installed action unpacking](../../.venv/lib/python3.11/site-packages/stable_worldmodel/policy.py#L420)
- The installed simulator runs at 10 control steps/second and 0.01-second physics substeps. Five environment actions span 0.5 seconds; the pilot's five-block horizon spans 2.5 seconds. Its receding horizon is one block, whereas the current upstream evaluation config specifies five. Preserve and report the actual pilot setting. [Installed environment](../../.venv/lib/python3.11/site-packages/stable_worldmodel/envs/pusht/env.py#L21), [local planner](../../experiments/helpers/safeCEM.py#L461), [upstream evaluation config](https://raw.githubusercontent.com/lucas-maes/le-wm/main/config/eval/pusht.yaml)
- `AutoCostModel` returns an eval-mode module with a `get_cost` method. Goal cost is terminal squared latent distance; the model also exposes `predicted_emb`. This supports reusing the existing planner after adaptation. [Loader](../../.venv/lib/python3.11/site-packages/stable_worldmodel/policy.py#L519), [cost implementation](../../.venv/lib/python3.11/site-packages/stable_worldmodel/wm/lewm/lewm.py#L110)
- Existing scripts fit StandardScalers on expert action, proprio and state columns, with goal duplicates, and preprocess images consistently. These parameters must stay fixed across adaptation arms. [build_process](../../experiments/scripts/safe_dial_pusht.py#L83), [pixel processing](../../experiments/helpers/probes.py#L42)
- Upstream training optimises the whole model with prediction loss plus SIGReg. Its target embeddings are not detached in the published end-to-end objective. It is not already a predictor-only targeted-fine-tuning runner. [Official train.py](https://raw.githubusercontent.com/lucas-maes/le-wm/main/train.py)

### Inferences

Recommended minimal adaptation path:

1. Load one verified checkpoint and record a frozen baseline. Freeze the visual encoder **and observation projector**, including running BatchNorm statistics. Cache real observation embeddings in eval mode.
2. Fit diagnostic readouts on a separate training split of actual encoded observations, validate them, then freeze them for every model comparison. Reusing the existing pusher probe is possible only after recovering its weights and validation provenance.
3. Collect the same number of new simulator transitions under targeted versus ordinary selection. Mix each arm with the same original-data replay subset. Count all simulator interactions used for candidate screening, not just retained transitions.
4. Train against encoded **new real future observations**, not the old model's imagined outputs. Strict “predictor-only” updates `.predictor` while keeping action encoder and prediction projector fixed. A practical alternative updates all three dynamics-side modules; call that **predictor-side adaptation** and state the distinction.
5. Evaluate both models with identical frozen readouts, scalers, constraints and planners, including held-out ordinary task performance to detect forgetting.

Freeze parameters with `requires_grad_(False)` and keep frozen modules in eval mode after any parent `.train()` call. Cache/detach teacher embeddings explicitly; do not accidentally detach trainable action embeddings. In this frozen-feature experiment, SIGReg on the fixed target latents supplies no training gradient. It should not be presented as an active representation-learning contribution. Match one-step versus multi-step loss, optimizer steps and data budget across arms.

### Gaps

- Acquisition scoring, transition storage, replay mixing, predictor-side optimization and candidate-level audit logging require implementation. Existing CEM diagnostics mostly summarise costs, not a complete branch dataset.
- Historical probe splitting is random by frame; the new study needs trajectory/root-separated train, development and test partitions. [Current split](../../experiments/helpers/probes.py#L161)
- Three-frame capacity does not prove the pilot always supplied three real history frames. The new collector must explicitly preserve the chosen observation/action history and temporal alignment.

## 3. What must be validated before the experience experiment is trustworthy?

### Takeaway

The most material hidden dependency is Push-T branch replay. The saved seven-number state is sufficient for existing pose visualisation, but it is not a verified complete simulator snapshot. A block-safety study also needs a new readout and a precise geometric event definition.

### Cited Findings

- The state contains pusher position, block position, block angle and pusher velocity. It omits block linear/angular velocity and solver/contact state. [Installed observation code](../../.venv/lib/python3.11/site-packages/stable_worldmodel/envs/pusht/env.py#L360)
- `_set_state` restores the listed pose/agent velocity and advances one physics substep. It does not restore a full contact root. Existing animation code explicitly pins poses after this call to prevent visual drift. That validates rendering, not dynamical branch equivalence. [State setter](../../.venv/lib/python3.11/site-packages/stable_worldmodel/envs/pusht/env.py#L514), [animation renderer](../../experiments/scripts/make_animations.py#L70)
- The repository's explicit replay verification script concerns MuJoCo locomotion. No equivalent verified Push-T arbitrary-root restoration test was found. [verify_replay.py](../../experiments/scripts/verify_replay.py#L1)
- Existing violation code can select block x/y, but only checks whether its **centre** is inside an axis-aligned box. This does not test the T-shaped footprint intersecting a region. Existing angle errors are metrics, not an orientation probe. [Violation function](../../experiments/helpers/linProbeHelpers.py#L282), [angle metric](../../experiments/helpers/hazardSweep.py#L99)

### Inferences

For the first pilot, branch from a fresh seeded reset followed by an identical recorded action prefix, and test repeated-prefix determinism including contact cases. This avoids prematurely designing full Pymunk snapshot restoration. If arbitrary-state restoration is later needed, save additional physical state and validate repeated suffix rollouts, including simulator history effects. Never assume the locomotion replay result transfers.

For block safety, train a readout for `(x, y, sin(theta), cos(theta))` using available physical labels. First verify it on real observations. For a “block centre enters a keep-out region” experiment, the existing centre predicate is a valid declared specification. For “any part of the T touches a region”, implement geometry from the actual simulator shape vertices transformed by pose. Record truth at a declared frequency; control-step checks are not automatically continuous-time collision checks.

Keep the physics/environment unchanged when adding a virtual keep-out rule. Adding a physical obstacle changes transition dynamics and becomes a separate transfer experiment. Use the pusher action-space guard consistently across arms so arena escape cannot masquerade as successful block avoidance.

### Gaps

- No block-position/orientation probe, footprint-intersection evaluator or new branch replay test was executed in this audit.
- A fixed encoder may not retain enough block/contact information. If actual-observation readouts fail, predictor-only adaptation cannot be assumed to solve it; that is a go/no-go result, not grounds to quietly expand to full model training.
