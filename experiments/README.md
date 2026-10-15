# Current experiment workspace

New implementation follows [oracle.md](../oracle.md). The manipulation environment,
model and adaptation method are still being selected.

Add reusable Python under `helpers/`, entry points under `scripts/`, tests under `tests/`
and study configs under `../configs/`. There is no `src/` package.

Earlier notebooks, helpers, runners and tests are preserved in
[archive/legacy/experiments](../archive/legacy/experiments/README.md). Start from the
[archive index](../archive/README.md) for the useful results and reproduction guidance.
Run archived commands through `bash archive/run.sh` from the repository root so their
working directory and shared environment remain consistent.

## UNISafe engineering checks

Use the current Vast environment, `/venv/main/bin/python`. First prepare the platform
with `bash scripts/vast/setup.sh` if it is not already installed. Then:

```bash
# Requires access to the private checkpoint mirror via HF_TOKEN or a saved HF login.
# HF_TOKEN may also be stored in the ignored repository .env.
bash scripts/setup_unisafe.sh

# Run frozen recorded-data inference, followed by the environment acceptance test.
bash scripts/smoke_unisafe.sh

# Explicit destination and seed; the destination must not exist.
bash scripts/smoke_unisafe.sh runs/unisafe/my-check 0
```

Each stage can also run separately:

```bash
/venv/main/bin/python experiments/scripts/unisafe_inference.py --seed 0
/venv/main/bin/python experiments/scripts/unisafe_environment.py --seed 0
```

Both accept `--output NEW_DIRECTORY`; inference also accepts `--filter-step` (default
200000) and `--compare-to PRIOR_INFERENCE_DIRECTORY`. The comparison requires bitwise
identical saved arrays and exact numerical traces, excluding timing, and is intended
for repeat runs with the same seed and platform. Outputs default to dated directories in ignored `runs/unisafe/`. The combined
runner saves logs and stops on the first failing stage. A successful stage writes
`summary.json`; failures return a nonzero process status. Environment assertion failures
also write `error.txt`. Run the two stages sequentially for uncontended timing.

### What these checks establish

The inference test strictly loads the frozen world model, task actor, uncertainty ensemble,
and safety actor plus two critics. It processes the three released failure recordings,
223 frames in total, without starting Isaac Sim. It saves:

- Per-frame failure predictions, uncertainty, hypothetical next-state safety values,
  reconstruction error, one-step image prediction error and timing in `trace.csv`.
- Latent features, proposed actions and recorded actions in `features_actions.npz`.
- Observed/reconstructed/predicted image panels and episode plots.
- Checkpoint hashes, source revisions, query counts, peak inference CUDA memory and elapsed time.

Recorded `action[t]` led to `observation[t]`. Therefore outgoing one-step predictions use
`action[t+1]`. Latent sampling uses modes (`sample=False`) for repeatable diagnostics;
this is not a replication of the evaluator's stochastic closed-loop rollout. The saved
training uncertainty threshold is plotted for context, not selected as a validated online
intervention threshold. No proposed actions are executed in this test.

The environment test runs one original normal block-plucking task on the migrated stack.
It checks seeded reset state, 128-by-128 RGB images from both cameras, model-facing wxyz
quaternions, relative IK arm movement, gripper closing and reopening, and the original
outcome functions. It deliberately injects middle-block drop, top-block drop, top-block
lateral displacement, success and timeout fixtures. It verifies both immediate predicates
and the termination/truncation flags after a real environment step. Those fixtures are
engineering tests, not autonomous successes or failure-rate samples.

Its output includes a camera image, an animated GIF, native reset state (`xyzw` quaternions),
complete resolved configuration,
per-step JSON records of normalized and executed actions, terminal observations, outcome
flags, and counts of environment steps, physics steps and resets. Simulator reset images
are explicitly refreshed without extra physics integration. Each run also saves a small
`implementation/` snapshot of our scripts, helpers, dependency pins and source manifest.

### Compatibility and provenance

[The compatibility helper](helpers/unisafe_source.py) requires the clean upstream commit
`ab0fdf87f38edefa48b54a291ec5b7156c17b6e0` and creates `data/unisafe/runtime/`.
It does not edit `third_party/UNISafe`. `compatibility.json` records the original and
adapted hash of every copied Python file. Only `Isaac-Takeoff-Franka-IK-Rel-v0` is registered
for this pilot; other upstream tasks and the original evaluation/training entry points
have not been ported.

The adapter updates import paths, PhysX configuration and tensor access; handles slice
indices in reset events; uses the legacy Franka asset; converts explicit configuration
quaternions from wxyz to xyzw; and converts observations back to wxyz for the checkpoint.
The original camera transforms, block dimensions and masses, physics timestep (0.01 s),
control decimation (5), 15 s episode limit and task outcome predicates are retained.
`sim.use_fabric=False` selects the supported tensor-to-renderer transform path because
native PhysX Fabric retained the robot's startup pose after reset on this stack. Camera
capture is also refreshed at reset without physics integration. The policy uses RGB and
end-effector state, not camera pose metadata. Native camera pose metadata is not validated
on this path; later geometric perception work must check it separately. No platform
package is patched.

Actions follow the original wrappers: `clip(normalized * 0.15, -0.15, 0.15)`, followed by
the IK controller's scale of 0.5 for the six arm coordinates. A negative gripper coordinate
closes the fingers; nonnegative opens them. The source's duplicate reset-event name is
preserved: its second definition randomizes cubes 2 and 3 together; the overwritten cube-1
randomizer is not silently restored.

For model-only use, the helper defers a simulator-only import. It removes compiled
`._orig_mod` key components and fixes the saved camera-matching regex, then demands a
strict checkpoint match. The upstream critics share their preprocessing network; loading
also verifies that the two released snapshots agree on those shared weights.

The original environment targeted Isaac Sim 4.2. Running on Sim 6.2 / Lab 3 does not
establish identical physics, rendering, historical success rates or identical trajectories
across devices. The current acceptance run uses an RTX 3090. The autonomous baseline described below separately connects the controller and records
explicit filter settings and physical outcomes. The engineering checks alone establish no
avoidable-stall claim. Neither workflow trains the models.

All three bundled recordings are labelled failure examples. Their collection code can
relabel preceding transitions and substitute previous observations at termination. Their
labels alone therefore cannot establish physical failure onset, warning lead time, false
alarm rates, successful-episode behaviour or generalization beyond the training data.

### Validated run on 15 October 2026

The combined runner completed successfully at `runs/unisafe/validated-20261015/` on the
RTX 3090. Frozen inference processed 223 frames; median per-frame diagnostic time was
about 15 ms and peak allocated CUDA memory was 0.79 GiB. A second seed-0 process matched
all saved feature/action arrays bitwise and all numerical trace fields exactly, excluding
timing (`runs/unisafe/repeat-20261015/`). Timing includes reconstruction,
one-step prediction and selected image saving; it is not a closed-loop controller benchmark.

The environment acceptance run completed 60 control steps and 300 physics steps, with
seven explicit resets and five automatic resets. Reset states matched exactly at seed 0.
Both reset images showed the block scene. Five upward arm commands moved the end effector
about 7.6 mm; finger positions closed from about 40 mm to 0.11 mm and reopened to 40 mm.
All five injected outcome checks passed. These counts describe the final acceptance run,
not the earlier migration/debugging attempts, which remain in separate local run folders.

### Dependency maintenance

[configs/unisafe/requirements-lock.txt](../configs/unisafe/requirements-lock.txt) is an
additive lock constrained by the current Vast platform. Do not install upstream's old
requirements or sync the legacy root `uv.lock` into this environment. Regenerate only
when deliberately changing these dependencies:

```bash
uv --no-config pip compile configs/unisafe/requirements.txt \
  --python /venv/main/bin/python \
  --constraint configs/vast/requirements-linux-py312.txt \
  --overrides configs/vast/uv-overrides.txt \
  --torch-backend cu130 --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match \
  --output-file configs/unisafe/requirements-lock.txt
bash scripts/vast/setup.sh --check
```

## UNISafe autonomous baseline and diagnosis

Run the frozen controller and analysis together:

```bash
bash scripts/baseline_unisafe.sh runs/unisafe/my-baseline

# Small integration pilot; do not pool its episodes into the primary estimate.
bash scripts/baseline_unisafe.sh runs/unisafe/my-pilot --seeds 0 1
```

The default [configuration](../configs/unisafe/baseline.json) runs 20 seeds with and
without filtering, alternating order. Both arms reset to the same physical state, checked
to within 1e-6, and use the same policy seed. The saved reset-image differences matter:
matching physical tensors does not establish identical rendered inputs, contact solver
state or exact counterfactual branching. The unfiltered arm computes a shadow filter to
match the model-query and random-draw schedule while executing its task proposal.

The filter uses snapshot 200000, uncertainty threshold 4.3 from the saved training config,
and reachability threshold 0.1 and action denominator 1 from the released function defaults.
The release does not establish the missing evaluation settings, so these values define
this migrated pilot. They are not a recovered paper evaluation configuration.

The released action postprocessor binarizes the last of six safety-action coordinates.
That coordinate controls arm rotation, while the task's actual gripper is coordinate seven.
`--backup-postprocess released` preserves this behavior for the primary baseline.
`--backup-postprocess arm_clip` removes that binarization as a separately labelled diagnostic.
It changes the controller and must not be pooled with the primary results.

The task and filter retain separate recurrent states: the task posterior is sampled, the
filter posterior uses its mode, and hypothetical proposal dynamics are sampled. Both tracks
receive the normalized action actually executed after clipping. The filter preserves the
task's gripper proposal when replacing the six arm coordinates. Ground-truth object states
and outcome flags are logged for evaluation; the learned encoder uses RGB and end-effector
state. No weights are trained or adapted.

Each episode saves `steps.jsonl` with proposed, backup, selected and executed actions,
intervention triggers, predicted safety, physical geometry and timing. `observations.npz`
contains both cameras and end-effector state, including the true terminal observation.
`rollout.gif` provides a visual check. A cloned terminal snapshot prevents the simulator's
automatic reset from overwriting terminal camera/state buffers. The engineering acceptance
test now also exercises this buffer ownership and verifies terminal injected geometry.

`summary.json` records outcomes, model provenance, settings, interaction counts, resets,
render calls, model queries and elapsed time. `analysis/` contains episode metrics, paired
outcomes, uncertainty intervals and plots for unfinished episodes. Its low-progress flag
means the middle block moved less than 2 mm over the last three seconds. It describes a
trace; it does not diagnose a restrictive filter or establish avoidability.

For the bounded follow-up protocol, after completing the primary run:

```bash
/venv/main/bin/python experiments/scripts/unisafe_followups.py \
    --baseline runs/unisafe/my-baseline --output runs/unisafe/my-followups
```

[The diagnostic plan](../configs/unisafe/diagnostics.json) records all selected seeds and
variants: extend the horizon to 30 seconds; change future stochastic draws after step 300
without resetting physical or recurrent state; remove the sixth-coordinate binarization;
and try different stochastic draws from the initial state. This plan was chosen after
inspecting the initial pilot and is exploratory. The RNG change affects both sampled task
posteriors and imagined filter predictions, not just task behavior.

After the follow-ups and a fresh environment acceptance run, audit the evidence together:

```bash
/venv/main/bin/python experiments/scripts/unisafe_diagnose.py \
    --baseline runs/unisafe/my-baseline --followups runs/unisafe/my-followups \
    --pilot runs/unisafe/my-pilot --environment-check runs/unisafe/my-acceptance
```

This compares the full recorded 300-step prefixes for the continuation probes and accounts
for every completed autonomous diagnostic, including unsuccessful alternatives. Even a
matching observed prefix is not validation of simulator snapshot restoration. A successful
alternative from an initial state is also weaker evidence than recovering from a validated
stalled state. Failed bounded probes do not prove that recovery is impossible.

The baseline-stage results and remaining research questions are recorded in
[the pilot report](../docs/research/reports/UNISafe%20baseline%20pilot.md).
