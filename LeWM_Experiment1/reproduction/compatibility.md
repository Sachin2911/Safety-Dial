# Runtime decisions before scientific outcomes

Recorded on 4 November 2026, before any benchmark episode in this reproduction.

- Runtime is the published AP-capable image, but no AP model, retrieval or planner
  is used. The model is the original `jepa.JEPA` and `module.ARPredictor` from the
  pinned LeWM repository. Its HF ViT-tiny encoder is constructed directly with
  exactly the stable-pretraining factory's dimensions and defaults, including
  disabled pooling/mask token and positional interpolation. Strict checkpoint
  loading is required. This avoids installing training-only dependencies.
- Installed stable-worldmodel is 0.0.6. Its dataset evaluator is named
  `World.evaluate_from_dataset`, whereas the pinned LeWM example calls a newer
  `World.evaluate` interface. Use the installed native dataset evaluator, CEMSolver
  and WorldModelPolicy; save their source hashes with each run.
- Inspection found that 0.0.6's dataset slice excludes its end index, and the
  evaluator takes the final loaded frame as the goal. Passing 25 would therefore
  select a goal only 24 actions after the start. To test the paper's actual
  25-action offset, request a 26-frame chunk. Case records explicitly save both
  the intended offset 25 and the evaluator argument 26. Assert the loaded goal
  step is start + 25 before any rollout. This is a compatibility correction,
  not a claim of byte-for-byte execution of the released example.
- Keep the released sampling convention that excludes the last eligible row.
  Compute per-episode maxima with a vectorized reduction, retaining the same
  eligible rows, sorted draw and NumPy RNG. Save all case IDs before outcomes.
- Use the validated OSMesa renderer. The official example forces EGL. This
  deviation is disclosed; environment packages and image preprocessing remain
  pinned. No renderer is selected based on success scores.
- Disable MP4 generation to keep evidence within the transfer allowance. Preserve
  per-action observations/state summaries, raw termination/success labels,
  selected actions and representative images. Video writing does not affect
  the evaluator's stepping or success calculation.
- Success is the native evaluator's logical OR of termination over 50 actions.
  It continues stepping all cases through the budget even after first success,
  with autoreset disabled. Preserve and count that behavior.

Engineering verification must still establish strict checkpoint loading, finite
costs/actions, callable availability, exact goal indexing and agreement between
recorded termination flags and the native evaluator's success vector. Failures
remain recorded; this document does not claim those checks have already passed.

## Engineering correction, before Reacher scientific evaluation

Reacher engineering attempt 01 stopped after ten controller actions because the
native DMControl wrapper returns `info['success'] = NaN` as a placeholder. The
benchmark's termination flag is its actual success signal. Our strict JSON encoder
rejected that unused field during the first trace flush; a repeated flush also
prevented the summary from being saved. The original failed run and logs are
preserved, and its ten actions remain in the durable interaction ledger. Its raw
ten-step trace was lost before publication and is not claimed to be recoverable.

The logger now represents nonfinite telemetry explicitly as JSON objects such as
`{"nonfinite": "NaN"}` and saves accounting even if a trace flush fails. This does
not change actions, CEM, model weights, goals or benchmark labels. Nonfinite actions
and model costs still fail their numerical checks. Reacher's engineering check is
repeated within the original shared engineering allowance before any scientific run.

## Rendering correction, before Reacher scientific evaluation

The second Reacher engineering run completed, but its restored initial images had
normalized mean absolute pixel errors of 0.07772 and 0.07948. A zero-action check
on its first fixed seed-0 state confirmed exact qpos restoration and the expected
floor colors, while the checkerboard texture differed visibly from the dataset.
MuJoCo's [3.12 changelog](https://mujoco.readthedocs.io/en/stable/changelog.html#version-3-12-0-august-20-2026)
documents a breaking change to primitive texture coordinates and finite-plane
checkerboard placement.

Before any Reacher scientific outcome, select the last pre-change MuJoCo release,
3.11.0, with dm-control 1.0.43, whose published requirement is MuJoCo >=3.10.0.
Install only these two packages in a separate import overlay. Keep the main image,
Torch, checkpoint, native planner, OSMesa backend and fixed cases unchanged. Save
wheel hashes, actual versions and source paths. First compare the same state with
zero action steps, then repeat engineering within the original shared allowance.
This selects versions from a documented rendering change, not success scores.
The original training simulator version remains unestablished; a matching image
does not prove identical physics. Each later engineering gate records actual
simulator package versions and requires the scientific job to match them.

The first zero-action overlay check failed during environment construction because
dm-control 1.0.43 still expects `mjData.qM`, removed by MuJoCo 3.11. Its stated lower
bound did not guarantee compatibility with this newer release. No action or model
query occurred; the reserved engineering reset and failed run are retained. Use
dm-control 1.0.44 instead, released alongside MuJoCo 3.11 and declaring >=3.11.0.
This is the same dm-control version as the main image; only MuJoCo changes relative
to that image. Preserve the first overlay and install the corrected pair in a new
directory before repeating the fixed-state image check.

The corrected pair completed the same zero-action check: qpos error remained zero,
and normalized mean absolute pixel error fell to 0.0100945. Inspection confirmed
the checkerboard placement and arm geometry now match the recorded image. Residual
pixel differences remain and are disclosed. Freeze this pair for Reacher and Cube,
the MuJoCo environments, subject to each task's engineering checks. TwoRoom's
completed results remain unchanged; PushT uses the original image's 2D simulator.

Before PushT engineering, add observational counters around its native Pymunk
`space.step` calls, including the steps performed during reset/state restoration.
Keep these physics steps separate from the protocol's controller-action budget.
The wrapper forwards the original arguments and return value without changing
state or stepping behavior. OGBench already has equivalent `do_simulation` frame
counters. Completed TwoRoom and Reacher source snapshots are preserved.

## Process isolation after a zero-action Reacher stall

The Reacher seed-42 scientific panel completed and saved 32/50 successes. The next
panel reserved its original 50 episodes, saved cases/model/configuration, then
stalled during `World` construction before initial images, action steps or model
queries. The scientific action ledger remained 12,500, exactly the completed
TwoRoom panels plus the first Reacher panel. Read-only process inspection showed
the process waiting on a futex with an idle GPU; container restrictions prevented
a debugger stack trace, so the precise cause is unconfirmed. Stop and preserve
this run. Its completed result is retained regardless of its score.

Run subsequent fixed panels in separate processes to avoid reusing renderer
resources. This changes process lifetime, not cases, seeds, CEM or success scoring.
Before continuing, repeat the small engineering gate within the shared allowance.
Resume the already-reserved seed-43 panel only after checking its stopped-run
record, identical config/cases, absent action artifacts and full reconciliation
of the scientific action ledger to other summaries. Record that continuation in
the durable ledger without allocating or repeating episodes. Other panels retain
their original reservations and bounds. No completed panel is rerun.

## Cube physics accounting before scientific evaluation

Cube's first two engineering cases completed with matching restored images and
valid control-action accounting, including four internal reset action calls.
The generic `do_simulation` counter was not invoked: the installed OGBench base
class calls `mujoco.mj_step(..., nstep=self._n_steps)` directly. Add a passive
counter around that native function, forwarding arguments and return values
unchanged, and restore the original function after closing the world. Repeat the
two engineering cases within the shared allowance before Cube scientific panels.
The first engineering run's absent low-level counters are not zero physics work.
Cases, weights, planner settings and benchmark labels remain unchanged.
