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
