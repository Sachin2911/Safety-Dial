# Shared protocol

Definitions and rules used by both [pushT.md](pushT.md) and [walker2d.md](walker2d.md).
Environment-specific details (the rule, timing, geometry) live in those files. The
interpretation rules in [researchDirection.md](../researchDirection.md) and
[AGENTS.md](../../AGENTS.md) apply throughout.

## Terms

- **Root:** a simulator state reached by a recorded procedure, together with the real
  observation history the model conditions on (three frames for every LeWM used here).
- **Tape:** an open-loop sequence of environment actions from a root, grouped into action
  blocks of `frameskip` actions. One block is one model step.
- **Branch:** one tape executed in the simulator from its root. Executing a branch costs
  simulator steps; imagining it costs only model compute.
- **Rule:** the supplied safety specification, evaluated on true simulator state at every
  environment step. A tape is **unsafe** (`U = 1`) if the rule is violated at any
  environment step within the horizon.
- **Clearance:** a signed distance to violating the rule, positive when safe. Units are
  environment specific: pixels for Push-T, m/s or a normalised health margin for Walker2d.

## The decision and the dial

A model accepts a tape when its predicted minimum clearance over the horizon is at least a
margin `m`, that is `A_m = 1` when `c_hat_min >= m`. The margin `m` is the Safety Dial.
Sweeping it traces a **dial curve** of false-safe acceptance against acceptance rate for one
fixed model, and better experience should move the whole curve down. The dial is an
evaluation instrument here, not the novelty claim.

## Metrics

- **False-safe acceptance:** `FSA(m) = #(A_m = 1 and U = 1) / #(A_m = 1)`. It is undefined,
  never zero, when nothing is accepted. Always report the accepted count, the total count
  and the acceptance rate `AR(m) = #(A_m = 1) / N` beside it.
- **False reject rate:** `#(A_m = 0 and U = 0) / #(U = 0)`, so that a model cannot look safe
  by rejecting everything.
- **Headline number:** FSA at matched acceptance. For each model, choose `m` on the
  development bank so that `AR` equals a fixed target, then apply that `m` unchanged on the
  test bank. The target is fixed once, before E2, as the development acceptance rate of the
  unadapted model at `m = 0`. Also report the area under the dial curve over a declared
  acceptance range.
- **Continuous errors:** pose or state error by horizon, and the signed clearance error
  `e = c_hat_min - c_min` (positive means optimistic) with its 90th, 95th and 99th
  percentiles. A low mean can hide a harmful optimistic tail.
- **Retention:** prediction error on held-out ordinary clips and, for Push-T, goal reaching
  on a fixed set of planning episodes without hazards.
- **Ordinary motion:** predicted against true displacement on ordinary tapes. A model that
  "improves" safety by predicting less movement must be caught here.

## The four-source decomposition

On identical tapes, compute the rule decision from four sources:

1. **Dense truth:** true state at every environment step.
2. **Endpoint truth:** true state at block endpoints only, interpolated between them.
3. **Real readout:** the frozen probe applied to encoded real frames at block endpoints.
4. **Imagined readout:** the same probe applied to the model's imagined latents.

A false-safe decision from source 4 is attributed to the first link in the chain that
already produces it: temporal sampling (2 against 1), readout (3 against 2) or imagination
(4 against 3). Report the attribution by horizon, by contact or near-failure regime, by
rotation or speed, and by distance to the boundary. Predictor repair is justified only
when imagination carries a material share.

## Splits and leakage

- Split whole source trajectories before drawing roots, tapes or branches. All branches of
  one root, and all roots of one source trajectory, stay in one split.
- Roles are disjoint: probe training, adaptation (seed data and acquired data),
  development (model and margin choices) and final test. Final-test outcomes never enter
  selection, calibration or training.
- Keep a **representative bank** of ordinary behaviour separate from a **stress bank** of
  contact, rotation or near-failure cases, and report each. A balanced stress bank does not
  estimate how often those cases occur in deployment.
- Relabelling one physical branch under several virtual hazards creates more labels, not
  more experience. Apply the same relabelling policy to every arm.

## Predictor-side adaptation

The single repair mechanism, fixed on development data before E3:

- Freeze the encoder and observation projector: gradients **and** running statistics, by
  keeping them in `eval()` after any parent `.train()` call. Freeze scalers and probes.
- Train the action encoder, predictor and prediction projector ("predictor-side").
  Updating the predictor alone is a different, separately named setting; choose one on
  development data before E3.
- Targets are cached, detached embeddings of the real next frames from the frozen encoder.
- Loss: latent prediction MSE, teacher-forced with the model's history length, optionally
  plus a short autoregressive rollout term. Choose on development data, then fix. SIGReg is
  omitted because frozen targets give it no gradient; say so in the thesis.
- Every batch mixes the same original-data replay subset with acquired data, initially
  half and half. Optimiser, learning rate, batch size and number of steps are identical
  across arms.
- Each data budget restarts from the same pretrained weights, so the choice of data is not
  confounded with accumulated optimisation. Pair training seeds across arms.

## Acquisition arms

All arms choose from the **same candidate pool** for each root, produced by one common
proposal generator. Choosing costs model compute; executing costs charged simulator steps.

1. **Random supported experience:** uniform choice from the pool, balanced across roots.
2. **Predicted boundary coverage:** candidates whose predicted minimum clearance lies in a
   band around the operating margin, balanced across roots and motion regimes.
3. **Learned optimistic-error risk (conditional on E2):** a small model on features known
   before execution (predicted clearance and when it occurs, predicted contact or health
   proxies, predicted displacement, action statistics). It is trained on branches already
   executed and retrained each round, and it never sees an unqueried actual future.
4. **Retrospective oracle (reference only):** executes the whole pool, then keeps the
   branches with the largest true optimistic error. It bounds how useful data can be; its
   simulator cost is not comparable with the other arms.

Acquisition runs in rounds. The model adapted in the previous round scores the next
round's candidates, the chosen batch is executed, and a fresh adaptation from the
pretrained weights uses all data acquired so far. Selection rules and band widths are
frozen after development.

## Budget accounting

Charge every simulator step: root generation, any replayed prefix, discarded or failed
branches, and every executed branch. Compare arms at equal **charged simulator steps** and
report branch counts beside them. Equal branch counts mean equal cost only when prefix
lengths and horizons match. Report model queries and GPU training time separately.

## Statistics and reporting

- Compare arms on the same test tapes (paired). Build 95% intervals with a bootstrap that
  resamples roots, not tapes.
- Use three acquisition seeds per arm for any claim about acquisition. Add independent
  training seeds before claiming robustness to training randomness.
- Report counts beside every rate. Report all-infeasible fallbacks, arena exits and
  observation-domain exits as bad outcomes, and censored futures as censored, never as
  safe.
- Provisional decision thresholds, to be confirmed on development data before use:
  - E2 proceeds if FSA at matched acceptance falls by at least 25% relative to no update,
    beats the readout-only correction, and retention stays within noise.
  - H3 is supported if the paired FSA difference excludes zero at two or more budgets,
    consistently across acquisition seeds.
