# Main track: Push-T with the released LeWM

Protocol for E0 to E5. Shared definitions are in [protocol.md](protocol.md); setup, storage
and file layout are in [infrastructure.md](infrastructure.md).

## Pinned facts

Checked on 26 September 2026 against the locked `stable-worldmodel` 0.1.1 install and the
[checkpoint audit](../research/checkpoints.md).

- **Model:** public `quentinll/lewm-pusht`. ViT-tiny encoder (patch 14, 224 px input),
  192-d latents, a predictor with a 3-frame history, and the modules `encoder`,
  `projector`, `action_encoder`, `predictor` and `pred_proj`. Rollouts are autoregressive
  and detach the initial embedding
  ([installed LeWM](../../.venv/lib/python3.11/site-packages/stable_worldmodel/wm/lewm/lewm.py)).
- **Timing:** 10 control steps per second, each made of 10 physics substeps of 0.01 s. An
  action block is five sequential 2-d actions (0.5 s); the five-block horizon is 2.5 s.
- **Actions:** relative. The pusher is a kinematic body driven by a PD controller toward
  `position + 100 * action`, and it passes through walls. The block is dynamic and walled in
  by segments from (5, 5) to (506, 506).
- **State:** 7 numbers: pusher xy, block xy, block angle, pusher velocity. Block linear and
  angular velocity are not included.
- **`_set_state`** sets the pusher's pose and velocity and the block's pose, **leaves the
  block's velocity at whatever it was**, then advances physics by one substep. It cannot
  restore a branch root. `reset` accepts `options={"state": ..., "goal_state": ...}` and
  builds a fresh pymunk space each time.
- **T geometry:** two `pymunk.Poly` shapes on one body. In body coordinates the bar spans
  x from -60 to 60 and y from 0 to 30; the stem spans x from -15 to 15 and y from 30 to 120.
  The block xy in the state is the body position, not the centre of mass. Transform the
  actual shape vertices by the recorded pose rather than hard-coding a centre.
- **Normalisers:** StandardScalers fitted on the full expert `action`, `proprio` and `state`
  columns, with `goal_` duplicates, exactly as the historical planner did. Rebuilt once and
  stored with the Push-T assets.
- **Expert data:** 18,685 episodes, 2,336,736 steps, episode lengths 49 to 246 (mean 125).
  Read contiguous chunks only.

## The rule

The **whole T footprint** must not intersect a virtual hazard, an axis-aligned box or a disc
in arena pixels. The hazard is not rendered and has no physics: it changes the rule, not the
dynamics. `U = 1` if the transformed T polygons intersect the hazard at any environment step
of the 25-step horizon. Clearance is the signed polygon-to-hazard distance in pixels.

The pusher's action-space arena guard (commanded positions stay inside the arena) is a
common engineering control in every arm, so an arena exit can never pass as avoidance.

## Layouts, roots and proposals

- **Layout generator:** a hazard is placed across the route the nominal planner actually
  takes (its swept T footprint), with the start and goal footprints clear by at least the
  widest margin tested. Sizes and shapes come from declared ranges. Tune the generator on
  development cases that have a demonstrated feasible route, then freeze it. Never drop a
  test case because a method fails on it.
- **Layout families for E4:** familiar layouts have hazard centres in one set of arena cells
  and one size range; held-out layouts use disjoint cells and a different size range.
  Starts and goals are split the same way. Fix both before E1.
- **Roots:** a seeded reset with a sampled start and goal, then `k` blocks of the nominal
  controller, with `k` in {0, 2, 4, 6}. A root is stored as (seed, reset options, recorded
  action prefix), and each replay charges `5k` environment steps. If expert episodes replay
  faithfully from their initial states, they become a cheaper second source of roots.
- **Nominal controller:** LeWM with CEM toward the goal, 300 samples and 30 iterations, as in
  the historical planner. About 1 s per solve with threads pinned.
- **Proposal generator:** the nominal 5-block plan plus bounded Gaussian perturbations of
  every action at two or three declared noise scales. The **stress bank** adds tapes aimed at
  the T's corners and edges to force contact and rotation, and tapes that push the T toward
  the hazard.

## E0: make the experiment trustworthy (week of 28 September)

- Rebuild the assets on the fresh instance ([infrastructure.md](infrastructure.md)):
  weights, expert data, scalers, the pinned upstream revision and a run manifest. Push the
  assets to Hugging Face before anything else.
- Reproduce one short baseline planning episode with the historical settings.
- **Replay test:** 50 roots, including roots in mid-contact, each replayed three times from
  reset. Compare pusher and block pose, plus block linear and angular velocity read from
  the pymunk bodies, at every environment step of a 25-step suffix. Pass if all repeats
  agree to a declared tolerance (target: bitwise, or better than 1e-6 px).
- **Geometry:** T polygons from the shapes, hazard intersection and signed clearance.
  Overlay about 10 development contact traces on rendered frames and check them by eye.
- **Timing:** log every environment step inside each block, compare dense truth with
  endpoint truth plus interpolation, and inspect a few physics-substep traces around
  contact.

**Exit evidence:** an asset manifest with Hugging Face revisions, a replay report, the
overlays and the declared timing. **Stop** if replay is not deterministic through contact,
and fix that before anything else. A saved pose or a rendered animation does not pass
this gate.

## E1: find the bottleneck (week of 5 October)

- **Probe:** block pose `(x, y, sin θ, cos θ)` and pusher `(x, y)` from single-frame 192-d
  latents. Fit linear and small MLP probes, choose capacity on development data, then
  freeze. Training frames are about 20,000 frames read contiguously from expert episodes in
  the probe split (disjoint from the replay split used in E2). Development and test frames
  come from our own roots and branches, so probe error is measured where it is used.
- **Banks (provisional sizes):** a development bank of 24 roots with 8 tapes each and a test
  bank of 96 roots with 16 tapes each, plus a stress bank of the same shape. Size the final
  test bank from the variance seen in E1, not in advance.
- **Decomposition:** the four sources from
  [protocol.md](protocol.md#the-four-source-decomposition) on identical tapes. Measure
  centre error (px), periodic angle error (degrees), signed clearance error and the
  resulting decisions, by horizon block, contact versus free motion, rotation and distance
  to the hazard.
- **References:** a stationary-block predictor (the block never moves) and a small
  coordinate-dynamics MLP trained on expert state transitions. The second uses privileged
  simulator state and is labelled as such.
- **Mechanistic check:** compare predicted and true block displacement and rotation during
  contact. The working hunch is that the predictor under-predicts motion in contact, which
  would make it optimistic about hazards that the T is pushed into.

**Outputs:** the decomposition figure, error by horizon and by regime, and dial curves for
each source. **Gate:** imagination causes a material share of false-safe decisions near the
boundary. If readout dominates, fix observation or probe coverage before blaming the
predictor. If temporal sampling dominates, resolve that first. If the coordinate reference
solves the problem, that is evidence against making the latent method central.

## E2: can extra experience repair it? (week of 12 October)

- Collect a modest, diverse adaptation bank of about 512 branches, contact-rich and ordinary,
  from adaptation-split roots. Every simulated candidate inspected while building it counts
  toward its cost.
- Apply the recipe in [protocol.md](protocol.md#predictor-side-adaptation). Choose, on
  development data only: teacher-forced or short-rollout loss, learning rate, number of
  steps, and predictor-side or predictor-only updates. Then fix them for E3.
- **Controls:** no update; a fixed margin calibrated on development data; and a
  readout-only residual correction (a small MLP on imagined latents) trained on the same
  new data.
- **Retention:** prediction error on held-out expert clips, 20 fixed goal-reaching episodes
  without hazards, and the ordinary-motion check.

**Outputs:** a paired before-and-after table with charged costs, and representative
successes and failures. **Gate:** the E2 threshold in
[protocol.md](protocol.md#statistics-and-reporting). If only the readout correction helps,
report better interpretation, not repaired dynamics. If nothing helps after one bounded
optimisation check, pause acquisition work and write up the diagnosis.

## E3: which experience (weeks of 19 and 26 October)

- **Arms:** random and boundary, plus the learned selector only if E2 shows a strong repair,
  with the retrospective oracle as a ceiling.
- **Budgets:** 128 common seed branches, then 64, 128, 256 and 512 additional branches.
  Root depths are matched so that equal branch counts are equal charged steps. Rounds of
  64, 64, 128 and 256 branches reach those totals.
- **Seeds:** first one paired run (priority item 4 in the [plan](README.md#priority-order)),
  then three acquisition seeds.
- Every budget restarts adaptation from the released weights with the fixed recipe.

**Output:** FSA at matched acceptance against charged simulator steps for each arm, with
root-clustered intervals, and branch counts and compute beside them.

## E4: transfer (week of 26 October)

- The 2 by 2 grid: familiar or held-out hazard layouts, crossed with familiar or held-out
  starts and goals. Dynamics stay unchanged.
- Evaluate the representative and stress banks separately, plus retention and ordinary
  motion.
- "Held out" means held out from **our** study. The released model's pretraining data is
  not fully known, so never claim these states were unseen by the checkpoint.

**Output:** the transfer figure and a direct answer to H4.

## E5: closed loop, only if E3 holds (week of 2 November)

- Freeze all weights. Compare the original and repaired models inside Safe CEM with the
  whole-T constraint, a fixed-margin variant, and one matched penalty-CEM reference, with
  nominal planning for context.
- 20 paired episodes per arm over two hazard layouts, equal candidate budgets, a declared
  all-infeasible fallback, and an audit of the action sequence CEM actually returns.
- Report violations against the true hazard at every step, final block pose error and
  coverage, arena exits, infeasible searches and planning latency.

The historical Safe CEM code constrains the pusher, not the T. The whole-T constraint is new
code in new files; the historical modules are imported and never edited.
