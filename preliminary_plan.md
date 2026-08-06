# Preliminary Experiments — State-Defined Hazards on Frozen LeWM

**Premise.** A hazard need not be rendered. Declare a region of state space, read position out of the latent with a probe, compute hazard cost geometrically. The world model sees unmodified in-distribution Push-T, so the released checkpoint is sufficient substrate. No environment modification, no retraining.

**Environment:** Push-T. **Checkpoint:** `quentinll/lewm-pusht` — pin a revision hash (two versions exist: Mar 27, Apr 16).

---

## Step 0 — Data check

Confirm the released dataset carries simulator state, not just pixels + actions. Probes need ground-truth labels. If absent, re-render or regenerate before proceeding.

## Step 1 — Position probe

Linear probe: post-projection latent `z` (192-d, CLS after MLP+BatchNorm — *not* raw CLS) → block position (x, y).

Validate on **imagined** latents, not just encoder outputs: report probe error at rollout depth t = 1 … 5. Trained on real frames, deployed on predictor outputs — decay here means the safety signal is unreliable exactly where it is used.

Expected to work: paper reports r ≈ 0.99 for Push-T positional probing.

## Step 2 — Hazard cost

Define a rectangle in workspace coordinates between block start and goal. Cost = penetration depth, **summed over the rollout** (`t = 1 … H`), not terminal-only.

Geometry, not a trained classifier. Region can be moved, resized, or duplicated without retraining.

## Step 3 — Penalty-CEM baseline

`C = ‖ẑ_H − z_g‖² + λ · hazard_cost`. Sweep λ over several orders of magnitude.

Each λ yields one (task success, violation rate) point; the sweep traces an empirical Pareto front. This is baseline (ii) *and* the demonstration of penalty fragility in own setup — more persuasive than citing it.

## Step 4 — NSGA-II

Same two quantities kept separate. One run returns the front.

Match **rollout budget** to CEM (300 candidates × 30 iterations = 9,000 latent rollouts per planning step), not generation count. Fix NSGA-II parameters once and hold them fixed, mirroring LeWM's treatment of λ.

Compare fronts by hypervolume against Step 3.

---

## Parallel diagnostics (cheap, retire known risks)

| Check | Question | Consequence if negative |
|---|---|---|
| Latent jerk variance | Does jerk discriminate anything, given emergent temporal straightening (App. H)? | Drop jerk as a Pareto axis now, not in September |
| `‖z‖²` vs χ²(192) | Does SIGReg deliver calibrated N(0, I) in practice? | If yes: imagination-time OOD signal is live. If no: publishable gap between guarantee and practice |

---

## Deferred (needs retraining / modified env)

- Real violation rates — a probe-defined hazard is virtual; the simulator does not enforce it, so only *predicted* violations are measurable
- Surprise as a safety signal — requires hazards that alter dynamics
- Any visual hazard rendering — LeWM discards control-irrelevant visual detail by design (Fig. 10: colour perturbations produce no significant surprise)

## Ordering rationale

If NSGA-II fails to reach CEM parity, that surfaces on a frozen checkpoint in an unmodified environment — not alongside a custom environment that also needs debugging.
