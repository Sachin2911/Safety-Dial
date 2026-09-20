# Animations

Two independent sets live here. Rebuild with

    uv run python experiments/scripts/make_animations.py             # Push-T Safe CEM
    uv run python experiments/scripts/make_locomotion_animations.py  # Phase 0 locomotion

Every frame is the real Push-T renderer replaying a **committed simulator state** from
`docs/safeDial/results/`, not a reconstruction, with the hazard drawn on top:

- **solid red box** — the true hazard. Violations are always measured against this box.
- **dotted box** — that hazard inflated by the dial `d`. This is the keep-out region the
  planner tests candidate plans against, and it is the only thing the dial changes.

Each is written twice: `.gif` for inline viewing, `.mp4` (about a tenth the size) for slides.

| File | What it shows |
|------|---------------|
| `dial_sweep` | The headline Safe CEM result. Seed 0 at `lambda = 0` (no hazard term) and at dials `d = 0, 20, 40`. The unconstrained planner cuts through the hazard on 7 of 50 steps; Safe CEM enters it on none, at any dial, and the route bends further out as `d` grows. |
| `penalty_vs_safe_cem` | A weight is not a threshold. Seed 1 under penalty CEM at `lambda = 0` and `lambda = 1`, against Safe CEM at `d = 40`. `lambda = 1` cuts the violation from 0.32 to 0.06 but not to zero: a weighted sum will still buy goal progress with a little violation, and constraint-priority ranking will not make that exchange. |
| `dial60_arena_escape` | Why a probe cannot enforce a constraint outside its own validity domain. Both panels are `d = 60`, same seed. Left, the arena constraint is read off the probe: the planner escapes the 512 px arena entirely, where no encoder output carries the pusher's position, and scores zero violations *vacuously* — the block never moves. Right, the same constraint computed in action space closes the exploit. |
| `dial_response` | What the operator is turning and what it costs. `d` sweeps 0 to 60 px, the keep-out region inflates, and the measured curves fill in beside it: violations at exactly 0.000 across the dial, task cost non-monotone within 3-seed noise, feasible set genuinely tightening. |

## Provenance, which matters for every `d = 60` panel

The sweep in `results.json` ran with the **probe-based** arena constraint, which is
structurally unable to detect a genuine arena exit; that is why its `d = 60` episode leaves
the arena. `dial60_arena_escape` animates the paired re-run of both variants
(`arena_fix.json`), and `dial_response` takes its `d = 60` point from the **fixed**
action-space variant, which is the honest number. Full discussion in
[`notes/safeCEM.md`](../notes/safeCEM.md) and [`docs/safeDial/SafeDialReport.pdf`](../docs/safeDial/SafeDialReport.pdf).

Three seeds and a reduced grid. The violation axis saturating at zero across the whole
usable dial means **H4 is not demonstrated** by this run; see the notes for the gate
geometry proposed to get a real trade-off.


## Phase 0 locomotion (`walker2d_*`, `hopper_*`)

Every frame is the real MuJoCo renderer replaying a simulator state the oracle actually scored,
with the state readout and a torso-height gauge showing the benchmark's healthy band. Rebuild
with `make_locomotion_animations.py`.

| File | What it shows |
|------|---------------|
| `walker2d_fall_and_getup` | The Phase 0 result in one clip. The robot walks, the benchmark declares failure **while it is still standing** at `z = 0.78`, it goes all the way to the ground at `z = 0.03`, and then a CEM plan stands it back up under the same action bounds any policy has. Nothing about this fall is irreversible, which is why Walker2d cannot support the proposal's experiment. |
| `hopper_getup` | A Hopper state 150 steps past the flag, torso at `z = 0.084`, recovered to `z = 1.087`. |
| `hopper_irrecoverable` | The contrast: a Hopper state the search could not recover. Read it with the caveat below. |

**Caveat on `hopper_irrecoverable`.** Hopper's irrecoverable set is a property of the recovery
predicate rather than of the dynamics. Relaxing the pitch tolerance from 0.10 to 0.15 rad, still
inside the benchmark's own 0.2 rad healthy band, empties it; so does shortening the dwell
requirement, and so does doubling the horizon. The clip shows the best attempt found, and that
attempt does reach an upright, healthy pose without ever holding it for 25 consecutive steps.
See [`docs/phase0/Phase0Report.pdf`](../docs/phase0/Phase0Report.pdf) section 4.4.
