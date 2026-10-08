# Safe CEM and the Safety Dial on Push-T

Code: `experiments/helpers/safeCEM.py`, `experiments/helpers/probes.py`,
`experiments/scripts/safe_dial_pusht.py`, `experiments/scripts/plot_safe_dial.py`.

## Why the earlier lambda sweep failed

The `hazardSweep.py` result (every non-zero lambda abandoned the task, block never moved,
pusher driven out of the arena) had three independent causes. All are fixed in `safeCEM.py`.

### 1. The start was inside the inflated hazard

`calibrate_box` places the box on the baseline path but never checks that the start clears
it once the probe margin is added. The box it chose, `(2, 82, 83, 163)`, inflates by
`PROBE_MARGIN = 35.5` to `(-33.5, 117.5, 47.5, 198.5)`, which **contains the pusher start
(70, 70)**. Every candidate plan was in violation from step zero, so the penalty was a
near-constant offset and the only way to reduce it was to leave. Notebook cell 41 works
around this by hardcoding a box "clears the start by 12.5px after margin".

### 2. The planner could escape the arena for free

The Push-T pusher is a `pymunk.Body(KINEMATIC)`
(`stable_worldmodel/envs/pusht/env.py:606`) driven by a PD controller toward
`agent.position + action * 100`, and the arena clip in `env.step` is **commented out**
(`env.py:320`). Kinematic bodies ignore collisions, so the pusher passes straight through
the walls. Verified directly: 40 steps of action `(-1, -1)` put it at `(-1380.9, -1313.1)`,
and 40 steps of `(+1, +1)` brought it back to `(224.7, 292.6)`.

So fleeing the arena drives any hazard penalty to zero at no cost. The lambda sweep's
"safety" was bought by escape, not avoidance, which is why every non-zero lambda had
`oob = 1.0`. `arena_exit_depth` makes leaving the arena a constraint violation inside the
planner.

The block, by contrast, is a dynamic body enclosed by static wall segments spanning
(5, 5) to (506, 506), so it cannot leave: pushed into the left wall it stops at x ~ 48.

### 3. A weight is not a threshold

With a scalarised cost `goal + lambda * hazard` there is no notion of a feasible plan, so
lambda has no units and no operating-point meaning. The dial has to be a threshold.

## Results (20 Sep 2026, 3 seeds, reduced grid)

Committed artifacts, because `outputs/` is gitignored and this box is **not** a persistent
volume (`workspace_is_volume = false`), so anything left there is destroyed on recycle:

- `docs/safeDial/results/results.json` and `results_states.npz`, the sweep
- `docs/safeDial/results/arena_fix.json` and `arena_fix_states.npz`, the paired dial-60 run
- `docs/safeDial/latex/figures/`, every figure
- `docs/safeDial/SafeDialReport.pdf`, the full report
- `animations/`, four animations of these same episodes, rebuilt by
  `experiments/scripts/make_animations.py` (see `animations/README.md`). Frames are the
  real Push-T renderer replaying the committed states, so nothing in them is a redraw of
  the summary numbers: `dial_sweep` (the headline, d = 0/20/40 against lambda = 0),
  `penalty_vs_safe_cem` (same seed, the exchange a weight makes and a threshold refuses),
  `dial60_arena_escape` (the paired probe-based/action-space run), and `dial_response`
  (the keep-out region inflating against the measured curves).

Hazard box `(30, 130, 149, 249)` calibrated on the baseline route, baseline violation 0.14.
Violations are always measured against the **true** box from simulator states.

> **Provenance, and it matters for the dial-60 row.** This sweep ran *before*
> `safeCEM.commanded_positions` existed, so its arena constraint was the **probe-based** one.
> The action-space constraint is now the default, so a plain rerun will **not** reproduce the
> dial-60 breakdown. Reproduce this exact table with
> `uv run python experiments/scripts/safe_dial_pusht.py --quick --seeds 3 --probe-arena`.
> The `arena_constraint_mode` field in the committed JSON records which variant was used.

    penalty CEM (arena constrained)        Safe CEM (dial d, px clearance)
     lam   viol frac    block err           dial   viol frac    block err   oob
       0  0.188+-0.097  11.2+-0.6              0  0.000+-0.000   9.0+-1.1  0.000
     0.1  0.073+-0.031   7.8+-2.1             20  0.000+-0.000  11.9+-1.9  0.000
       1  0.020+-0.028   4.9+-1.7             40  0.000+-0.000   6.7+-2.0  0.000
      10  0.011+-0.015   7.9+-2.4             60  0.000+-0.000  60.0+-0.0  0.953

**The headline, stated precisely.** Safe CEM achieves exactly zero violations on **every**
episode at every usable dial: 9 of 9 across d = 0, 20, 40. Penalty CEM achieves zero on only
**4 of 9** episodes at lambda > 0, so no weight achieves it *reliably* and every weight's mean
is non-zero (lambda = 10 still averages 0.011, with one seed at 0.032).

Per-episode violation fractions:

    lam=0     0.140  0.323  0.100     zeros 0/3
    lam=0.1   0.114  0.040  0.065     zeros 0/3
    lam=1     0.000  0.060  0.000     zeros 2/3
    lam=10    0.000  0.000  0.032     zeros 2/3
    d=0       0.000  0.000  0.000     zeros 3/3
    d=20      0.000  0.000  0.000     zeros 3/3
    d=40      0.000  0.000  0.000     zeros 3/3

That difference is structural, not a matter of tuning. A weighted sum will accept a small
violation in exchange for a large enough goal improvement, so whether a given episode comes
out clean depends on the particular trajectory; constraint-priority ranking never makes that
exchange, so it is clean every time. The price is about 4 px of task accuracy against the
best-tuned lambda (9.0 vs 4.9 px).

Do not state this as "no lambda ever reaches zero": individual penalty episodes do. The claim
that survives the data is about *reliability*.

**Correction to the project's motivation story.** With the geometry fixed, penalty CEM is
*not* fragile here: violation falls monotonically in lambda (spearman -1.0) and
out-of-bounds is 0.000 at every lambda, even in the arm with no arena constraint at all.
The earlier "every non-zero lambda abandoned the task and fled the arena" result is
therefore **confounded by cause 1**, the start sitting inside the inflated hazard.
Earlier guidance and the
[historical proposal](https://github.com/Sachin2911/Safety-Dial/blob/a7799a22f76b7af3284a2e94b0ca67f6c65d4f95/docs/revisedProp/latex/main.tex)
presented that sweep as clean evidence of penalty fragility. The corrected geometry
invalidates that interpretation. The usable Safe-CEM runs recorded zero true-box
violations; that is an empirical result, not a zero-violation guarantee. The
[current plan](../docs/researchDirection.md) retains these corrected experiments as evidence.

**H4 is not demonstrated by this run.** The safety axis saturates at 0.000 for every
usable dial, and block error is non-monotone within seed noise (9.0, 11.9, 6.7 px at
+-1 to 2 px). `detour_feasible` reported 694 px of slack against a 1062 px budget, so
tightening the dial costs almost nothing. The dial does bite on the planner's internal
feasible set (frac feasible 0.974, 0.964, 0.796), but that is not a trade-off curve.

To get a real trade-off, use a **gate**: two boxes separated by a gap of width w instead
of one box. For d < w/2 the planner passes through, for d > w/2 the gate closes and it
must detour or give up. Sharp, interpretable, and still "a hazard in between".

### The dial 60 breakdown, and why a probe cannot enforce an arena constraint

At d = 60 every seed collapses: block moved 0.0 px (final error 60.0 +- 0.0 px, exactly
the start to goal block distance), out-of-bounds on 47 to 48 of 50 steps, pusher reaching
x = -276, y = -562. The reported violation of 0.000 is achieved **vacuously**, by leaving
the arena so the true box is never entered.

The arena constraint did not stop it, and could not have. It was evaluated on
`probe(z_hat)`, and the probe is a regressor trained on in-arena frames: once the pusher
truly leaves the 512 px arena it is not visible in the 224 px frame at all, so no encoder
output carries its position. The planner reported **80.6% of candidates feasible** while
driving off-screen. A probe-based constraint cannot police the probe's own validity
domain, and a conservative dial pushes the planner straight toward that blind spot,
because escaping is the cheapest way to satisfy a tight hazard constraint.

The fix is `safeCEM.commanded_positions`: compute the arena constraint in **action
space**. The pusher is position-controlled (`agent.position + action * 100`), so the
commanded trajectory is an exact function of the candidate actions and the current
position, with no world model and no probe. Enabled by `action_space_arena=True` in
`run_episode_dial`. The pragmatic alternative is to restore the commented-out action clip
in `env.step`, which makes the arena physically inescapable.

This failure mode is itself a result worth keeping: it is a concrete, measured instance of
why latent safety filters need an uncertainty or validity-domain term, which is exactly
UNISafe's contribution.

**Fix validated.** Re-running dial 60 on the same three seeds with
`action_space_arena=True` and nothing else changed:

    seed   viol   block err   oob    block moved   pusher x range      frac feasible
       0  0.000      9.4 px  0.000       66.8 px   [ 58.6,  301.7]            0.613
       1  0.000     15.8 px  0.000       67.8 px   [ 60.6,  307.1]            0.692
       2  0.000     24.8 px  0.000       84.0 px   [ 63.1,  326.0]            0.644

Out-of-bounds 0.953 -> 0.000, block moved 0.0 -> 67 to 84 px, block error 60.0 -> 9.4 to
24.8 px, violation still 0.000 but now genuinely rather than vacuously. Frac feasible
falls from 0.806 to about 0.65, which is the honest figure: the constraint is real now, so
fewer candidates satisfy it.

This also partly rescues H4. With the action-space constraint the dial gives mean block
error 9.0 px at d = 0 against 16.7 px at d = 60, both at **zero** violations, which is a
genuine conservatism cost on the task axis. Seed variance is large (9.4, 15.8, 24.8), so
re-run with more seeds and the full dial grid before quoting it.

## Safe CEM

`CEMSolver` selects elites with `torch.topk(costs, largest=False)`, a pure scalar ranking,
so constraint-priority ranking can be encoded in the scalar and the upstream solver needs
no modification:

    feasible:   cost = goal_cost
    infeasible: cost = (max feasible goal cost + 1) + violation

Every feasible candidate sorts ahead of every infeasible one; feasible ones sort by goal
progress; infeasible ones sort by violation magnitude, so the search still has a gradient
toward feasibility when nothing is safe yet. This is the constraint-priority ranking of
MPC-RCE (`docs/papers/myPapers/MPC-RCE.pdf`).

Keep the infeasible branch **graded**, not a constant: CEM sets
`var = topk_candidates.std(dim=1)`, so a constant cost makes the elite set arbitrary and
the search collapses.

The dial `d` is the required clearance in px: the hazard box is inflated by `d` for the
feasibility test, while violations are always **measured against the true box**.

## Hazard placement needs two conditions at once

Each existing helper has one and lacks the other:

- `hazardSweep.calibrate_box` puts the box on the observed baseline path, so there is
  genuinely something to avoid, but never checks start/goal clearance (cause 1 above).
- A box placed on the straight line between start and goal clears the endpoints but the
  planner need not follow that line. Measured directly: baseline violation fraction
  **0.00**, so the sweep had nothing to measure.

`safeCEM.calibrate_box_on_path` requires both: on the route the unconstrained planner
actually took, and clear of start and goal once inflated by the widest dial. It prints the
full candidate table.

## Box gotcha: CPU quota versus nproc

This container has an **11.52-CPU cgroup quota** (`/sys/fs/cgroup/cpu.max` = `1152000
100000`) while `nproc` and `os.sched_getaffinity` both report **96**. Torch sizes its
intra-op thread pool from the affinity mask, so it spawns 96 threads into 11.5 cores and
thrashes. Symptom: a CEM solve takes 15 s instead of the 1.0 s the same planner reaches on
a quiet box, with the **GPU at 0 to 4 percent** the whole time and the process pinned near
half a core. `/sys/fs/cgroup/cpu.stat` showed `nr_throttled 15228`.

Fix, before importing torch:

```python
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "8")
import torch
torch.set_num_threads(8)
```

Measured at `n_steps=10`, two solves each, while the box was shared: bare LeWM goal cost
4.4 s, `DialCostModel` (safe) 6.6 s, notebook `HazardAugmentedCostModel` 15.8 s. The
constraint machinery is cheap; the thread thrash was the cost.

**Check `cpu.max` before blaming the model.** Also check whether another agent is running
on the same box: `ps -eo pid,pcpu,etime,comm --sort=-pcpu`.

## Probe cache

`probes.load_or_train_pusher_probe` caches the MLP probe to `data/probes/pusher_mlp.pt`
(gitignored), so a fresh process no longer re-encodes 100k frames. Trained result:
**R2 = 0.9991, RMSE 3.03 px, radial p95 7.5 px** on held-out real latents, reproducing the
notebook. Note this is error on *real* latents; on *imagined* latents at rollout depth 5
the p95 radial error is 35.5 px, which is the principled floor for the dial.

The encoder streams chunk by chunk. An earlier version accumulated all 100k frames with
`np.concatenate`, needing about 30 GB peak, and on a shared box it thrashed: 164 s of
system time against 16 s of user time, never reaching the encoder. Streaming holds peak
RSS at 1.7 GB.

End-to-end encode rate including cold spread-offset reads is **939 frames/s**, so a full
2.34M-frame latent bank is about 44 min. The 8k frames/s figure from a GPU-only
microbenchmark is not the rate you get.

## Reading the HDF5

Contiguous reads run at ~19,000 frames/s; a sorted scattered fancy-index runs at
**46 frames/s**, a 415x penalty. Never fancy-index the pixels array; read contiguous spans
at spread offsets.
