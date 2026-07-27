# Stage 0: place the hazard

Part of [plan.md](plan.md). Written 2026-07-26.

**Purpose:** download LeWM's Push-T training data, plot where the pusher and block spend their time,
and choose a forbidden region that produces a real safety-performance trade-off.

**Budget:** about 1 hour, realistically 1 to 2 with HDF5 spelunking.

**Environment: Push-T, confirmed.** It has a standalone pip package (`gym-pusht`), its dataset is
HDF5 rather than a tarball, its containment test is point-in-region rather than segment-in-region, and
it is already named in the research proposal.

## Stage 0 stops before the planner

No NSGA-II here. Stage 0 ends when the region is chosen and the numbers are written down.

The split exists to fail fast. Stage 0 is an hour and it tells you whether Stage 1 can work at all. If
the planner gets built first and the hazard turns out to be badly placed, that is three hours burnt
debugging a planner that is behaving correctly on a broken problem.

## What it achieves

### 1. It stops the Pareto front from being degenerate

Objective 2 only produces a *front* if avoiding the hazard genuinely costs task performance.

| Regime | Hazard position | Result |
|--------|-----------------|--------|
| 1 | Off the natural path | Every plan avoids it for free. Safety objective flat. Front collapses to one point. |
| 2 | Blocking the only route, or on the contact geometry | No plan can avoid it and still succeed. Safety objective saturated. Front collapses at the other end. |
| **3** | **On the busy path, with a real alternative** | **Cutting through is fast and unsafe, going around is slow and safe. Real curve.** |

Only regime 3 is worth showing, and the heatmap identifies the regime before any planner code exists.

### 2. The heatmap doubles as a map of world-model competence

The visitation density *is* LeWM's training distribution. Dense means the model is well trained there
and prediction error will be low. Sparse means it degrades.

So one plot answers two questions: where to put the hazard so the trade-off is real, and where an
out-of-distribution signal could actually fire. Whether those two locations coincide is directly
relevant to the violation-of-expectation study, and it is the more interesting half of the
conversation with the supervisor.

### 3. It produces the baseline numbers

"Expert demonstrations enter this region in X% of episodes, spending a mean of Y steps inside" is row
one of the results table. Every later result is a comparison against it.

### 4. It builds the state logger used for the rest of the project

Violation counting, probe training labels, and evaluation metrics all need ground-truth state per
timestep. Write the containment check and the logging once, here.

## The data

Hugging Face dataset `quentinll/lewm-pusht`, file `pusht_expert_train.h5.zst`. Zstd-compressed HDF5 of
expert demonstrations. This is the exact data the checkpoint was trained on, so no policy needs to be
run to produce the heatmap.

Decompress with the `zstd` CLI or the `zstandard` Python package, then open with `h5py`. Walk the keys
to find the structure rather than assuming it.

**What to look for.** Push-T's conventional state is 5-dimensional: pusher x, pusher y, block x,
block y, block angle. Actions are 2D absolute target positions for the pusher. The arena is
conventionally 512 x 512. Verify all of this against the actual data ranges instead of trusting the
convention.

Also find how episodes are delimited, whether by an index array, an episode-id column, or separate
groups. Every per-episode statistic below depends on getting that right.

## Choosing the region

### Use a circle, not a box

Decide the shape now, with Stage 1's objective in mind. A circle gives you containment and penetration
depth as one-liners:

```
inside          : ||p - c|| < r
penetration     : max(0, r - ||p - c||)
```

An axis-aligned box needs a signed-distance computation for graded penetration, which is fiddlier and
buys nothing.

### Push-T specific guidance

The target T pose is fixed, so every expert trajectory converges toward the same goal region. That
creates a reliable hotspot, but **do not put the hazard on the goal itself.** That is regime 2: the
task becomes impossible and the front saturates. Place it on the *approach* to the goal.

If the pusher density looks diffuse (starts are randomised, so it will be somewhat spread), work
inward from the goal along the convergence funnel until you find a band that most trajectories pass
through.

### The two-heatmap test

Plot pusher density and block density **separately**, then check your candidate against both:

- **Overlaps pusher transit paths:** good. The pusher passes through on its way somewhere, so avoiding
  it costs a detour.
- **Overlaps the block's occupied region:** bad. The pusher has to reach the block to push it, so a
  hazard sitting on the block's territory covers the contact approach. Avoiding it does not cost a
  longer path, it costs the task. That is regime 2 in disguise and it is the most likely way to get
  this wrong.

The region you want sits on pusher transit, clear of where the block lives.

### Sanity check on escapability

In open 2D Push-T a plain detour almost always exists, so regime 1 (free to avoid) is the more likely
trap, not regime 2. If your entry rate comes out low, the region is in the wrong place. Move it, do
not shrink the criterion.

## Numbers to record

For the chosen region:

| Quantity | Why |
|----------|-----|
| Centre `(x, y)` and radius `r` | Needed by every later stage. Write it into a config, not a notebook cell. |
| Entry rate: fraction of episodes with at least one state inside | The headline baseline number. Wants to be substantial, not 2%. |
| Mean dwell: steps inside per episode, averaged over all episodes | Second baseline number. |
| Mean summed penetration depth per episode | This is literally the expert's objective-2 value, so it gives a reference point on the Pareto front before the planner exists. |
| Expert success rate, if the dataset records it | Gives the ideal corner of the front. |

Report mean dwell over all episodes, not only over episodes that entered, and say which you used.

## Coordinate frames

`gym-pusht` (standalone, Diffusion Policy lineage) and `swm/PushT-v1` (inside `stable-worldmodel`) may
not share an identical coordinate frame.

**Treat the dataset's own frame as ground truth**, since Stage 2 runs against the same data the model
was trained on. When you do the live environment check, confirm the units line up rather than assuming
they do. A silent factor-of-two or a flipped y axis here would poison everything downstream.

## Gotchas

- Make the placement decision on a 2D **histogram of visited positions**, not on a spaghetti plot of
  overlaid trajectories. The spaghetti plot is a good second visual for the slide; the histogram is
  what you read the decision off.
- Check whether the y axis is flipped between the data convention and matplotlib's default. Push-T
  renders with the origin at the top left.
- If episodes have variable length, do not average across a padded array without masking.
- Store the region definition somewhere importable (`configs/env/`), not inline in a plotting script.
  Stages 1 and 2 both need it.

## Checklist

**Data**

- [ ] Download `pusht_expert_train.h5.zst` from `quentinll/lewm-pusht`
- [ ] Decompress the zstd layer
- [ ] Open with `h5py` and walk the key structure, printing shapes and dtypes
- [ ] Identify the state array and confirm which columns are pusher x,y and block x,y,theta
- [ ] Identify how episodes are delimited
- [ ] Print min/max of each state column and confirm the arena frame (expect roughly 0 to 512)

**Heatmaps**

- [ ] 2D histogram of pusher position over all trajectories
- [ ] 2D histogram of block centre position over all trajectories
- [ ] Render both in the arena frame, correct axis orientation
- [ ] Optional second visual: overlay a sample of trajectories as lines

**Region selection**

- [ ] Propose two or three candidate circles on the pusher-density hotspots
- [ ] Reject any candidate overlapping the block's occupied region (two-heatmap test)
- [ ] Reject any candidate sitting on the goal pose itself
- [ ] For each survivor compute entry rate, mean dwell, mean summed penetration depth
- [ ] Pick the regime-3 candidate and record centre, radius, and all three numbers
- [ ] Save the region definition to `configs/env/`

**Live environment**

- [ ] `pip install gym-pusht`
- [ ] Reset and step the env
- [ ] Read pusher and block state from it
- [ ] Confirm the units and axis orientation match the dataset frame

**Reusable code left behind**

- [ ] Containment test function
- [ ] Penetration-depth function (same shape as Stage 1's objective 2)
- [ ] Per-timestep state logger

**Done when**

- [ ] A heatmap image exists with the chosen region drawn on it
- [ ] Centre, radius, entry rate, mean dwell, and mean penetration depth are written down
- [ ] The region lives in a config file, not a notebook cell
