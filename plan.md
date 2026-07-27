# SafetyDial: build plan

Working document. Written 2026-07-26, ahead of the supervisor meeting on 2026-07-27.

**Goal:** get a running demonstration of multi-objective evolutionary planning with a forbidden
region in Push-T, staged so that each stage is independently presentable if the next one does not
land.

## The two target images

Everything below exists to produce these.

**Image 1: trajectory overlay.** The Push-T arena, forbidden region shaded, and four or five planned
paths drawn on top, taken from different points on the Pareto front. One cuts the corner straight
through the region, one takes a wide detour, the rest sit in between. Same planner, same model, no
retuning: the operating point is selected by picking a point on the front. This picture is the thesis.

**Image 2: the Pareto front.** Task cost on x, hazard cost on y, non-dominated set highlighted, with
scalarised CEM plotted as a handful of separate dots at different penalty weights. The contrast
between one curve and a scatter of committed-in-advance points is the argument for the method.

## Stage 0: place the hazard (about 1 hour)

Download LeWM's Push-T training data, plot where the pusher and block spend their time, and place the
forbidden region on the busy part.

Dataset is public: `quentinll/lewm-pusht` on Hugging Face, containing `pusht_expert_train.h5.zst`
(zstd-compressed HDF5 of expert demonstrations). This is the exact data the checkpoint was trained on,
so no policy needs to be run to get the heatmap.

### Why this matters

**It stops the Pareto front from being degenerate.** Objective 2 only produces a *front* if avoiding
the hazard genuinely costs task performance. Three regimes:

| Regime | Hazard position | Result |
|--------|-----------------|--------|
| Off the natural path | Every plan avoids it for free | Safety objective flat, front collapses to one point |
| Blocking the only route | No plan can avoid it and still succeed | Safety objective saturated, front collapses again |
| **On the greedy path, detour available** | Cutting through is fast and unsafe, going around is slow and safe | **Real trade-off, real curve** |

Only the third regime gives anything worth showing. The heatmap identifies the regime before any
planner code exists.

**The heatmap doubles as a map of world-model competence.** The visitation density *is* LeWM's
training distribution. Dense regions are where the model is well trained and prediction error will be
low. Sparse regions are where it degrades. So one plot answers two questions: where to put the hazard
so the trade-off is real, and where an out-of-distribution signal could actually fire. Whether those
two locations coincide is directly relevant to the violation-of-expectation study.

**It produces the baseline number.** "Expert demonstrations enter this region in X% of episodes,
spending a mean of Y steps inside" is row one of the results table. Every later result is measured
against it. The number needs to be substantial, not 2%.

**It builds the state logger used for the rest of the project.** Violation counting, probe training
labels, and evaluation metrics all need ground-truth state per timestep. Write the containment check
and the logging once, here.

### Checklist

- [ ] Pull the dataset, decompress the zstd, open the HDF5, inspect the structure
- [ ] Locate the state arrays: pusher x,y and block x,y,theta per timestep
- [ ] 2D histogram of pusher position over all trajectories, rendered in the arena coordinate frame
- [ ] Same for block centre
- [ ] Verify the coordinate frame against the data ranges (Push-T conventionally uses 512x512)
- [ ] Overlay two or three candidate regions (circle or axis-aligned box, whichever is easier to test containment against)
- [ ] For each candidate compute: fraction of episodes entering, mean steps inside, whether a clear detour exists
- [ ] Pick the regime-three candidate, record its coordinates
- [ ] Separately: `pip install gym-pusht`, reset, step, confirm the same state variables are readable live and in the same units as the dataset

### Done when

A heatmap image with the chosen region drawn on it, plus four numbers: entry rate, mean dwell time,
and the region's arena coordinates.

## Stage 1: the evolutionary planner (2 to 3 hours)

Build NSGA-II over action sequences against a cheap analytic surrogate. **This does not need LeWM**
and is the guaranteed demo.

### What it proves

That the planner returns a *set* of plans spanning a trade-off, at the same compute budget where CEM
returns one. Everything downstream depends on this working.

Keep the dynamics deliberately trivial. This is a test harness where the right answer is already
known, so any weirdness is unambiguously a planner bug. That property disappears once a 15M-parameter
world model is behind it.

### The surrogate

Push-T's action is an absolute target position for the pusher, and the sim drives the pusher toward
it. So:

```
p_{t+1} = p_t + clip(a_t - p_t, -v_max, +v_max)
```

Move toward the commanded point, capped speed. **Do not model the block.** Contact dynamics are the
hard part and are exactly what LeWM exists to handle.

The demo task becomes "pusher goes from A to B with a hazard between them."

### Encoding

Decision variable is an entire action sequence. Horizon H=16 with a 2D action gives a 32-dimensional
real vector, box-bounded by the action space.

**Use the direct flattened encoding** `[a_0 ... a_{H-1}]`, because it is exactly the space CEM
searches. A spline control-point encoding converges faster and produces prettier paths, but then the
baseline comparison is no longer apples to apples and the first question will be whether the win came
from the encoding rather than the algorithm. Keep spline encoding as a fallback if convergence is bad.

At 32 variables with 2 objectives, NSGA-II should be comfortable.

**Trap:** since actions are absolute positions, the optimiser can produce sequences that jump across
the arena between consecutive steps. The speed cap absorbs this, so the path stays continuous even
when the action sequence looks jumpy.

**Rule to adopt now:** compute trajectory-based objectives on the rolled-out states, never on the
decision variables. When latent jerk arrives in Phase 3, it is jerk of the latent trajectory, not of
the action vector.

### Objectives

**Objective 1 (task):** terminal distance to goal, `||p_H - p_goal||`. Terminal only, mirroring
LeWM's `||z_H - z_g||^2`.

**Objective 2 (hazard):** summed penetration depth,

```
sum_t max(0, r - dist(p_t, centre))
```

The alternatives are worse and the reason generalises. Binary "did it enter" takes two distinct values
across the whole population, so non-dominated sorting cannot separate anything, most individuals tie,
and the search stalls. Step count is better but coarse and integer-valued.

**General principle: evolutionary selection needs a graded signal.** An objective that is flat across
most of the population provides no selection pressure, and selection pressure is the entire thesis.
When surprise and jerk arrive, check their spread across a population before trusting them.

### NSGA-II setup

pymoo, `Problem` subclass, `n_var=32`, `n_obj=2`, `xl`/`xu` from the action bounds. SBX crossover and
polynomial mutation, defaults are fine.

Two things to get right:

**Write `_evaluate` vectorised.** Take the full population `(pop_size, n_var)` and return
`(pop_size, n_obj)` in one call rather than looping. It barely matters with the surrogate. It matters
enormously once LeWM is behind it, because the whole population becomes a single batched GPU forward
pass. Designing for it now avoids a rewrite.

**Match the compute budget exactly.** LeWM's Push-T CEM configuration is 300 samples over 30
iterations, top 30 elites, so 9000 rollouts. Run NSGA-II at 300 population over 30 generations, or 100
over 90. State the number on the plot, otherwise the first question is whether the win came from
spending more compute.

### The comparison that makes the point

Do not plot one CEM dot. Implement scalarised CEM with cost `task + lambda * hazard` and **sweep
lambda** over about five values across a few orders of magnitude. Plot all five dots against the
single NSGA-II front.

That image does three things at once:

1. Each lambda buys exactly one operating point, chosen before the run.
2. The mapping from lambda to outcome is nonlinear and unintuitive. Expect two values landing nearly
   on top of each other and then a large jump. That is the reward-engineering problem, made visible.
3. One NSGA-II run at the same total budget covers the whole curve.

CEM itself is about thirty lines: sample Gaussian, evaluate, keep top-K elites, refit mean and
covariance, repeat.

### Metrics

Hypervolume via pymoo needs a reference point dominated by everything. Fix it once, write it down,
reuse it forever. HV values are meaningless across different reference points.

IGD needs a reference front that does not exist for this problem. Generate one by running NSGA-II once
at a very large budget and treating that as ground truth. Note in the writeup that it is an
approximation.

### Failure modes

| Symptom | Likely cause |
|---------|--------------|
| Front collapses to a point | Hazard placement wrong (Stage 0 should have caught it), or objective 2 is not graded |
| Paths look like noise | Too few generations, or bounds too loose |
| Front is a straight 45-degree line | Objectives trivially anti-correlated; move the hazard off the direct line for richer geometry |
| Everything scores identically on objective 2 | Hazard is somewhere nothing goes |

### Done when

Both target images exist, the compute budget is stated on the plot, and `src/safetydial/planners/`
and `src/safetydial/objectives/` are populated.

## Stage 2: swap in real LeWM (risky, time-boxed)

A surface-area problem. Only two functions are needed out of LeWM:

- `encode(obs) -> z`, images to 192-dim latents
- `predict(z_history, action) -> z_next`

Everything from Stage 1 stays. With a clean `WorldModel` interface this is an adapter, not a rewrite.

### Bypass their eval loop

Two routes: register a custom planner inside `stable_worldmodel`'s policy abstraction and run their
`eval.py`, or load the weights directly and drive an own MPC loop.

**Take the second.** Fighting an unfamiliar Hydra config hierarchy under time pressure is a bad trade
when only two tensors' worth of functionality is needed. Route one becomes worth it later, when
reproducing their exact reported baseline numbers under their exact eval protocol, which is a separate
task.

### Install reality

Their README says `uv venv --python=3.10` and `uv pip install stable-worldmodel[train,env]`. This repo
pins 3.11. Try 3.11 first, the 3.10 may just be what the authors happened to use. If it fights, make a
throwaway venv outside the repo and reconcile the pin later. Do not let a version-pin question block
the research question.

The conversion step in their README loads `weights.pt` plus `config.json` and writes an `_object.ckpt`
under `$STABLEWM_HOME`. Expect to lose time here. It is the most likely failure point.

### Three landmines

**`num_frames: 3`.** The predictor is conditioned on a three-frame window, not a single latent. The
rollout signature is `rollout(z_history, actions)`, and at episode start the history needs padding by
repeating the first observation. Getting this wrong still runs, still returns tensors, and quietly
produces garbage.

**The action encoder is almost certainly chunked.** Across the published checkpoints:

| Env | Raw action dim | `action_encoder.input_dim` |
|---------|----|----|
| Reacher | 2  | 10 |
| Push-T  | 2  | 10 |
| Cube    | 5  | 25 |

Every one is exactly 5 x action_dim, which suggests each predictor step consumes a chunk of 5
primitive actions. **Verify this in the code before setting bounds.** If it holds, a latent horizon of
H predictor steps means 5 * H * action_dim decision variables. For Push-T at H=4 that is 40 variables.

**A goal observation is required.** The cost is `||z_H - z_g||^2` with `z_g = enc(o_g)`, so an actual
rendered image of the goal state is needed, not just goal coordinates. Find how their Push-T eval
config constructs it.

### Validate before trusting

Do not wire the model straight into NSGA-II. Three checks, in order:

1. Encode two near-identical observations. Latents should be close. If not, preprocessing is wrong,
   probably image size or normalisation. Config says 224px, patch 14.
2. Encode frames along a demo trajectory approaching the goal. `||z_t - z_g||` should decrease roughly
   monotonically. If it does not, the cost function has no signal and the planner cannot work.
3. **Open-loop rollout error versus horizon.** Take a real demo trajectory from the Stage 0 dataset,
   feed its true actions through the predictor, compare predicted latents against encoded
   ground-truth latents at each step, plot error against horizon.

Check 3 is not just a sanity test. It identifies the horizon at which the model stops being
trustworthy, which is how H gets *chosen* rather than guessed. It also makes a good figure.

### Batching

Encode once per replan, on the current observation. The entire population then rolls out through the
predictor only. The predictor is small (6 layers, 192 dims), so 300 candidates by H steps is cheap.
The ViT encoder is the expensive part and it runs once. That asymmetry is why LeWM plans in under a
second, and why the vectorised `_evaluate` from Stage 1 pays off here.

### Done when (realistic for one evening)

Checks 1 to 3 passing, plus a rollout-error-versus-horizon plot. That alone says the checkpoint loads,
its interface is understood, and its degradation point has been measured.

## Time-boxing

Pick a wall-clock cutoff now, roughly three hours before intended sleep. **If the checkpoint is not
loading by then, stop and polish Stage 1's plots.** A crisp Stage 1 demo beats a half-working Stage 2
with nothing to show.

## Framing notes for the meeting

**Say "objective", not "constraint".** The ideation document argues that constrained RL fails
precisely because it forces specification of a cost function and a threshold, which is as hard as
specifying the reward. Saying "safety constraints" describes CPO, which is the baseline, not the
method. The pitch is a *second objective*, unthresholded, with the trade-off resolved by the Pareto
front at deployment.

**Label the demo honestly.** Two substitutions need stating out loud:

> "For this demo I used hazard distance directly as the second objective, so the trade-off is visible.
> In the real system that slot gets filled by structural signals, surprise and latent jerk, or by a
> probe, and the hazard goes back to being the evaluation metric."

> "The planner is validated on analytic pusher dynamics. The LeWM swap is next."

Do not let the surrogate be mistaken for the world model.

**Why the frozen model is not a limitation.** It makes the search affordable (tens of thousands of
imagined rollouts per decision, impossible to get from the environment). Its latents already encode
pusher position and block pose, since pixel dynamics cannot be predicted otherwise, and the paper
decodes physical state from them in App. D. And its ignorance of the hazard is what makes the result
mean something: a reward-free, hazard-agnostic, frozen model plus selection pressure yielding safe
behaviour is a real claim, whereas a model trained to know about hazard X avoiding hazard X is not.
It also means the hazard can be changed at deployment with nothing retrained.

## Questions for the supervisor

1. **Hazard placement.** Show the heatmap and the chosen box. Does the placement create the intended
   trade-off?
2. **Ruler or target.** Should the hazard stay purely an evaluation instrument (optimise structural
   signals only, then *measure* hazard entry), or is a probe-based hazard objective acceptable given
   it never requires picking lambda?
3. **Breadth.** Bring Reacher up as a second environment, or go deep on Push-T?

## Open research questions

**Surprise cannot be prediction error at plan time.** Prediction error needs a ground-truth next
observation, and during imagination there is none. So surprise has to be an *out-of-distribution
score on the imagined latents*. The Stage 0 dataset supplies the ingredient: encode the training
trajectories once, fit something cheap (k-NN distance or a Gaussian in latent space), and score
imagined trajectories by drift from the trained-on region. Still faithful to the proposal's intent,
different mechanism from what the phrasing suggests. Worth resolving before it gets probed.

This also closes a loop: the Stage 0 heatmap that places the hazard becomes the reference distribution
for the OOD signal.

**Structural signals may not track an arbitrary hazard.** Latent jerk measures smoothness and has no
relationship to a drawn box. The cleanest experiment optimises surprise and jerk with zero hazard
knowledge in the loop, then measures hazard entry against penalty-tuned CEM. The hazard is the ruler,
not the target.

## Fallback ladder

None of these require unfreezing the world model or training LeWM.

1. Structural signals alone reduce hazard entry. Strongest result.
2. They do not, but a probe on frozen latents does. Still a working dial, still no lambda tuning.
3. Neither works, and the finding is *why* structural signals fail to capture specified hazards.
   A genuine negative result.
