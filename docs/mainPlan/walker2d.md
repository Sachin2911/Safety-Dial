# Parallel track: a LeWM trained on Safety-Gymnasium Walker2d

A second, controlled testbed for the same question. Shared definitions are in
[protocol.md](protocol.md). **Hard go/no-go on 18 October 2026.** This track is not a
dependency of the Push-T study; when the two compete for time, Push-T wins.

## Why this track, and why Walker2d

Training our own LeWM adds three things the released Push-T model cannot:

1. **Known training data.** Test states are genuinely unseen, and the pretraining data can
   be made expert-like on purpose (mostly competent behaviour, few near-failures), which is
   how released checkpoints are typically built. The question then becomes which added
   experience repairs the model's optimism about failures it rarely saw, and whether that
   experience helps more in pretraining or later (S5).
2. **Exact branching.** In Phase 0, restoring MuJoCo `(qpos, qvel)` and replaying the
   recorded actions reproduced `qpos` to 7.4e-15, and zeroing `qacc_warmstart` and `ctrl`
   made labels bit-reproducible. Branches need no replayed prefix.
3. **A benchmark safety cost** that safe-RL readers recognise, with published OmniSafe
   results for context.

Walker2d is chosen because Phase 0 already built and verified its infrastructure: the
vendored task matches upstream, state-only storage and lazy rendering feed
`stable_worldmodel` datasets, and throughput is measured. Phase 0 ruled Walker2d out as a
source of **irreversible** failures. It did not rule it out as a benchmark with supplied
rules, which is how it is used here.

**Documented alternatives, not scheduled:**

- **`SafetyPointGoal1`** is the closest parallel to Push-T (spatial hazards plus goal
  reaching), and its hazards are drawn in the image, so "can the model see the danger"
  becomes testable. SafeDreamer already trains world models on it from front and rear
  camera views. Here it is blocked by safety-gymnasium's pins (gymnasium 0.28.1,
  mujoco 2.3.3, pygame 2.1.0) and a Python 3.11 import crash in the navigation assets. It is
  good future work; switching to it now would need a 3-day time limit on setup.
- **`SafetyHalfCheetahVelocity`** never terminates, has long episodes and is a classic
  pixel world-model task, but speed is its only rule. It needs its own vendored `step()`.

## Pinned environment

- `safetydial/SafetyWalker2dVelocity-v1`, the vendored task in
  [locoEnv.py](../../experiments/helpers/locoEnv.py) (imported, never edited), on
  gymnasium's Walker2d-v4 physics.
- Control step 0.008 s (4 physics steps of 0.002 s). Actions: 6 torques in [-1, 1].
  Observation: 17 numbers, where `obs[0]` is torso height and `obs[1]` is pitch.
- 1000-step time limit. Benchmark termination is on for pretraining data, and off only for
  explicitly labelled diagnostic continuations and branch truth.
- Rendering: the default camera, which tracks the torso from a distance of 4 at an
  elevation of -20, rendered at 224 by 224 through EGL with the size fixed at `make()`. The
  floor is a checkerboard, so forward speed is visible only as floor motion between frames.

## The rules

Two supplied rules, evaluated at every environment step and reported separately:

1. **Speed, the benchmark's own cost:** unsafe if signed forward velocity exceeds
   2.3415 m/s (`v1`). Clearance is `2.3415 - v` in m/s. `x_velocity` is stored in float64,
   so the threshold can be changed without recollecting.
2. **Health, a supplied rule and not a claim of irreversibility:** unsafe if torso height
   leaves (0.8, 2.0) or pitch leaves (-1, 1) rad, the benchmark's own healthy band.
   Clearance is the smaller of the height and pitch margins, each divided by a declared
   scale. In Phase 0 this flag fired while the robot was still upright, so the rule flags
   excessive leaning as well as falling. Name it accordingly.

## Timing and observability

Push-T's frameskip of 5 would make a five-block horizon only 0.2 s here, shorter than a fall
(about 0.5 s in Phase 0). Default: **frameskip 10** (0.08 s per model step and 60-d action
blocks), history 3 (0.24 s of context), and an evaluation horizon of **10 blocks (0.8 s)**.
Frameskips of 5 and 20 are checked in S0.

Before training any LeWM, run a cheap observability check: regress speed, height and pitch
from stacks of three rendered frames at each candidate frameskip with a small CNN. If speed
cannot be recovered from frames, drop the speed rule rather than feeding velocity in, since
supplying the predicted quantity would change the information regime.

## S0: rules, timing and behaviour policies (week of 28 September)

- Fix the rules, frameskip, horizon and camera in a config.
- Train behaviour policies with OmniSafe in the isolated Python 3.10 environment.
  [AGENTS.md](../../AGENTS.md#engineering) allows policy weights from it, never datasets or
  reported numbers. Train **PPO** (fast, with frequent speed violations) and
  **PPO-Lagrangian** (speed-limited), saving checkpoints at intervals to get a ladder of
  competence and speed.
- Export each actor and its observation normaliser, then run them in the vendored
  environment through `TorchActorPolicy`
  ([locoPolicies.py](../../experiments/helpers/locoPolicies.py)). Measure return, speed
  distribution and fall rate there, not in the isolated environment.
- Upload the policy weights to Hugging Face. Start policy training on day one: it is CPU
  time that runs while E0 proceeds.

## S1: data (week of 5 October)

State-only HDF5 with lazy rendering, collected by a new script that imports
[locoCollect.py](../../experiments/helpers/locoCollect.py) rather than editing it.

| Set | Purpose | Composition (provisional) |
|---|---|---|
| **A** | LeWM-A pretraining, expert-like | About 3M steps from mid and late PPO and PPO-Lagrangian checkpoints spanning speeds around 2.34 m/s, action noise σ in {0, 0.05, 0.1}, benchmark termination on. Early checkpoints make up at most 10% of steps, so near-failures are rare but present |
| **Probe** | Readout training | Separate episodes covering the full range of pose and speed, including labelled continuations past health violations |
| **Roots** | Development and test branching | Held-out episodes of the set-A policies, plus a stress bank of roots within 0.5 s before a health violation or at speeds near the limit |
| **B** | LeWM-B pretraining (S5) | Collected in S5, once S4 has fixed its proposal generator |

- About 3M steps matches the scale of LeWM's own datasets (roughly 1M to 4M frames).
  Storage is about 0.8 GB, and collection takes minutes to an hour.
- Split whole episodes into pretraining, probe, development and test, and store the episode
  lists with the data. Test episodes never enter pretraining.
- Check label alignment on hand-inspected traces: the collector stores the state before
  each action, so the cost and health at row `t` belong to the transition caused by
  action `t`.
- Upload the data and split lists to a private Hugging Face dataset repository.

## S2: train LeWM-A on the 5090 (weeks of 5 and 12 October)

- A new entry point builds the lazy-rendering dataset
  ([locoData.py](../../experiments/helpers/locoData.py), imported) and runs the upstream
  LeWM training code at a pinned commit with a Walker2d data config: 224 px images,
  frameskip 10, history 3, and 6-d actions, which makes the action encoder input 60-d.
- Otherwise use the upstream defaults: SIGReg weight 0.1 with 1,024 projections, batch 128,
  10 epochs, and a linear warmup with cosine schedule. Log curves to W&B.
- **Smoke run first**, for 1,000 to 2,000 steps: measure iterations per second and render
  throughput, check the render fingerprint and that frames are not black (the EGL fork
  pitfall), and confirm that both loss terms fall.
- **Full run:** the upstream README reports a few hours on a single GPU; measure it rather
  than assuming it. Push intermediate checkpoints during the run, and the final checkpoint,
  config, fitted normalisers and manifest when it finishes.

## S3: go/no-go (18 October)

On real encoded frames, with trajectory-split linear and small MLP probes:

- **Pass** if torso height and pitch reach R² of at least 0.9 and speed at least 0.8
  (provisional thresholds), imagined errors grow measurably but not uselessly with horizon,
  and real-readout decisions track dense truth.
- If speed fails but health passes, continue with the health rule only.
- If health also fails, or training is still unstable after one fix, **stop the track** and
  write the outcome up as a limitation. Do not switch environments this late.

## S4: the minimum study on Walker2d (weeks of 19 and 26 October)

- The four-source decomposition for both rules: dense truth every 0.008 s, endpoint truth
  every 0.08 s, real readout and imagined readout, split by horizon, speed regime and
  distance to the health band.
- Predictor-side adaptation with the shared recipe; random versus boundary at two budgets
  (for example 128 and 512 additional branches), with three acquisition seeds.
- **Proposal generator:** the behaviour policy's own actions from the root, recorded as an
  open-loop tape, plus Gaussian perturbations at declared scales and short torque bursts
  for the stress bank.
- Branches restore `(qpos, qvel)` exactly and zero `qacc_warmstart` and `ctrl`, as Phase 0's
  oracle does, so the charged cost is the branch horizon plus root generation.

## S5 stretch: pretraining versus adaptation (weeks of 26 October and 2 November)

The question: does the same experience help more when it is in the pretraining data, where
it also shapes the encoder, than when it is added later to the predictor?

- **LeWM-B** is trained exactly like LeWM-A with the same number of steps, except that `N`
  ordinary steps are replaced by branches drawn the way S4's random arm draws them (same
  proposal generator, same root distribution). `N` equals the largest S4 budget in
  simulator steps. Collecting set B takes minutes; training takes a few hours.
- **Optional dose-response:** LeWM-B+ with about 25% of its steps from perturbed and
  near-failure episodes, only if the 5090 is otherwise idle.
- **Comparisons on the same test banks:** LeWM-A unadapted; LeWM-A plus random adaptation
  with `N` steps; LeWM-A plus boundary adaptation with `N` steps; LeWM-B unadapted; and
  LeWM-B plus boundary adaptation if time allows.
- Every model gets **fresh probes** trained with the same protocol and capacity. An old
  probe can fail simply because feature coordinates moved.

If LeWM-B beats LeWM-A plus adaptation, seeing failures during pretraining matters beyond
what the predictor can learn later, which is evidence about what the representation
encodes. If LeWM-A plus adaptation matches LeWM-B, later targeted experience is enough.
Report simulator steps for both routes.

## Risks and stop rules

| Risk | Early sign | Response |
|---|---|---|
| Behaviour policies too slow to train or too narrow | S0 checkpoints do not span speeds around 2.34 m/s | Add noisier and action-scaled rollouts; accept a narrower speed range and say so |
| Speed not visible from frames | S0 observability check | Keep the health rule only |
| LeWM does not learn locomotion dynamics | Smoke-run losses, S3 probes | One fix attempt, then stop at 18 October |
| Data loading starves the GPU | Smoke-run throughput | More render workers within the CPU quota, or pre-render a subset |
| The track crowds out Push-T | E0 or E1 slipping | Push-T wins; defer S4 and S5 |
