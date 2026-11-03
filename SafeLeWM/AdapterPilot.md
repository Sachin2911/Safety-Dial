# Recurrent adapter mechanism pilot

Prospective pilot, 1 November 2026. Sachin authorized renting GPUs and implementing
the learned-adapter pilot, then steered provisioning to one eight-RTX-4090 Vast
machine. Replacement instance 58676387, machine 19781, quoted $3.6637/hour with
100 GB disk. Sachin explicitly approved replacing the stalled first rental at up to
$3.70/hour. Instance 58673041 was deleted and provider absence confirmed before
replacement; it ran no experiments. Failed RunPod requests created no rentals.
The existing L4 remains unchanged.
This protocol supersedes the previous pause for this pilot only. The immediate
architecture-transfer targets are TwoRoom, PushT and Cube; only TwoRoom runs here.

## Scope and limits

One implementation and one exploratory optimizer repeat, with native AP, untrained
adapter, evolved adapter and equal-budget parameter random search. No PPO run,
three-environment claim, optimizer-superiority claim, reward-penalty sweep or risk
certification. No repeated attempts based on unfavorable outcomes. Engineering fixes
before training require new source snapshots and full interaction accounting.

Target overall work time 2-4 hours. One run's wall-clock limit is 2 hours after
launch; maximum 592 simulator episodes/118,400 actions including engineering and
validation witnesses. No further campaign follows this pilot. Rental shutdown is
manual, as requested; the job deadline does not shut down the machine. At the quoted
rate, four replacement rental hours are about $14.65 plus network transfer; idle rental
and storage can continue until Sachin requests stopping.

## Controller

Freeze the AP checkpoint, model, action scaler, observation-only memory and CEM
recipe from the qualified baseline. Reuse all original memory exclusions and cache
bytes. Native actions are normalized to [-1,1]. AP replans every five controls from
the actual observed state, after any corrections to the preceding chunk.

The generic adapter has a 32-unit tanh feature layer, a 32-unit GRU and four linear
outputs in TwoRoom (two per action coordinate). Every network parameter is eligible
for learning; no handcrafted safe controller, waypoint route or expert weights are
embedded. The actor is evaluated at every executed control. Its memory resets at
each episode. Current visual features refresh when AP replans; causal proprioception
and action history refresh every control. No additional model queries are required.

For each normalized base action b, output gain g=1+tanh(v), residual r=tanh(u), and
execute clip(g*b+r,-1,1). Neutral output heads give g=1,r=0 exactly. These bounds
provide substantial authority, including slowing and redirection. They also permit
large overrides, which are logged rather than assumed to preserve base-policy skill.
Legal-action clipping is not a safety filter. The passive hazard monitor never
changes actions, observations, rewards or termination.

TwoRoom's input vector has 396 coordinates: separately layer-normalized current and
original-goal frozen embeddings (192+192), native agent position scaled to [-1,1]
(2), supplied hazard center relative to current position divided by 224 (2), the
18-unit exclusion radius divided by 224 (1), base action (2), previous executed
action (2), most recent native displacement divided by 5 (2), and elapsed fraction
of the 200-action cap (1). No true goal coordinates, future states, outcome flags,
scripted-route information or per-layout ID reach the network. The actor itself is
environment-independent; PushT/Cube observation bindings and dimensions remain
future work. This is architecture reuse, not same-weight transfer.

## Initialization, search and objective

Use generic seeded PyTorch initialization, seed 8301, with neutral output heads.
Search standardized coordinates z: theta=theta_neutral + fan_in_scale*z. All biases
and weights use the documented block fan-in scale; no successful coefficients are
supplied. Initial z=0 reproduces AP. Initial exploration sigma=0.5. The untrained
reference is one fixed N(0,0.5^2) draw with seed 8302, locked before any episodes.

Use pycma 4.4.4, diagonal covariance, population 8, four complete generations:
32 candidates, each evaluated on all eight training layouts, 256 episodes. Random
search uses 32 independent draws from the identical initial distribution, also 256
episodes. CMA seed 8303; random-search seed 8304. All candidates use the same
training planner seed 8201. Keep ties equal; record tied generations and randomize
presentation order to the optimizer with a separate seeded permutation. No evaluation
results are imputed. No incomplete candidate is passed to ask/tell.

These are equal episode and action caps, not guaranteed equal realized simulator
steps: successful episodes may terminate early. Report both actual totals and all
model work. This pilot cannot establish superiority under exactly matched interactions.

For this mechanism pilot, rank candidates lexicographically by (safe completions,
all completions, fewer violation episodes). This directly targets the desired
endpoint and prevents a stationary zero-risk controller outranking a task-completing
controller when neither has a safe completion. It introduces no weighted safety
penalty, clearance shaping or task-distance reward. With n=8 the equivalent integer
score is 81*safe + 9*complete - violations; these coefficients only encode the exact
lexicographic order. This is deliberately narrower than the prospective main
feasibility-first, fixed-risk-limit study; no risk-budget/dial claim follows here.
Select each search's best observed candidate using training scores only, breaking
exact ties by earliest candidate ID. Do not select the unevaluated CMA mean. Save
all candidates, scores, optimizer state and initial/random seeds.

## Panels and allocation

`configs/adapter-pilot/panel.json` freezes all inputs before the first reset.
Training: all eight previously qualified same-room layouts, including baseline
failures. They are known development inputs. Final pilot evaluation: four new
constructed same-family layouts, fixed before training, seeds 8401/8402/8403.
These are held out from adapter search and candidate selection, not claimed absent
from pretrained data or representative of arbitrary TwoRoom scenes. No evaluation
AP outcomes are read until both nominees are locked.

Eight GPUs on the same physical host run independent persistent workers. Every
candidate receives every training case, with the same seeds and base configuration.
Evaluation arms for each case/seed use the same device assignment. Verify native
parity across devices before training. Native and untrained references are evaluated
on the same four new layouts and three seeds as the two selected candidates:
4 arms * 4 layouts * 3 seeds = 48 evaluation episodes. All have a 200-action cap.

## Engineering checks and accounting

On the rented host only, check generic adapter dimensions, recurrent reset,
serialization, finite bounded outputs, neutral parity, nonzero influence, gradients
for future RL use, and optimizer ask/tell/checkpoint semantics. Synthetic software
checks have zero simulator/model interactions. Document every failure.

Native/neutral parity: all eight training layouts, two arms, 16 episodes.
Cross-device parity: repeat the first native case on the other seven GPUs, seven
additional episodes. This is 23 engineering episodes; reserve at most one additional
failed-start interaction within the 24-episode engineering cap. Verify states,
actions, observations and native task outcomes exactly, along with model work.

Run one privileged safe and one risky witness on each new evaluation layout, eight
episodes, to validate geometric feasibility only. Witness results do not change
panels or expose AP outcomes; a failed witness stops the pilot without replacements.
The planned total is 23+8+256+256+48=591 episodes (592 absolute cap). Report simulator
resets/actions, model batch calls, predicted blocks, encoder frames, adapter calls,
clipping, correction norms, engagement and elapsed time. Preserve failed starts,
partial traces and all discarded candidates; no local experiment computation.

## Interpretation and closeout

Report native, untrained, evolved and random-search safe completion, all completion,
violations and unfinished episodes together on all evaluation cases. Positive
results are exploratory, from one optimizer repeat and a small constructed panel.
Improvement over native/untrained establishes a candidate mechanism, not optimizer
superiority or transfer. Ties with random search must remain ties. A negative result
is retained and discussed before changing architecture, information or objective.
Independent saved-trace audits and private verified preservation precede closeout.

## Implementation and launch

The generic network is `experiments/helpers/recurrent_adapter.py`. The TwoRoom input
binding and exact ranking live in `experiments/helpers/pilot_common.py`.
`experiments/scripts/adapter_worker.py` owns one persistent AP model and memory per
GPU; `adapter_pilot.py` freezes sources, schedules checks and search, locks nominees,
then evaluates. `audit_adapter_pilot.py` independently checks saved trajectories,
resource counts, retrieval exclusions, candidate scores and optimizer checkpoints.

Fresh-host preparation uses `setup_adapter_host.sh` and the pinned public assets in
`fetch_adapter_assets.py`. Restore the already verified memory cache rather than
re-encoding it. Run `experiments/tests/test_recurrent_adapter.py` with the host's own
venv interpreter, save its output and create `software-checks-passed` only on success.
The runner rejects other instances, absent software-check confirmation and an
existing output directory. `adapter_pilot.py run` is a one-shot bounded job; it has
no provider stop or restart calls. Do not launch until runtime and source bindings
have been checked on the rented host.
