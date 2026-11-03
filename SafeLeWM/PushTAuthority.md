# PushT scene and adapter authority qualification

Prospective protocol, 2 November 2026. Sachin authorized the proposed bounded
PushT screen on the existing eight-RTX-4090 Vast instance 58676387, host
c45db82630e6. No new rental, optimizer training, Cube run or safety-dial study.
Rental shutdown remains manual. This fresh protocol does not resume an older run.

## Question and scope

Can fixed Anchored Planning and the recurrent action-adapter family support useful
safe task completion across distinct PushT starting scenes, without retrieving
sibling demonstrations from those scenes? The earlier hazard screen used one
starting-scene group and is historical development evidence, not replication.

Freeze this protocol, code, panel and parameter draws before the first episode.
Use the public PushT data and checkpoint pinned by fetch_manip_data.py. Download
once if absent. Use a separate simulator dependency overlay; do not modify existing
TwoRoom evidence, models, caches or active PPO jobs. All computation runs on Vast.

## Cases and retrieval memory

Use the existing on-path island recipe in helpers/manip.py, unchanged. The protected
body is the pusher disc only: 15 px radius, 12 px island, 6 px margin, hence 33 px
centre exclusion. Swept segment contact, including tangency, labels a violation;
the monitor never changes actions. The T-block is not protected. Original joint
pusher/block position and angle success is required; no proxy reaching task.

The data contain 101-episode starting-scene groups. Verify first-state equality
within each group (absolute tolerance 1e-4); stop on a grouping mismatch. Retrieval
uses the first 8,000 episodes with more than five rows, restricted to groups 0-79.
No new query group may occur in memory. Encode only those memory rows; keep the
original dataset row indices and report all encoder work.

Exclude every group used by any prior manipulation-screen case (long, hazard or
witness development). Scan remaining groups numbered 81 and above in ascending
order and episodes within each group in ascending order. Select the first eligible
on-path layout per group, at the episode's first row with goal 140 rows later.
Reject initially successful cases and reuse the existing geometric feasibility
recipe. Between selected starts require Euclidean distance of the first four state
coordinates at least 40 px, or wrapped block-angle difference at least pi/9.
Take the first eight such groups. If fewer than eight exist, stop and report the
shortfall; do not relax the recipe or replace cases after rollout outcomes.
These are development cases, not a final generalization test or a claim of absence
from pretrained LeWM data. Store rejected groups and selection reasons.

## Fixed planner and adapter

Retain the prior PushT AP settings: released frozen checkpoint, full-dataset action
scaler, observed five-row successor, top-one retrieval, remaining horizon max(5,
140-t), 300 CEM candidates, 30 iterations, 30 elites, standardized-zero prior,
standard deviation 1/3, five executed actions and 280-action episode cap.
Legal normalized actions are in [-1,1]^2; target = current pusher position +
100*action. Hazard geometry is supplied separately, not drawn into images.

Reuse RecurrentActionAdapter: tanh feature layer 32, GRU32, four output logits for
two gains and two bounded residuals, with clipping only at native action limits.
The feature vector has 402 entries: standardized current and goal latents (192
each), pusher XY /256 -1, block-minus-pusher XY /512, block sin/cos angle,
pusher velocity /100, hazard-minus-pusher XY /512, exclusion radius /512, base
command, previous executed command, last pusher displacement /100 and time /280.
Current latent refreshes with each five-action replan, as in TwoRoom; current native
state and history refresh every action. Goal coordinates, witness routes and costs
to go are never actor inputs. This changes mechanical task features, not the actor
family. New weights are trained per environment in any future study; no weight
transfer is claimed.

Freeze generic initialization seed 13101, neutral zero output heads and fan-in
coordinate scaling. Draw sixteen independent 132-dimensional head-coordinate
vectors from N(0, 0.5^2), RNG seed 13103, before outcomes. Other weights stay fixed.
These are authority probes, not evolution, a winning-policy claim or a tuning sweep.
There is no adaptation of parameters based on outcomes.

## Stages and declared gates

Planner seeds 13201 and 13202, paired across all arms. Assign each scene to a fixed
GPU (scene index 0-7) for every main comparison.

1. Software checks and one cross-GPU native reference per GPU on scene 0 (8 episodes,
   seed 13201). Require bitwise equal states, actions and frames across GPUs.
2. For each of eight scenes: replay risky witness and scripted safe witness under
   seed 13201; clean AP and monitored AP under both seeds; neutral adapter under
   both seeds (64 episodes). Witnesses use privileged state only as feasibility
   diagnostics. Preserve any failure; a failed witness is not a safe route.
   Require clean/monitored and monitored/neutral bitwise equality in states,
   actions, frames and rewards on all sixteen pairs.
3. Admit a scene for interpretation only if its safe witness completes without
   violation or invalid scripted contact, its risky witness completes with violation,
   and native AP completes under at least one seed. Report all eight scenes anyway.
   Continue to the parameter screen only with at least four admitted scenes,
   native completion at least 8/16, and native violations in at least two admitted
   scenes. These are engineering gates, not statistical claims. No waived gate.
4. If qualified, evaluate all sixteen pre-drawn adapters on all eight scenes and
   both seeds (256 episodes). Do not remove failed-admission scenes from this run;
   report complete-panel and admitted-subset results separately. Report each
   genotype's safe completion, all completion, violations, unfinished, engagement,
   action changes, clipping and candidate spread. Useful safe authority requires
   at least a two-episode spread in safe completions among probes, safe completion
   on at least two admitted scenes by the probes collectively, and at least one
   probe exceeding neutral safe completion by two episodes on the admitted subset.
   Report action authority even if useful safe authority fails. No training follows
   automatically from a pass.

## Budgets, failure handling and preservation

At most 328 scheduled resets and 91,840 actions (all 280-action ceilings). One
engineering-failure reserve adds at most two resets and 560 actions, only after a
recorded implementation correction, never as a favorable-outcome retry. Absolute
cap 330 resets and 92,400 attempted actions. Stop dispatch on a worker error;
retain every partial episode, and charge any in-flight work. No automatic restart.
Four-hour active-work deadline from asset preparation bounds preparation plus
execution; at the quoted $3.6637/hour this is at most $14.6548 of active rental
use, separate from already-authorized idle time. The deadline stops experiment
work, never the rental. Target execution under one hour once assets are ready.
At most 60 GB new disk including the 46.3 GB public HDF5 and cache; refuse launch
without enough space. Audit, packaging and transfer add no simulator interactions.

Audit source/asset bindings, cluster separation, native success, swept contacts,
neutral/monitor/cross-GPU parity, exact actor replay and feature construction,
retrieval membership, every attempted action and model call. Save source freezes,
initialization, all parameter probes, job specifications, planning logs and traces.
Report gates and negative outcomes without retuning. Preserve an independently
hash-verified private local evidence archive and a concise result record. No new
private cloud upload or rental shutdown is implied.
