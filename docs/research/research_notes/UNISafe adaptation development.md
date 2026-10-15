# UNISafe adaptation development

**15 October 2026, ongoing.** Sachin authorized continued diagnosis, evolutionary
experiments and other environments if needed, with a solid empirical result as the goal.
This is an experiment log, not a completed result or a claim of evolutionary novelty.

The starting evidence is the
[matched release diagnosis](../reports/UNISafe%20release%20and%20switching%20diagnosis.md).
The task policy opens before extraction finishes; a hand-timed gripper hold prevents that
loss of progress in one episode. The original filter remains enabled. The next question is
whether an automatic, limited change to task behavior generalizes to unseen conditions.

## Fixed evaluation contract

The primary task retains the released compound success predicate, including greater than
7 cm middle/base XY separation, all physical failure predicates and the end-effector
boundary. The frozen safety filter retains snapshot 200000, uncertainty threshold 4.3,
value threshold 0.1 and released backup postprocessing. No pretrained weights are updated.
A task adapter changes the proposed gripper command before safety assessment; arm proposals
are unchanged. All actual violations and unfinished episodes remain visible in the results.

[The development protocol](../../../configs/unisafe/adaptation-study.json) separates
previously inspected seeds 0–19, training seeds 20–23, validation seeds 100–107, and reserved
test seeds 1000–1063. Reserved stress seeds 2000–2031 have not been used. Development may
change the representation; final candidates and implementation must be frozen before test
outcomes are viewed. Other environments remain an option if this task does not support a
meaningful study. Success is not defined as making evolution beat every comparator.

The bounded initial search compares elitist mutation, uniform random search and categorical
REINFORCE over the same task-adapter parameters. The latter is an episodic policy-gradient
comparison in this restricted controller family, not full Dreamer fine-tuning. Each method
receives the same initial population and evaluation cap. Candidates are evaluated under the
same filter; elite reevaluations and unsuccessful candidates all count. Selection utility is
1 minus a small completion-time penalty for success, -1 for failure or boundary exit, and
0 for unfinished episodes. Report actual control steps as well as the common episode cap.

A fixed-delay controller and the frozen task policy are essential controls. If a simple
rule performs as well as evolution, that is evidence about the adaptation problem, not
an evolutionary advantage. Sequential development runs randomize method order within seed
and retain initial states and camera hashes. Matching physical tensors alone does not make
their rendered inputs identical. Fresh-process matched checks are required for individual
causal claims.

## Specification and learned-head checks

The [paper's Table 12](https://arxiv.org/html/2505.00779v2#A4.T12) lists a success condition
without the extra 7 cm XY requirement in the released source. Recomputing the other released
geometry conditions on existing recordings finds that 18/20 filtered and 19/20 unfiltered
episodes satisfy that reduced predicate before their recorded termination. The same table
is not sufficient to recover the checkpoint's exact historical training environment. This
is a specification discrepancy and a possible explanatory factor, not proof that the
checkpoint was trained with a different target. It does not relabel the primary results.

A new 300-frame replay again matches original actions, safety values and uncertainty with
maximum error 0.0. The continuation head is approximately 1 around steps 40–50, including
the opening command at step 43. It therefore does not support an explanation in which the
model predicts termination there. The learned reward/continuation outputs are predictions;
they do not establish the actor's causal reason for opening. This offline diagnostic adds
600 reward-head and 600 continuation-head queries, plus the recorded replay query schedule,
68.47 seconds of measured runtime, zero simulator steps and zero training updates.

Raw evidence: `runs/unisafe/release-heads-20261015/`.

## Runner acceptance and the first negative control

With adaptation disabled, the new runner reproduces all 900 steps and full camera arrays
of the first three original episodes bitwise. Omitting deterministic decoder diagnostics
does not change the control trajectory. These 900 control steps and 4,500 physics steps
are counted engineering interactions, not new held-out performance evidence.

The first adapter delays opening after three consecutive closing proposals, using command
history alone. An eight-seed development pilot compares delays of 0, 10, 20 and 40 control
steps, with method order shuffled within each seed.

| Delay | Successes / 8 | Physical failures / 8 | Boundary exits / 8 | Timeouts / 8 |
| --- | ---: | ---: | ---: | ---: |
| 0, frozen control | 1 | 1 | 0 | 6 |
| 10 steps | 2 | 2 | 0 | 4 |
| 20 steps | 1 | 1 | 0 | 6 |
| 40 steps | 5 | 1 | 1 | 1 |

All 32 episodes and 7,484 control steps (37,420 physics steps) are included. The artifact
audit passes action transforms, source hashes, physical predicates and the fixed-filter
contract. These small, previously inspected development starts cannot support a general
safety or improvement claim. The longer delay has more successes but also a boundary exit.

The trace reveals a structural weakness: the policy sometimes closes its fingers around
empty space, then needs to reopen to acquire the block. Command duration alone cannot
reliably distinguish those attempts from a sustained grasp. Delaying every such opening
can prevent acquisition. Raw evidence: `runs/unisafe/adaptation-pilot-20261015/`.

## Gripper-aware adapter

The next version uses the measured finger joint positions as well as command history.
It requires a stable intermediate finger aperture during sustained closing before delaying
release. Fully closed empty fingers and rapid passage through an intermediate aperture do
not qualify. This adds an available proprioceptive signal to the adapter, not privileged
object geometry. Evolution, random search, policy gradient and the fixed-delay comparator
all receive the same inputs. The pretrained model's encoder inputs remain unchanged.

With a 20-step delay, three qualifying closing commands, a minimum 5 mm aperture per finger,
a maximum 25 mm aperture and less than 2 mm change between observations, the rule detects
the original seed-1 release at step 43 automatically. Its first 43 steps match the original
recording exactly, and its complete 56-step successful trajectory matches the earlier
hand-timed intervention bitwise. No episode-time trigger is supplied to this adapter.
The two warmup episodes also match exactly. This adds 656 control steps and 3,280 physics
steps, with the task completing at 2.80 seconds and no intervention or physical failure.
It is an acceptance case, not a generalization estimate.

Raw evidence: `runs/unisafe/adaptation-grasp-acceptance-20261015/acceptance.json`.
The gripper-aware development pilot is now complete:

| Delay | Successes / 8 | Physical failures / 8 | Boundary exits / 8 | Timeouts / 8 |
| --- | ---: | ---: | ---: | ---: |
| 0, frozen control | 0 | 1 | 0 | 7 |
| 10 steps | 5 | 1 | 0 | 2 |
| 20 steps | 7 | 0 | 0 | 1 |
| 40 steps | 6 | 2 | 0 | 0 |

This adds 32 episodes, 4,571 control steps and 22,855 physics steps. The artifact audit
passes. The seven successful 20-step-delay cases include two with one safety intervention;
most finish without an intervention. The remaining timeout never acquires qualifying grasp
evidence, so the adapter does not override its gripper. This is primarily a grasp-maintenance
result, not yet evidence of a filter-specific adaptation benefit. The unchanged-policy count
differs from the first pilot because preceding trajectories change renderer history. Both
pilots retain all outcomes; these development runs do not establish exact counterfactuals.

Raw evidence: `runs/unisafe/adaptation-grasp-pilot-20261015/`.
The initial optimization comparisons are complete below. Independent validation and reserved tests remain outstanding.

## Evaluation execution and selection

A concurrency trial started two fresh simulator processes while the pilot was running.
Their startup consumed CPU and slowed the active pilot considerably. Both were deliberately
stopped before creating an episode or making model queries. Their retained status records
confirm zero additional control steps and preserve elapsed startup time. This does not prove
that parallel Isaac execution is impossible. The container has a CPU quota equivalent to
7.68 cores despite exposing 32 CPUs; subsequent fresh-process evaluation uses one worker.

Validation will compare training finalists and fixed delays of 10, 20 and 40 steps on
separate seeds. The common selection rule first prefers candidates with no more observed
violations than the baseline on validation, then maximizes utility. If no candidate passes
that empirical screen, the least-violating candidate remains explicitly flagged. This is
not a safety guarantee. The final test plan freezes selected parameters and source hashes
before using reserved seeds. An unfiltered ablation for the selected evolutionary adapter
and baseline is planned on the first 32 reserved seeds, with shadow queries retained, to
distinguish a general task improvement from an interaction specific to the filter.


## Initial search comparison and fresh-process acceptance

The first search seed, 2026, is complete. Each method used three generations of four
candidates, each evaluated on the same four training starts: 48 episodes per method.
These are outcomes during optimization, not performance estimates of the final policies.

| Method | Successes / 48 | Physical failures | Boundary exits | Timeouts | Control steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Evolution | 32 | 5 | 1 | 10 | 5,360 |
| Random search | 28 | 5 | 2 | 13 | 5,837 |
| Categorical policy gradient | 31 | 3 | 1 | 13 | 5,631 |

All 144 episodes, 16,828 control steps and 84,140 physics steps pass the action, geometry,
filter and source-hash audit. The first generation has identical candidates and ordering
across the three methods. All 16 trajectories and 2,143 steps per method match bitwise,
including physical observations, camera hashes, model values and actions; only timing
fields differ. All three actual evaluations remain counted. Later repeated candidate
fitness varies with preceding trajectories and renderer history, which limits conclusions
from training fitness alone. The small development differences do not rank the optimizers.

A sequential fresh-process check on previously inspected seed 8 also passes. Baseline and
20-step grasp delay have identical physical states, camera hashes and actions until the
adapter first changes the gripper at step 32. The baseline times out after 300 steps; the
adapter succeeds after 51 steps. Both actual runs add 351 control steps and 1,755 physics
steps. This accepts the fresh-process evaluation procedure on this host, not arbitrary
snapshot restoration or a general success-rate claim. The stopped concurrency trial remains
separately recorded.

Before viewing validation or test outcomes, the
[replicated protocol](../../../configs/unisafe/adaptation-replicated.json) adds the fixed
control's 40-step and 5 mm settings to every optimizer's domain. Three new search RNG seeds
(2027, 2028, 2029), each with four generations of four candidates on four training starts,
provide 192 training episodes per method. Only their finalists enter validation; the 2026
runs remain separately counted development. Validation selects one deployment controller
per method. Reserved tests therefore compare selected controllers, not the average of all
optimizer replicas, and cannot by themselves establish broad optimizer superiority.
