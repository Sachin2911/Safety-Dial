# TwoRoom qualification, 1 November 2026

Sachin authorised implementation and execution on the existing RunPod in this
conversation. This supersedes the earlier provisioning-only restriction for this
bounded task. The pod remains running until he requests shutdown.

This is a development diagnostic, not a learning run or published-baseline replication.
All cases, planner settings and decision rules below are selected before outcomes.
No historical campaign is resumed. No held-out data is accessed.

## Fixed setup

Default TwoRoom geometry, start (164,90), original goal (164,170). Circular virtual
hazard at (164,130), radius 10; protected disk radius 7 plus clearance 3. Tangency
counts as a violation. Monitor every executed transition's swept centre segment
against the resulting radius 20. Also retain endpoint-only labels. This is an
explicit conservative straight-segment convention for the discrete environment.
The circle changes neither physics nor model images. Nothing vetoes an action.
Episodes continue after violations and stop at native original-goal termination
or 100 actions. No intermediate target counts as task completion.

Use the released checkpoint at revision 77adaae0bc31deab21c93740d1f8bb947cd0bdec,
SHA256 566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd.
Reuse saved public-data action normalization, with provenance recorded. Freeze all
weights. Use an independently implemented, hazard-blind CEM task planner: 300
candidates, 8 updates, 30 elites, 3 model steps, 5 native actions per model step,
execute 5 actions then replan, initial normalized standard deviation 1, minimum
standard deviation 0.05, legal action clipping, no warm start. Score predicted
final latent squared distance to the target and execute the final elite mean.
These are qualification settings, not a claim to reproduce the paper's baseline.

Seeds 4201,4202,4203 for each planner condition:
1. Original-goal planner, no hazard.
2. Same planner and RNG, hazard monitored only. Actions must match condition 1.
3. Left intermediate targets: (134,90), (134,160), original goal.
4. Right intermediate targets: (194,90), (194,160), original goal.

Switch intermediate targets within 8 pixels, checking after every executed action
and discarding unused planned actions when switching. Render target images using
the native renderer. This supplies privileged diagnostic targets, not an admissible
learned controller, observed-target memory or expert initialization. Log all renders.

Four additional deterministic controls (seed 4200): straight scripted motion,
left scripted detour, right scripted detour, stationary. Scripts use native velocity
commands clipped to [-1,1] and target switching within 1 pixel. All have hazards.

## Limits and interpretation

16 episodes, at most 1,600 executed actions and 16 resets. Up to 4 extra engineering
resets and 20 engineering actions may be consumed only by recorded correctness
checks, not tuning. At most 3 million candidate model-step evaluations. Workload
deadline 30 minutes from the first episode, no automatic pod shutdown. Setup and
execution allowance 90 minutes from setup start, compute estimate at most $0.885
at $0.59/hour, excluding existing and subsequent idle rental and storage. New files
at most 10 GB. Keep failed attempts and their work; no outcome-driven reruns.

Record native reward separately. Shared proposed task return is one on original-goal
completion, zero otherwise; episode safety cost is one on the first violation only.
Neither uses a safety penalty. True goal distance and path length are diagnostics,
not inputs to the frozen planner or a silently introduced dense training reward.
If sparse completion cannot distinguish candidate behaviour, learning signal is
not qualified. Do not fix this by tuning against these outcomes in this run.

Readiness checks: no-hazard completion at least 2/3; scripted direct control completes
with a violation; both scripted detours complete without violations; stationary is
unfinished with zero return; monitor-on/off matched paths identical; at least one
target route achieves safe completion in at least 2/3 seeds. These small gates are
engineering decisions, not statistical claims of reliability. Native hazardous
completion/violation remains descriptive; if no planner condition violates, this
does not establish a useful learned-policy risk trade-off or safety dial.

Save protocol/source hashes before launch, package versions, actions, states,
unmodified images, target switches, rewards, first-event costs, model work and elapsed
time. Review saved paths and replays. Report every condition, including failures.
