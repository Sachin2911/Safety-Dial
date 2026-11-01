# LeWM proposal adapter: bounded development pilot

Frozen on 30 October 2026 before pilot episodes. The user authorized proceeding from
the completed single-episode integration smoke to a bounded training pilot. This is
a separate RunPod allocation with a fresh $1 cap and one-hour cutoff. The unavailable
RTX 2000 Ada request allocated no pod; the selected RTX 3090 costs $0.50 per hour.
Pod `8ldukx5cq1o6qo` must stop by 23:50:57 SAST. Provider credentials remain local.

The frozen protocol is [pilot.json](../../configs/lewm_proposal_20261030/pilot.json).
The existing 510-parameter network proposes one action sequence per replan, occupying
one existing candidate slot in each CEM iteration. The frozen LeWM supplies task
cost; the unchanged learned native-proprioception motion model checks explicit
virtual hazard geometry. The full 15 px pusher retains its 6 px margin. The monitor
observes actual dense physics and never overrides commands. Hazard graphics remain
outside policy images. Fourteen software checks passed on the rented GPU, including
the real planner injection path, safety refusal, completion-dominant fitness,
checkpoint recovery without duplicate jobs, and selection before validation.

The schedule contains 76 episodes: two initial-adapter references; four generations
of six candidates on two cases plus two contemporary native controls per generation;
six selection episodes comparing native, initial and training elite; and twelve
paired validation episodes comparing native with the frozen selected arm. Candidate
and native evaluations in each generation share planner seeds. Candidate order and
arm ordering are declared in advance. Initialisation, CMA, ordering, training,
selection and validation seeds are separate. A completed job is recovered from its
saved result rather than replayed after interruption.

Diagonal CMA-ES uses population six, four updates and initial sigma 0.15. Any panel
violation makes its fitness worse than every safe panel. Within safe panels,
completion dominates bounded goal progress and a small action-count tie-break.
Progress feedback uses privileged simulator task-state error only as a scalar
training reward; the adapter cannot read object or goal state directly. Future
optimizer comparisons must receive the same information. Selection chooses the
best training candidate rather than assuming the final distribution mean is best.
It then compares that candidate with native and the seeded initial adapter on two
new planner seeds, freezes the winning weights, and opens validation only afterward.
Native wins exact selection ties.

This pilot reuses one known source scene and three known virtual hazard layouts.
Validation seeds are reserved within this run, not unseen tasks. There is one
optimizer seed. Parameter random search, tuned fixed and ordinary-learning controls
have not run. A positive result could establish a development signal, not an
evolution-specific advantage or cross-scene generalisation. All unfinished episodes,
violations, engagement, discarded candidates, resets, failed starts, model queries
and elapsed time must remain explicit.

Each episode has at most 150 actions and 337,680 predictor steps. The schedule allows
at most two technical failed starts, or 78 starts total: at most 11,700 actions and
26,339,040 predictor steps including failures. A technical error stops for diagnosis;
refusal or safe non-completion is a valid outcome. No outcome-based retries or
extensions are permitted. Starting a new episode requires a ten-minute preservation
reserve before the cutoff. A local provider-stop guard and a credential-free remote
process guard enforce the lease. Evidence must be downloaded and verified before
deleting the exact pod.

The full schedule completed: 76 valid episodes, zero technical failed starts, four
real CMA-ES updates and 24 evaluated genotypes. All computation and fourteen
software checks ran on the rented host. The
[results receipt](../../progress/receipts/20261030-lewm-proposal-pilot-results.json)
records every stage and resource total.

On selection, native, initial and evolved elite each completed 1/2 episodes safely,
with 0/2 violations, 1/2 unfinished and 2/2 engaged. All three used 190 actions across
their two episodes. Native achieved better safe task progress in the unfinished
case and won the declared fitness comparison: 4.62548 versus 4.56938 for initial
and 4.57166 for elite. The training elite was generation 1, candidate 0; its weights
are retained independently of the selected native configuration. This pilot did
not establish a completion gain from the adapter.

Validation therefore repeated native in both arms. Each recorded 5/6 safe
completions, 0/6 violations, 1/6 unfinished and 6/6 engaged, using 358 actions.
These are repeated native evaluations, not validation of the evolved adapter and
not twelve independent cases. The evolved elite did not reach reserved validation.
No optimizer comparison or unseen-scene claim follows.

Across all training, controls, selection and validation episodes, the descriptive
totals are 47/76 safe completions, 0/76 observed violations, 29/76 unfinished and
76/76 engaged. These heterogeneous, correlated episodes are not a single efficacy
test. The run used 6,072 environment actions, 60,720 execution physics steps,
152 reset physics steps, 20,758 model batch calls, 4,599,868 model candidate rollouts
and 13,799,604 predictor steps. Episode runtime summed to 315.6 seconds; this
excludes setup, software checks, orchestration and preservation.

The adapter supplied 874 raw proposals, evaluated 13,110 times within existing CEM
slots. None was selected directly. This does not prove absence of influence through
the fitted CEM distribution. Saved-action inspection found multiple candidate
trajectories in four of the eight generation-by-case panels, and one shared
candidate trajectory in the other four. Candidate fitness was frequently identical.
Those observations motivate examining proposal influence before increasing the
training budget; they do not establish the cause of the negative result.

Independent audits replayed every saved proposal from its frozen weights and
features, checked dense safety and native goal outcomes, recomputed all candidate
fitness values, verified all 76 job identities and controller genotypes, and bound
selection and validation to the declared rule. All
[895 archived files](../../outputs/lewm-proposal-pilot-20261030/release/lewm-proposal-pilot-manifest.json)
were downloaded and hash-verified, with zero configured-credential matches. All six
[paired videos](../../outputs/lewm-proposal-pilot-20261030/remote/results/pilot-v1/media/provenance.json)
decoded successfully and the
[gallery](../../outputs/lewm-proposal-pilot-20261030/remote/results/pilot-v1/media/gallery.png)
was inspected. Original raw frames, source freezes, checkpoints, environment
records and the
[trained elite](../../outputs/lewm-proposal-pilot-20261030/remote/results/pilot-v1/controllers/training-elite.json)
are preserved locally. No external upload was performed.

Pod `8ldukx5cq1o6qo` was deleted after verification. Provider get and list both
confirmed absence at 23:08:09 SAST, and both shutdown guards retired. The
[closeout receipt](../../progress/receipts/20261030-lewm-proposal-pilot-pod-closeout.json)
records a conservative rental estimate of $0.145 under the separate $1 cap, not an
invoice. No other rental was changed. This completed pilot authorizes no extension
or further experiment.
