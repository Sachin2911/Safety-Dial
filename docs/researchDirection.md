# SafetyDial: which experience makes LeWM safer to use?

**Status: adopted research direction, 21 September 2026.** Sachin Mohan approved
this direction after the Push-T pilot, Phase 0 triage and targeted-experience review.
This is the authority for the research question, scope and experiment sequence.
It supersedes the 20 September planner-selection audit and earlier proposals.
Adoption does not imply supervisor approval, completed experiments or new trained weights.

Student: Sachin Mohan (2699183), BSc Honours Computer Science, University of the
Witwatersrand. Supervisor: Geraud Nangue Tasse. Project name: **SafetyDial**.
Working title: **Which Experience Makes LeWM Safer to Use?**

Start execution with the [pilot checklist](research/pilot.md). The
[checkpoint reference](research/checkpoints.md) records the asset audit, and
[related work](research/relatedWork.md) bounds the prospective contribution.
Completed [Push-T](../notes/safeCEM.md) and [Phase 0](../notes/phase0Report.md)
experiments remain evidence and reusable infrastructure.

**The adopted study keeps LeWM's visual representation fixed and tests which additional experience improves its safety-relevant predictions.** Start with the released Push-T checkpoint, first establish what physical information can already be read from it, then change only the experience used to adapt its dynamics predictor. This connects your interest in what JEPA encodes to a controllable experiment: distinguish information already available in observations from information the predictor fails to carry into imagined futures. It also reuses your strongest assets without first training locomotion. The main question is:

> **At the same interaction budget, which additional experience improves a released LeWM's predictions of unsafe outcomes, and does that improvement transfer to new safety rules?**

The main study should proceed only if the initial diagnosis finds a meaningful, repairable prediction error. A result that simple boundary sampling beats a sophisticated selector could be valuable. A result that only a new readout is needed would change the claim. We should let those outcomes determine the project rather than assume a new acquisition algorithm is necessary.

**“Encoding safety” needs a precise meaning.**

In Push-T, the model can encode physical information such as the T's position, orientation and motion. You supply the safety rule, for example “no part of the T may enter this region”. The experiment asks whether the model preserves enough information to answer that question. It does not test whether the model independently discovers what is dangerous.

There are three distinct claims. A readout trained on actual encoded observations measures information accessible to that readout. Applying the same readout to imagined futures measures how well the predictor carries that information forward. Updating the encoder would change the representation itself. **Improving a predictor while its encoder stays frozen cannot demonstrate that the encoder learned new safety features.** Likewise, a weak probe does not prove that information is absent: the readout, observation history or training coverage could be inadequate. FARM, posted 10 September 2026, already extracts failure information from frozen predictive states using supervised readouts, so “a JEPA latent contains failure information” alone is not a fresh contribution. ([FARM](https://arxiv.org/html/2609.11445v1))

The released checkpoint supports a study of **additional adaptation experience**. It cannot establish what caused its original representation to form without controlled pretraining comparisons. That broader question would be a separate, substantially larger study.

**What we can reuse, and what still needs building.**

The official release lists Push-T, Cube, TwoRooms and Reacher checkpoints. No compatible locomotion checkpoint is listed there. Start with Push-T; the others are possible later replications using their own models and action conventions, not zero-shot transfers of Push-T weights. ([Official LeWM repository](https://github.com/lucas-maes/le-wm))

- **Released Push-T LeWM:** a pretrained encoder and action-conditioned predictor, with 192-dimensional latents.
- **Existing project code:** penalty-CEM, Safe-CEM, pusher constraints, preprocessing and task metrics.
- **Existing probe implementation:** a pusher-position MLP, not a validated block-position/orientation readout.
- **Work still required:** block geometry and readout, verified branch replay, transition collection, acquisition scoring and a fine-tuning runner.

These distinctions come from the [checkpoint audit](research/checkpoints.md), [published configuration](https://huggingface.co/quentinll/lewm-pusht/blob/main/config.json) and [existing probe code](../experiments/helpers/probes.py).

The current local checkout lacks the weights, expert data and upstream source under the expected `data/` and `third_party/` paths. The public weights are about 72 MB, but the expert archive is 13.1 GB compressed. Existing scripts fit action/state normalisation from expert data; the inspected model repository does not include fitted scalers. Prefer recovering the historical checkpoint, source revision, scalers and a small replay subset from the previous experiment machine before downloading a large dataset. Installed packages alone do not establish that the old run is reproducible. ([Weight metadata](https://huggingface.co/quentinll/lewm-pusht/blob/main/weights.pt), [Dataset files](https://huggingface.co/datasets/quentinll/lewm-pusht/tree/main), [Model files](https://huggingface.co/quentinll/lewm-pusht/tree/main))

**The literature makes the contribution narrower, but still testable.**

WMPO already adapts a pretrained world model using base-policy rollouts because expert demonstrations underrepresent failures. AdaJEPA already compares predictor-only and joint adaptation, and varies data amount and diversity with fixed optimisation steps. Therefore, neither collecting failures nor comparing frozen and unfrozen components is sufficient novelty. ([WMPO, November 2025](https://arxiv.org/html/2511.09515v1), [AdaJEPA, June 2026](https://arxiv.org/html/2606.32026v1))

ReDRAW studies small latent-dynamics repairs and the experience that supports transfer; FOSP already varies safe/unsafe data balance and dataset size. Our prospective distinction is a controlled intervention on **which counterfactual action branches are acquired**, isolating its effect on false-safe decisions and transfer while the repair mechanism stays fixed. This search has not established that the exact comparison is absent everywhere. The result must explain which experience fixes which error, rather than merely show that more data helps. ([ReDRAW, April 2025](https://arxiv.org/html/2504.02252v1), [FOSP, ICLR 2025](https://arxiv.org/html/2407.04942v2))

**Experiment 0: make the physical question and replay trustworthy.**

Use the whole T-shaped footprint avoiding a virtual forbidden region as the main constraint. Predict block centre and `(sin θ, cos θ)`, then transform the simulator's actual shape polygons. A centre outside the region does not establish that the T is clear. Keep the pusher's action-space arena guard as a common engineering baseline. A virtual constraint changes the rule, not the physics; introducing a solid obstacle would be a different experiment.

Build layouts independently of each evaluated method, with both initial and target footprints clear. Use development cases with a demonstrated feasible route to define the layout generator, then freeze it. Do not remove test cases because a method fails, or create only tiny hazards positioned between prediction timestamps.

Replay is an unresolved dependency. The saved seven-number observation omits block linear/angular velocity, and the inspected `_set_state` path advances physics. It is sufficient for pose visualisation, not a verified arbitrary-root snapshot. Start from a fresh seeded reset and replay the recorded action prefix before branching. Verify repeated suffix outcomes, including contact. Charge those prefixes to interaction cost. Replace this with snapshot restoration only after equivalence is tested. ([Replay and geometry audit](research/checkpoints.md))

Preserve the released action convention: a block contains **five sequential two-dimensional commands**, not necessarily one repeated command. In the inspected 10 Hz setup, a block spans 0.5 seconds and five predicted blocks span 2.5 seconds. Record every environment transition inside each block. Compare dense truth with coarse true endpoints and interpolation, and inspect a few physics-substep contact traces. Environment-step logging is not continuous collision certification. ([Timing and action audit](research/checkpoints.md))

The output is a small replay report, geometry overlays and roughly ten checked contact traces. Stop here if replay or event definitions remain unreliable.

**Experiment 1: locate the untouched checkpoint's bottleneck.**

Create separate trajectory-level banks for probe training, development and final evaluation. Fit linear and small MLP block-pose readouts on real encoded observations, choose capacity on development data, then freeze the chosen readout. The historical pusher probe used random frame splitting, so its results do not replace this validation.

An illustrative screening budget is 20,000 probe-training frames and 24 evaluation roots with eight action tapes each, each tape covering five model steps. These are provisional sizes, not claims about sufficient statistical power. Include ordinary bounded perturbations around useful pushing and a separately labelled contact/rotation stress bank.

On identical executed actions compare dense simulator geometry, coarse true poses plus interpolation, the readout on actual future images, and that same readout on imagined latents. Measure centre, periodic-angle and polygon-clearance errors, then resulting hazard decisions. Stratify by contact, rotation, horizon and distance to the boundary. Include a stationary-block reference and a small coordinate-dynamics predictor, clearly labelling privileged simulator state or velocity inputs.

Good actual-image readout with poor imagination supports predictor repair. Poor readout first calls for better observation/probe checks. If temporal sampling explains the misses, resolve that before blaming representation learning. If a simple physical reference solves the problem, that is evidence against making the latent method central.

**Experiment 2: establish that extra experience can repair the error.**

Collect a modest, diverse bank of additional contact outcomes and action perturbations. Before developing a sophisticated selector, test whether these transitions improve the diagnosed failure at all. An error-enriched subset can test repairability, but every simulated candidate inspected to find those errors counts as a query.

The preferred main repair is **predictor-side adaptation**. Freeze the visual encoder and observation projector, including BatchNorm running statistics by keeping those modules in evaluation mode. Freeze preprocessing, scalers and the diagnostic physical readout. Cache detached target embeddings from new real observations. Adapt the action encoder, predictor and prediction projector using the latent-prediction objective, mixed with the same original-style replay subset for every arm. Strictly updating only the predictor is another possible setting, but choose one before the main comparison and name it accurately.

An illustrative replay mixture is half original-style and half acquired transitions. Fix one-step versus rollout training, optimiser settings and update count on development data. With frozen target features, their SIGReg term supplies no representation-learning gradient. The upstream end-to-end training script does not already implement this controlled adaptation experiment. ([Training implementation](https://raw.githubusercontent.com/lucas-maes/le-wm/main/train.py), [Module audit](research/checkpoints.md))

Include no update, a calibrated fixed margin and one cheap readout-only residual correction. These diagnose what needs changing; they are not the start of an architecture-by-selector factorial. If only readout correction helps, report better interpretation rather than repaired dynamics. If ordinary extra data does nothing after one reasonable optimisation check, pause acquisition development.

**Experiment 3: compare three ways of spending the same experience budget.**

Every arm starts from the same paid seed data and uses the same chosen repair. The core comparison is:

1. **Random supported experience:** sample ordinary nominal-plan perturbations from a common proposal generator.
2. **Predicted boundary coverage:** select branches near predicted polygon contact, balanced across roots and motion regimes.
3. **Learned optimistic-error risk:** train a small scorer on previously queried branches to predict false-safe events or optimistic clearance errors from available context and imagined trajectories.

The third arm cannot inspect an unqueried actual future. Its training data, scoring cost and update schedule are part of the method. Ensemble disagreement can be a later reference if affordable; it should not double the initial project. A selector ranked by true future errors is an oracle upper reference, not a deployable acquisition rule.

A provisional pilot grid is 128 common seed branches, then 64, 128, 256 and 512 additional branches, using matched prefix lengths and equal branch horizons. **The main prospective comparison uses equal charged simulator-step budgets, including prefix replay**, since equal branch counts can otherwise have unequal costs. Compare learning curves at common charged-step budgets and report branch counts alongside them. Count root-generation rollouts, reset-prefix execution, unsuccessful queries and discarded examples. Report model-query and training compute separately. A precomputed pool is useful for debugging training-set selection, but generating every outcome first does not demonstrate reduced simulator interaction. Prospective acquisition chooses branches before their outcomes are known.

At each data checkpoint, start adaptation from the same released predictor weights, with equal optimiser steps, replay ratio and paired training seed. This separates data choice from accumulated optimisation. Use the latest available adapted model to score the next acquisition batch, with rules frozen after development. Begin with one paired pilot run; confirm a useful effect with several acquisition seeds, then additional training seeds if making a training-robustness claim.

**Experiment 4: test what the repair actually learned.**

Use an untouched transfer grid crossing familiar versus held-out hazard layouts with familiar versus held-out starts/goals. Keep dynamics unchanged initially. Evaluate both representative branches and the separately declared stress bank. All descendants of a source trajectory belong to one split, and final-test outcomes never enter adaptive selection or calibration. New trajectory splits do not establish that all states were unseen during the released model's pretraining; that provenance is not fully known.

Relabelling one physical trajectory under several virtual hazards creates extra constraint labels, not extra simulator experience or independent episodes. Apply the same relabelling policy to every arm. Test ordinary motion and goal prediction too, so a predictor that simply forecasts less movement cannot pass as an improved model.

Report unsafe accepted plans divided by all accepted plans, with counts and acceptance rate; the fraction is undefined when nothing is accepted. Compare methods where acceptance and useful progress overlap, and group uncertainty by root or source trajectory. Preserve arena exits as bad outcomes. Report a composite of hazard violation or observation-domain exit, and identify censored hazard-only futures rather than calling unseen steps safe.

Continue if a data strategy produces a repeatable transferable improvement or reveals a clear failure of an apparently sensible strategy. Stop claiming targeted acquisition if gains vanish when all queries are charged, disappear on new roots, or merely reflect greater rejection. Random data matching the learned selector is a valid outcome, not a reason to keep adding complexity.

**Experiment 5: show the control consequence only after the offline result.**

Freeze all weights and compare repaired versus original Safe-CEM, fixed margins and one matched penalty-CEM reference, with nominal planning for context. Pair starts, goals and hazards, use equal candidate budgets and declare all-infeasible fallbacks. Audit the sequence actually returned and executed, including CEM averaging. Report block coverage, pose error, hazard exposure, arena failures and latency. A second released task is optional replication after its data and action interface are verified.

Encoder adaptation should remain a separate extension if the encoding gate reveals a plausible limitation. Each adapted encoder would need fresh equal-capacity probes trained on the same reference split, because an old probe can fail simply when feature coordinates move. Do not quietly fold that different representation study into the fixed-encoder acquisition comparison.

**The minimum next pilot is much smaller than the full trajectory.** Restore and fingerprint the previous assets, complete Experiments 0 and 1, then compare random versus boundary-selected data with one predictor-side repair at one modest budget. Produce a decomposition plot, a paired before/after table and representative successes and failures. Develop the learned acquisition scorer only if this exposes a repairable signal. A rough allocation is validity and diagnosis first, repairability next, then acquisition curves and transfer; GPU memory, collection speed and training time remain unmeasured. The first gate determines whether to proceed to the main acquisition comparison. The learned selector, closed-loop study and second task remain conditional extensions.


**Scope and planning horizon.** The minimum study is a reproducible diagnosis,
repairability test and equal-budget comparison of ordinary versus boundary-focused
experience. A learned error-risk selector follows only if that comparison exposes a
useful signal. Transfer is required for a claim beyond memorising acquired cases.
New locomotion training, irreversibility estimation, a new safety-filter architecture,
encoder adaptation and full-model pretraining are not core dependencies. SafetyDial
remains the project name; a conservatism dial is an evaluation tool, not the novelty claim.

The thesis target remains November 2026, with a possible ICLR 2027 workshop afterward.
Measure pilot runtime before committing to a larger schedule. Check the eventual
workshop's actual call; no paper deadline or acceptance is assumed.
