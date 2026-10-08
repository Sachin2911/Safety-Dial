# Physical robustness in safe manipulation with learned world models

Research date: 8 October 2026. This note deepens the shortlisted robustness problem.
It distinguishes published mechanisms from our proposed research roles and does not
select a controller representation or claim that generic domain randomization is new.

## What changes, what can the robot know, and what can its world model represent?

### Takeaway

The promising question is bounded safe task completion under specified physical changes,
not unrestricted OOD generalization. The crucial scientific choice is whether the robot
must tolerate unknown dynamics, infer them through experience, or update its model.
Those choices need different information, baselines and interaction budgets.

### Cited Findings

UNISafe's hard block-plucking evaluation changes size, weight and friction. Table 15
reports 38% safe success, 31% failure and 31% incompletion for its diffusion-policy
combination. These are setting-specific outcomes, not an attribution of failures to the
world model. Appendix D.2 says entirely different tower configurations cannot be reliably
filtered. The table's evaluation denominator was not established in this audit; the
100 trajectories in its calibration table are not automatically evaluation trials.
[UNISafe](https://arxiv.org/html/2505.00779)

The source audit finds that normal/hard configurations also change initial arm pose and
outcome thresholds. Released reset events vary joints and block poses, rather than
providing a ready mass/friction sweep. Therefore toggling the two tasks is not a clean
physical-parameter intervention. The world-model interface takes images, end-effector
pose and recurrent action history, with no explicit mass/friction input.
[Normal configuration](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/takeoff/config/franka/takeoff_joint_pos_env_cfg.py);
[hard configuration](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/takeoff/config/franka/hard_takeoff_joint_pos_env_cfg.py);
[wrapper](https://github.com/CMU-IntentLab/UNISafe/blob/ab0fdf87f38edefa48b54a291ec5b7156c17b6e0/dreamer_wrapper.py)

DROPO provides a concrete identifiability warning: forward-pushing demonstrations weakly
identify lateral friction, and position-controlled pushing can weakly identify object
mass. Different parameter combinations can explain similar observed motion. The authors
still obtain useful transferred policies from distributions that do not exactly recover
the generating parameters. [DROPO, section 4.3.2](https://arxiv.org/html/2201.08434)

### Inferences

Use the following distinctions when specifying the problem. These are analytical
categories, not findings that every listed shift defeats UNISafe.

| Change | Main distinction for the research question |
| --- | --- |
| Mass, inertia, centre of mass | Often visually hidden; changes load and acceleration. An appearance change need not reveal the relevant parameter. |
| Friction | Directional, contact-dependent or spatially varying; a single episode-wide coefficient is a simplifying case. |
| Compliance/damping | Changes deformation and contact transients; a rigid-object simulator may not express the intended variation faithfully. |
| Object geometry | Alters contacts, reachability and images together; a harder task may become physically infeasible. |
| Initial object/robot pose | Primarily changes starting state and contact configuration, not necessarily the dynamics law. |
| Camera, lighting, background | Observation shift without necessarily changing physical dynamics; isolate from physical robustness. |

A constant unknown mass sampled at reset is different from an external push or a sudden
within-episode change. Deterministic parameter bias is different from stochastic forces
or sensor noise. Report these separately. Broadening a declared parameter interval is
bounded robustness; unseen mechanisms or geometries are stronger generalization claims.

Four adaptation types must remain distinct: a fixed feedforward policy can choose
behaviour tolerant of variation; a frozen recurrent network can update its memory;
explicit identification can estimate a parameter or belief and condition control;
gradient updates can change model or policy weights. Frozen weights do not imply frozen
knowledge. Conversely, recurrence does not establish that useful identification occurred.

A frozen predictor receiving the same latent state and action has the same conditional
prediction law regardless of an unobserved mass we wish to test. Scenario search can
evaluate actual simulator variants, explicitly conditioned models, or validated histories
that induce different beliefs. It cannot make an unconditioned model simulate arbitrary
physics merely by attaching parameter labels. Independent Gaussian latent noise is not
a validated substitute for a persistent friction change. Longer history can reveal
dynamics only when interactions are informative; longer prediction horizons can amplify
an undetected bias before observation feedback arrives.

### Gaps

It remains unverified which released states expose useful information about changed
physics before intervention is necessary. Readout error, transition-model error, actuator
behaviour and genuinely infeasible tasks must not be conflated. The companion source
audit supports interface claims, not executed counterfactual replay or robustness results.

## Which established methods are the closest comparators?

### Takeaway

Robust policy training, online identification, model adaptation, uncertainty-aware MPC
and evolutionary manipulation already have substantial precedent. Their information
requirements and safety meanings differ. The ten comparisons below are alternatives
to distinguish, not a requirement to implement everything.

### Cited Findings

**1. EPOpt and adversarial policy training.** EPOpt trains a state-conditioned policy
on the worst-return fraction of trajectories from sampled physics models, approximating
a CVaR objective. Its direct-transfer tests keep policy weights fixed; another stage
adapts the source parameter distribution using target data. Evidence includes Hopper and
HalfCheetah, not arm manipulation with separate damage constraints. Low-tail task return
is its robustness objective, not a general violation-probability guarantee. RARL is a
related adversarial-training precedent; adversarial disturbances and fixed parameter
uncertainty should not be treated as identical. EPOpt is ICLR 2017; RARL is ICML 2017.
[EPOpt](https://arxiv.org/pdf/1610.01283);
[RARL](https://proceedings.mlr.press/v70/pinto17a.html)

**2. Automatic domain randomization with recurrent manipulation policies.** OpenAI's
Rubik's Cube work trains under varied physical and observation conditions. At deployment,
fixed recurrent-policy weights update hidden state using observations rather than true
physical parameters. Its perturbation analysis resets memory, resamples dynamics during
execution and disables joints; adaptation includes a vulnerable adjustment period.
This directly establishes that memory-based adaptation in manipulation predates our
proposal. The principal measures concern successful manipulation and drops, not an
independently specified learned safety filter. [Paper, section 9](https://arxiv.org/html/1910.07113)

**3. Calibrate the simulator distribution, then train the policy.** SimOpt alternates
real rollouts, trajectory-discrepancy-based distribution updates and policy training,
demonstrating swing-peg insertion and drawer opening. DROPO instead uses offline
transitions and CMA-ES to fit distribution means and variances through likelihood
matching, then trains a task policy in the calibrated physics simulator. Its real
manipulation evidence includes puck sliding and Franka pushing. DROID previously used
CMA-ES with demonstrated joint-torque discrepancy for door opening. These methods search
physical-model distributions, not directly safety-filtered task-policy parameters.
Their transfer/success evidence should not be advertised as explicit hard-safety
guarantees. Runtime policy execution does not require receiving the true parameter
values from the experimenter.
[SimOpt](https://arxiv.org/html/1810.05687);
[DROPO, Robotics and Autonomous Systems 2023](https://arxiv.org/html/2201.08434);
[DROID, RA-L/ICRA 2021](https://arxiv.org/html/2102.11003)

**4. CMA-ES already improves compact manipulation policies.** Behavior Policy Learning
first learns a behaviour representation from state-only solution sketches, then
fine-tunes policy parameters with CMA-ES. Candidate evaluation averages sparse task
success across randomized starts and observation noise. Runtime control uses object and
end-effector information with a model-based low-level controller; the work includes
simulated manipulation and preliminary physical-robot transfer. This is a direct
precedent for evolutionary behaviour improvement, but its reported objective is task
success, not a separate safety constraint under hidden mass/friction shifts. The paper
explicitly allows other policy optimizers.
[Frontiers in Robotics and AI, 2022](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.974537/full)

**5. Adapt model weights or recurrent model state online.** GrBAL meta-trains dynamics
for gradient adaptation from recent transitions; ReBAL instead adapts recurrent state.
MPC acts through the adapted model without true physical-parameter input. GrBAL resets
to the learned initialization and adapts again at each control step. The planning horizon
is shorter than the adaptation-validity horizon. Real evidence uses a legged millirobot,
with simulation studies of changing dynamics; this is a model-adaptation comparator,
not established safe arm manipulation. Runtime computation and adaptation experience
must be charged. [Learning to Adapt, methods and experiments](https://arxiv.org/html/1803.11347)

**6. Robust constrained CEM with learned dynamics.** MPC-RCE fits a state-transition
ensemble and a binary cost classifier from environment interactions. It optimizes action
sequences using worst sampled trajectory cost, selects feasible elites first, and
executes one action before replanning. Models are trained again as the data buffer grows.
The stated predictor consumes state/action rather than a physical-parameter oracle.
Safety Gym results measure constraint costs; the authors acknowledge early training
violations. Sampled ensemble robustness is not exhaustive physical-parameter robustness,
and this is not a visual latent manipulation demonstration.
[2022 workshop paper, Algorithms 1-2](https://learn-to-race.org/workshop-sl4ad-icml2022/assets/papers/paper_16.pdf)

**7. Frozen latent models with adaptive safety assessment.** UNISafe trains recurrent
latent dynamics, a failure readout, an uncertainty ensemble and a safety policy/value;
runtime observation updates support filtering without explicit mass/friction input.
*When World Models Lie* adds observed latent prediction errors, an adaptively calibrated
error neighbourhood and pessimistic value optimization. The online change is to the
uncertainty assessment, not a demonstrated retraining of dynamics weights. It includes
simulated and physical manipulation, but acknowledges reactive response, calibration
mismatch and absence of formal safety guarantees. Large latent uncertainty sets can
include physically unrealizable states. Thus online error adaptation is already prior.
[UNISafe, CoRL 2025](https://arxiv.org/html/2505.00779);
[28 September 2026 preprint](https://arxiv.org/html/2609.34300)

**8. Physics-based safety evaluation under uncertain mass and friction.** A 2025
manipulation paper uses an explicit physics simulator, nominal trajectories and sparse
reevaluation of critical transitions under a parameter distribution. Safety quantities
refer to grasp slip and actuator load, requiring physical state/contact information.
Its simulated bimanual handover is relevant, but an important correction applies:
the authors lack a suitable fallback policy, and emulate probing by manually changing
the parameter distribution. It demonstrates an evaluation pipeline, not a complete
closed-loop system performing real probing. Sparse evaluation also assumes nominal
critical transitions remain informative after parameter changes.
[Paper, sections III-IV](https://arxiv.org/html/2509.12674)

**9. Explicit online dynamics identification for shielding.** Adaptive Shielding uses
pretrained function-encoder bases and recent transition data to infer dynamics
coefficients, combining this representation with safety-regularized learning and an
uncertainty-aware shield. Its experiments vary hidden physics per episode and include
Safety Gymnasium tasks; the inspected evidence is not a released arm benchmark. Some
comparison predictors receive true parameters, an explicit oracle advantage. Safety
bounds depend on assumptions about prediction error and observable safety information;
they are not assumption-free physical guarantees. A method requiring initial transition
context has a different information budget from a controller acting immediately.
[2025 arXiv manuscript, current version](https://arxiv.org/html/2506.11033)

**10. Risk-sensitive belief-space manipulation.** A 2026 dexterous-grasping manuscript
uses learned transition/observation updates for a Gaussian-mixture belief over contact
parameters, then optimizes a smooth CVaR grasp-quality objective. Runtime inputs include
pose estimates, tactile signals and joint information, rather than a visual RSSM alone.
Simulation varies friction, stiffness and mass; hardware tests address pose uncertainty.
The hardware quality measure is explicitly a tactile proxy rather than a force-closure
certificate. This demonstrates occupied territory around multimodal beliefs and
risk-sensitive action selection, with greater sensing/model assumptions than UNISafe.
Venue status was not independently established here.
[Variational Neural Belief Parameterizations](https://arxiv.org/html/2604.25897)

### Inferences

An ensemble's disagreement estimates something about its fitted predictors; it is not
automatically a calibrated posterior over friction or mass. Neither a CVaR return
objective nor success under domain randomization entails a chosen failure-rate bound.
The runtime information and the quantity called safety must accompany every comparison.

Quality-diversity can retain alternative behaviours, but diversity is not evidence of
robustness across physical conditions. Existing model-based repertoire work already
screens imagined candidates before expensive validation, so that mechanism alone is
occupied. [M-QD](https://arxiv.org/html/2008.04589);
[GuSS](https://arxiv.org/abs/2206.09743)

### Gaps

Most comparator evidence addresses either task robustness or safety; substantially fewer
examples match visual world models, manipulation and explicit safety simultaneously.
That mismatch is an opportunity for a careful comparison, not proof of an original method.
Acceptance was verified for the named published venues; unspecified newer manuscripts
remain manuscript evidence. No comparator implementation was executed.

## What could make a modest EA study scientifically useful?

### Takeaway

A defensible contribution would explain and improve a specific safe-completion weakness
under controlled physical variation. A standard EA is sufficient as a tool if the result
distinguishes behavioural robustness from better sensing, identification or model repair.
Merely optimizing average return over randomized masses would repeat established work.

### Cited Findings

A comparative adaptive-domain-randomization study already documents limitations on both
sides: online calibration depends on the current policy's useful data, while offline
approaches can fail when replaying open-loop trajectories. This supports accounting for
data collection and identifying what the chosen experience can reveal, rather than
assuming parameter estimation is solved. [ADR benchmark](https://arxiv.org/abs/2206.14661)

The adaptive latent-filter paper separately identifies nominal-policy difficulty after
safety intervention. That is a competing, narrower source of incompletion, not proof
that every failure under changed physics requires a new model or robust policy.
[When World Models Lie, V-B](https://arxiv.org/html/2609.34300)

### Inferences

**Contribution options.** One option is a controlled result on whether adapting task
behaviour can preserve safe completion while the world model and filter remain fixed.
Another is comparing ordinary robust-search objectives or controller families under a
declared safety requirement, identifying which failure mechanism each addresses. A
parameter-region benchmark that separates apparent safety from actual outcomes can also
be useful. These are potential result/benchmark contributions; a method claim needs a
specific mechanism beyond CMA-ES replacing another optimizer. Genotypes should follow the
diagnosed behaviour, not precede it. Adversarial scenario selection is already established;
discovering a bad mass setting alone is weak novelty.

**What the world model contributes.** It can provide perception, predictive features,
action evaluation or a reusable filter. Evolving a controller with this fixed interface
still uses a world model, even if candidate fitness comes from actual simulator outcomes.
If fitness comes mainly from imagination, model error can reverse candidate rankings;
robust-looking latent outcomes require independent physical validation. Changing real
simulator parameters tests actual simulated physics. Perturbing latents tests an assumed
model-error distribution unless the two are empirically linked.

**Measurement controls.** Hold safety definitions, initial-state generation and action
interfaces constant when varying physics. Separate training/calibration ranges, held-out
regions and unseen parameter combinations. Report safe completion, failure, incompletion
and intervention by region as well as overall, because frequent easy cases can dominate
an average. Check task feasibility at extreme settings and include simple cautious
behaviour so stopping is not mistaken for useful robustness. A privileged parameter-aware
reference helps locate information limits but must be labelled as an oracle.

Charge all simulator interactions, including identification probes, warm-up, discarded
candidates and model-training data; report model queries and computation separately.
Match information and budgets when comparing evolution with ordinary domain-randomized
fine-tuning, robust MPC or online adaptation. Repeated episodes and shared seeds reduce
some comparison noise, but do not create independent parameter regions. With zero failures
in independent Bernoulli episodes, the approximate 95% upper bound is 3/n, not zero risk.
Distinguish errors of the learned world model relative to its simulator from simulator
errors relative to physical hardware; a simulator-only study establishes the former.

**Conditional comparison with task continuation.** Robustness offers a controllable
intervention axis and strong evolutionary precedent, but demands clean parameter controls
and careful observability distinctions. Continuation is the narrower behavioural question
if stalling already occurs at nominal physics. Robustness becomes more attractive when
feasible physical variations reproducibly break safe completion and the cause can be
localized. If frozen-model beliefs never contain needed information, robustness can
expand into model/sensor adaptation and lose its low entry cost. Neither candidate
requires a complicated planning controller.

### Gaps

Searches covered EPOpt/CVaR, robust manipulation with CMA-ES, DROPO/DROID/SimOpt, recurrent
domain randomization, adversarial RL, learned robust MPC, online model adaptation and
risk-sensitive grasping, followed by primary-method checks. No exhaustive novelty claim
is made. Precise UNISafe evaluation denominators, useful hidden-parameter observability,
and clean runnable physics hooks remain unresolved. No experiments, installations,
rentals or data downloads were performed. Earlier notes overstated the demonstrated
probing capability of the physics-filter paper; the correction above supersedes that
interpretation while preserving the older files.
