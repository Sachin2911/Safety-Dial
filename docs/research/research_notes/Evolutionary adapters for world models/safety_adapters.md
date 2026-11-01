# Safety aware evolutionary adapters

Research checked 30 October 2026. Scope: reusable controllers trained by evolution, their safety evidence, and the closest frozen-base residual architectures. These notes distinguish controller evolution from online action-sequence search and from tuning planner settings. They authorize no computation or rental.

## Have evolution trained reusable safety controllers or residual adapters worked?

### Takeaway

There are direct examples of safety-aware evolution of reusable neural controllers, including residual controllers around a fixed baseline. The strongest recent residual example also contains an important physical-transfer failure. None of the sources below establishes the exact proposed combination of a frozen LeWM, learned motion checker and hazard-conditioned proposal adapter.

### Cited Findings

- **Micro Neural Policies for Safe Real-Time Robotic Control**, Cao, Curcio, Ottaviano and Caccamo, arXiv preprint submitted 6 October 2026, explicitly studies ES-Res: a neural correction added to a fixed certified linear feedback controller. It uses safety-aware ES and statistical model checking, with Cartpole and Quadrotor state/action constraints. It is simulation-based policy synthesis, not training through a frozen learned world model. [Abstract and date](https://arxiv.org/abs/2610.08541), [full methods, sections IV-A and V-B](https://arxiv.org/html/2610.08541v1).
- In Quadrotor simulation, ES-Res verifies for **5/5 tested network architectures** at both 50 and 100 Hz; pure ES verifies for 3/5 and 4/5. These denominators count architectures, not independent trials or optimizer seeds. In physical Quadrotor tests, ES-Res **crashes at both frequencies**, whereas the selected pure-ES policy completes waypoint tracking. Authors attribute the residual failure to baseline model mismatch. Cartpole ES-Res also oscillates. Simulation verification therefore does not establish physical adapter safety. [Tables III and section VI-B](https://arxiv.org/html/2610.08541v1).
- **Synthesizing Safe Policies under Probabilistic Constraints with Reinforcement Learning and Bayesian Model Checking**, Belzner and Wirsing, arXiv 2020 and Science of Computer Programming 2021, introduces PSyCo/Safe Neural Evolutionary Strategies. Neural-policy perturbations are ranked using returns and constraint costs, with a Bayesian-confidence-dependent weighting. The study uses Particle Dance and Obstacle Run, not pretrained vision models. Obstacle Run has five repeats, 20,000 training episodes per repeat and verification every 1,000 episodes, capped at 1,000 verification episodes. Reported results satisfy specified probabilistic requirements; the paper also notes policies can still fail specification in the conflicting Particle Dance task. [Full methods and experiments](https://arxiv.org/html/2005.03898v2), [published article](https://doi.org/10.1016/j.scico.2021.102620).
- PSyCo releases experiment code. It needs an executable generative environment, not an analytically known transition model. Its statistical conclusions remain finite-horizon and the online confidence estimates use non-stationary policy-generated data, both explicit limitations. [Paper, sections 1 and 6](https://arxiv.org/html/2005.03898v2), [author code](https://github.com/lenzbelzner/psyco).

### Inferences

- These papers support the feasibility of evolving a small reusable controller under safety constraints. They do not establish that a frozen-base residual is always easier or safer than learning a controller directly. The MNP residual transfer failure is particularly relevant to any assumption that keeping the base fixed automatically preserves useful behaviour.
- Safety predicates and verification assumptions still have to be supplied. Evolution does not remove safety specification; PSyCo changes the balancing procedure rather than learning values without a cost or constraint definition.

### Gaps

- No primary public implementation for MNP was identified in its paper or the targeted repository search. Its cited predecessor is [Safe and Robust Neural Policy Learning with Statistical Verification for Sim-to-Real Deployment in Robotics](https://arxiv.org/abs/2608.06481), first submitted 6 August 2026 and revised 2 October. The predecessor was identified but not used here for additional performance claims.
- The physical MNP text describes selected challenging cases and trajectories, not a broad physical success-rate panel with a usable trial denominator. Do not turn its demonstrations into a general safety rate.

## Is there a working evolved controller that proposes behaviour to another controller?

### Takeaway

Behavior Policy Learning is a close reusable behavioural-proposal precedent: a neural policy outputs future controllable states for a fixed model-based low-level controller, and CMA-ES fine-tunes it. It does not use a learned latent world model or introduced hazard constraints.

### Cited Findings

- **Behavior policy learning: Learning multi-stage tasks via solution sketches and model-based controllers**, Tsinganos et al., Frontiers in Robotics and AI, 12 October 2022, learns a neural high-level policy from state-only demonstration sketches. Its output is a future target state followed by a fixed low-level model-based controller. Inputs include end-effector state and object locations/distances. CMA-ES then fine-tunes policy weights in simulation with randomized initial states and observation noise, using binary task success. [Primary paper, sections 2.2-2.4](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.974537/full).
- The perception-based simulated pick-and-place policies improve from **66, 64 and 60 percent** to **75, 82 and 76 percent** for the three selected imitation policies. The physical demonstration evaluates one optimized policy on 12 configurations and reports **8/12 task successes**. This is preliminary task-completion evidence, not a separately measured collision-free completion benchmark. [Table 5 and section 3.4](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.974537/full).
- The authors explicitly describe CMA-ES as small fine-tuning around an imitation initialization, not discovery of a new policy from scratch. Simulation interaction and demonstration sketches are required; this is not training entirely in the imagination of a pretrained world model. [Sections 2.4 and 4](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.974537/full).

### Inferences

- This is credible evidence that evolving a compact structured behavioural proposal can improve a composed controller and transfer a useful task behaviour to a physical robot.
- It argues for a restricted adapter output and a useful initialization, rather than evolving an unconstrained large action policy from scratch. Applying that lesson to LeWM is an architectural inference, not an already demonstrated result.

### Gaps

- No author code link was found in the inspected article. No newly supplied virtual hazard, learned-safety-checker or evolutionary-planner-refinement comparison is reported in this paper.

## What do frozen base safety and residual learning results establish?

### Takeaway

The recent VLA-Stem and VLaRL titles in this repository are externally verifiable papers. They demonstrate that frozen-base behavioural adapters can be useful, but both use gradient RL rather than evolution. They are adjacent feasibility evidence, not proof of an evolutionary or LeWM-specific advantage.

### Cited Findings

- **VLA-Stem**, Zou et al., arXiv preprint submitted 27 October 2026, keeps the VLA frozen, converts action chunks into motion references and trains a separate execution policy conditioned on reference trajectory, proprioception and explicit obstacle geometry. PPO, teacher-student distillation and a staged curriculum train it. It reports **52.8 percent mean collision-free task success versus 23.7 percent for the strongest safety baseline** across six constrained simulation settings. Each trained policy/task/scenario is evaluated over 3,200 episodes. Training costs are about **38.0M, 23.7M and 20.6M simulated transitions per seed** across its three stages, including teacher and student. These are not lightweight training costs merely because the deployed adapter is small. [Dated abstract](https://arxiv.org/abs/2610.37378), [methods and evaluation, sections III-IV](https://arxiv.org/html/2610.37378v1).
- VLA-Stem's trajectory-level guidance explicitly addresses corrections that preserve immediate safety but leave subsequent nominal actions misaligned. That mechanism is directly relevant to stalled recovery after intervention. It uses a simulator, privileged teacher information and obstacle observations; it does not evolve a proposal in a frozen learned world model. No primary code link was identified in the inspected paper. [Sections III-B and IV-A](https://arxiv.org/html/2610.37378v1).
- **VLaRL**, Saito et al., submitted 25 September 2026, keeps a task-fine-tuned VLA frozen while learning bounded additive residual actions conditioned on internal VLM latents, nominal action, proprioception and wrist force. A learned sim-to-real latent mapper is frozen during residual TD3 training. The actor is a four-layer width-1024 MLP, trained for 60,000 simulator steps with demonstrations and a privileged critic. Real conditions use 40 trials and show task gains, but the paper does not establish explicit hazard-constrained safety or use evolution. [Primary full paper, sections III-IV](https://arxiv.org/html/2609.30868v1).

### Inferences

- VLA-Stem is the strongest adjacent reference here for the actual goal of maintaining task completion while adapting a frozen base to new obstacles. BPL is the closer example of evolutionary fine-tuning of a behavioural proposal, and MNP is the closer safety-aware ES residual study.
- A defensible novelty statement must describe the particular adapter interface, search/training method and evaluation of the combination. These sources establish overlapping ingredients, not absence of earlier work on the exact combination.
- Training a proposal adapter entirely in imagination would additionally require a consistent closed-loop rollout of every adapter input. A task-predictive latent rollout and a path-only safety forecaster do not by themselves establish that capability.

### Gaps

- In this focused search, no primary paper was verified demonstrating all of the following together: a frozen LeWM-like latent predictor, a separately frozen learned motion safety checker, supplied novel hazards, an ES-trained reusable proposal adapter and refinement by an evolutionary receding-horizon planner. This is a search gap, not a claim of novelty.
- This review did not independently reproduce any published experiment. Figures without numeric tables were not converted into invented trial counts or safety guarantees.
