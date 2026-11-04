# Reinforcement learning and world models for manipulation

## What has changed in reinforcement learning and dexterous manipulation?

### Takeaway

Reinforcement learning can refine physical manipulation using outcomes and corrective experience. Recent systems combine it with demonstrations, simulation and conventional control rather than treating these as mutually exclusive alternatives. Capability remains tied to the evaluated tasks and transfer conditions. [HIL-SERL](https://arxiv.org/html/2410.21845v3); [Dexplore](https://arxiv.org/html/2509.09671v1).

### Cited Findings

- **HIL-SERL, 2024 preprint / 2025 Science Robotics:** combines a small demonstration set, a learned binary reward classifier, off-policy RL and human corrections. Demonstrated hardware tasks include motherboard assembly, bimanual handover, timing-belt assembly and dynamic manipulation. Most tasks trained within 1 to 2.5 hours; timing-belt assembly took six hours. The reported evaluation achieved 100/100 successes for individual tasks, with a separate 10-trial whole-IKEA-assembly test. These are controlled task evaluations, not universal reliability estimates. Contact tasks use an impedance controller with reference limiting, and humans correct undesirable training behaviour. The learned high-level policy is therefore one part of the safety arrangement. [Paper, Sections 3 and 4](https://arxiv.org/html/2410.21845v3); [author demonstrations](https://hil-serl.github.io/).
- **Dexplore, CoRL 2025:** uses human hand-object motion capture as flexible guidance for RL in simulation, allowing robot-specific motions rather than requiring exact copying of human finger movements. A distilled controller using depth and proprioception transfers to a physical robot hand. This illustrates the interaction between imitation, RL and sim-to-real transfer. The paper reports difficulty with small or thin objects, studies single-hand interactions, and acknowledges limited training diversity. It establishes dexterous task capability under its evaluation, rather than a broad safety guarantee. [Paper, including Section 6](https://arxiv.org/html/2509.09671v1); [official project and demos](https://research.nvidia.com/labs/srl/projects/dexplore/).

### Inferences

- RL and imitation are not cleanly separated camps: demonstrations can guide exploration, while outcome-based improvement goes beyond reproducing recorded actions. HIL-SERL and Dexplore make this overlap concrete.
- It is useful to ask where physical trial and error happens: on hardware with intervention, in simulation, or in a learned predictive model. Each moves rather than eliminates the engineering and safety burden.

### Gaps

- The cited task-success results do not measure every human-injury, object-damage or semantic hazard. Do not classify either method as globally safe or unsafe.
- These five selected exemplars form a conceptual map, not an exhaustive review of all results through 3 November 2026.

## What do predictive world models contribute, and where do they fail?

### Takeaway

A world model predicts consequences conditional on actions. An action policy selects actions. The two can be combined, but a vision-language-action policy does not become a predictive planning model simply by processing images and language. V-JEPA 2-AC illustrates online planning; UNISafe illustrates safety checking and learning fallback behaviour from imagined dynamics. [V-JEPA 2](https://arxiv.org/html/2506.09985v1); [UNISafe](https://arxiv.org/html/2505.00779v2).

### Cited Findings

- **V-JEPA 2-AC, June 2025:** a video-pretrained representation is extended with action-conditioned latent prediction using less than 62 hours of robot video. Planning against image goals enabled reaching, grasping and pick-and-place on Franka arms in two labs without collecting training data in those deployment environments. The authors explicitly report sensitivity to camera placement, compounding prediction errors, planning-search difficulty over long horizons, and reliance on image goals. They manually selected workable camera positions. This is a meaningful demonstration of predictive control and transfer, but its robot experiments do not establish general physical safety. [Paper, Sections 3, 4 and 4.3](https://arxiv.org/html/2506.09985v1).
- **The Planning Limits of Latent World Models, September 2026 preprint:** studies five frozen visual backbones on Meta-World tasks and BridgeData V2 interactions. In the studied setup, short imagined trajectories rank actions usefully only for relatively nearby goals. Importantly, a horizon mismatch remains even when simulator predictions are perfect, separating planning-design limitations from model prediction error. The authors also show a model can improve selection among actions proposed by a VLA. These are findings about those tested systems, not a theorem that every world model has the same limit. [Abstract and paper](https://arxiv.org/abs/2609.39235).

### Inferences

- Predictive quality, planning quality, task competence and safety are distinct. Better prediction can help, but the plan horizon, goal specification, candidate actions and represented information also matter.
- The model may support imagined training, online comparison of candidate plans, or checking a separate policy's proposals. These are different roles and should be stated explicitly when reading a paper.
- A plausible predicted future should not be called a certified counterfactual. The model can be wrong precisely on the unfamiliar actions that an optimizer considers attractive.

### Gaps

- No broad safety guarantee was established by the V-JEPA manipulation evaluations examined here.
- The September 2026 planning-limits paper is a recent preprint; preserve that status rather than presenting its findings as settled consensus.

## What changes when safety is an explicit learning objective or intervention?

### Takeaway

Safety-specific systems add hazard models, uncertainty estimates, fallback behaviour or action filtering. These make the safety claim more explicit, while introducing their own assumptions and failure modes. [UNISafe](https://arxiv.org/html/2505.00779v2).

### Cited Findings

- **UNISafe, CoRL 2025:** uses a latent world model, uncertainty estimation and reachability-based learning to filter a task policy. It evaluates simulated block plucking and hardware Jenga alongside a navigation benchmark. The aim is to avoid both known failures and unreliable imagined transitions. Its paper explicitly states that it does not guarantee zero system-level failures: its formal statistical assurance concerns a component of uncertainty detection. It assumes an in-distribution safe initial state and excludes unknown disturbances or visual distractions from that operating assumption. The authors discuss stopping when even the fallback is unreliable. Thus, it is evidence of explicit safety intervention and reduced failures in tested settings, with bounded assurances. [Paper, Sections 6, Limitations and Appendix B.4](https://arxiv.org/html/2505.00779v2); [official demonstrations](https://cmu-intentlab.github.io/UNISafe/).

### Inferences

- A useful answer to “what is safe?” identifies the protected property, operating conditions, enforcement mechanism and evidence. Calling an entire model family safe or unsafe hides these distinctions.
- HIL-SERL uses human and low-level control safeguards while learning competent skills; UNISafe explicitly learns an intervention around a task policy. These solve different parts of safe manipulation and neither removes all human specification.
- Safety can reside in hardware, low-level control, policy learning, runtime filtering or human supervision. A capability gain at one layer does not establish safety for the complete system.

### Gaps

- The cited studies do not establish that evolutionary optimization is preferable to RL or other search methods for the user's research objective. Selecting that role requires a separate literature and problem analysis.
