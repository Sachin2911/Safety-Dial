# Imitation learning and robot foundation models

## What developments changed learned manipulation capabilities?

### Takeaway

The useful progression is from learning individual demonstrated skills, to learning reusable visual-language-action policies, to composing skills and monitoring longer tasks. These are overlapping developments, not a replacement sequence or a ranking of safety. This synthesis is based on the primary examples below, checked against the 3 November 2026 cutoff.

### Cited Findings

- **ACT and ALOHA, RSS 2023:** ALOHA is a bimanual teleoperation platform; ACT is the imitation-learning algorithm. ACT predicts chunks of future actions from camera images and robot state. The authors demonstrate learned cup opening, battery insertion and related precision tasks. Their page includes both successful and failed responses to disturbances. Crucially, some other impressive ALOHA clips illustrate human teleoperation, not autonomous learned policies. The learned-policy results use a bounded object-position distribution. [Author project and demonstrations](https://tonyzhaozh.github.io/aloha/).
- **Diffusion Policy, RSS 2023 and IJRR 2024:** generates action sequences through conditional denoising, executes a portion, then observes and replans. This helps represent several plausible ways of acting without averaging them into an invalid intermediate action. Demonstrations include mug flipping, sauce pouring/spreading and Push-T. The paper evaluates task performance across twelve tasks from four benchmarks; this is a policy-generation advance, not a general safety guarantee. [Author project](https://diffusion-policy.cs.columbia.edu/).
- **OpenVLA, June 2024, revision September 2024:** an open 7B-parameter vision-language-action model trained on 970,000 robot episodes. It connects camera observations and language instructions directly to robot actions and supports adaptation to new setups. The authors show pick-and-place, wiping and other tasks; project videos explicitly distinguish zero-shot WidowX/Google-robot results from fine-tuned Franka results. [Author project](https://openvla.github.io/).
- **Physical Intelligence pi0.7, 16 April 2026:** the company reports one model performing laundry folding, espresso preparation, box building and new appliance-related tasks, with controlled tests of instruction and cross-embodiment generalization. It can be prompted with language, visual subgoals and strategy information. These are author-reported capabilities and controlled comparisons, not independent evidence of universal household autonomy. [Dated company research post and demos](https://www.pi.website/blog/pi07).
- Pi0.7 combines demonstrations, autonomous experience including failures, human interventions, human video and other data. It can distill behaviours from RL-trained specialist policies. This shows that imitation, foundation models and RL are increasingly combined rather than cleanly separated. [Technical paper](https://www.pi.website/download/pi07.pdf).
- **Gemini Robotics ER 2, 30 July 2026:** Google describes a higher-level model that watches video, tracks progress, orchestrates tools and hands motor execution to a lower-level VLA or robot API. Its official demonstration includes Spot fetching an object. The release evaluates orchestration with real VLA, simulated VLA and human teleoperation, which are distinct execution conditions. These are company-reported results. [Dated release](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/).
- **Gemini Robotics On-Device 2, 30 July 2026:** a separate local VLA whose inputs include images, language and robot state and whose outputs are numerical robot actions. The model card reports improvements on new platforms after post-training and makes it available to trusted testers. ER 2 is not itself the direct numerical action model. [Official model card](https://deepmind.google/models/model-cards/gemini-robotics-on-device-2/).

### Inferences

- For a reader, three axes are more informative than model size: the range of tasks and objects handled, precision during contact, and the length of a sequence completed without human rescue. Different papers improve different axes.
- Language competence makes task specification more flexible, but choosing the correct object does not establish safe contact with it. More instruction generalization is not automatically better force control.
- ALOHA and Diffusion Policy are accessible foundations; OpenVLA illustrates reusable pretrained policies; pi0.7 and the Gemini models illustrate the verified 2026 frontier. This is a selective map, not an exhaustive list or a cross-paper leaderboard.

### Gaps

- No comparable common evaluation across all these systems establishes a single winner in autonomous reliability or safety. Different robots, tasks, adaptation data and reporting conditions prevent a clean ranking from these sources.
- This review does not establish availability of pi0.7 weights or broad unrestricted access. Do not imply that every impressive model is a downloadable research baseline.

## What safety and reliability evidence actually exists?

### Takeaway

These sources support important capability advances, while their safety claims vary markedly. The strongest accurate characterization is property- and system-specific: a learned action generator, a reasoning safeguard, a physical controller and a hardware interlock contribute different things.

### Cited Findings

- The original OpenVLA paper explicitly states that it does not yet provide very high reliability, with typical success below 90% on the tested tasks. It also identifies single-image inputs and inference speed as limitations, with further work needed for high-frequency bimanual tasks. This is the 2024 model's limitation, not a statement about all later derivatives. [OpenVLA, Section 6](https://arxiv.org/html/2406.09246v3).
- Google's On-Device 2 card reports safety evaluation and red-teaming on specific embodiments, including collision rates, motion quality and out-of-distribution robustness. It also says the model has limitations on out-of-distribution tasks and high-degree-of-freedom robots. Evaluation primarily concerns standing bimanual manipulation; mobile and whole-body risks lie outside that scope. It recommends semantic reasoning, low-level motion/force/balance control and hardware safety mechanisms together. [On-Device 2 model card](https://deepmind.google/models/model-cards/gemini-robotics-on-device-2/).
- Google explicitly describes its human-distance stopping behaviour as ongoing research rather than a guaranteed safety-rated system. Its broader approach combines semantic, physical and operational safeguards, and its VLA can be combined with lower-level safety mechanisms. [Official robotics safety page](https://deepmind.google/models/gemini-robotics/responsibly-advancing-ai-and-robotics/).
- The ER 2 model card reports semantic-safety and human-safety-monitoring benchmark gains while also excluding safety-critical uses in its intended-use guidance. It defines ER 2 as a VLM producing text, with visual, spatial, temporal and tool-orchestration functions. Such evaluations should not be relabelled as proof that every composed robot system is safe. [ER 2 model card](https://deepmind.google/models/model-cards/gemini-robotics-er-2/).

### Inferences

- ACT, Diffusion Policy, OpenVLA and pi0.7 capability demonstrations do not establish a universal safety guarantee. That is a limit on the evidence, not proof that every execution is unsafe.
- A physical controller could prevent certain collisions while allowing a semantically hazardous task. A reasoning model could reject a hazardous instruction while the motor controller still drops an object. Safety therefore needs a claim about the assembled system and the particular hazards being measured.
- Task-success percentages alone cannot tell us whether failures were harmless, whether successful episodes contained violations, or how much human supervision was required.

### Gaps

- The retrieved pi0.7 post and paper concentrate on capability/generalization; this review did not identify a quantified general safety claim there. Searching those texts for safety terminology is insufficient to infer that the company's overall process lacks safeguards.
- Public company results here are not independent field deployment audits. Do not describe author demos as evidence of reliable unattended operation in arbitrary homes.

## Which demonstrations best explain the developments?

### Takeaway

Use one demonstration per conceptual step and inspect what the robot actually receives and controls. The best first page is ALOHA because it explicitly separates teleoperation, learned policies and failed disturbance trials.

### Cited Findings

- **Watch first:** the ALOHA page's learned-policy and reactiveness sections, then compare with its teleoperation section. The contrast makes the human's contribution visible. [ALOHA](https://tonyzhaozh.github.io/aloha/).
- **Then:** Diffusion Policy's Push-T and pouring examples illustrate continuous action generation and repeated replanning. [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/).
- **For the current frontier:** pi0.7's dated research page provides appliance, laundry and other examples alongside controlled evaluations. [Pi0.7](https://www.pi.website/blog/pi07).
- **For explicit separation of reasoning and action:** the July 2026 Gemini ER 2 release explains orchestration and demonstrates tool-driven fetching. Read its safety claims alongside the model card, rather than treating the demo as a complete system validation. [ER 2 release](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/).

### Inferences

- For every demo, ask: autonomous or teleoperated; familiar or held-out setting; task-specific adaptation or a shared model; number of attempts and interventions; safety outcomes separate from completion. These questions make the evidence comparable at a conceptual level even when numerical comparisons are inappropriate.

### Gaps

- Embedded videos were identified through their official project pages; no claim is made that every player was tested or every video watched end to end.
