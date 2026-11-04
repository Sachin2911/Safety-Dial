# Videos for controller adapter research

## Which videos should Sachin watch first?

### Takeaway

Start with the short MPC explanations, the original residual-control demonstration, and the World Models author talk. These connect directly to selecting action sequences, correcting an existing controller, and evolving a compact controller around learned representations. The recommendations below are ordered for this project, rather than by popularity.

### Cited Findings

1. **Why Use Model Predictive Control? | Understanding MPC, Part 1**. [YouTube](https://www.youtube.com/watch?v=8U0xiOkDcmw). Official MATLAB channel, published 15 May 2018. The official series lists 4:50. Introductory explanation of prediction, coupled controls and constraints; an accessible first watch before considering a candidate-ranking adapter. [Official series and duration](https://www.mathworks.com/videos/series/understanding-model-predictive-control.html).
2. **What is Model Predictive Control? | Understanding MPC, Part 2**. [YouTube](https://www.youtube.com/watch?v=cEWnixjNdzs). Official MATLAB channel, published 30 May 2018. The official series lists 6:06. Explains optimizing an action sequence, executing its first step and replanning. Particularly useful for locating where our ranking adapter could intervene. This teaches the planning foundation, not a learned adapter. [Official series and duration](https://www.mathworks.com/videos/series/understanding-model-predictive-control.html).
3. **Residual Reinforcement Learning for Robot Control**, authors' supplementary video. [YouTube](https://www.youtube.com/watch?v=RRllUka-r2A). Treat as a research demonstration, not a full lecture. The project combines conventional feedback control with a learned action correction and demonstrates real-world block assembly involving contact and unstable objects. It is especially relevant to the TwoRoom residual and our unsuccessful Block Plucking residual screen. The official project page embeds this exact video identifier. [Project](https://residualrl.github.io/) and [official page source containing the embed](https://github.com/residualrl/residualrl.github.io/blob/master/index.html).
4. **Recurrent World Models Facilitate Policy Evolution**. [YouTube talk](https://www.youtube.com/watch?v=HzA8LRqhujk). The authors' World Models site links this as its talk. Read alongside the interactive project article to understand the division between learned visual/memory representations and a compact evolved controller. It is a foundational conceptual relative of our work, not an experiment that adds safety to an existing frozen planner. [Author project page](https://worldmodels.github.io/) and [official source with the talk link](https://github.com/worldmodels/worldmodels.github.io/blob/master/draft_header.html).

### Inferences

- My suggested study order is MPC Parts 1 and 2, residual demonstration plus its paper, then World Models talk plus its interactive article. The two introductory MPC videos total about eleven minutes; other durations were not verified and should be omitted.
- For each resource, ask where the learned component sits, what inputs it receives, what it can change, and what remains fixed. This makes the connection to our different adapter interfaces explicit.

### Gaps

- These links and source descriptions were checked on 4 November 2026. Videos were not played end to end. Direct YouTube fetches intermittently failed; links for the residual demonstration and World Models talk are verified through official project pages/source. Do not claim full playback verification.
- The residual video's exact displayed YouTube title, channel name and runtime were not retrievable. The label above is the associated research title. World Models title was returned by a direct YouTube page, but channel and runtime were not exposed.

## Which videos explain safety filters and their assumptions?

### Takeaway

UC San Diego's Safety for Autonomous Systems course has two especially relevant guest lectures. They provide a contrasting way to modify an existing controller while making the assumptions behind safety claims explicit.

### Cited Findings

5. **Jason Choi: Introduction to Control Lyapunov Functions and Control Barrier Functions**. [YouTube](https://www.youtube.com/watch?v=_Tkn_Hzo4AA). Listed by Sylvia Herbert's Safe Autonomous Systems Lab as a MAE 207 guest lecture by Jason Choi, UC Berkeley. More mathematical than the MPC introductions. The official teaching page confirms the speaker, topic and exact link. [Official course page](https://sylvia-herbert-fs9e.squarespace.com/teaching).
6. **Lukas Hewing: Model Predictive Control and Predictive Safety Filters**. [YouTube](https://www.youtube.com/watch?v=Zx8QKSUqIW0). Listed in the same course as a guest lecture by Lukas Hewing, ETH Zurich. Useful follow-on for studying a layer that checks or modifies proposed commands through predictive control. [Official course page](https://sylvia-herbert-fs9e.squarespace.com/teaching).

### Inferences

- These are advanced follow-ons, not prerequisites to understanding our paper. For Choi, focus on how a safety requirement becomes a restriction on permissible controls. For Hewing, focus on the interface between the proposed controller action, predictive checks and the executed action.
- A safety filter and a learned adapter are different design choices. These lectures help us articulate the distinction; they do not imply that our adapter inherits any formal safety guarantee.

### Gaps

- The official course confirms lecture titles, speakers and video links; direct YouTube retrieval failed. Runtime, upload date, channel name and playback availability were not verified. Do not invent timestamps or claim a transcript was read.

## Is there an optional video about general adaptation across conditions?

### Takeaway

Rapid Motor Adaptation is an optional extension after the core six. Its adaptation module illustrates conditioning an existing policy through an inferred latent context, rather than directly correcting each action.

### Cited Findings

- **RMA: Rapid Motor Adaptation for Legged Robots**, method video: [YouTube](https://www.youtube.com/watch?v=qKF6dr_S-wQ), with a separate [results video](https://www.youtube.com/watch?v=nBy1piJrq1A). These direct video URLs were verified by the parallel transfer researcher through the [authors' project page](https://ashish-kmr.github.io/rma-legged-robots/). That researcher reported channel Legged Robots and publication date 1 July 2021. Primary content claims and exact architecture should be drawn from the transfer notes rather than inferred solely from the video titles.

### Inferences

- This is useful for asking whether an adapter should infer the current situation and condition a policy, rather than directly outputting corrections. It is not evidence that the same weights transfer from navigation to manipulation.

### Gaps

- Direct video playback and runtime were not verified. This is an optional extension, so the report can omit it to keep the core watchlist manageable.
