---
pretty_name: LeWMExp1 Baseline Reproduction
language:
  - en
tags:
  - robotics
  - world-models
  - reinforcement-learning
  - video
  - reproducibility
viewer: false
---
# LeWM Experiment 1 baseline reproduction

Results and videos from the frozen LeWorldModel checkpoint reproduction, completed on **4 November 2026**.
Original CEM planning was evaluated on TwoRoom, Reacher, PushT and OGBench Cube.
**Three environments met the practical competence check; Reacher fell short.**

[Experiment report](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/README.md) · [All setups](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/setups.md) · [One-page PDF](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/reproduction.pdf) · [Evidence guide](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/runs/README.md)

## Experiments

| Experiment | What was tested | Status | Browse |
| --- | --- | --- | --- |
| Frozen checkpoint reproduction | Four environments, three evaluation seeds, plus paired CEM sensitivity | Complete: 750 episodes, 15 audited panels | [Results and settings](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/README.md) · [Videos by environment](#videos-by-environment) |

## Videos by environment

Each environment has one recorded success and one failure from its primary seed-42 panel. Press play below or use **Open video**. These five-second clips reconstruct 50 saved post-action states at 10 displayed actions per second. They are illustrative examples, not original recordings or real-time playback. Sign in with access to this private dataset to load the videos and images.

[TwoRoom](#tworoom) · [Reacher](#reacher) · [PushT](#pusht) · [OGBench Cube](#ogbench-cube)

### TwoRoom

Navigate the two-room environment to a recorded nearby goal. Seed 42 · CEM 10 iterations · primary result **45/50 (90%)**.

<table>
<tr>
<td width="50%" valign="top">
<strong>Success</strong><br>
<video controls playsinline preload="none" width="100%" poster="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/tworoom/success-trajectory.png" src="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/tworoom/success.mp4" title="TwoRoom: recorded success"><a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/tworoom/success.mp4">Watch TwoRoom success</a></video>
<p>Case 0 · First success: action 19.<br><a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/tworoom/success.mp4">Open video</a> · <a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/tworoom/success-trajectory.png">View trajectory</a></p>
</td>
<td width="50%" valign="top">
<strong>Failure</strong><br>
<video controls playsinline preload="none" width="100%" poster="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/tworoom/failure-trajectory.png" src="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/tworoom/failure.mp4" title="TwoRoom: recorded failure"><a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/tworoom/failure.mp4">Watch TwoRoom failure</a></video>
<p>Case 3 · No success within 50 actions.<br><a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/tworoom/failure.mp4">Open video</a> · <a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/tworoom/failure-trajectory.png">View trajectory</a></p>
</td>
</tr>
</table>

Both examples reconstruct the saved positions directly.

[TwoRoom: setup, results and trajectories](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/tworoom.md)

### Reacher

Control the arm to reach a recorded nearby goal. Seed 42 · CEM 10 iterations · primary result **32/50 (64%)**.

<table>
<tr>
<td width="50%" valign="top">
<strong>Success</strong><br>
<video controls playsinline preload="none" width="100%" poster="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/reacher/success-trajectory.png" src="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/reacher/success.mp4" title="Reacher: recorded success"><a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/reacher/success.mp4">Watch Reacher success</a></video>
<p>Case 2 · First success: action 9.<br><a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/reacher/success.mp4">Open video</a> · <a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/reacher/success-trajectory.png">View trajectory</a></p>
</td>
<td width="50%" valign="top">
<strong>Failure</strong><br>
<video controls playsinline preload="none" width="100%" poster="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/reacher/failure-trajectory.png" src="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/reacher/failure.mp4" title="Reacher: recorded failure"><a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/reacher/failure.mp4">Watch Reacher failure</a></video>
<p>Case 0 · No success within 50 actions.<br><a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/reacher/failure.mp4">Open video</a> · <a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/reacher/failure-trajectory.png">View trajectory</a></p>
</td>
</tr>
</table>

The successful example has no retained goal image, as labelled in the clip. The original benchmark success flag is preserved.

[Reacher: setup, results and trajectories](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/reacher.md)

### PushT

Push the T-shaped block into the recorded target pose. Seed 42 · CEM 30 iterations · primary result **48/50 (96%)**.

<table>
<tr>
<td width="50%" valign="top">
<strong>Success</strong><br>
<video controls playsinline preload="none" width="100%" poster="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/pusht/success-trajectory.png" src="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/pusht/success.mp4" title="PushT: recorded success"><a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/pusht/success.mp4">Watch PushT success</a></video>
<p>Case 0 · First success: action 16.<br><a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/pusht/success.mp4">Open video</a> · <a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/pusht/success-trajectory.png">View trajectory</a></p>
</td>
<td width="50%" valign="top">
<strong>Failure</strong><br>
<video controls playsinline preload="none" width="100%" poster="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/pusht/failure-trajectory.png" src="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/pusht/failure.mp4" title="PushT: recorded failure"><a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/pusht/failure.mp4">Watch PushT failure</a></video>
<p>Case 15 · No success within 50 actions.<br><a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/pusht/failure.mp4">Open video</a> · <a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/pusht/failure-trajectory.png">View trajectory</a></p>
</td>
</tr>
</table>

Block poses are recorded; agent motion is reconstructed from the saved actions. All 100 saved rewards and termination flags match the reconstruction.

[PushT: setup, results and trajectories](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/pusht.md)

### OGBench Cube

Control the OGBench single-cube task toward a recorded goal. Seed 42 · CEM 10 iterations · primary result **36/50 (72%)**.

<table>
<tr>
<td width="50%" valign="top">
<strong>Success</strong><br>
<video controls playsinline preload="none" width="100%" poster="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/cube/success-trajectory.png" src="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/cube/success.mp4" title="OGBench Cube: recorded success"><a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/cube/success.mp4">Watch OGBench Cube success</a></video>
<p>Case 2 · First success: action 1.<br><a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/cube/success.mp4">Open video</a> · <a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/cube/success-trajectory.png">View trajectory</a></p>
</td>
<td width="50%" valign="top">
<strong>Failure</strong><br>
<video controls playsinline preload="none" width="100%" poster="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/cube/failure-trajectory.png" src="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/cube/failure.mp4" title="OGBench Cube: recorded failure"><a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/cube/failure.mp4">Watch OGBench Cube failure</a></video>
<p>Case 0 · No success within 50 actions.<br><a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/cube/failure.mp4">Open video</a> · <a href="https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/cube/failure-trajectory.png">View trajectory</a></p>
</td>
</tr>
</table>

The successful example first met the native success condition at action 1. Its nearly stationary cube is shown at a physical plot scale.

[OGBench Cube: setup, results and trajectories](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/cube.md)

## Results at a glance

| Environment | Paper reference | Primary seed 42 | Seed 43 | Seed 44 | Primary check |
| --- | ---: | ---: | ---: | ---: | --- |
| [TwoRoom](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/tworoom.md) | 87% | 45/50 (90%) | 37/50 (74%) | 45/50 (90%) | Within tolerance |
| [Reacher](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/reacher.md) | 86% | 32/50 (64%) | 34/50 (68%) | 43/50 (86%) | Below tolerance |
| [PushT](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/pusht.md) | 96% | 48/50 (96%) | 44/50 (88%) | 44/50 (88%) | Within tolerance |
| [OGBench Cube](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/cube.md) | 74% | 36/50 (72%) | 42/50 (84%) | 32/50 (64%) | Within tolerance |

The predeclared primary check allows a score at most 10 percentage points below the paper reference.
Each panel contains 50 cases, with goals 25 recorded actions ahead and 50 evaluation actions per case.
These results establish nearby-goal competence in three environments. They do not establish formal
equivalence, safety, long-horizon completion or replication across training seeds.

## Find your way around

| Folder | Contents |
| --- | --- |
| [reports/](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/README.md) | Experiment summaries, all setup results, protocols, PDF and audits |
| [media/](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/media/README.md) | The eight final clips and trajectory images, grouped by experiment and environment |
| [runs/](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/runs/README.md) | Original scientific, engineering and media run records with pinned source links |
| [infrastructure-checks/](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/main/infrastructure-checks) | Storage and transfer checks; separate from scientific results |

The gallery uses the final reviewed media. Original runs, failed attempts and earlier media packages
remain available through the evidence guide. [Media provenance](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/media/reproduction-20261104/index.json)
records the original revision and SHA-256 of every curated file.
