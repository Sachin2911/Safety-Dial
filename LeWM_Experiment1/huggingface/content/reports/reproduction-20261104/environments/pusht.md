# PushT

[Dataset home and videos](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction) · [Experiment report](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/README.md) · [All setups](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/setups.md)

Push the T-shaped block into the recorded target pose. The primary setup uses the frozen released checkpoint, original CEM with
**30 iterations**, evaluation **seed 42**, goals 25 actions ahead and a 50-action window.
Primary result: **48/50 (96%)**.

## Watch success and failure

Each environment has one recorded success and one failure from its primary seed-42 panel. Press play below or use **Open video**. These five-second clips reconstruct 50 saved post-action states at 10 displayed actions per second. They are illustrative examples, not original recordings or real-time playback. Sign in with access to this private dataset to load the videos and images.

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

### Trajectories

Frames after actions 1, 25 and 50.

**Success**

![PushT success: actions 1, 25 and 50](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/pusht/success-trajectory.png)

**Failure**

![PushT failure: actions 1, 25 and 50](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/pusht/failure-trajectory.png)



Block poses are recorded; agent motion is reconstructed from the saved actions. All 100 saved rewards and termination flags match the reconstruction.

## Results for every setup

| Environment | Condition | Seed | CEM iterations | Success | Saved evidence | Videos |
| --- | --- | ---: | ---: | ---: | --- | --- |
| PushT | Primary | 42 | 30 | 48/50 (96%) | [Cases](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/90afabbf3ad488e5005eab885879d8ac2ec186ba/runs/repro-20261104-eval-pusht-seed42-cem30/artifacts/pusht-scientific-seed42-cem30/cases.json) · [Outcomes](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/90afabbf3ad488e5005eab885879d8ac2ec186ba/runs/repro-20261104-eval-pusht-seed42-cem30/metrics/pusht-scientific-seed42-cem30/episodes.json) · [Summary](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/90afabbf3ad488e5005eab885879d8ac2ec186ba/runs/repro-20261104-eval-pusht-seed42-cem30/metrics/pusht-scientific-seed42-cem30/summary.json) | [Success / failure](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/pusht.md) |
| PushT | Repeat | 43 | 30 | 44/50 (88%) | [Cases](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/21c73823e35673c703d518e435a6741a92429ffa/runs/repro-20261104-eval-pusht-seed43-cem30/artifacts/pusht-scientific-seed43-cem30/cases.json) · [Outcomes](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/21c73823e35673c703d518e435a6741a92429ffa/runs/repro-20261104-eval-pusht-seed43-cem30/metrics/pusht-scientific-seed43-cem30/episodes.json) · [Summary](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/21c73823e35673c703d518e435a6741a92429ffa/runs/repro-20261104-eval-pusht-seed43-cem30/metrics/pusht-scientific-seed43-cem30/summary.json) | No clips produced |
| PushT | Repeat | 44 | 30 | 44/50 (88%) | [Cases](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/c5cb75fb30b13ba27cfcd963f75c6fd1ea4c31f9/runs/repro-20261104-eval-pusht-seed44-cem30/artifacts/pusht-scientific-seed44-cem30/cases.json) · [Outcomes](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/c5cb75fb30b13ba27cfcd963f75c6fd1ea4c31f9/runs/repro-20261104-eval-pusht-seed44-cem30/metrics/pusht-scientific-seed44-cem30/episodes.json) · [Summary](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/c5cb75fb30b13ba27cfcd963f75c6fd1ea4c31f9/runs/repro-20261104-eval-pusht-seed44-cem30/metrics/pusht-scientific-seed44-cem30/summary.json) | No clips produced |

## Trace the examples

The examples are the first success and first failure in saved case order, selected from the primary
panel. Success means the native benchmark condition was met at least once during the 50 actions;
the original evaluator continues stepping after first success. Case indices are zero-based.

[Media provenance and exact case IDs](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/media/reproduction-20261104/index.json) · [Original reconstruction records](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/a00abce4219395d4f8fbdd1d7fbcd28b211c05ab/runs/repro-20261104-media-06/artifacts/pusht) · [Media verification](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/media-receipt.json)
