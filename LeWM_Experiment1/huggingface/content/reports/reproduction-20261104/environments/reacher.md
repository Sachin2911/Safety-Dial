# Reacher

[Dataset home and videos](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction) · [Experiment report](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/README.md) · [All setups](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/setups.md)

Control the arm to reach a recorded nearby goal. The primary setup uses the frozen released checkpoint, original CEM with
**10 iterations**, evaluation **seed 42**, goals 25 actions ahead and a 50-action window.
Primary result: **32/50 (64%)**.

## Watch success and failure

Each environment has one recorded success and one failure from its primary seed-42 panel. Press play below or use **Open video**. These five-second clips reconstruct 50 saved post-action states at 10 displayed actions per second. They are illustrative examples, not original recordings or real-time playback. Sign in with access to this private dataset to load the videos and images.

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

### Trajectories

Frames after actions 1, 25 and 50.

**Success**

![Reacher success: actions 1, 25 and 50](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/reacher/success-trajectory.png)

**Failure**

![Reacher failure: actions 1, 25 and 50](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/reacher/failure-trajectory.png)



The successful example has no retained goal image, as labelled in the clip. The original benchmark success flag is preserved.

## Results for every setup

| Environment | Condition | Seed | CEM iterations | Success | Saved evidence | Videos |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Reacher | Primary | 42 | 10 | 32/50 (64%) | [Cases](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/b7d0c0cbe56d0f454a9bff712d0bc2da2652d8c7/runs/repro-20261104-eval-reacher/artifacts/reacher-scientific-seed42-cem10/cases.json) · [Outcomes](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/b7d0c0cbe56d0f454a9bff712d0bc2da2652d8c7/runs/repro-20261104-eval-reacher/metrics/reacher-scientific-seed42-cem10/episodes.json) · [Summary](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/b7d0c0cbe56d0f454a9bff712d0bc2da2652d8c7/runs/repro-20261104-eval-reacher/metrics/reacher-scientific-seed42-cem10/summary.json) | [Success / failure](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/reacher.md) |
| Reacher | Repeat | 43 | 10 | 34/50 (68%) | [Cases](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/7342ebd59b2f4e1cb5aa3423cef9535dffdbaae0/runs/repro-20261104-eval-reacher-seed43-cem10/artifacts/reacher-scientific-seed43-cem10/cases.json) · [Outcomes](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/7342ebd59b2f4e1cb5aa3423cef9535dffdbaae0/runs/repro-20261104-eval-reacher-seed43-cem10/metrics/reacher-scientific-seed43-cem10/episodes.json) · [Summary](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/7342ebd59b2f4e1cb5aa3423cef9535dffdbaae0/runs/repro-20261104-eval-reacher-seed43-cem10/metrics/reacher-scientific-seed43-cem10/summary.json) | No clips produced |
| Reacher | Repeat | 44 | 10 | 43/50 (86%) | [Cases](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/d98e1a2254a4e1e944a2ed541027fd4048a264ac/runs/repro-20261104-eval-reacher-seed44-cem10/artifacts/reacher-scientific-seed44-cem10/cases.json) · [Outcomes](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/d98e1a2254a4e1e944a2ed541027fd4048a264ac/runs/repro-20261104-eval-reacher-seed44-cem10/metrics/reacher-scientific-seed44-cem10/episodes.json) · [Summary](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/d98e1a2254a4e1e944a2ed541027fd4048a264ac/runs/repro-20261104-eval-reacher-seed44-cem10/metrics/reacher-scientific-seed44-cem10/summary.json) | No clips produced |
| Reacher | Sensitivity | 42 | 30 | 39/50 (78%) | [Cases](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/c8de2b7b13f5d304c173e4d7688c23ae381cce42/runs/repro-20261104-eval-reacher-seed42-cem30/artifacts/reacher-scientific-seed42-cem30/cases.json) · [Outcomes](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/c8de2b7b13f5d304c173e4d7688c23ae381cce42/runs/repro-20261104-eval-reacher-seed42-cem30/metrics/reacher-scientific-seed42-cem30/episodes.json) · [Summary](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/c8de2b7b13f5d304c173e4d7688c23ae381cce42/runs/repro-20261104-eval-reacher-seed42-cem30/metrics/reacher-scientific-seed42-cem30/summary.json) | No clips produced |

## Trace the examples

The examples are the first success and first failure in saved case order, selected from the primary
panel. Success means the native benchmark condition was met at least once during the 50 actions;
the original evaluator continues stepping after first success. Case indices are zero-based.

[Media provenance and exact case IDs](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/media/reproduction-20261104/index.json) · [Original reconstruction records](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/a00abce4219395d4f8fbdd1d7fbcd28b211c05ab/runs/repro-20261104-media-06/artifacts/reacher) · [Media verification](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/media-receipt.json)
