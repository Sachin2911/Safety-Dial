# OGBench Cube

[Dataset home and videos](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction) · [Experiment report](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/README.md) · [All setups](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/setups.md)

Control the OGBench single-cube task toward a recorded goal. The primary setup uses the frozen released checkpoint, original CEM with
**10 iterations**, evaluation **seed 42**, goals 25 actions ahead and a 50-action window.
Primary result: **36/50 (72%)**.

## Watch success and failure

Each environment has one recorded success and one failure from its primary seed-42 panel. Press play below or use **Open video**. These five-second clips reconstruct 50 saved post-action states at 10 displayed actions per second. They are illustrative examples, not original recordings or real-time playback. Sign in with access to this private dataset to load the videos and images.

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

### Trajectories

Frames after actions 1, 25 and 50.

**Success**

![OGBench Cube success: actions 1, 25 and 50](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/cube/success-trajectory.png)

**Failure**

![OGBench Cube failure: actions 1, 25 and 50](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/cube/failure-trajectory.png)



The successful example first met the native success condition at action 1. Its nearly stationary cube is shown at a physical plot scale.

## Results for every setup

| Environment | Condition | Seed | CEM iterations | Success | Saved evidence | Videos |
| --- | --- | ---: | ---: | ---: | --- | --- |
| OGBench Cube | Primary | 42 | 10 | 36/50 (72%) | [Cases](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/e3aa7017aa421ff7518af893a64ccb325bf9fb6b/runs/repro-20261104-eval-cube-seed42-cem10/artifacts/cube-scientific-seed42-cem10/cases.json) · [Outcomes](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/e3aa7017aa421ff7518af893a64ccb325bf9fb6b/runs/repro-20261104-eval-cube-seed42-cem10/metrics/cube-scientific-seed42-cem10/episodes.json) · [Summary](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/e3aa7017aa421ff7518af893a64ccb325bf9fb6b/runs/repro-20261104-eval-cube-seed42-cem10/metrics/cube-scientific-seed42-cem10/summary.json) | [Success / failure](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/cube.md) |
| OGBench Cube | Repeat | 43 | 10 | 42/50 (84%) | [Cases](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/ce1c2de4313478255dcd509a55e867b1ce041804/runs/repro-20261104-eval-cube-seed43-cem10/artifacts/cube-scientific-seed43-cem10/cases.json) · [Outcomes](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/ce1c2de4313478255dcd509a55e867b1ce041804/runs/repro-20261104-eval-cube-seed43-cem10/metrics/cube-scientific-seed43-cem10/episodes.json) · [Summary](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/ce1c2de4313478255dcd509a55e867b1ce041804/runs/repro-20261104-eval-cube-seed43-cem10/metrics/cube-scientific-seed43-cem10/summary.json) | No clips produced |
| OGBench Cube | Repeat | 44 | 10 | 32/50 (64%) | [Cases](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/f6fe3a6ba339c5eebaefea53db43d8d32bee52a0/runs/repro-20261104-eval-cube-seed44-cem10/artifacts/cube-scientific-seed44-cem10/cases.json) · [Outcomes](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/f6fe3a6ba339c5eebaefea53db43d8d32bee52a0/runs/repro-20261104-eval-cube-seed44-cem10/metrics/cube-scientific-seed44-cem10/episodes.json) · [Summary](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/f6fe3a6ba339c5eebaefea53db43d8d32bee52a0/runs/repro-20261104-eval-cube-seed44-cem10/metrics/cube-scientific-seed44-cem10/summary.json) | No clips produced |
| OGBench Cube | Sensitivity | 42 | 30 | 34/50 (68%) | [Cases](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/90b379572054bd9e6384b88cabea784e4875cc4b/runs/repro-20261104-eval-cube-seed42-cem30/artifacts/cube-scientific-seed42-cem30/cases.json) · [Outcomes](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/90b379572054bd9e6384b88cabea784e4875cc4b/runs/repro-20261104-eval-cube-seed42-cem30/metrics/cube-scientific-seed42-cem30/episodes.json) · [Summary](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/90b379572054bd9e6384b88cabea784e4875cc4b/runs/repro-20261104-eval-cube-seed42-cem30/metrics/cube-scientific-seed42-cem30/summary.json) | No clips produced |

## Trace the examples

The examples are the first success and first failure in saved case order, selected from the primary
panel. Success means the native benchmark condition was met at least once during the 50 actions;
the original evaluator continues stepping after first success. Case indices are zero-based.

[Media provenance and exact case IDs](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/media/reproduction-20261104/index.json) · [Original reconstruction records](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/a00abce4219395d4f8fbdd1d7fbcd28b211c05ab/runs/repro-20261104-media-06/artifacts/cube) · [Media verification](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/media-receipt.json)
