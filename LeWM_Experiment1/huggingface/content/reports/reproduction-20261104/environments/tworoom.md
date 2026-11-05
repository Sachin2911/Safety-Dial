# TwoRoom

[Dataset home and videos](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction) · [Experiment report](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/README.md) · [All setups](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/setups.md)

Navigate the two-room environment to a recorded nearby goal. The primary setup uses the frozen released checkpoint, original CEM with
**10 iterations**, evaluation **seed 42**, goals 25 actions ahead and a 50-action window.
Primary result: **45/50 (90%)**.

## Watch success and failure

Each environment has one recorded success and one failure from its primary seed-42 panel. Press play below or use **Open video**. These five-second clips reconstruct 50 saved post-action states at 10 displayed actions per second. They are illustrative examples, not original recordings or real-time playback. Sign in with access to this private dataset to load the videos and images.

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

### Trajectories

Frames after actions 1, 25 and 50.

**Success**

![TwoRoom success: actions 1, 25 and 50](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/tworoom/success-trajectory.png)

**Failure**

![TwoRoom failure: actions 1, 25 and 50](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/resolve/main/media/reproduction-20261104/tworoom/failure-trajectory.png)



Both examples reconstruct the saved positions directly.

## Results for every setup

| Environment | Condition | Seed | CEM iterations | Success | Saved evidence | Videos |
| --- | --- | ---: | ---: | ---: | --- | --- |
| TwoRoom | Primary | 42 | 10 | 45/50 (90%) | [Cases](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/1400c508aa326eb18bcda9ea7fea2b8050b80e41/runs/repro-20261104-eval-tworoom/artifacts/tworoom-scientific-seed42-cem10/cases.json) · [Outcomes](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/1400c508aa326eb18bcda9ea7fea2b8050b80e41/runs/repro-20261104-eval-tworoom/metrics/tworoom-scientific-seed42-cem10/episodes.json) · [Summary](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/1400c508aa326eb18bcda9ea7fea2b8050b80e41/runs/repro-20261104-eval-tworoom/metrics/tworoom-scientific-seed42-cem10/summary.json) | [Success / failure](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/tworoom.md) |
| TwoRoom | Repeat | 43 | 10 | 37/50 (74%) | [Cases](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/1400c508aa326eb18bcda9ea7fea2b8050b80e41/runs/repro-20261104-eval-tworoom/artifacts/tworoom-scientific-seed43-cem10/cases.json) · [Outcomes](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/1400c508aa326eb18bcda9ea7fea2b8050b80e41/runs/repro-20261104-eval-tworoom/metrics/tworoom-scientific-seed43-cem10/episodes.json) · [Summary](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/1400c508aa326eb18bcda9ea7fea2b8050b80e41/runs/repro-20261104-eval-tworoom/metrics/tworoom-scientific-seed43-cem10/summary.json) | No clips produced |
| TwoRoom | Repeat | 44 | 10 | 45/50 (90%) | [Cases](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/1400c508aa326eb18bcda9ea7fea2b8050b80e41/runs/repro-20261104-eval-tworoom/artifacts/tworoom-scientific-seed44-cem10/cases.json) · [Outcomes](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/1400c508aa326eb18bcda9ea7fea2b8050b80e41/runs/repro-20261104-eval-tworoom/metrics/tworoom-scientific-seed44-cem10/episodes.json) · [Summary](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/1400c508aa326eb18bcda9ea7fea2b8050b80e41/runs/repro-20261104-eval-tworoom/metrics/tworoom-scientific-seed44-cem10/summary.json) | No clips produced |
| TwoRoom | Sensitivity | 42 | 30 | 42/50 (84%) | [Cases](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/1400c508aa326eb18bcda9ea7fea2b8050b80e41/runs/repro-20261104-eval-tworoom/artifacts/tworoom-scientific-seed42-cem30/cases.json) · [Outcomes](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/1400c508aa326eb18bcda9ea7fea2b8050b80e41/runs/repro-20261104-eval-tworoom/metrics/tworoom-scientific-seed42-cem30/episodes.json) · [Summary](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/1400c508aa326eb18bcda9ea7fea2b8050b80e41/runs/repro-20261104-eval-tworoom/metrics/tworoom-scientific-seed42-cem30/summary.json) | No clips produced |

## Trace the examples

The examples are the first success and first failure in saved case order, selected from the primary
panel. Success means the native benchmark condition was met at least once during the 50 actions;
the original evaluator continues stepping after first success. Case indices are zero-based.

[Media provenance and exact case IDs](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/media/reproduction-20261104/index.json) · [Original reconstruction records](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/a00abce4219395d4f8fbdd1d7fbcd28b211c05ab/runs/repro-20261104-media-06/artifacts/tworoom) · [Media verification](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/media-receipt.json)
