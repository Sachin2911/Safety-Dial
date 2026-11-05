# Frozen LeWM checkpoint reproduction

**Completed 4 November 2026.** The released frozen checkpoints and original CEM planner were tested
on nearby goals. TwoRoom, PushT and OGBench Cube met the predeclared practical tolerance; Reacher did not.

[Dataset home and videos](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction) · [One-page PDF](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/reproduction.pdf) · [Detailed results](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/results.md) · [All setups](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/setups.md)

## What was tested

| Setting | Value |
| --- | --- |
| Models | Released frozen LeWM checkpoints; original JEPA model |
| Controller | Original Cross-Entropy Method (CEM); 300 candidates, 30 elites |
| Goals | Recorded trajectory states exactly 25 actions after each start |
| Evaluation | 50 actions per case, with native benchmark success scoring |
| Primary comparison | 50 fixed cases per environment, evaluation seed 42 |
| Robustness repeats | 50 cases each for seeds 43 and 44, using the same checkpoint |
| Main CEM iterations | 10 for TwoRoom, Reacher and Cube; 30 for PushT |
| Sensitivity | CEM 30 on the same seed-42 cases for TwoRoom, Reacher and Cube |
| Total | 600 main episodes + 150 sensitivity episodes; 37,500 scientific actions |

## Results

| Environment | Paper reference | Primary seed 42 | Seed 43 | Seed 44 | Primary check |
| --- | ---: | ---: | ---: | ---: | --- |
| [TwoRoom](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/tworoom.md) | 87% | 45/50 (90%) | 37/50 (74%) | 45/50 (90%) | Within tolerance |
| [Reacher](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/reacher.md) | 86% | 32/50 (64%) | 34/50 (68%) | 43/50 (86%) | Below tolerance |
| [PushT](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/pusht.md) | 96% | 48/50 (96%) | 44/50 (88%) | 44/50 (88%) | Within tolerance |
| [OGBench Cube](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/cube.md) | 74% | 36/50 (72%) | 42/50 (84%) | 32/50 (64%) | Within tolerance |

The primary practical tolerance is no more than 10 percentage points below the paper reference.
It is an engineering criterion, not a formal equivalence test. Seeds 43 and 44 are evaluation repeats,
not independently trained models. Descriptive uncertainty intervals are in the detailed results.

The separate CEM-30 sensitivity scored TwoRoom **42/50 (84%)**, Reacher **39/50 (78%)** and Cube
**34/50 (68%)**. These results do not replace the primary comparison.

## Explore each environment

| Environment | Videos and trajectories | Primary setup |
| --- | --- | --- |
| TwoRoom | [Watch success and failure](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/tworoom.md) | Seed 42, CEM 10 |
| Reacher | [Watch success and failure](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/reacher.md) | Seed 42, CEM 10 |
| PushT | [Watch success and failure](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/pusht.md) | Seed 42, CEM 30 |
| OGBench Cube | [Watch success and failure](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/environments/cube.md) | Seed 42, CEM 10 |

Eight clips reconstruct saved trajectories, one success and failure per environment. No new evaluation
actions or model queries were used to produce them. Full reconstruction notes accompany each gallery.

## Evidence and limitations

All 15 scientific panels passed the saved evidence audit. Engineering used 19 reserved episodes or
attempts and 710 controller actions. Ten raw trace rows were lost in an early Reacher logging failure;
their action count remains in the ledger. No completed scientific panel was repeated.

Runtime differences include OSMesa rendering, a goal-index correction to obtain an actual 25-action
offset, released checkpoint history slots, and the documented MuJoCo compatibility overlay.
The baseline concerns nearby-goal completion; safety and long-horizon performance remain untested here.

[Prospective protocol](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/protocol.md) · [Runtime compatibility](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/compatibility.md) · [Full audit](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/audit.json) · [Media audit](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/media-audit.json) · [Evidence guide](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/runs/README.md)

The reproduction instance was destroyed after backup verification at Sachin's direction.
Recorded provider charges for the session were **$3.147**, including transfers and media work.
[Closeout](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/closeout.json) · [Billing receipt](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/billing.json)
