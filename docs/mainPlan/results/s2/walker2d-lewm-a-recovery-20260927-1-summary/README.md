# walker2d-lewm-a-recovery-20260927-1: finalized training summary

The saved model reached **214,780 optimizer updates across 10 epochs**. The full history contains 10,739 records at 20-update intervals. This document covers training evidence only; S3/S4 outcomes are outside its scope.

![Saved losses and learning-rate schedule](learning_curves.png)

Recovery restored the atomic bundle at **step 172,000**, then executed 42,780 updates to the same 214,780-step horizon. The original worker had logged through 174,400, so **2,400 already logged updates were repeated after rollback**. This duplicate compute is separate from the final optimizer-step index; no unlogged tail is inferred.

Final saved validation prediction loss: **0.0039440737** at step 214,780. This is the saved scalar from train_state.json and the final manifest, not an invented validation curve.

Saved wall_clock_s: **10231.992s**, for the recovered invocation only. It includes that invocation's setup and checkpoint publication and excludes the original run's elapsed time. A whole-run duration or throughput cannot be obtained by summing inter-run it_per_s values; none is reported.

| Saved configuration/provenance | Value |
|---|---|
| Parameters | 18,034,978 |
| Dataset clips | 2,808,971 |
| Dataset | `data/study/walker2d/continuation-20260926-2/setA.h5` |
| Dataset SHA256 | `4602eeb952c918a3a9de787605d496c5a751fd692396f539e3e82fcf9c96f267` |
| Split counts | `{"training": 4544, "validation": 93}` |
| Splits SHA256 | `e7a8083f82d7b7701587b3dcaf1c73043384c18603d617470d293de58221b4b6` |
| Recipe | `{"batch": 128, "clip_stride": 1, "epochs": 10, "lr": 5e-05, "max_steps": 0, "push_every": 4000, "seed": 3072, "smoke": false, "warmup": 500, "wd": 0.001, "workers": 6}` |
| Final checkpoint revision | `9e05478658484d0f26ea740fead9e0ab7625367e` |
| Resume bundle SHA256 | `50279ce50cac07cc74ef0fa8757415a71e2fedc9f69a09db8699a232174b0d5c` |

The fully saved model configuration, render validation, source/data provenance and input hashes are retained in summary.json. The workflow's completed-stage hashes were verified before reporting. Every curve point comes from the saved history; no model, simulator, optimizer or network work was performed.
