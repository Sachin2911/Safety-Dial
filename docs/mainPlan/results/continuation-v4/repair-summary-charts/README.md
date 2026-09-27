# Saved E2 summary charts

These figures present immutable saved results. No bootstrap, model, simulator, training, recalibration or example selection was performed.

## Test decisions

All four point FSAs are undefined. Lines show censor identification bounds, not confidence intervals or midpoint estimates. Each arm retains its development-set threshold; achieved test acceptance rates differ.

| Arm | Margin px | Accepted / total | AR | Known false-safe | Accepted unresolved | FSA bounds |
|---|---:|---:|---:|---:|---:|---|
| no_update | 0.036740 | 3139/4096 | 76.64% | 756 | 49 | 24.08% to 25.65% |
| adapted | 6.071947 | 2950/4096 | 72.02% | 620 | 50 | 21.02% to 22.71% |
| readout_correction | 3.802506 | 2905/4096 | 70.92% | 795 | 34 | 27.37% to 28.54% |
| fixed_margin | 80.000000 | 132/4096 | 3.22% | 4 | 12 | 3.03% to 12.12% |

The fixed-margin control uses the recorded 80 px fallback because its development FSA target is undefined. The chosen adaptation recipe also used the recorded fallback: zero of 16 candidates met the saved eligibility rule. Neither fallback is reselected here.

The conservative fixed-set relative-reduction outer bound is 5.697695% to 18.046952%, computed exactly from the integer counts. Its upper bound is below the 25% gate target. Shared labels need not jointly attain the endpoints. This is not a confidence interval, population-effect estimate, or evidence of no positive gain.

## Retention

Saved 400-clip errors are shown in separate units. Pose error is against the real-image readout, not ground-truth geometry. No evaluation was regenerated.

| Metric | No update | Adapted | Units |
|---|---:|---:|---|
| latent_mse_tf | 0.008681825 | 0.006402136 | latent squared units |
| latent_mse_rollout | 0.043945361 | 0.030984864 | latent squared units |
| pose_h5_centre_px_vs_real_readout | 5.819512367 | 3.871506214 | pixels, versus frozen real-image readout (not simulator ground truth) |

Goal-retention execution: 18 baseline and 16 adapted cases reached all 250 planned actions; 2 and 4 ended early without verified goal completion. These are not goal-success counts. All 20 paired cases per model are retained; incomplete outcomes prevent a completed paired-retention conclusion.

## Cost

58,475 charged E2 simulator steps. Failed/discarded collection and repeated context queries are included; upstream E1 bank generation and diagnosis are excluded. Model queries and optimizer work are separate costs.

| Category | Steps |
|---|---:|
| Adaptation root generation | 16,972 |
| Queried adaptation branches | 21,855 |
| Training / readout context replay | 3,140 |
| Evaluation context replay | 6,880 |
| Goal retention (both models) | 9,628 |

Each chart is provided as PNG and vector PDF. chart_data.json retains all plotted numbers, exact fractions, case identities and source pointers. The original saved scientific gate remains false.
